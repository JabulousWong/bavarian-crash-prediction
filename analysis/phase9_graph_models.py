"""Ego-graph attention (ego-GAT) learners under the Phase 6 label treatments (U column).

The v2 network is a set of directed chains, so the graph around a row's edge is its
k upstream and k downstream edges plus, optionally, the opposite-carriageway edge
(`phase9_graph_dataset`). Every edge in that ego graph is a node; its input is the
raw 5-min sequence of its two cross-sections over [t_end-120, t_end), encoded by
the same 1-layer GRU (64 units) as `phase9_sequence_models`, plus a learned role
embedding (up2, up1, ego, dn1, dn2, opp). Two graph-attention layers (4 heads,
residual) pass messages along the chain (and ego<->opp); missing neighbours are
masked out. The ego node is read out and joined with the same context block.

  gat_k0      ego node only: GRU encoder + attention over the self loop (control)
  gat_k1      + one upstream and one downstream edge
  gat_k2      + two upstream and two downstream edges
  gat_k2_opp  + opposite carriageway

Same units, U sample, positives, weights, priors, training loop and evaluation as
`phase9_sequence_models`, which also supplies the MLP and GRU reference scores.
Training 2024, evaluation 2025 (previously examined): exploratory.
Run one process per GPU with disjoint --learners, then --evaluate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from phase6_minute_pu import (ALL_TO_INJURY_RATIO, CONTEXT, PRIMARY_BUFFER, SEED, Preprocessor, evaluate, interval,
                              label_weights, positive_mask, raw_features)
from phase9_graph_dataset import SECTIONS
from phase9_sequence_models import ARMS, SequencePreprocessor, batched_scores, fit, load


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "v2" / "phase6"
SEQUENCES = ROOT / "analysis" / "results" / "v2" / "phase9"
REFERENCE = SEQUENCES / "sequence_models"
OUT = SEQUENCES / "graph_models"
# Section order in the stacked input: the six neighbour sections around the two ego sections.
ORDER = ["s_m2", "s_m1", "s0", "s1", "s_p2", "s_p3", "opp_up", "opp_dn"]
ROLES = {"up2": ("s_m2", "s_m1"), "up1": ("s_m1", "s0"), "ego": ("s0", "s1"),
         "dn1": ("s1", "s_p2"), "dn2": ("s_p2", "s_p3"), "opp": ("opp_up", "opp_dn")}
LINKS = [("up2", "up1"), ("up1", "ego"), ("ego", "dn1"), ("dn1", "dn2"), ("opp", "ego")]
LEARNERS = {"gat_k0": ("ego",), "gat_k1": ("up1", "ego", "dn1"),
            "gat_k2": ("up2", "up1", "ego", "dn1", "dn2"),
            "gat_k2_opp": ("up2", "up1", "ego", "dn1", "dn2", "opp")}
REFERENCE_LEARNERS = ("mlp", "gru")


class GraphAttention(torch.nn.Module):
    """Dense multi-head graph attention (Velickovic et al. 2018) over a small fixed graph."""

    def __init__(self, width: int, heads: int = 4):
        super().__init__()
        self.heads, self.size = heads, width // heads
        self.project = torch.nn.Linear(width, width, bias=False)
        self.source = torch.nn.Parameter(torch.empty(heads, self.size))
        self.target = torch.nn.Parameter(torch.empty(heads, self.size))
        torch.nn.init.xavier_uniform_(self.source)
        torch.nn.init.xavier_uniform_(self.target)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, nodes, width = h.shape
        z = self.project(h).view(batch, nodes, self.heads, self.size)
        logits = (z * self.target).sum(-1).unsqueeze(2) + (z * self.source).sum(-1).unsqueeze(1)  # (B, i, j, H)
        logits = F.leaky_relu(logits, 0.2).masked_fill(~adjacency.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(logits, dim=2)
        return torch.einsum("bijh,bjhd->bihd", attention, z).reshape(batch, nodes, width)


class EgoGraphNet(torch.nn.Module):
    def __init__(self, roles: tuple[str, ...], channels: int, context: int, width: int = 64, layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.ego = roles.index("ego")
        pairs = [[ORDER.index(section) for section in ROLES[role]] for role in roles]
        self.register_buffer("pairs", torch.tensor(pairs))
        self.register_buffer("role_slots", torch.tensor([list(ROLES).index(role) for role in roles]))
        links = torch.eye(len(roles), dtype=torch.bool)
        for a, b in LINKS:
            if a in roles and b in roles:
                links[roles.index(a), roles.index(b)] = links[roles.index(b), roles.index(a)] = True
        self.register_buffer("links", links)
        self.encoder = torch.nn.GRU(2 * channels, width, batch_first=True)
        self.role = torch.nn.Embedding(len(ROLES), width)
        torch.nn.init.normal_(self.role.weight, std=0.02)
        self.attention = torch.nn.ModuleList(GraphAttention(width) for _ in range(layers))
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(width) for _ in range(layers))
        self.head = torch.nn.Sequential(torch.nn.Linear(width + context, 32), torch.nn.ReLU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(32, 1))

    def forward(self, sections: torch.Tensor, present: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, _, bins, channels = sections.shape
        nodes = len(self.pairs)
        x = sections[:, self.pairs]  # (B, nodes, 2, bins, channels)
        x = x.permute(0, 1, 3, 2, 4).reshape(batch * nodes, bins, 2 * channels)
        output, _ = self.encoder(x)
        h = output[:, -1].view(batch, nodes, -1) + self.role(self.role_slots)
        mask = present[:, self.role_slots]
        # Absent nodes keep only their self loop, so nothing attends to them and their rows stay finite.
        adjacency = (self.links & mask.unsqueeze(1) & mask.unsqueeze(2)) | self.links.diagonal().diag()
        for attention, norm in zip(self.attention, self.norms):
            h = norm(h + F.elu(attention(h, adjacency)))
        return self.head(torch.cat([h[:, self.ego], context], dim=1))


def graph_inputs(year: int, ego: np.ndarray, rows: pd.DataFrame, edges: pd.DataFrame,
                 prep: SequencePreprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Stacked, standardised sections in ORDER and the per-row presence of each role's edge."""
    neighbours = np.load(SEQUENCES / f"graph_sequences_{year}.npy", mmap_mode="r")
    stacked = np.concatenate([neighbours[:, :2], ego, neighbours[:, 2:]], axis=1)
    assert [*SECTIONS[:2], "s0", "s1", *SECTIONS[2:]] == ORDER
    table = edges.set_index("edge_id")
    present = np.ones((len(rows), len(ROLES)), dtype=bool)
    for slot, role in enumerate(ROLES):
        if role != "ego":
            present[:, slot] = rows["edge_id"].map(table[role]).notna().to_numpy()
    return prep.transform(stacked), present


def train_learners(args) -> None:
    device = args.device
    torch.set_num_threads(8)
    train, test = load()
    summary = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    units = summary["years"]["2024"]["population_edge_5min_units"]
    x_train_raw, x_test_raw = raw_features(train), raw_features(test)
    is_u = (train["kind"] == "marginal").to_numpy()
    context = Preprocessor(CONTEXT).fit(x_train_raw.loc[is_u])
    labels = label_weights(train, test, x_train_raw, x_test_raw, is_u)
    prior_injury = int((train["kind"] == "official_case").sum()) / units
    prior_all = prior_injury * ALL_TO_INJURY_RATIO

    edges = pd.read_csv(SEQUENCES / "graph_edges.csv", dtype={r: "Int64" for r in ("up1", "up2", "dn1", "dn2", "opp")})
    ego = {year: np.load(SEQUENCES / f"sequences_{year}.npy", mmap_mode="r") for year in (2024, 2025)}
    prep = SequencePreprocessor().fit(np.asarray(ego[2024][is_u]))  # as in phase9_sequence_models
    to_device = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    inputs = {}
    for year, frame, x in ((2024, train, x_train_raw), (2025, test, x_test_raw)):
        sections, present = graph_inputs(year, ego[year], frame, edges, prep)
        inputs[year] = [to_device(sections), to_device(present, torch.bool), to_device(context.transform(x))]
        del sections

    OUT.mkdir(parents=True, exist_ok=True)
    u_rows = np.flatnonzero(is_u)
    for learner in args.learners:
        roles = LEARNERS[learner]
        make_model = lambda: EgoGraphNet(roles, inputs[2024][0].shape[-1], inputs[2024][2].shape[1],
                                         layers=args.layers, dropout=args.dropout)
        scores, fits = {}, {}
        for name, (which, weighting, mode, target) in ARMS.items():
            if args.arms and name not in args.arms:
                continue
            mask = positive_mask(train, PRIMARY_BUFFER, which)
            weight = np.ones(mask.sum())
            if "credibility" in weighting:
                weight = weight * labels["credibility_weight"][mask]
            if "propensity" in weighting:
                weight = weight * labels["inverse_propensity"][mask]
            prior = {"labeled": weight.sum() / units, "injury": prior_injury, "all": prior_all}[target]
            probabilities, notes, started = [], [], time.perf_counter()
            for seed in range(args.seeds):
                model, note = fit(learner, inputs[2024], np.flatnonzero(mask), weight, u_rows, prior, mode,
                                  SEED + seed, device, max_steps=args.max_steps, patience=args.patience,
                                  lr=args.lr, dropout=args.dropout, make_model=make_model)
                probabilities.append(torch.sigmoid(batched_scores(model, inputs[2025], size=8192)).cpu().numpy())
                notes.append(note)
            key = f"{learner}__{name}__b{PRIMARY_BUFFER}"
            scores[key] = np.mean(probabilities, axis=0)
            fits[key] = {"roles": list(roles), "layers": args.layers, "lr": args.lr, "patience": args.patience,
                         "max_steps": args.max_steps, "dropout": args.dropout,
                         "positives": int(mask.sum()), "effective_positive_weight": float(weight.sum()),
                         "prior": float(prior), "mode": mode, "weighting": weighting,
                         "parameters": notes[0]["parameters"], "seconds": time.perf_counter() - started,
                         "best_steps": [n["best_step"] for n in notes],
                         "steps_with_negative_risk": [n["steps_with_negative_risk"] for n in notes]}
            print(f"fitted {key}: positives={mask.sum()}, prior={prior:.3e}, steps={fits[key]['best_steps']}, "
                  f"{fits[key]['seconds']:.0f}s", flush=True)
        pd.DataFrame(scores).assign(row_id=test["row_id"].to_numpy()).to_parquet(OUT / f"scores_{learner}.parquet", index=False)
        (OUT / f"fits_{learner}.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()


def evaluate_all(args) -> None:
    _, test = load()
    scores, fits = {}, {}
    sources = [(REFERENCE, learner) for learner in REFERENCE_LEARNERS] + [(OUT, learner) for learner in LEARNERS]
    for directory, learner in sources:
        path = directory / f"scores_{learner}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not (frame["row_id"].to_numpy() == test["row_id"].to_numpy()).all():
            raise ValueError(f"{path} is not in dataset row order")
        scores.update({key: frame[key].to_numpy() for key in frame.columns if key != "row_id"})
        fits.update(json.loads((directory / f"fits_{learner}.json").read_text(encoding="utf-8")))
    rng = np.random.default_rng(SEED)
    metrics, place, conditional = evaluate(test, scores, rng, args.replicates)

    learners = [learner for _, learner in sources if any(key.startswith(f"{learner}__") for key in scores)]
    key = lambda learner, arm: f"{learner}__{arm}__b{PRIMARY_BUFFER}"

    def paired(a: str, b: str) -> dict:
        output = {}
        for store, endpoint in ((place, "official_all"), (place, "waze_police_confirmed"),
                                (conditional, "waze_all"), (conditional, "official_all")):
            delta = store[(a, endpoint)] - store[(b, endpoint)]
            label = ("population_" if store is place else "matched_") + endpoint
            output[label] = {"delta": float(delta.mean()), "ci95": interval(delta, rng, args.replicates)}
        return output

    comparisons = {}
    ladder = (("gat_k0", "gru", "attention block on the ego node alone vs plain GRU (control)"),
              ("gat_k1", "gat_k0", "one upstream and one downstream edge"),
              ("gat_k2", "gat_k1", "second hop"),
              ("gat_k2_opp", "gat_k2", "opposite carriageway"),
              ("gat_k2_opp", "mlp", "full ego graph vs aggregated-feature MLP"))
    for arm in ("L2_pn_union", "L5_mswspu"):
        for a, b, meaning in ladder:
            if key(a, arm) in scores and key(b, arm) in scores:
                comparisons[f"{a}_minus_{b}__{arm}"] = {"meaning": meaning, **paired(key(a, arm), key(b, arm))}
    for learner in LEARNERS:
        for a, b, meaning in (("L2_pn_union", "L1_pn_reliable", "weak Waze positives added (no PU)"),
                              ("L4_nnpu_reliable", "L1_pn_reliable", "nnPU vs U-as-negative, reliable positives"),
                              ("L5_mswspu", "L3_pn_union_credibility", "nnPU vs U-as-negative, credibility-weighted union"),
                              ("L6_mswspu_propensity", "L5_mswspu", "reporting-propensity (SAR) correction")):
            if key(learner, a) in scores and key(learner, b) in scores:
                comparisons[f"{learner}__{a}_minus_{b}"] = {"meaning": meaning, **paired(key(learner, a), key(learner, b))}

    summary = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    kind = test["kind"].to_numpy()
    observed = (kind == "official_case").sum() / (summary["years"]["2025"]["population_edge_5min_units"] / 12)
    for name, score in scores.items():
        hourly = 1 - (1 - score[kind == "marginal"]) ** 12
        metrics[name]["predicted_over_observed_hourly"] = float(hourly.mean() / observed)

    graph = json.loads((SEQUENCES / "graph_summary.json").read_text(encoding="utf-8"))
    results = {
        "design": {"unit": "edge x 5-min", "train_year": 2024, "test_year": 2025, "buffer_min": PRIMARY_BUFFER,
                   "graph": "directed chain; ego edge, k up/downstream edges, opposite edge within "
                            f"{graph['opposite_max_m']:.0f} m; node = 2 sections x 24 bins x 8 channels",
                   "training": "as phase9_sequence_models (mini-batch AdamW, 20% hold-out early stopping, mean of seeds)",
                   "test_status": "2025 previously examined; exploratory"},
        "graph_edges_with": graph["edges_with"],
        "observed_injury_rate_per_edge_hour_2025": float(observed),
        "fits": fits, "metrics": metrics, "paired_comparisons": comparisons,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, learners)


def write_report(results: dict, learners: list[str]) -> None:
    metrics, fits = results["metrics"], results["fits"]
    b = PRIMARY_BUFFER
    arms = [arm for arm in ARMS if all(f"{learner}__{arm}__b{b}" in metrics for learner in learners)]

    def ci(value, bounds):
        return f"{value:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"

    with_ = results["graph_edges_with"]
    lines = ["# Phase 9: ego graph attention (ego-GAT) × label handling (U column, 30-minute buffer)", "",
             "The v2 network is a set of directed chains: every edge has exactly one upstream and one downstream neighbour (except at chain ends), so the graph model learns "
             "the traffic state k hops up- and downstream along the route (and on the opposite carriageway). Every edge is a node whose input is the raw "
             "5-minute series of its two detector sections over [t_end−120, t_end), encoded with the same encoder as the GRU learner plus a role embedding; two layers of 4-head graph attention with missing neighbours masked; the target edge is read out "
             "and joined with the same context block. Data, U, positives, weights, priors, training procedure and evaluation are the same as for the sequence learners; the mlp and gru columns are taken from "
             "`sequence_models`. Trained on 2024, evaluated on 2025; 2025 has been inspected before, so the results are exploratory.", "",
             f"Of the 590 edges, {with_['up1']}/{with_['dn1']} have a 1-hop up/downstream neighbour, {with_['up2']}/{with_['dn2']} a 2-hop one, "
             f"and {with_['opp']} an opposite edge.", ""]
    for title, column in (("Population ROC-AUC: all official crashes (hourly) vs marginal units", ("population", "official_all", "roc_auc", "roc_auc_ci95")),
                          ("Within-set concordance: all Waze-timed events", ("matched", "waze_all", "within_set_concordance", "ci95")),
                          ("Within-set concordance: official crash hours (Phase 2 risk sets)", ("matched", "official_all", "within_set_concordance", "ci95"))):
        lines += [f"## {title} [95% CI]", "", "| Label handling | " + " | ".join(learners) + " |",
                  "|---|" + "---:|" * len(learners)]
        for arm in arms:
            cells = []
            for learner in learners:
                entry = metrics[f"{learner}__{arm}__b{b}"][column[0]][column[1]]
                cells.append(ci(entry[column[2]], entry[column[3]]))
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Probability level: predicted/observed (injury crashes per edge-hour, against the complete register)", "",
              "| Label handling | " + " | ".join(learners) + " |", "|---|" + "---:|" * len(learners)]
    for arm in arms:
        lines.append(f"| {arm} | " + " | ".join(
            f"{metrics[f'{learner}__{arm}__b{b}']['predicted_over_observed_hourly']:.2f}" for learner in learners) + " |")
    lines += ["", "## Paired differences [95% CI]", "",
              "| Contrast | Meaning | Population: all official | Population: Waze-timed and police-confirmed | Matched: all Waze-timed | Matched: official |", "|---|---|---:|---:|---:|---:|"]
    for name, item in results["paired_comparisons"].items():
        cells = [f"{item[e]['delta']:+.4f} [{item[e]['ci95'][0]:+.4f}, {item[e]['ci95'][1]:+.4f}]"
                 for e in ("population_official_all", "population_waze_police_confirmed", "matched_waze_all", "matched_official_all")]
        lines.append(f"| {name} | {item['meaning']} | " + " | ".join(cells) + " |")
    lines += ["", "## Learner size and training time (per label handling, all seeds)", "", "| Learner | Parameters | Mean seconds | Early-stopping steps (L5) |", "|---|---:|---:|---|"]
    for learner in learners:
        entries = [fits[f"{learner}__{arm}__b{b}"] for arm in arms]
        l5 = fits.get(f"{learner}__L5_mswspu__b{b}", {}).get("best_steps", "")
        lines.append(f"| {learner} | {entries[0]['parameters']} | {np.mean([e['seconds'] for e in entries]):.0f} | {l5} |")
    lines += ["", "Intervals are a bootstrap over positives (or risk sets) with fixed models, without retraining uncertainty, spatio-temporal correlation or multiple-comparison correction.", "",
              "Reproduction: `python analysis/phase9_graph_dataset.py`; `python analysis/phase9_graph_models.py --learners ...` (can be split across GPUs); "
              "`python analysis/phase9_graph_models.py --evaluate`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", nargs="+", choices=list(LEARNERS), default=list(LEARNERS))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), help="subset of label treatments (smoke tests)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8, help="hold-out checks (every 50 steps) without improvement")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--out-name", default="graph_models", help="result directory under results/v2/phase9")
    args = parser.parse_args()
    global OUT
    OUT = SEQUENCES / args.out_name
    if args.evaluate:
        evaluate_all(args)
    else:
        train_learners(args)


if __name__ == "__main__":
    main()
