"""Sequence learners under the Phase 6 label treatments (matrix rows L1-L6, U column).

Same units, U sample, positives, weights, priors and evaluation as
`phase6_minute_pu`; only the learner changes:
  mlp          64-32 on the 34 aggregated window features + context (reference)
  gru, lstm    1 layer, 64 units, over 24 bins x (2 sections x 8 channels)
  tcn          3 causal dilated conv layers (kernel 3, dilation 1/3/9), 64 channels
  transformer  2 encoder layers, d=64, 4 heads, one token per section x bin,
               bin and section embeddings, CLS read-out
Sequence learners see the raw 5-min bins of [t_end-120, t_end)
(`phase9_sequence_dataset`) plus the same context block as the MLP (hour,
weekend, season, edge length, mean flow), so the comparison is hand-made window
aggregates versus learned ones. Every learner is trained with the same risks
(PN with U as negative, or nnPU) on mini-batches of U; a 20% hold-out of the
2024 P and U selects the stopping step. Training 2024, evaluation 2025, which
has been examined before: exploratory.

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

from phase6_minute_pu import (ALL_TO_INJURY_RATIO, CONTEXT, PRIMARY_BUFFER, SEED, TRAFFIC, Preprocessor,
                              evaluate, interval, label_weights, positive_mask, raw_features)
from phase9_sequence_dataset import CHANNELS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "v2" / "phase6"
SEQUENCES = ROOT / "analysis" / "results" / "v2" / "phase9"
OUT = SEQUENCES / "sequence_models"
LEARNERS = ("mlp", "gru", "lstm", "tcn", "transformer")
ARMS = {
    # name: (positives, weighting, risk, prior)
    "L1_pn_reliable": ("reliable", "uniform", "pn", "labeled"),
    "L2_pn_union": ("union", "uniform", "pn", "labeled"),
    "L3_pn_union_credibility": ("union", "credibility", "pn", "labeled"),
    "L4_nnpu_reliable": ("reliable", "uniform", "nnpu", "injury"),
    "L5_mswspu": ("union", "credibility", "nnpu", "all"),
    "L6_mswspu_propensity": ("union", "credibility+propensity", "nnpu", "all"),
}


class SequenceNet(torch.nn.Module):
    def __init__(self, learner: str, channels: int, context: int, width: int = 64, bins: int = 24,
                 dropout: float = 0.0):
        super().__init__()
        self.learner = learner
        if learner in ("gru", "lstm"):
            cell = torch.nn.GRU if learner == "gru" else torch.nn.LSTM
            self.encoder = cell(2 * channels, width, batch_first=True)
        elif learner == "tcn":
            self.dilations = (1, 3, 9)
            self.convs = torch.nn.ModuleList(
                torch.nn.Conv1d(2 * channels if i == 0 else width, width, 3, dilation=d)
                for i, d in enumerate(self.dilations))
            self.skip = torch.nn.Conv1d(2 * channels, width, 1)
        elif learner == "transformer":
            self.embed = torch.nn.Linear(channels, width)
            self.position = torch.nn.Parameter(torch.zeros(1, 1, bins, width))
            self.side = torch.nn.Parameter(torch.zeros(1, 2, 1, width))
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, width))
            for parameter in (self.position, self.side, self.cls):
                torch.nn.init.normal_(parameter, std=0.02)
            layer = torch.nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        else:
            raise ValueError(learner)
        self.head = torch.nn.Sequential(torch.nn.Linear(width + context, 32), torch.nn.ReLU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(32, 1))

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, sides, bins, channels = sequence.shape
        if self.learner in ("gru", "lstm"):
            output, _ = self.encoder(sequence.permute(0, 2, 1, 3).reshape(batch, bins, sides * channels))
            summary = output[:, -1]
        elif self.learner == "tcn":
            x = sequence.permute(0, 1, 3, 2).reshape(batch, sides * channels, bins)
            h = x
            for conv, dilation in zip(self.convs, self.dilations):
                h = F.relu(conv(F.pad(h, (2 * dilation, 0))))
            summary = (h + self.skip(x))[:, :, -1]
        else:
            tokens = (self.embed(sequence) + self.position + self.side).reshape(batch, sides * bins, -1)
            tokens = torch.cat([self.cls.expand(batch, -1, -1), tokens], dim=1)
            summary = self.encoder(tokens)[:, 0]
        return self.head(torch.cat([summary, context], dim=1))


class FlatNet(torch.nn.Module):
    """The Phase 6 MLP on aggregated features."""

    def __init__(self, features: int, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(features, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(32, 1))

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.net(flat)


class SequencePreprocessor:
    """Per-channel log/standardisation fitted on U; NaN -> 0 plus a missing-bin channel."""

    LOG = ("flow", "valid")

    def fit(self, sequences: np.ndarray) -> "SequencePreprocessor":
        values = self._log(sequences)
        self.mean = np.nanmean(values, axis=(0, 1, 2))
        self.scale = np.nanstd(values, axis=(0, 1, 2))
        self.scale[self.scale == 0] = 1.0
        return self

    def _log(self, sequences: np.ndarray) -> np.ndarray:
        values = sequences.copy()
        for name in self.LOG:
            index = CHANNELS.index(name)
            values[..., index] = np.log1p(np.clip(values[..., index], 0, None))
        return values

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        values = (self._log(sequences) - self.mean) / self.scale
        missing = np.isnan(values[..., CHANNELS.index("flow")])[..., None]
        values = np.clip(np.nan_to_num(values, nan=0.0), -8, 8)
        return np.concatenate([values, missing], axis=-1).astype(np.float32)


def risk_parts(model, p_inputs, wp, u_inputs, prior):
    zp, zu = model(*p_inputs).squeeze(1), model(*u_inputs).squeeze(1)
    weight = wp / wp.sum()
    positive = prior * (weight * F.softplus(-zp)).sum()
    negative = F.softplus(zu).mean() - prior * (weight * F.softplus(zp)).sum()
    return positive, negative, F.softplus(zu).mean()


def batched_scores(model, inputs, rows=None, size=16384) -> torch.Tensor:
    first = inputs[0]
    rows = torch.arange(len(first), device=first.device) if rows is None else rows
    with torch.no_grad():
        return torch.cat([model(*(t[rows[i:i + size]] for t in inputs)).squeeze(1)
                          for i in range(0, len(rows), size)])


def fit(learner, inputs, p_rows, wp, u_rows, prior, mode, seed, device, batch_u=4096, batch_p=1024,
        max_steps=4000, check_every=50, patience=8, lr=1e-3, dropout=0.0, make_model=None):
    """Mini-batch training; a 20% hold-out of P and U selects the stopping step.

    `make_model` builds a learner not defined here (it must end in `.head`)."""
    rng = np.random.default_rng(seed)
    hold_p, hold_u = rng.random(len(p_rows)) < 0.2, rng.random(len(u_rows)) < 0.2
    tensor = lambda a, dtype=torch.long: torch.as_tensor(a, dtype=dtype, device=device)
    tp, tw, tu = tensor(p_rows[~hold_p]), tensor(wp[~hold_p], torch.float32), tensor(u_rows[~hold_u])
    vp, vw, vu = tensor(p_rows[hold_p]), tensor(wp[hold_p], torch.float32), tensor(u_rows[hold_u])
    torch.manual_seed(seed)
    if make_model is not None:
        model = make_model().to(device)
    elif learner == "mlp":
        model = FlatNet(inputs[0].shape[1], dropout).to(device)
    else:
        model = SequenceNet(learner, inputs[0].shape[-1], inputs[1].shape[1], dropout=dropout).to(device)
    last = model.net[-1] if learner == "mlp" else model.head[-1]
    with torch.no_grad():
        last.bias.fill_(float(np.log(prior / (1 - prior))))
        last.weight.mul_(0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed)

    def validation() -> float:
        model.eval()
        zp, zu = batched_scores(model, inputs, vp), batched_scores(model, inputs, vu)
        weight = vw / vw.sum()
        positive = prior * (weight * F.softplus(-zp)).sum()
        if mode == "pn":
            value = positive + F.softplus(zu).mean()
        else:
            value = positive + torch.clamp(F.softplus(zu).mean() - prior * (weight * F.softplus(zp)).sum(), min=0)
        model.train()
        return value.item()

    best, best_state, best_step, stale, clipped = float("inf"), None, 0, 0, 0
    order, cursor = torch.randperm(len(tu), device=device, generator=generator), 0
    model.train()
    for step in range(1, max_steps + 1):
        if cursor + batch_u > len(order):
            order, cursor = torch.randperm(len(tu), device=device, generator=generator), 0
        u_batch = tu[order[cursor:cursor + batch_u]]
        cursor += batch_u
        if len(tp) > batch_p:
            pick = torch.randint(len(tp), (batch_p,), device=device, generator=generator)
            p_batch, w_batch = tp[pick], tw[pick]
        else:
            p_batch, w_batch = tp, tw
        optimiser.zero_grad()
        positive, negative, unlabeled_negative = risk_parts(
            model, [t[p_batch] for t in inputs], w_batch, [t[u_batch] for t in inputs], prior)
        if mode == "nnpu" and negative.item() < 0:
            clipped += 1
            loss = -negative  # Kiryo et al.: step back when the negative risk is below zero
        else:
            loss = positive + (unlabeled_negative if mode == "pn" else negative)
        (loss / prior).backward()
        optimiser.step()
        if step % check_every == 0:
            value = validation()
            if value < best - 1e-12:
                best, best_step, stale = value, step, 0
                best_state = {key: tensor_.detach().clone() for key, tensor_ in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_step": best_step, "validation_risk": best, "steps_with_negative_risk": clipped,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}


def load():
    train = pd.read_parquet(DATA / "minute_dataset_2024.parquet")
    test = pd.read_parquet(DATA / "minute_dataset_2025.parquet")
    for frame in (train, test):
        frame["official_confirmed"] = frame["official_confirmed"].fillna(False).astype(bool)
    return train, test


def train_learners(args) -> None:
    device = args.device
    torch.set_num_threads(8)
    train, test = load()
    summary = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    units = summary["years"]["2024"]["population_edge_5min_units"]
    x_train_raw, x_test_raw = raw_features(train), raw_features(test)
    is_u = (train["kind"] == "marginal").to_numpy()
    flat = Preprocessor(TRAFFIC + CONTEXT).fit(x_train_raw.loc[is_u])
    context = Preprocessor(CONTEXT).fit(x_train_raw.loc[is_u])
    labels = label_weights(train, test, x_train_raw, x_test_raw, is_u)
    prior_injury = int((train["kind"] == "official_case").sum()) / units
    prior_all = prior_injury * ALL_TO_INJURY_RATIO

    sequence_inputs = {}
    if any(learner != "mlp" for learner in args.learners):
        raw_train, raw_test = (np.load(SEQUENCES / f"sequences_{year}.npy", mmap_mode="r") for year in (2024, 2025))
        sequence_prep = SequencePreprocessor().fit(np.asarray(raw_train[is_u]))
        sequence_inputs = {year: sequence_prep.transform(np.asarray(raw)) for year, raw in ((2024, raw_train), (2025, raw_test))}
    to_device = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

    OUT.mkdir(parents=True, exist_ok=True)
    u_rows = np.flatnonzero(is_u)
    for learner in args.learners:
        if learner == "mlp":
            inputs = {2024: [to_device(flat.transform(x_train_raw))], 2025: [to_device(flat.transform(x_test_raw))]}
        else:
            inputs = {year: [to_device(sequence_inputs[year]), to_device(c.transform(x))]
                      for year, c, x in ((2024, context, x_train_raw), (2025, context, x_test_raw))}
        scores, fits = {}, {}
        for name, (which, weighting, mode, target) in ARMS.items():
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
                                  lr=args.lr, dropout=args.dropout)
                probabilities.append(torch.sigmoid(batched_scores(model, inputs[2025])).cpu().numpy())
                notes.append(note)
            key = f"{learner}__{name}__b{PRIMARY_BUFFER}"
            scores[key] = np.mean(probabilities, axis=0)
            fits[key] = {"lr": args.lr, "patience": args.patience, "max_steps": args.max_steps, "dropout": args.dropout,
                         "positives": int(mask.sum()), "effective_positive_weight": float(weight.sum()),
                         "prior": float(prior), "mode": mode, "weighting": weighting,
                         "parameters": notes[0]["parameters"], "seconds": time.perf_counter() - started,
                         "best_steps": [n["best_step"] for n in notes],
                         "steps_with_negative_risk": [n["steps_with_negative_risk"] for n in notes]}
            print(f"fitted {key}: positives={mask.sum()}, prior={prior:.3e}, steps={fits[key]['best_steps']}, "
                  f"{fits[key]['seconds']:.0f}s", flush=True)
        pd.DataFrame(scores).assign(row_id=test["row_id"].to_numpy()).to_parquet(OUT / f"scores_{learner}.parquet", index=False)
        (OUT / f"fits_{learner}.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
        del inputs
        torch.cuda.empty_cache()


def evaluate_all(args) -> None:
    _, test = load()
    scores, fits = {}, {}
    for learner in LEARNERS:
        path = OUT / f"scores_{learner}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not (frame["row_id"].to_numpy() == test["row_id"].to_numpy()).all():
            raise ValueError(f"{path} is not in dataset row order")
        scores.update({key: frame[key].to_numpy() for key in frame.columns if key != "row_id"})
        fits.update(json.loads((OUT / f"fits_{learner}.json").read_text(encoding="utf-8")))
    rng = np.random.default_rng(SEED)
    metrics, place, conditional = evaluate(test, scores, rng, args.replicates)

    learners = [learner for learner in LEARNERS if any(key.startswith(f"{learner}__") for key in scores)]
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
    for learner in learners:
        if learner != "mlp" and "mlp" in learners:
            for arm in ("L2_pn_union", "L5_mswspu"):
                comparisons[f"{learner}_minus_mlp__{arm}"] = {
                    "meaning": f"learned sequence encoder vs aggregated features, {arm}",
                    **paired(key(learner, arm), key("mlp", arm))}
        for a, b, meaning in (("L2_pn_union", "L1_pn_reliable", "weak Waze positives added (no PU)"),
                              ("L4_nnpu_reliable", "L1_pn_reliable", "nnPU vs U-as-negative, reliable positives"),
                              ("L5_mswspu", "L3_pn_union_credibility", "nnPU vs U-as-negative, credibility-weighted union"),
                              ("L6_mswspu_propensity", "L5_mswspu", "reporting-propensity (SAR) correction")):
            if key(learner, a) not in scores or key(learner, b) not in scores:
                continue
            comparisons[f"{learner}__{a}_minus_{b}"] = {"meaning": meaning, **paired(key(learner, a), key(learner, b))}

    # Probability level against the complete police register (edge-hour injury crashes).
    summary = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    kind = test["kind"].to_numpy()
    observed = (kind == "official_case").sum() / (summary["years"]["2025"]["population_edge_5min_units"] / 12)
    for name, score in scores.items():
        hourly = 1 - (1 - score[kind == "marginal"]) ** 12
        metrics[name]["predicted_over_observed_hourly"] = float(hourly.mean() / observed)

    results = {
        "design": {"unit": "edge x 5-min", "train_year": 2024, "test_year": 2025, "buffer_min": PRIMARY_BUFFER,
                   "sequence": "2 sections x 24 bins x 8 channels over [t_end-120, t_end)",
                   "training": "mini-batch AdamW (U 4096, P <=1024), 20% hold-out early stopping, mean of seeds",
                   "test_status": "2025 previously examined; exploratory"},
        "arms": {name: dict(zip(("positives", "weighting", "risk", "prior"), arm)) for name, arm in ARMS.items()},
        "observed_injury_rate_per_edge_hour_2025": float(observed),
        "fits": fits, "metrics": metrics, "paired_comparisons": comparisons,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, learners)


def write_report(results: dict, learners: list[str]) -> None:
    metrics, fits = results["metrics"], results["fits"]
    b = PRIMARY_BUFFER

    def ci(value, bounds):
        return f"{value:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"

    lines = ["# Phase 9: sequence learners × label handling (U column, 30-minute buffer)", "",
             "Same data, U, positives and weights, priors and evaluation as Phase 6; only the learner changes. The MLP uses the 34 aggregated features, "
             "the sequence learners the raw 5-minute section series over [t_end−120, t_end) (up/downstream × 24 bins × 8 channels) plus the same context block. "
             "All learners are retrained in this script with the same mini-batch procedure, so the MLP row differs slightly from the full-batch MLP of Phase 6. "
             "Trained on 2024, evaluated on 2025; 2025 has been inspected before, so the results are exploratory.", ""]
    setting = next(iter(fits.values()))
    lines += [f"Training settings: learning rate {setting.get('lr', 1e-3):g}, early-stopping patience {setting.get('patience', 8)} checks (every 50 steps), "
              f"at most {setting.get('max_steps', 4000)} steps, dropout before the output layer {setting.get('dropout', 0.0):g}.", ""]
    for title, column in (("Population ROC-AUC: all official crashes (hourly) vs marginal units", ("population", "official_all", "roc_auc", "roc_auc_ci95")),
                          ("Within-set concordance: all Waze-timed events", ("matched", "waze_all", "within_set_concordance", "ci95")),
                          ("Within-set concordance: official crash hours (Phase 2 risk sets)", ("matched", "official_all", "within_set_concordance", "ci95"))):
        lines += [f"## {title} [95% CI]", "", "| Label handling | " + " | ".join(learners) + " |",
                  "|---|" + "---:|" * len(learners)]
        for arm in ARMS:
            cells = []
            for learner in learners:
                entry = metrics[f"{learner}__{arm}__b{b}"][column[0]][column[1]]
                cells.append(ci(entry[column[2]], entry[column[3]]))
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Probability level: predicted/observed (injury crashes per edge-hour, against the complete register)", "",
              "| Label handling | " + " | ".join(learners) + " |", "|---|" + "---:|" * len(learners)]
    for arm in ARMS:
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
        entries = [fits[f"{learner}__{arm}__b{b}"] for arm in ARMS]
        lines.append(f"| {learner} | {entries[0]['parameters']} | {np.mean([e['seconds'] for e in entries]):.0f} | "
                     f"{fits[f'{learner}__L5_mswspu__b{b}']['best_steps']} |")
    lines += ["", "Intervals are a bootstrap over positives (or risk sets) with fixed models, without retraining uncertainty, spatio-temporal correlation or multiple-comparison correction.", "",
              "Reproduction: `python analysis/phase9_sequence_dataset.py`; `python analysis/phase9_sequence_models.py --learners ...` (can be split across GPUs); "
              "`python analysis/phase9_sequence_models.py --evaluate`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", nargs="+", choices=LEARNERS, default=list(LEARNERS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8, help="hold-out checks (every 50 steps) without improvement")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--dropout", type=float, default=0.0, help="before the output layer (and in the MLP hidden layers)")
    parser.add_argument("--out-name", default="sequence_models", help="result directory under results/v2/phase9")
    args = parser.parse_args()
    global OUT
    OUT = SEQUENCES / args.out_name
    if args.evaluate:
        evaluate_all(args)
    else:
        train_learners(args)


if __name__ == "__main__":
    main()
