"""Checks requested in the internal review of the AAP manuscript (2026-10-28).

A  Strong baselines (no refit, v2 phase 6/7 predictions): paired differences of
   the Waze-trained arms against models trained on all official crash hours,
   with a bootstrap over positives (as in the paper) and over edges (clusters).
B  Calibration decomposition: calibration-in-the-large, logistic recalibration
   intercept and slope, and log skill per edge-hour. The class prior sets the
   intercept; the slope is a property of the ranking.
C  Time anchor: for Waze-timed events with an official record, the position of
   the first report and of the window end relative to the recorded crash hour,
   and the within-set concordance split by whether the window ends before the
   recorded hour.
D  Spatial hold-out by motorway (4 folds, both carriageways together) and a
   reverse-year replication (train 2025, evaluate 2024), refitting the key arms
   with the frozen phase 6 protocol. 2025 had been examined before, so D is a
   robustness check, not a confirmatory test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import phase6_minute_pu as p6
from phase6_minute_pu import CONTEXT, TRAFFIC, Preprocessor, fit_mlp, label_weights, placements, predict, raw_features, within_stratum

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "analysis" / "results" / "v2"
OUT = V2 / "phase10"
SEED = 20261028
REPS = 1000
FOLDS = {"A8": 0, "A99": 1, "A3": 1, "A9": 2, "A7": 2, "A93": 2, "A96": 3, "A94": 3, "A92": 3, "A95": 3}


def road(route: pd.Series) -> pd.Series:
    return route.str.split("_").str[0].str.split("-").str[0]


def ci(values: np.ndarray, rng, reps=REPS) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def cluster_ci(values: np.ndarray, groups: np.ndarray, rng, reps=REPS) -> list[float]:
    """Bootstrap over clusters (edges): mean of the values of the drawn clusters."""
    codes, inverse = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(codes), len(codes))
        draws.append(sums[pick].sum() / counts[pick].sum())
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def hourly(p):
    return 1 - (1 - np.clip(p, 0, 1)) ** 12


# ---------------------------------------------------------------- A, B, C
def part_abc(units: dict) -> dict:
    rng = np.random.default_rng(SEED)
    pred = pd.read_parquet(V2 / "phase6" / "minute_pu" / "test_predictions_2025.parquet")
    kind = pred["kind"].to_numpy()
    marginal, case = kind == "marginal", kind == "official_case"
    official_sets = np.isin(kind, ["official_case", "official_control"])
    edges_case = pred.loc[case, "edge_id"].to_numpy()
    out = {"A_strong_baselines": {}, "B_calibration": {}, "C_time_anchor": {}}

    def place(col):
        return placements(pred.loc[case, col].to_numpy(), np.sort(pred.loc[marginal, col].to_numpy()))

    def within(col):
        sub = pred.loc[official_sets]
        return within_stratum(sub, sub[col].to_numpy(), (sub["kind"] == "official_case").to_numpy())

    strata_edge = pred.loc[case].set_index("stratum_id")["edge_id"]
    pairs = [
        ("pn_union__b30", "pn_reliable__b30", "weak positives added (MLP, PN)"),
        ("pn_union__b30", "pn_official_hour__b30", "all Waze-timed vs all official crash hours (MLP, PN)"),
        ("mswspu_credibility__b30", "pn_official_hour__b30", "MSWS-PU vs all official crash hours (MLP)"),
        ("pn_reliable__b30", "pn_official_hour__b30", "484 reliable vs all official crash hours (MLP, PN)"),
        ("hgb_pn_union__b30", "hgb_pn_official_hour__b30", "all Waze-timed vs all official crash hours (HGB)"),
        ("hgb_pn_union__b30", "pn_official_hour__b30", "HGB all Waze-timed vs MLP official"),
        ("nnpu_reliable__b30", "pn_reliable__b30", "nnPU vs PN, reliable positives"),
    ]
    sets_index = pred.loc[official_sets & case, "stratum_id"].to_numpy()
    for a, b, meaning in pairs:
        dp = place(a) - place(b)
        dw = within(a) - within(b)
        # within_stratum returns strata in sorted order of stratum id
        strata_sorted = np.sort(pred.loc[official_sets, "stratum_id"].unique())
        strata_sorted = strata_sorted[np.isin(strata_sorted, sets_index)]
        w_edges = strata_edge.reindex(strata_sorted).to_numpy()
        out["A_strong_baselines"][f"{a}_minus_{b}"] = {
            "meaning": meaning,
            "population_auc": {"a": float(place(a).mean()), "b": float(place(b).mean()), "delta": float(dp.mean()),
                               "ci95_positives": ci(dp, rng), "ci95_edges": cluster_ci(dp, edges_case, rng)},
            "within_set": {"a": float(within(a).mean()), "b": float(within(b).mean()), "delta": float(dw.mean()),
                           "ci95_sets": ci(dw, rng), "ci95_edges": cluster_ci(dw, w_edges, rng) if len(w_edges) == len(dw) else None},
        }

    two = pd.read_parquet(V2 / "phase7" / "two_stage_predictions_2025.parquet")
    tk = two["kind"].to_numpy()
    tm, tc = tk == "marginal", tk == "official_case"
    te = two.loc[tc, "edge_id"].to_numpy()
    tsets = np.isin(tk, ["official_case", "official_control"])
    for a, b in [("pn_union__context+frequency+traffic", "pn_official_hour__context+frequency+traffic"),
                 ("mswspu__context+frequency+traffic", "pn_official_hour__context+frequency+traffic")]:
        pa = placements(two.loc[tc, a].to_numpy(), np.sort(two.loc[tm, a].to_numpy()))
        pb = placements(two.loc[tc, b].to_numpy(), np.sort(two.loc[tm, b].to_numpy()))
        sub = two.loc[tsets]
        wa = within_stratum(sub, sub[a].to_numpy(), (sub["kind"] == "official_case").to_numpy())
        wb = within_stratum(sub, sub[b].to_numpy(), (sub["kind"] == "official_case").to_numpy())
        out["A_strong_baselines"][f"two_stage::{a}_minus_{b}"] = {
            "population_auc": {"a": float(pa.mean()), "b": float(pb.mean()), "delta": float((pa - pb).mean()),
                               "ci95_positives": ci(pa - pb, rng), "ci95_edges": cluster_ci(pa - pb, te, rng)},
            "within_set": {"a": float(wa.mean()), "b": float(wb.mean()), "delta": float((wa - wb).mean()), "ci95_sets": ci(wa - wb, rng)},
        }
    # Edge-cluster intervals for the absolute AUC of the key arms.
    for col in ("pn_reliable__b30", "pn_union__b30", "pn_official_hour__b30", "mswspu_credibility__b30"):
        v = place(col)
        out["A_strong_baselines"][f"auc::{col}"] = {"auc": float(v.mean()), "ci95_positives": ci(v, rng),
                                                    "ci95_edges": cluster_ci(v, edges_case, rng)}

    # B: calibration decomposition per edge-hour. Sampled units stand for all edge-hours.
    n_hours = units[2025] / 12
    fitted = json.loads((V2 / "phase6" / "minute_pu" / "metrics.json").read_text())
    prior = fitted["priors_per_unit"]["injury_from_police_register"]
    labelled = fitted["fits"]["pn_reliable__b30"]["prior"]
    arms = {
        "constant_2024": np.full(len(pred), prior),
        "pn_reliable": pred["pn_reliable__b30"].to_numpy(),
        "pn_reliable_rescaled": np.clip(pred["pn_reliable__b30"].to_numpy() * prior / labelled, 0, 1),
        "nnpu_reliable": pred["nnpu_reliable__b30"].to_numpy(),
        "nnpu_reliable_propensity": pred["nnpu_reliable_propensity__b30"].to_numpy(),
        "pn_official_hour": pred["pn_official_hour__b30"].to_numpy(),
        "pn_union": pred["pn_union__b30"].to_numpy(),
        "mswspu_credibility": pred["mswspu_credibility__b30"].to_numpy(),
    }
    n_u, n_c = int(marginal.sum()), int(case.sum())
    for name, p in arms.items():
        hu, hc = np.clip(hourly(p[marginal]), 1e-12, 1 - 1e-12), np.clip(hourly(p[case]), 1e-12, 1 - 1e-12)
        entry = {"predicted_over_observed": float(hu.mean() * n_hours / n_c),
                 "mean_hourly_risk": float(hu.mean())}
        if name != "constant_2024":
            # Weighted logistic recalibration: units weighted to the population of edge-hours (label 0),
            # crash hours label 1 with weight 1; positives inside the units are negligible (2e-4).
            z = np.log(np.r_[hu, hc] / (1 - np.r_[hu, hc]))
            y = np.r_[np.zeros(n_u), np.ones(n_c)]
            w = np.r_[np.full(n_u, (n_hours - n_c) / n_u), np.ones(n_c)]
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y, sample_weight=w)
            slope = float(lr.coef_[0, 0])
            # intercept with slope fixed at 1 (calibration-in-the-large on the logit scale)
            off = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=True)
            from scipy.optimize import brentq
            g = lambda a: (w * (1 / (1 + np.exp(-(z + a))) - y)).sum()
            entry.update({"recalibration_slope": slope, "recalibration_intercept": float(lr.intercept_[0]),
                          "intercept_given_slope1": float(brentq(g, -10, 10))})
        out["B_calibration"][name] = entry

    # C: time anchor of Waze-timed events with an official record.
    for year in (2024, 2025):
        d = pd.read_parquet(V2 / "phase6" / f"minute_dataset_{year}.parquet")
        off = d.loc[d["kind"] == "official_case", ["edge_id", "t_end"]].rename(columns={"t_end": "H"})
        ev = d.loc[(d["kind"] == "waze_case") & (d["buffer_min"] == 30) & d["official_confirmed"].fillna(False).astype(bool),
                   ["edge_id", "reference_time", "stratum_id"]]
        m = ev.merge(off, on="edge_id")
        m["lag"] = (m["reference_time"] - m["H"]).dt.total_seconds() / 60
        m = m.loc[m["lag"].between(-15, 105)].sort_values("lag").drop_duplicates("stratum_id")
        lag = m["lag"].to_numpy()
        entry = {"confirmed_events": int(len(ev)), "matched_to_crash_hour_same_edge": int(len(m)),
                 "lag_first_report_minus_hour_start_min": {"median": float(np.median(lag)),
                                                          "q25": float(np.quantile(lag, 0.25)), "q75": float(np.quantile(lag, 0.75))},
                 "share_report_after_hour_end": float((lag >= 60).mean())}
        for b in (15, 30, 45):
            entry[f"share_window_end_after_hour_start_b{b}"] = float((lag - b > 0).mean())
        out["C_time_anchor"][str(year)] = entry
        if year == 2025:
            pre = set(m.loc[m["lag"] - 30 <= 0, "stratum_id"])  # window (b=30) ends at or before H
            post = set(m.loc[m["lag"] - 30 > 0, "stratum_id"])
            res = {}
            for b in (15, 30, 45):
                pre_b = set(m.loc[m["lag"] - b <= 0, "stratum_id"])
                post_b = set(m.loc[m["lag"] - b > 0, "stratum_id"])
                rows_all = (pred["buffer_min"] == b).to_numpy() & np.isin(kind, ["waze_case", "waze_control"])
                for col in (f"pn_union__b{b}", f"pn_reliable__b{b}"):
                    for label, s in (("window_ends_before_hour", pre_b), ("window_ends_inside_hour", post_b)):
                        # strata of the 30-min design are identified per buffer via the case's cluster
                        sub = pred.loc[rows_all]
                        cl = d.loc[(d["kind"] == "waze_case") & (d["buffer_min"] == 30)].set_index("stratum_id")["cluster_id"]
                        keep_clusters = set(cl.reindex(list(s)).dropna())
                        dcase = d.loc[(d["kind"] == "waze_case") & (d["buffer_min"] == b)]
                        strata_b = set(dcase.loc[dcase["cluster_id"].isin(keep_clusters), "stratum_id"])
                        sub = sub.loc[sub["stratum_id"].isin(strata_b)]
                        v = within_stratum(sub, sub[col].to_numpy(), (sub["kind"] == "waze_case").to_numpy())
                        res[f"{col}::{label}"] = {"risk_sets": int(len(v)), "concordance": float(v.mean()) if len(v) else None,
                                                  "ci95": ci(v, rng) if len(v) > 5 else None}
            out["C_time_anchor"]["within_set_by_window_position_2025"] = res
    return out


# ---------------------------------------------------------------- D
ARMS = {
    "pn_official_hour": ("official_hour", "uniform", "pn", "labeled"),
    "pn_reliable": ("reliable", "uniform", "pn", "labeled"),
    "nnpu_reliable": ("reliable", "uniform", "nnpu", "injury"),
    "pn_union": ("union", "uniform", "pn", "labeled"),
    "mswspu_credibility": ("union", "credibility", "nnpu", "all"),
}


def fit_arms(train, test, units_train, train_rows, device, seeds=5):
    """Phase 6 protocol on the rows `train_rows` of `train`; returns scores on `test`."""
    x_tr_raw, x_te_raw = raw_features(train), raw_features(test)
    is_u = (train["kind"] == "marginal").to_numpy() & train_rows
    prep = Preprocessor(TRAFFIC + CONTEXT).fit(x_tr_raw.loc[is_u])
    x_tr, x_te = prep.transform(x_tr_raw), prep.transform(x_te_raw)
    labels = label_weights(train.loc[train_rows].reset_index(drop=True), test, x_tr_raw.loc[train_rows].reset_index(drop=True),
                           x_te_raw, is_u[train_rows])
    cred = np.ones(len(train))
    cred[train_rows] = labels["credibility_weight"]
    n_official = int(((train["kind"] == "official_case").to_numpy() & train_rows).sum())
    prior_injury = n_official / units_train
    prior_all = prior_injury * p6.ALL_TO_INJURY_RATIO
    scores, info = {}, {}
    for name, (which, weighting, mode, target) in ARMS.items():
        mask = p6.positive_mask(train, 30, which) & train_rows
        weight = cred[mask] if weighting == "credibility" else np.ones(mask.sum())
        prior = {"labeled": mask.sum() / units_train, "injury": prior_injury, "all": prior_all}[target]
        models = [fit_mlp(x_tr[mask], weight, x_tr[is_u], prior, mode, SEED + s, device)[0] for s in range(seeds)]
        scores[name] = predict(models, x_te, device)
        info[name] = {"positives": int(mask.sum()), "prior": float(prior)}
    for name, which in (("hgb_official_hour", "official_hour"), ("hgb_union", "union")):
        mask = p6.positive_mask(train, 30, which) & train_rows
        rows = mask | is_u
        target = mask[rows].astype(int)
        w = np.where(target == 1, len(target) / (2 * target.sum()), len(target) / (2 * (len(target) - target.sum())))
        tree = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=False, random_state=SEED)
        tree.fit(x_tr_raw.loc[rows, TRAFFIC + CONTEXT], target, sample_weight=w)
        scores[name] = tree.predict_proba(x_te_raw[TRAFFIC + CONTEXT])[:, 1]
    return scores, info, {"prior_injury": prior_injury}


def evaluate_block(test, scores, eval_rows, units_eval, rng):
    """Placements of official crash hours among sampled units of the same block; within-set; level."""
    kind = test["kind"].to_numpy()
    m, c = (kind == "marginal") & eval_rows, (kind == "official_case") & eval_rows
    sets = np.isin(kind, ["official_case", "official_control"]) & eval_rows
    out = {}
    for name, s in scores.items():
        pl = placements(s[c], np.sort(s[m]))
        sub = test.loc[sets]
        w = within_stratum(sub, s[sets], (sub["kind"] == "official_case").to_numpy())
        level = float(hourly(s[m]).mean() * units_eval / 12 / c.sum())
        out[name] = {"placements": pl, "within": w, "level": level}
    return out


def summarize(blocks: list[dict], rng) -> dict:
    names = blocks[0].keys()
    pooled = {n: {"placements": np.concatenate([b[n]["placements"] for b in blocks]),
                  "within": np.concatenate([b[n]["within"] for b in blocks])} for n in names}
    res = {"arms": {}, "paired": {}}
    for n in names:
        res["arms"][n] = {"population_auc": float(pooled[n]["placements"].mean()), "ci95": ci(pooled[n]["placements"], rng),
                          "within_set": float(pooled[n]["within"].mean()), "within_ci95": ci(pooled[n]["within"], rng),
                          "level_by_block": [b[n]["level"] for b in blocks]}
    for a, b in [("pn_union", "pn_reliable"), ("pn_union", "pn_official_hour"), ("mswspu_credibility", "pn_official_hour"),
                 ("nnpu_reliable", "pn_reliable"), ("mswspu_credibility", "pn_union"), ("hgb_union", "hgb_official_hour")]:
        dp = pooled[a]["placements"] - pooled[b]["placements"]
        dw = pooled[a]["within"] - pooled[b]["within"]
        res["paired"][f"{a}_minus_{b}"] = {"population_delta": float(dp.mean()), "ci95": ci(dp, rng),
                                           "within_delta": float(dw.mean()), "within_ci95": ci(dw, rng)}
    return res


def part_d(units: dict) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED + 1)
    data = {}
    for year in (2024, 2025):
        d = pd.read_parquet(V2 / "phase6" / f"minute_dataset_{year}.parquet")
        d["official_confirmed"] = d["official_confirmed"].fillna(False).astype(bool)
        d["fold"] = road(d["route"]).map(FOLDS)
        assert d["fold"].notna().all(), d.loc[d["fold"].isna(), "route"].unique()
        data[year] = d
    out = {"folds": {str(k): sorted(r for r, f in FOLDS.items() if f == k) for k in range(4)}}

    # Spatial hold-out: train 2024 on the other motorways, evaluate 2025 on the held-out ones.
    train, test = data[2024], data[2025]
    blocks, fold_info = [], {}
    for k in range(4):
        tr_rows = (train["fold"] != k).to_numpy()
        ev_rows = (test["fold"] == k).to_numpy()
        share_tr = ((train["kind"] == "marginal") & tr_rows).sum() / (train["kind"] == "marginal").sum()
        share_ev = ((test["kind"] == "marginal") & ev_rows).sum() / (test["kind"] == "marginal").sum()
        scores, info, pri = fit_arms(train, test, units[2024] * share_tr, tr_rows, device)
        blocks.append(evaluate_block(test, scores, ev_rows, units[2025] * share_ev, rng))
        fold_info[k] = {"train_unit_share": float(share_tr), "eval_unit_share": float(share_ev),
                        "eval_official_crash_hours": int(((test["kind"] == "official_case").to_numpy() & ev_rows).sum()), **info}
        print(f"spatial fold {k} done", flush=True)
    out["spatial_holdout"] = summarize(blocks, rng)
    out["spatial_holdout"]["folds"] = fold_info

    # Reverse years: train 2025, evaluate 2024 (all edges).
    tr_rows = np.ones(len(data[2025]), bool)
    scores, info, pri = fit_arms(data[2025], data[2024], units[2025], tr_rows, device)
    block = evaluate_block(data[2024], scores, np.ones(len(data[2024]), bool), units[2024], rng)
    out["reverse_years"] = summarize([block], rng)
    out["reverse_years"]["fits"] = info
    print("reverse years done", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = json.loads((V2 / "phase6" / "minute_dataset_summary.json").read_text())
    units = {y: summary["years"][str(y)]["population_edge_5min_units"] for y in (2024, 2025)}
    result = part_abc(units)
    (OUT / "checks_abc.json").write_text(json.dumps(result, indent=2))
    print("A-C done", flush=True)
    result_d = part_d(units)
    (OUT / "checks_d.json").write_text(json.dumps(result_d, indent=2))
    print(f"elapsed {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
