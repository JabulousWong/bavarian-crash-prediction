"""Reproduce Phase 2 at 50/100/150 m without modifying Phase 2 artifacts.

This varies the distance to reconstructed detector chords, NOT membership in an
authoritative road network. Within each threshold all label treatments share an
official 2025 matched test cohort and the 2024 union cohort's controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Keep concurrent thesis analyses from oversubscribing the machine.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from phase2_baseline import BASE_FEATURES, features, metrics
from phase2_build_dataset import add_site_features, sample_matched
from phase2_label_comparison import model_pair, soft_expand


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "analysis" / "results" / "phase2"
OUT = ROOT / "analysis" / "results" / "phase3" / "network_sensitivity"
SEED = 20260929
TREATMENTS = ("official_only", "hard_union", "soft_waze_q_0.50")
MODEL_NAMES = ("logistic", "hist_gradient_boosting")
META_COLUMNS = ["stratum_id", "edge_id", "outcome_hour", "target_official", "source_pattern"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_dataset(threshold: int, year: int, mode: str, destination: Path) -> pd.DataFrame:
    suffix = "_union" if mode == "union" else ""
    name = f"matched_edge_hours_{year}_{threshold}m{suffix}.parquet"
    if threshold == 100:
        # These are the exact frozen rows used for the user's Phase 2 results.
        frame = pd.read_parquet(PHASE2 / "datasets" / name)
    else:
        events = pd.read_parquet(PHASE2 / f"label_events_{year}_{threshold}m.parquet")
        events["event_hour"] = pd.to_datetime(events["event_hour"])
        frame = add_site_features(sample_matched(year, events, mode), year)
    frame.to_parquet(destination / name, index=False)
    return frame


def audit_dataset(frame: pd.DataFrame, events: pd.DataFrame, coverage: dict) -> dict:
    """Validate matching and lagging; quantify inherited calendar/capture limits."""
    grouped = frame.groupby("stratum_id")["target_union"].agg(["sum", "count"])
    assert (grouped["sum"] == 1).all() and (grouped["count"] == 5).all()
    assert frame["feature_hour"].eq(frame["outcome_hour"] - pd.Timedelta(hours=1)).all()
    assert frame["outcome_hour"].dt.year.eq(frame["year"]).all()
    assert not frame.duplicated(["edge_id", "outcome_hour"]).any()
    # Source first/last reports establish only an envelope, not uninterrupted uptime.
    start = pd.Timestamp(coverage["coverage_start_local"]).tz_localize(None).ceil("h")
    end = pd.Timestamp(coverage["coverage_end_local"]).tz_localize(None).floor("h")
    within = frame["outcome_hour"].ge(start) & frame["outcome_hour"].lt(end)
    controls = frame["target_union"].eq(0)
    all_missing = frame[BASE_FEATURES].isna().all(axis=1)
    all_flow_missing = frame[["from_n_kfz", "to_n_kfz"]].isna().all(axis=1)
    event_keys = pd.MultiIndex.from_frame(events[["edge_id", "event_hour"]])
    control_keys = pd.MultiIndex.from_frame(
        frame.loc[controls, ["edge_id", "outcome_hour"]].rename(columns={"outcome_hour": "event_hour"})
    )
    assert not control_keys.isin(event_keys).any()
    return {
        "rows": len(frame),
        "strata": len(grouped),
        "official_case_edge_hours": int(frame["target_official"].sum()),
        "union_case_edge_hours": int(frame["target_union"].sum()),
        "control_edge_hours": int(controls.sum()),
        "unique_edges": int(frame["edge_id"].nunique()),
        "rows_with_all_traffic_missing": int(all_missing.sum()),
        "rows_missing_flow_at_both_sites": int(all_flow_missing.sum()),
        "rows_with_complete_traffic": int(frame[BASE_FEATURES].notna().all(axis=1).sum()),
        "strict_complete_capture_envelope": [start.isoformat(), end.isoformat()],
        "rows_outside_capture_envelope": int((~within).sum()),
        "cases_outside_capture_envelope": int((~within & ~controls).sum()),
        "controls_outside_capture_envelope": int((~within & controls).sum()),
        "invariants": {
            "exact_1_to_4_risk_sets": True,
            "features_precede_outcome_by_one_hour": True,
            "unique_edge_hour_rows": True,
            "no_observed_event_controls": True,
        },
    }


def fit_comparison(train: pd.DataFrame, test: pd.DataFrame, *, hgb_early_stopping=None) -> tuple[dict, pd.DataFrame]:
    official = train.loc[train["target_union"].eq(0) | train["official_observed"]].copy()
    specifications = {
        "official_only": (official, official["official_observed"].astype(int).to_numpy(), np.ones(len(official))),
        "hard_union": (train, train["target_union"].astype(int).to_numpy(), np.ones(len(train))),
        "soft_waze_q_0.50": soft_expand(train, 0.50),
    }
    results = {}
    predictions = test[META_COLUMNS].copy()
    x_test, y_test = features(test), test["target_official"].to_numpy()
    for treatment, (frame, y_train, weights) in specifications.items():
        x_train = features(frame)
        positive, negative = weights[y_train == 1].sum(), weights[y_train == 0].sum()
        fit_weights = weights.copy()
        fit_weights[y_train == 1] *= (positive + negative) / (2 * positive)
        fit_weights[y_train == 0] *= (positive + negative) / (2 * negative)
        results[treatment] = {
            "train_rows_after_soft_expansion": len(frame),
            "effective_positive_weight": float(positive),
            "effective_negative_weight": float(negative),
            "models": {},
        }
        for name, model in model_pair().items():
            if name == "hist_gradient_boosting" and hgb_early_stopping is not None:
                model.set_params(model__early_stopping=hgb_early_stopping)
            model.fit(x_train, y_train, model__sample_weight=fit_weights)
            score = model.predict_proba(x_test)[:, 1]
            results[treatment]["models"][name] = metrics(y_test, score)
            predictions[f"{treatment}_{name}"] = score
    return results, predictions


def paired_bootstrap(predictions: pd.DataFrame, replicates: int) -> dict:
    """Use the same RNG/stratum ordering as Phase 2; all models share each draw."""
    groups = {key: group.index.to_numpy() for key, group in predictions.groupby("stratum_id")}
    keys = np.array(list(groups), dtype=object)
    columns = [f"{t}_{m}" for t in TREATMENTS for m in MODEL_NAMES]
    y_all = predictions["target_official"].to_numpy()
    scores = predictions[columns].to_numpy()
    estimates = np.empty((replicates, len(columns), 2))
    rng = np.random.default_rng(SEED)
    for replicate in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in selected])
        y = y_all[indices]
        for column in range(len(columns)):
            score = scores[indices, column]
            estimates[replicate, column] = roc_auc_score(y, score), average_precision_score(y, score)
    comparisons = {}
    pairs = [("hard_union", "official_only"), ("soft_waze_q_0.50", "official_only"), ("soft_waze_q_0.50", "hard_union")]
    for model in MODEL_NAMES:
        for treatment, reference in pairs:
            candidate_col, reference_col = f"{treatment}_{model}", f"{reference}_{model}"
            c, r = columns.index(candidate_col), columns.index(reference_col)
            comparison = {}
            for metric_i, metric_name in enumerate(("roc_auc", "pr_auc")):
                metric = roc_auc_score if metric_i == 0 else average_precision_score
                deltas = estimates[:, c, metric_i] - estimates[:, r, metric_i]
                comparison[metric_name] = {
                    "point_delta": float(metric(y_all, scores[:, c]) - metric(y_all, scores[:, r])),
                    "mean_delta": float(deltas.mean()),
                    "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
                    "bootstrap_probability_delta_gt_0": float((deltas > 0).mean()),
                }
            comparisons[f"{candidate_col}_minus_{reference_col}"] = comparison
    return {
        "replicates": replicates,
        "seed": SEED,
        "resampling_unit": "matched risk-set stratum",
        "inference": "conditional on fitted training models and this threshold-specific test cohort; no refitting",
        "comparisons": comparisons,
    }


def write_report(summary: dict) -> None:
    lines = [
        "# Detector-corridor distance sensitivity",
        "",
        "2024 training; 2025 official injury-crash 1:4 matched evaluation. "
        "Every treatment shares the same test rows within each distance threshold. "
        "Traffic features use the completed interval [h-1,h) to predict [h,h+1).",
        "",
        "| Distance | 2024 union cases | 2025 official cases | Label treatment | HGB ROC-AUC | HGB average precision |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for threshold, entry in summary["thresholds"].items():
        for treatment in TREATMENTS:
            result = entry["specifications"][treatment]["models"]["hist_gradient_boosting"]
            lines.append(f"| {threshold} m | {entry['datasets']['train_union']['union_case_edge_hours']} | {entry['datasets']['test_official']['official_case_edge_hours']} | {treatment} | {result['roc_auc']:.4f} | {result['pr_auc_matched_prevalence']:.4f} |")
    lines += ["", "Paired risk-set bootstrap differences (HGB; percentile 95% intervals):", "", "| Distance | Contrast | Delta ROC-AUC [95% CI] | Delta average precision [95% CI] |", "|---:|---|---|---|"]
    for threshold, entry in summary["thresholds"].items():
        for treatment, reference in (("hard_union", "official_only"), ("soft_waze_q_0.50", "hard_union")):
            contrast = entry["bootstrap"]["comparisons"][f"{treatment}_hist_gradient_boosting_minus_{reference}_hist_gradient_boosting"]
            cells = []
            for metric in ("roc_auc", "pr_auc"):
                result = contrast[metric]
                cells.append(f"{result['point_delta']:+.4f} [{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]")
            lines.append(f"| {threshold} m | {treatment} minus {reference} | {' | '.join(cells)} |")
    lines += ["", "Interpretation and limits:", ""]
    for note in summary["limitations"]:
        lines.append(f"- {note}")
    lines += ["", "Coverage and missingness audit:", "", "| Distance | Cohort | Rows outside complete capture envelope | Cases outside | Rows missing both flow measures |", "|---:|---|---:|---:|---:|"]
    for threshold, entry in summary["thresholds"].items():
        for cohort, audit in entry["datasets"].items():
            lines.append(f"| {threshold} m | {cohort} | {audit['rows_outside_capture_envelope']} | {audit['cases_outside_capture_envelope']} | {audit['rows_missing_flow_at_both_sites']} |")
    lines += ["", "Run with:", "", "```bash", "python analysis/phase3_network_sensitivity.py --replicates 1000", "```", "", "Machine-readable details, fitted-model predictions and exact matched rows are retained in this directory. "
              "Phase 2 source files are read only; the frozen 100 m cohort is copied into Phase 3."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[50, 100, 150], choices=[50, 100, 150])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse completed thresholds if input hashes and bootstrap settings match")
    args = parser.parse_args()
    if args.replicates < 1 or args.threads < 1:
        parser.error("replicates and threads must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((ROOT / "analysis/results/phase1/source_summary.json").read_text(encoding="utf-8"))
    coverage = {entry["year"]: entry for entry in source_summary}
    input_paths = [ROOT / "analysis/results/phase1/detector_corridor_edges.csv", ROOT / "analysis/results/phase1/source_summary.json"]
    input_paths += [PHASE2 / f"label_events_{year}_{threshold}m.parquet" for threshold in args.thresholds for year in (2024, 2025)]
    input_paths += [PHASE2 / "datasets" / name for name in ("matched_edge_hours_2024_100m_union.parquet", "matched_edge_hours_2025_100m.parquet")]
    code_paths = [Path(__file__), ROOT / "analysis/phase2_baseline.py", ROOT / "analysis/phase2_build_dataset.py", ROOT / "analysis/phase2_label_comparison.py"]
    summary = {
        "design": {
            "train_year": 2024, "test_year": 2025, "seed": SEED,
            "predictor_window": "[h-1,h)", "outcome_window": "[h,h+1)",
            "control_matching": "same edge, hour-of-day and weekday/weekend status; +/-28 days; +/-2-hour event exclusion; no control reuse",
            "official_training_controls": "all controls from threshold-specific union cohort; identical controls in every treatment",
            "models_and_weights": "unchanged Phase 2 model_pair; class-balanced fitting weights; soft q fixed at 0.50",
            "test_prevalence": 0.2,
        },
        "limitations": [
            "Distances refer to straight detector-to-detector corridor geometry. This is a proxy-scope sensitivity, not an authoritative road-network sensitivity or verified motorway membership.",
            "Thresholds change both training and test cohorts. Contrasts between label treatments are paired within threshold; differences across thresholds are descriptive and cannot be interpreted as paired effects.",
            "Average precision (reported as PR-AUC), Brier score and log loss refer to the artificial 20% case prevalence. ROC-AUC also describes these matched case/control distributions, not all population edge-hours. Scores are not calibrated population crash probabilities.",
            "The official-only comparator retains all union-cohort controls and removes Waze-only cases, matching Phase 2. It is not the separate Phase 2 official-risk-set baseline.",
            "The 1,000-replicate bootstrap resamples complete risk sets with models fixed; it does not include training uncertainty, between-risk-set temporal/spatial dependence, or correction for multiple comparisons.",
            "Soft q=0.50 is a sensitivity parameter, not an estimated Waze positive predictive value. Duplicated soft rows and default HGB internal early stopping are inherited from Phase 2.",
            "Phase 2 calendar sampling and missing-feature imputation are preserved. Waze ends on 2025-12-18; controls outside the complete capture envelope and rows missing both traffic flow measures are explicitly audited. First/last reports do not prove uninterrupted observation uptime.",
            "The already examined 2025 set is reused for this prespecified sensitivity; these results must not be used to tune the threshold and then claim a fresh independent validation.",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": args.threads},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in input_paths + code_paths},
        "thresholds": {},
    }
    with threadpool_limits(limits=args.threads):
        if args.resume and (OUT / "network_sensitivity.json").exists():
            previous = json.loads((OUT / "network_sensitivity.json").read_text(encoding="utf-8"))
            assert all(previous["source_sha256"].get(str(p.relative_to(ROOT))) == digest(p) for p in input_paths)
            for key, value in previous["thresholds"].items():
                assert value["bootstrap"]["replicates"] == args.replicates
            summary["thresholds"] = previous["thresholds"]
        for threshold in args.thresholds:
            if str(threshold) in summary["thresholds"]:
                print(f"{threshold}m: reusing completed threshold", flush=True)
                continue
            started = time.monotonic()
            destination = OUT / f"{threshold}m"
            destination.mkdir(exist_ok=True)
            print(f"{threshold}m: building matched cohorts", flush=True)
            train = load_dataset(threshold, 2024, "union", destination)
            test = load_dataset(threshold, 2025, "official", destination)
            audits = {}
            for cohort, year, frame in (("train_union", 2024, train), ("test_official", 2025, test)):
                events = pd.read_parquet(PHASE2 / f"label_events_{year}_{threshold}m.parquet")
                events["event_hour"] = pd.to_datetime(events["event_hour"])
                audits[cohort] = audit_dataset(frame, events, coverage[year])
            print(f"{threshold}m: fitting six models ({len(train)} train, {len(test)} test rows)", flush=True)
            specifications, predictions = fit_comparison(train, test)
            predictions.to_parquet(destination / "test_predictions_2025.parquet", index=False)
            write_json(destination / "metrics.json", specifications)
            print(f"{threshold}m: {args.replicates} paired risk-set bootstrap replicates", flush=True)
            bootstrap = paired_bootstrap(predictions, args.replicates)
            write_json(destination / "paired_bootstrap.json", bootstrap)
            entry = {"datasets": audits, "specifications": specifications, "bootstrap": bootstrap}
            if threshold == 100:
                original = pd.read_parquet(PHASE2 / "label_comparison/test_predictions_2025.parquet")
                assert predictions[META_COLUMNS].equals(original[META_COLUMNS])
                differences = {col: float(np.abs(predictions[col].to_numpy() - original[col].to_numpy()).max()) for col in predictions if col not in META_COLUMNS}
                entry["phase2_reproduction_max_absolute_prediction_difference"] = differences
                # BLAS thread-count changes produce ~2e-8 logistic probability
                # differences here; HGB predictions should reproduce exactly.
                entry["phase2_reproduction_numerical_tolerance"] = 1e-7
                assert max(differences.values()) < 1e-7, differences
            entry["elapsed_seconds"] = time.monotonic() - started
            summary["thresholds"][str(threshold)] = entry
            write_json(OUT / "network_sensitivity.json", summary)
            write_report(summary)
            print(f"{threshold}m: complete in {entry['elapsed_seconds']:.1f}s", flush=True)
    summary["phase2_inputs_unchanged"] = all(digest(path) == summary["source_sha256"][str(path.relative_to(ROOT))] for path in input_paths)
    assert summary["phase2_inputs_unchanged"]
    write_json(OUT / "network_sensitivity.json", summary)
    rows = []
    for threshold, entry in summary["thresholds"].items():
        for treatment, treatment_results in entry["specifications"].items():
            for model, result in treatment_results["models"].items():
                rows.append({"threshold_m": int(threshold), "treatment": treatment, "model": model, **result})
    pd.DataFrame(rows).to_csv(OUT / "metrics_summary.csv", index=False)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
