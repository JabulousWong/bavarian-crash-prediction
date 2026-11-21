"""Fixed-iteration HGB label sensitivity on the frozen 100 m cohort."""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from phase3_network_sensitivity import (
    ROOT, PHASE2, TREATMENTS, digest, fit_comparison, paired_bootstrap, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--strict-cohort", action="store_true",
                        help="Retain whole risk sets within capture envelope with flow available")
    args = parser.parse_args()
    out = ROOT / "analysis/results/phase3" / ("strict_cohort_sensitivity" if args.strict_cohort else "early_stopping_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    paths = [PHASE2 / "datasets/matched_edge_hours_2024_100m_union.parquet",
             PHASE2 / "datasets/matched_edge_hours_2025_100m.parquet",
             ROOT / "analysis/results/phase3/network_sensitivity/100m/metrics.json"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    train, test = (pd.read_parquet(p) for p in paths[:2])
    cohort_audit = {}
    if args.strict_cohort:
        import json
        coverage_path = ROOT / "analysis/results/phase1/source_summary.json"
        hashes[str(coverage_path.relative_to(ROOT))] = digest(coverage_path)
        paths.append(coverage_path)
        coverage = {v["year"]: v for v in json.loads(coverage_path.read_text(encoding="utf-8"))}
        def restrict(frame, year):
            start = pd.Timestamp(coverage[year]["coverage_start_local"]).tz_localize(None).ceil("h")
            end = pd.Timestamp(coverage[year]["coverage_end_local"]).tz_localize(None).floor("h")
            eligible = frame.outcome_hour.ge(start) & frame.outcome_hour.lt(end)
            eligible &= ~frame[["from_n_kfz", "to_n_kfz"]].isna().all(axis=1)
            keep = eligible.groupby(frame.stratum_id).transform("all")
            result = frame.loc[keep].reset_index(drop=True)
            assert result.groupby("stratum_id").size().eq(5).all()
            cohort_audit[str(year)] = {"original_rows": len(frame), "retained_rows": len(result),
                                       "retained_risk_sets": result.stratum_id.nunique(),
                                       "capture_envelope": [start.isoformat(), end.isoformat()]}
            return result
        train, test = restrict(train, 2024), restrict(test, 2025)
    for frame in (train, test):
        assert frame.feature_hour.eq(frame.outcome_hour - pd.Timedelta(hours=1)).all()
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        print("Fitting fixed 200-iteration HGB and logistic comparators", flush=True)
        results, predictions = fit_comparison(train, test, hgb_early_stopping=False)
        predictions.to_parquet(out / "test_predictions_2025.parquet", index=False)
        write_json(out / "metrics.json", results)
        print(f"Running {args.replicates} paired risk-set bootstrap draws", flush=True)
        bootstrap = paired_bootstrap(predictions, args.replicates)
    write_json(out / "paired_bootstrap.json", bootstrap)
    import json
    original = json.loads(paths[2].read_text(encoding="utf-8"))
    unchanged = all(digest(p) == hashes[str(p.relative_to(ROOT))] for p in paths)
    assert unchanged
    summary = {
        "design": {"train_year": 2024, "test_year": 2025, "threshold_m": 100,
                   "hgb_early_stopping": False, "hgb_iterations": 200,
                   "soft_q": 0.5, "test_endpoint": "matched official injury crashes",
                   "purpose": "Remove random validation split of duplicated soft-label rows",
                   "test_status": "previously examined; exploratory robustness, not fresh blind validation"},
        "metrics": results, "bootstrap": bootstrap,
        "strict_cohort": args.strict_cohort, "cohort_audit": cohort_audit,
        "source_sha256": hashes, "inputs_unchanged": unchanged,
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                        [Path(__file__), ROOT / "analysis/phase3_network_sensitivity.py",
                         ROOT / "analysis/phase2_label_comparison.py"]},
        "python": platform.python_version(), "elapsed_seconds": time.perf_counter()-start,
    }
    write_json(out / "summary.json", summary)
    lines = ["# Sensitivity to HGB internal early stopping", "",
             "Fixed 100 m data, trained on 2024 and evaluated on the 2025 official injury case-control data. HGB internal random early stopping is disabled and all models train for 200 iterations; features, class balancing and the soft weight q=0.50 are unchanged. Logistic regression serves as a reproduction control.",
             "", "Soft labels use weighted positive and negative copies of the same feature row; the default random validation split can put the copies into both training and validation. Disabling early stopping removes this mechanism, but since this experiment also changes the number of iterations, the difference cannot be attributed entirely to the duplicated rows. Imputation follows the existing procedure.",
             "", "| Label | Original HGB ROC-AUC | No early stopping ROC-AUC | Original AP | No early stopping AP |",
             "|---|---:|---:|---:|---:|"]
    if args.strict_cohort:
        lines[0] = "# Sensitivity to the complete collection envelope and traffic availability"
        lines.insert(2, "Only complete 1:4 risk sets are kept in which every member lies inside the complete collection envelope and has flow at at least one end. HGB early stopping stays disabled. The original and current results below differ in both cohort and early stopping and cannot be read as a single-mechanism comparison; the cohort filter is no proof of continuous collection and may introduce availability selection bias.")
        lines.insert(3, json.dumps(cohort_audit, ensure_ascii=False))
    for treatment in TREATMENTS:
        old = original[treatment]["models"]["hist_gradient_boosting"]
        new = results[treatment]["models"]["hist_gradient_boosting"]
        lines.append(f"| {treatment} | {old['roc_auc']:.5f} | {new['roc_auc']:.5f} | {old['pr_auc_matched_prevalence']:.5f} | {new['pr_auc_matched_prevalence']:.5f} |")
    lines += ["", "| HGB contrast without early stopping | ΔROC-AUC [95% CI] | ΔAP [95% CI] |",
              "|---|---:|---:|"]
    for name, values in bootstrap["comparisons"].items():
        if "hist_gradient_boosting" not in name:
            continue
        cells = []
        for metric in ("roc_auc", "pr_auc"):
            v = values[metric]
            cells.append(f"{v['point_delta']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", f"{args.replicates} paired risk-set bootstrap replicates with fixed models. Intervals exclude retraining uncertainty, spatio-temporal correlation across risk sets and multiple-comparison correction. AP reflects the artificial 20% case rate, not population-level warning performance; 2025 has been inspected, so this experiment cannot be used to choose the best number of iterations and then claim independent validation.",
              "", f"Input hashes unchanged: {unchanged}; run time {summary['elapsed_seconds']:.1f} s.",
              "", "Reproduction: `python analysis/phase3_early_stopping.py --replicates 1000" + (" --strict-cohort" if args.strict_cohort else "") + "`."]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
