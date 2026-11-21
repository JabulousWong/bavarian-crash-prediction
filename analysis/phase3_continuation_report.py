"""Render additional Phase 3 evidence directly from saved results."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "analysis/results/phase3"


def evidence_lines():
    lines = ["", "## Phase 3 follow-up experiments", ""]
    for directory in ("early_stopping_sensitivity", "strict_cohort_sensitivity"):
        early_path = OUT / directory / "summary.json"
        if not early_path.exists():
            continue
        early = json.loads(early_path.read_text(encoding="utf-8"))
        lines += ["### " + ("Complete collection envelope and traffic availability" if early.get("strict_cohort") else "HGB internal early stopping disabled"), "",
                  "Frozen 100 m data, trained on 2024 and evaluated on 2025; HGB for all three labellings is trained for a fixed 200 iterations without the internal random validation split. Duplicated soft-label rows can straddle the default train/validation split; since this experiment also changes the number of iterations, it is not an isolated causal estimate of that mechanism.",
                  "", "| Label contrast | ΔROC-AUC [95% CI] | ΔAP [95% CI] |",
                  "|---|---:|---:|"]
        for name, values in early["bootstrap"]["comparisons"].items():
            if "hist_gradient_boosting" not in name:
                continue
            formatted = []
            for metric in ("roc_auc", "pr_auc"):
                v = values[metric]
                formatted.append(f"{v['point_delta']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")
            lines.append(f"| {name} | {' | '.join(formatted)} |")
        if early.get("strict_cohort"):
            lines += ["", "Filtered to complete risk sets: every row lies inside the first/last observation envelope of the source and has flow at at least one end. This filter does not prove continuous collection and may introduce availability selection bias."]
            for year, audit in early["cohort_audit"].items():
                lines.append(f"{year}: {audit['original_rows']:,} rows → {audit['retained_rows']:,} rows, {audit['retained_risk_sets']:,} risk sets kept.")
        lines += ["", f"Intervals come from a paired risk-set bootstrap of fixed models, without retraining or spatio-temporal correlation across risk sets; AP reflects the artificial case rate. 2025 has already been inspected, so these are exploratory robustness results. See the [experiment report]({directory}/REPORT.md)."]
    grid_path = OUT / "pu_simulation_rare_grid/simulation_summary.json"
    if grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        cfg = grid["configuration"]
        lines += ["", "### PU simulation over several rare-event rates", "",
                  f"Each prior × four mechanisms × {cfg['repetitions']} seeds; training population {cfg['n_train']:,}, marginal U {cfg['n_u']:,}, independent test population {cfg['n_test']:,}. {grid['optimizer']['fits']:,} fits in total, {grid['optimizer']['failed_fits']} optimiser failures.",
                  "", "The priors are synthetic scenario parameters, not estimates of the actual local crash prior. The original training population of 150,000 had too few observed positives at the lowest prior, so the run was stopped and enlarged to 500,000 for all priors; the planned 30 seeds were kept and no difficult seed was dropped.",
                  "", "| Prior | Mechanism | Official AUC | Correct PU AUC | Official mean risk | Correct PU mean risk | Matched U − equal marginal U ΔAUC [95% MC CI] |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for key, models in grid["metrics"].items():
            prior, scenario = key.removeprefix("prior_").split("__")
            official = models["official_supervised"]
            pu = models["nnpu_official_marginal_true_prior"]
            d = grid["paired_differences"][key]["nnpu_official_matched_1to4_u_minus_nnpu_official_equal_budget_marginal_u"]["roc_auc"]
            ci = d["mc_mean_ci95"]
            lines.append(f"| {prior} | {scenario} | {official['roc_auc']['mean']:.5f} | {pu['roc_auc']['mean']:.5f} | {official['mean_prediction']['mean']:.7f} | {pu['mean_prediction']['mean']:.7f} | {d['mean']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
        lines += ["", "Intervals are Monte Carlo intervals of the mean across seeds. At rare rates the fixed synthetic false-positive mechanism can produce low Waze precision, which must not be read as a real Waze measurement. The SAR and contaminated-positive scenarios violate the standard PU assumptions; a correct global prior does not remove these biases. See the [full simulation](pu_simulation_rare_grid/REPORT.md).",
                  "", "Reproduction: `python analysis/phase3_pu_simulation.py --repetitions 30 --n-train 500000 --n-u 50000 --n-test 250000 --priors 0.0002 0.0005 0.0015 --out analysis/results/phase3/pu_simulation_rare_grid`."]
    pilot_path = OUT / "pu_guided_oversampling/summary.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        diagnostic = json.loads((OUT / "pu_guided_oversampling/diagnostics.json").read_text(encoding="utf-8"))
        lines += ["", "### Prototype of PU-confidence-guided oversampling", "",
                  "Cross-fitted PU scores select the top-1% candidates, choose neighbours by score and assign synthetic weights. The original P and the independent marginal U are unchanged; augmentation enters only an additional auxiliary loss. The soft interpolated label is an untested assumption; the modified objective does not inherit the consistency guarantee of PU.",
                  "", f"Three mechanisms with {pilot['configuration']['repetitions']} seeds each and three preset auxiliary strengths; {pilot['final_fits']:,} final-model fits, {pilot['failed_final_fits']} failed, {pilot['failed_teacher_fits']} teacher failures. All strengths are reported; no best method is chosen from the test truth.",
                  "", "| Mechanism | True-positive share of top-1% candidates | True-positive share of selected neighbours |",
                  "|---|---:|---:|"]
        for scenario in sorted({d["scenario"] for d in diagnostic}):
            selected = [d for d in diagnostic if d["scenario"] == scenario]
            candidate = sum(d["candidate_purity"] for d in selected)/len(selected)
            neighbor = sum(d["chosen_neighbor_purity"] for d in selected)/len(selected)
            lines.append(f"| {scenario} | {candidate:.5%} | {neighbor:.5%} |")
        lines += ["", "A relatively high score is not a high crash probability. With rare events, the top-1% candidates can still be mostly negatives; the truth is used only for the audit above, not for the generation rules.",
                  "", "| Mechanism / strength / contrast | ΔROC-AUC [95% MC CI] |",
                  "|---|---:|"]
        for key, values in pilot["paired_differences"].items():
            if not (key.endswith("minus_smote_positive") or key.endswith("minus_pu_guided_real_endpoints") or key.endswith("minus_pu_base")):
                continue
            v = values["roc_auc"]
            ci = v["mc_mean_ci95"]
            lines.append(f"| {key} | {v['mean']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
        lines += ["", "This is a mechanism experiment with continuous Gaussian simulation and linear models, without real road or time constraints. The contamination mechanism deliberately uses the source union including false positives as P; the PU prior, target-crash definition and collection mechanism of the real data are not yet validated. See the [full prototype experiment](pu_guided_oversampling/REPORT.md)."]
    return lines


if __name__ == "__main__":
    import hashlib
    import pandas as pd
    import numpy as np
    grid_path = OUT / "pu_simulation_rare_grid/simulation_summary.json"
    manifest = {"results": {}}
    if grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(OUT / "pu_simulation_rare_grid/per_seed_metrics.csv")
        config = grid["configuration"]
        cells = frame.groupby(["prior", "scenario", "model"]).seed.nunique()
        assert cells.eq(config["repetitions"]).all()
        assert len(cells) == len(config["priors"])*len(config["scenarios"])*12
        assert not frame.duplicated(["prior", "scenario", "model", "seed"]).any()
        assert np.isfinite(frame[list(grid["metrics"][next(iter(grid["metrics"]))]["official_supervised"])].to_numpy()).all()
        assert int((~frame.converged).sum()) == grid["optimizer"]["failed_fits"]
        manifest["simulation_validation"] = {"rows": len(frame), "cells": len(cells),
                                             "seeds_per_cell": config["repetitions"],
                                             "duplicate_records": False,
                                             "failed_fits": grid["optimizer"]["failed_fits"]}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        for ax, scenario in zip(axes, ("scar_clean", "sar_clean")):
            priors = sorted(config["priors"])
            ax.plot(priors, priors, "--", color="0.5", label="True prior")
            for model, label in (("official_supervised", "Official supervised"),
                                 ("nnpu_official_marginal_true_prior", "Correct-prior PU")):
                stats = [grid["metrics"][f"prior_{p:g}__{scenario}"][model]["mean_prediction"] for p in priors]
                means = [v["mean"] for v in stats]
                errors = [[v["mean"]-v["mc_mean_ci95"][0] for v in stats],
                          [v["mc_mean_ci95"][1]-v["mean"] for v in stats]]
                ax.errorbar(priors, means, yerr=errors, marker="o", capsize=3, label=label)
            ax.set_title(scenario)
            ax.set_xlabel("Synthetic generating prior")
            ax.set_ylabel("Mean predicted probability")
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("Synthetic rare-event calibration: 95% Monte Carlo mean intervals")
        fig.savefig(OUT / "pu_simulation_rare_grid/calibration.png", dpi=180)
        fig.savefig(OUT / "pu_simulation_rare_grid/calibration.svg")
        plt.close(fig)
    for directory in ("early_stopping_sensitivity", "strict_cohort_sensitivity", "pu_simulation_rare_grid", "pu_guided_oversampling"):
        for result in sorted((OUT / directory).glob("*")):
            if result.is_file():
                manifest["results"][str(result.relative_to(OUT))] = hashlib.sha256(result.read_bytes()).hexdigest()
    manifest["code_sha256"] = {str(p.relative_to(OUT.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob("phase3_*.py")}
    (OUT / "continuation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    path = OUT / "CONTINUATION.md"
    path.write_text("# Phase 3 follow-up results\n" + "\n".join(evidence_lines()) + "\n", encoding="utf-8")
    print(path)
