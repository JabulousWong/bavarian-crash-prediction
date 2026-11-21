"""Build the Phase 3 evidence report from saved, auditable results."""
from pathlib import Path
import json
import hashlib
import platform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results/phase3"


def load(relative):
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def main():
    network = load("network_sensitivity/network_sensitivity.json")
    population = load("population/population_summary.json")
    prior = load("class_prior/class_prior_constraints.json")
    s100 = prior["training_prior_constraints_2024"]["100"]; ts = s100["transport_scenarios"]
    simulation = load("pu_simulation/simulation_summary.json")
    lines = ["# Phase 3: official statistical constraints, corridor sensitivity and PU method checks", "",
        "Current evidence: Waze labels give a small cross-year ranking improvement that is reasonably robust to the corridor distance threshold; this is not yet enough to make PU the main method. Completed here: proxy-corridor sensitivity, extraction of the official statistics, label-independent marginal sampling of the population, and known-truth mechanism simulations. Validation against the real operated network and identification of the all-crash prior for the target corridor are still open.", "",
        "The crash statistics used are official counts. What is unknown is how these known totals map onto the edge-hour population of this study, not the Bavarian crash totals themselves. Raw data and Phase 2 results are kept.", "",
        "## Official crash statistics", "",
        "| Scope | 2024 | 2025 |", "|---|---:|---:|"]
    for scope, text in (("bavaria_all_roads", "Bavaria, all roads"), ("bavaria_motorways", "Bavaria, motorways")):
        for field, label in (("total", "all crashes"), ("injury", "injury crashes"), ("pdo_total", "property-damage-only crashes")):
            a, b = (prior["actual_counts"][y][scope][field] for y in ("2024", "2025"))
            lines.append(f"| {text}: {label} | {a:,} | {b:,} |")
    lines += ["", "All-road counts come from the h1101c December files, cells S.7!J14/J15/J17; motorway injury and serious property-damage crashes from the h1100d annual files, cells 21!C34/K34. All motorway crashes and the remaining property-damage crashes were checked against [GENESIS 46241-0020](https://genesis.destatis.de/datenbank/online/statistic/46241/table/46241-0020). Item-level sources and file SHA256 are in [statistical constraints](class_prior/class_prior_constraints.json) and [count CSV](class_prior/actual_official_counts.csv).", "",
        "Exported 2,496 rows of [district motorway detail](class_prior/district_motorway_monthly.csv) for 24 months, including districts, cities, administrative regions and state totals, so the table must not be summed as a whole. In the district table, insgesamt excludes the remaining/minor property-damage crashes; its property-damage column covers serious property damage and other specified crashes such as drink-driving. Monthly first releases are revised later: the monthly files sum to 2,771 motorway injury crashes in 2024 against an annual value of 3,236, and to 2,825 against 3,199 in 2025. Annual analyses use the cumulative/annual figures.", "",
        "## 50 / 100 / 150 m proxy-corridor sensitivity", "",
        "All models are trained on 2024 and evaluated on the 2025 official injury case-control data of the respective threshold. Within a threshold, all labelling methods share the test rows. Models and the soft weight q=.50 keep their preset configuration.", "",
        "| Distance | Official HGB ROC-AUC | Hard union HGB ROC-AUC | Soft weight HGB ROC-AUC | Hard union − official ΔROC-AUC [95% CI] | ΔAP [95% CI] |",
        "|---:|---:|---:|---:|---|---|"]
    for threshold, entry in network["thresholds"].items():
        values = [entry["specifications"][name]["models"]["hist_gradient_boosting"]["roc_auc"]
                  for name in ("official_only", "hard_union", "soft_waze_q_0.50")]
        contrast = entry["bootstrap"]["comparisons"]["hard_union_hist_gradient_boosting_minus_official_only_hist_gradient_boosting"]
        deltas = [f"{contrast[k]['point_delta']:+.4f} [{contrast[k]['ci95'][0]:+.4f}, {contrast[k]['ci95'][1]:+.4f}]"
                  for k in ("roc_auc", "pr_auc")]
        lines.append(f"| {threshold}m | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {' | '.join(deltas)} |")
    lines += ["", "![Proxy-corridor threshold sensitivity](network_sensitivity.png)", "", "1,000 paired risk-set bootstrap replicates per threshold. At all three thresholds the hard-union interval lies above zero, but the lower bound of the ROC difference at 150 m is close to zero; these are per-item conditional intervals without multiple-comparison correction. The ROC/AP intervals of soft weight versus hard union all cross zero. Absolute predictive ability remains weak and does not support claims of high-performance warning.", "",
        "AP (called PR-AUC in earlier outputs) reflects the artificial 20% case rate; ROC-AUC is also bounded by the matched case/control distribution. Neither these metrics nor uncalibrated scores represent population-level warning performance. The bootstrap does not cover retraining uncertainty or edge/time correlation across risk sets. 2025 has already been inspected, so later sensitivity analyses cannot be called fresh blind tests.", "",
        "At 100 m the HGB predictions of all three labellings match Phase 2 row by row; for logistic regression the maximum probability difference is 2.08e-8 due to thread count, within the 1e-7 tolerance. Phase 2 input hashes are unchanged. See the [sensitivity report](network_sensitivity/REPORT.md).", "",
        "## Correct marginal sampling for PU", "",
        "The population is rebuilt from all 1,061 inherited corridor edge IDs: edges are not selected by crash occurrence, cases are not matched, crashes and adjacent hours are not excluded, and the population is not filtered by missing features. Units are drawn with equal probability without replacement and source labels are attached afterwards. Known positives in U are kept: U estimates p(X) and must not be read as known negatives.", "",
        "| Year | Usable hours | Population edge-hours | Sampled | 100 m official positives in sample |",
        "|---:|---:|---:|---:|---:|"]
    for year, entry in population["years"].items():
        lines.append(f"| {year} | {entry['eligible_outcome_hours']:,} | {entry['population_edge_hours']:,} | {entry['sample_edge_hours']:,} | {entry['thresholds']['100']['sample_official_edge_hours']} |")
    lines += ["", "The original cache had 943 sites per year; this run adds 44 sites and 70 detector channels to reach 987 sites, 140 new channel-years in total. The added cache is stored separately. Sample files store inclusion probabilities and inverse-probability weights; official positives are additionally stored as a full table, so their known count is not estimated from the few sampled positives. 2024 is used for training/priors; the 2025 files only for locked evaluation or coverage audits.", "",
        "Only prediction hours fully inside the first/last Waze observation envelope are kept; windows around daylight-saving changes where local clock ambiguity cannot be resolved are excluded. Raw traffic timestamps carry no time zone and are currently interpreted as local Europe/Berlin, which still needs confirmation from the provider. The observation envelope is not proof of complete hourly collection.", "",
        "The sample is valid for the declared edge-ID population, which is not yet a verified physical road population: the original geometry contains 44 duplicate WKTs, 48 duplicate unordered site pairs and one zero-length edge. These IDs are kept for compatibility with the existing labels, so the denominator must not be read as the de-duplicated operated network of southern Bavaria. No directly usable shp/gpkg/geojson/osm road file was available. The [operator page](https://www.autobahn.de/ueber-uns/niederlassungen/suedbayern) has a map of the operated area that can be used for the next check.", "",
        "Files: [population summary](population/population_summary.json), [2024 marginal sample](population/marginal_edge_hours_2024.parquet), [2025 marginal sample](population/marginal_edge_hours_2025.parquet).", "",
        "## Class prior: official values and conditions for local identification", "",
        "Define π = P(at least one target crash in a population edge-hour). Neither crash counts, the injury share of all crashes, nor the case share of the 1:4 sample can be plugged in for π directly. Geography, time, crash severity and multiple crashes per hour have to be aligned first.", "",
        f"In 2024, {s100['observed_injury_crash_records']:,} official crashes were observed in the current 100 m labelling / declared population, merged into {s100['observed_injury_edge_hours']:,} positive hours; the observed fraction is {s100['observed_injury_edge_hours']:,}/{s100['population_edge_hours']:,} = {s100['observed_injury_edge_hour_fraction']:.9f} ({100*s100['observed_injury_edge_hour_fraction']:.5f}%). This is the computable observation rate of the source.", "",
        f"The 2024 ratio of all police-recorded crashes to injury crashes on Bavarian motorways is {ts['all_police_recorded_crashes']['actual_2024_motorway_count_ratio']:.6f}; adding only serious property damage gives {ts['injury_plus_serious_pdo']['actual_2024_motorway_count_ratio']:.6f}. Assuming in addition that the regional ratio transfers, the injury sample is complete and spatial assignment is correct, conditional occupancy ranges for the 100 m population follow: injury + serious property damage [{ts['injury_plus_serious_pdo']['conditional_occupancy_lower']:.9f},{ts['injury_plus_serious_pdo']['conditional_occupancy_upper']:.9f}], all police-recorded crashes [{ts['all_police_recorded_crashes']['conditional_occupancy_lower']:.9f},{ts['all_police_recorded_crashes']['conditional_occupancy_upper']:.9f}]. The lower bound lets the additional crashes fall into existing positive hours, the upper bound puts each into a new hour. These are neither confidence intervals nor an assumption-free identification of the local prior.", "",
        "The state-wide injury records in the Unfallatlas input (46,446/46,938) are fewer than the official injury statistics (49,374/49,925); the gap needs further auditing and must not simply be closed with a multiplier. The 2025 statistics are used only as a descriptive check, not to choose the 2024 training prior.", "",
        "## Known-truth simulation and choice of the main method", "",
        f"Completed 30 random seeds × 2 example priors × 4 collection mechanisms; {simulation['optimizer']['fits']:,} fits, {simulation['optimizer']['failed_fits']} failed. Compared: official, hard union, soft weight, PU with correct and wrong prior, U with events wrongly excluded, 1:4 matched U, and the true model.", "",
        "In the SCAR scenario with a 1% example prior, the main gain of correct nnPU is calibration: the mean predicted rate moves from about 0.00289 to 0.01002 while AUC stays almost unchanged at about 0.698. Using 1:4 matched controls as U instead of an equally sized correct marginal U changes AUC by −0.0531 (Monte Carlo mean 95% interval [−0.0598, −0.0465]). When positive collection depends on features or Waze has correlated false positives, the correct global prior does not repair these biases.", "",
        "These are mechanism checks on synthetic data and cannot replace validation with real property-damage labels. The default simulated priors of 1%/5% are higher than the rare-crash magnitude implied by the statistical constraints; the simulation does not reproduce cross-year drift, crash severity, map errors or spatio-temporal correlation. Analytic gradients and the finite-population PU risk identity pass numerical checks; the low-capacity models never triggered the non-negative correction, so nnPU cannot be claimed to beat uPU from this. [Simulation report](pu_simulation/REPORT.md).", "",
        "Current recommendation: the main line keeps the cross-year comparison and robustness analysis of source-label fusion; PU is an extension/appendix with stated assumptions. PU can only become the main method once the real network population, the target-crash prior, the positive selection mechanism and a matching validation target are supported. Soft-weight labels are not treated as PU here, and the official injury test is not used to claim detection of all crashes.", "",
        "## Reproduction", "", "```bash",
        "python analysis/phase3_population.py --workers 2",
        "python analysis/phase3_class_prior.py",
        "python analysis/phase3_network_sensitivity.py --replicates 1000",
        "python analysis/phase3_pu_simulation.py",
        "python analysis/phase3_report.py",
        "```", "",
        "Original PU methods: [Elkan & Noto 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf); [Kiryo et al. 2017](https://papers.nips.cc/paper/6765-positive-unlabeled-learning-with-non-negative-risk-estimator.pdf)."]
    rare = OUT / "pu_simulation_rare/simulation_summary.json"
    if rare.exists():
        value = json.loads(rare.read_text())
        lines += ["", "A rare-event magnitude simulation was added (prior 0.001, a magnitude scenario only, not an estimate of the actual prior):",
                  f"{value['optimizer']['fits']:,} fits, {value['optimizer']['failed_fits']} failed. Under SCAR each seed has 34–60 observed official positives and 126–175 true test events; correct PU still mainly improves mean-risk calibration (0.000305 to 0.001001), AUC about 0.681. Matched U versus an equally sized correct marginal U: ΔAUC −0.0532, Monte Carlo 95% interval [−0.0663, −0.0401]. The SAR bias remains. See the [rare-event note](pu_simulation_rare/RARE_EVENT_NOTE.md) and the [full report](pu_simulation_rare/REPORT.md).",
                  "The synthetic false-positive rate carried over into this scenario lowers Waze precision to 11.9%; this is an artefact of the rare-rate setting and in no way an estimate of real Waze precision.",
                  "Reproduce the rare scenario: `python analysis/phase3_pu_simulation.py --repetitions 30 --n-train 150000 --n-u 50000 --n-test 150000 --priors 0.001 --out analysis/results/phase3/pu_simulation_rare`."]
    from phase3_continuation_report import evidence_lines
    lines += evidence_lines()
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # A standalone figure gives the threshold comparisons a common visual scale.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    thresholds = [50, 100, 150]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout="constrained")
    for axis, metric, title in zip(axes, ("roc_auc", "pr_auc"), ("ROC-AUC difference", "Average precision difference")):
        results = [network["thresholds"][str(t)]["bootstrap"]["comparisons"][
            "hard_union_hist_gradient_boosting_minus_official_only_hist_gradient_boosting"][metric]
            for t in thresholds]
        estimates = np.array([r["point_delta"] for r in results])
        errors = np.array([[r["point_delta"]-r["ci95"][0] for r in results],
                           [r["ci95"][1]-r["point_delta"] for r in results]])
        axis.errorbar(estimates, [0, 1, 2], xerr=errors, fmt="o", capsize=4,
                      color="#246b8a", linewidth=1.7)
        axis.axvline(0, color="0.5", linestyle="--", linewidth=1)
        axis.set_yticks([0, 1, 2], [f"{t} m" for t in thresholds])
        axis.set_xlim(-.002, .036)
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Detector-corridor threshold")
    fig.suptitle("Hard union minus official-only: 2025 matched test\n1,000 paired risk-set bootstraps; pointwise 95% intervals", fontsize=11)
    fig.savefig(OUT / "network_sensitivity.png", dpi=180)
    fig.savefig(OUT / "network_sensitivity.svg")
    plt.close(fig)
    manifest = {"python": platform.python_version(), "phase2_inputs_unchanged": network["phase2_inputs_unchanged"],
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "analysis").glob("phase3_*.py"))},
                "required_results_present": {name: (OUT / name).exists() for name in (
                    "network_sensitivity/network_sensitivity.json", "population/population_summary.json",
                    "class_prior/class_prior_constraints.json", "pu_simulation/simulation_summary.json")}}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT / "REPORT.md")


if __name__ == "__main__":
    main()
