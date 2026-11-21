"""What PU learning changes when labels are incomplete: absolute risk, alerts, label hiding.

Ranking metrics cannot show it. Under SCAR the labelled-event probability is a
constant multiple of the crash probability, so treating U as negative ranks units
exactly as the PU risk does; and with a class prior near 1e-4 the positives inside
U are negligible. The PU correction acts on the level of the probability, and on
the ranking only where labelling depends on x (SAR).

Part 1 scores the existing Phase 6 arms against the complete 2025 police register
per edge-hour: expected crash count, logarithmic and Brier score, and the alerts
each arm raises at fixed absolute-risk thresholds.

Part 2 treats the police injury register as complete and removes labels from the
2024 positives before training:
  scar      every crash keeps its label with the same probability
  sar_flow  the probability rises with traffic volume (crashes in light traffic
            are reported less often); logit e(x) = a + SELECTIVITY * z(log flow)
  waze      the real mechanism: only crashes that Waze reported keep their label
Each setting trains the Phase 6 MLP as
  pn               U as negative, prior = labelled rate
  pn_rescaled      pn scores multiplied by pi / labelled rate (Elkan & Noto)
  nnpu             nnPU with the register prior pi
  nnpu_propensity  nnPU, positives weighted by 1 / e_hat(x); e_hat is a logistic
                   model of label status on the full 2024 register, the
                   information the real register gives about Waze reporting
  nnpu_propensity_oracle  the same with the true e(x) (sar_flow only)
and evaluates on the full 2025 register. The 2025 data were examined in earlier
phases, so every result here is exploratory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from phase6_minute_pu import CONTEXT, TRAFFIC, Preprocessor, fit_mlp, placements, predict, raw_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "phase6"
OUT = ROOT / "analysis" / "results" / "phase8"
SEED = 20260930
ALERT_MULTIPLES = (2, 3, 5)
LABELLED_SHARES = (1.0, 0.7, 0.5, 0.35, 0.2, 0.1)
SELECTIVITY = 1.5  # log-odds of keeping a label per standard deviation of log window flow
PROPENSITY_COLUMNS = ["hour_sin", "hour_cos", "weekend", "up_flow_w", "dn_flow_w", "up_mean_flow_2024", "length_km"]
METHOD_LABELS = {
    "pn": "PN (U as negative)", "pn_rescaled": "PN rescaled by the known prior",
    "nnpu": "nnPU, register prior", "nnpu_propensity": "nnPU + estimated propensity",
    "nnpu_propensity_oracle": "nnPU + true propensity",
}


def hourly(p: np.ndarray) -> np.ndarray:
    return 1 - (1 - np.clip(p, 0, 1)) ** 12


def log_score(h_units: np.ndarray, h_cases: np.ndarray, n_hours: float) -> float:
    """Mean logarithmic loss per edge-hour; the unit sample stands for all hours, cases correct it."""
    h_units, h_cases = np.clip(h_units, 1e-12, 1 - 1e-12), np.clip(h_cases, 1e-12, 1 - 1e-12)
    return float(-(np.log1p(-h_units).mean() + (np.log(h_cases) - np.log1p(-h_cases)).sum() / n_hours))


def brier_score(h_units: np.ndarray, h_cases: np.ndarray, n_hours: float) -> float:
    return float((h_units ** 2).mean() + (1 - 2 * h_cases).sum() / n_hours)


def score_existing(units: dict, replicates: int) -> dict:
    predictions = pd.read_parquet(DATA / "minute_pu" / "test_predictions_2025.parquet")
    fitted = json.loads((DATA / "minute_pu" / "metrics.json").read_text(encoding="utf-8"))
    prior = fitted["priors_per_unit"]["injury_from_police_register"]
    labelled = fitted["fits"]["pn_reliable__b30"]["prior"]
    unit = (predictions["kind"] == "marginal").to_numpy()
    case = (predictions["kind"] == "official_case").to_numpy()
    n_hours, n_cases = units[2025] / 12, int(case.sum())
    score = lambda name: predictions[f"{name}__b30"].to_numpy()
    arms = {
        "constant_2024_rate": np.full(len(predictions), prior),
        "pn_reliable": score("pn_reliable"),
        "pn_reliable_rescaled": np.clip(score("pn_reliable") * prior / labelled, 0, 1),
        "nnpu_reliable": score("nnpu_reliable"),
        "nnpu_reliable_propensity": score("nnpu_reliable_propensity"),
        "pn_official_hour": score("pn_official_hour"),
    }
    risk = {name: (hourly(values[unit]), hourly(values[case])) for name, values in arms.items()}
    base = float(hourly(np.array([prior]))[0])
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name in arms}
    for _ in range(replicates):
        u, c = rng.integers(0, unit.sum(), unit.sum()), rng.integers(0, n_cases, n_cases)
        reference = log_score(*[risk["constant_2024_rate"][0][u], risk["constant_2024_rate"][1][c]], n_hours)
        for name, (h_u, h_c) in risk.items():
            draws[name].append(1 - log_score(h_u[u], h_c[c], n_hours) / reference)
    reference_log = log_score(*risk["constant_2024_rate"], n_hours)
    reference_brier = brier_score(*risk["constant_2024_rate"], n_hours)
    output = {"observed_crash_hours_2025": n_cases, "edge_hours_2025": n_hours, "rate_2024_per_edge_hour": base,
              "label_frequency_reliable": labelled / prior, "arms": {}, "alerts": {}}
    for name, (h_u, h_c) in risk.items():
        output["arms"][name] = {
            "expected_crash_hours": float(h_u.mean() * n_hours),
            "predicted_over_observed": float(h_u.mean() * n_hours / n_cases),
            "log_score_x1e3": log_score(h_u, h_c, n_hours) * 1e3,
            "log_skill": 1 - log_score(h_u, h_c, n_hours) / reference_log,
            "log_skill_ci95": [float(np.quantile(draws[name], q)) for q in (0.025, 0.975)],
            "brier_skill": 1 - brier_score(h_u, h_c, n_hours) / reference_brier,
        }
        output["alerts"][name] = {}
        for multiple in ALERT_MULTIPLES:
            flagged, caught = float((h_u >= multiple * base).mean()), float((h_c >= multiple * base).mean())
            output["alerts"][name][str(multiple)] = {
                "share_of_edge_hours_flagged": flagged, "edge_hours_flagged": flagged * n_hours,
                "crash_hours_caught": caught * n_cases, "share_of_crash_hours_caught": caught,
                "observed_rate_over_base": caught * n_cases / (flagged * n_hours) / (n_cases / n_hours) if flagged else float("nan"),
                "predicted_rate_over_base": float(h_u[h_u >= multiple * base].mean() / base) if flagged else float("nan"),
            }
    return output


def hiding_experiment(units: dict, reps: int, seeds: int, replicates: int) -> tuple[pd.DataFrame, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_parquet(DATA / "minute_dataset_2024.parquet")
    test = pd.read_parquet(DATA / "minute_dataset_2025.parquet")
    raw_train, raw_test = raw_features(train), raw_features(test)
    is_u = (train["kind"] == "marginal").to_numpy()
    pos = (train["kind"] == "official_case").to_numpy()
    unit, case = (test["kind"] == "marginal").to_numpy(), (test["kind"] == "official_case").to_numpy()
    prep = Preprocessor(TRAFFIC + CONTEXT).fit(raw_train.loc[is_u])
    xu, xp = prep.transform(raw_train.loc[is_u]), prep.transform(raw_train.loc[pos])
    x_unit, x_case = prep.transform(raw_test.loc[unit]), prep.transform(raw_test.loc[case])
    prior, n_hours, n_cases = pos.sum() / units[2024], units[2025] / 12, int(case.sum())
    reference = log_score(hourly(np.full(unit.sum(), prior)), hourly(np.full(n_cases, prior)), n_hours)

    # Window flow (log scale, standardised on U) drives the biased mechanism and defines the traffic strata.
    mean, scale = raw_train.loc[is_u, "up_flow_w"].mean(), raw_train.loc[is_u, "up_flow_w"].std()
    z = lambda raw, rows: ((raw.loc[rows, "up_flow_w"] - mean) / scale).fillna(0.0).to_numpy()
    z_pos, z_unit, z_case = z(raw_train, pos), z(raw_test, unit), z(raw_test, case)
    design = Preprocessor(PROPENSITY_COLUMNS).fit(raw_train.loc[is_u])
    g_pos, g_unit, g_case = (design.transform(raw.loc[rows]) for raw, rows in
                             ((raw_train, pos), (raw_test, unit), (raw_test, case)))
    waze_train = train.loc[pos, "waze_matched"].fillna(False).astype(bool).to_numpy()
    waze_test = test.loc[case, "waze_matched"].fillna(False).astype(bool).to_numpy()

    def evaluate(p_unit, p_case, hidden, low, high) -> dict:
        h_unit, h_case = hourly(p_unit), hourly(p_case)
        place = placements(p_case, np.sort(p_unit))
        row = {"auc_all": place.mean(), "auc_hidden_type": place[hidden].mean() if hidden.any() else np.nan,
               "auc_labelled_type": place[~hidden].mean() if (~hidden).any() else np.nan,
               "predicted_over_observed": h_unit.mean() * n_hours / n_cases,
               "log_skill": 1 - log_score(h_unit, h_case, n_hours) / reference}
        for name, (rows_unit, rows_case) in (("low", low), ("high", high)):
            row[f"predicted_over_observed_{name}_third"] = h_unit[rows_unit].mean() * n_hours * rows_unit.mean() / max(rows_case.sum(), 1)
        return row, place

    settings = [("scar", share) for share in LABELLED_SHARES] + [("sar_flow", share) for share in LABELLED_SHARES if share < 1] + [("waze", None)]
    rows, waze_places, started = [], {}, time.perf_counter()
    for index, (mechanism, share) in enumerate(settings):
        for rep in range(reps):
            rng = np.random.default_rng(SEED + 1000 * index + rep)
            if mechanism == "waze":
                keep, hidden, e_pos = waze_train, ~waze_test, None
            else:
                if mechanism == "scar":
                    e_pos, e_case = np.full(len(z_pos), share), np.full(n_cases, share)
                else:
                    intercept = brentq(lambda a: expit(a + SELECTIVITY * z_pos).mean() - share, -30, 30)
                    e_pos, e_case = expit(intercept + SELECTIVITY * z_pos), expit(intercept + SELECTIVITY * z_case)
                keep, hidden = rng.random(len(z_pos)) < e_pos, rng.random(n_cases) >= e_case
            estimated = None
            if mechanism != "scar":
                model = LogisticRegression(max_iter=1000).fit(g_pos, keep)
                estimated = model.predict_proba(g_pos)[:, 1]
                stratum_unit, stratum_case = (model.predict_proba(g)[:, 1] for g in (g_unit, g_case))
            if mechanism != "waze":  # traffic strata; for Waze the strata follow the estimated reporting propensity
                stratum_unit, stratum_case = z_unit, z_case
            cut_low, cut_high = np.quantile(stratum_unit, [1 / 3, 2 / 3])
            low, high = (stratum_unit <= cut_low, stratum_case <= cut_low), (stratum_unit >= cut_high, stratum_case >= cut_high)
            labelled_rate = keep.sum() / units[2024]

            def fit(mode, weight, target_prior):
                models = [fit_mlp(xp[keep], weight, xu, target_prior, mode, SEED + 100 * rep + seed, device)[0] for seed in range(seeds)]
                return predict(models, x_unit, device), predict(models, x_case, device)

            ones = np.ones(int(keep.sum()))
            scores = {"pn": fit("pn", ones, labelled_rate)}
            scores["pn_rescaled"] = tuple(np.clip(p * prior / labelled_rate, 0, 1) for p in scores["pn"])
            scores["nnpu"] = fit("nnpu", ones, prior)
            if estimated is not None:
                scores["nnpu_propensity"] = fit("nnpu", 1 / np.clip(estimated[keep], 0.1, 1), prior)
            if mechanism == "sar_flow":
                scores["nnpu_propensity_oracle"] = fit("nnpu", 1 / np.clip(e_pos[keep], 0.05, 1), prior)
            for method, (p_unit, p_case) in scores.items():
                row, place = evaluate(p_unit, p_case, hidden, low, high)
                rows.append({"mechanism": mechanism, "labelled_share": float(keep.mean()) if share is None else share,
                             "rep": rep, "method": method, "labelled_positives": int(keep.sum()), **row})
                if mechanism == "waze":
                    waze_places.setdefault(method, []).append(place)
        print(f"{mechanism} {share}: done after {time.perf_counter() - started:.0f}s", flush=True)

    # Real mechanism: paired bootstrap over 2025 crashes on the seed-averaged placements.
    rng = np.random.default_rng(SEED)
    mean_place = {method: np.mean(values, axis=0) for method, values in waze_places.items()}
    waze_pairs = {}
    for a, b in (("nnpu", "pn"), ("nnpu_propensity", "nnpu"), ("nnpu_propensity", "pn")):
        waze_pairs[f"{a}_minus_{b}"] = {}
        for endpoint, mask in (("all", np.ones(n_cases, dtype=bool)), ("not_waze_reported", ~waze_test), ("waze_reported", waze_test)):
            delta = (mean_place[a] - mean_place[b])[mask]
            draws = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
            waze_pairs[f"{a}_minus_{b}"][endpoint] = {"delta": float(delta.mean()),
                                                      "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}
    details = {"device": device, "prior_per_unit": float(prior), "train_positives": int(pos.sum()), "test_crash_hours": n_cases,
               "waze_labelled_share_2024": float(waze_train.mean()), "waze_pairs": waze_pairs,
               "share_of_2025_crashes_in_low_traffic_third": float((z_case <= np.quantile(z_unit, 1 / 3)).mean())}
    return pd.DataFrame(rows), details


def draw_figure(summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, secondary, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    styles = {"pn": ("#2a78d6", "o"), "pn_rescaled": ("#eb6834", "s"), "nnpu": ("#1baf7a", "^"), "nnpu_propensity": ("#4a3aa7", "D")}
    panels = [("predicted_over_observed", "Predicted / observed crashes,\nwhole network"),
              ("predicted_over_observed_low_third", "Predicted / observed crashes,\nthird of units with lightest traffic"),
              ("auc_hidden_type", "ROC-AUC on crashes\nwhose label would be hidden")]
    columns = [("scar", "Labels removed at random"), ("sar_flow", "Labels removed more often in light traffic")]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(3, 2, figsize=(9, 8.6), sharex=True, sharey="row", layout="constrained", facecolor=surface)
    for column, (mechanism, title) in enumerate(columns):
        for row, (metric, label) in enumerate(panels):
            ax = axes[row, column]
            ax.set_facecolor(surface)
            if metric != "auc_hidden_type":
                ax.axhline(1.0, color=axis, linewidth=1.0, zorder=1)
            for offset, (method, (color, marker)) in zip((-1.2, -0.4, 0.4, 1.2), styles.items()):
                # With every label kept the mechanisms coincide, and propensity weights are all one.
                start = summary.loc[(summary["mechanism"] == "scar") & (summary["labelled_share"] == 1.0)
                                    & (summary["method"] == ("nnpu" if method == "nnpu_propensity" else method))]
                part = summary.loc[(summary["mechanism"] == mechanism) & (summary["method"] == method) & (summary["labelled_share"] < 1.0)]
                part = pd.concat([start, part]).sort_values("labelled_share", ascending=False).dropna(subset=[(metric, "mean")])
                if part.empty:
                    continue
                x = 100 * (1 - part["labelled_share"]) + offset  # small dodge: PN rescaled and nnPU nearly coincide
                ax.errorbar(x, part[(metric, "mean")], yerr=part[(metric, "std")], color=color, linewidth=1.6, elinewidth=0.8,
                            marker=marker, markersize=6.5, markeredgecolor=surface, markeredgewidth=1.4, zorder=3,
                            label=METHOD_LABELS[method] if (row, column) == (0, 1) else None)
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(axis)
            ax.tick_params(colors=muted, length=0, labelsize=8.5)
            if column == 0:
                ax.set_ylabel(label, color=secondary)
            if row == 0:
                ax.set_title(title, color=ink, fontsize=10, loc="left")
            if row == 2:
                ax.set_xlabel("Police-recorded crashes without a label in training (%)", color=secondary)
    fig.legend(loc="outside upper left", ncol=4, frameon=False, fontsize=8.5, labelcolor=secondary, handlelength=2.4)
    fig.savefig(OUT / "label_hiding.png", dpi=200, facecolor=surface)
    fig.savefig(OUT / "label_hiding.svg", facecolor=surface)
    plt.close(fig)


def write_report(results: dict, summary: pd.DataFrame) -> None:
    scoring, details = results["scoring"], results["hiding"]
    names = {"constant_2024_rate": "Constant (2024 mean rate)", "pn_reliable": "PN, 364 reliable positives", "pn_reliable_rescaled": "PN × π / label rate",
             "nnpu_reliable": "nnPU, register prior", "nnpu_reliable_propensity": "nnPU + reporting propensity", "pn_official_hour": "PN, complete police labels (reference)"}
    lines = [
        "# Phase 8: what PU learning changes when labels are incomplete", "",
        "Ranking metrics do not show the effect of PU: under random missingness (SCAR) the labelling probability is a constant multiple of the crash probability, so the ranking is unchanged; with a crash prior of about 1e-4, the positives in U are negligible. "
        "PU corrects the absolute level of the probabilities, and the ranking when labelling depends on features (SAR). 2025 has been inspected before, so the results are exploratory.", "",
        "## 1. Existing Phase 6 models against the complete police register (per edge-hour)", "",
        f"Registered crash edge-hours in 2025: {scoring['observed_crash_hours_2025']}; total edge-hours: {scoring['edge_hours_2025']:.3e}. "
        f"The 'reliable positives' are only {scoring['label_frequency_reliable']:.2f} of the registered 2024 crashes (label frequency c). "
        "Log-score skill = 1 − model log loss / constant-rate log loss; values above 0 beat the constant. Intervals are a bootstrap over units and crashes.", "",
        "| Model | Expected crash hours | Predicted/observed | Log loss ×1e-3 | Log-score skill [95% CI] | Brier skill |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in scoring["arms"].items():
        lines.append(f"| {names[name]} | {item['expected_crash_hours']:.0f} | {item['predicted_over_observed']:.2f} | {item['log_score_x1e3']:.4f} | "
                     f"{item['log_skill']:+.4f} [{item['log_skill_ci95'][0]:+.4f}, {item['log_skill_ci95'][1]:+.4f}] | {item['brier_skill']:+.5f} |")
    lines += ["", "At an event rate of about 2e-4 the Brier score is almost entirely driven by (1−h)² of the crash hours and is insensitive to the probability level, so its skill is close to 0; the log score discriminates better.", "",
              "### Alarms at fixed absolute risk thresholds", "",
              f"Thresholds are multiples of the 2024 mean hourly risk ({scoring['rate_2024_per_edge_hour']:.2e}). The 'actual multiple' is the observed crash rate in alarmed hours divided by the mean rate and should be at least the threshold multiple.", "",
              "| Model | Threshold | Alarmed edge-hours | Share | Crash hours covered | Coverage | Alarmed hours: predicted multiple | Actual multiple |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, by_threshold in scoring["alerts"].items():
        if name == "constant_2024_rate":
            continue
        for multiple, item in by_threshold.items():
            lines.append(f"| {names[name]} | {multiple}× | {item['edge_hours_flagged']:,.0f} | {item['share_of_edge_hours_flagged']:.3%} | "
                         f"{item['crash_hours_caught']:.0f} | {item['share_of_crash_hours_caught']:.1%} | "
                         f"{item['predicted_rate_over_base']:.1f} | {item['observed_rate_over_base']:.1f} |")
    lines += ["", "## 2. Label-hiding experiment", "",
              f"The police injury-crash register is taken as complete truth ({details['train_positives']} crash edge-hours in 2024, prior {details['prior_per_unit']:.3e} per unit); "
              "part of the labels is hidden before training and evaluation uses the complete 2025 register. scar: hidden at random; sar_flow: the lower the traffic flow, the more likely a label is hidden "
              f"(logit e = a + {SELECTIVITY}·z(log flow)); waze: the real mechanism, only crashes reported in Waze keep their label "
              f"(kept: {details['waze_labelled_share_2024']:.2f}). Values are mean ± SD over repeated hiding. "
              f"The 'lowest-flow third of units' contains {details['share_of_2025_crashes_in_low_traffic_third']:.1%} of the 2025 crashes; under the waze mechanism the strata follow the estimated reporting propensity.", ""]
    metrics = [("predicted_over_observed", "Predicted/observed"), ("predicted_over_observed_low_third", "Predicted/observed: lowest-flow third (waze: lowest reporting propensity)"),
               ("predicted_over_observed_high_third", "Predicted/observed: highest-flow third (waze: highest reporting propensity)"), ("auc_all", "AUC: all crashes"),
               ("auc_hidden_type", "AUC: crashes of the hidden type"), ("log_skill", "Log-score skill")]
    titles = {"scar": "Hidden at random (SCAR)", "sar_flow": "Hidden with flow-dependent bias (SAR)", "waze": "Real mechanism: crashes not reported in Waze have no label"}
    for mechanism, title in titles.items():
        lines += [f"### {title}", "", "| Share of labels kept | Method | " + " | ".join(label for _, label in metrics) + " |", "|---:|---|" + "---:|" * len(metrics)]
        part = summary.loc[summary["mechanism"] == mechanism].sort_values("labelled_share", ascending=False, kind="stable")
        for _, row in part.iterrows():
            cells = ["—" if np.isnan(row[(metric, "mean")]) else f"{row[(metric, 'mean')]:.3f} ± {row[(metric, 'std')]:.3f}" for metric, _ in metrics]
            lines.append(f"| {row['labelled_share'].item():.2f} | {row['method'].item()} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Paired differences under the real mechanism (population AUC, bootstrap over 2025 crashes)", "", "| Contrast | All crashes | Not reported in Waze | Reported in Waze |", "|---|---:|---:|---:|"]
    for name, item in details["waze_pairs"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{item[e]['delta']:+.4f} [{item[e]['ci95'][0]:+.4f}, {item[e]['ci95'][1]:+.4f}]" for e in ("all", "not_waze_reported", "waze_reported")) + " |")
    lines += ["", "Figure: `label_hiding.png` / `label_hiding.svg` (mean and SD across repeats; the line at 1 marks the correct crash count; methods are slightly offset on the x-axis). "
              "Stratified calibration should be compared with the same stratum when all labels are kept: the model with complete labels is not exactly 1 in the lowest-flow third either.", "",
              "Limitations: the prior π is taken from the complete register and would need an external source in practice; the propensity model uses the labelling status on the complete 2024 register, which assumes a register sample that can be checked. "
              "With a known prior, pn_rescaled and nnpu are equivalent for the overall level; they differ only in stratified calibration and in ranking under biased labelling. "
              "The SD across repeats reflects only the randomness of hiding and initialisation, not the sampling error of the 2025 crashes.", "",
              "Reproduction: `python analysis/phase8_pu_value.py` (requires the Phase 6 datasets and predictions)."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    units = {year: dataset["years"][str(year)]["population_edge_5min_units"] for year in (2024, 2025)}
    scoring = score_existing(units, args.replicates)
    runs, details = hiding_experiment(units, args.reps, args.seeds, args.replicates)
    runs.to_csv(OUT / "label_hiding_runs.csv", index=False)
    measures = [column for column in runs.columns if column not in ("mechanism", "labelled_share", "rep", "method", "labelled_positives")]
    summary = runs.groupby(["mechanism", "labelled_share", "method"], sort=False)[measures].agg(["mean", "std"]).reset_index()
    if args.reps == 1:
        summary = summary.fillna({(measure, "std"): 0.0 for measure in measures})
    results = {"scoring": scoring, "hiding": details,
               "hiding_summary": [{"mechanism": row["mechanism"].item(), "labelled_share": row["labelled_share"].item(), "method": row["method"].item(),
                                   **{measure: {"mean": float(row[(measure, "mean")]), "std": float(row[(measure, "std")])} for measure in measures}}
                                  for _, row in summary.iterrows()]}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    draw_figure(summary)
    write_report(results, summary)


if __name__ == "__main__":
    main()
