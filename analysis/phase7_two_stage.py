"""Stage 2: does the 2016-2023 frequency prior improve short-term prediction?

The edge x 5-min units, positives, marginal U, models and metrics are those of
Phase 6 (2024 training, 2025 evaluation, 30-min buffer before the first Waze
report). Feature blocks:
  context    hour, season, weekend, edge length, 2024 mean flow
  frequency  empirical-Bayes edge rate from 2016-2023 crashes, its SPF and raw
             count components, and the rate scaled by the network hour/month profile
  traffic    one-minute detector features before t_end
The prior alone, without any learning, is scored as a reference. A frequency
prior is constant within a matched risk set up to the month profile, so it can
only change population-level ranking, not within-set discrimination.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase6_minute_pu import (
    ALL_TO_INJURY_RATIO, CONTEXT, PRIMARY_BUFFER, SEED, TRAFFIC, Preprocessor, fit_mlp, interval,
    placements, predict, raw_features, within_stratum,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / "analysis" / "results" / "phase6"
OUT = ROOT / "analysis" / "results" / "phase7"
FREQUENCY = ["freq_log_eb", "freq_log_spf", "freq_log_hist", "freq_log_unit"]
UNITS_PER_YEAR = 365.25 * 24 * 12


def frequency_features(frame: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_csv(OUT / "edge_frequency_prior.csv").set_index("edge_id")
    hour = pd.read_csv(OUT / "temporal_profile_hour.csv").set_index(["weekend", "hour"])["multiplier"]
    month = pd.read_csv(OUT / "temporal_profile_month.csv").set_index("month")["multiplier"]
    edge = frame["edge_id"]
    keys = pd.MultiIndex.from_arrays([frame["t_end"].dt.dayofweek >= 5, frame["t_end"].dt.hour])
    temporal = hour.reindex(keys).to_numpy() * frame["t_end"].dt.month.map(month).to_numpy()
    eb = edge.map(prior["eb_per_year"]).to_numpy()
    return pd.DataFrame({
        "freq_log_eb": np.log(eb + 0.01),
        "freq_log_spf": np.log(edge.map(prior["spf_per_year"]).to_numpy() + 0.01),
        "freq_log_hist": np.log1p(edge.map(prior["hist_count"]).to_numpy()),
        "freq_log_unit": np.log(eb * temporal + 0.01),
        "prior_unit_probability": eb * temporal / UNITS_PER_YEAR,
        "has_history": edge.map(prior["hist_count"]).to_numpy() > 0,
    }, index=frame.index)


def km_time_coverage(score, length_km, positive, background, share: float) -> float:
    """Share of positives inside the riskiest `share` of network length x time.

    Units are ranked by risk per kilometre and weighted by their length, so a
    long edge cannot rank high merely for being long. Uniform risk gives `share`.
    """
    density = score / length_km
    if np.ptp(density[background]) < 1e-12:
        return share  # uniform risk per kilometre, e.g. a score equal to edge length
    order = np.argsort(-density[background])
    cumulative = np.cumsum(length_km[background][order]) / length_km[background].sum()
    threshold = density[background][order][np.searchsorted(cumulative, share)]
    return float((density[positive] > threshold).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_parquet(PHASE6 / "minute_dataset_2024.parquet")
    test = pd.read_parquet(PHASE6 / "minute_dataset_2025.parquet")
    dataset_summary = json.loads((PHASE6 / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    units = {year: dataset_summary["years"][str(year)]["population_edge_5min_units"] for year in (2024, 2025)}
    for frame in (train, test):
        frame["official_confirmed"] = frame["official_confirmed"].fillna(False).astype(bool)
    f_train, f_test = frequency_features(train), frequency_features(test)
    x_train_raw = pd.concat([raw_features(train), f_train[FREQUENCY]], axis=1)
    x_test_raw = pd.concat([raw_features(test), f_test[FREQUENCY]], axis=1)
    is_u = (train["kind"] == "marginal").to_numpy()
    blocks = {
        "context": CONTEXT, "context+frequency": CONTEXT + FREQUENCY,
        "context+traffic": CONTEXT + TRAFFIC, "context+frequency+traffic": CONTEXT + FREQUENCY + TRAFFIC,
    }
    prep = {name: Preprocessor(columns).fit(x_train_raw.loc[is_u]) for name, columns in blocks.items()}
    x_train = {name: p.transform(x_train_raw) for name, p in prep.items()}
    x_test = {name: p.transform(x_test_raw) for name, p in prep.items()}

    waze = ((train["kind"] == "waze_case") & (train["buffer_min"] == PRIMARY_BUFFER)).to_numpy()
    positives = {"union": waze, "reliable": waze & train["official_confirmed"].to_numpy(),
                 "official_hour": (train["kind"] == "official_case").to_numpy()}
    prior_injury = positives["official_hour"].sum() / units[2024]
    prior_all = prior_injury * ALL_TO_INJURY_RATIO
    arms = {
        # name: (positives, mode, prior target, feature block)
        "pn_union__context": ("union", "pn", "labeled", "context"),
        "pn_union__context+frequency": ("union", "pn", "labeled", "context+frequency"),
        "pn_union__context+traffic": ("union", "pn", "labeled", "context+traffic"),
        "pn_union__context+frequency+traffic": ("union", "pn", "labeled", "context+frequency+traffic"),
        "mswspu__context+frequency+traffic": ("union", "nnpu", "all", "context+frequency+traffic"),
        "nnpu_reliable__context+frequency+traffic": ("reliable", "nnpu", "injury", "context+frequency+traffic"),
        "pn_official_hour__context+traffic": ("official_hour", "pn", "labeled", "context+traffic"),
        "pn_official_hour__context+frequency+traffic": ("official_hour", "pn", "labeled", "context+frequency+traffic"),
    }
    length_km = (test["length_straight_m"].clip(lower=50) / 1000).to_numpy()
    scores, fits = {"edge_length_alone": length_km.copy(),
                    "frequency_prior_alone": f_test["prior_unit_probability"].to_numpy()}, {}
    for name, (which, mode, target, block) in arms.items():
        mask = positives[which]
        prior = {"labeled": mask.sum() / units[2024], "injury": prior_injury, "all": prior_all}[target]
        models = [fit_mlp(x_train[block][mask], np.ones(mask.sum()), x_train[block][is_u], prior, mode, SEED + seed, device)[0]
                  for seed in range(args.seeds)]
        scores[name] = predict(models, x_test[block], device)
        fits[name] = {"positives": int(mask.sum()), "prior": float(prior), "mode": mode, "features": block}
        print(f"fitted {name}", flush=True)

    rng = np.random.default_rng(SEED)
    kind = test["kind"].to_numpy()
    marginal = kind == "marginal"
    reachable = f_test["has_history"].to_numpy()
    official = kind == "official_case"
    waze_test = (kind == "waze_case") & (test["buffer_min"] == PRIMARY_BUFFER).to_numpy()
    endpoints = {
        "official_all": (official, marginal),
        "official_on_edges_with_history": (official & reachable, marginal & reachable),
        "waze_police_confirmed": (waze_test & test["official_confirmed"].to_numpy(), marginal),
    }
    complete = test["complete_risk_set"].fillna(False).astype(bool).to_numpy()
    matched = {
        "official": (np.isin(kind, ["official_case", "official_control"]), official),
        "waze_all": (np.isin(kind, ["waze_case", "waze_control"]) & (test["buffer_min"] == PRIMARY_BUFFER).to_numpy() & complete,
                     kind == "waze_case"),
    }
    metrics, place, conditional = {}, {}, {}
    for name, score in scores.items():
        metrics[name] = {"population": {}, "matched": {}}
        for endpoint, (positive, background) in endpoints.items():
            ordered = np.sort(score[background])
            values = placements(score[positive], ordered)
            place[(name, endpoint)] = values
            metrics[name]["population"][endpoint] = {
                "n_positives": int(positive.sum()), "roc_auc": float(values.mean()),
                "roc_auc_ci95": interval(values, rng, args.replicates),
                "share_in_top1pct_units": float((score[positive] > np.quantile(ordered, 0.99)).mean()),
                "share_in_top5pct_units": float((score[positive] > np.quantile(ordered, 0.95)).mean()),
                "share_in_top10pct_units": float((score[positive] > np.quantile(ordered, 0.90)).mean()),
                "share_in_top10pct_km_time": km_time_coverage(score, length_km, positive, background, 0.10),
            }
        for endpoint, (rows, case) in matched.items():
            values = within_stratum(test.loc[rows], score[rows], case[rows])
            conditional[(name, endpoint)] = values
            metrics[name]["matched"][endpoint] = {"risk_sets": int(len(values)), "within_set_concordance": float(values.mean()),
                                                  "ci95": interval(values, rng, args.replicates)}

    def paired(a: str, b: str) -> dict:
        output = {}
        for endpoint in endpoints:
            delta = place[(a, endpoint)] - place[(b, endpoint)]
            output[f"population:{endpoint}"] = {"delta": float(delta.mean()), "ci95": interval(delta, rng, args.replicates)}
        for endpoint in matched:
            delta = conditional[(a, endpoint)] - conditional[(b, endpoint)]
            output[f"matched:{endpoint}"] = {"delta": float(delta.mean()), "ci95": interval(delta, rng, args.replicates)}
        return output

    comparisons = {
        "frequency added to context": paired("pn_union__context+frequency", "pn_union__context"),
        "frequency added to context+traffic": paired("pn_union__context+frequency+traffic", "pn_union__context+traffic"),
        "traffic added to context+frequency": paired("pn_union__context+frequency+traffic", "pn_union__context+frequency"),
        "nnPU vs U-as-negative, all features": paired("mswspu__context+frequency+traffic", "pn_union__context+frequency+traffic"),
        "frequency added, police hour-timed labels": paired("pn_official_hour__context+frequency+traffic", "pn_official_hour__context+traffic"),
        "learned two-stage model vs frequency prior alone": paired("pn_union__context+frequency+traffic", "frequency_prior_alone"),
    }

    population_hours = units[2025] / 12
    observed_rate = official.sum() / population_hours
    calibration = {"observed_injury_rate_per_edge_hour_2025": float(observed_rate), "scores": {}}
    for name in ("frequency_prior_alone", "nnpu_reliable__context+frequency+traffic"):
        hourly = 1 - (1 - scores[name]) ** 12
        cuts = np.quantile(hourly[marginal], np.linspace(0.1, 0.9, 9))
        decile_u, decile_p = np.digitize(hourly[marginal], cuts), np.digitize(hourly[official], cuts)
        calibration["scores"][name] = {
            "predicted_over_observed": float(hourly[marginal].mean() / observed_rate),
            "deciles": [{"decile": d + 1, "predicted": float(hourly[marginal][decile_u == d].mean()),
                         "observed": float((decile_p == d).sum() / (population_hours * (decile_u == d).mean()))}
                        for d in range(10)],
        }

    frequency_summary = json.loads((OUT / "frequency_summary.json").read_text(encoding="utf-8"))
    results = {
        "design": {"unit": "edge x 5-min", "train_year": 2024, "test_year": 2025, "buffer_min": PRIMARY_BUFFER,
                   "prior_source": "police injury crashes 2016-2023", "seeds": args.seeds,
                   "test_status": "2025 previously examined; exploratory"},
        "fits": fits, "metrics": metrics, "paired_comparisons": comparisons, "hourly_calibration": calibration,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "two_stage_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    predictions = test[["row_id", "kind", "stratum_id", "edge_id", "t_end", "buffer_min"]].copy()
    for name, score in scores.items():
        predictions[name] = score
    predictions.to_parquet(OUT / "two_stage_predictions_2025.parquet", index=False)

    lines = ["# Phase 7: crash-frequency prior first, then short-term prediction", "",
             "Stage 1 estimates the long-term crash rate of every edge from 2016–2023 police injury crashes only (SPF + empirical Bayes) and multiplies it by the network-wide hour × weekday/weekend and month profiles to obtain a prior for every edge × 5-minute unit. "
             "Stage 2 reuses the units, positives, U and models of Phase 6 (trained on 2024, evaluated on 2025). 2025 has been inspected before, so the results are exploratory.", "",
             "## Stage 1: predicting later edge crash counts from 2016–2023", "",
             "| Predictor | 2024 Spearman | 2024 crashes on top 10% of edges | 2025 Spearman | 2025 crashes on top 10% of edges | 2025 Poisson deviance |", "|---|---:|---:|---:|---:|---:|"]
    validation = frequency_summary["validation"]
    for name in ("length only", "SPF (length, volume proxy, lanes, road)", "observed 2016-2023 count", "empirical Bayes"):
        a, b = validation["2024"][name], validation["2025"][name]
        lines.append(f"| {name} | {a['spearman']:.3f} | {a['share_of_crashes_on_top10pct_edges']:.3f} | {b['spearman']:.3f} | "
                     f"{b['share_of_crashes_on_top10pct_edges']:.3f} | {b['poisson_deviance']:.3f} |")
    lines += ["", f"Oracle coverage of the top 10% of edges (ranked by the actual crash count of that year): 2024 {validation['2024']['share_on_top10pct_edges_oracle']:.3f}, 2025 {validation['2025']['share_on_top10pct_edges_oracle']:.3f}. "
              f"{frequency_summary['edges_without_history']}/{frequency_summary['edges']} edges had no crash at all in 2016–2023.", "",
              "## Stage 2: 2025 official crashes against marginal units", "",
              "| Model | ROC-AUC [95% CI] | Edges with crash history only | Top 1% unit coverage | Top 5% unit coverage | Top 10% unit coverage | Top 10% km·time coverage | Within-set concordance, official risk sets |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in metrics.items():
        p, r, m = item["population"]["official_all"], item["population"]["official_on_edges_with_history"], item["matched"]["official"]
        lines.append(f"| {name} | {p['roc_auc']:.3f} [{p['roc_auc_ci95'][0]:.3f}, {p['roc_auc_ci95'][1]:.3f}] | {r['roc_auc']:.3f} | "
                     f"{p['share_in_top1pct_units']:.3f} | {p['share_in_top5pct_units']:.3f} | {p['share_in_top10pct_units']:.3f} | {p['share_in_top10pct_km_time']:.3f} | "
                     f"{m['within_set_concordance']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] |")
    lines += ["", "Edge lengths range from 0 to 24 km and long edges have more crashes by construction, so part of the unit-ranked AUC and coverage is just a length effect (see the edge_length_alone row). "
              "'Top 10% km·time coverage' ranks by risk per km and weights by length; it is 0.10 for uniform risk."]
    lines += ["", "## Paired differences [95% CI]", "", "| Contrast | Population: all official | Population: edges with crash history | Population: Waze-timed and police-confirmed | Matched: official | Matched: all Waze-timed |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in comparisons.items():
        cells = [f"{item[key]['delta']:+.4f} [{item[key]['ci95'][0]:+.4f}, {item[key]['ci95'][1]:+.4f}]" for key in
                 ("population:official_all", "population:official_on_edges_with_history", "population:waze_police_confirmed",
                  "matched:official", "matched:waze_all")]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "## Absolute risk: predicted and observed injury-crash rate per edge-hour (by decile, ×1e-4)", "", f"Observed 2025 rate {observed_rate:.3e}.", "",
              "| Score | Predicted/observed | " + " | ".join(f"D{d}" for d in range(1, 11)) + " |", "|---|---:|" + "---:|" * 10]
    for name, item in calibration["scores"].items():
        cells = " | ".join(f"{d['predicted'] * 1e4:.2f}/{d['observed'] * 1e4:.2f}" for d in item["deciles"])
        lines.append(f"| {name} | {item['predicted_over_observed']:.2f} | {cells} |")
    lines += ["", "Intervals are a bootstrap over positives or risk sets with fixed models, without retraining uncertainty. The edge population contains duplicate and zero-length edges, and edges without crash history inflate the population AUC, so results restricted to edges with crash history are also given. "
              "The flow proxy is the 2024 detector mean, not the actual 2016–2023 flow.", "",
              "Reproduction: `python analysis/phase7_frequency_prior.py`, then `python analysis/phase7_two_stage.py`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
