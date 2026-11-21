"""Does prior-hour roadside weather add to prior-hour traffic on the matched cohort?

2024 training, 2025 official injury-crash matched evaluation, as in Phase 2/3.
HGB runs a fixed 200 iterations without internal early stopping. The 2025 set
has been examined before; this is exploratory, not a fresh blind test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase2_baseline import features as traffic_features, metrics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "phase5" / "datasets"
PHASE1 = ROOT / "analysis" / "results" / "phase1"
OUT = ROOT / "analysis" / "results" / "phase5" / "weather_eval"
SEED = 20260929
# Visibility devices saturate at about 1, 2 or 16 km; only the range below the
# lowest ceiling is on a common scale.
VISIBILITY_CAP_M = 1000.0


def weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["wx_ni", "wx_wfd", "wx_fbz", "wx_fbt", "wx_tt1"]].copy()
    result["wx_sw_capped"] = frame["wx_sw"].clip(upper=VISIBILITY_CAP_M)
    result["wx_wet"] = (frame["wx_wfd"] > 0).astype(float).where(frame["wx_wfd"].notna())
    return result


def design(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    parts = []
    if "traffic" in feature_set:
        parts.append(traffic_features(frame))
    if "weather" in feature_set:
        parts.append(weather_features(frame))
    return pd.concat(parts, axis=1)


def models() -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def balanced_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y))
    pos, neg = (y == 1).sum(), (y == 0).sum()
    weights[y == 1] *= (pos + neg) / (2 * pos)
    weights[y == 0] *= (pos + neg) / (2 * neg)
    return weights


def paired_bootstrap(predictions: pd.DataFrame, pairs: list[tuple[str, str]], replicates: int) -> dict:
    groups = {key: group.index.to_numpy() for key, group in predictions.groupby("stratum_id")}
    keys = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(SEED)
    columns = sorted({name for pair in pairs for name in pair})
    draws = {column: {"roc_auc": [], "pr_auc": []} for column in columns}
    y_all = predictions["target_official"].to_numpy()
    scores = {column: predictions[column].to_numpy() for column in columns}
    for _ in range(replicates):
        indices = np.concatenate([groups[key] for key in rng.choice(keys, size=len(keys), replace=True)])
        y = y_all[indices]
        for column in columns:
            draws[column]["roc_auc"].append(roc_auc_score(y, scores[column][indices]))
            draws[column]["pr_auc"].append(average_precision_score(y, scores[column][indices]))
    output = {}
    for candidate, reference in pairs:
        output[f"{candidate}_minus_{reference}"] = {}
        for metric, function in (("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)):
            delta = np.asarray(draws[candidate][metric]) - np.asarray(draws[reference][metric])
            output[f"{candidate}_minus_{reference}"][metric] = {
                "point_delta": float(function(y_all, scores[candidate]) - function(y_all, scores[reference])),
                "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
                "bootstrap_probability_delta_gt_0": float((delta > 0).mean()),
            }
    return output


def case_control_contrast(frame: pd.DataFrame) -> dict:
    """Descriptive prior-hour weather shares for cases and their matched controls."""
    flags = pd.DataFrame(
        {
            "wet_film": (frame["wx_wfd"] > 0).where(frame["wx_wfd"].notna()),
            "precipitation": (frame["wx_ni"] > 0).where(frame["wx_ni"].notna()),
            "surface_not_dry": (frame["wx_fbz"] > 0).where(frame["wx_fbz"].notna()),
            "surface_below_0C": (frame["wx_fbt"] < 0).where(frame["wx_fbt"].notna()),
            "visibility_below_500m": (frame["wx_sw"] < 500).where(frame["wx_sw"].notna()),
        }
    ).astype(float)
    y = frame["target_official"].to_numpy()
    return {
        name: {
            "cases": float(flags.loc[y == 1, name].mean()),
            "controls": float(flags.loc[y == 0, name].mean()),
            "n_cases_observed": int(flags.loc[y == 1, name].notna().sum()),
        }
        for name in flags
    }


def police_concordance(frame: pd.DataFrame, year: int) -> dict:
    """Prior-hour sensor wet film against the police road-state record (decoding check)."""
    crashes = pd.read_csv(PHASE1 / f"crash_network_assignment_{year}.csv", dtype={"uid": str})
    states = pd.read_csv(
        ROOT / "accidents" / f"Unfallorte{year}_standardized.csv", sep=None, engine="python",
        dtype={"uid": str}, usecols=["uid", "USTRZUSTAND"],
    )
    crashes = crashes.merge(states, on="uid")
    crashes["outcome_hour"] = pd.to_datetime(crashes["date"]) + pd.to_timedelta(crashes["USTUNDE"], unit="h")
    police = crashes.groupby(["nearest_edge", "outcome_hour"])["USTRZUSTAND"].max().rename("police_state")
    cases = frame.loc[frame["target_official"] == 1].join(police, on=["edge_id", "outcome_hour"])
    cases = cases.dropna(subset=["police_state", "wx_wfd"])
    sensor_wet, police_wet = cases["wx_wfd"] > 0, cases["police_state"] > 0
    return {
        "cases_compared": int(len(cases)),
        "police_not_dry_share_given_sensor_dry": float(police_wet[~sensor_wet].mean()),
        "police_not_dry_share_given_sensor_wet": float(police_wet[sensor_wet].mean()),
        "raw_agreement": float((sensor_wet == police_wet).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA / "matched_edge_hours_2024_100m_union_wx.parquet")
    test = pd.read_parquet(DATA / "matched_edge_hours_2025_100m_wx.parquet")
    y_test = test["target_official"].to_numpy()
    official_train = train.loc[train["target_union"].eq(0) | train["official_observed"]]
    labels = {
        "official_only": (official_train, official_train["official_observed"].astype(int).to_numpy()),
        "hard_union": (train, train["target_union"].astype(int).to_numpy()),
    }
    feature_sets = ("traffic", "weather", "traffic+weather")
    results = {
        "design": {
            "train_year": 2024, "test_year": 2025, "test_endpoint": "matched official injury crashes (1:4)",
            "as_of_rule": "traffic and weather from [h-1,h); outcome in [h,h+1)",
            "hgb_iterations": 200, "hgb_early_stopping": False,
            "test_status": "previously examined; exploratory, not fresh blind validation",
        },
        "metrics": {},
    }
    predictions = test[["stratum_id", "edge_id", "outcome_hour", "target_official"]].copy()
    for label, (frame, y_train) in labels.items():
        weights = balanced_weights(y_train)
        for feature_set in feature_sets:
            x_train, x_test = design(frame, feature_set), design(test, feature_set)
            for model_name, model in models().items():
                model.fit(x_train, y_train, model__sample_weight=weights)
                score = model.predict_proba(x_test)[:, 1]
                name = f"{label}__{feature_set}__{model_name}"
                results["metrics"][name] = metrics(y_test, score)
                predictions[name] = score
    predictions.to_parquet(OUT / "test_predictions_2025.parquet", index=False)
    pairs = [
        (f"{label}__traffic+weather__{model}", f"{label}__traffic__{model}")
        for label in labels for model in ("logistic", "hist_gradient_boosting")
    ]
    bootstrap = paired_bootstrap(predictions, pairs, args.replicates)
    official_2024 = pd.read_parquet(DATA / "matched_edge_hours_2024_100m_wx.parquet")
    results["case_control_weather_shares"] = {
        "2024": case_control_contrast(official_2024), "2025": case_control_contrast(test),
    }
    results["police_road_state_check"] = {
        "2024": police_concordance(official_2024, 2024), "2025": police_concordance(test, 2025),
    }
    results["paired_bootstrap"] = {"replicates": args.replicates, "resampling_unit": "matched risk-set stratum",
                                   "comparisons": bootstrap}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Phase 5: roadside weather features of the preceding hour", "",
        "Trained on 2024, evaluated on the 2025 official injury crashes with 1:4 matching; traffic and weather are both taken from [h−1, h). HGB uses a fixed 200 iterations without internal early stopping and the same class-balancing weights as Phase 2.",
        "2025 has been inspected before, so these results are exploratory.", "",
        "| Training labels | Features | Logistic regression ROC-AUC | HGB ROC-AUC | HGB AP |", "|---|---|---:|---:|---:|",
    ]
    for label in labels:
        for feature_set in feature_sets:
            logistic = results["metrics"][f"{label}__{feature_set}__logistic"]
            tree = results["metrics"][f"{label}__{feature_set}__hist_gradient_boosting"]
            lines.append(f"| {label} | {feature_set} | {logistic['roc_auc']:.4f} | {tree['roc_auc']:.4f} | {tree['pr_auc_matched_prevalence']:.4f} |")
    lines += ["", "| Gain from adding weather (paired risk-set bootstrap) | ΔROC-AUC [95% CI] | ΔAP [95% CI] |", "|---|---:|---:|"]
    for name, values in bootstrap.items():
        cells = [f"{values[m]['point_delta']:+.4f} [{values[m]['ci95'][0]:+.4f}, {values[m]['ci95'][1]:+.4f}]" for m in ("roc_auc", "pr_auc")]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "| Share of weather state in the preceding hour | 2024 cases | 2024 controls | 2025 cases | 2025 controls |", "|---|---:|---:|---:|---:|"]
    shares = results["case_control_weather_shares"]
    for name in shares["2024"]:
        lines.append(f"| {name} | {shares['2024'][name]['cases']:.4f} | {shares['2024'][name]['controls']:.4f} | {shares['2025'][name]['cases']:.4f} | {shares['2025'][name]['controls']:.4f} |")
    lines += ["", f"Intervals come from {args.replicates} paired risk-set bootstrap replicates of fixed models, without retraining uncertainty or spatio-temporal correlation across risk sets, and without multiple-comparison correction. AP reflects the artificial 20% case rate.",
              "Visibility sensors have three range limits of about 1/2/16 km; the model truncates visibility at 1000 m for all of them.",
              "", "Reproduction: `python analysis/phase5_weather.py`, then `python analysis/phase5_weather_eval.py --replicates 1000`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(results["police_road_state_check"], indent=2))


if __name__ == "__main__":
    main()
