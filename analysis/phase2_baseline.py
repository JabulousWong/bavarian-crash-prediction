"""Train a strict temporal baseline: 2024 development, untouched 2025 test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "phase2" / "datasets"
OUT = ROOT / "analysis" / "results" / "phase2" / "baseline"


BASE_FEATURES = [
    f"{side}_{name}"
    for side in ("from", "to")
    for name in (
        "n_kfz", "n_lkw", "n_pkw", "v_kfz", "v_lkw", "v_pkw",
        "occupancy_mean", "net_gap_mean", "speed_sd_mean",
        "valid_lane_minutes_flow", "valid_lane_minutes_speed", "lane_channels",
    )
]


def load(year: int) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"matched_edge_hours_{year}_100m.parquet")


def features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[BASE_FEATURES].copy()
    for side in ("from", "to"):
        result[f"{side}_heavy_share"] = result[f"{side}_n_lkw"] / result[f"{side}_n_kfz"].replace(0, np.nan)
        result[f"{side}_flow_coverage"] = result[f"{side}_valid_lane_minutes_flow"] / (
            60 * result[f"{side}_lane_channels"].replace(0, np.nan)
        )
        result[f"{side}_speed_coverage"] = result[f"{side}_valid_lane_minutes_speed"] / (
            60 * result[f"{side}_lane_channels"].replace(0, np.nan)
        )
    result["spatial_speed_difference"] = result["to_v_kfz"] - result["from_v_kfz"]
    result["spatial_flow_difference"] = result["to_n_kfz"] - result["from_n_kfz"]
    return result.replace([np.inf, -np.inf], np.nan)


def detection_at_fraction(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(score) * fraction)))
    selected = np.argsort(score)[-count:]
    return float(y[selected].sum() / y.sum())


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_matched_prevalence": float(average_precision_score(y, score)),
        "brier_matched_prevalence": float(brier_score_loss(y, score)),
        "log_loss_matched_prevalence": float(log_loss(y, score)),
        "detection_rate_at_top_1pct": detection_at_fraction(y, score, 0.01),
        "detection_rate_at_top_5pct": detection_at_fraction(y, score, 0.05),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = load(2024), load(2025)
    x_train, x_test = features(train), features(test)
    y_train = train["target_official"].to_numpy()
    y_test = test["target_official"].to_numpy()
    logistic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=20260929
                ),
            ),
        ]
    )
    tree = Pipeline(
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
                    class_weight="balanced",
                    random_state=20260929,
                ),
            ),
        ]
    )
    results = {
        "design": {
            "train_year": 2024,
            "test_year": 2025,
            "target": "official injury-crash edge-hour",
            "predictor_window": "completed hour [h-1,h)",
            "test_prevalence_is_matched_not_population": float(y_test.mean()),
            "network_scope": "detector-derived corridor, 100 m provisional threshold",
        },
        "models": {},
    }
    predictions = test[
        ["stratum_id", "edge_id", "outcome_hour", "target_official", "source_pattern"]
    ].copy()
    for name, model in (("logistic", logistic), ("hist_gradient_boosting", tree)):
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        results["models"][name] = metrics(y_test, score)
        predictions[f"score_{name}"] = score
    predictions.to_parquet(OUT / "test_predictions_2025.parquet", index=False)
    (OUT / "baseline_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
