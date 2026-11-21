"""Compare official-only, hard-union, and soft Waze supervision on one cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase2_baseline import features, metrics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "phase2" / "datasets"
OUT = ROOT / "analysis" / "results" / "phase2" / "label_comparison"


def model_pair() -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=20260929)),
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
                        random_state=20260929,
                    ),
                ),
            ]
        ),
    }


def soft_expand(frame: pd.DataFrame, q: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    is_control = frame["target_union"].eq(0)
    is_official = frame["official_observed"]
    is_waze_only = frame["target_union"].eq(1) & ~is_official
    fixed = frame.loc[is_control | is_official].copy()
    fixed_y = np.where(fixed["official_observed"], 1, 0)
    weak = frame.loc[is_waze_only].copy()
    expanded = pd.concat([fixed, weak, weak], ignore_index=True)
    y = np.concatenate(
        [fixed_y, np.ones(len(weak), dtype=int), np.zeros(len(weak), dtype=int)]
    )
    weights = np.concatenate(
        [np.ones(len(fixed)), np.full(len(weak), q), np.full(len(weak), 1 - q)]
    )
    return expanded, y, weights


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA / "matched_edge_hours_2024_100m_union.parquet")
    test = pd.read_parquet(DATA / "matched_edge_hours_2025_100m.parquet")
    x_test = features(test)
    y_test = test["target_official"].to_numpy()
    specifications = {}

    official_train = train.loc[train["target_union"].eq(0) | train["official_observed"]].copy()
    specifications["official_only"] = (
        official_train,
        official_train["official_observed"].astype(int).to_numpy(),
        np.ones(len(official_train)),
    )
    specifications["hard_union"] = (
        train,
        train["target_union"].astype(int).to_numpy(),
        np.ones(len(train)),
    )
    for q in (0.10, 0.25, 0.50, 0.75):
        specifications[f"soft_waze_q_{q:.2f}"] = soft_expand(train, q)

    results = {
        "evaluation_endpoint": "2025 official injury-crash matched test set",
        "warning": "Soft q is a sensitivity parameter, not P(true crash | Waze report).",
        "specifications": {},
    }
    predictions = test[
        ["stratum_id", "edge_id", "outcome_hour", "target_official", "source_pattern"]
    ].copy()
    for specification, (train_frame, y_train, weights) in specifications.items():
        x_train = features(train_frame)
        results["specifications"][specification] = {
            "train_rows_after_soft_expansion": int(len(train_frame)),
            "effective_positive_weight": float(weights[y_train == 1].sum()),
            "effective_negative_weight": float(weights[y_train == 0].sum()),
            "models": {},
        }
        for model_name, model in model_pair().items():
            fit_weights = weights.copy()
            pos = fit_weights[y_train == 1].sum()
            neg = fit_weights[y_train == 0].sum()
            fit_weights[y_train == 1] *= (pos + neg) / (2 * pos)
            fit_weights[y_train == 0] *= (pos + neg) / (2 * neg)
            model.fit(x_train, y_train, model__sample_weight=fit_weights)
            score = model.predict_proba(x_test)[:, 1]
            results["specifications"][specification]["models"][model_name] = metrics(
                y_test, score
            )
            predictions[f"{specification}_{model_name}"] = score
    predictions.to_parquet(OUT / "test_predictions_2025.parquet", index=False)
    (OUT / "label_comparison_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
