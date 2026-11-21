"""Paired risk-set bootstrap for label-treatment comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis" / "results" / "phase2" / "label_comparison"
SEED = 20260929
REPLICATES = 1000


def main() -> None:
    data = pd.read_parquet(BASE / "test_predictions_2025.parquet")
    score_columns = [column for column in data if column not in {
        "stratum_id", "edge_id", "outcome_hour", "target_official", "source_pattern"
    }]
    groups = {key: group.index.to_numpy() for key, group in data.groupby("stratum_id")}
    keys = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(SEED)
    estimates = {column: {"roc_auc": [], "pr_auc": []} for column in score_columns}
    for _ in range(REPLICATES):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        sample = data.loc[indices]
        y = sample["target_official"].to_numpy()
        for column in score_columns:
            score = sample[column].to_numpy()
            estimates[column]["roc_auc"].append(roc_auc_score(y, score))
            estimates[column]["pr_auc"].append(average_precision_score(y, score))
    comparisons = {}
    for model in ("logistic", "hist_gradient_boosting"):
        reference = f"official_only_{model}"
        for treatment in (
            "hard_union", "soft_waze_q_0.10", "soft_waze_q_0.25",
            "soft_waze_q_0.50", "soft_waze_q_0.75",
        ):
            candidate = f"{treatment}_{model}"
            key = f"{candidate}_minus_{reference}"
            comparisons[key] = {}
            for metric in ("roc_auc", "pr_auc"):
                delta = np.asarray(estimates[candidate][metric]) - np.asarray(
                    estimates[reference][metric]
                )
                comparisons[key][metric] = {
                    "mean_delta": float(delta.mean()),
                    "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
                    "bootstrap_probability_delta_gt_0": float((delta > 0).mean()),
                }
    output = {
        "replicates": REPLICATES,
        "resampling_unit": "matched risk-set stratum",
        "comparisons": comparisons,
    }
    (BASE / "paired_bootstrap.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
