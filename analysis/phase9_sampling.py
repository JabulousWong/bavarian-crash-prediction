"""Conventional resampling of the training set (matrix B, labels fixed, no PU).

The question is the one most crash-prediction studies answer implicitly: what
does random undersampling, random oversampling or SMOTE of the training data do
to a crash prediction model? Labels are the conventional ones: police-recorded
injury crash hours of 2024 are positives and the label-independent marginal
sample U (300,000 edge x 5-min units) is treated as negative. Only the
training pool changes; features, learners and the 2025 evaluation do not.

  none          all positives + all of U (pool ratio about 1:260)
  rus_1to10     U randomly undersampled to 10 per positive
  rus_1to4      ... 4 per positive
  rus_1to1      ... 1 per positive
  ros_1to1      positives duplicated at random up to |U|
  smote_1to1    SMOTE (k=5) positives up to |U|
  adasyn_1to1   ADASYN (k=5) positives up to |U|
  rus_smote     U undersampled to 1:10, then SMOTE to 1:1

Resampling acts on the training part of 2024 only; a 20 % hold-out of the
original 2024 rows (not resampled) selects the MLP stopping epoch by ROC-AUC.
Learners: logistic regression, histogram gradient boosting (thesis settings),
MLP 64-32. Each condition is repeated over five seeds (resampling draw and
model initialisation) and the seed-averaged score is evaluated.

Evaluation on 2025 (previously examined: exploratory):
  ranking      population ROC-AUC of police crash hours against U, within-risk-set
               concordance, share of crashes in the top 1 % of units
  threshold    at the default 0.5 cut-off: recall, alerts per day on the network,
               precision at the natural prevalence; and the same model scored on a
               1:1 balanced test set, as many studies report it
  level        mean predicted unit risk / observed rate, before and after the
               prior-shift correction with the known sampling ratio
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from phase6_minute_pu import CONTEXT, SEED, TRAFFIC, Preprocessor, evaluate, interval, raw_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "v2" / "phase6"
OUT = ROOT / "analysis" / "results" / "v2" / "phase9" / "sampling"
BUFFER_KEY = "__b30"  # `evaluate` parses a buffer from the key; police endpoints do not use it
METHODS = ("none", "rus_1to10", "rus_1to4", "rus_1to1", "ros_1to1", "smote_1to1", "adasyn_1to1", "rus_smote")
LEARNERS = ("lr", "hgb", "mlp")


def resample(method: str, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return x, y
    if method.startswith("rus_1to"):
        ratio = 1 / int(method.split("to")[1])
        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed).fit_resample(x, y)
    if method == "ros_1to1":
        return RandomOverSampler(random_state=seed).fit_resample(x, y)
    if method == "smote_1to1":
        return SMOTE(k_neighbors=5, random_state=seed).fit_resample(x, y)
    if method == "adasyn_1to1":
        return ADASYN(n_neighbors=5, random_state=seed).fit_resample(x, y)
    if method == "rus_smote":
        x, y = RandomUnderSampler(sampling_strategy=0.1, random_state=seed).fit_resample(x, y)
        return SMOTE(k_neighbors=5, random_state=seed).fit_resample(x, y)
    raise ValueError(method)


def fit_mlp(x, y, x_val, y_val, seed, device, epochs=40, patience=5, batch=4096):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1),
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tx, ty = torch.as_tensor(x, device=device), torch.as_tensor(y, dtype=torch.float32, device=device)
    vx = torch.as_tensor(x_val, device=device)
    best, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(x)), device=device)
        for start in range(0, len(x), batch):
            index = order[start:start + batch]
            optimiser.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx[index]).squeeze(1), ty[index])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            value = roc_auc_score(y_val, model(vx).squeeze(1).cpu().numpy())
        if value > best + 1e-5:
            best, stale = value, 0
            best_state = {key: t.detach().clone() for key, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def fit_predict(learner, x, y, x_val, y_val, x_test, seed, device) -> np.ndarray:
    if learner == "lr":
        model = LogisticRegression(max_iter=2000).fit(x, y)
        return model.predict_proba(x_test)[:, 1]
    if learner == "hgb":
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=40,
                                               l2_regularization=1.0, early_stopping=False, random_state=seed)
        return model.fit(x, y).predict_proba(x_test)[:, 1]
    model = fit_mlp(x, y, x_val, y_val, seed, device)
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x_test, device=device)).squeeze(1)).cpu().numpy()


def prior_shift(p: np.ndarray, train_prevalence: float, population_prevalence: float) -> np.ndarray:
    """Intercept correction for a known sampling ratio (King & Zeng 2001; Elkan 2001)."""
    odds = p / np.clip(1 - p, 1e-12, None)
    factor = (population_prevalence / (1 - population_prevalence)) / (train_prevalence / (1 - train_prevalence))
    corrected = odds * factor
    return corrected / (1 + corrected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--learners", nargs="+", default=list(LEARNERS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)

    train = pd.read_parquet(DATA / "minute_dataset_2024.parquet")
    test = pd.read_parquet(DATA / "minute_dataset_2025.parquet")
    test["official_confirmed"] = test["official_confirmed"].fillna(False).astype(bool)
    summary = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    units = {year: summary["years"][str(year)]["population_edge_5min_units"] for year in (2024, 2025)}
    window = summary["years"]["2025"]["eligible_window"]
    days_2025 = (pd.Timestamp(window[1]) - pd.Timestamp(window[0])).total_seconds() / 86400

    x_train_raw, x_test_raw = raw_features(train), raw_features(test)
    kind_train = train["kind"].to_numpy()
    rows = np.isin(kind_train, ["official_case", "marginal"])
    prep = Preprocessor(TRAFFIC + CONTEXT).fit(x_train_raw.loc[kind_train == "marginal"])
    x_all = prep.transform(x_train_raw.loc[rows])
    y_all = (kind_train[rows] == "official_case").astype(np.int64)
    x_test = prep.transform(x_test_raw)
    hold = np.random.default_rng(SEED).random(len(y_all)) < 0.2
    x_fit, y_fit, x_val, y_val = x_all[~hold], y_all[~hold], x_all[hold], y_all[hold]
    population_prevalence = int(y_all.sum()) / units[2024]

    scores, fits = {}, {}
    for learner in args.learners:
        for method in METHODS:
            t0 = time.perf_counter()
            probabilities, pool = [], None
            for seed in range(args.seeds):
                x_res, y_res = resample(method, x_fit, y_fit, SEED + seed)
                pool = {"rows": int(len(y_res)), "positives": int(y_res.sum()),
                        "train_prevalence": float(y_res.mean())}
                probabilities.append(fit_predict(learner, x_res.astype(np.float32), y_res, x_val, y_val, x_test,
                                                 SEED + seed, device))
            key = f"{learner}__{method}{BUFFER_KEY}"
            scores[key] = np.mean(probabilities, axis=0)
            fits[key] = {**pool, "seconds": time.perf_counter() - t0}
            print(f"fitted {key}: {pool}, {fits[key]['seconds']:.0f}s", flush=True)

    # ---- Evaluation on 2025 ----
    rng = np.random.default_rng(SEED)
    metrics, place, conditional = evaluate(test, scores, rng, args.replicates)
    kind = test["kind"].to_numpy()
    marginal, cases = kind == "marginal", kind == "official_case"
    observed = int(cases.sum()) / units[2025]
    balanced_controls = np.random.default_rng(SEED).choice(np.flatnonzero(marginal), size=int(cases.sum()), replace=False)

    def at_threshold(score: np.ndarray, threshold: float) -> dict:
        recall = float((score[cases] >= threshold).mean())
        alert_share = float((score[marginal] >= threshold).mean())
        true_alerts, alerts = recall * cases.sum(), alert_share * units[2025]
        precision = float(true_alerts / alerts) if alerts > 0 else float("nan")
        return {"recall": recall, "alert_share_of_units": alert_share, "alerts_per_day": alerts / days_2025,
                "precision_natural": precision,
                "f1_natural": float(2 * precision * recall / (precision + recall)) if alerts > 0 and recall > 0 else 0.0}

    def balanced(score: np.ndarray, threshold: float) -> dict:
        s = np.concatenate([score[cases], score[balanced_controls]])
        y = np.r_[np.ones(cases.sum()), np.zeros(len(balanced_controls))]
        hit = s >= threshold
        tp, fp = (hit & (y == 1)).sum(), (hit & (y == 0)).sum()
        recall, precision = tp / y.sum(), (tp / (tp + fp) if tp + fp else float("nan"))
        return {"accuracy": float((hit == y).mean()), "recall": float(recall), "precision": float(precision),
                "f1": float(2 * precision * recall / (precision + recall)) if tp else 0.0,
                "roc_auc": float(roc_auc_score(y, s))}

    def deciles(p: np.ndarray) -> list[dict]:
        edges = np.quantile(p[marginal], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        du, dp = np.digitize(p[marginal], edges[1:-1]), np.digitize(p[cases], edges[1:-1])
        return [{"decile": d + 1, "predicted": float(p[marginal][du == d].mean()),
                 "observed": float((dp == d).sum() / (units[2025] * (du == d).mean()))} for d in range(10)]

    for key, score in scores.items():
        prevalence = fits[key]["train_prevalence"]
        corrected = prior_shift(score, prevalence, population_prevalence)
        metrics[key]["pr_auc_natural"] = float(average_precision_score(
            np.r_[np.ones(cases.sum()), np.zeros(marginal.sum())], np.r_[score[cases], score[marginal]],
            sample_weight=np.r_[np.ones(cases.sum()), np.full(marginal.sum(), units[2025] / marginal.sum())]))
        metrics[key]["threshold_0.5"] = at_threshold(score, 0.5)
        metrics[key]["balanced_test_0.5"] = balanced(score, 0.5)
        metrics[key]["level"] = {
            "predicted_over_observed_raw": float(score[marginal].mean() / observed),
            "predicted_over_observed_corrected": float(corrected[marginal].mean() / observed),
            "deciles_raw": deciles(score), "deciles_corrected": deciles(corrected),
        }

    comparisons = {}
    for key in scores:
        learner, method = key.split("__")[0], key.split("__")[1]
        if method == "none":
            continue
        reference = f"{learner}__none{BUFFER_KEY}"
        population = place[(key, "official_all")] - place[(reference, "official_all")]
        matched = conditional[(key, "official_all")] - conditional[(reference, "official_all")]
        comparisons[key] = {
            "population_auc_delta": float(population.mean()), "population_ci95": interval(population, rng, args.replicates),
            "within_set_delta": float(matched.mean()), "within_set_ci95": interval(matched, rng, args.replicates),
        }

    results = {
        "design": {"unit": "edge x 5-min", "train_year": 2024, "test_year": 2025,
                   "positives": "police injury crash hours (official_case)", "negatives": "marginal sample U",
                   "methods": METHODS, "learners": args.learners, "seeds": args.seeds, "device": device,
                   "population_prevalence_2024": population_prevalence, "observed_rate_2025": observed,
                   "days_2025": days_2025, "validation": "20% of original 2024 rows, not resampled",
                   "test_status": "2025 previously examined; exploratory"},
        "fits": fits,
        "metrics": {key: {"population": value["population"]["official_all"],
                          "matched": value["matched"]["official_all"],
                          **{k: value[k] for k in ("pr_auc_natural", "threshold_0.5", "balanced_test_0.5", "level")}}
                    for key, value in metrics.items()},
        "paired_vs_none": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(scores).assign(row_id=test["row_id"].to_numpy(), kind=kind).to_parquet(
        OUT / "test_predictions_2025.parquet", index=False)
    write_report(results)


def write_report(results: dict) -> None:
    lines = ["# Comparison of sampling methods (conventional labels: police crashes = positives, U = negatives; no PU)", "",
             "Trained on 2024, evaluated on 2025. Each cell evaluates the scores averaged over 5 seeds.", "",
             "| Learner | Sampling | Training rows | Training prevalence | Population AUC [95% CI] | Within-set concordance | Natural PR-AUC | Top 1% unit coverage | Recall at 0.5 | Alarms per day | Natural precision | Balanced-test F1 | Balanced-test precision | Predicted/observed raw | Corrected |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, m in results["metrics"].items():
        learner, method = key.split("__")[:2]
        f, p, t, b, lv = results["fits"][key], m["population"], m["threshold_0.5"], m["balanced_test_0.5"], m["level"]
        lines.append(
            f"| {learner} | {method} | {f['rows']:,} | {f['train_prevalence']:.3f} | {p['roc_auc']:.3f} "
            f"[{p['roc_auc_ci95'][0]:.3f}, {p['roc_auc_ci95'][1]:.3f}] | {m['matched']['within_set_concordance']:.3f} | {m['pr_auc_natural']:.2e} | "
            f"{p['share_in_top1pct_units']:.3f} | {t['recall']:.3f} | {t['alerts_per_day']:,.0f} | {t['precision_natural']:.2e} | "
            f"{b['f1']:.3f} | {b['precision']:.3f} | {lv['predicted_over_observed_raw']:.1f} | {lv['predicted_over_observed_corrected']:.2f} |")
    lines += ["", "## Paired difference to no sampling (none) [95% CI]", "", "| Model | Population AUC difference | Within-set concordance difference |", "|---|---:|---:|"]
    for key, c in results["paired_vs_none"].items():
        lines.append(f"| {key.split('__b')[0]} | {c['population_auc_delta']:+.4f} [{c['population_ci95'][0]:+.4f}, {c['population_ci95'][1]:+.4f}] | "
                     f"{c['within_set_delta']:+.4f} [{c['within_set_ci95'][0]:+.4f}, {c['within_set_ci95'][1]:+.4f}] |")
    lines += ["", "Natural precision = true positives at threshold 0.5 / alarmed units across the network (all 2025 5-minute edge units). Balanced test = 2025 official crashes + an equal number of random U units, the reporting style common in the literature.",
              "Corrected = intercept shift from the known training prevalence to the 2024 per-unit prior. Reproduction: `python analysis/phase9_sampling.py`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
