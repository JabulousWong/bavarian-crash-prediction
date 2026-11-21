"""Minute-level MSWS-PU experiment on Waze-timed accident events.

Units are edge x 5-min intervals. Training uses 2024, evaluation 2025.
  P_r  Waze-timed events matched to a police injury record (reliable positives)
  P_w  Waze-timed events without such a record (weak positives)
  U    label-independent marginal sample of units (events are not removed)

All MLP arms share architecture, features and U, and differ only in the risk:
  PN     pi_L * R_p^+ + R_u^-                         (U treated as negative)
  nnPU   pi * R_p^+ + max(0, R_u^- - pi * R_p^-)      (Kiryo et al. 2017, logistic loss)
with positives weighted by w_i (normalised):
  uniform      w_i = 1
  credibility  w_i = 1 for P_r; for P_w a relative credibility from Waze report signals
  propensity   w_i additionally divided by e(x) = P(Waze report | police crash, x),
               estimated on the complete 2024 police register (SAR correction)

Class priors per unit: the police injury register fixes pi for the injury target;
the all-crash target scales it by the Bavarian motorway all/injury accident ratio,
which assumes that regional ratio transfers to this network. Ranking under SCAR
does not depend on pi; absolute risk does. The 2025 data have been examined in
earlier phases, so every result here is exploratory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase6_minute_dataset import BUFFERS, SITE_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "phase6"
OUT = DATA / "minute_pu"
HOURLY_PREDICTIONS = ROOT / "analysis" / "results" / "phase2" / "label_comparison" / "test_predictions_2025.parquet"
SEED = 20260930
PRIMARY_BUFFER = 30
ALL_TO_INJURY_RATIO = 28_004 / 3_236  # Bavarian motorways 2024, Phase 3 class-prior audit
LOG_COLUMNS = ["up_flow_w", "dn_flow_w", "up_flow_c", "dn_flow_c", "up_mean_flow_2024", "dn_mean_flow_2024"]
TRAFFIC = [f"{side}_{name}" for side in ("up", "dn") for name in SITE_FEATURES] + ["dv_updn", "docc_updn", "dflow_updn"]
CONTEXT = ["hour_sin", "hour_cos", "weekend", "season_sin", "season_cos", "length_km",
           "up_mean_flow_2024", "dn_mean_flow_2024"]


def raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[[f"{side}_{name}" for side in ("up", "dn") for name in SITE_FEATURES]
              + ["up_mean_flow_2024", "dn_mean_flow_2024"]].astype(float).copy()
    x["dv_updn"] = x["dn_v_w"] - x["up_v_w"]
    x["docc_updn"] = x["dn_occ_w"] - x["up_occ_w"]
    x["dflow_updn"] = x["dn_flow_w"] - x["up_flow_w"]
    for column in LOG_COLUMNS:
        x[column] = np.log1p(x[column])
    # Whole hours only: police rows sit exactly on the hour, marginal rows do not,
    # and a finer clock would let a model recognise the row type.
    hour = frame["t_end"].dt.hour
    x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    x["weekend"] = (frame["t_end"].dt.dayofweek >= 5).astype(float)
    day = frame["t_end"].dt.dayofyear
    x["season_sin"], x["season_cos"] = np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)
    x["length_km"] = frame["length_straight_m"].clip(upper=25_000) / 1000
    return x.replace([np.inf, -np.inf], np.nan)


class Preprocessor:
    """Median imputation with missing indicators and standardisation, fitted on U."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, x: pd.DataFrame) -> "Preprocessor":
        x = x[self.columns]
        self.median = x.median()
        self.flagged = [column for column in self.columns if x[column].isna().any()]
        filled = x.fillna(self.median)
        self.mean, self.scale = filled.mean(), filled.std().replace(0, 1.0)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x = x[self.columns]
        values = ((x.fillna(self.median) - self.mean) / self.scale).clip(-8, 8).to_numpy()
        flags = x[self.flagged].isna().to_numpy(dtype=float)
        return np.hstack([values, flags]).astype(np.float32)


def risk_parts(model, xp, wp, xu, prior):
    zp, zu = model(xp).squeeze(1), model(xu).squeeze(1)
    weight = wp / wp.sum()
    positive = prior * (weight * F.softplus(-zp)).sum()
    negative = F.softplus(zu).mean() - prior * (weight * F.softplus(zp)).sum()
    return positive, negative, F.softplus(zu).mean()


def fit_mlp(xp, wp, xu, prior, mode, seed, device, max_steps=800, check_every=10, patience=12):
    """Full-batch training; a 20% hold-out of P and U selects the stopping step."""
    rng = np.random.default_rng(seed)
    hold_p, hold_u = rng.random(len(xp)) < 0.2, rng.random(len(xu)) < 0.2
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    tp, tw, tu = tensor(xp[~hold_p]), tensor(wp[~hold_p]), tensor(xu[~hold_u])
    vp, vw, vu = tensor(xp[hold_p]), tensor(wp[hold_p]), tensor(xu[hold_u])
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(xp.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1),
    ).to(device)
    with torch.no_grad():
        model[-1].bias.fill_(float(np.log(prior / (1 - prior))))
        model[-1].weight.mul_(0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def objective(positive, negative, unlabeled_negative):
        if mode == "pn":
            return positive + unlabeled_negative
        return positive + torch.clamp(negative, min=0)

    best, best_state, best_step, stale, clipped = float("inf"), None, 0, 0, 0
    for step in range(1, max_steps + 1):
        optimiser.zero_grad()
        positive, negative, unlabeled_negative = risk_parts(model, tp, tw, tu, prior)
        if mode == "nnpu" and negative.item() < 0:
            clipped += 1
            loss = -negative  # Kiryo et al.: step back when the negative risk is below zero
        else:
            loss = positive + (unlabeled_negative if mode == "pn" else negative)
        (loss / prior).backward()
        optimiser.step()
        if step % check_every == 0:
            with torch.no_grad():
                value = objective(*risk_parts(model, vp, vw, vu, prior)).item()
            if value < best - 1e-12:
                best, best_step, stale = value, step, 0
                best_state = {key: tensor_.detach().clone() for key, tensor_ in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_step": best_step, "validation_risk": best, "steps_with_negative_risk": clipped}


def predict(models, x, device) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        return np.mean([torch.sigmoid(model(tensor).squeeze(1)).cpu().numpy() for model in models], axis=0)


def placements(positive_scores: np.ndarray, marginal_sorted: np.ndarray) -> np.ndarray:
    """Share of marginal units scored below each positive (ties count half)."""
    left = np.searchsorted(marginal_sorted, positive_scores, side="left")
    right = np.searchsorted(marginal_sorted, positive_scores, side="right")
    return (left + right) / (2 * len(marginal_sorted))


def interval(values: np.ndarray, rng: np.random.Generator, replicates: int) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def within_stratum(frame: pd.DataFrame, score: np.ndarray, is_case: np.ndarray) -> np.ndarray:
    """Per risk set: share of its controls scored below the case (ties half)."""
    data = pd.DataFrame({"stratum": frame["stratum_id"].to_numpy(), "score": score, "case": is_case})
    case = data.loc[data["case"]].set_index("stratum")["score"]
    controls = data.loc[~data["case"]].join(case.rename("case_score"), on="stratum").dropna(subset=["case_score"])
    wins = (controls["case_score"] > controls["score"]) + 0.5 * (controls["case_score"] == controls["score"])
    return wins.groupby(controls["stratum"]).mean().to_numpy()


def label_weights(train: pd.DataFrame, test: pd.DataFrame, x_train_raw: pd.DataFrame,
                  x_test_raw: pd.DataFrame, is_u: np.ndarray) -> dict:
    """Credibility weights for weak positives and inverse reporting propensities (2024 fits)."""
    # Relative credibility of weak positives from Waze report signals (2024 only).
    def signals(frame):
        return pd.DataFrame({
            "log_reports": np.log1p(frame["reports"].astype(float)),
            "confidence": frame["max_confidence"].astype(float).fillna(0),
            "reliability": frame["max_reliability"].astype(float).fillna(0),
            "major": (frame["subtype"] == "ACCIDENT_MAJOR").astype(float),
            "minor": (frame["subtype"] == "ACCIDENT_MINOR").astype(float),
        })

    events_2024 = train.loc[(train["kind"] == "waze_case") & (train["buffer_min"] == PRIMARY_BUFFER)]
    events_2025 = test.loc[(test["kind"] == "waze_case") & (test["buffer_min"] == PRIMARY_BUFFER)]
    credibility = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        signals(events_2024), events_2024["official_confirmed"])
    reference = credibility.predict_proba(signals(events_2024.loc[events_2024["official_confirmed"]]))[:, 1].mean()
    waze_rows = (train["kind"] == "waze_case").to_numpy()
    credibility_weight = np.ones(len(train))
    credibility_weight[waze_rows] = np.where(
        train.loc[waze_rows, "official_confirmed"], 1.0,
        np.clip(credibility.predict_proba(signals(train.loc[waze_rows]))[:, 1] / reference, 0.05, 1.0))

    # Waze reporting propensity among police-recorded crashes (2024 only).
    propensity_columns = ["hour_sin", "hour_cos", "weekend", "up_flow_w", "dn_flow_w", "up_mean_flow_2024", "length_km"]
    propensity_prep = Preprocessor(propensity_columns).fit(x_train_raw.loc[is_u])
    official_cases_2024 = (train["kind"] == "official_case").to_numpy()
    reported_2024 = train.loc[official_cases_2024, "waze_matched"].fillna(False).astype(bool).to_numpy()
    propensity = LogisticRegression(max_iter=1000).fit(
        propensity_prep.transform(x_train_raw.loc[official_cases_2024]), reported_2024)
    e_train = propensity.predict_proba(propensity_prep.transform(x_train_raw))[:, 1]
    official_cases_2025 = (test["kind"] == "official_case").to_numpy()
    reported_2025 = test.loc[official_cases_2025, "waze_matched"].fillna(False).astype(bool).to_numpy()
    e_2025 = propensity.predict_proba(propensity_prep.transform(x_test_raw.loc[official_cases_2025]))[:, 1]
    return {
        "credibility_weight": credibility_weight,
        "inverse_propensity": 1.0 / np.clip(e_train, 0.1, 1.0),
        "credibility_model": {
            "auc_2024_in_sample": float(roc_auc_score(events_2024["official_confirmed"], credibility.predict_proba(signals(events_2024))[:, 1])),
            "auc_2025": float(roc_auc_score(events_2025["official_confirmed"], credibility.predict_proba(signals(events_2025))[:, 1])),
            "mean_weight_weak_positives": float(credibility_weight[waze_rows & ~train["official_confirmed"].to_numpy()].mean()),
        },
        "propensity_model": {
            "waze_reported_share_of_police_crashes": {"2024": float(reported_2024.mean()), "2025": float(reported_2025.mean())},
            "auc_2024_in_sample": float(roc_auc_score(reported_2024, e_train[official_cases_2024])),
            "auc_2025": float(roc_auc_score(reported_2025, e_2025)),
        },
    }


def positive_mask(train: pd.DataFrame, buffer: int, which: str) -> np.ndarray:
    if which == "official_hour":
        return (train["kind"] == "official_case").to_numpy()
    mask = (train["kind"] == "waze_case").to_numpy() & (train["buffer_min"] == buffer).to_numpy()
    return mask & train["official_confirmed"].to_numpy() if which == "reliable" else mask


def evaluate(test: pd.DataFrame, scores: dict[str, np.ndarray], rng: np.random.Generator,
             replicates: int) -> tuple[dict, dict, dict]:
    """Population (vs marginal units) and within-risk-set metrics for each `name__b<buffer>` score."""
    kind = test["kind"].to_numpy()
    marginal = kind == "marginal"
    official_cases_2025 = kind == "official_case"
    waze_matched_2025 = test["waze_matched"].fillna(False).astype(bool).to_numpy()

    def endpoint_masks(buffer: int) -> dict[str, np.ndarray]:
        waze = (kind == "waze_case") & (test["buffer_min"] == buffer).to_numpy()
        return {
            "waze_police_confirmed": waze & test["official_confirmed"].to_numpy(),
            "waze_all": waze,
            "official_all": official_cases_2025,
            "official_waze_reported": official_cases_2025 & waze_matched_2025,
            "official_not_waze_reported": official_cases_2025 & ~waze_matched_2025,
        }

    def matched_sets(buffer: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        waze = np.isin(kind, ["waze_case", "waze_control"]) & (test["buffer_min"] == buffer).to_numpy() \
            & test["complete_risk_set"].fillna(False).astype(bool).to_numpy()
        confirmed = test.loc[(kind == "waze_case") & test["official_confirmed"].to_numpy(), "stratum_id"].unique()
        official = np.isin(kind, ["official_case", "official_control"])
        return {
            "waze_police_confirmed": (waze & test["stratum_id"].isin(confirmed).to_numpy(), kind == "waze_case"),
            "waze_all": (waze, kind == "waze_case"),
            "official_all": (official, kind == "official_case"),
        }

    metrics, place, conditional = {}, {}, {}
    for key, score in scores.items():
        buffer = int(key.split("__b")[1])
        sorted_marginal = np.sort(score[marginal])
        thresholds = {"top1pct": np.quantile(sorted_marginal, 0.99), "top5pct": np.quantile(sorted_marginal, 0.95)}
        metrics[key] = {"population": {}, "matched": {}}
        for endpoint, mask in endpoint_masks(buffer).items():
            values = placements(score[mask], sorted_marginal)
            place[(key, endpoint)] = values
            metrics[key]["population"][endpoint] = {
                "n_positives": int(mask.sum()), "roc_auc": float(values.mean()),
                "roc_auc_ci95": interval(values, rng, replicates),
                "share_in_top1pct_units": float((score[mask] > thresholds["top1pct"]).mean()),
                "share_in_top5pct_units": float((score[mask] > thresholds["top5pct"]).mean()),
            }
        for endpoint, (rows, case) in matched_sets(buffer).items():
            values = within_stratum(test.loc[rows], score[rows], case[rows])
            conditional[(key, endpoint)] = values
            metrics[key]["matched"][endpoint] = {
                "risk_sets": int(len(values)), "within_set_concordance": float(values.mean()),
                "ci95": interval(values, rng, replicates),
                "pooled_roc_auc": float(roc_auc_score(case[rows], score[rows])),
            }
        metrics[key]["mean_predicted_unit_risk_2025"] = float(score[marginal].mean())
    return metrics, place, conditional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    train = pd.read_parquet(DATA / "minute_dataset_2024.parquet")
    test = pd.read_parquet(DATA / "minute_dataset_2025.parquet")
    dataset_summary = json.loads((DATA / "minute_dataset_summary.json").read_text(encoding="utf-8"))
    units = {year: dataset_summary["years"][str(year)]["population_edge_5min_units"] for year in (2024, 2025)}
    for frame in (train, test):
        frame["official_confirmed"] = frame["official_confirmed"].fillna(False).astype(bool)
    x_train_raw, x_test_raw = raw_features(train), raw_features(test)
    is_u = (train["kind"] == "marginal").to_numpy()
    full = Preprocessor(TRAFFIC + CONTEXT).fit(x_train_raw.loc[is_u])
    context_only = Preprocessor(CONTEXT).fit(x_train_raw.loc[is_u])
    x_train = {"full": full.transform(x_train_raw), "context": context_only.transform(x_train_raw)}
    x_test = {"full": full.transform(x_test_raw), "context": context_only.transform(x_test_raw)}

    # Priors per edge x 5-min unit.
    official_2024 = int((train["kind"] == "official_case").sum())
    prior_injury = official_2024 / units[2024]
    prior_all = prior_injury * ALL_TO_INJURY_RATIO

    labels = label_weights(train, test, x_train_raw, x_test_raw, is_u)
    credibility_weight, inverse_propensity = labels["credibility_weight"], labels["inverse_propensity"]
    official_cases_2025 = (test["kind"] == "official_case").to_numpy()

    def positives(buffer: int, which: str) -> np.ndarray:
        return positive_mask(train, buffer, which)

    arms = {
        # name: (positives, weighting, mode, prior target, feature set)
        "pn_official_hour": ("official_hour", "uniform", "pn", "labeled", "full"),
        "pn_reliable": ("reliable", "uniform", "pn", "labeled", "full"),
        "nnpu_reliable": ("reliable", "uniform", "nnpu", "injury", "full"),
        "nnpu_reliable_propensity": ("reliable", "propensity", "nnpu", "injury", "full"),
        "pn_union": ("union", "uniform", "pn", "labeled", "full"),
        "pn_union_context_only": ("union", "uniform", "pn", "labeled", "context"),
        "mswspu_uniform": ("union", "uniform", "nnpu", "all", "full"),
        "mswspu_credibility": ("union", "credibility", "nnpu", "all", "full"),
        "mswspu_credibility_propensity": ("union", "credibility+propensity", "nnpu", "all", "full"),
        "mswspu_credibility_half_prior": ("union", "credibility", "nnpu", "all*0.5", "full"),
        "mswspu_credibility_double_prior": ("union", "credibility", "nnpu", "all*2", "full"),
    }
    sensitivity_arms = ("pn_reliable", "pn_union", "mswspu_credibility")
    runs = [(name, PRIMARY_BUFFER) for name in arms] + [
        (name, buffer) for buffer in BUFFERS if buffer != PRIMARY_BUFFER for name in sensitivity_arms]

    scores, fits = {}, {}
    for name, buffer in runs:
        which, weighting, mode, target, feature_set = arms[name]
        mask = positives(buffer, which)
        weight = np.ones(mask.sum())
        if "credibility" in weighting:
            weight = weight * credibility_weight[mask]
        if "propensity" in weighting:
            weight = weight * inverse_propensity[mask]
        labeled_rate = mask.sum() / units[2024]
        prior = {"labeled": labeled_rate, "injury": prior_injury, "all": prior_all,
                 "all*0.5": prior_all * 0.5, "all*2": prior_all * 2}[target]
        xp, xu = x_train[feature_set][mask], x_train[feature_set][is_u]
        models, notes = [], []
        for seed in range(args.seeds):
            model, note = fit_mlp(xp, weight, xu, prior, mode, SEED + seed, device)
            models.append(model)
            notes.append(note)
        key = f"{name}__b{buffer}"
        scores[key] = predict(models, x_test[feature_set], device)
        fits[key] = {"positives": int(mask.sum()), "effective_positive_weight": float(weight.sum()),
                     "prior": float(prior), "mode": mode, "weighting": weighting, "features": feature_set,
                     "best_steps": [note["best_step"] for note in notes],
                     "steps_with_negative_risk": [note["steps_with_negative_risk"] for note in notes]}
        print(f"fitted {key}: positives={mask.sum()}, prior={prior:.3e}, steps={fits[key]['best_steps']}", flush=True)

    # Tree references with U as negatives (ranking only).
    for name, which in (("hgb_pn_official_hour", "official_hour"), ("hgb_pn_reliable", "reliable"), ("hgb_pn_union", "union")):
        mask = positives(PRIMARY_BUFFER, which)
        rows = mask | is_u
        target = mask[rows].astype(int)
        weight = np.where(target == 1, len(target) / (2 * target.sum()), len(target) / (2 * (len(target) - target.sum())))
        columns = TRAFFIC + CONTEXT
        tree = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=False, random_state=SEED)
        tree.fit(x_train_raw.loc[rows, columns], target, sample_weight=weight)
        scores[f"{name}__b{PRIMARY_BUFFER}"] = tree.predict_proba(x_test_raw[columns])[:, 1]
        print(f"fitted {name}", flush=True)

    # ---- Evaluation on 2025 ----
    rng = np.random.default_rng(SEED)
    marginal = test["kind"].to_numpy() == "marginal"
    metrics, place, conditional = evaluate(test, scores, rng, args.replicates)

    def paired(a: str, b: str, endpoint: str, store: dict) -> dict:
        delta = store[(f"{a}__b{PRIMARY_BUFFER}", endpoint)] - store[(f"{b}__b{PRIMARY_BUFFER}", endpoint)]
        return {"delta": float(delta.mean()), "ci95": interval(delta, rng, args.replicates)}

    comparisons = {}
    pairs = [
        ("pn_union", "pn_reliable", "weak Waze positives added (no PU)"),
        ("nnpu_reliable", "pn_reliable", "nnPU vs U-as-negative, reliable positives"),
        ("mswspu_uniform", "pn_union", "nnPU vs U-as-negative, all Waze positives"),
        ("mswspu_credibility", "mswspu_uniform", "credibility weights for weak positives"),
        ("mswspu_credibility_propensity", "mswspu_credibility", "reporting-propensity (SAR) correction"),
        ("nnpu_reliable_propensity", "nnpu_reliable", "reporting-propensity (SAR) correction, reliable only"),
        ("mswspu_credibility", "pn_official_hour", "Waze-timed MSWS-PU vs hour-timed police labels"),
        ("pn_union", "pn_union_context_only", "minute traffic added to time/location context"),
    ]
    for a, b, meaning in pairs:
        comparisons[f"{a}_minus_{b}"] = {
            "meaning": meaning,
            "population": {endpoint: paired(a, b, endpoint, place) for endpoint in
                           ("waze_police_confirmed", "official_all", "official_not_waze_reported")},
            "matched": {endpoint: paired(a, b, endpoint, conditional) for endpoint in
                        ("waze_police_confirmed", "waze_all", "official_all")},
        }

    # Absolute-risk check against the complete police register at hour level.
    official_hours_2025 = int(official_cases_2025.sum())
    population_hours_2025 = units[2025] / 12
    calibration = {"observed_injury_rate_per_edge_hour_2025": official_hours_2025 / population_hours_2025, "arms": {}}
    for name in ("pn_reliable", "nnpu_reliable", "nnpu_reliable_propensity", "pn_official_hour"):
        score = scores[f"{name}__b{PRIMARY_BUFFER}"]
        hourly = 1 - (1 - score) ** 12
        edges_ = np.quantile(hourly[marginal], np.linspace(0, 1, 11))
        edges_[0], edges_[-1] = -np.inf, np.inf
        decile_u = np.digitize(hourly[marginal], edges_[1:-1])
        decile_p = np.digitize(hourly[official_cases_2025], edges_[1:-1])
        rows = []
        for decile in range(10):
            share = (decile_u == decile).mean()
            rows.append({"decile": decile + 1,
                         "predicted": float(hourly[marginal][decile_u == decile].mean()),
                         "observed": float((decile_p == decile).sum() / (population_hours_2025 * share))})
        calibration["arms"][name] = {
            "mean_predicted_hourly_risk": float(hourly[marginal].mean()),
            "predicted_over_observed": float(hourly[marginal].mean() / calibration["observed_injury_rate_per_edge_hour_2025"]),
            "deciles": rows,
        }

    results = {
        "design": {
            "unit": "edge x 5-min", "train_year": 2024, "test_year": 2025, "primary_buffer_min": PRIMARY_BUFFER,
            "feature_window": "W=[t_end-30,t_end), C=[t_end-120,t_end-30); Waze rows: t_end = first report - buffer",
            "model": "MLP 64-32, full batch AdamW, logistic loss, mean of seeds", "seeds": args.seeds, "device": device,
            "test_status": "2025 previously examined; exploratory",
        },
        "priors_per_unit": {"injury_from_police_register": prior_injury, "all_crashes_assumed": prior_all,
                            "all_to_injury_ratio": ALL_TO_INJURY_RATIO},
        "credibility_model": labels["credibility_model"],
        "propensity_model": labels["propensity_model"],
        "hourly_reference": hourly_reference(test),
        "fits": fits, "metrics": metrics, "paired_comparisons": comparisons, "hourly_calibration": calibration,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    predictions = test[["row_id", "kind", "stratum_id", "edge_id", "t_end", "buffer_min", "official_confirmed"]].copy()
    for key, score in scores.items():
        predictions[key] = score
    predictions.to_parquet(OUT / "test_predictions_2025.parquet", index=False)
    write_report(results)


def hourly_reference(test: pd.DataFrame) -> dict:
    """Phase 2 hourly models on the same official risk sets, in the same metric."""
    hourly = pd.read_parquet(HOURLY_PREDICTIONS)
    case = hourly["target_official"].to_numpy() == 1
    rng = np.random.default_rng(SEED)
    output = {}
    for column in ("official_only_hist_gradient_boosting", "hard_union_hist_gradient_boosting"):
        values = within_stratum(hourly, hourly[column].to_numpy(), case)
        output[column] = {"risk_sets": int(len(values)), "within_set_concordance": float(values.mean()),
                          "ci95": interval(values, rng, 1000)}
    return output


def write_report(results: dict) -> None:
    b = results["design"]["primary_buffer_min"]
    metrics, years = results["metrics"], None
    lines = [
        "# Phase 6: minute-level MSWS-PU sub-study", "",
        f"Units are edge × 5 minutes. Trained on 2024, evaluated on 2025. Waze events are referenced to the first-report time t0, and the feature window ends {b} minutes before t0 (W is 30 minutes, C the 90 minutes before it); "
        "official crashes only have an hour, so their feature window ends at the start of the crash hour. U is a label-independent marginal sample from which known events are not removed. 2025 has been inspected before, so the results are exploratory.", "",
        f"Prior per unit: injury crashes {results['priors_per_unit']['injury_from_police_register']:.3e} (from the complete police register); "
        f"all crashes {results['priors_per_unit']['all_crashes_assumed']:.3e} (multiplied by the Bavarian motorway all/injury crash ratio {results['priors_per_unit']['all_to_injury_ratio']:.2f}; an assumption).", "",
        "## Population view: ROC-AUC of 2025 positives against marginal units [95% CI]", "",
        "| Model | Training positives | Waze-timed and police-confirmed | All official crashes (hourly) | Of which not reported in Waze | Official crashes covered by top 5% of units |", "|---|---:|---:|---:|---:|---:|",
    ]

    def cell(entry):
        return f"{entry['roc_auc']:.3f} [{entry['roc_auc_ci95'][0]:.3f}, {entry['roc_auc_ci95'][1]:.3f}]"

    primary = [key for key in metrics if key.endswith(f"__b{b}")]
    for key in primary:
        population = metrics[key]["population"]
        count = results["fits"].get(key, {}).get("positives", "")
        lines.append(f"| {key.split('__')[0]} | {count} | {cell(population['waze_police_confirmed'])} | {cell(population['official_all'])} | "
                     f"{cell(population['official_not_waze_reported'])} | {population['official_all']['share_in_top5pct_units']:.3f} |")
    lines += ["", "## Matched view: share of risk sets in which the case scores above its controls [95% CI] (0.5 = no information)", "",
              "| Model | Waze-timed and police-confirmed | All Waze-timed events | Official crashes (hourly, Phase 2 risk sets) |", "|---|---:|---:|---:|"]
    for key in primary:
        matched = metrics[key]["matched"]
        cells = [f"{matched[e]['within_set_concordance']:.3f} [{matched[e]['ci95'][0]:.3f}, {matched[e]['ci95'][1]:.3f}]"
                 for e in ("waze_police_confirmed", "waze_all", "official_all")]
        lines.append(f"| {key.split('__')[0]} | {' | '.join(cells)} |")
    lines += ["", "Reference: the same metric for the Phase 2 hourly traffic-feature models on the same official risk sets: " + "; ".join(
        f"{name} {item['within_set_concordance']:.3f} [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}]"
        for name, item in results["hourly_reference"].items()) + "."]
    lines += ["", "## Paired differences [95% CI]", "",
              "| Contrast | Meaning | Population: Waze-timed and police-confirmed | Population: all official | Population: official not reported in Waze | Matched: all Waze-timed | Matched: official |",
              "|---|---|---:|---:|---:|---:|---:|"]

    def delta(entry):
        return f"{entry['delta']:+.4f} [{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}]"

    for name, item in results["paired_comparisons"].items():
        lines.append(f"| {name} | {item['meaning']} | {delta(item['population']['waze_police_confirmed'])} | "
                     f"{delta(item['population']['official_all'])} | {delta(item['population']['official_not_waze_reported'])} | "
                     f"{delta(item['matched']['waze_all'])} | {delta(item['matched']['official_all'])} |")
    lines += ["", "## Sensitivity to the buffer before the first report (matched view, all Waze-timed events)", "", "| Model | " + " | ".join(f"buffer {x} min" for x in BUFFERS) + " |",
              "|---|" + "---:|" * len(BUFFERS)]
    for name in ("pn_reliable", "pn_union", "mswspu_credibility"):
        cells = []
        for buffer in BUFFERS:
            entry = metrics[f"{name}__b{buffer}"]["matched"]["waze_all"]
            cells.append(f"{entry['within_set_concordance']:.3f} [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}]")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    calibration = results["hourly_calibration"]
    lines += ["", "## Absolute-risk calibration against the complete police register (injury crashes per edge-hour)", "",
              f"Observed 2025 rate: {calibration['observed_injury_rate_per_edge_hour_2025']:.3e}. Predicted hourly risk is 1−(1−p)^12.", "",
              "| Model | Mean predicted hourly risk | Predicted/observed | Top decile: predicted | Top decile: observed |", "|---|---:|---:|---:|---:|"]
    for name, item in calibration["arms"].items():
        top = item["deciles"][-1]
        lines.append(f"| {name} | {item['mean_predicted_hourly_risk']:.3e} | {item['predicted_over_observed']:.2f} | {top['predicted']:.3e} | {top['observed']:.3e} |")
    credibility, propensity = results["credibility_model"], results["propensity_model"]
    lines += ["", "## Auxiliary models", "",
              f"- Credibility model (Waze report count, confidence, reliability, subtype → police injury record present): 2025 AUC {credibility['auc_2025']:.3f}; mean weight of weak positives {credibility['mean_weight_weak_positives']:.3f}. "
              "The weight is a relative credibility, not a posterior probability of a 'real crash'.",
              f"- Reporting-propensity model (whether a police crash was reported in Waze): reported share 2024/2025 {propensity['waze_reported_share_of_police_crashes']['2024']:.3f}/{propensity['waze_reported_share_of_police_crashes']['2025']:.3f}; 2025 AUC {propensity['auc_2025']:.3f}.",
              "", "Intervals are a bootstrap over positives (or risk sets) with fixed models, without retraining uncertainty, spatio-temporal correlation or multiple-comparison correction. The first Waze report lags the crash, so the buffer cannot guarantee that the window lies entirely before the crash.",
              "The official register has no direction field, but the coordinates fall on a specific carriageway: the crash carriageway is inferred from the travel direction of Waze reports at other times within 15–25 m of the coordinates, otherwise from the crash's own Waze crash report; about half of the crashes have no such evidence and keep the nearest edge (see official_direction in minute_dataset_summary.json). Control hours are not re-selected after the carriageway adjustment.", "",
              "Reproduction: `python analysis/phase6_minute_dataset.py`, then `python analysis/phase6_minute_pu.py`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
