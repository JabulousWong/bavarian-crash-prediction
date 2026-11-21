"""Known-truth PU stress tests; illustrative mechanisms, not a crash-data model.

All learners use the same linear logistic score and unbalanced population losses.
nnPU minimizes Kiryo et al. (2017), Eq. 6, with logistic loss and L2 penalty:
    pi * mean_P softplus(-g) + max(0, mean_U softplus(g)
                                      - pi * mean_P softplus(g)).
The correctly sampled U is an independent draw from p(x), retaining event rows.
No latent training labels enter a non-oracle learner or hyperparameter choice.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import check_grad, minimize
from scipy.special import expit, logit
from scipy.stats import t
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "results" / "phase3" / "pu_simulation"
SHIFT = np.array([0.55, 0.45, 0.25, 0.0, 0.0])
SCENARIOS = ("scar_clean", "scar_waze_fp", "sar_clean", "sar_waze_fp")
BASE_SEED = 20260929
L2 = 0.0001
METRICS = ("roc_auc", "average_precision", "brier", "log_loss", "mean_prediction", "calibration_bias")
SOURCES = [
    {"title": "Kiryo et al. (2017), Positive-Unlabeled Learning with Non-Negative Risk Estimator",
     "url": "https://papers.nips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf",
     "used_for": "Independent P~p(x|Y=1), U~p(x); class prior and risk equations (2), (6)."},
    {"title": "du Plessis et al. (2015), Convex Formulation for Learning from Positive and Unlabeled Data",
     "url": "https://proceedings.mlr.press/v37/plessis15.html",
     "used_for": "Unbiased risk decomposition and convex linear logistic formulation."},
]


def draw_population(rng: np.random.Generator, n: int, prior: float):
    y = rng.random(n) < prior
    x = rng.normal(size=(n, len(SHIFT))) + y[:, None] * SHIFT
    return x, y.astype(np.int8)


def observe_labels(x, y, official_uniform, waze_uniform, scenario):
    # Perfect specificity for official labels. SAR selection distorts p(x|Y=1).
    capture = expit(-1.0 + 1.6 * x[:, 0] - 1.0 * x[:, 1]) if scenario.startswith("sar") else 0.30
    official = y.astype(bool) & (official_uniform < capture)
    # Waze true-positive capture is 0.5 in all scenarios. False reports depend
    # on nuisance feature x3, producing an explicit spurious association.
    fp = expit(-6.3 + 1.2 * x[:, 3]) if scenario.endswith("fp") else np.zeros(len(y))
    waze = waze_uniform < np.where(y, 0.50, fp)
    return official, waze


def with_intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def supervised_objective(theta, x, target):
    z = x @ theta
    penalty = 0.5 * L2 * np.dot(theta[1:], theta[1:])
    gradient = x.T @ (expit(z) - target) / len(x)
    gradient[1:] += L2 * theta[1:]
    return np.mean(np.logaddexp(0.0, z) - target * z) + penalty, gradient


def pu_parts(theta, p, u, prior):
    zp, zu = p @ theta, u @ theta
    positive = prior * np.logaddexp(0.0, -zp).mean()
    negative = np.logaddexp(0.0, zu).mean() - prior * np.logaddexp(0.0, zp).mean()
    gp = prior * (p.T @ (expit(zp) - 1)) / len(p)
    gn = u.T @ expit(zu) / len(u) - prior * p.T @ expit(zp) / len(p)
    return positive, negative, gp, gn


def pu_objective(theta, p, u, prior):
    positive, negative, gp, gn = pu_parts(theta, p, u, prior)
    gradient = gp + (gn if negative > 0 else 0)
    gradient[1:] += L2 * theta[1:]
    loss = positive + max(0.0, negative) + 0.5 * L2 * np.dot(theta[1:], theta[1:])
    return loss, gradient


def fit_logistic(x, target):
    theta = np.zeros(x.shape[1])
    theta[0] = logit(np.clip(np.mean(target), 1e-6, 1 - 1e-6))
    result = minimize(supervised_objective, theta, args=(x, target), jac=True,
                      method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7})
    return result.x, {"converged": bool(result.success), "iterations": int(result.nit),
                      "objective": float(result.fun), "message": str(result.message),
                      "negative_component": None, "solver": "L-BFGS-B"}


def fit_pu(p, u, prior):
    theta = np.zeros(p.shape[1])
    theta[0] = logit(prior)
    result = minimize(pu_objective, theta, args=(p, u, prior), jac=True,
                      method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7})
    solver = "L-BFGS-B"
    negative = pu_parts(result.x, p, u, prior)[1]
    if not result.success or abs(negative) < 1e-6:
        # Exact epigraph of max(A, B), where A=pi*Rp+ and
        # B=Ru- - pi*mean_P(g). Both branches are convex for linear logistic.
        # SLSQP resolves the nondifferentiable crossing without smoothing Eq. 6.
        def branches(v):
            positive, neg, gp, gn = pu_parts(v[:-1], p, u, prior)
            return np.array([v[-1] - positive, v[-1] - positive - neg])

        def branch_jac(v):
            _, _, gp, gn = pu_parts(v[:-1], p, u, prior)
            return np.vstack((np.r_[-gp, 1.0], np.r_[-gp - gn, 1.0]))

        def epigraph(v):
            gradient = np.zeros_like(v)
            gradient[1:-1] = L2 * v[1:-1]
            gradient[-1] = 1.0
            return v[-1] + 0.5 * L2 * np.dot(v[1:-1], v[1:-1]), gradient

        positive, neg, _, _ = pu_parts(result.x, p, u, prior)
        fallback = minimize(epigraph, np.r_[result.x, positive + max(0.0, neg)],
                            jac=True, method="SLSQP",
                            constraints={"type": "ineq", "fun": branches, "jac": branch_jac},
                            options={"maxiter": 250, "ftol": 1e-10})
        result = fallback
        theta = fallback.x[:-1]
        solver = "SLSQP epigraph"
    else:
        theta = result.x
    positive, negative, _, _ = pu_parts(theta, p, u, prior)
    return theta, {"converged": bool(result.success), "iterations": int(result.nit),
                   "objective": float(pu_objective(theta, p, u, prior)[0]),
                   "message": str(result.message), "negative_component": float(negative), "solver": solver}


def evaluate(y, score):
    return {"roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, score)),
            "log_loss": float(log_loss(y, score)),
            "mean_prediction": float(np.mean(score)),
            "calibration_bias": float(np.mean(score) - np.mean(y))}


def mc_summary(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if n > 1 else None
    half_width = float(t.ppf(0.975, n - 1) * se) if n > 1 else None
    return {"n_seeds": n, "mean": mean, "sd_across_seeds": sd, "mc_standard_error": se,
            "mc_mean_ci95": [mean - half_width, mean + half_width] if n > 1 else None}


def verification():
    rng = np.random.default_rng(12233)
    p = with_intercept(rng.normal(size=(30, 5)))
    u = with_intercept(rng.normal(size=(100, 5)))
    theta = np.array([-2.0, .2, -.1, .3, .2, -.4])
    errors = {}
    # Check both active/non-active negative-risk branches away from the kink.
    for prior in (.1, .9):
        point = theta if prior == .1 else np.array([2.0, .2, -.1, .3, .2, -.4])
        error = check_grad(lambda z: pu_objective(z, p, u, prior)[0],
                           lambda z: pu_objective(z, p, u, prior)[1], point)
        errors[str(prior)] = float(error)
    # Force a negative correction with different P and U support.
    p2 = with_intercept(np.tile([3., 0, 0, 0, 0], (30, 1)))
    u2 = with_intercept(np.tile([-3., 0, 0, 0, 0], (100, 1)))
    point2 = np.array([0., 1., 0., 0., 0., 0.])
    error = check_grad(lambda z: pu_objective(z, p2, u2, .2)[0],
                       lambda z: pu_objective(z, p2, u2, .2)[1], point2)
    errors["clipped_negative_branch"] = float(error)
    positive, negative, _, _ = pu_parts(point2, p2, u2, .2)
    assert negative < 0
    assert max(errors.values()) < 2e-6, errors
    # Algebraic identity is exact when P/U are the same fully labeled finite
    # population and pi equals its positive fraction; labels only verify math.
    x, y = draw_population(rng, 2000, .1)
    design = with_intercept(x)
    prior = y.mean()
    pos, neg, _, _ = pu_parts(theta, design[y == 1], design, prior)
    supervised = supervised_objective(theta, design, y)[0]
    decomposed = pos + neg + .5 * L2 * np.dot(theta[1:], theta[1:])
    identity_error = float(abs(supervised - decomposed))
    assert identity_error < 1e-12
    return {"gradient_absolute_l2_errors": errors, "finite_population_risk_identity_error": identity_error}


def run_one(seed, prior, scenario, n_train, n_u, n_test):
    # Common random numbers across scenarios/priors improve paired comparisons.
    rng = np.random.default_rng(seed)
    x, y = draw_population(rng, n_train, prior)
    xu, yu = draw_population(rng, n_u, prior)
    # Independent candidate pool for event-excluded and matched controls.
    xb, yb = draw_population(rng, max(3 * n_u, n_train), prior)
    xt, yt = draw_population(rng, n_test, prior)
    official, waze = observe_labels(x, y, rng.random(n_train), rng.random(n_train), scenario)
    off_b, waze_b = observe_labels(xb, yb, rng.random(len(yb)), rng.random(len(yb)), scenario)
    union = official | waze
    eligible = np.flatnonzero(~(off_b | waze_b))
    if len(eligible) < n_u or official.sum() < 10:
        raise RuntimeError("Insufficient positives/candidate controls; increase population size.")
    excluded_ids = eligible[:n_u]
    p = with_intercept(x[official])
    p_union = with_intercept(x[union])
    u = with_intercept(xu)
    design = with_intercept(x)
    test = with_intercept(xt)
    # Four event-free controls per observed official event; matching on x0
    # coarsened into fixed bins distorts the overall x0 distribution in U.
    bins = np.array([-np.inf, -.75, 0.0, .75, 1.5, np.inf])
    p_stratum = np.digitize(x[official, 0], bins[1:-1])
    b_stratum = np.digitize(xb[:, 0], bins[1:-1])
    matched_ids = []
    for stratum, count in zip(*np.unique(p_stratum, return_counts=True)):
        candidates = eligible[b_stratum[eligible] == stratum]
        matched_ids.extend(rng.choice(candidates, 4 * int(count), replace=True).tolist())
    matched_ids = np.asarray(matched_ids)
    # Equal number of marginal controls, to isolate matching from U sample size.
    budget_x, budget_y = draw_population(rng, len(matched_ids), prior)

    learned = {}
    # Official-only excludes Waze-only rows as in the real label comparison.
    keep = (~union) | official
    learned["official_supervised"] = fit_logistic(design[keep], official[keep].astype(float))
    learned["hard_union_supervised"] = fit_logistic(design, union.astype(float))
    soft = np.where(official, 1.0, np.where(waze, .5, 0.0))
    learned["soft_waze_q_0.50"] = fit_logistic(design, soft)
    learned["oracle_latent_supervised"] = fit_logistic(design, y.astype(float))
    learned["nnpu_official_marginal_true_prior"] = fit_pu(p, u, prior)
    learned["nnpu_union_marginal_true_prior"] = fit_pu(p_union, u, prior)
    learned["nnpu_official_marginal_half_prior"] = fit_pu(p, u, prior / 2)
    learned["nnpu_official_marginal_double_prior"] = fit_pu(p, u, prior * 2)
    learned["nnpu_official_event_excluded_u"] = fit_pu(p, with_intercept(xb[excluded_ids]), prior)
    learned["nnpu_official_matched_1to4_u"] = fit_pu(p, with_intercept(xb[matched_ids]), prior)
    learned["nnpu_official_equal_budget_marginal_u"] = fit_pu(p, with_intercept(budget_x), prior)
    records = []
    for model, (theta, diagnostics) in learned.items():
        records.append({"seed": seed, "prior": prior, "scenario": scenario, "model": model,
                        **evaluate(yt, expit(test @ theta)), **diagnostics})
    bayes_theta = np.r_[logit(prior) - 0.5 * np.dot(SHIFT, SHIFT), SHIFT]
    records.append({"seed": seed, "prior": prior, "scenario": scenario, "model": "bayes_true_posterior",
                    **evaluate(yt, expit(test @ bayes_theta)), "converged": True, "iterations": 0,
                    "objective": None, "negative_component": None, "message": "known DGP", "solver": "none"})

    # Risk decomposition for a fixed known Bayes score diagnoses sampling bias
    # independently of fitted-model outcomes and optimization variability.
    fixed_p = np.logaddexp(0.0, p @ bayes_theta).mean()
    fixed_p_positive = np.logaddexp(0.0, -(p @ bayes_theta)).mean()
    fixed_truth_risk = float(np.mean(np.logaddexp(0.0, test @ bayes_theta) - yt * (test @ bayes_theta)))
    sample_checks = {}
    for name, xx, yy in (("marginal", xu, yu), ("event_excluded", xb[excluded_ids], yb[excluded_ids]),
                         ("matched_1to4", xb[matched_ids], yb[matched_ids]),
                         ("equal_budget_marginal", budget_x, budget_y)):
        risk = prior * fixed_p_positive + np.logaddexp(0.0, with_intercept(xx) @ bayes_theta).mean() - prior * fixed_p
        sample_checks[name] = {"n": len(xx), "latent_positive_fraction": float(yy.mean()),
                               "mean_x0": float(xx[:, 0].mean()), "fixed_bayes_pu_risk": float(risk),
                               "fixed_bayes_pu_risk_error": float(risk - fixed_truth_risk)}
    counts = {"seed": seed, "prior": prior, "scenario": scenario,
              "train_latent_positives": int(y.sum()), "official_positives": int(official.sum()),
              "waze_reports": int(waze.sum()), "union_positives": int(union.sum()),
              "waze_precision": float(y[waze].mean()), "union_precision": float(y[union].mean()),
              "official_sensitivity": float(official.sum() / y.sum()), "test_prevalence": float(yt.mean()),
              "positive_x0_mean": float(x[official, 0].mean()), "latent_positive_x0_mean": float(x[y == 1, 0].mean()),
              "fixed_bayes_true_test_risk": fixed_truth_risk, "u_sampling": sample_checks}
    return records, counts


def summarize(frame, counts, config, checks, elapsed):
    summary = {"configuration": config, "verification": checks, "elapsed_seconds": elapsed,
               "sources": SOURCES, "uncertainty": "95% Student-t Monte Carlo intervals for the mean across independent seeds; not real-data confidence intervals.",
               "metrics": {}, "paired_differences": {}, "sampling_diagnostics": {},
               "optimizer": {"fits": int((frame.solver != "none").sum()),
                             "failed_fits": int((~frame.converged).sum()),
                             "nnpu_fits_with_clipped_final_negative_component": int((frame.negative_component.fillna(np.inf) <= 0).sum()),
                             "solvers": frame.solver.value_counts().to_dict()}}
    contrasts = [
        ("hard_union_supervised", "official_supervised"),
        ("soft_waze_q_0.50", "hard_union_supervised"),
        ("nnpu_official_marginal_true_prior", "official_supervised"),
        ("nnpu_official_marginal_true_prior", "hard_union_supervised"),
        ("nnpu_union_marginal_true_prior", "nnpu_official_marginal_true_prior"),
        ("nnpu_official_marginal_half_prior", "nnpu_official_marginal_true_prior"),
        ("nnpu_official_marginal_double_prior", "nnpu_official_marginal_true_prior"),
        ("nnpu_official_event_excluded_u", "nnpu_official_marginal_true_prior"),
        ("nnpu_official_matched_1to4_u", "nnpu_official_equal_budget_marginal_u"),
    ]
    for (prior, scenario), group in frame.groupby(["prior", "scenario"]):
        key = f"prior_{prior:g}__{scenario}"
        summary["metrics"][key] = {model: {metric: mc_summary(rows[metric]) for metric in METRICS}
                                   for model, rows in group.groupby("model")}
        indexed = group.set_index(["model", "seed"])
        summary["paired_differences"][key] = {}
        for a, b in contrasts:
            delta = indexed.loc[a, list(METRICS)] - indexed.loc[b, list(METRICS)]
            summary["paired_differences"][key][f"{a}_minus_{b}"] = {metric: mc_summary(delta[metric]) for metric in METRICS}
        selected = [row for row in counts if row["prior"] == prior and row["scenario"] == scenario]
        diag = {name: mc_summary([row[name] for row in selected]) for name in
                ("official_positives", "waze_precision", "union_precision", "official_sensitivity", "positive_x0_mean", "latent_positive_x0_mean")}
        diag["u_sampling"] = {name: {metric: mc_summary([row["u_sampling"][name][metric] for row in selected])
                                      for metric in ("n", "latent_positive_fraction", "mean_x0", "fixed_bayes_pu_risk_error")}
                               for name in selected[0]["u_sampling"]}
        summary["sampling_diagnostics"][key] = diag
    return summary


def write_report(summary, path):
    config = summary["configuration"]
    prior_text = "/".join(f"{100 * value:g}%" for value in config["priors"])
    lines = ["# Known-truth PU mechanism simulation", "",
             f"Run: {config['repetitions']} independent seeds per cell; {len(config['priors'])} priors x 4 mechanisms. "
             f"Training population {config['n_train']:,}, marginal U {config['n_u']:,}, independent population test {config['n_test']:,} per seed.", "",
             "This is a synthetic mechanism check, not evidence that the real official/Waze labels satisfy PU assumptions. "
             f"The illustrative {prior_text} probabilities are chosen for controlled stress experiments; "
             "they are not estimates for the detector network. A rare-event run can match an empirically relevant order of magnitude without identifying the true prior. "
             "Synthetic observations are IID; there is no claim to reproduce cross-year drift, crash severity, location error, or temporal clustering.", "",
             "Y is Bernoulli(pi); X|Y is N(Y*[0.55,0.45,0.25,0,0], I). Under SCAR, official capture is 0.30. "
             "Under SAR it is sigmoid(-1+1.6*x0-x1). Official positives have no false reports. Waze captures each true positive with probability 0.50; "
             "in the noise scenario its false-positive probability is sigmoid(-6.3+1.2*x3), otherwise zero. "
             "The same random draws are reused across mechanisms within a seed.", "",
             "All fitted models use a linear logistic score with fixed L2=0.0001, no class balancing, and no tuning on latent test labels. "
             "Official-only supervision omits Waze-only rows. Hard union uses every observed report; soft supervision uses fractional target 0.5 for Waze-only rows. "
             "The oracle is trained on latent labels; Bayes is the known DGP posterior. These baselines differ from the real matched/boosting experiment and their absolute metrics must not be compared with it.", "",
             "nnPU uses pi*E_P[softplus(-g)] + max(0, E_U[softplus(g)] - pi*E_P[softplus(g)]). "
             "The positive sample uses official observations unless the method name says union. Correct U is independent p(x), with no event exclusions. "
             "Wrong event-excluded U conditions on no official/Waze report; wrong matched U contains four event-free controls per official positive, sampled with replacement in fixed x0 strata. "
             "Equal-budget marginal U has the same number of rows as matched U. Priors in each method name refer to the known generating prior; half/double explicitly misspecify it. "
             "SCAR and perfect positive specificity are deliberately violated in the SAR and noisy-union variants.", "",
             "Intervals below are 95% Student-t Monte Carlo intervals for mean performance across independent seeds. "
             "Paired differences use the same seed and test set. They quantify simulation precision, not empirical uncertainty about traffic accidents. "
             "AP denotes average precision, not trapezoidal PR area. Brier/log loss and calibration bias refer to latent population outcomes.", ""]
    first_prior = min(config["priors"])
    clean = summary["metrics"][f"prior_{first_prior:g}__scar_clean"]
    sar = summary["metrics"][f"prior_{first_prior:g}__sar_clean"]
    matched = summary["paired_differences"][f"prior_{first_prior:g}__scar_clean"]["nnpu_official_matched_1to4_u_minus_nnpu_official_equal_budget_marginal_u"]["roc_auc"]
    noisy = summary["paired_differences"][f"prior_{first_prior:g}__scar_waze_fp"]["hard_union_supervised_minus_official_supervised"]["roc_auc"]
    lines += ["## Observed simulation findings", "",
              f"At prior={first_prior:g} under SCAR without Waze noise, official-only ROC-AUC is {clean['official_supervised']['roc_auc']['mean']:.4f} "
              f"and correct-prior official nnPU is {clean['nnpu_official_marginal_true_prior']['roc_auc']['mean']:.4f}. "
              f"Mean predicted probability changes from {clean['official_supervised']['mean_prediction']['mean']:.5f} "
              f"to {clean['nnpu_official_marginal_true_prior']['mean_prediction']['mean']:.5f}. "
              "The main improvement here is probability calibration, with little ranking change.", "",
              f"At the same prior with SAR ascertainment, correct-prior official nnPU has ROC-AUC "
              f"{sar['nnpu_official_marginal_true_prior']['roc_auc']['mean']:.4f} and Brier "
              f"{sar['nnpu_official_marginal_true_prior']['brier']['mean']:.6f}; official supervision has "
              f"{sar['official_supervised']['roc_auc']['mean']:.4f} and {sar['official_supervised']['brier']['mean']:.6f}. "
              "Forcing the correct overall mean risk does not fix conditional bias in the positive sample.", "",
              f"Under SCAR at prior={first_prior:g}, matched U versus equal-budget marginal U changes ROC-AUC by {matched['mean']:+.4f}. "
              f"With feature-dependent Waze false reports, hard union versus official supervision changes ROC-AUC by {noisy['mean']:+.4f}. "
              "These results show possible failure mechanisms, not expected effect sizes for the real data. Exact Monte Carlo intervals are in the paired-difference table and JSON.", "",
             "## Main mean metrics", "",
             "| Prior | Mechanism | Model | ROC-AUC | AP | Brier | Log loss | Mean predicted risk |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    main_models = ("official_supervised", "hard_union_supervised", "soft_waze_q_0.50", "nnpu_official_marginal_true_prior",
                   "nnpu_union_marginal_true_prior", "oracle_latent_supervised", "bayes_true_posterior")
    for cell, models in summary["metrics"].items():
        prior, scenario = cell.removeprefix("prior_").split("__")
        for model in main_models:
            row = models[model]
            values = " | ".join(f"{row[metric]['mean']:.5f}" for metric in ("roc_auc", "average_precision", "brier", "log_loss", "mean_prediction"))
            lines.append(f"| {prior} | {scenario} | {model} | {values} |")
    lines += ["", "## Paired stress-test differences", "",
              "Differences are first model minus reference; positive ROC-AUC/AP is better, positive Brier/log loss is worse.", "",
              "| Cell | Contrast | ROC-AUC difference [95% MC CI] | Brier difference [95% MC CI] |",
              "|---|---|---:|---:|"]
    for cell, contrasts in summary["paired_differences"].items():
        for name, values in contrasts.items():
            formatted = []
            for metric in ("roc_auc", "brier"):
                stat = values[metric]
                ci = stat["mc_mean_ci95"]
                formatted.append(f"{stat['mean']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else f"{stat['mean']:+.5f}")
            lines.append(f"| {cell} | {name} | {' | '.join(formatted)} |")
    lines += ["", "## Interpretation boundaries", "",
              "- Correct marginal sampling and a correct prior fix the PU risk identity only when the supplied positives represent the target positive class. SAR ascertainment and contaminated positive sets break that condition.",
              "- The marginal risk includes both observed and unobserved event rows. Event-excluded controls are not a draw from p(x); matching also changes covariate frequencies. At low prevalence the event-exclusion effect can be small, which does not make the sampling identity valid.",
              "- A correct prior can chiefly improve probability calibration while providing little ranking benefit under SCAR; missing-at-random label thinning preserves ranking in the infinite-sample ideal. Ranking metrics alone do not establish PU benefit.",
              "- No class prior is estimated here: half/true/double inputs are controlled stress tests. No claim is made that aggregate official financial-damage counts identify the target network's site-hour prior.",
              "- Official injury crashes and all-severity crashes may be different target outcomes, not merely randomly observed versions of the same outcome. The simulation cannot resolve this endpoint choice.",
              "- No neural network, tree learner, real-data PU performance claim, or decision threshold is validated by this simulation. Fixed regularization and one Gaussian DGP limit generalization.", "",
              f"Optimization: {summary['optimizer']['fits']} fits; {summary['optimizer']['failed_fits']} failures. "
              f"Runtime {summary['elapsed_seconds']:.1f} seconds. Analytic gradients and the finite-population risk identity passed checks. "
              f"The negative risk was clipped at the final fit in {summary['optimizer']['nnpu_fits_with_clipped_final_negative_component']} nnPU fits; "
              "the low-capacity linear model therefore does not by itself demonstrate the overfitting-protection advantage of clipping over ordinary unbiased PU.", "",
              "## Sources", ""]
    lines += [f"- [{source['title']}]({source['url']}): {source['used_for']}" for source in SOURCES]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=30000)
    parser.add_argument("--n-u", type=int, default=12000)
    parser.add_argument("--n-test", type=int, default=40000)
    parser.add_argument("--priors", type=float, nargs="+", default=[.01, .05])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.repetitions < 1 or not all(0 < prior < .5 for prior in args.priors):
        parser.error("Need positive repetitions and priors in (0, 0.5).")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    all_records, all_counts = [], []
    with threadpool_limits(limits=1):
        checks = verification()
        for prior in args.priors:
            for scenario in SCENARIOS:
                for repetition in range(args.repetitions):
                    seed = BASE_SEED + repetition
                    records, counts = run_one(seed, prior, scenario, args.n_train, args.n_u, args.n_test)
                    all_records.extend(records)
                    all_counts.append(counts)
                print(f"Completed prior={prior:g} scenario={scenario} seeds={args.repetitions}; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    frame = pd.DataFrame(all_records)
    config = {"repetitions": args.repetitions, "n_train": args.n_train, "n_u": args.n_u, "n_test": args.n_test,
              "priors": args.priors, "scenarios": list(SCENARIOS), "base_seed": BASE_SEED,
              "feature_positive_mean_shift": SHIFT.tolist(), "l2": L2,
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
              "model_family": "linear logistic", "test_endpoint": "independent population latent Y",
              "prior_status": "illustrative known DGP inputs; not inferred from real accident statistics"}
    result = summarize(frame, all_counts, config, checks, time.perf_counter() - start)
    frame.to_csv(args.out / "per_seed_metrics.csv", index=False)
    (args.out / "sampling_diagnostics.json").write_text(json.dumps(all_counts, indent=2), encoding="utf-8")
    (args.out / "simulation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.out / "REPORT.md")
    print(json.dumps({"output": str(args.out), "optimizer": result["optimizer"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
