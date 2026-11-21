"""Known-truth pilot of PU-guided augmentation; heuristic auxiliary risk."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from phase3_pu_simulation import (
    BASE_SEED, L2, METRICS, draw_population, observe_labels, with_intercept,
    fit_pu, pu_objective, evaluate, mc_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def auxiliary_objective(theta, p, u, prior, a, target, weight, strength):
    loss, gradient = pu_objective(theta, p, u, prior)
    z = a @ theta
    normalized = weight / weight.sum()
    loss += strength * np.dot(normalized, np.logaddexp(0, z)-target*z)
    gradient += strength * (a.T @ (normalized*(expit(z)-target)))
    return loss, gradient


def fit_augmented(initial, p, u, prior, a, target, weight, strength):
    result = minimize(auxiliary_objective, initial, args=(p, u, prior, a, target, weight, strength),
                      jac=True, method="L-BFGS-B",
                      options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7})
    return result.x, bool(result.success)


def crossfit_scores(x, positives, u, prior):
    scores = np.empty(len(x))
    failures = 0
    for train, held in StratifiedKFold(3, shuffle=True, random_state=BASE_SEED).split(x, positives):
        # Candidate rows are not used in their own teacher fit. U is independent.
        theta, diagnostics = fit_pu(with_intercept(x[train][positives[train]]), u, prior)
        failures += int(not diagnostics["converged"])
        scores[held] = expit(with_intercept(x[held]) @ theta)
    return scores, failures


def build_augmentations(x, positives, scores, rng):
    px = x[positives]
    count = 4 * len(px)
    assert len(px) >= 6
    # Same number of anchors and interpolation draws in each synthetic method.
    anchors = np.arange(count) % len(px)
    rng.shuffle(anchors)
    fraction = rng.random(count)
    nn_p = NearestNeighbors(n_neighbors=6).fit(px).kneighbors(px, return_distance=False)[:, 1:]
    p_neighbor = nn_p[anchors, rng.integers(0, 5, count)]
    smote = (1-fraction[:, None])*px[anchors] + fraction[:, None]*px[p_neighbor]
    # A fixed top 1% of non-P candidates: score is not a hard positive label.
    candidates = np.flatnonzero(~positives)
    selected = candidates[np.argsort(scores[candidates])[-max(5, int(.01*len(candidates))):]]
    neighbors = NearestNeighbors(n_neighbors=5).fit(x[selected]).kneighbors(px, return_distance=False)
    neighbor_scores = np.maximum(scores[selected[neighbors]], 1e-12)
    probabilities = neighbor_scores / neighbor_scores.sum(axis=1, keepdims=True)
    chosen = np.array([rng.choice(5, p=probabilities[a]) for a in anchors])
    partner_ids = selected[neighbors[anchors, chosen]]
    confidence = scores[partner_ids]
    guided = (1-fraction[:, None])*px[anchors] + fraction[:, None]*x[partner_ids]
    target = (1-fraction) + fraction*confidence
    # Exact real-endpoint control: same pairs, fractions, targets and weight mass.
    endpoint_x = np.concatenate([px[anchors], x[partner_ids]])
    endpoint_target = np.r_[np.ones(count), confidence]
    endpoint_weight = np.r_[(1-fraction)*confidence, fraction*confidence]
    ones = np.ones(count)
    augmentation = {
        "random_positive": (px[anchors], ones, ones),
        "smote_positive": (smote, ones, ones),
        "pu_guided_real_endpoints": (endpoint_x, endpoint_target, endpoint_weight),
        "pu_guided_synthetic_uniform": (guided, target, ones),
        "pu_guided_synthetic_weighted": (guided, target, confidence),
    }
    assert np.all((target >= 0) & (target <= 1))
    assert np.isclose(endpoint_weight.sum(), confidence.sum())
    return augmentation, selected, partner_ids, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--prior", type=float, default=.001)
    parser.add_argument("--n-train", type=int, default=150000)
    parser.add_argument("--n-u", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=150000)
    parser.add_argument("--strength-factors", type=float, nargs="+", default=[.25, 1, 4])
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/results/phase3/pu_guided_oversampling")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    records, diagnostics = [], []
    rng = np.random.default_rng(1234)
    p, u = with_intercept(rng.normal(size=(20, 5))), with_intercept(rng.normal(size=(80, 5)))
    a = with_intercept(rng.normal(size=(12, 5)))
    target, weight = rng.random(12), rng.random(12)+.01
    point = np.r_[-4., rng.normal(size=5)*.1]
    objective_args = (p, u, .001, a, target, weight, .004)
    error = check_grad(lambda t: auxiliary_objective(t, *objective_args)[0],
                       lambda t: auxiliary_objective(t, *objective_args)[1], point)
    assert error < 2e-6, error
    with threadpool_limits(limits=1):
        for scenario in ("scar_clean", "sar_clean", "scar_contaminated_positive"):
            for repetition in range(args.repetitions):
                seed = BASE_SEED + repetition
                rng = np.random.default_rng(seed)
                x, latent = draw_population(rng, args.n_train, args.prior)
                xu, _ = draw_population(rng, args.n_u, args.prior)
                xt, yt = draw_population(rng, args.n_test, args.prior)
                observed, waze = observe_labels(x, latent, rng.random(len(x)), rng.random(len(x)),
                                               "scar_waze_fp" if scenario.endswith("positive") else scenario)
                positives = (observed | waze) if scenario.endswith("positive") else observed
                u, p = with_intercept(xu), with_intercept(x[positives])
                teacher, teacher_diag = fit_pu(p, u, args.prior)
                scores, teacher_failures = crossfit_scores(x, positives, u, args.prior)
                augmentations, selected, partners, confidence = build_augmentations(x, positives, scores, rng)
                diagnostics.append({"seed": seed, "scenario": scenario, "positive_count": int(positives.sum()),
                                    "observed_positive_purity": float(latent[positives].mean()),
                                    "candidate_purity": float(latent[selected].mean()),
                                    "chosen_neighbor_purity": float(latent[partners].mean()),
                                    "mean_neighbor_score": float(confidence.mean()),
                                    "generated_count": len(partners),
                                    "teacher_failed_fits": teacher_failures + int(not teacher_diag["converged"])})
                records.append({"seed": seed, "scenario": scenario, "method": "pu_base", "strength_factor": 0.,
                                "converged": teacher_diag["converged"], **evaluate(yt, expit(with_intercept(xt)@teacher))})
                for factor in args.strength_factors:
                    for method, (augmented, targets, weights) in augmentations.items():
                        theta, converged = fit_augmented(teacher, p, u, args.prior, with_intercept(augmented),
                                                       targets, weights, args.prior*factor)
                        records.append({"seed": seed, "scenario": scenario, "method": method,
                                        "strength_factor": factor, "converged": converged,
                                        **evaluate(yt, expit(with_intercept(xt)@theta))})
                # Checkpoint completed seeds; no difficult seeds are omitted.
                pd.DataFrame(records).to_csv(args.out / "per_seed_metrics.csv", index=False)
                (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            print(f"Completed {scenario}: {args.repetitions} seeds; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    frame = pd.DataFrame(records)
    assert not frame.duplicated(["seed", "scenario", "method", "strength_factor"]).any()
    assert np.isfinite(frame[list(METRICS)].to_numpy()).all()
    cells = frame.groupby(["scenario", "strength_factor", "method"])
    assert cells.size().eq(args.repetitions).all()
    summary = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "gradient_check_error": float(error), "elapsed_seconds": time.perf_counter()-start,
               "failed_final_fits": int((~frame.converged).sum()),
               "failed_teacher_fits": sum(d["teacher_failed_fits"] for d in diagnostics),
               "final_fits": len(frame), "metrics": {}, "paired_differences": {},
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), Path(__file__).with_name("phase3_pu_simulation.py")]}}
    for (scenario, factor, method), rows in cells:
        summary["metrics"][f"{scenario}__{factor:g}__{method}"] = {m: mc_summary(rows[m]) for m in METRICS}
    contrasts = [("smote_positive", "random_positive"),
                 ("pu_guided_synthetic_weighted", "smote_positive"),
                 ("pu_guided_synthetic_weighted", "pu_guided_real_endpoints"),
                 ("pu_guided_synthetic_weighted", "pu_guided_synthetic_uniform")]
    for scenario in frame.scenario.unique():
        base = frame[(frame.scenario == scenario) & (frame.method == "pu_base")].set_index("seed")
        for factor in args.strength_factors:
            subset = frame[(frame.scenario == scenario) & (frame.strength_factor == factor)].set_index(["method", "seed"])
            for first, second in contrasts + [("pu_guided_synthetic_weighted", "pu_base")]:
                reference = base if second == "pu_base" else subset.loc[second]
                delta = subset.loc[first, list(METRICS)] - reference[list(METRICS)]
                summary["paired_differences"][f"{scenario}__{factor:g}__{first}_minus_{second}"] = {m: mc_summary(delta[m]) for m in METRICS}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# PU-confidence-guided oversampling: known-truth experiment", "",
             "This experiment lets PU scores drive candidate selection, the number of neighbour draws and the weights of synthetic samples. It uses three-fold cross-fitted scores, top-1% candidates, 5 neighbours and a generation budget of 4 per observed positive. Candidate selection and neighbour rules were not tuned on the latent truth.",
             "", "The original P, the independent marginal U and the true prior are unchanged; synthetic samples enter only an additional auxiliary loss and replace neither P nor U. All auxiliary losses are normalised to the same total mass, with strengths prior × {0.25, 1, 4}. Generation weights use the neighbour score and the soft target is (1−λ)+λq. This interpolated label is a heuristic assumption, not a proven posterior probability, and the auxiliary objective does not inherit the risk-consistency guarantee of PU.",
             "", "Random positives and plain SMOTE use the same number of generated samples. The real-endpoint control uses the same anchors, neighbours, interpolation ratios, soft labels and weight mass, to test whether synthetic geometry is more useful than weighting the endpoints. The uniform-weight synthetic control tests the confidence weights.",
             "", "Three mechanisms: SCAR, SAR and contaminated positives; the contamination mechanism wrongly uses the Waze union including synthetic false positives as P, as a stress test. Training features are continuous Gaussian variables only, without real road or spatio-temporal constraints. Confidence scores are not crash probabilities validated on real data. All results are simulations and do not represent real traffic prediction performance.",
             "", "| Mechanism / auxiliary strength / contrast | ΔROC-AUC [95% MC CI] | ΔAP [95% MC CI] |",
             "|---|---:|---:|"]
    for key, values in summary["paired_differences"].items():
        formatted = []
        for metric in ("roc_auc", "average_precision"):
            v = values[metric]
            ci = v["mc_mean_ci95"]
            formatted.append(f"{v['mean']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]")
        lines.append(f"| {key} | {' | '.join(formatted)} |")
    lines += ["", f"{len(frame)} final-model fits, {summary['failed_final_fits']} failed; {summary['failed_teacher_fits']} teacher failures. Gradient error {error:.3g}. Intervals are Monte Carlo intervals of the mean across seeds, not corrected for multiple comparisons; they must not be used to pick the best strength and then claim independent validation.",
              "", "The truth is used only for evaluation and the candidate-purity audit. The prior is known in the simulation, while the prior and selection mechanism of the real data still need support; the method is a prototype of PU-assisted data augmentation and cannot be claimed to improve the theoretical PU estimator.",
              "", "Reproduction: `python analysis/phase3_pu_guided_oversampling.py --repetitions 30`."]
    (args.out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "final_fits": len(frame), "failed_fits": summary["failed_final_fits"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
