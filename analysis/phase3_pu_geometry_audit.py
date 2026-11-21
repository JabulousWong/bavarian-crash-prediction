"""Paired known-truth geometry and label-mass audit; no truth-based selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from phase3_pu_simulation import BASE_SEED, mc_summary, with_intercept
from phase3_pu_guided_oversampling import crossfit_scores


def features(x, nonlinear):
    return np.column_stack((x, x*x)) if nonlinear else x


def population(rng, n, prior, separation):
    y = rng.random(n) < prior
    cluster = rng.choice([-1, 1], n)
    x = rng.normal(size=(n, 2)) * np.where(y[:, None], .65, 1.)
    x[:, 0] += y * cluster * separation
    return x, y, np.where(y, cluster, 0)


def posterior(x, prior, separation):
    positive = []
    for sign in (-1, 1):
        distance = x - np.array([sign*separation, 0.])
        positive.append(-np.sum(distance*distance, axis=1)/(2*.65**2)-2*np.log(.65)-np.log(2))
    lp = np.log(prior) + logsumexp(positive, axis=0)
    ln = np.log1p(-prior) - np.sum(x*x, axis=1)/2
    return expit(lp-ln)


def run(args):
    rows, failures = [], 0
    with threadpool_limits(limits=1):
        for separation in (1., 2.5):
            for contamination in (0., .3):
                for nonlinear in (False, True):
                    for repetition in range(args.repetitions):
                        seed = BASE_SEED + repetition
                        rng = np.random.default_rng(seed)
                        x, y, cluster = population(rng, args.n_train, args.prior, separation)
                        xu, _, _ = population(rng, args.n_train, args.prior, separation)
                        observed = y & (rng.random(len(x)) < .3)
                        false_ids = rng.choice(np.flatnonzero(~y), int(observed.sum()*contamination/(1-contamination)), replace=False)
                        observed[false_ids] = True
                        scores, failed = crossfit_scores(features(x, nonlinear), observed, with_intercept(features(xu, nonlinear)), args.prior)
                        failures += failed
                        pids = np.flatnonzero(observed)
                        assert len(pids) >= 6
                        anchors = np.tile(pids, 4)
                        fractions = rng.random(len(anchors))
                        pool = np.flatnonzero(~observed)
                        selected = pool[np.argsort(scores[pool])[-max(5, int(.01*len(pool))):]]
                        pn = NearestNeighbors(n_neighbors=6).fit(x[pids]).kneighbors(x[pids], return_distance=False)[:, 1:]
                        gn = NearestNeighbors(n_neighbors=5).fit(x[selected]).kneighbors(x[pids], return_distance=False)
                        probabilities = np.maximum(scores[selected[gn]], 1e-12)
                        probabilities /= probabilities.sum(axis=1, keepdims=True)
                        pick = np.array([rng.choice(5, p=v) for v in np.tile(probabilities, (4, 1))])
                        partners = selected[np.tile(gn, (4, 1))[np.arange(len(anchors)), pick]]
                        smote_partners = pids[np.tile(pn, (4, 1))[np.arange(len(anchors)), rng.integers(0, 5, len(anchors))]]
                        for method in ('smote', 'smote_soft_weighted', 'pu_guided', 'pu_guided_anchor_weight'):
                            b = smote_partners if method.startswith('smote') else partners
                            q = scores[b]
                            target = np.ones(len(b)) if method == 'smote' else 1-fractions+fractions*q
                            weight = np.ones(len(b)) if method == 'smote' else np.maximum(q, 1e-12)
                            if method == 'pu_guided_anchor_weight':
                                weight = weight * np.maximum(scores[anchors], 1e-12)
                            weight /= weight.sum()
                            synthetic = (1-fractions[:, None])*x[anchors]+fractions[:, None]*x[b]
                            truth = posterior(synthetic, args.prior, separation)
                            both_positive = y[anchors] & y[b]
                            cross = both_positive & (cluster[anchors] != cluster[b])
                            rows.append(dict(seed=seed, separation=separation, contamination=contamination,
                                             teacher='quadratic' if nonlinear else 'linear', method=method,
                                             cross_cluster_rate=float(cross.mean()),
                                             cross_cluster_given_positive_pair=float(cross.sum()/max(1, both_positive.sum())),
                                             negative_endpoint_rate=float((~y[b]).mean()),
                                             weighted_false_anchor_mass=float(weight @ (~y[anchors])),
                                             low_posterior_mass=float(weight @ (truth < .5)),
                                             target_overstatement=float(weight @ np.maximum(target-truth, 0)),
                                             positive_label_error_mass=float(weight @ (target*(1-truth))),
                                             generated_count=len(b), positive_count=len(pids)))
                    print(f'Completed separation={separation}, contamination={contamination}, quadratic={nonlinear}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out/'per_seed_geometry.csv', index=False)
    metrics = ['cross_cluster_rate', 'cross_cluster_given_positive_pair', 'negative_endpoint_rate',
               'weighted_false_anchor_mass', 'low_posterior_mass', 'target_overstatement', 'positive_label_error_mass']
    assert np.isfinite(frame[metrics].to_numpy()).all()
    assert frame.groupby(['separation', 'contamination', 'teacher', 'method']).size().eq(args.repetitions).all()
    summary = dict(configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   failed_teacher_fits=failures, metrics={}, paired_differences={},
                   code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), Path(__file__).with_name('phase3_pu_guided_oversampling.py'),
                                 Path(__file__).with_name('phase3_pu_simulation.py')]})
    lines = ['# PU guidance: direct audit of cross-cluster interpolation and label inflation', '',
             'Two Gaussian positive clusters and central Gaussian negatives; prior fixed at 0.02, cluster separation 1 / 2.5, positive contamination 0 / 30%. This is a geometric stress test, not a validation of rare-crash performance.', '',
             'Three-fold cross-fitted teachers with linear and with squared features. Neighbours are always selected in the original 2-D space. Top-1% non-P candidates, 5 neighbours and a budget of four per P are fixed; all methods share anchors and interpolation ratios. The truth and the analytic posterior are used for the audit only.', '',
             'PU guidance keeps the original soft labels and neighbour confidence weights; anchor_weight additionally multiplies by the anchor score and only checks a weight ablation without changing the generation geometry. smote_soft_weighted keeps the SMOTE pairing and geometry and only uses PU soft labels and partner weights of the same form. The weights of every method are normalised to the same total mass.', '',
             'The cross-cluster rate only counts pairs whose two true-positive endpoints come from different generating clusters; negative endpoints are listed separately so that negatives are not assigned to a positive cluster. Label inflation is the weighted max(target − true posterior, 0); false positive label mass is the weighted target × (1 − true posterior), an expected error mass rather than a count of newly added errors.', '',
             '| Distance / contamination / teacher / method (vs SMOTE) | Δ cross-cluster rate [95% MC CI] | Δ label inflation [95% MC CI] | Δ false positive label mass [95% MC CI] |',
             '|---|---:|---:|---:|']
    for cell, group in frame.groupby(['separation', 'contamination', 'teacher']):
        base = group[group.method == 'smote'].set_index('seed')
        for method, values in group.groupby('method'):
            key = '__'.join(map(str, (*cell, method)))
            summary['metrics'][key] = {m: mc_summary(values[m]) for m in metrics}
            if method == 'smote':
                continue
            delta = values.set_index('seed')[metrics]-base[metrics]
            result = {m: mc_summary(delta[m]) for m in metrics}
            summary['paired_differences'][key] = result
            formatted = []
            for m in ('cross_cluster_rate', 'target_overstatement', 'positive_label_error_mass'):
                v = result[m]
                ci = v['mc_mean_ci95']
                formatted.append(f"{v['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else f"{v['mean']:+.4f}")
            lines.append(f"| {key} | {' | '.join(formatted)} |")
        soft = group[group.method == 'smote_soft_weighted'].set_index('seed')
        for method in ('pu_guided', 'pu_guided_anchor_weight'):
            delta = group[group.method == method].set_index('seed')[metrics]-soft[metrics]
            summary['paired_differences']['__'.join(map(str, (*cell, method)))+'_minus_smote_soft_weighted'] = {m: mc_summary(delta[m]) for m in metrics}
    lines += ['', f'Failed teacher fits: {failures}; all seeds kept. Intervals are Monte Carlo intervals of the paired seed means, without multiple-comparison correction.', '',
              'This audit does not train the final predictor on the augmented data, so it can only show whether the generation mechanism reduces geometric/label risk, not that downstream prediction improves. Contaminated P violates the standard clean-positive PU assumption, and the anchor score does not necessarily identify false positives. With very low cross-cluster rates there is a floor effect, which is not enough to claim that PU guidance reduces cross-cluster risk.', '',
              'Reproduction: python analysis/phase3_pu_geometry_audit.py --repetitions 30']
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--n-train', type=int, default=30000)
    parser.add_argument('--prior', type=float, default=.02)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results/phase3/pu_geometry_audit')
    run(parser.parse_args())

