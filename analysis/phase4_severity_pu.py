"""Severity PU benchmarks: real labeled injury crashes and known-truth accidents."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from phase3_pu_geometry_audit import population, features
from phase3_pu_guided_oversampling import build_augmentations, crossfit_scores, fit_augmented
from phase3_pu_simulation import BASE_SEED, L2, METRICS, evaluate, fit_pu, mc_summary, pu_parts, with_intercept

ROOT = Path(__file__).resolve().parents[1]
TRAFFIC = [f'{side}_{name}' for side in ('from', 'to') for name in
           ('n_kfz', 'n_lkw', 'v_kfz', 'v_lkw', 'occupancy_mean', 'net_gap_mean', 'speed_sd_mean')]


def robust_augmented(initial, p, u, prior, a, target, weight, strength):
    theta, converged = fit_augmented(initial, p, u, prior, a, target, weight, strength)
    if converged:
        return theta, True, False
    # Exact epigraph resolves the nnPU max() kink without smoothing the objective.
    normalized = weight/weight.sum()
    def constraints(v):
        positive, negative, _, _ = pu_parts(v[:-1], p, u, prior)
        return np.array([v[-1]-positive, v[-1]-positive-negative])
    def constraint_gradient(v):
        _, _, gp, gn = pu_parts(v[:-1], p, u, prior)
        return np.vstack((np.r_[-gp, 1.], np.r_[-gp-gn, 1.]))
    def objective(v):
        z = a@v[:-1]
        gradient = strength*(a.T@(normalized*(expit(z)-target)))
        gradient[1:] += L2*v[1:-1]
        loss = v[-1]+.5*L2*np.dot(v[1:-1], v[1:-1])+strength*np.dot(normalized, np.logaddexp(0, z)-target*z)
        return loss, np.r_[gradient, 1.]
    positive, negative, _, _ = pu_parts(theta, p, u, prior)
    result = minimize(objective, np.r_[theta, positive+max(0., negative)], jac=True, method='SLSQP',
                      constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_gradient},
                      options={'maxiter': 500, 'ftol': 1e-10})
    feasible = constraints(result.x).min() >= -1e-7
    return result.x[:-1], bool(result.success and feasible), True


def official_cohort(year):
    source = ROOT/'analysis/results/phase1'/f'crash_network_assignment_{year}.csv'
    frame = pd.read_csv(source, dtype={'uid': str})
    frame = frame[frame.nearest_corridor_m.le(100)].copy()
    assert frame.uid.is_unique
    assert frame.UKATEGORIE.isin([1, 2, 3]).all()
    frame['edge_id'] = frame.nearest_edge
    frame['outcome_hour'] = pd.to_datetime(frame.date)+pd.to_timedelta(frame.USTUNDE, unit='h')
    # Existing traffic cache is joined by location and hour, never by severity.
    cache = pd.read_parquet(ROOT/'analysis/results/phase2/datasets'/f'matched_edge_hours_{year}_100m_union.parquet')
    cache = cache[cache.target_union.eq(1)].drop_duplicates(['edge_id', 'outcome_hour'])
    frame = frame.merge(cache[['edge_id', 'outcome_hour', *TRAFFIC]], on=['edge_id', 'outcome_hour'],
                        how='left', validate='many_to_one', indicator=True)
    assert frame['_merge'].eq('both').all()
    frame['severe'] = frame.UKATEGORIE.le(2).astype(int)
    return frame


def augmentations(x, observed, scores, rng):
    result, selected, partners, confidence = build_augmentations(x, observed, scores, rng)
    # build_augmentations generates four copies of every positive, shuffled.
    # Recover the exact anchor endpoint from its endpoint control.
    n = len(partners)
    endpoints, _, _ = result['pu_guided_real_endpoints']
    anchors = endpoints[:n]
    pids = np.flatnonzero(observed)
    # Map exact anchor rows to cross-fit scores (duplicates are resolved to same feature score mean).
    score_map = {}
    for row, score in zip(x[pids], scores[pids]):
        score_map.setdefault(tuple(row), []).append(score)
    anchor_score = np.array([np.mean(score_map[tuple(row)]) for row in anchors])
    synthetic, target, weight = result['pu_guided_synthetic_weighted']
    result['pu_guided_anchor_weight'] = (synthetic, target, weight*np.maximum(anchor_score, 1e-12))
    # Label/weight-only SMOTE control, using scores at synthetic locations is deliberately avoided.
    # Interpolated target and weight from guided method are attached to the same anchor budgets;
    # thus this is a label-mass control, not an exact partner-score ablation.
    result['smote_same_soft_mass'] = (result['smote_positive'][0], target, weight)
    return result, selected, partners


def benchmark(x, y, xu, xt, yt, observed, prior, seed, factors, metadata, records, diagnostics,
              geometry_x=None, feature_transform=None):
    p, u = with_intercept(x[observed]), with_intercept(xu)
    teacher, diag = fit_pu(p, u, prior)
    scores, failed = crossfit_scores(x, observed, u, prior)
    rng = np.random.default_rng(seed+1801)
    augmented, selected, partners = augmentations(x if geometry_x is None else geometry_x, observed, scores, rng)
    if feature_transform is not None:
        augmented = {method: (feature_transform(a), target, weight)
                     for method, (a, target, weight) in augmented.items()}
    diagnostics.append(dict(**metadata, seed=seed, prior_used=prior, positives=int(observed.sum()),
                            positive_purity=float(y[observed].mean()), candidate_purity=float(y[selected].mean()),
                            partner_purity=float(y[partners].mean()), failed_teachers=failed+int(not diag['converged'])))
    records.append(dict(**metadata, seed=seed, method='pu_base', strength_factor=0.,
                        converged=diag['converged'], solver_fallback=False, **evaluate(yt, expit(with_intercept(xt)@teacher))))
    for factor in factors:
        for method, (a, target, weight) in augmented.items():
            theta, converged, fallback = robust_augmented(teacher, p, u, prior, with_intercept(a), target, weight, prior*factor)
            records.append(dict(**metadata, seed=seed, method=method, strength_factor=factor,
                                converged=converged, solver_fallback=fallback, **evaluate(yt, expit(with_intercept(xt)@theta))))


def summarize(frame, cell_columns):
    metrics, paired = {}, {}
    for cell, group in frame.groupby(cell_columns, dropna=False):
        key = '__'.join(map(str, cell if isinstance(cell, tuple) else (cell,)))
        for (factor, method), rows in group.groupby(['strength_factor', 'method']):
            metrics[f'{key}__{factor:g}__{method}'] = {m: mc_summary(rows[m]) for m in METRICS}
        base = group[group.method.eq('pu_base')].set_index('seed')
        for factor, subset in group[group.strength_factor.gt(0)].groupby('strength_factor'):
            for method in ('smote_positive', 'pu_guided_synthetic_weighted', 'pu_guided_anchor_weight',
                           'smote_same_soft_mass', 'pu_guided_real_endpoints'):
                rows = subset[subset.method.eq(method)].set_index('seed')
                paired[f'{key}__{factor:g}__{method}_minus_pu_base'] = {m: mc_summary(rows[m]-base[m]) for m in METRICS}
            guided = subset[subset.method.eq('pu_guided_synthetic_weighted')].set_index('seed')
            for reference in ('smote_positive', 'smote_same_soft_mass', 'pu_guided_real_endpoints', 'pu_guided_synthetic_uniform'):
                other = subset[subset.method.eq(reference)].set_index('seed')
                paired[f'{key}__{factor:g}__guided_minus_{reference}'] = {m: mc_summary(guided[m]-other[m]) for m in METRICS}
    return dict(metrics=metrics, paired_differences=paired)


def real_experiment(args):
    train, test = official_cohort(2024), official_cohort(2025)
    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out/'official_severity_cohort_2024.parquet', index=False)
    test.to_parquet(args.out/'official_severity_cohort_2025.parquet', index=False)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(train[TRAFFIC]))
    xt = scaler.transform(imputer.transform(test[TRAFFIC]))
    y, yt = train.severe.to_numpy(), test.severe.to_numpy()
    # Group same edge-hour so shared traffic rows cannot occur in both P pool and marginal U.
    group_table = train.groupby(['edge_id', 'outcome_hour']).severe.max().reset_index()
    group_ids = pd.MultiIndex.from_frame(train[['edge_id', 'outcome_hour']])
    records, diagnostics, supervised = [], [], {}
    for balanced in (False, True):
        model = LogisticRegression(C=1., class_weight='balanced' if balanced else None, max_iter=2000)
        model.fit(x, y)
        supervised['full_labels_logistic_balanced' if balanced else 'full_labels_logistic'] = evaluate(yt, model.predict_proba(xt)[:, 1])
    hgb = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=7, min_samples_leaf=30,
                                         l2_regularization=1., early_stopping=False, random_state=BASE_SEED)
    hgb.fit(x, y)
    supervised['full_labels_hgb'] = evaluate(yt, hgb.predict_proba(xt)[:, 1])
    for capture in (.3, .7):
        for prior_factor in (.5, 1., 1.5):
            for repetition in range(args.repetitions):
                seed = BASE_SEED+repetition
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=seed)
                a, b = next(splitter.split(group_table, group_table.severe))
                keys = pd.MultiIndex.from_frame(group_table.iloc[a][['edge_id', 'outcome_hour']])
                pool = group_ids.isin(keys)
                px, py, ux = x[pool], y[pool], x[~pool]
                assert not set(group_ids[pool]) & set(group_ids[~pool])
                rng = np.random.default_rng(seed+71)
                observed = py.astype(bool) & (rng.random(len(py)) < capture)
                assert observed.sum() >= 6
                prior = float(y.mean()*prior_factor)
                metadata = dict(task='real_severe_vs_slight', capture=capture, prior_factor=prior_factor)
                benchmark(px, py, ux, xt, yt, observed, prior, seed, [1.], metadata, records, diagnostics)
                naive = LogisticRegression(class_weight='balanced', max_iter=2000).fit(px, observed.astype(int))
                records.append(dict(**metadata, seed=seed, method='unknown_as_negative_balanced', strength_factor=0.,
                                    converged=bool(naive.n_iter_[0] < 2000), solver_fallback=False, **evaluate(yt, naive.predict_proba(xt)[:, 1])))
            print(f'Real severity complete: capture={capture}, prior_factor={prior_factor}', flush=True)
            pd.DataFrame(records).to_csv(args.out/'real_per_seed_metrics.csv', index=False)
    frame = pd.DataFrame(records)
    assert frame.groupby(['capture', 'prior_factor', 'method']).size().eq(args.repetitions).all()
    assert np.isfinite(frame[list(METRICS)].to_numpy()).all()
    summary = summarize(frame, ['capture', 'prior_factor'])
    summary.update(cohorts={str(year): dict(accidents=len(f), severe=int(f.severe.sum()),
                                          categories=f.UKATEGORIE.value_counts().to_dict()) for year, f in [(2024, train), (2025, test)]},
                   full_labels_baselines=supervised, features=TRAFFIC,
                   train_severity_prior=float(y.mean()), test_severity_prior=float(yt.mean()),
                   final_fits=len(frame), failed_final_fits=int((~frame.converged).sum()),
                   solver_fallbacks=int(frame.solver_fallback.sum()),
                   failed_teacher_fits=sum(d['failed_teachers'] for d in diagnostics), diagnostics=diagnostics)
    (args.out/'real_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def simulation(args):
    records, diagnostics = [], []
    counts = pd.read_csv(ROOT/'analysis/results/phase3/class_prior/actual_official_counts.csv')
    motorway = counts[counts.year.eq(2024) & counts.road_scope.eq('motorways')].set_index('category')
    reference_prior = float(motorway.loc['injury', 'count']/motorway.loc['total', 'count'])
    for prior in (reference_prior, .3):
        for mechanism in ('scar', 'sar'):
            for repetition in range(args.repetitions):
                seed = BASE_SEED+repetition
                rng = np.random.default_rng(seed)
                raw, y, _ = population(rng, args.n_train, prior, 2.5)
                raw_u, _, _ = population(rng, args.n_train, prior, 2.5)
                raw_t, yt, _ = population(rng, args.n_test, prior, 2.5)
                # All rows represent accidents, Y=1 means injury; no non-accident controls.
                x, xu, xt = [features(v, True) for v in (raw, raw_u, raw_t)]
                scaler = StandardScaler().fit(x)
                x, xu, xt = [scaler.transform(v) for v in (x, xu, xt)]
                capture = .3 if mechanism == 'scar' else expit(-1+1.2*raw[:, 0])
                observed = y & (rng.random(len(y)) < capture)
                metadata = dict(task='sim_injury_vs_no_injury', prior=prior, mechanism=mechanism)
                benchmark(x, y, xu, xt, yt, observed, prior, seed, [.25, 1., 4.], metadata, records, diagnostics,
                          geometry_x=raw, feature_transform=lambda a: scaler.transform(features(a, True)))
            print(f'Injury simulation complete: prior={prior}, mechanism={mechanism}', flush=True)
            pd.DataFrame(records).to_csv(args.out/'simulation_per_seed_metrics.csv', index=False)
    frame = pd.DataFrame(records)
    assert frame.groupby(['prior', 'mechanism', 'strength_factor', 'method']).size().eq(args.repetitions).all()
    assert np.isfinite(frame[list(METRICS)].to_numpy()).all()
    summary = summarize(frame, ['prior', 'mechanism'])
    summary.update(final_fits=len(frame), failed_final_fits=int((~frame.converged).sum()),
                   solver_fallbacks=int(frame.solver_fallback.sum()),
                   reference_prior=reference_prior,
                   reference_counts=dict(injury=int(motorway.loc['injury', 'count']), total=int(motorway.loc['total', 'count'])),
                   failed_teacher_fits=sum(d['failed_teachers'] for d in diagnostics), diagnostics=diagnostics)
    (args.out/'simulation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def report(args, real, simulated, elapsed):
    lines = ['# Severity PU: validation on real severity and injury simulation', '',
             'The main question is whether a crash causes injury. Property-damage-only crashes are available only as H1100D/H1101C totals, without time or location of individual crashes. The injury status of unmatched Waze records is unknown and there is no real test set of non-injury crashes, so this question is studied by known-truth simulation only. Waze-only records are not used as negatives, and road periods without a crash do not enter the severity cohort.', '',
             'Supplementary task on real data: among official injury crashes, predict serious injury/fatal (UKATEGORIE 1/2) versus slight injury (3). This is a different endpoint from injury versus no injury. Each official crash is one row, kept by uid; the most severe edge-hour label is not used in place of the individual label.', '',
             'Trained on 2024, fixed evaluation on 2025; 2025 was inspected in earlier work, so this remains an exploratory out-of-time validation. Features are 14 continuous traffic features over the full hour before the crash, with median imputation and standardisation fitted on the training year. Waze confidence, report count, source flags, injury category and post-crash traffic are not used as features; crashes in the same edge-hour are grouped when splitting the P candidate pool and the independent U.', '',
             'The real-data PU experiment artificially keeps only 30% / 70% of the serious-crash labels, and slight-injury labels are hidden when training PU. The real data actually have complete severity labels, so the fully supervised model is the practical baseline. The PU prior is the 2024 complete-label serious share × {0.5, 1, 1.5}; the middle value is an oracle benchmark and does not represent a prior that would be identifiable in a real unknown-label task.', '',
             f"In the simulation each row is a crash and the positive class means injury; the reference injury share uses the 2024 official Bavarian motorway totals {simulated['reference_counts']['injury']} / {simulated['reference_counts']['total']} = {simulated['reference_prior']:.6f}, with 0.3 as an additional stress condition. The former only fixes the simulated class share and is not a known prior of the Waze report sample or the local corridor. SCAR and feature-dependent SAR, two positive clusters and central negatives. A PU teacher with squared features can represent both clusters; the truth is used only for positive collection, evaluation and the oracle prior, not for candidate selection. Neighbour search and interpolation happen in the original 2-D space and the squared features are recomputed afterwards, which keeps the feature manifold.", '',
             'All methods keep the original P/U, with a generation budget of four samples per P, three-fold teacher scores, top-1% candidates and 5 neighbours. The additional auxiliary losses have equal mass; the real-data strength is fixed at the prior, the simulated strengths are prior × {0.25, 1, 4}, and no strength is chosen by test performance. Anchor weights are multiplied by the anchor score; the real-endpoint control isolates the effect of interpolation; smote_same_soft_mass attaches the same soft labels and weights to SMOTE as a control for total label mass and is not a strict ablation of the partner score.', '',
             '| Real task: label capture / prior multiple / method | ΔAP vs PU base [95% MC CI] | ΔROC-AUC [95% MC CI] |',
             '|---|---:|---:|']
    def add_rows(summary, prefix):
        for key, values in summary['paired_differences'].items():
            if not key.endswith('_minus_pu_base') or not any(name in key for name in ('smote_positive', 'pu_guided_synthetic_weighted', 'pu_guided_anchor_weight')):
                continue
            formatted = []
            for m in ('average_precision', 'roc_auc'):
                v = values[m]; ci = v['mc_mean_ci95']
                formatted.append(f"{v['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else f"{v['mean']:+.4f}")
            lines.append(f"| {prefix}{key} | {' | '.join(formatted)} |")
    add_rows(real, '')
    lines += ['', '| Simulation: injury share / collection mechanism / strength / method | ΔAP vs PU base [95% MC CI] | ΔROC-AUC [95% MC CI] |', '|---|---:|---:|']
    add_rows(simulated, '')
    lines += ['', '## Fully supervised baseline (real data)', '', '| Model | AP | ROC-AUC | Brier |', '|---|---:|---:|---:|']
    for name, values in real['full_labels_baselines'].items():
        lines.append(f"| {name} | {values['average_precision']:.4f} | {values['roc_auc']:.4f} | {values['brier']:.4f} |")
    lines += ['', f"Real data: {real['final_fits']} fits, {real['failed_final_fits']} failed; simulation: {simulated['final_fits']} fits, {simulated['failed_final_fits']} failed. Teacher failures {real['failed_teacher_fits']} / {simulated['failed_teacher_fits']}. Epigraph solver fallbacks in the auxiliary fits {real['solver_fallbacks']} / {simulated['solver_fallbacks']}. All seeds kept; total time {elapsed:.1f} s.", '',
              'MC intervals reflect variation of the training split / label-hiding seed on a fixed test set; they are not sampling confidence intervals for the road/crash population and are not corrected for multiple comparisons. The real 2025 test crashes are not resampled per seed; quantifying test-set uncertainty would need a bootstrap grouped by crash time/location. Serious injury and fatal share one binary class; fatal crashes alone are too few to be evaluated separately. The road scope is still the 100 m detector corridor, not a verified operated network.', '',
              'Category definitions: [Unfallatlas documentation](https://www.statistikportal.de/de/karten/unfallatlas), [data field documentation](https://www.forschungsdatenzentrum.de/sites/default/files/stvu_ab_2013_on-site_dsb.pdf).', '',
              'Reproduction: `python analysis/phase4_severity_pu.py --repetitions 30`.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-test', type=int, default=20000)
    parser.add_argument('--out', type=Path, default=ROOT/'analysis/results/phase4/severity_pu')
    args = parser.parse_args()
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        real = real_experiment(args)
        simulated = simulation(args)
    report(args, real, simulated, time.perf_counter()-started)
    sources = [Path(__file__), Path(__file__).with_name('phase3_pu_guided_oversampling.py'),
               Path(__file__).with_name('phase3_pu_simulation.py'), Path(__file__).with_name('phase3_pu_geometry_audit.py')]
    for year in (2024, 2025):
        sources.extend([ROOT/'analysis/results/phase1'/f'crash_network_assignment_{year}.csv',
                        ROOT/'analysis/results/phase2/datasets'/f'matched_edge_hours_{year}_100m_union.parquet'])
    sources.append(ROOT/'analysis/results/phase3/class_prior/actual_official_counts.csv')
    manifest = dict(configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    elapsed_seconds=time.perf_counter()-started,
                    input_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(out=str(args.out), elapsed=manifest['elapsed_seconds'])), flush=True)
