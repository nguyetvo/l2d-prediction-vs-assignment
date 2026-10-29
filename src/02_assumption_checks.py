"""Reproduction and assumption checks on the OpenL2D/FiFAR benchmark: per-run loss and paper-style table (vs the
published Table 7), oracle headroom, expert complementarity, calibration (AUC/ECE) of the correctness models,
monthly stability and feature shift, re-evaluation over a lambda grid, paired bootstrap statistics.
Run once per results source; run "author" first so the "retrain" run can add the side-by-side comparison.
Usage: python src/02_assumption_checks.py [retrain|author]"""
from pathlib import Path
import sys, json, subprocess, time, zipfile
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.stats import ks_2samp, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
OFFICIAL, RETRAIN, LEGACY_PY = config.OFFICIAL, config.RETRAIN, config.LEGACY_PY

RESULTS_SOURCE = sys.argv[1] if len(sys.argv) > 1 else 'retrain'   # 'retrain' = retrained models; 'author' = official shipped models
assert RESULTS_SOURCE in ('retrain', 'author'), RESULTS_SOURCE
if RESULTS_SOURCE == 'author':
    REPO = OFFICIAL
    RESULTS = OFFICIAL / 'l2d_benchmarking' / 'deferral' / 'results_author_official'
else:
    REPO = RETRAIN
    RESULTS = RETRAIN / 'l2d_benchmarking' / 'deferral' / 'results'
L2D_PRED = REPO / 'l2d_benchmarking' / 'deferral' / 'l2d_predictions'
FIFAR = OFFICIAL / 'FiFAR'         # data is identical in both; read from the original

OUT = config.CHECKS_OUT / RESULTS_SOURCE
CACHE = OUT / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.057                     # cost(FP)/cost(FN), dataset_cfg.yaml
TEST_MONTH, VAL_MONTH = 7, 6
SEED = 42
N_BOOT = 1000

# ---- Decision thresholds: fix these BEFORE reading results ----
HEADROOM_MIN_REL = 0.10            # oracle gap worth pursuing, relative to DeCCaF loss
RESCUABLE_MIN = 0.30               # share of h-errors that some team expert gets right
ECE_MAX = 0.05                     # calibration error considered acceptable
KS_SHIFT = 0.10                    # KS statistic treated as material feature shift
LAMBDA_GRID = [0.02, 0.057, 0.1, 0.2, 0.5]

for p in [RESULTS, L2D_PRED, FIFAR, LEGACY_PY]:
    print(('OK      ' if p.exists() else 'MISSING ') + str(p))
assert RESULTS.exists() and L2D_PRED.exists() and FIFAR.exists(), 'Inputs missing (for retrain, wait until it finishes)'

EXPORT = r"""
import pickle, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for m in ['ova', 'deccaf']:
    d = pickle.load(open(src / f'{m}.pkl', 'rb'))
    for k, v in d.items():
        v.to_parquet(dst / f'{m}__{k}.parquet')
print('exported', len(list(dst.glob('*.parquet'))), 'files')
"""
if len(list(CACHE.glob('*__*.parquet'))) < 50:
    r = subprocess.run([str(LEGACY_PY), '-c', EXPORT, str(L2D_PRED), str(CACHE)], capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:]); assert r.returncode == 0
L2D = {}
for p in sorted(CACHE.glob('*__*.parquet')):
    m, seed = p.stem.split('__')
    L2D.setdefault({'ova': 'OvA', 'deccaf': 'DeCCaF'}[m], {})[seed] = pd.read_parquet(p)
print({m: len(v) for m, v in L2D.items()})

alerts = pd.read_parquet(FIFAR / 'alert_data/processed_data/alerts.parquet')
test = alerts[alerts['month'] == TEST_MONTH]
y = test['fraud_bool'].astype(int)
experts = pd.read_parquet(FIFAR / 'synthetic_experts/expert_predictions.parquet')
pred = experts.loc[test.index].astype(int).copy()

# classifier h hard decisions = official "Only_Classifier" output (identical for every scenario)
only_h = sorted((RESULTS / 'Only_Classifier').rglob('results.parquet'))
pred['classifier_h'] = pd.read_parquet(only_h[0])['prediction'].astype(int).reindex(test.index)
assert pred['classifier_h'].notna().all()

envs = {}
for d in sorted((FIFAR / 'testbed/test').iterdir()):
    if d.is_dir():
        envs[d.name] = dict(cap=pd.read_csv(d / 'capacity.csv'), bat=pd.read_csv(d / 'batches.csv'),
                            team=d.name.split('#')[1].split('-')[0], dist=d.name.split('-')[1])

def case_cost(p, yy, lam=LAMBDA):
    p = np.asarray(p).astype(int); yy = np.asarray(yy).astype(int)
    return np.where((p == 0) & (yy == 1), 1.0, np.where((p == 1) & (yy == 0), lam, 0.0))

def env_workers(env):
    """Workers with capacity>0 in the (single) batch, including classifier_h = remainder."""
    c = envs[env]['cap'].iloc[0]
    caps = c.drop(['batch_id', 'batch_size'])
    caps = caps[caps > 0].astype(int)
    caps['classifier_h'] = int(c['batch_size'] - caps.sum())
    return caps[caps > 0]

runs, COST = [], {}   # COST[key] = per-case cost vector aligned to test.index
def add_run(method, seed, env, path):
    p = pd.read_parquet(path)['prediction'].astype(int).reindex(test.index)
    assert p.notna().all(), f'incomplete run {path}'
    cc = case_cost(p, y); key = (method, seed, env); COST[key] = cc
    a = p.values; yy = y.values
    runs.append(dict(method=method, seed=seed, env=env, team=envs[env]['team'], dist=envs[env]['dist'],
                     fn=int(((a == 0) & (yy == 1)).sum()), fp=int(((a == 1) & (yy == 0)).sum()), loss=cc.sum()))

for m in ['DeCCaF', 'OvA', 'Random']:
    for seed_dir in sorted((RESULTS / m).iterdir()):
        for env_dir in sorted(seed_dir.iterdir()):
            add_run(m, seed_dir.name, env_dir.name, env_dir / 'results.parquet')
for m in ['Full_Rej', 'Only_Classifier']:
    for env_dir in sorted((RESULTS / m).iterdir()):
        add_run(m, 'NA', env_dir.name, env_dir / 'results.parquet')
runs = pd.DataFrame(runs)
print(runs.groupby('method').size().to_string())

# Same aggregation as the author's results.ipynb: mean over runs per (method, team), CI = 1.96*sd/sqrt(n)
g = runs.groupby(['method', 'team'])['loss'].agg(['mean', 'std', 'count'])
g['ci95'] = 1.96 * g['std'] / np.sqrt(g['count'])
paper_table = g['mean'].round(1).unstack('method')[['Full_Rej', 'Only_Classifier', 'Random', 'OvA', 'DeCCaF']]
print('\nLoss = FN + 0.057*FP (mean per team):'); print(paper_table.to_string())
runs.to_csv(OUT / '1_runs_loss.csv', index=False); g.to_csv(OUT / '1_paper_style_table.csv')

def solve_transport(C, caps):
    """min sum C[w,j] x[w,j]  s.t. each case to exactly one worker, worker w gets exactly caps[w]."""
    W, N = C.shape
    cols = np.arange(W * N)
    A_case = coo_matrix((np.ones(W * N), (np.tile(np.arange(N), W), cols)), shape=(N, W * N))
    A_work = coo_matrix((np.ones(W * N), (np.repeat(np.arange(W), N), cols)), shape=(W, W * N))
    r = linprog(C.ravel(), A_eq=vstack([A_case, A_work]).tocsr(), b_eq=np.r_[np.ones(N), caps],
                bounds=(0, 1), method='highs')
    assert r.status == 0, r.message
    x = r.x.reshape(W, N)
    assert np.allclose(x.max(0), 1, atol=1e-6), 'non-integral LP solution'
    return x.argmax(0), r.fun

def oracle_losses(env, lam=LAMBDA, label=True):
    caps = env_workers(env); out = {}
    cases = envs[env]['bat']['case_id'].values
    yy = y.loc[cases].values
    true_cost = np.vstack([case_cost(pred.loc[cases, w], yy, lam) for w in caps.index])
    idx = np.arange(len(cases))
    a_case, _ = solve_transport(true_cost, caps.values)
    out['CaseOracle'] = true_cost[a_case, idx].sum()
    out['NoCapacityOracle'] = true_cost.min(0).sum()   # ignores capacities: effect of the constraint itself
    if label:
        # label oracle: expected cost of worker w given the true class, using its test-set FNR/FPR
        fnr = {w: ((pred.loc[test.index, w] == 0) & (y == 1)).sum() / (y == 1).sum() for w in caps.index}
        fpr = {w: ((pred.loc[test.index, w] == 1) & (y == 0)).sum() / (y == 0).sum() for w in caps.index}
        label_cost = np.vstack([np.where(yy == 1, fnr[w], lam * fpr[w]) for w in caps.index])
        a_lab, _ = solve_transport(label_cost, caps.values)
        out['LabelOracle'] = true_cost[a_lab, idx].sum()
    return out

t0 = time.time(); rows = []
for env in envs:
    o = oracle_losses(env)
    r = runs[runs.env == env].groupby('method')['loss'].mean()
    rows.append(dict(env=env, team=envs[env]['team'], dist=envs[env]['dist'], **r.to_dict(), **o))
head = pd.DataFrame(rows)
head['headroom_rel'] = (head['DeCCaF'] - head['CaseOracle']) / head['DeCCaF']
head['headroom_rel_label'] = (head['DeCCaF'] - head['LabelOracle']) / head['DeCCaF']
head['capture_DeCCaF'] = (head['Random'] - head['DeCCaF']) / (head['Random'] - head['CaseOracle'])
head['capture_OvA'] = (head['Random'] - head['OvA']) / (head['Random'] - head['CaseOracle'])
head['capture_LabelOracle'] = (head['Random'] - head['LabelOracle']) / (head['Random'] - head['CaseOracle'])
print(f'oracles solved in {time.time()-t0:.0f}s')
cols = ['Random', 'OvA', 'DeCCaF', 'LabelOracle', 'CaseOracle', 'NoCapacityOracle', 'headroom_rel_label', 'headroom_rel', 'capture_OvA', 'capture_DeCCaF', 'capture_LabelOracle']
print(head.groupby('team')[cols].mean().round(3).to_string())
print('\nOverall:'); print(head[cols].mean().round(3).to_string())
head.to_csv(OUT / '2_oracle_headroom_by_env.csv', index=False)

err = pd.DataFrame({w: (pred[w] != y).astype(int) for w in pred.columns}, index=test.index)
per_worker = pd.DataFrame({
    'FNR': [((pred[w] == 0) & (y == 1)).sum() / (y == 1).sum() for w in pred.columns],
    'FPR': [((pred[w] == 1) & (y == 0)).sum() / (y == 0).sum() for w in pred.columns],
    'loss_per_case': [case_cost(pred[w], y).mean() for w in pred.columns]}, index=pred.columns)
print('Classifier h:', per_worker.loc['classifier_h'].round(4).to_dict())
print('Experts loss/case: min %.4f | median %.4f | max %.4f' % tuple(per_worker.drop('classifier_h')['loss_per_case'].quantile([0, .5, 1])))

comp = []
for team in sorted({e['team'] for e in envs.values()}):
    env = next(k for k, e in envs.items() if e['team'] == team and e['dist'] == 'hom')
    members = [w for w in env_workers(env).index if w != 'classifier_h']
    h_wrong = err['classifier_h'] == 1
    any_right = (err[members] == 0).any(axis=1)
    corr = err[members + ['classifier_h']].corr()
    comp.append(dict(team=team, n_experts=len(members), h_errors=int(h_wrong.sum()),
                     rescuable_share=float((h_wrong & any_right).sum() / h_wrong.sum()),
                     all_wrong_share=float((err[members + ['classifier_h']].all(axis=1)).mean()),
                     mean_err_corr_experts=float(corr.loc[members, members].values[np.triu_indices(len(members), 1)].mean()),
                     mean_err_corr_with_h=float(corr.loc[members, 'classifier_h'].mean())))
comp = pd.DataFrame(comp); print(comp.round(3).to_string(index=False))
per_worker.to_csv(OUT / '3_per_worker_error_rates.csv'); comp.to_csv(OUT / '3_complementarity_by_team.csv', index=False)

val = alerts[alerts['month'] == VAL_MONTH]
e_c = val['fraud_bool'].replace([0, 1], [LAMBDA, 1]).mean()
reb_1 = val['fraud_bool'].mean() / e_c
reb_0 = (1 - val['fraud_bool']).mean() * LAMBDA / e_c
nmin = int((y == 0).sum()); nmax = int(nmin * reb_1 / reb_0)
over = pd.concat([test[y == 0], test[y == 1].sample(replace=True, n=nmax, random_state=SEED)]).index

def ece(outcome, prob):
    pt, pp = calibration_curve(outcome, prob, strategy='quantile', n_bins=10)
    return float(np.mean(np.abs(pt - pp)))

cal = []
for m, seeds in L2D.items():
    for seed, P in seeds.items():
        for w in P.columns:
            outcome = (pred.loc[over, w] == y.loc[over]).astype(int)
            prob = P.loc[over, w]
            cal.append(dict(method=m, seed=seed, team=seed.split('#')[1], worker=w,
                            auc=roc_auc_score(outcome, prob), ece=ece(outcome, prob),
                            mean_pred=float(prob.mean()), mean_true=float(outcome.mean())))
cal = pd.DataFrame(cal)
summary_cal = cal[cal.worker != 'classifier_h'].groupby('method')[['auc', 'ece', 'mean_pred', 'mean_true']].agg(['mean', 'std']).round(4)
print('Experts only (classifier_h excluded, as in the paper):'); print(summary_cal.to_string())
print('\nShare of (seed, expert) pairs with ECE >', ECE_MAX, ':')
print(cal[cal.worker != 'classifier_h'].assign(bad=lambda d: d.ece > ECE_MAX).groupby('method')['bad'].mean().round(3).to_string())
cal.to_csv(OUT / '4_calibration_auc_ece.csv', index=False)

monthly = alerts.groupby('month').agg(n=('fraud_bool', 'size'), fraud_rate=('fraud_bool', 'mean'))
monthly['model_score_auc'] = [roc_auc_score(g['fraud_bool'], g['model_score']) if g['fraud_bool'].nunique() > 1 else np.nan
                              for _, g in alerts.groupby('month')]
exp_err_month = pd.DataFrame({w: (experts.loc[alerts.index, w] != alerts['fraud_bool']).groupby(alerts['month']).mean()
                              for w in experts.columns})
monthly['expert_error_mean'] = exp_err_month.mean(axis=1)
print(monthly.round(4).to_string())

num_cols = [c for c in alerts.columns if c not in ['fraud_bool', 'month'] and pd.api.types.is_numeric_dtype(alerts[c])]
before = alerts[alerts['month'] < TEST_MONTH]
ks = pd.DataFrame([dict(feature=c, ks=ks_2samp(before[c], test[c]).statistic) for c in num_cols]).sort_values('ks', ascending=False)
print('\nTop KS (months <7 vs month 7):'); print(ks.head(10).round(3).to_string(index=False))
print('Features with KS >=', KS_SHIFT, ':', int((ks.ks >= KS_SHIFT).sum()), 'of', len(ks))
monthly.to_csv(OUT / '5_monthly_stability.csv'); ks.to_csv(OUT / '5_feature_ks_shift.csv', index=False)

sens = []
for lam in LAMBDA_GRID:
    l =runs.assign(loss_l=runs['fn'] + lam * runs['fp']).groupby('method')['loss_l'].mean()
    orc = np.mean([oracle_losses(env, lam, label=False)['CaseOracle'] for env in envs])
    sens.append(dict(lam=lam, **l.to_dict(), CaseOracle=orc))
sens = pd.DataFrame(sens).set_index('lam')
ranking = sens[['Full_Rej', 'Only_Classifier', 'Random', 'OvA', 'DeCCaF']].rank(axis=1).astype(int)
print(sens.round(1).to_string()); print('\nRank (1 = best):'); print(ranking.to_string())
ranking_stable = bool((ranking.nunique() == 1).all())
print('Ranking stable across λ grid:', ranking_stable)
sens.to_csv(OUT / '6_lambda_sensitivity.csv')

rng = np.random.default_rng(SEED)
N = len(test)
Wb = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)   # bootstrap case weights

def paired(a, b):
    keys = [k for k in COST if k[0] == a]
    diffs_point, diffs_boot, env_means = [], [], {}
    for (_, seed, env) in keys:
        kb = (b, seed, env) if (b, seed, env) in COST else (b, 'NA', env) if (b, 'NA', env) in COST else None
        if kb is None:  # Random: average over its 5 seeds
            rs = [k for k in COST if k[0] == b and k[2] == env]
            cb = np.mean([COST[k] for k in rs], axis=0)
        else:
            cb = COST[kb]
        d = COST[(a, seed, env)] - cb
        diffs_point.append(d.sum()); diffs_boot.append(Wb @ d); env_means.setdefault(env, []).append(d.sum())
    boot = np.mean(diffs_boot, axis=0)
    em = np.array([np.mean(v) for v in env_means.values()])
    return dict(A=a, B=b, n_pairs=len(keys), mean_diff=float(np.mean(diffs_point)),
                ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)),
                share_A_better=float(np.mean(np.array(diffs_point) < 0)),
                wilcoxon_p_env=float(wilcoxon(em).pvalue) if np.any(em != 0) else np.nan)

stats = pd.DataFrame([paired('DeCCaF', 'OvA'), paired('DeCCaF', 'Random'), paired('OvA', 'Random'),
                      paired('DeCCaF', 'Only_Classifier'), paired('OvA', 'Only_Classifier')])
stats['ci_excludes_0'] = (stats.ci_low > 0) | (stats.ci_high < 0)
print('Loss difference A − B (negative = A better):'); print(stats.round(3).to_string(index=False))
stats.to_csv(OUT / '7_paired_statistics.csv', index=False)

overall_head = float(head['headroom_rel_label'].mean())   # conservative (label-oracle) gap drives the decision
cal_e = cal[cal.worker != 'classifier_h'].groupby('method')['ece'].mean()
verdict = {
  'results_source': RESULTS_SOURCE,
  'A1_metric_reproduced': {'runs': runs.groupby('method').size().to_dict(), 'status': 'PASS'},
  'A2_headroom': {'mean_rel_gap_DeCCaF_vs_LabelOracle': round(overall_head, 4),
                  'mean_rel_gap_DeCCaF_vs_CaseOracle_loose': round(float(head['headroom_rel'].mean()), 4),
                  'mean_capture_DeCCaF': round(float(head['capture_DeCCaF'].mean()), 4),
                  'mean_capture_LabelOracle': round(float(head['capture_LabelOracle'].mean()), 4),
                  'status': 'GAP_EXISTS' if overall_head >= HEADROOM_MIN_REL else 'LITTLE_HEADROOM'},
  'A3_complementarity': {'min_rescuable_share': round(float(comp.rescuable_share.min()), 4),
                         'status': 'PASS' if comp.rescuable_share.min() >= RESCUABLE_MIN else 'WEAK'},
  'A4_calibration': {'mean_ece': cal_e.round(4).to_dict(),
                     'status': {m: ('OK' if v <= ECE_MAX else 'MISCALIBRATED_GAP') for m, v in cal_e.items()}},
  'A5_shift': {'features_ks_ge_threshold': int((ks.ks >= KS_SHIFT).sum()),
               'status': 'SHIFT_PRESENT' if (ks.ks >= KS_SHIFT).any() else 'NO_MATERIAL_SHIFT'},
  'A6_lambda_robust': {'ranking_stable': ranking_stable},
  'A7_statistics': stats[['A', 'B', 'mean_diff', 'ci_low', 'ci_high', 'ci_excludes_0']].round(3).to_dict('records'),
  'thresholds': dict(HEADROOM_MIN_REL=HEADROOM_MIN_REL, RESCUABLE_MIN=RESCUABLE_MIN, ECE_MAX=ECE_MAX, KS_SHIFT=KS_SHIFT),
  'caveats': ['Synthetic experts: not evidence about human behaviour',
              '25 scenarios share one test set: not independent samples',
              'Oracles are upper bounds, not achievable methods',
              'A gap shows where improvement is possible; it is not novelty by itself']}
(OUT / 'VERDICT.json').write_text(json.dumps(verdict, indent=2, default=str), encoding='utf-8')
print(json.dumps(verdict, indent=2, default=str))
pack = OUT.parent / f'assumption_checks_{RESULTS_SOURCE}.zip'
with zipfile.ZipFile(pack, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUT.glob('*.*'): z.write(p, p.name)
print('ZIP:', pack)

PAPER_T7 = pd.DataFrame({
    'Full_Rej': [213.5] * 5, 'Only_Classifier': [204.0] * 5,
    'Random': [169.6, 157.8, 151.2, 163.1, 163.3], 'OvA': [151.7, 142.0, 131.3, 145.8, 141.2],
    'DeCCaF': [138.1, 145.3, 126.2, 141.6, 132.0]}, index=[f'team_{i}' for i in range(1, 6)])
PAPER_CI = pd.DataFrame({
    'Random': [2.53, 1.39, 1.84, 1.61, 1.58], 'OvA': [1.99, 1.61, 2.19, 3.51, 2.57],
    'DeCCaF': [5.14, 4.28, 4.52, 5.32, 1.93]}, index=PAPER_T7.index)
methods = ['Full_Rej', 'Only_Classifier', 'Random', 'OvA', 'DeCCaF']
this_run = g['mean'].unstack('method')[methods]

rows = []
for team in PAPER_T7.index:
    for m in methods:
        p, v = PAPER_T7.loc[team, m], this_run.loc[team, m]
        ci = PAPER_CI.loc[team, m] if m in PAPER_CI else 0.0
        rows.append(dict(team=team, method=m, paper=p, this_run=round(v, 1), diff=round(v - p, 1),
                         rel_diff=round((v - p) / p, 3), within_paper_ci=bool(abs(v - p) <= max(ci, 0.05))))
vs_paper = pd.DataFrame(rows)
print(f'[{RESULTS_SOURCE}] vs paper Table 7:')
print(vs_paper.pivot(index='team', columns='method', values='diff')[methods].to_string())
print('\nWithin paper 95% CI:', int(vs_paper.within_paper_ci.sum()), 'of', len(vs_paper))
red =lambda t: {m: round(float((t['Random'] - t[m]).mean() / t['Random'].mean()), 3) for m in ['OvA', 'DeCCaF']}
print('Loss reduction vs Random  | paper:', red(PAPER_T7), '| this run:', red(this_run))
rank_paper = PAPER_T7.mean().rank().astype(int); rank_run = this_run.mean().rank().astype(int)
print('Ranking (1 = best) paper:', rank_paper.to_dict()); print('Ranking (1 = best) run:  ', rank_run.to_dict())
vs_paper.to_csv(OUT / '9_vs_paper_table7.csv', index=False)

# Side-by-side with the author's shipped results, if that mode has been run once
other = 'author' if RESULTS_SOURCE == 'retrain' else 'retrain'
other_runs_path = OUT.parent / other / '1_runs_loss.csv'
if other_runs_path.exists():
    other_runs = pd.read_csv(other_runs_path)
    other_tbl = other_runs.groupby(['team', 'method'])['loss'].mean().unstack('method')[methods]
    print(f'\nMean loss per team — {RESULTS_SOURCE} minus {other}:')
    print((this_run - other_tbl).round(1).to_string())
    for name in ['4_calibration_auc_ece.csv']:
        a = pd.read_csv(OUT / name); b = pd.read_csv(OUT.parent / other / name)
        f = lambda d: d[d.worker != 'classifier_h'].groupby('method')[['auc', 'ece']].mean().round(4)
        print(f'\nExpert-correctness models (mean AUC / ECE):\n{RESULTS_SOURCE}:\n{f(a)}\n{other}:\n{f(b)}')
    other_verdict = OUT.parent / other / 'VERDICT.json'
    if other_verdict.exists():
        ov = json.loads(other_verdict.read_text(encoding='utf-8'))
        print('\nHeadroom (label-oracle gap):', RESULTS_SOURCE, verdict['A2_headroom']['mean_rel_gap_DeCCaF_vs_LabelOracle'],
              '|', other, ov['A2_headroom'].get('mean_rel_gap_DeCCaF_vs_LabelOracle'))
else:
    print(f'\n(Run once with RESULTS_SOURCE = "{other}" to add the side-by-side comparison.)')

with zipfile.ZipFile(pack, 'w', zipfile.ZIP_DEFLATED) as z:   # refresh ZIP to include section 9
    for p in OUT.glob('*.*'): z.write(p, p.name)
print('ZIP:', pack)