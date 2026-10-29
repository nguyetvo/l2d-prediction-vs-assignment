"""Gap decomposition (v2): v1 levels plus case-weighted objectives (L0-cw, L1-cw), an equal-size month-6 model (L2eq)
and within-case discrimination of the correctness estimates. Shares the prediction cache with 03_gap_decomposition.py.
Usage: python src/04_gap_decomposition_v2.py [retrain|author]"""
from pathlib import Path
import sys, json, subprocess, time, zipfile, warnings
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from joblib import Parallel, delayed
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
OFFICIAL, RETRAIN, LEGACY_PY = config.OFFICIAL, config.RETRAIN, config.LEGACY_PY

RESULTS_SOURCE = sys.argv[1] if len(sys.argv) > 1 else 'retrain'   # 'retrain' = retrained models; 'author' = official shipped models
assert RESULTS_SOURCE in ('retrain', 'author'), RESULTS_SOURCE
REPO = RETRAIN if RESULTS_SOURCE == 'retrain' else OFFICIAL
RESULTS = REPO / 'l2d_benchmarking' / 'deferral' / ('results' if RESULTS_SOURCE == 'retrain' else 'results_author_official')
FIFAR = OFFICIAL / 'FiFAR'
OUT = config.GAP2_OUT / RESULTS_SOURCE; OUT.mkdir(parents=True, exist_ok=True)
CACHE = config.GAP_OUT / RESULTS_SOURCE / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)   # shared with v1

LAMBDA, TEST_MONTH, VAL_MONTH = 0.057, 7, 6
SEED, N_BOOT, N_FOLDS, N_JOBS = 42, 1000, 5, 6
for p in [REPO, RESULTS, FIFAR, LEGACY_PY]:
    print(('OK      ' if p.exists() else 'MISSING ') + str(p))

EXPORT = r"""
import pickle, yaml, os, sys, numpy as np, pandas as pd
from pathlib import Path
repo, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load(open(repo / 'Code/alert_data/dataset_cfg.yaml'))
cat = cfg['categorical_dict']; CAT = cfg['data_cols']['categorical']
cat['assignment'] = yaml.safe_load(open(repo / 'FiFAR/synthetic_experts/expert_ids.yaml'))['human_ids']
def cc(d, fs):
    d = d.copy()
    for f in fs: d[f] = pd.Categorical(d[f].values, categories=cat[f])
    return d
A = pd.read_parquet(repo / 'FiFAR/alert_data/processed_data/alerts.parquet')
B = repo / 'l2d_benchmarking'
h = pickle.load(open(B / 'classifier_h/selected_model/best_model.pickle', 'rb'))
hp = yaml.safe_load(open(B / 'classifier_h/selected_model/model_properties.yaml'))
hprob = 1 / (1 + np.exp(-(h.predict(A.drop(columns=['fraud_bool', 'model_score', 'month']), raw_score=True) + hp['init_score'])))
pd.DataFrame({'h_prob': hprob}, index=A.index).to_parquet(out / 'h_prob.parquet')
pk = pickle.load(open(B / 'deferral/l2d_predictions/deccaf.pkl', 'rb'))
for s, df in pk.items(): df.to_parquet(out / f'deccaf_m7__{s}.parquet')
def predict(m, tr, D):
    X = D.drop(columns=['month', 'fraud_bool']); t = pd.DataFrame(index=D.index)
    for e in tr['assignment'].unique():
        if e == 'classifier_h': continue
        X['assignment'] = e; X = cc(X, CAT + ['assignment']); t[e] = m.predict_proba(X)[:, 1]
    hd = hprob[A.index.get_indexer(D.index)]; t['classifier_h'] = np.maximum(hd, 1 - hd)
    return t
V = A[A['month'] == 6]; T = A[A['month'] == 7]
for i, s in enumerate(sorted(os.listdir(B / 'expert_models/deccaf'))):
    m = pickle.load(open(B / f'expert_models/deccaf/{s}/best_model.pickle', 'rb'))
    tr = pd.read_parquet(repo / f'FiFAR/testbed/train_alert/{s}/train.parquet')
    predict(m, tr, V).to_parquet(out / f'deccaf_m6__{s}.parquet')
    if i == 0:  # check that our preprocessing reproduces the official month-7 predictions
        chk = predict(m, tr, T); ref = pk[s][chk.columns]
        print('month-7 reproduction max |diff|:', float(np.abs(chk.values - ref.loc[chk.index].values).max()))
print('exported', len(pk), 'seeds')
"""
if len(list(CACHE.glob('deccaf_m6__*.parquet'))) < 25:
    t0 = time.time()
    r = subprocess.run([str(LEGACY_PY), '-c', EXPORT, str(REPO), str(CACHE)], capture_output=True, text=True)
    print(r.stdout, r.stderr[-1500:]); assert r.returncode == 0
    print(f'export: {time.time()-t0:.0f}s')
P6 = {p.stem.split('__')[1]: pd.read_parquet(p) for p in sorted(CACHE.glob('deccaf_m6__*.parquet'))}
P7 = {p.stem.split('__')[1]: pd.read_parquet(p) for p in sorted(CACHE.glob('deccaf_m7__*.parquet'))}
h_prob = pd.read_parquet(CACHE / 'h_prob.parquet')['h_prob']
print(len(P6), 'month-6 and', len(P7), 'month-7 prediction tables')

alerts = pd.read_parquet(FIFAR / 'alert_data/processed_data/alerts.parquet')
experts = pd.read_parquet(FIFAR / 'synthetic_experts/expert_predictions.parquet').loc[alerts.index].astype(int)
y_all = alerts['fraud_bool'].astype(int)
decisions = experts.copy(); decisions['classifier_h'] = (h_prob.loc[alerts.index] >= 0.5).astype(int)
correct = decisions.eq(y_all, axis=0).astype(int)                 # is worker j right on case i (all months)
weight = y_all.map({0: LAMBDA, 1: 1.0})                               # DeCCaF cost weights
m7, m6 = alerts.index[alerts['month'] == TEST_MONTH], alerts.index[alerts['month'] == VAL_MONTH]
past = alerts.index[alerts['month'].isin([3, 4, 5])]
y = y_all.loc[m7]

envs = {}
for d in sorted((FIFAR / 'testbed/test').iterdir()):
    if d.is_dir():
        c = pd.read_csv(d / 'capacity.csv').iloc[0]; caps = c.drop(['batch_id', 'batch_size']); caps = caps[caps > 0].astype(int)
        caps['classifier_h'] = int(c['batch_size'] - caps.sum())
        envs[d.name] = dict(caps=caps, cases=pd.read_csv(d / 'batches.csv')['case_id'].values, team=d.name.split('#')[1].split('-')[0])
teams = sorted({e['team'] for e in envs.values()})
team_workers = {t: list(next(e['caps'].index for e in envs.values() if e['team'] == t)) for t in teams}
seeds_of = {t: sorted(s for s in P7 if s.split('#')[1] == t) for t in teams}

def case_cost(p, yy, lam=LAMBDA):
    p, yy = np.asarray(p), np.asarray(yy)
    return np.where((p == 0) & (yy == 1), 1.0, np.where((p == 1) & (yy == 0), lam, 0.0))

def solve_transport(C, caps):
    W, N = C.shape; cols = np.arange(W * N)
    A1 = coo_matrix((np.ones(W * N), (np.tile(np.arange(N), W), cols)), shape=(N, W * N))
    A2 = coo_matrix((np.ones(W * N), (np.repeat(np.arange(W), N), cols)), shape=(W, W * N))
    r = linprog(C.ravel(), A_eq=vstack([A1, A2]).tocsr(), b_eq=np.r_[np.ones(N), caps], bounds=(0, 1), method='highs')
    assert r.status == 0, r.message
    x = r.x.reshape(W, N); assert np.allclose(x.max(0), 1, atol=1e-6)
    return x.argmax(0)

def allocate(env, P, cw=None):
    """Maximize sum of (optionally case-weighted) estimated correctness under exact capacities; return per-case realized costs."""
    caps, cases = envs[env]['caps'], envs[env]['cases']
    S = P.loc[cases, caps.index].values.astype(float)
    if cw is not None: S = S * cw.loc[cases].values[:, None]
    a = solve_transport(-S.T, caps.values)
    chosen = decisions.loc[cases, caps.index].values[np.arange(len(cases)), a]
    return case_cost(chosen, y_all.loc[cases].values)

def run_level(jobs):
    """jobs: list of (key, env, P[, case_weights]). Returns {key: cost vector}."""
    out = Parallel(n_jobs=N_JOBS)(delayed(allocate)(*j[1:]) for j in jobs)
    return {j[0]: v for j, v in zip(jobs, out)}

# sanity: classifier h hard decisions match the official Only_Classifier output
oc = pd.read_parquet(sorted((RESULTS / 'Only_Classifier').rglob('results.parquet'))[0])['prediction'].astype(int)
print('classifier h agreement with official output:', float((oc.loc[m7] == decisions.loc[m7, 'classifier_h']).mean()))
print('teams:', teams, '| seeds per team:', {t: len(s) for t, s in seeds_of.items()})

COST = {}   # (level, seed_or_team, env) -> per-case cost vector (aligned with envs[env]['cases'])
for env, e in envs.items():
    for s in seeds_of[e['team']]:
        p = pd.read_parquet(RESULTS / 'DeCCaF' / s / env / 'results.parquet')['prediction'].astype(int).reindex(e['cases'])
        COST[('L0', s, env)] = case_cost(p.values, y_all.loc[e['cases']].values)
t0 = time.time()
res = run_level([(('L0-LP', s, env), env, P7[s]) for env, e in envs.items() for s in seeds_of[e['team']]])
COST.update(res); print(f'L0-LP: {len(res)} allocations in {time.time()-t0:.0f}s')
l0 = np.mean([v.sum() for k, v in COST.items() if k[0] == 'L0']); l0lp = np.mean([v.sum() for k, v in COST.items() if k[0] == 'L0-LP'])
print(f'mean loss  official CP-SAT: {l0:.1f} | exact LP: {l0lp:.1f}')

def isotonic_map(p_fit, c_fit, w_fit, p_apply):
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(p_fit, c_fit, sample_weight=w_fit)
    return iso.predict(p_apply) + 1e-6 * np.asarray(p_apply)   # tiny tie-break keeps the original order inside flat steps

def recal_val(s):       # deployable: fit on validation month 6, apply to test month 7
    P = P7[s].copy()
    for w in P.columns:
        P[w] = isotonic_map(P6[s][w].values, correct.loc[m6, w].values, weight.loc[m6].values, P7[s][w].values)
    return P

kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
def recal_cross(s):     # diagnostic: cross-fitted on test month 7
    P = P7[s].copy(); idx = P.index.values
    for w in P.columns:
        out = np.zeros(len(idx))
        for tr, te in kf.split(idx):
            out[te] = isotonic_map(P7[s][w].values[tr], correct.loc[idx[tr], w].values, weight.loc[idx[tr]].values, P7[s][w].values[te])
        P[w] = out
    return P

P_L1 = {s: recal_val(s) for s in P7}; P_L1s = {s: recal_cross(s) for s in P7}
t0 = time.time()
for lvl, PP in [('L1', P_L1), ('L1*', P_L1s)]:
    COST.update(run_level([((lvl, s, env), env, PP[s]) for env, e in envs.items() for s in seeds_of[e['team']]]))
print(f'L1, L1*: {time.time()-t0:.0f}s')

# Case weights D = P(fraud) + λ·P(legit), from classifier h recalibrated (cost-weighted) on month 6
h_w = pd.Series(isotonic_map(h_prob.loc[m6].values, y_all.loc[m6].values, weight.loc[m6].values, h_prob.values),
                index=h_prob.index).clip(1e-6, 1 - 1e-6)
CW = LAMBDA / (1 - h_w + LAMBDA * h_w)
print('case weight D: min %.3f | median %.3f | max %.3f' % tuple(CW.loc[m7].quantile([0, .5, 1])))
t0 = time.time()
for lvl, PP in [('L0-cw', P7), ('L1-cw', P_L1)]:
    COST.update(run_level([((lvl, s, env), env, PP[s], CW) for env, e in envs.items() for s in seeds_of[e['team']]]))
print(f'L0-cw, L1-cw: {time.time()-t0:.0f}s')

FEATS = [c for c in alerts.columns if c not in ['fraud_bool', 'month']]   # same inputs as DeCCaF (incl. model_score)

def long_table(idx, workers):
    X = alerts.loc[idx, FEATS]
    parts = []
    for w in workers:
        d = X.copy(); d['expert'] = w; d['case_id'] = idx
        d['target'] = correct.loc[idx, w].values; d['w'] = weight.loc[idx].values
        parts.append(d)
    L = pd.concat(parts, ignore_index=True)
    L['expert'] = pd.Categorical(L['expert'], categories=workers)
    return L

def hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
                                          validation_fraction=0.15, categorical_features='from_dtype', random_state=SEED)

def predict_table(model, idx, workers):
    L = long_table(idx, workers)
    p = model.predict_proba(L[FEATS + ['expert']])[:, 1]
    return pd.DataFrame(p.reshape(len(workers), len(idx)).T, index=idx, columns=workers)

def h_column(fit_idx, apply_idx):
    ph = np.maximum(h_prob, 1 - h_prob)
    return isotonic_map(ph.loc[fit_idx].values, correct.loc[fit_idx, 'classifier_h'].values, weight.loc[fit_idx].values, ph.loc[apply_idx].values)

P_L2, P_L3 = {}, {}
t0 = time.time()
for t in teams:
    ex = [w for w in team_workers[t] if w != 'classifier_h']
    Ltr = long_table(past, ex)
    m = hgb().fit(Ltr[FEATS + ['expert']], Ltr['target'], sample_weight=Ltr['w'])
    P = predict_table(m, m7, ex); P['classifier_h'] = h_column(m6, m7); P_L2[t] = P
    idx = m7.values; P3 = pd.DataFrame(index=m7, columns=ex, dtype=float)
    for tr, te in kf.split(idx):
        Lf = long_table(idx[tr], ex)
        mf = hgb().fit(Lf[FEATS + ['expert']], Lf['target'], sample_weight=Lf['w'])
        P3.iloc[te] = predict_table(mf, idx[te], ex).values
    hc = pd.Series(index=m7, dtype=float)
    for tr, te in kf.split(idx): hc.iloc[te] = h_column(idx[tr], idx[te])
    P3['classifier_h'] = hc.values; P_L3[t] = P3
    print(f'{t}: models fitted ({time.time()-t0:.0f}s)')
for lvl, PP in [('L2', P_L2), ('L3', P_L3)]:
    COST.update(run_level([((lvl, e['team'], env), env, PP[e['team']]) for env, e in envs.items()]))
print(f'L2, L3 done: {time.time()-t0:.0f}s')

# L2eq: train on month 6 with exactly the size of an L3 training fold (5 repetitions), predict month 7
n_fold = len(next(kf.split(m7.values))[0]); rng_eq = np.random.default_rng(SEED)
P_L2eq = {}
for t in teams:
    ex = [w for w in team_workers[t] if w != 'classifier_h']
    for r in range(N_FOLDS):
        sub = pd.Index(rng_eq.choice(m6.values, size=n_fold, replace=False))
        Lf = long_table(sub, ex)
        mf = hgb().fit(Lf[FEATS + ['expert']], Lf['target'], sample_weight=Lf['w'])
        P = predict_table(mf, m7, ex); P['classifier_h'] = h_column(sub, m7); P_L2eq[(t, r)] = P
    print(f'{t}: L2eq models fitted ({time.time()-t0:.0f}s)')
COST.update(run_level([(('L2eq', f'{t}#r{r}', env), env, P_L2eq[(t, r)]) for env, e in envs.items()
                       for (t, r) in P_L2eq if t == e['team']]))
print(f'L2eq done (training size {n_fold} cases per model): {time.time()-t0:.0f}s')

def oracle_costs(env):
    caps, cases = envs[env]['caps'], envs[env]['cases']; yy = y_all.loc[cases].values
    true = np.vstack([case_cost(decisions.loc[cases, w].values, yy) for w in caps.index])
    fnr = {w: ((decisions.loc[m7, w] == 0) & (y == 1)).sum() / (y == 1).sum() for w in caps.index}
    fpr = {w: ((decisions.loc[m7, w] == 1) & (y == 0)).sum() / (y == 0).sum() for w in caps.index}
    lab = np.vstack([np.where(yy == 1, fnr[w], LAMBDA * fpr[w]) for w in caps.index])
    n = np.arange(len(cases))
    return true[solve_transport(lab, caps.values), n], true[solve_transport(true, caps.values), n]
t0 = time.time()
orc = Parallel(n_jobs=N_JOBS)(delayed(oracle_costs)(env) for env in envs)
for env, (lab, case) in zip(envs, orc):
    COST[('LabelOracle', 'oracle', env)] = lab; COST[('CaseOracle', 'oracle', env)] = case
print(f'oracles: {time.time()-t0:.0f}s')

LEVELS = ['L0', 'L0-LP', 'L0-cw', 'L1', 'L1-cw', 'L1*', 'L2', 'L2eq', 'L3', 'LabelOracle', 'CaseOracle']
rows = []
for env, e in envs.items():
    r = dict(env=env, team=e['team'])
    for lvl in LEVELS:
        r[lvl] = float(np.mean([v.sum() for k, v in COST.items() if k[0] == lvl and k[2] == env]))
    rows.append(r)
lad = pd.DataFrame(rows)
gap = lad['L0-LP'] - lad['LabelOracle']
for lvl in ['L0-cw', 'L1', 'L1-cw', 'L1*', 'L2', 'L2eq', 'L3']:
    lad[f'closed_{lvl}'] = (lad['L0-LP'] - lad[lvl]) / gap
print('Mean loss per level (lower is better):')
print(lad.groupby('team')[LEVELS].mean().round(1).to_string())
print('\nOverall:'); print(lad[LEVELS].mean().round(2).to_string())
print('\nShare of the DeCCaF → label-oracle gap closed (mean over scenarios):')
print(lad[[c for c in lad if c.startswith('closed_')]].mean().round(3).to_string())
lad.to_csv(OUT / '1_ladder_by_scenario.csv', index=False)

rng = np.random.default_rng(SEED)
def level_vec(lvl, env):  # per-case cost averaged over training seeds (if any)
    return np.mean([v for k, v in COST.items() if k[0] == lvl and k[2] == env], axis=0)
N = len(next(iter(envs.values()))['cases'])
Wb = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)
def step(a, b):
    d = np.mean([level_vec(b, env) - level_vec(a, env) for env in envs], axis=0)   # mean over scenarios, per case
    boot = Wb @ d
    return dict(step=f'{a} → {b}', mean_change=float(d.sum()), ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)))
steps = pd.DataFrame([step('L0', 'L0-LP'), step('L0-LP', 'L0-cw'), step('L0-LP', 'L1'), step('L1', 'L1-cw'),
                      step('L0-LP', 'L1-cw'), step('L0-LP', 'L1*'), step('L0-LP', 'L2'), step('L2', 'L2eq'),
                      step('L2eq', 'L3'), step('L3', 'LabelOracle'), step('L0-LP', 'LabelOracle')])
steps['significant'] = (steps.ci_high < 0) | (steps.ci_low > 0)
print('Change in loss (negative = improvement), mean over 25 scenarios:'); print(steps.round(2).to_string(index=False))
steps.to_csv(OUT / '2_ladder_steps_ci.csv', index=False)

val = alerts.loc[m6]
e_c = val['fraud_bool'].replace([0, 1], [LAMBDA, 1]).mean()
reb_1, reb_0 = val['fraud_bool'].mean() / e_c, (1 - val['fraud_bool']).mean() * LAMBDA / e_c
nmax = int((y == 0).sum() * reb_1 / reb_0)
over = pd.concat([y[y == 0], y[y == 1].sample(replace=True, n=nmax, random_state=SEED)]).index   # as in results.ipynb
def ece(c, p):
    p = np.clip(np.asarray(p, dtype=float), 0, 1)   # the isotonic tie-break can exceed 1 by ~1e-6
    pt, pp = calibration_curve(c, p, strategy='quantile', n_bins=10); return float(np.mean(np.abs(pt - pp)))
q = []
for lvl, src in [('L0 DeCCaF', P7), ('L1 recal (m6)', P_L1), ('L1* recal (m7 cross-fit)', P_L1s)]:
    for s, P in src.items():
        for w in P.columns.drop('classifier_h'):
            c = correct.loc[over, w]; q.append(dict(level=lvl, auc=roc_auc_score(c, P.loc[over, w]), ece=ece(c, P.loc[over, w])))
for lvl, src in [('L2 past re-estimate', P_L2), ('L2eq month-6 equal size', P_L2eq), ('L3 test cross-fit', P_L3)]:
    for t, P in src.items():
        for w in P.columns.drop('classifier_h'):
            c = correct.loc[over, w]; q.append(dict(level=lvl, auc=roc_auc_score(c, P.loc[over, w].astype(float)), ece=ece(c, P.loc[over, w].astype(float))))
qual = pd.DataFrame(q).groupby('level')[['auc', 'ece']].mean()

# Within-case discrimination: for each test case, does the estimate rank an expert who is right above one who is wrong?
# (Allocation compares experts on the same case, which per-expert AUC does not measure.)
def within_case(P, team):
    ex = [w for w in team_workers[team] if w != 'classifier_h']
    Pm, Cm = P.loc[m7, ex].values.astype(float), correct.loc[m7, ex].values
    G = (Pm[:, :, None] > Pm[:, None, :]) + 0.5 * (Pm[:, :, None] == Pm[:, None, :])
    V = Cm[:, :, None] * (1 - Cm[:, None, :])
    return float((G * V).sum() / V.sum())
wc = {}
for lvl, src, team_of in [('L0 DeCCaF', P7, lambda k: k.split('#')[1]), ('L1 recal (m6)', P_L1, lambda k: k.split('#')[1]),
                          ('L1* recal (m7 cross-fit)', P_L1s, lambda k: k.split('#')[1]), ('L2 past re-estimate', P_L2, lambda k: k),
                          ('L2eq month-6 equal size', P_L2eq, lambda k: k[0]), ('L3 test cross-fit', P_L3, lambda k: k)]:
    wc[lvl] = np.mean([within_case(P, team_of(k)) for k, P in src.items()])
qual['within_case_auc'] = pd.Series(wc)
qual = qual.round(4)
print(qual.to_string()); qual.to_csv(OUT / '3_estimate_quality.csv')

o = lad[LEVELS].mean()
summary = {
  'results_source': RESULTS_SOURCE,
  'mean_loss': o.round(2).to_dict(),
  'gap_L0LP_to_label_oracle': round(float(o['L0-LP'] - o['LabelOracle']), 2),
  'share_closed': lad[[c for c in lad if c.startswith('closed_')]].mean().round(3).to_dict(),
  'objective_case_weighting_L0LP_to_L0cw': round(float(o['L0-LP'] - o['L0-cw']), 2),
  'shift_effect_L2eq_to_L3_equal_data': round(float(o['L2eq'] - o['L3']), 2),
  'steps': steps.round(2).to_dict('records'),
  'estimate_quality': qual.to_dict('index'),
  'notes': ['L1* and L3 use test-month information (cross-fitted): diagnostic bounds, not deployable methods',
            'Levels change one ingredient at a time and are not perfectly additive',
            'L2/L2eq/L3 use every team expert decision on past/test cases, i.e. more supervision than DeCCaF had',
            'Case weights D are estimated from classifier h recalibrated on month 6 (deployable)',
            'Synthetic experts; 25 scenarios share one test month']}
(OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
pack = OUT.parent / f'gap_decomposition_v2_{RESULTS_SOURCE}.zip'
with zipfile.ZipFile(pack, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUT.glob('*.*'): z.write(p, p.name)
print('ZIP:', pack)