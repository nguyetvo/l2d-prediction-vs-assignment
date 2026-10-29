"""OvA vs DeCCaF when models are RETRAINED at each cost ratio λ (authors' code), compared with
re-weighting the evaluation only. One training testbed per team (shuffle_1#team_1..5), all 25 test scenarios,
same LP assignment for both methods. Outputs CSV into config.LAMBDA_OUT and the figure into config.FIG_DIR.
The setup part of this file (up to the export loop) is also executed by 08, 09 and 10 via exec."""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
OFFICIAL = config.OFFICIAL; FIFAR = OFFICIAL / 'FiFAR'
RETRAIN = config.RETRAIN
LEGACY_PY = config.LEGACY_PY
OUT = config.LAMBDA_OUT; OUT.mkdir(parents=True, exist_ok=True)
FIG = config.FIG_DIR
EXPORT = Path(__file__).resolve().parent / '06_lambda_export_legacy.py'
LAMS = [0.02, 0.057, 0.1, 0.2, 0.5]
WS = {0.057: RETRAIN, **{l: config.lambda_workspace(l) for l in LAMS if l != 0.057}}
SEED, N_BOOT = 42, 1000

alerts = pd.read_parquet(FIFAR / 'alert_data/processed_data/alerts.parquet')
experts = pd.read_parquet(FIFAR / 'synthetic_experts/expert_predictions.parquet').loc[alerts.index].astype(int)
y = alerts['fraud_bool'].astype(int)
envs = {}
for d in sorted((FIFAR / 'testbed/test').iterdir()):
    c = pd.read_csv(d / 'capacity.csv').iloc[0]; caps = c.drop(['batch_id', 'batch_size']); caps = caps[caps > 0].astype(int)
    caps['classifier_h'] = int(c['batch_size'] - caps.sum())
    envs[d.name] = dict(caps=caps, cases=pd.read_csv(d / 'batches.csv')['case_id'].values, team=d.name.split('#')[1].split('-')[0])

def solve_transport(C, caps):
    W, N = C.shape; cols = np.arange(W * N)
    A1 = coo_matrix((np.ones(W * N), (np.tile(np.arange(N), W), cols)), shape=(N, W * N))
    A2 = coo_matrix((np.ones(W * N), (np.repeat(np.arange(W), N), cols)), shape=(W, W * N))
    r = linprog(C.ravel(), A_eq=vstack([A1, A2]).tocsr(), b_eq=np.r_[np.ones(N), caps], bounds=(0, 1), method='highs')
    assert r.status == 0, r.message
    return r.x.reshape(W, N).argmax(0)

def assign(env, P, decisions):
    """Returns per-case (fn, fp) indicators of the LP assignment that maximizes the sum of predicted correctness."""
    caps, cases = envs[env]['caps'], envs[env]['cases']
    a = solve_transport(-P.loc[cases, caps.index].values.astype(float).T, caps.values)
    pred = decisions.loc[cases, caps.index].values[np.arange(len(cases)), a]; yy = y.loc[cases].values
    return ((pred == 0) & (yy == 1)).astype(float), ((pred == 1) & (yy == 0)).astype(float)

ok = {}
for lam, ws in WS.items():
    cache = OUT / f'cache_{lam:g}'
    pk = ws / 'l2d_benchmarking/deferral/l2d_predictions/deccaf.pkl'
    if not pk.exists(): print(f'λ={lam}: models not ready, skipped'); continue
    stale = not (cache / 'h_prob.parquet').exists() or (cache / 'h_prob.parquet').stat().st_mtime < pk.stat().st_mtime
    if stale:
        r = subprocess.run([str(LEGACY_PY), str(EXPORT), str(ws), str(cache)], capture_output=True, text=True)
        print(r.stdout.strip(), r.stderr[-500:]); assert r.returncode == 0
    ok[lam] = cache
print('available training λ:', sorted(ok))

FNFP = {}   # (train_lam, method, env) -> (fn vector, fp vector)
for lam, cache in ok.items():
    h = pd.read_parquet(cache / 'h_prob.parquet')['h_prob']
    dec = experts.copy(); dec['classifier_h'] = (h.loc[alerts.index] >= 0.5).astype(int)
    jobs = []
    for m in ['ova', 'deccaf']:
        for f in sorted(cache.glob(f'{m}__*.parquet')):
            P = pd.read_parquet(f); seed = f.stem.split('__')[1]; team = seed.split('#')[1]
            jobs += [((lam, m, seed, env), env, P, dec) for env, e in envs.items() if e['team'] == team]
    res = Parallel(n_jobs=6)(delayed(assign)(*j[1:]) for j in jobs)
    FNFP.update({j[0]: v for j, v in zip(jobs, res)})

rng = np.random.default_rng(SEED); N = len(next(iter(envs.values()))['cases'])
Wb = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)
def seeds_of(lam): return sorted({k[2] for k in FNFP if k[0] == lam})
def loss_vec(train_lam, m, eval_lam, seeds):
    return np.mean([FNFP[(train_lam, m, sd, env)][0] + eval_lam * FNFP[(train_lam, m, sd, env)][1]
                    for sd in seeds for env in envs if envs[env]['team'] == sd.split('#')[1]], axis=0)
def compare(eval_lam, kind, train_lam, seeds, design):
    o, d = loss_vec(train_lam, 'ova', eval_lam, seeds), loss_vec(train_lam, 'deccaf', eval_lam, seeds)
    diff = d - o; boot = Wb @ diff
    return dict(design=design, eval_lambda=eval_lam, setting=kind, n_training_testbeds=len(seeds), OvA=o.sum(), DeCCaF=d.sum(),
                ratio_OvA_to_DeCCaF=o.sum() / d.sum(), DeCCaF_minus_OvA=diff.sum(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5))
S1 = [f'shuffle_1#team_{i}' for i in range(1, 6)]
rows = []
for eval_lam in LAMS:      # design A: lambda grid, one training testbed per team
    for kind, train_lam in [('evaluation only (models trained at 0.057)', 0.057), ('retrained at this λ', eval_lam)]:
        if train_lam in ok and all(sd in seeds_of(train_lam) for sd in S1):
            rows.append(compare(eval_lam, kind, train_lam, S1, 'A: λ grid, 1 testbed/team'))
for eval_lam in LAMS:      # design B: every lambda that has all five training testbeds per team (authors' design)
    if eval_lam != 0.057 and eval_lam in ok and len(seeds_of(eval_lam)) == 25 and len(seeds_of(0.057)) == 25:
        rows.append(compare(eval_lam, 'evaluation only (models trained at 0.057)', 0.057, seeds_of(0.057), 'B: 5 testbeds/team'))
        rows.append(compare(eval_lam, 'retrained at this λ', eval_lam, seeds_of(eval_lam), 'B: 5 testbeds/team'))
        rows.append(compare(0.057, 'retrained at this λ', 0.057, seeds_of(0.057), 'B: 5 testbeds/team'))
res = pd.DataFrame(rows).drop_duplicates(['design', 'eval_lambda', 'DeCCaF', 'OvA'])

# solver check: LP assignment vs the official CP-SAT results at λ = 0.057 (retrained models, same testbeds)
chk = []
for m, name in [('ova', 'OvA'), ('deccaf', 'DeCCaF')]:
    for sd in S1:
        for env in envs:
            if envs[env]['team'] != sd.split('#')[1]: continue
            f = RETRAIN / 'l2d_benchmarking/deferral/results' / name / sd / env / 'results.parquet'
            if not f.exists() or (0.057, m, sd, env) not in FNFP: continue
            pr = pd.read_parquet(f)['prediction'].astype(int).reindex(envs[env]['cases']).values; yy = y.loc[envs[env]['cases']].values
            fn, fp = FNFP[(0.057, m, sd, env)]
            chk.append(dict(method=name, official_cp_sat=((pr == 0) & (yy == 1)).sum() + 0.057 * ((pr == 1) & (yy == 0)).sum(), our_lp=fn.sum() + 0.057 * fp.sum()))
if chk:
    chk = pd.DataFrame(chk).groupby('method').mean().round(2); print('solver check (λ = 0.057, shuffle_1):'); print(chk.to_string())
    chk.to_csv(OUT / 'solver_check_lp_vs_cpsat.csv')
res['winner'] = np.where(res.ci_high < 0, 'DeCCaF', np.where(res.ci_low > 0, 'OvA', 'no significant difference'))
pd.set_option('display.width', 200); print(res.round(3).to_string(index=False))
res.round(4).to_csv(OUT / 'lambda_retrain_results.csv', index=False)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for kind, col, mk, lab in [('evaluation only (models trained at 0.057)', '#6b6b6b', 'o', 'Re-weighted evaluation only'),
                           ('retrained at this λ', '#2a78d6', 's', 'Both methods retrained at λ')]:
    A = res[res.design.str.startswith('A')]
    g = A[(A.setting == kind) | ((A.eval_lambda == 0.057) & (kind == 'retrained at this λ'))].sort_values('eval_lambda')
    ax.plot(g.eval_lambda, g.ratio_OvA_to_DeCCaF, color=col, lw=2, marker=mk, ms=5, markeredgecolor='white', label=lab)
ax.axhline(1, color='#9a9a9a', lw=1, ls='--'); ax.text(0.021, 1.004, 'equal loss', fontsize=6.5, color='#6b6b6b')
ax.set_xscale('log'); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks(LAMS); ax.set_xticklabels([f'{v:g}' for v in LAMS])
ax.set_xlabel('Cost ratio λ (log scale)'); ax.set_ylabel('OvA loss / DeCCaF loss'); ax.legend(fontsize=6.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e3e3e3', lw=0.6)
FIG.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'): fig.savefig(FIG / f'F3b_lambda_retrained.{ext}', dpi=300, bbox_inches='tight')
print('saved', OUT / 'lambda_retrain_results.csv')
