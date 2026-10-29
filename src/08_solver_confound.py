"""How much of DeCCaF's advantage over OvA comes from the assignment procedure?
Official code: DeCCaF -> CP-SAT (optimal batch assignment); OvA -> greedy case-by-case. Here both are also
re-assigned with the same exact LP. Retrained models, lambda = 0.057, all 5 training testbeds x 25 scenarios.
Usage: python src/08_solver_confound.py [retrain|author]"""
import os, sys, subprocess
SRC = sys.argv[1] if len(sys.argv) > 1 else 'retrain'
sys.argv = ['x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '07_lambda_eval.py'), encoding='utf-8').read().split('ok = {}')[0])
import numpy as np, pandas as pd
if SRC == 'author':
    REPO, RESDIR = OFFICIAL, OFFICIAL / 'l2d_benchmarking/deferral/results_author_official'
    cache = OUT / 'cache_author'
    if not (cache / 'h_prob.parquet').exists():
        r = subprocess.run([str(LEGACY_PY), str(EXPORT), str(OFFICIAL), str(cache)], capture_output=True, text=True); print(r.stdout, r.stderr[-400:])
else:
    REPO, RESDIR = RETRAIN, RETRAIN / 'l2d_benchmarking/deferral/results'
    cache = OUT / 'cache_0.057'
h = pd.read_parquet(cache / 'h_prob.parquet')['h_prob']
dec = experts.copy(); dec['classifier_h'] = (h.loc[alerts.index] >= 0.5).astype(int)
LAM = 0.057
jobs = []
for m in ['ova', 'deccaf']:
    for f in sorted(cache.glob(f'{m}__*.parquet')):
        P = pd.read_parquet(f); seed = f.stem.split('__')[1]; team = seed.split('#')[1]
        jobs += [((m, seed, env), env, P, dec) for env, e in envs.items() if e['team'] == team]
res = Parallel(n_jobs=6)(delayed(assign)(*j[1:]) for j in jobs)
LP = {j[0]: v[0] + LAM * v[1] for j, v in zip(jobs, res)}
OFF = {}
for (m, seed, env) in LP:
    name = {'ova': 'OvA', 'deccaf': 'DeCCaF'}[m]
    pr = pd.read_parquet(RESDIR / name / seed / env / 'results.parquet')['prediction'].astype(int).reindex(envs[env]['cases']).values
    yy = y.loc[envs[env]['cases']].values
    OFF[(m, seed, env)] = ((pr == 0) & (yy == 1)) + LAM * ((pr == 1) & (yy == 0))
keys = sorted({(s, e) for (_, s, e) in LP})
def vec(D, m): return np.mean([D[(m, s, e)] for s, e in keys], axis=0)
rng = np.random.default_rng(42); N = len(next(iter(envs.values()))['cases']); Wb = rng.multinomial(N, np.ones(N) / N, size=1000).astype(float)
def ci(v): b = Wb @ v; return f'{v.sum():.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]'
out = {
 'runs per method': len(keys),
 'OvA official greedy': vec(OFF, 'ova').sum(), 'OvA exact LP': vec(LP, 'ova').sum(),
 'DeCCaF official CP-SAT': vec(OFF, 'deccaf').sum(), 'DeCCaF exact LP': vec(LP, 'deccaf').sum(),
 'DeCCaF - OvA, official procedures': ci(vec(OFF, 'deccaf') - vec(OFF, 'ova')),
 'DeCCaF - OvA, same LP for both': ci(vec(LP, 'deccaf') - vec(LP, 'ova')),
 'OvA: LP - greedy (assignment effect)': ci(vec(LP, 'ova') - vec(OFF, 'ova')),
}
d_off = (vec(OFF, 'deccaf') - vec(OFF, 'ova')).sum(); d_lp = (vec(LP, 'deccaf') - vec(LP, 'ova')).sum()
out['share of the official DeCCaF advantage explained by the assignment procedure'] = f'{100 * (1 - d_lp / d_off):.1f}%'
by_team = pd.DataFrame({t: {'OvA greedy': np.mean([OFF[('ova', s, e)].sum() for s, e in keys if envs[e]['team'] == t]),
                            'OvA LP': np.mean([LP[('ova', s, e)].sum() for s, e in keys if envs[e]['team'] == t]),
                            'DeCCaF CP-SAT': np.mean([OFF[('deccaf', s, e)].sum() for s, e in keys if envs[e]['team'] == t]),
                            'DeCCaF LP': np.mean([LP[('deccaf', s, e)].sum() for s, e in keys if envs[e]['team'] == t])} for t in sorted({e['team'] for e in envs.values()})}).T.round(1)
for k, v in out.items(): print(f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}')
print(by_team.to_string())
pd.Series({k: str(v) for k, v in out.items()}).to_csv(OUT / f'solver_confound_summary_{SRC}.csv'); by_team.to_csv(OUT / f'solver_confound_by_team_{SRC}.csv')
