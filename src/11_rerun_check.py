"""Released-artifact consistency check.
1. Compares the re-run of the authors' deferral script (00 --rerun-deferral, l2d_benchmarking/deferral/results)
   with the shipped results (results_author_official), file by file.
2. Recomputes the released OvA greedy assignment (run_alert.py, ova_deferral_func: batch by batch, each case to
   the available worker with the highest predicted score) from the shipped OvA predictions (ova.pkl), and measures
   its agreement with the shipped and the re-run assignments.
3. Repeats the author column of the solver comparison (08_solver_confound.py author) with the recomputed OvA greedy
   in place of the shipped OvA results; same LP, same bootstrap.
Needs 00_setup_official_run.py --rerun-deferral (for part 1) and 08_solver_confound.py author (cache_author).
Outputs into config.LAMBDA_OUT: rerun_vs_shipped_files.csv, ova_greedy_recomputed_agreement.csv,
solver_confound_summary_author_ova_recomputed.csv, solver_confound_by_team_author_ova_recomputed.csv.
Usage: python src/11_rerun_check.py"""
import os, sys
sys.argv = ['x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '07_lambda_eval.py'), encoding='utf-8').read().split('ok = {}')[0])
import numpy as np, pandas as pd

DEF = OFFICIAL / 'l2d_benchmarking/deferral'
SHIPPED, RERUN = DEF / 'results_author_official', DEF / 'results'
cache = OUT / 'cache_author'
assert (cache / 'h_prob.parquet').exists(), 'run 08_solver_confound.py author first (creates cache_author)'
LAM = 0.057

# 1. re-run vs shipped, file by file
if RERUN.exists():
    rows = []
    for f in sorted(SHIPPED.rglob('*.parquet')):
        r = f.relative_to(SHIPPED); g = RERUN / r
        same = g.exists() and pd.read_parquet(f).reset_index(drop=True).equals(pd.read_parquet(g).reset_index(drop=True))
        rows.append(dict(method=r.parts[0], file=str(r.as_posix()), identical=same))
    cmp = pd.DataFrame(rows); cmp.to_csv(OUT / 'rerun_vs_shipped_files.csv', index=False)
    print('re-run vs shipped, files not identical by method:'); print(cmp.groupby('method')['identical'].agg(['size', lambda s: int((~s).sum())]).rename(columns={'<lambda_0>': 'different'}).to_string())
else:
    print('no re-run found (00_setup_official_run.py --rerun-deferral); part 1 skipped')

# 2. OvA greedy recomputed from ova.pkl, exactly as ova_deferral_func in run_alert.py
h = pd.read_parquet(cache / 'h_prob.parquet')['h_prob']
dec = experts.copy(); dec['classifier_h'] = (h.loc[alerts.index] >= 0.5).astype(int)

def ova_greedy(env, P):
    d = FIFAR / 'testbed/test' / env
    capacities, batches = pd.read_csv(d / 'capacity.csv'), pd.read_csv(d / 'batches.csv')
    out = {}
    for i in np.arange(1, batches['batch'].max() + 1):
        c = capacities.loc[capacities['batch_id'] == i, :].iloc[0]
        c.loc['classifier_h'] = c['batch_size'] - c[2:].sum()
        c = c.loc[c != 0]
        preds = P.loc[batches.loc[batches['batch'] == i]['case_id'], c.index.drop(['batch_id', 'batch_size'])]
        for ix, row in preds.iterrows():
            for choice in row.sort_values(ascending=False).index:
                if c[choice] > 0:
                    c[choice] -= 1; out[ix] = choice; break
    return pd.Series(out)

jobs = []
for f in sorted(cache.glob('ova__*.parquet')):
    P = pd.read_parquet(f); seed = f.stem.split('__')[1]; team = seed.split('#')[1]
    jobs += [(seed, env, P) for env, e in envs.items() if e['team'] == team]
greedy = dict(zip([(s, e) for s, e, _ in jobs], Parallel(n_jobs=6)(delayed(ova_greedy)(e, P) for _, e, P in jobs)))
rows = []
for (seed, env), a in greedy.items():
    r = dict(seed=seed, env=env)
    for name, base in [('shipped', SHIPPED), ('rerun', RERUN)]:
        f = base / 'OvA' / seed / env / 'assignments.parquet'
        if f.exists():
            s = pd.read_parquet(f)['assignment']; r[f'agree_{name}'] = float((a.reindex(s.index) == s).mean())
    rows.append(r)
agr = pd.DataFrame(rows); agr.to_csv(OUT / 'ova_greedy_recomputed_agreement.csv', index=False)
print(f'\nOvA greedy recomputed from ova.pkl: {len(agr)} runs')
for c in [c for c in agr.columns if c.startswith('agree_')]:
    bad = agr[agr[c] < 1]
    print(f'  {c}: identical in {int((agr[c] == 1).sum())}/{len(agr)} runs; others: ' + (', '.join(f'{s}/{e} ({v:.3f})' for s, e, v in bad[['seed', 'env', c]].values) or 'none'))

# 3. author column of the solver comparison with the recomputed OvA greedy (same LP and bootstrap as 08)
lp_jobs = []
for m in ['ova', 'deccaf']:
    for f in sorted(cache.glob(f'{m}__*.parquet')):
        P = pd.read_parquet(f); seed = f.stem.split('__')[1]; team = seed.split('#')[1]
        lp_jobs += [((m, seed, env), env, P, dec) for env, e in envs.items() if e['team'] == team]
res = Parallel(n_jobs=6)(delayed(assign)(*j[1:]) for j in lp_jobs)
LP = {j[0]: v[0] + LAM * v[1] for j, v in zip(lp_jobs, res)}
OFF = {}
for (m, seed, env) in LP:
    cases = envs[env]['cases']; yy = y.loc[cases].values
    if m == 'ova':
        a = greedy[(seed, env)].reindex(cases).values
        pr = dec.loc[cases].values[np.arange(len(cases)), dec.columns.get_indexer(a)]
    else:
        pr = pd.read_parquet(SHIPPED / 'DeCCaF' / seed / env / 'results.parquet')['prediction'].astype(int).reindex(cases).values
    OFF[(m, seed, env)] = ((pr == 0) & (yy == 1)) + LAM * ((pr == 1) & (yy == 0))
keys = sorted({(s, e) for (_, s, e) in LP})
def vec(D, m): return np.mean([D[(m, s, e)] for s, e in keys], axis=0)
rng = np.random.default_rng(42); N = len(next(iter(envs.values()))['cases']); Wb = rng.multinomial(N, np.ones(N) / N, size=1000).astype(float)
def ci(v): b = Wb @ v; return f'{v.sum():.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]'
out = {
 'runs per method': len(keys),
 'OvA greedy recomputed from ova.pkl': vec(OFF, 'ova').sum(), 'OvA exact LP': vec(LP, 'ova').sum(),
 'DeCCaF official CP-SAT (shipped)': vec(OFF, 'deccaf').sum(), 'DeCCaF exact LP': vec(LP, 'deccaf').sum(),
 'DeCCaF - OvA, official procedures': ci(vec(OFF, 'deccaf') - vec(OFF, 'ova')),
 'DeCCaF - OvA, same LP for both': ci(vec(LP, 'deccaf') - vec(LP, 'ova')),
 'OvA: LP - greedy (assignment effect)': ci(vec(LP, 'ova') - vec(OFF, 'ova')),
}
d_off = (vec(OFF, 'deccaf') - vec(OFF, 'ova')).sum(); d_lp = (vec(LP, 'deccaf') - vec(LP, 'ova')).sum()
out['share of the official DeCCaF advantage explained by the assignment procedure'] = f'{100 * (1 - d_lp / d_off):.1f}%'
by_team = pd.DataFrame({t: {'OvA greedy (recomputed)': np.mean([OFF[('ova', s, e)].sum() for s, e in keys if envs[e]['team'] == t]),
                            'OvA LP': np.mean([LP[('ova', s, e)].sum() for s, e in keys if envs[e]['team'] == t]),
                            'DeCCaF CP-SAT': np.mean([OFF[('deccaf', s, e)].sum() for s, e in keys if envs[e]['team'] == t]),
                            'DeCCaF LP': np.mean([LP[('deccaf', s, e)].sum() for s, e in keys if envs[e]['team'] == t])} for t in sorted({e['team'] for e in envs.values()})}).T.round(1)
print('\nAuthor artefacts, OvA greedy recomputed from ova.pkl:')
for k, v in out.items(): print(f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}')
print(by_team.to_string())
pd.Series({k: str(v) for k, v in out.items()}).to_csv(OUT / 'solver_confound_summary_author_ova_recomputed.csv')
by_team.to_csv(OUT / 'solver_confound_by_team_author_ova_recomputed.csv')
