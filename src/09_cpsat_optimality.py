"""Was the released CP-SAT assignment optimal? Compare the objective (sum of predicted scores) of the
saved DeCCaF assignments with the exact LP optimum on the same scores, for author artifacts and retrained models.
Requires the prediction caches written by 07_lambda_eval.py (cache_0.057) and 08_solver_confound.py author (cache_author)."""
import os, sys
sys.argv = ['x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '07_lambda_eval.py'), encoding='utf-8').read().split('ok = {}')[0])
import numpy as np, pandas as pd
SRCS = {'author': (OFFICIAL / 'l2d_benchmarking/deferral/results_author_official', OUT / 'cache_author'),
        'retrain': (RETRAIN / 'l2d_benchmarking/deferral/results', OUT / 'cache_0.057')}
rows = []
for src, (resdir, cache) in SRCS.items():
    for f in sorted(cache.glob('deccaf__*.parquet')):
        P = pd.read_parquet(f); seed = f.stem.split('__')[1]; team = seed.split('#')[1]
        for env, e in envs.items():
            if e['team'] != team: continue
            caps, cases = e['caps'], e['cases']
            S_ = P.loc[cases, caps.index].values.astype(float)
            a_lp = solve_transport(-S_.T, caps.values); obj_lp = S_[np.arange(len(cases)), a_lp].sum()
            asg = pd.read_parquet(resdir / 'DeCCaF' / seed / env / 'assignments.parquet').iloc[:, 0].reindex(cases).astype(str)
            col = {w: k for k, w in enumerate(caps.index)}
            a_cp = asg.map(col).values.astype(int); obj_cp = S_[np.arange(len(cases)), a_cp].sum()
            used = np.bincount(a_cp, minlength=len(caps)); feas = bool((used == caps.values).all())
            rows.append(dict(src=src, seed=seed, env=env, obj_cpsat=obj_cp, obj_lp=obj_lp, rel_gap=(obj_lp - obj_cp) / obj_lp, cap_match=feas))
r = pd.DataFrame(rows); r.to_csv(OUT / 'cpsat_optimality_check.csv', index=False)
print(r.groupby('src').agg(n=('env', 'size'), mean_gap=('rel_gap', 'mean'), max_gap=('rel_gap', 'max'),
      n_gap_gt_1e4=('rel_gap', lambda s: (s > 1e-4).sum()), caps_ok=('cap_match', 'mean')).to_string())
