"""Does the assignment confound hold at every cost ratio? Models retrained at each lambda;
OvA assigned with the released greedy rule (re-implemented from run_alert.ova_deferral_func) vs the exact LP;
DeCCaF assigned with the exact LP (equal to CP-SAT at optimality). One training testbed per team (shuffle_1). Requires the caches of 07_lambda_eval.py for every lambda."""
import os, sys
sys.argv = ['x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '07_lambda_eval.py'), encoding='utf-8').read().split('ok = {}')[0])
import numpy as np, pandas as pd

def greedy(env, P, decisions):
    """Released OvA rule: cases in batch order, each to the highest-scoring worker with capacity left."""
    caps, cases = envs[env]['caps'], envs[env]['cases']
    S = P.loc[cases, caps.index].values.astype(float); left = caps.values.copy()
    order = np.argsort(-S, axis=1, kind='stable'); a = np.empty(len(cases), int)
    for i in range(len(cases)):
        for j in order[i]:
            if left[j] > 0: left[j] -= 1; a[i] = j; break
    pred = decisions.loc[cases, caps.index].values[np.arange(len(cases)), a]; yy = y.loc[cases].values
    return ((pred == 0) & (yy == 1)).astype(float), ((pred == 1) & (yy == 0)).astype(float)

rng = np.random.default_rng(SEED); N = len(next(iter(envs.values()))['cases'])
Wb = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)
def ci(v): b = Wb @ v; return v.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)

rows = []
for lam in [0.02, 0.057, 0.1, 0.2, 0.5]:
    cache = OUT / f'cache_{lam:g}'
    h = pd.read_parquet(cache / 'h_prob.parquet')['h_prob']
    dec = experts.copy(); dec['classifier_h'] = (h.loc[alerts.index] >= 0.5).astype(int)
    for seeds_kind in (['shuffle_1'] if lam != 0.057 else ['shuffle_1', 'all']):
        L = {}
        for m in ['ova', 'deccaf']:
            for f in sorted(cache.glob(f'{m}__*.parquet')):
                seed = f.stem.split('__')[1]
                if seeds_kind == 'shuffle_1' and not seed.startswith('shuffle_1#'): continue
                P = pd.read_parquet(f); team = seed.split('#')[1]
                for env, e in envs.items():
                    if e['team'] != team: continue
                    for proc, fn in ([('greedy', greedy), ('LP', assign)] if m == 'ova' else [('LP', assign)]):
                        fnv, fpv = fn(env, P, dec); L[(m, proc, seed, env)] = fnv + lam * fpv
        def vec(m, proc):
            ks = sorted(k for k in L if k[0] == m and k[1] == proc); return np.mean([L[k] for k in ks], axis=0), len(ks)
        og, n = vec('ova', 'greedy'); ol, _ = vec('ova', 'LP'); dl, _ = vec('deccaf', 'LP')
        off, lp = ci(dl - og), ci(dl - ol)
        rows.append(dict(train_eval_lambda=lam, testbeds=seeds_kind, runs=n, OvA_greedy=og.sum(), OvA_LP=ol.sum(), DeCCaF_LP=dl.sum(),
                         diff_released=off[0], released_lo=off[1], released_hi=off[2], diff_sameLP=lp[0], sameLP_lo=lp[1], sameLP_hi=lp[2],
                         share_assignment=(1 - lp[0] / off[0]) if off[0] < 0 else np.nan))
        print(rows[-1], flush=True)
res = pd.DataFrame(rows).round(2)
print(res.to_string()); res.to_csv(OUT / 'solver_confound_by_lambda.csv', index=False)
