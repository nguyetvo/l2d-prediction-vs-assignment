"""Runs in legacy_env. Exports OvA/DeCCaF month-7 prediction tables (testbeds shuffle_1#team_*) and classifier-h
probabilities of one workspace to parquet. Called by 07_lambda_eval.py and 08_solver_confound.py.
Usage: <legacy python> src/06_lambda_export_legacy.py REPO OUT"""
import pickle, sys, yaml, numpy as np, pandas as pd
from pathlib import Path
repo, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
A = pd.read_parquet(repo / 'FiFAR/alert_data/processed_data/alerts.parquet')
B = repo / 'l2d_benchmarking'
h = pickle.load(open(B / 'classifier_h/selected_model/best_model.pickle', 'rb'))
hp = yaml.safe_load(open(B / 'classifier_h/selected_model/model_properties.yaml'))
p = 1 / (1 + np.exp(-(h.predict(A.drop(columns=['fraud_bool', 'model_score', 'month']), raw_score=True) + hp['init_score'])))
pd.DataFrame({'h_prob': p}, index=A.index).to_parquet(out / 'h_prob.parquet')
for m in ['ova', 'deccaf']:
    if not (B / f'deferral/l2d_predictions/{m}.pkl').exists(): continue   # e.g. DeCCaF-only workspaces
    pk = pickle.load(open(B / f'deferral/l2d_predictions/{m}.pkl', 'rb'))
    for s, df in pk.items():
        df.astype(float).to_parquet(out / f'{m}__{s}.parquet')
print('exported', repo)
