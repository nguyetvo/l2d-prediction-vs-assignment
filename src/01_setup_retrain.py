"""Independent retraining at lambda = 0.057 in a clean workspace (config.RETRAIN).

The official workspace is copied without the authors' trained models and results (l2d_benchmarking/, alert_model/
and their ZIPs), because the OpenL2D scripts skip training when those folders exist. Then the authors' scripts run
unchanged with the legacy interpreter, each from its own directory:
classifier_h/training.py -> expert_models/run_ova.py -> expert_models/run_deccaf.py -> deferral/run_alert.py.
A stage is retried (fresh process; the scripts resume from saved models) only after out-of-memory/native crashes.
Hyper-parameter search is long: allow many hours per stage (timeout 24 h per stage).

Usage: python src/01_setup_retrain.py
"""
import os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SOURCE_REPO, WORK_REPO, LEGACY_PY = config.OFFICIAL, config.RETRAIN, config.LEGACY_PY
LOGS = config.SETUP_LOGS; LOGS.mkdir(parents=True, exist_ok=True)
STAGE_TIMEOUT = 24 * 3600
MAX_ATTEMPTS = 5
AUTHOR_ARTIFACTS = {'l2d_benchmarking', 'alert_model', 'l2d_benchmarking.zip', 'alert_model.zip'}
CLEAN_MARKER = WORK_REPO / '.clean_workspace_ok'   # once present, l2d_benchmarking/ holds the retrain outputs


def ignore_repo(dirname, names):
    if Path(dirname) == SOURCE_REPO:
        return [x for x in names if x in AUTHOR_ARTIFACTS | {'.git', '__pycache__', '.ipynb_checkpoints'}]
    return [x for x in names if x in {'__pycache__', '.ipynb_checkpoints'}]


def junction(link, target):
    if link.exists(): return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':
        link.symlink_to(target, target_is_directory=True); return
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)


def run_stage(name, script):
    path = WORK_REPO / 'Code' / script
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log = LOGS / f'retrain_{name}.log'
        if log.exists(): log.rename(LOGS / f'retrain_{name}_prev_{datetime.now():%Y%m%d_%H%M%S}.log')
        print(f'{name}: attempt {attempt}/{MAX_ATTEMPTS} (log: {log})', flush=True); t0 = time.time()
        with open(log, 'w', encoding='utf-8', errors='replace') as f:
            r = subprocess.run([str(LEGACY_PY), path.name], cwd=path.parent, stdout=f, stderr=subprocess.STDOUT, timeout=STAGE_TIMEOUT)
        print(f'{name}: return code {r.returncode}, {(time.time() - t0) / 60:.1f} min', flush=True)
        if r.returncode == 0: return True
        tail = log.read_text(encoding='utf-8', errors='replace')[-750:]
        retryable = ('bad allocation' in tail) or ('MemoryError' in tail) or (r.returncode < 0) or (r.returncode > 255)
        if not retryable: print(tail); return False
        time.sleep(30)
    return False


if __name__ == '__main__':
    for p in [SOURCE_REPO / 'Code/classifier_h/training.py', SOURCE_REPO / 'FiFAR', LEGACY_PY]:
        assert p.exists(), f'missing {p} (run 00_setup_official_run.py / create the legacy env first)'
    if not CLEAN_MARKER.exists():
        if not WORK_REPO.exists():
            WORK_REPO.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SOURCE_REPO, WORK_REPO, ignore=ignore_repo)
        leftovers = [n for n in AUTHOR_ARTIFACTS if (WORK_REPO / n).exists()]
        assert not leftovers, f'Author artifacts in the retrain workspace: {leftovers}'
        CLEAN_MARKER.write_text(datetime.now().isoformat(), encoding='utf-8')
    # classifier_h/training.py writes models under FiFAR/classifier_h/models but later reads them from
    # l2d_benchmarking/classifier_h/models: link the two locations.
    (WORK_REPO / 'FiFAR/classifier_h/models').mkdir(parents=True, exist_ok=True)
    junction(WORK_REPO / 'l2d_benchmarking/classifier_h/models', WORK_REPO / 'FiFAR/classifier_h/models')
    ok = run_stage('classifier_h', 'classifier_h/training.py')
    ok = ok and run_stage('ova', 'expert_models/run_ova.py')
    ok = ok and run_stage('deccaf', 'expert_models/run_deccaf.py')
    ok = ok and run_stage('deferral', 'deferral/run_alert.py')
    print('retraining finished' if ok else 'retraining stopped: see logs in ' + str(LOGS))
