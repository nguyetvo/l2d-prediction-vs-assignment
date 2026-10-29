"""Retrain classifier h, OvA and DeCCaF with the authors' code at another cost ratio λ
(one training testbed per team: shuffle_1#team_1..5). Every stage resumes if restarted.
Usage: python src/05_lambda_retrain.py 0.5 [all] [deccaf_only]"""
import os, re, shutil, subprocess, sys, time, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
ALL = 'all' in sys.argv[2:]          # 'all' = the authors' five training testbeds per team (shuffle_1..5)
DECCAF_ONLY = 'deccaf_only' in sys.argv[2:]   # skip OvA (OvA takes ~10 h); the OvA vs DeCCaF comparison needs both
OFFICIAL = config.OFFICIAL
LEGACY_PY = config.LEGACY_PY
ROOT = config.lambda_dir(LAM)
WS = ROOT / 'openl2d'
TESTBEDS = [f'shuffle_{k}#team_{i}' for k in (range(1, 6) if ALL else [1]) for i in range(1, 6)]
MAX_ATTEMPTS = 5

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(ROOT / 'progress.log', 'a', encoding='utf-8') as f: f.write(line + '\n')

def junction(link, target):
    if link.exists(): return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':   # non-Windows: directory symlink instead of an NTFS junction
        link.symlink_to(target, target_is_directory=True); return
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)

def setup():
    if (WS / '.setup_done').exists(): return
    if (WS / 'Code').exists(): shutil.rmtree(WS / 'Code')          # only the copied code; junctions are never removed
    shutil.copytree(OFFICIAL / 'Code', WS / 'Code', ignore=shutil.ignore_patterns('__pycache__', '*.ipynb'))
    junction(WS / 'FiFAR/alert_data', OFFICIAL / 'FiFAR/alert_data')
    junction(WS / 'FiFAR/synthetic_experts', OFFICIAL / 'FiFAR/synthetic_experts')
    for t in TESTBEDS: junction(WS / 'FiFAR/testbed/train_alert' / t, OFFICIAL / 'FiFAR/testbed/train_alert' / t)
    (WS / 'FiFAR/classifier_h/models').mkdir(parents=True, exist_ok=True)
    junction(WS / 'l2d_benchmarking/classifier_h/models', WS / 'FiFAR/classifier_h/models')   # authors' path bug (writes FiFAR/, reads l2d_benchmarking/)
    p = WS / 'Code/alert_data/dataset_cfg.yaml'; t = p.read_text(encoding='utf-8')
    assert 'lambda: 0.057' in t; p.write_text(t.replace('lambda: 0.057', f'lambda: {LAM}'), encoding='utf-8')
    p = WS / 'Code/expert_models/run_deccaf.py'; t = p.read_text(encoding='utf-8')
    assert 'l = 0.057' in t; p.write_text(t.replace('l = 0.057', f'l = {LAM}'), encoding='utf-8')
    (WS / '.setup_done').write_text('ok')

def reduce_h_budget():
    """Classifier h: the authors run 2 search spaces x 10 initial scores x 300 trials (about 6 h). To reduce
    runtime we keep both search spaces but use 3 initial scores x 60 trials (40 random start-up trials);
    the authors use 300 trials (space 0, DART) and 1,700 trials (space 1).
    OvA and DeCCaF keep the authors' full budget. Idempotent."""
    for f in (WS / 'Code/classifier_h/param_spaces').glob('*.yaml'):
        t = f.read_text(encoding='utf-8')
        t = re.sub(r'(?m)^startups:\s*\d+', 'startups: 40', t); t = re.sub(r'(?m)^total:\s*\d+', 'total: 60', t); f.write_text(t, encoding='utf-8')
        assert 'total: 60' in t and 'startups: 40' in t
    f = WS / 'Code/classifier_h/training.py'; t = f.read_text(encoding='utf-8')
    t = t.replace('np.arange(init_train, init_train + 2, 0.2)', 'np.arange(init_train, init_train + 2, 0.7)')
    assert 'init_train + 2, 0.7' in t; f.write_text(t, encoding='utf-8')

def stage(name, script, done):
    if done(): log(f'{name}: already done'); return
    path = WS / 'Code' / script
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time(); log(f'{name}: attempt {attempt}/{MAX_ATTEMPTS}')
        with open(ROOT / f'{name}.log', 'a', encoding='utf-8') as f:
            r = subprocess.run([str(LEGACY_PY), path.name], cwd=str(path.parent), stdout=f, stderr=subprocess.STDOUT)
        log(f'{name}: return code {r.returncode}, {(time.time()-t0)/60:.1f} min')
        if done(): return
    raise RuntimeError(f'{name} failed {MAX_ATTEMPTS} times — see {ROOT / (name + ".log")}')

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True); log(f'λ = {LAM}')
    setup(); reduce_h_budget()
    for t in TESTBEDS: junction(WS / 'FiFAR/testbed/train_alert' / t, OFFICIAL / 'FiFAR/testbed/train_alert' / t)   # adds shuffles 2-5 when ALL
    B = WS / 'l2d_benchmarking'
    stage('classifier_h', 'classifier_h/training.py', lambda: (B / 'classifier_h/selected_model/model_properties.yaml').exists())
    def fresh(pkl, models):   # predictions exist and are newer than every trained model of every testbed
        f = B / 'deferral/l2d_predictions' / pkl
        return f.exists() and len(models) > 0 and all(m.exists() for m in models) and f.stat().st_mtime > max(m.stat().st_mtime for m in models)
    ova_models = lambda: [p for t in TESTBEDS for p in (B / 'expert_models/ova' / t).glob('*/best_model.pickle')] if all((B / 'expert_models/ova' / t).exists() for t in TESTBEDS) else []
    dec_models = lambda: [B / 'expert_models/deccaf' / t / 'best_model.pickle' for t in TESTBEDS]
    if not DECCAF_ONLY: stage('ova', 'expert_models/run_ova.py', lambda: fresh('ova.pkl', ova_models()) and len(ova_models()) == 10 * len(TESTBEDS))
    stage('deccaf', 'expert_models/run_deccaf.py', lambda: fresh('deccaf.pkl', dec_models()))
    log('all stages finished')
