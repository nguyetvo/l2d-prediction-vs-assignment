"""Prepare the official OpenL2D workspace at config.OFFICIAL.

Steps (each is skipped if already done):
 1. clone github.com/feedzai/openl2d (or copy nothing and use --repo to point at an existing clone);
 2. copy the FiFAR payload (the folder of the dataset ZIP that holds alert_data/) into <workspace>/FiFAR;
 3. fetch the authors' released artifacts (alert_model.zip, l2d_benchmarking.zip) and extract them without
    overwriting existing files;
 4. move the shipped deferral results to l2d_benchmarking/deferral/results_author_official
    (run_alert.py skips every scenario whose results folder already exists);
 5. optional (--rerun-deferral): re-run Code/deferral/run_alert.py with the legacy interpreter on the released
    models; results go to l2d_benchmarking/deferral/results.

Usage: python src/00_setup_official_run.py [--data-zip PATH] [--repo PATH] [--rerun-deferral]
"""
import argparse, shutil, subprocess, sys, zipfile, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ARTIFACTS = {'alert_model.zip': 'https://media.githubusercontent.com/media/feedzai/openl2d/main/alert_model.zip',
             'l2d_benchmarking.zip': 'https://media.githubusercontent.com/media/feedzai/openl2d/main/l2d_benchmarking.zip'}

ap = argparse.ArgumentParser()
ap.add_argument('--data-zip', type=Path, default=config.FIFAR_ZIP, help='FiFAR dataset archive')
ap.add_argument('--repo', type=Path, default=None, help='existing openl2d clone to copy into the workspace')
ap.add_argument('--rerun-deferral', action='store_true', help='re-run the authors\' deferral script on released models')
args = ap.parse_args()
REPO = config.OFFICIAL
LOGS = config.SETUP_LOGS; LOGS.mkdir(parents=True, exist_ok=True)

# 1. repository
if not (REPO / 'Code/deferral/run_alert.py').exists():
    REPO.parent.mkdir(parents=True, exist_ok=True)
    if args.repo is not None:
        shutil.copytree(args.repo, REPO, ignore=shutil.ignore_patterns('.git'))
    else:
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/feedzai/openl2d.git', str(REPO)], check=True)
print('repository:', REPO)

# 2. FiFAR data
root = REPO / 'FiFAR'
assert args.data_zip.exists(), f'Missing dataset archive: {args.data_zip} (set L2D_FIFAR_ZIP or --data-zip)'
with zipfile.ZipFile(args.data_zip) as z:
    # payload root = folder holding alert_data/ (e.g. 'FiFAR/' or '.../FiFAR/FiFAR/', depending on the download)
    anchor = [n for n in z.namelist() if n.endswith('alert_data/processed_data/alerts.parquet')]
    assert len(anchor) == 1, f'Expected one alert_data/processed_data/alerts.parquet in the data ZIP, found {len(anchor)}'
    prefix = anchor[0][:-len('alert_data/processed_data/alerts.parquet')]
    members = [i for i in z.infolist() if i.filename.startswith(prefix)]
    copied = 0
    for item in members:
        rel = item.filename[len(prefix):]
        if not rel or item.is_dir(): continue
        dest = root / rel
        if dest.exists() and dest.stat().st_size == item.file_size: continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(item) as src, open(dest, 'wb') as dst: shutil.copyfileobj(src, dst)
        copied += 1
print('FiFAR files copied:', copied)
for p in ['alert_data/processed_data/alerts.parquet', 'synthetic_experts/expert_predictions.parquet', 'testbed/test']:
    assert (root / p).exists(), f'missing {root / p}'

# 3. released artifacts (never overwrite existing files)
for fname, url in ARTIFACTS.items():
    path = REPO / fname
    # a clone without Git LFS holds only a small pointer file in place of the archive
    if path.exists() and path.stat().st_size < 1024 and path.read_bytes().startswith(b'version https://git-lfs'):
        path.unlink()
    if not path.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=90) as resp, open(path, 'wb') as target: shutil.copyfileobj(resp, target)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for m in z.infolist():
            if m.is_dir(): continue
            rel = Path(m.filename)
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe ZIP path')
            target = REPO / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, 'wb') as dst: shutil.copyfileobj(src, dst)
    print('artifact extracted:', fname)

# 4. keep the shipped results apart
RESULTS = REPO / 'l2d_benchmarking/deferral/results'
AUTHOR_RESULTS = REPO / 'l2d_benchmarking/deferral/results_author_official'
if RESULTS.exists() and not AUTHOR_RESULTS.exists():
    shutil.move(str(RESULTS), str(AUTHOR_RESULTS)); print('shipped results moved to', AUTHOR_RESULTS)

# 5. optional re-run of the authors' deferral script (from its own directory, as the OpenL2D README requires)
if args.rerun_deferral:
    assert config.LEGACY_PY.exists(), f'Missing legacy interpreter {config.LEGACY_PY} (set L2D_LEGACY_PY)'
    script = REPO / 'Code/deferral/run_alert.py'
    with open(LOGS / 'official_deferral.log', 'w', encoding='utf-8', errors='replace') as f:
        r = subprocess.run([str(config.LEGACY_PY), script.name], cwd=str(script.parent), stdout=f, stderr=subprocess.STDOUT)
    print('run_alert.py return code:', r.returncode, '| log:', LOGS / 'official_deferral.log')
