"""Single source of paths for every script in src/.

All locations derive from one root directory, set with the environment variable L2D_ROOT
(default: ./workspace next to this file). Importing this module has no side effects:
no directory is created here.

Layout under ROOT:
    official_run/openl2d/          clone of feedzai/openl2d + FiFAR data + released artifacts
    retrain/openl2d/               clean copy of the official code, retrained from scratch (lambda = 0.057)
    lambda_retrain/lambda_<l>/     retraining workspaces at other cost ratios (openl2d/ inside)
    legacy_env/                    Python 3.9 environment with the OpenL2D package versions
    outputs/                       everything produced by src/
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('L2D_ROOT', REPO_DIR / 'workspace')).resolve()

# OpenL2D workspaces
OFFICIAL = ROOT / 'official_run' / 'openl2d'      # authors' code, data and released models/results
RETRAIN = ROOT / 'retrain' / 'openl2d'            # independent retraining at lambda = 0.057
LAMBDA_ROOT = ROOT / 'lambda_retrain'


def lambda_dir(lam):
    """Working directory (logs + workspace) of the retraining run at cost ratio `lam`."""
    return LAMBDA_ROOT / f'lambda_{lam:g}'


def lambda_workspace(lam):
    """OpenL2D workspace retrained at cost ratio `lam`."""
    return lambda_dir(lam) / 'openl2d'


# Legacy interpreter that runs the OpenL2D code and loads its pickles (Python 3.9, requirements-legacy.txt)
LEGACY_ENV = ROOT / 'legacy_env'
_LEGACY_DEFAULT = LEGACY_ENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
LEGACY_PY = Path(os.environ.get('L2D_LEGACY_PY', _LEGACY_DEFAULT))

# FiFAR dataset archive (as obtained following the OpenL2D README)
FIFAR_ZIP = Path(os.environ.get('L2D_FIFAR_ZIP', ROOT / 'fifar_data.zip'))

# Outputs
OUTPUTS = ROOT / 'outputs'
CHECKS_OUT = OUTPUTS / 'assumption_checks'          # <source>/ with source in {author, retrain}
GAP_OUT = OUTPUTS / 'gap_decomposition'             # <source>/ (+ shared prediction cache)
GAP2_OUT = OUTPUTS / 'gap_decomposition_v2'         # <source>/
LAMBDA_OUT = OUTPUTS / 'lambda_results'             # lambda grid, solver confound, CP-SAT check
FIG_DIR = OUTPUTS / 'figures'
TAB_DIR = OUTPUTS / 'tables'
SETUP_LOGS = OUTPUTS / 'setup_logs'

if __name__ == '__main__':
    for k in ['ROOT', 'OFFICIAL', 'RETRAIN', 'LAMBDA_ROOT', 'LEGACY_PY', 'FIFAR_ZIP', 'OUTPUTS',
              'CHECKS_OUT', 'GAP_OUT', 'GAP2_OUT', 'LAMBDA_OUT', 'FIG_DIR', 'TAB_DIR']:
        print(f'{k:12s} {globals()[k]}')
