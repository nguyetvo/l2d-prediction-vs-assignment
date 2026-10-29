# Separating Prediction Quality from Expert Assignment in Capacity-Constrained Learning to Defer

Code for the paper *"Separating Prediction Quality from Expert Assignment in Capacity-Constrained Learning to
Defer: A Replication and Assumption Audit"* (ICCIES 2027, Springer LNCS).

The code replicates the OpenL2D benchmark on the FiFAR dataset (Alves et al.; DeCCaF and OvA,
<https://github.com/feedzai/openl2d>) at two levels: re-running the authors' released artifacts and
retraining all models from scratch with the authors' code. It then

- compares OvA and DeCCaF under one common exact assignment (a transportation LP solved with HiGHS via
  `scipy.optimize.linprog`), instead of the released procedures (greedy for OvA, CP-SAT for DeCCaF);
- checks whether the released CP-SAT assignments reach the LP optimum;
- retrains OvA and DeCCaF at other cost ratios λ and compares re-weighted evaluation with retraining;
- audits benchmark assumptions: temporal drift, calibration of the expert-correctness estimates,
  oracle headroom and expert complementarity.

No data, trained models or results are included. Everything is regenerated from the OpenL2D repository,
the FiFAR dataset and the scripts below.

## Repository layout

```
config.py                         all paths (one root: L2D_ROOT)
requirements.txt                  analysis environment (Python 3.13)
requirements-legacy.txt           OpenL2D environment (Python 3.9)
src/
  00_setup_official_run.py        official workspace: clone, FiFAR data, released artifacts
  01_setup_retrain.py             clean workspace + retraining with the authors' scripts (λ = 0.057)
  02_assumption_checks.py         reproduction table, paired statistics, headroom, calibration, drift, λ re-evaluation
  03_gap_decomposition.py         gap decomposition between DeCCaF and the label oracle (v1)
  04_gap_decomposition_v2.py      gap decomposition with case weighting and equal-size controls (v2)
  05_lambda_retrain.py            retraining at another cost ratio λ
  06_lambda_export_legacy.py      helper run by the legacy interpreter: exports predictions to parquet
  07_lambda_eval.py               OvA vs DeCCaF, re-evaluated vs retrained at each λ, shared LP
  08_solver_confound.py           OvA vs DeCCaF under official procedures vs a shared exact LP
  09_cpsat_optimality.py          objective of released CP-SAT assignments vs the LP optimum
  10_lambda_greedy.py             assignment confound at every λ (greedy vs LP for OvA)
  figures/
    fig_overview.py               study overview diagram
    fig_tables_main.py            reproduction, drift, λ-sensitivity and headroom figures/tables
    fig_assignment_confound.py    assignment-confound figure
```

## Script to paper mapping

Table and figure numbers refer to the submitted manuscript.

| Paper artifact | Script(s) | Output (under `outputs/`) |
|---|---|---|
| Fig. 1 (study overview) | `figures/fig_overview.py` | `figures/F0_study_overview.{pdf,png}` |
| Table 2 (published vs released-artifact vs retrained loss by team) | `02_assumption_checks.py author`, `02_assumption_checks.py retrain`, then `figures/fig_tables_main.py` | `assumption_checks/<src>/9_vs_paper_table7.csv`, `tables/T1_reproduction_vs_paper.csv` |
| Fig. 2 (published vs reproduced loss) | `figures/fig_tables_main.py` | `figures/F1_reproduction_scatter.*` |
| Table 3 (paired differences, bootstrap CIs) | `02_assumption_checks.py retrain` | `assumption_checks/retrain/7_paired_statistics.csv`, `tables/T2_paired_statistics.csv` |
| Table 4 (assignment-controlled comparison) | `08_solver_confound.py author`, `08_solver_confound.py retrain` | `lambda_results/solver_confound_summary_<src>.csv`, `solver_confound_by_team_<src>.csv` |
| Fig. 3 (assignment confound) | `figures/fig_assignment_confound.py` | `figures/F5_assignment_confound.*` |
| CP-SAT optimality check (solver optimality subsection, Table 9) | `09_cpsat_optimality.py` | `lambda_results/cpsat_optimality_check.csv` |
| Table 5 (monthly counts, prevalence, expert error, score AUC) | `02_assumption_checks.py retrain` | `assumption_checks/retrain/5_monthly_stability.csv`, `5_feature_ks_shift.csv` |
| Fig. 4 (monthly drift) | `figures/fig_tables_main.py` | `figures/F2_monthly_drift.*` |
| Table 6 (re-evaluation at other λ, models fixed) | `02_assumption_checks.py retrain` | `assumption_checks/retrain/6_lambda_sensitivity.csv`, `tables/T3_lambda_sensitivity.csv` (also `figures/F3_lambda_sensitivity.*`) |
| Table 7 and Fig. 5 (re-evaluated vs retrained at each λ) | `05_lambda_retrain.py` (per λ), `07_lambda_eval.py` | `lambda_results/lambda_retrain_results.csv`, `figures/F3b_lambda_retrained.*` |
| Calibration (AUC/ECE of expert-correctness estimates, Sect. on calibration) | `02_assumption_checks.py retrain` | `assumption_checks/retrain/4_calibration_auc_ece.csv` |
| Table 8 (retrained losses and label oracle by team) | `02_assumption_checks.py retrain`, `figures/fig_tables_main.py` | `assumption_checks/retrain/2_oracle_headroom_by_env.csv`, `tables/T4_headroom_by_team.csv`, `figures/F4_headroom_by_team.*` |
| Table 9 (assumption summary) | summarises the outputs above; `VERDICT.json` of `02_assumption_checks.py` | `assumption_checks/<src>/VERDICT.json` |
| Key numbers quoted in the text | `figures/fig_tables_main.py` | `key_numbers.json` |
| Solver confound at every λ | `10_lambda_greedy.py` | `lambda_results/solver_confound_by_lambda.csv` (see script docstring) |
| Gap decomposition (same-LP ladder, recalibration, oracles, bootstrap) | `03_gap_decomposition.py`, `04_gap_decomposition_v2.py` | `gap_decomposition*/<src>/` (see script docstrings) |
| LP vs CP-SAT loss at λ = 0.057 (solver check) | `07_lambda_eval.py` | `lambda_results/solver_check_lp_vs_cpsat.csv` |

Table 1 of the paper (implementation obstacles) is descriptive and has no script.

## Prerequisites

1. **OpenL2D code**: <https://github.com/feedzai/openl2d> (cloned by `00_setup_official_run.py`, or pass an
   existing clone with `--repo`).
2. **FiFAR data**: obtain the dataset archive as described in the OpenL2D README. The setup script expects a ZIP
   whose FiFAR payload lies under a `.../FiFAR/FiFAR/` folder (the layout of the published archive) and copies it
   to `<workspace>/FiFAR`. Point to it with `L2D_FIFAR_ZIP` or `--data-zip`.
3. **Released artifacts**: `alert_model.zip` and `l2d_benchmarking.zip` from the OpenL2D repository (Git LFS);
   fetched by `00_setup_official_run.py`.
4. **Two Python environments**
   - *Legacy* (Python 3.9, `requirements-legacy.txt`): runs the authors' training/deferral scripts and loads their
     pickles (pandas 1.3.5, scikit-learn 0.24.2, LightGBM 3.2.1, OR-Tools 9.5). The authors' own environment is
     Python 3.7 (`openl2d-environment.yml` in their repository); 3.9 with the same package pins was used here.
   - *Analysis* (Python 3.13, `requirements.txt`): all scripts in `src/` except `06_lambda_export_legacy.py`.
     Scripts call the legacy interpreter themselves where needed.

Example (any virtual-environment tool works):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt          # analysis env
uv venv --python 3.9 workspace/legacy_env
uv pip install --python workspace/legacy_env/bin/python -r requirements-legacy.txt
```

## Configuration

All paths come from `config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `L2D_ROOT` | `./workspace` | root for workspaces and outputs |
| `L2D_LEGACY_PY` | `$L2D_ROOT/legacy_env/Scripts/python.exe` (Windows) or `.../bin/python` | legacy interpreter |
| `L2D_FIFAR_ZIP` | `$L2D_ROOT/fifar_data.zip` | FiFAR dataset archive |

```
$L2D_ROOT/official_run/openl2d/        official code + FiFAR + released models/results
$L2D_ROOT/retrain/openl2d/             independent retraining (λ = 0.057)
$L2D_ROOT/lambda_retrain/lambda_<λ>/   retraining at λ ∈ {0.02, 0.1, 0.2, 0.5}
$L2D_ROOT/outputs/                     all results, tables and figures
```

`python config.py` prints the resolved paths.

## Run order

Run from the repository root with the analysis interpreter.

```bash
# 1. Official workspace (+ optional re-run of the authors' deferral script on the released models)
python src/00_setup_official_run.py --rerun-deferral

# 2. Independent retraining at λ = 0.057 (classifier h, OvA, DeCCaF, deferral); many hours
python src/01_setup_retrain.py

# 3. Reproduction and assumption checks (author first, so that retrain adds the side-by-side comparison)
python src/02_assumption_checks.py author
python src/02_assumption_checks.py retrain

# 4. Gap decomposition (optional, supplementary)
python src/03_gap_decomposition.py retrain
python src/04_gap_decomposition_v2.py retrain

# 5. Retraining at other cost ratios (one training testbed per team; add "all" for the five testbeds)
python src/05_lambda_retrain.py 0.02
python src/05_lambda_retrain.py 0.1
python src/05_lambda_retrain.py 0.2
python src/05_lambda_retrain.py 0.5

# 6. λ comparison with a shared LP, solver confound, CP-SAT optimality
python src/07_lambda_eval.py
python src/08_solver_confound.py author
python src/08_solver_confound.py retrain
python src/09_cpsat_optimality.py        # needs caches from 07 (cache_0.057) and 08 author (cache_author)
python src/10_lambda_greedy.py           # needs caches for every λ from 07

# 7. Figures and tables
python src/figures/fig_overview.py
python src/figures/fig_tables_main.py
python src/figures/fig_assignment_confound.py
```

`06_lambda_export_legacy.py` is not run directly; `07` and `08` call it with the legacy interpreter.
`08`, `09` and `10` execute the setup part of `07_lambda_eval.py` (everything before its export loop) via `exec`.

## Fixed settings

- Cost ratio λ = 0.057 (cost of a false positive relative to a false negative, as in OpenL2D); λ grid
  {0.02, 0.057, 0.1, 0.2, 0.5}.
- Test month 7, validation month 6; 25 test scenarios (5 teams × 5 capacity settings).
- Bootstrap: 1000 multinomial case-weight resamples, `numpy.random.default_rng(42)`; percentile 95% CIs.
- Assignment LP: each case to exactly one worker, each worker exactly its capacity; `linprog(method='highs')`,
  with a check that the solution is integral.

## Runtime notes

- Retraining with the authors' full budget takes many hours per stage (`01_setup_retrain.py` allows 24 h per
  stage; the OvA stage takes about 10 h). Stages resume from saved models when restarted.
- `05_lambda_retrain.py` reduces only the classifier-h search (3 initial scores × 60 trials instead of
  10 × 300/1,700) and keeps the authors' full budget for OvA and DeCCaF; see its docstring.
- The analysis scripts use 6 parallel workers (`joblib`) for the LP assignments; `03`/`04` also fit gradient
  boosting models for the re-estimated correctness levels.

## Citation

Original benchmark and methods (please cite when using this code):

```bibtex
@article{alves2024deccaf,
  author  = {Alves, Jean V. and Leit{\~a}o, Diogo and Jesus, S{\'e}rgio and Sampaio, Marco O. P. and
             Li{\'e}bana, Javier and Saleiro, Pedro and Figueiredo, M{\'a}rio A. T. and Bizarro, Pedro},
  title   = {Cost-Sensitive Learning to Defer to Multiple Experts with Workload Constraints},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}

@article{alves2025openl2d,
  author  = {Alves, Jean V. and Leit{\~a}o, Diogo and Jesus, S{\'e}rgio and Sampaio, Marco O. P. and
             Li{\'e}bana, Javier and Saleiro, Pedro and Figueiredo, M{\'a}rio A. T. and Bizarro, Pedro},
  title   = {A Benchmarking Framework and Dataset for Learning to Defer in Human-{AI} Decision-Making},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {506},
  year    = {2025},
  doi     = {10.1038/s41597-025-04664-y}
}

@misc{alves2023fifar,
  author       = {Alves, Jean V. and Leit{\~a}o, Diogo and Jesus, S{\'e}rgio and Sampaio, Marco O. P. and
                  Saleiro, Pedro and Figueiredo, M{\'a}rio A. T. and Bizarro, Pedro},
  title        = {{FiFAR}: A Fraud Detection Dataset for Learning to Defer},
  howpublished = {arXiv preprint arXiv:2312.13218},
  year         = {2023}
}
```

This paper: citation to be added after publication.

## License

MIT (see `LICENSE`). OpenL2D and FiFAR are distributed under their own licenses.
