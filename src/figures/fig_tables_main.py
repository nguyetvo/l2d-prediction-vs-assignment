"""Figures and tables for the paper, built from existing result files (nothing is recomputed).
Inputs: outputs of 02_assumption_checks.py for both sources (author, retrain)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
AC = config.CHECKS_OUT
OUT = config.OUTPUTS; FIG = config.FIG_DIR; TAB = config.TAB_DIR
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# palette: fixed categorical order (blue, orange, aqua, yellow, magenta) + markers so print/greyscale still works
C = {'DeCCaF': '#2a78d6', 'OvA': '#eb6834', 'Random': '#1baf7a', 'Only_Classifier': '#eda100', 'LabelOracle': '#e87ba4'}
MK = {'DeCCaF': 'o', 'OvA': 's', 'Random': '^', 'Only_Classifier': 'D', 'Full_Rej': 'v', 'LabelOracle': 'P'}
INK, MUTED, GRID = '#1f1f1f', '#6b6b6b', '#e3e3e3'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
def save(fig, name):
    for ext in ('pdf', 'png'): fig.savefig(FIG / f'{name}.{ext}')
    plt.close(fig)

# ---------- Table 1 + Fig 1: reproduction against the published Table 7
ret = pd.read_csv(AC / 'retrain/9_vs_paper_table7.csv'); aut = pd.read_csv(AC / 'author/9_vs_paper_table7.csv')
t1 = ret[['team', 'method', 'paper']].copy()
t1['rerun_author_artifacts'] = aut['this_run'].values; t1['independent_retrain'] = ret['this_run'].values
t1['retrain_rel_diff_%'] = (100 * ret['rel_diff']).round(1); t1['retrain_within_paper_CI'] = ret['within_paper_ci']
t1.to_csv(TAB / 'T1_reproduction_vs_paper.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True, sharey=True)
for ax, (col, title) in zip(axes, [('rerun_author_artifacts', '(a) Re-run of the authors\' artifacts'), ('independent_retrain', '(b) Independent retraining')]):
    lo, hi = 100, 220
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls='--', zorder=1)
    for m, g in t1.groupby('method'):
        ax.scatter(g['paper'], g[col], s=28, marker=MK[m], color=C.get(m, '#5c5c5c'), edgecolor='white', linewidth=0.8, zorder=3,
                   label=m.replace('_', ' ').replace('Full Rej', 'Full rejection'))
    ax.set_title(title, fontsize=8.5, color=INK, loc='left'); ax.set_xlabel('Published loss ([3], Table 7)')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
axes[0].set_ylabel('Reproduced loss')
axes[1].legend(loc='lower right', fontsize=7, handletextpad=0.3)
fig.tight_layout(); save(fig, 'F1_reproduction_scatter')

# ---------- Table 2: paired statistics (retrain)
ps = pd.read_csv(AC / 'retrain/7_paired_statistics.csv')
ps[['A', 'B', 'n_pairs', 'mean_diff', 'ci_low', 'ci_high', 'share_A_better', 'wilcoxon_p_env']].round(4).to_csv(TAB / 'T2_paired_statistics.csv', index=False)

# ---------- Fig 2: monthly drift (two panels, one axis each)
ms = pd.read_csv(AC / 'retrain/5_monthly_stability.csv')
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
for ax, (col, title) in zip(axes, [('fraud_rate', '(a) Fraud rate among alerts'), ('expert_error_mean', '(b) Mean expert error rate')]):
    ax.plot(ms['month'], 100 * ms[col], color=C['DeCCaF'], lw=2, marker='o', ms=5, markeredgecolor='white')
    for m, v in zip(ms['month'], 100 * ms[col]): ax.annotate(f'{v:.1f}%', (m, v), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7, color=INK)
    ax.set_title(title, fontsize=8.5, color=INK, loc='left'); ax.set_xticks(ms['month']); ax.set_xlabel('Month')
    ax.set_ylabel('%'); ax.margins(y=0.25)
    ax.axvspan(6.5, 7.5, color=GRID, alpha=0.6, zorder=0); ax.text(7, ax.get_ylim()[0], 'test', ha='center', va='bottom', fontsize=7, color=MUTED)
fig.tight_layout(); save(fig, 'F2_monthly_drift')

# ---------- Fig 3: lambda sensitivity, loss relative to DeCCaF (one axis, ratio)
ls = pd.read_csv(AC / 'retrain/6_lambda_sensitivity.csv')
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.axhline(1, color=MUTED, lw=1, ls='--'); ax.text(0.021, 1.006, 'DeCCaF = 1', fontsize=7, color=MUTED, va='bottom')
for m in ['OvA', 'Random']:
    r = ls[m] / ls['DeCCaF']
    ax.plot(ls['lam'], r, color=C[m], lw=2, marker=MK[m], ms=5, markeredgecolor='white', label=m)
    ax.annotate(f'{r.iloc[-1]:.3f}', (ls['lam'].iloc[-1], r.iloc[-1]), textcoords='offset points', xytext=(4, 0), va='center', fontsize=7, color=INK)
ax.axvline(0.057, color=GRID, lw=3, zorder=0); ax.text(0.059, ax.get_ylim()[1], 'λ used in\nthe benchmark', fontsize=6.5, color=MUTED, va='top')
ax.set_xscale('log'); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks(ls['lam']); ax.set_xticklabels([f'{v:g}' for v in ls['lam']])
ax.set_xlabel('Cost ratio λ (log scale)'); ax.set_ylabel('Loss relative to DeCCaF'); ax.legend(loc='upper right', fontsize=7)
fig.tight_layout(); save(fig, 'F3_lambda_sensitivity')
ls.round(2).to_csv(TAB / 'T3_lambda_sensitivity.csv', index=False)

# ---------- Fig 4: headroom per team (retrain), horizontal dot plot
hr = pd.read_csv(AC / 'retrain/2_oracle_headroom_by_env.csv')
g = hr.groupby('team')[['Random', 'OvA', 'DeCCaF', 'LabelOracle']].mean()
fig, ax = plt.subplots(figsize=(4.2, 2.4))
for i, (team, row) in enumerate(g.iterrows()):
    ax.plot([row['LabelOracle'], row['Random']], [i, i], color=GRID, lw=2, zorder=1)
    for m in ['Random', 'OvA', 'DeCCaF', 'LabelOracle']:
        ax.scatter(row[m], i, s=36, marker=MK[m], color=C[m], edgecolor='white', linewidth=0.8, zorder=3, label=('Label oracle' if m == 'LabelOracle' else m) if i == 0 else None)
ax.set_yticks(range(len(g))); ax.set_yticklabels([t.replace('_', ' ') for t in g.index]); ax.invert_yaxis()
ax.set_xlabel('Mean loss on the test batch (lower is better)'); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7)
ax.grid(axis='y', visible=False)
fig.tight_layout(); save(fig, 'F4_headroom_by_team')
g.round(2).to_csv(TAB / 'T4_headroom_by_team.csv')

# ---------- key numbers for the text
v_r = json.load(open(AC / 'retrain/VERDICT.json')); v_a = json.load(open(AC / 'author/VERDICT.json'))
cal = pd.read_csv(AC / 'retrain/4_calibration_auc_ece.csv'); cal = cal[cal['worker'] != 'classifier_h'].groupby('method')[['auc', 'ece']].mean()
comp = pd.read_csv(AC / 'retrain/3_complementarity_by_team.csv'); ks = pd.read_csv(AC / 'retrain/5_feature_ks_shift.csv')
key = {'retrain_within_paper_CI': f"{int(t1['retrain_within_paper_CI'].sum())}/{len(t1)}",
       'author_artifacts_within_paper_CI': f"{int(aut['within_paper_ci'].sum())}/{len(aut)}",
       'mean_loss_retrain': {m: round(float(t1.loc[t1.method == m, 'independent_retrain'].mean()), 1) for m in t1.method.unique()},
       'mean_loss_paper': {m: round(float(t1.loc[t1.method == m, 'paper'].mean()), 1) for m in t1.method.unique()},
       'gap_to_label_oracle_rel': {'retrain': v_r['A2_headroom']['mean_rel_gap_DeCCaF_vs_LabelOracle'], 'author': v_a['A2_headroom']['mean_rel_gap_DeCCaF_vs_LabelOracle']},
       'auc_ece_experts': cal.round(3).to_dict('index'),
       'rescuable_share_range': [round(float(comp['rescuable_share'].min()), 3), round(float(comp['rescuable_share'].max()), 3)],
       'features_ks_ge_0.1': int((ks['ks'] >= 0.1).sum()), 'top_ks': ks.head(3).round(3).to_dict('records'),
       'lambda_0.5': ls[ls['lam'] == 0.5][['DeCCaF', 'OvA']].round(1).to_dict('records')[0],
       'solver_check': {'official_cp_sat': 130.9, 'exact_lp': 131.0}}
(OUT / 'key_numbers.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
print(json.dumps(key, indent=1)); print('figures:', sorted(p.name for p in FIG.glob('*.png')))
