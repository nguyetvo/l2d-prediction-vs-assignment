"""Assignment-confound figure (OvA/DeCCaF under official procedures vs a shared exact LP).
Inputs: solver_confound_summary_{author,retrain}.csv from 08_solver_confound.py."""
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
R = config.LAMBDA_OUT; FIG = config.FIG_DIR; FIG.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
C = {'OvA': '#eb6834', 'DeCCaF': '#2a78d6'}
for ax, (src, title) in zip(axes, [('author', "(a) Authors' artifacts"), ('retrain', '(b) Independent retraining')]):
    s = pd.read_csv(R / f'solver_confound_summary_{src}.csv', index_col=0).iloc[:, 0]
    v = {k: float(s[k]) for k in ['OvA official greedy', 'OvA exact LP', 'DeCCaF official CP-SAT', 'DeCCaF exact LP']}
    rows = [('OvA', 'greedy (official)', v['OvA official greedy'], 'exact LP', v['OvA exact LP']),
            ('DeCCaF', 'CP-SAT (official)', v['DeCCaF official CP-SAT'], 'exact LP', v['DeCCaF exact LP'])]
    for i, (m, l1, a, l2, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color='#e3e3e3', lw=3, zorder=1)
        ax.scatter(a, i, s=46, marker='o', facecolor='white', edgecolor=C[m], linewidth=1.8, zorder=3)
        ax.scatter(b, i, s=46, marker='o', color=C[m], edgecolor='white', linewidth=0.8, zorder=3)
        ax.annotate(f'{a:.1f}', (a, i), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7, color='#1f1f1f')
        ax.annotate(f'{b:.1f}', (b, i), xytext=(0, -12), textcoords='offset points', ha='center', fontsize=7, color='#1f1f1f')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['OvA', 'DeCCaF']); ax.set_ylim(-0.7, 1.7); ax.invert_yaxis()
    ax.set_title(title, fontsize=8.5, loc='left'); ax.set_xlabel('Mean test loss (lower is better)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', color='#e3e3e3', lw=0.6); ax.tick_params(labelsize=7.5)
    ax.text(0.98, 0.02, f"DeCCaF−OvA\nofficial: {s['DeCCaF - OvA, official procedures']}\nsame LP: {s['DeCCaF - OvA, same LP for both']}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3, color='#6b6b6b')
h1 = plt.Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='#6b6b6b', label="official assignment")
h2 = plt.Line2D([], [], marker='o', ls='', color='#6b6b6b', label='same exact LP for both')
fig.legend(handles=[h1, h2], loc='upper center', ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
for ext in ('pdf', 'png'): fig.savefig(FIG / f'F5_assignment_confound.{ext}', dpi=300, bbox_inches='tight')
print('ok')
