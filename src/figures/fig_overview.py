"""Fig. 1 (study overview): recalculation from released results, retraining, solver-controlled comparison, assumption audit."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
OUT = config.FIG_DIR; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = '#1a1a1a', '#666666'
fig, ax = plt.subplots(figsize=(11, 4.0)); ax.set_xlim(0, 110); ax.set_ylim(0, 40); ax.axis('off')

def box(x, y, w, h, color, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='white', ec=color, lw=2.6))
    ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='top', fontsize=10.5, weight='bold', color=INK)
    ax.text(x + w / 2, y + h - 8.2, body, ha='center', va='top', fontsize=8.6, color=MUTED, linespacing=1.45)
def arrow(x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.8, mutation_scale=14))

box(1, 13, 16, 14, '#6b6b6b', 'FiFAR data', '30,622 alerts\n50 synthetic experts\n25 test scenarios')
box(22, 22, 22, 14, '#2a76d2', 'Level 1: recalculation', "authors' released result\nand prediction files\n(no retraining, no re-run)")
box(22, 4, 22, 14, '#e8673a', 'Level 2: retrain', 'classifier h, 250 OvA,\n25 DeCCaF (TPE, 120 trials)\n+ released assignment')
box(49, 13, 17, 14, '#1a1a1a', 'Reproduction', 'published Table 7\npaired bootstrap CIs')
box(71, 13, 17, 14, '#8a4fd0', 'Solver control', 'same exact LP\nfor OvA and DeCCaF')
box(93, 1, 16, 38, '#1aa67a', 'Assumption\naudit', 'temporal drift\n\ncost ratio λ:\nre-evaluated and\nretrained at\n0.02, 0.1, 0.2, 0.5\n\ncalibration\n\noracle headroom\ncomplementarity')
ax.texts[-1].set_position((101, 39 - 11))
arrow(17, 22, 22, 28); arrow(17, 18, 22, 11)
arrow(44, 28, 49, 22); arrow(44, 11, 49, 18)
arrow(66, 20, 71, 20); arrow(88, 20, 93, 20)
fig.tight_layout(pad=0.2)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'F0_study_overview.{ext}', dpi=300, bbox_inches='tight')
print('ok')
