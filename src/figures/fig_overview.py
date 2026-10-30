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
# drawn at about the LNCS text width so that the fonts stay legible after scaling
fig, ax = plt.subplots(figsize=(6.8, 2.15)); ax.set_xlim(0, 110); ax.set_ylim(0, 40); ax.axis('off')

def box(x, y, w, h, color, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='white', ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=7.8, weight='bold', color=INK, linespacing=1.1)
    ax.text(x + w / 2, y + h - 3.6 - 2.4 * (title.count('\n') + 1), body, ha='center', va='top', fontsize=6.8,
            color=MUTED, linespacing=1.25)
def arrow(x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.1, mutation_scale=9))

box(0.5, 10, 16, 20, '#6b6b6b', 'FiFAR data', '30,622 alerts\n50 synthetic\nexperts\n25 test scenarios')
box(20.5, 21, 27, 18, '#2a76d2', 'Level 1: recalculation', "released result and\nprediction files\n(no retraining, no re-run)")
box(20.5, 1, 27, 18, '#e8673a', 'Level 2: retrain', 'classifier h, 250 OvA,\n25 DeCCaF (TPE, 120 trials)\n+ released assignment')
box(51.5, 11, 16, 18, '#1a1a1a', 'Reproduction', 'published\nTable 7\npaired\nbootstrap CIs')
box(71.5, 11, 16, 18, '#8a4fd0', 'Solver control', 'same exact LP\nfor OvA and\nDeCCaF')
box(91.5, 1, 18, 38, '#1aa67a', 'Assumption\naudit', 'temporal drift\ncost ratio λ:\nre-evaluated and\nretrained at\n0.02, 0.1, 0.2, 0.5\ncalibration\noracle headroom\ncomplementarity')
arrow(16.5, 22, 20.5, 30); arrow(16.5, 18, 20.5, 10)
arrow(47.5, 30, 51.5, 23); arrow(47.5, 10, 51.5, 17)
arrow(67.5, 20, 71.5, 20); arrow(87.5, 20, 91.5, 20)
fig.tight_layout(pad=0.2)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'F0_study_overview.{ext}', dpi=300, bbox_inches='tight')
print('ok')
