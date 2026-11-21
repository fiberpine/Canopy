#!/usr/bin/env python3
"""Orca SIGCOMM'20 Style Fig. 7: 2x2 4-Flow Join-and-Leave Fairness & Convergence
(Fiber vs. Official Orca vs. BBR vs. CUBIC on 48 Mbps, 100 ms RTT, 5 BDP, 120s)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.8,
    'axes.titlesize': 10.0, 'axes.titleweight': 'bold',
    'axes.labelsize': 9.0, 'axes.spines.top': False, 'axes.spines.right': False,
})

DATA = Path(r'D:\Canopy\experiments\paper-evaluation-20261120\data\fresh_20261121')
OUT  = Path(r'D:\Canopy\experiments\paper-evaluation-20261120\figures_fiber')

rows = json.loads((DATA / 'fig7_orca_join_leave.json').read_text())
by_alg = {r['scheme']: r for r in rows}

fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), dpi=300, sharex=True, sharey=True)

PANELS = [
    ('fiber',         '(a) Fiber (Ours)'),
    ('official-orca', '(b) Official Orca'),
    ('bbr',           '(c) BBR'),
    ('cubic',         '(d) CUBIC'),
]
FLOW_COLORS = ['#D62728', '#0077B6', '#009E73', '#FF7F0E']
FLOW_LABELS = ['Flow 1 (0–120s)', 'Flow 2 (15–105s)', 'Flow 3 (30–90s)', 'Flow 4 (45–75s)']

# Fair-share staircase reference: 0-15:48, 15-30:24, 30-45:16, 45-75:12, 75-90:16, 90-105:24, 105-120:48
t_fair = [0, 15, 15, 30, 30, 45, 45, 75, 75, 90, 90, 105, 105, 120]
y_fair = [48, 48, 24, 24, 16, 16, 12, 12, 16, 16, 24, 24, 48, 48]

for ax, (alg, title) in zip(axes.flat, PANELS):
    r = by_alg[alg]
    # Shaded 4-flow overlap zone
    ax.axvspan(45, 75, color='#F3F4F6', alpha=0.8, zorder=0)
    ax.plot(t_fair, y_fair, color='#333333', ls='--', lw=1.2, alpha=0.75, label='Optimal Fair Share', zorder=2)

    for i, pf in enumerate(r['per_flow']):
        st = float(pf['start_s'])
        ys = np.array(pf['mbps_per_s'], dtype=float)
        ts = st + np.arange(len(ys)) + 0.5
        ax.plot(ts, ys, color=FLOW_COLORS[i], lw=1.85, label=FLOW_LABELS[i], zorder=4 + i)

    ax.grid(True, ls='--', alpha=0.35)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 54)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120])
    q_col = '#D62728' if alg == 'fiber' else '#333333'
    ax.set_title(f"{title}  |  4-Flow Jain = {r['jain4']:.3f}, Queue = {r['bottleneck_q_ms']:.1f} ms",
                 color=q_col if alg == 'fiber' else '#111111', pad=6)
    ax.text(60, 49.5, '4-Flow Overlap', ha='center', va='top', fontsize=7.6, color='#6B7280', style='italic')

axes[0, 0].set_ylabel('Throughput (Mbps)')
axes[1, 0].set_ylabel('Throughput (Mbps)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 1].set_xlabel('Time (s)')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, fontsize=9.0, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.96])

for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'fig7_intra_fairness.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('[SAVED] Orca SIGCOMM-20 Style 2x2 Join-and-Leave Fig. 7!')
