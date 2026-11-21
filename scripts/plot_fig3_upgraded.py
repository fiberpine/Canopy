#!/usr/bin/env python3
"""Upgraded Publication-Grade Fig. 3: 2x4 Dual-Metric Sensitivity Matrix
(Top Row: Link Utilization %, Bottom Row: Mean Queue Delay ms)
strictly from 100% Fresh Rerun data (stage1, stage2_e2, stage2_e3)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
})

DATA = Path(r'D:\Canopy\experiments\paper-evaluation-20261120\data\fresh_20261121')
OUT  = Path(r'D:\Canopy\experiments\paper-evaluation-20261120\figures_fiber')

s1 = json.loads((DATA / 'stage1_40scen_11algs.json').read_text())
e2 = json.loads((DATA / 'stage2_e2_buffer.json').read_text())
e3 = json.loads((DATA / 'stage2_e3_loss.json').read_text())

ALGS = ['fiber', 'official-orca', 'orca-base', 'official-canopy', 'bbr', 'cubic']
STYLE = {
    'fiber':           dict(label='Fiber (Ours)', color='#D62728', marker='*', ms=8.5, lw=2.3, zo=10),
    'official-orca':   dict(label='Orca',         color='#0077B6', marker='v', ms=5.5, lw=1.5, zo=7),
    'orca-base':       dict(label='Orca-Base',    color='#4895EF', marker='s', ms=5.0, lw=1.4, zo=6),
    'official-canopy': dict(label='Canopy',       color='#009E73', marker='^', ms=5.5, lw=1.5, zo=6),
    'bbr':             dict(label='BBR',          color='#2CA02C', marker='o', ms=5.2, lw=1.5, zo=5),
    'cubic':           dict(label='CUBIC',        color='#555555', marker='d', ms=5.2, lw=1.4, zo=4),
}

def ok_rows(rows):
    return [r for r in rows if 'util_pct' in r and np.isfinite(r['util_pct']) and r['util_pct'] > 0.5]

fig, axes = plt.subplots(2, 4, figsize=(15.2, 5.8), dpi=300, sharex='col')

# Helper to plot one metric on an axis
def plot_series(ax, xs, ys, alg):
    st = STYLE[alg]
    ax.plot(xs, ys, color=st['color'], marker=st['marker'], markersize=st['ms'],
            lw=st['lw'], label=st['label'], zorder=st['zo'],
            markeredgecolor='black' if alg == 'fiber' else 'none', markeredgewidth=0.5)

# ---- Col 0: Buffer Size Sweep (0.25 - 10 BDP) ----
bdps = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
for alg in ALGS:
    xs, us, qs = [], [], []
    for b in bdps:
        sub = [r for r in ok_rows(e2) if r['scheme'] == alg and abs(r['bdp'] - b) < 1e-6]
        if sub:
            xs.append(b); us.append(sub[0]['util_pct']); qs.append(max(1.0, sub[0]['q_mean_ms']))
    plot_series(axes[0, 0], xs, us, alg)
    plot_series(axes[1, 0], xs, qs, alg)
axes[0, 0].set_title('(a) Buffer Size (0.25–10 BDP)')
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(bdps); axes[1, 0].set_xticklabels(['0.25', '0.5', '1', '2', '5', '10'])
axes[1, 0].set_xlabel('Bottleneck Buffer (BDP)')

# ---- Col 1: Random Packet Loss (0 - 2%) ----
losses = [0.0, 0.1, 0.5, 1.0, 2.0]
for alg in ALGS:
    xs, us, qs = [], [], []
    for lp in losses:
        sub = [r for r in ok_rows(e3) if r['scheme'] == alg and abs(r['loss_pct'] - lp) < 1e-6]
        if sub:
            xs.append(lp); us.append(sub[0]['util_pct']); qs.append(max(1.0, sub[0]['q_mean_ms']))
    plot_series(axes[0, 1], xs, us, alg)
    plot_series(axes[1, 1], xs, qs, alg)
axes[0, 1].set_title('(b) Random Packet Loss (0–2%)')
axes[1, 1].set_xticks(losses); axes[1, 1].set_xticklabels(['0', '0.1', '0.5', '1.0', '2.0'])
axes[1, 1].set_xlabel('Random Loss Rate (%)')

# ---- Col 2: Base RTT Scaling (20 - 200 ms) ----
wired = [r for r in ok_rows(s1) if r.get('family') in ('Wired-Broadband', 'Long-Haul-WAN')]
rtts = [20, 50, 100, 200]
for alg in ALGS:
    xs, us, qs = [], [], []
    for rtt in rtts:
        sub = [r for r in wired if r['scheme'] == alg and r['base_rtt_ms'] == rtt]
        if sub:
            xs.append(rtt)
            us.append(float(np.mean([r['util_pct'] for r in sub])))
            qs.append(float(np.mean([r['q_mean_ms'] for r in sub])))
    plot_series(axes[0, 2], xs, us, alg)
    plot_series(axes[1, 2], xs, qs, alg)
axes[0, 2].set_title('(c) Base RTT Scaling (20–200 ms)')
axes[1, 2].set_xticks(rtts)
axes[1, 2].set_xlabel('Base RTT (ms)')

# ---- Col 3: Bottleneck Bandwidth Scaling (12 - 72 Mbps) ----
bws = [12, 24, 36, 42, 48, 72]
for alg in ALGS:
    xs, us, qs = [], [], []
    for bw in bws:
        sub = [r for r in wired if r['scheme'] == alg and r['nominal_mbps'] == bw]
        if sub:
            xs.append(bw)
            us.append(float(np.mean([r['util_pct'] for r in sub])))
            qs.append(float(np.mean([r['q_mean_ms'] for r in sub])))
    plot_series(axes[0, 3], xs, us, alg)
    plot_series(axes[1, 3], xs, qs, alg)
axes[0, 3].set_title('(d) Bandwidth Scaling (12–72 Mbps)')
axes[1, 3].set_xticks(bws)
axes[1, 3].set_xlabel('Bottleneck Bandwidth (Mbps)')

# Styling both rows uniformly
for c in range(4):
    ax_u = axes[0, c]; ax_q = axes[1, c]
    ax_u.grid(True, ls='--', alpha=0.35)
    ax_q.grid(True, which='major', ls='--', alpha=0.35)
    ax_q.set_yscale('log')
    ax_q.axhspan(1.0, 20.0, color='#E8F5E9', alpha=0.45, zorder=0)
    if c == 0:
        ax_u.set_ylim(60, 103)
        ax_q.text(0.28, 1.35, 'Low-Queue Zone (<20 ms)', fontsize=7.0, color='#2E7D32', style='italic')
    elif c == 1:
        ax_u.set_ylim(0, 105)
    elif c == 2:
        ax_u.set_ylim(90, 102)
    else:
        ax_u.set_ylim(94, 102)
    ax_q.set_ylim(1.0, 550)

axes[0, 0].set_ylabel('Link Utilization (%)\n(Higher → Better)')
axes[1, 0].set_ylabel('Mean Queue Delay (ms)\n(Lower → Better)')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=9.0, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'fig3_sensitivity_2x2.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('[SAVED] Upgraded 2x4 Dual-Metric Fig. 3!')
