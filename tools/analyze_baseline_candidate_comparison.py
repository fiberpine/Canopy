"""Compare archived H0 and the BN-repair endpoint using existing local evidence only."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'evidence/baseline-online-bn-repair-20261112'
KEYS = ('utilization', 'queue_mean_ms', 'queue_p95_ms', 'drop_fraction')


def trim(row):
    return next(v for v in row['windows'] if v['window'] == 'trim10')


def aggregate(cells):
    return {key: mean(c[key] for c in cells) for key in KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((SOURCE / name).read_bytes())
    rows = [r for h in load('frozen-final-latest.json') for r in h['rows']]
    current = defaultdict(list)
    h0 = {}
    for row in rows:
        name = row['job']['condition']['condition']
        if row['job']['group'] == 'onlinebn-final6m':
            assert row['status'] == 'COMPLETED'
            assert row['policy_sha256'] == '631683b939596cabc85262e2294123c462913bf6b836dff10b84501644ad60e0'
            assert row['model_sha256'] == row['job']['model_sha256']
            current[name].append(row)
        elif row['status'] == 'COMPLETED':
            h0[name] = row
    assert len(current) == 24 and all(len(v) == 3 for v in current.values())
    assert len(h0) == 23 and 'G12-R200' not in h0
    candidate = {name: {key: mean(trim(r)[key] for r in rs) for key in KEYS} for name, rs in current.items()}
    cells = [dict(condition=name, base_rtt_ms=h0[name]['job']['condition']['base_rtt_ms'],
                  candidate=candidate[name], H0=trim(h0[name]),
                  queue_delta_ms=candidate[name]['queue_mean_ms'] - trim(h0[name])['queue_mean_ms'])
             for name in sorted(h0)]
    matched = dict(candidate=aggregate([c['candidate'] for c in cells]), H0=aggregate([c['H0'] for c in cells]),
                   cells=23, candidate_repeats=3, H0_repeats=1,
                   queue_improved=sum(c['queue_delta_ms'] < 0 for c in cells),
                   queue_worsened=sum(c['queue_delta_ms'] > 0 for c in cells))
    other_windows = {}
    for window in ('full', 'startup0_10'):
        metric = lambda r: next(w for w in r['windows'] if w['window'] == window)
        candidate_window = {name: {key: mean(metric(r)[key] for r in rs) for key in KEYS} for name, rs in current.items()}
        other_windows[window] = dict(candidate_all24=aggregate(list(candidate_window.values())),
                                    candidate_same23=aggregate([candidate_window[name] for name in h0]),
                                    H0_same23=aggregate([metric(r) for r in h0.values()]))
    sensitivity = []
    for excluded in (['G48-R200'], ['G48-R200', 'G72-R200']):
        cc = [c for c in cells if c['condition'] not in excluded]
        sensitivity.append(dict(excluded=excluded, cells=len(cc), candidate=aggregate([c['candidate'] for c in cc]),
                                H0=aggregate([c['H0'] for c in cc]), scope='Descriptive sensitivity only; original acceptance unchanged'))
    by_rtt = []
    for rtt in (20, 50, 100, 200):
        cc = [c for c in cells if c['base_rtt_ms'] == rtt]
        by_rtt.append(dict(base_rtt_ms=rtt, cells=len(cc), candidate=aggregate([c['candidate'] for c in cc]), H0=aggregate([c['H0'] for c in cc])))
    archive_path = ROOT / 'evidence/baseline/metrics/evaluation_runs.csv'
    archive = [r for r in csv.DictReader(archive_path.open()) if r['group'] in ('trained', 'screen-trained') and r['window'] == 'trim10']
    assert len(archive) == 72 and len({r['condition'] for r in archive}) == 24
    assert {r['policy_hash'] for r in archive} == {'b580dc803e511d458d8d78d1248e2b572d0576475f91086ad9a0007e679bb3c8'}
    history = dict(cells=24, repeats=3, statistics={key: mean(float(r[key]) for r in archive) for key in KEYS},
                   weak_G12_R200=[float(r['utilization']) for r in archive if r['condition'] == 'G12-R200'],
                   scope='Historical campaign, different execution date/chain; not the matched current reference')
    details, seconds = [], []
    for row in load('weak-existing-log-audit.json')['runs']:
        if row['job']['condition']['condition'] != 'G12-R200':
            continue
        bins = row['second_bins']
        assert len(bins) == 60
        values = [b['departure_bytes'] / b['service_bytes'] for b in bins]
        for window in row['windows']:
            lo, hi = int(window['start_s']), int(window['end_s'])
            rebuilt = sum(b['departure_bytes'] for b in bins[lo:hi]) / sum(b['service_bytes'] for b in bins[lo:hi])
            assert abs(rebuilt - window['utilization']) < 1e-12
            assert window['complete_window']
        first = row['first_controls'][0]
        headers = '\n'.join(row['down_log_headers'])
        init = float(re.search(r'# init timestamp: (\d+)', headers).group(1))
        base = float(re.search(r'# base timestamp: (\d+)', headers).group(1))
        handoff = (float(first['time_us']) / 1000 - init - base) / 1000
        original = next(r for r in rows if r['job']['run'] == row['job']['run'])
        if original['status'] == 'COMPLETED':
            assert abs(handoff - original['handoff_s']) < 1e-6
        recovery = next((s for s in range(10, 56) if min(values[s:s+5]) >= .9), None)
        sustained = next((s for s in range(10, 60) if min(values[s:]) >= .9), None)
        details.append(dict(run=row['job']['run'], group=row['job']['group'], host_slot=row['job']['host'],
                            original_status=original['status'], diagnostic_status=row['status'], windows=row['windows'],
                            first_control=first, handoff_s=handoff, control_period_ms=original.get('control_period_ms'),
                            first_five_consecutive_seconds_ge90_after10_s=recovery,
                            sustained_ge90_to_end_after10_s=sustained,
                            bins10=[dict(start_s=s, utilization=sum(b['departure_bytes'] for b in bins[s:s+10]) /
                                         sum(b['service_bytes'] for b in bins[s:s+10])) for s in range(0, 60, 10)],
                            model_hash_unchanged=row['model_hash_unchanged'], state_rows=row['state_rows'],
                            state_intervals_on_RL_clock=row['state_time_intervals']))
        seconds += [dict(run=row['job']['run'], group=row['job']['group'], network_second=s, utilization=u) for s, u in enumerate(values)]
    assert len(details) == 4
    overall = aggregate(list(candidate.values()))
    for key in KEYS:
        assert abs(overall[key] - load('health-assessment.json')['overall'][key]) < 1e-12
    result = dict(status='OFFLINE_REANALYSIS_NO_CLOUD_NO_TF_NO_NEW_EXPERIMENT',
                  current_overall=overall, historical_H0=history, matched=matched, other_windows=other_windows,
                  sensitivity=sensitivity, by_rtt=by_rtt,
                  cells=cells, weak_condition_runs=details,
                  hypothetical_other_23_unchanged=dict(weak_cell_mean_util90=(23 * matched['candidate']['utilization'] + .9) / 24,
                                                      weak_cell_mean_util100=(23 * matched['candidate']['utilization'] + 1) / 24),
                  checks=dict(all_three_windows_rebuilt_from_60_second_bins=True, native_to_packet_clock_handoff_rebuilt=True,
                              original_FAIL_and_H0_execution_error_unchanged=True),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      archive_path, SOURCE/'frozen-final-latest.json', SOURCE/'weak-existing-log-audit.json', SOURCE/'health-assessment.json')})
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'weak-second-utilization.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(seconds[0])); writer.writeheader(); writer.writerows(seconds)
    if args.plot:
        plots(out, seconds, cells)
    print(json.dumps(dict(status=result['status'], matched_cells=23, weak_runs=4,
                          queue_improved=matched['queue_improved'], queue_worsened=matched['queue_worsened'],
                          all_window_utilizations_rebuilt=True)))


def plots(out, seconds, cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    styles = [('onlinebn-confirm-r0-c03', '#198754', 'Current r0: 89.58%', '-'),
              ('onlinebn-confirm-r1-c03', '#c13b32', 'Current r1: 3.43%', '-'),
              ('onlinebn-confirm-r2-c03', '#2768b5', 'Current r2: 32.00%', '-'),
              ('H0-reference-c03', '#777777', 'H0: diagnostic packet trace; execution failed', '--')]
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for name, color, label, style in styles:
        rr = [s for s in seconds if s['run'] == name]
        ax.step([r['network_second'] for r in rr] + [60], [100*r['utilization'] for r in rr] + [100*rr[-1]['utilization']],
                where='post', color=color, label=label, linestyle=style, linewidth=1.7)
    ax.axvspan(0, 10, color='#eff1f3', alpha=.6)
    ax.axvline(10, color='#666666', linewidth=.8)
    ax.axvspan(5.0, 5.2, color='#9467bd', alpha=.3)
    ax.text(5.3, 108, 'RL handoff ~5.1 s', color='#72528e', va='top')
    ax.annotate('r0 recovers at ~16 s', (16, 99), (20, 78), arrowprops=dict(arrowstyle='->', color='#198754'), color='#198754')
    ax.annotate('r2 recovers at ~45 s', (45, 94), (35, 62), arrowprops=dict(arrowstyle='->', color='#2768b5'), color='#2768b5')
    ax.set(xlim=(0, 60), ylim=(0, 112), xlabel='Packet-log time (s); shaded 0-10 s excluded from trim10 score',
           ylabel='Link utilization per 1-second bin (%)', title='G12-R200: same frozen policy, three fresh TCP connections')
    ax.grid(axis='y', alpha=.2); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, frameon=False, fontsize=9)
    for suffix in ('.png', '.svg'): fig.savefig(out / ('weak-repeat-trajectories' + suffix), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = {20:'#198754', 50:'#e69f00', 100:'#2768b5', 200:'#a348a3'}
    for rtt in colors:
        rr = [c for c in cells if c['base_rtt_ms'] == rtt]
        ax.scatter([c['H0']['queue_mean_ms'] for c in rr], [c['candidate']['queue_mean_ms'] for c in rr],
                   label='RTT %d ms (n=%d)' % (rtt, len(rr)), color=colors[rtt], s=42)
    ax.plot([1.5, 400], [1.5, 400], color='#777777', linestyle='--', linewidth=1)
    for name in ('G48-R200', 'G72-R200', 'G36-R200', 'G36-R100'):
        c = next(c for c in cells if c['condition'] == name)
        offset = (-5, -16) if name == 'G48-R200' else (5, 6)
        ax.annotate(name, (c['H0']['queue_mean_ms'], c['candidate']['queue_mean_ms']), xytext=offset,
                    ha='right' if name == 'G48-R200' else 'left', textcoords='offset points', fontsize=8)
    ax.set(xscale='log', yscale='log', xlim=(1.5, 450), ylim=(1.5, 100), xlabel='H0 mean queue (ms; one repeat per cell)',
           ylabel='Current mean queue (ms; three-repeat mean)', title='Same 23 conditions: 12 queue improvements, 11 regressions')
    ax.text(2, 70, 'Above diagonal: current has more queue', fontsize=8)
    ax.grid(alpha=.15); ax.legend(loc='lower right', fontsize=8)
    for suffix in ('.png', '.svg'): fig.savefig(out / ('matched-queue-comparison' + suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
