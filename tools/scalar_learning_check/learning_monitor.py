#!/usr/bin/env python3
"""Zero-TCP monitor for a limited single-action Orca learning run (read-only).

For every actor directory <root>/logs/<run>-actor<i>/states.csv (evaluate.py rows) and its condition
(config conditions[i]), summarize consecutive windows of `window_s` seconds of control time:

  det_mean / det_p10 / det_p90  deterministic action (action_no_noise)
  sat_low / sat_high            fraction with pi <= -0.95 / pi >= 0.95 (tanh saturation)
  clip_fraction                 fraction with executed |a| >= 0.999 (exploration pinned at the boundary)
  cwnd_p50 / cwnd_max / low_window_fraction (cwnd <= 10)
  utilization                   thr / goodput capacity; thr = throughput_term * max_bw (bytes/s) and
                                goodput capacity = capacity_mbps*1e6/8 * 1448/1504
  anchor                        max_bw / goodput capacity at the end of the window
  reward_mean                   the agent's own reward
  privileged_reward_mean        reward * max_bw / goodput capacity (same formula, true capacity as scale)
  alias                         utilization < 0.6 while mean throughput_term >= 0.7
  queue_ms                      mean(srtt - min_rtt)
  loss_fraction                 fraction of rows with loss_term > 0
  cumulative_exploration_max    max |sum(a_exec - pi)| since the connection started

Registered rules on each window index (actors aligned by control time):
  S1 saturation   actors with sat_low >= .5 or sat_high >= .5   flag >= 3 actors, stop >= 12
  S2 low window   actors with low_window_fraction >= .5          flag >= 3 actors, stop >= 12
  S3 explosion    any cwnd >= 1e6                                stop
  alias           actors flagged alias                           reported only
Optional learner logs/<run>/training.csv: q_mean, actor_grad, critic_loss by inserted transitions;
a vanishing actor gradient (< 1e-5 after warmup + 50k) is flagged.
Python 3.6+, stdlib only.
"""
import argparse
import csv
import json
import math
import pathlib
import re
import sys

GOODPUT = 1448.0 / 1504.0
FLAG_ACTORS, STOP_ACTORS = 3, 12


def quantile(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def rtt_group(condition):
    m = re.search(r'-R(\d+)', condition['condition'])
    return 'R' + m.group(1) if m else 'R%d' % condition.get('base_rtt_ms', 0)


def window_stats(rows, condition, cum_start):
    cap = condition['capacity_mbps'] * 1e6 / 8 * GOODPUT
    n = len(rows)
    det = [float(r['action_no_noise']) for r in rows]
    act = [float(r['action']) for r in rows]
    cwnd = [float(r['cwnd']) for r in rows]
    thr_term = [float(r['throughput_term']) for r in rows]
    max_bw = [float(r['max_bw']) for r in rows]
    thr = [t * m for t, m in zip(thr_term, max_bw)]
    reward = [float(r['reward']) for r in rows]
    cum, peak = cum_start, 0.0
    for a, p in zip(act, det):
        cum += a - p
        peak = max(peak, abs(cum))
    util = sum(thr) / n / cap
    thr_term_mean = sum(thr_term) / n
    return dict(
        rows=n,
        det_mean=sum(det) / n, det_p10=quantile(det, .1), det_p90=quantile(det, .9),
        sat_low=sum(p <= -0.95 for p in det) / n, sat_high=sum(p >= 0.95 for p in det) / n,
        clip_fraction=sum(abs(a) >= 0.999 for a in act) / n,
        cwnd_p50=quantile(cwnd, .5), cwnd_max=max(cwnd), low_window_fraction=sum(c <= 10 for c in cwnd) / n,
        utilization=util, anchor=max_bw[-1] / cap, throughput_term_mean=thr_term_mean,
        reward_mean=sum(reward) / n,
        privileged_reward_mean=sum(r * m / cap for r, m in zip(reward, max_bw)) / n,
        alias=bool(util < 0.6 and thr_term_mean >= 0.7),
        queue_ms=sum(float(r['srtt_ms']) - float(r['min_rtt_ms']) for r in rows) / n,
        loss_fraction=sum(float(r['loss_term']) > 0 for r in rows) / n,
        cumulative_exploration_max=peak), cum


def actor_windows(states_csv, condition, window_s):
    with open(str(states_csv), newline='') as f:
        rows = list(csv.DictReader(f))
    buckets = {}
    for r in rows:
        buckets.setdefault(int(float(r['elapsed_s']) // window_s), []).append(r)
    out, cum = [], 0.0
    for k in sorted(buckets):
        stats, cum = window_stats(buckets[k], condition, cum)
        stats.update(window=k, start_s=k * window_s)
        out.append(stats)
    return out


def learner_summary(training_csv, warmup):
    with open(str(training_csv), newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return dict(status='EMPTY')
    pts = []
    for r in rows:
        try:
            pts.append(dict(inserted=int(float(r['inserted'])), critic_iterations=int(float(r['critic_iterations'])),
                            q_mean=float(r['q_mean']), actor_grad=float(r['actor_grad']), critic_loss=float(r['critic_loss'])))
        except (KeyError, ValueError):
            continue
    late = [p for p in pts if p['inserted'] > warmup + 50000]
    vanishing = [p['inserted'] for p in late if abs(p['actor_grad']) < 1e-5]
    thin = pts[::max(1, len(pts) // 40)]
    return dict(points=len(pts), first=pts[0] if pts else None, last=pts[-1] if pts else None,
                first_vanishing_actor_grad_inserted=vanishing[0] if vanishing else None,
                vanishing_fraction_after_warmup=len(vanishing) / len(late) if late else None, trajectory=thin)


def monitor(root, run, config, window_s=30.0):
    root = pathlib.Path(root)
    conditions = config['conditions']
    actors, rules = {}, []
    for i, cond in enumerate(conditions):
        path = root / 'logs' / ('%s-actor%d' % (run, i)) / 'states.csv'
        if path.exists():
            actors[i] = dict(condition=cond['condition'], rtt=rtt_group(cond), windows=actor_windows(path, cond, window_s))
    if not actors:
        raise ValueError('no actor states.csv found under ' + str(root / 'logs'))
    first = {}
    last_window = max(w['window'] for a in actors.values() for w in a['windows'])
    for k in range(last_window + 1):
        cur = {i: next((w for w in a['windows'] if w['window'] == k), None) for i, a in actors.items()}
        cur = {i: w for i, w in cur.items() if w is not None}
        sat = sorted(i for i, w in cur.items() if w['sat_low'] >= .5 or w['sat_high'] >= .5)
        low = sorted(i for i, w in cur.items() if w['low_window_fraction'] >= .5)
        boom = sorted(i for i, w in cur.items() if w['cwnd_max'] >= 1e6)
        alias = sorted(i for i, w in cur.items() if w['alias'])
        by_rtt = {}
        for i, w in cur.items():
            g = by_rtt.setdefault(actors[i]['rtt'], dict(actors=0, det_mean=0.0, utilization=0.0, queue_ms=0.0))
            g['actors'] += 1
            for key in ('det_mean', 'utilization', 'queue_ms'):
                g[key] += w[key]
        for g in by_rtt.values():
            for key in ('det_mean', 'utilization', 'queue_ms'):
                g[key] /= g['actors']
        row = dict(window=k, start_s=k * window_s, actors=len(cur), saturated=sat, low_window=low, exploded=boom,
                   alias=alias, by_rtt=by_rtt,
                   S1=('STOP' if len(sat) >= STOP_ACTORS else 'FLAG' if len(sat) >= FLAG_ACTORS else 'OK'),
                   S2=('STOP' if len(low) >= STOP_ACTORS else 'FLAG' if len(low) >= FLAG_ACTORS else 'OK'),
                   S3=('STOP' if boom else 'OK'))
        for rule in ('S1', 'S2', 'S3'):
            if row[rule] != 'OK' and (rule, row[rule]) not in first:
                first[(rule, row[rule])] = k * window_s
        rules.append(row)
    out = dict(run=run, window_s=window_s, actors=len(actors), windows=len(rules),
               first_trigger_s={'%s_%s' % key: v for key, v in sorted(first.items())},
               stop=any(r[x] == 'STOP' for r in rules for x in ('S1', 'S2', 'S3')),
               rules=rules, actors_detail={str(i): a for i, a in actors.items()})
    training = root / 'logs' / run / 'training.csv'
    if training.exists():
        out['learner'] = learner_summary(training, config.get('warmup', 12288))
    return out


def _synthetic(root, run):
    import random
    rng = random.Random(7)
    conds = []
    for i, (name, cap, rtt) in enumerate([('G24-R20-q2', 23.99, 20), ('G24-R20-q2', 23.99, 20),
                                          ('G48-R200-q2', 47.98, 200), ('G12-R50-q2', 11.99, 50)]):
        conds.append(dict(condition=name, capacity_mbps=cap, base_rtt_ms=rtt, queue_packets=10))
        d = pathlib.Path(root) / 'logs' / ('%s-actor%d' % (run, i))
        d.mkdir(parents=True, exist_ok=True)
        capb = cap * 1e6 / 8 * GOODPUT
        with open(str(d / 'states.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['elapsed_s', 'action', 'action_no_noise', 'reward', 'throughput_term', 'loss_term', 'delay_term',
                        'max_bw', 'cwnd', 'srtt_ms', 'min_rtt_ms'])
            for t in range(3000):
                el = t * 0.02
                if i == 0:      # healthy: near-neutral policy, full utilization
                    pi, cw, tt, mb = -0.4, 60.0, 0.97, capb
                elif i == 1:    # collapses to the lower bound after 20 s
                    pi = -0.2 if el < 20 else -0.999
                    cw = 60.0 if el < 20 else 5.0
                    tt, mb = (0.97, capb) if el < 20 else (0.08, capb)
                elif i == 2:    # explodes after 40 s
                    pi, cw, tt, mb = 0.01, (3000.0 if el < 40 else 2e6), 0.98, capb
                else:           # alias: never anchored, sees full reward at 30% utilization
                    pi, cw, tt, mb = -0.1, 15.0, 0.97, 0.3 * capb
                a = max(-1.0, min(1.0, pi + rng.gauss(0, .2)))
                w.writerow([el, a, pi, tt, tt, 0.0, 1.0, mb, cw, rtt + 2, rtt])
    return dict(conditions=conds, warmup=12288)


def self_test(tmp):
    cfg = _synthetic(tmp, 'selftest')
    out = monitor(tmp, 'selftest', cfg, window_s=10.0)
    by_window = {r['window']: r for r in out['rules']}
    assert by_window[0]['saturated'] == [] and by_window[0]['low_window'] == [], by_window[0]
    assert 1 in by_window[3]['saturated'] and 1 in by_window[3]['low_window'], by_window[3]
    assert by_window[5]['S3'] == 'STOP' and by_window[3]['S3'] == 'OK', (by_window[3], by_window[5])
    assert all(3 in r['alias'] and 0 not in r['alias'] for r in out['rules']), 'alias detector'
    a3 = out['actors_detail']['3']['windows'][0]
    assert abs(a3['utilization'] - 0.29) < 0.02 and abs(a3['privileged_reward_mean'] - 0.29) < 0.02, a3
    assert out['stop'] and out['first_trigger_s']['S3_STOP'] == 40.0, out['first_trigger_s']
    return dict(first_trigger_s=out['first_trigger_s'], window3=by_window[3]['by_rtt'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', help='framework root containing logs/ and configs/')
    p.add_argument('--run')
    p.add_argument('--config', help='training config JSON (default <root>/configs/<run>.json)')
    p.add_argument('--window-s', type=float, default=30.0)
    p.add_argument('--out')
    p.add_argument('--self-test', metavar='TMPDIR')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test(a.self_test)), indent=1))
        return 0
    config = json.loads(pathlib.Path(a.config or pathlib.Path(a.root) / 'configs' / (a.run + '.json')).read_text())
    out = monitor(a.root, a.run, config, a.window_s)
    text = json.dumps(out, indent=1, allow_nan=False)
    if a.out:
        pathlib.Path(a.out).write_text(text)
        print(json.dumps(dict(out=a.out, actors=out['actors'], windows=out['windows'], stop=out['stop'],
                              first_trigger_s=out['first_trigger_s'])))
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
