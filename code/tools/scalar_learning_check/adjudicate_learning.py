#!/usr/bin/env python3
"""Apply the registered decision rules of the limited learning check (read-only, no TF/TCP).

Inputs
  --evaluations  CSV of frozen evaluation rows: checkpoint, arm (A0 | A4g | CUBIC), condition, window
                 (full | trim10), utilization, queue_mean_ms, queue_p95_ms, queue_p99_ms, drop_fraction,
                 reward_mean, action_no_noise_mean; optional throughput_term_mean. CUBIC rows use
                 checkpoint "reference". The step-0 checkpoint is labelled 0.
  --registration registration JSON from make_learning_registration.py (cells, thresholds)
  --monitor      learning_monitor.py output (stop/flag history)
All rules use trim10 rows and the final (largest) checkpoint unless stated.
  A4g_good / A0_good  utilization >= .90, queue_mean <= CUBIC queue_mean, drop <= max(.01, CUBIC drop)
  gap                 utilization(A4g) - utilization(A0)
  alias run           utilization < .6 while reward_mean (or throughput_term_mean) >= .7
Decision, in priority order (all triggered conditions are also listed under flags)
  COLLAPSE_REPRODUCED  the monitor registered a STOP
  POLICY_LIMITED       A4g utilization < .80 in >= 2 RTT-20 cells
  ANCHORING_LIMITED    gap >= .15 and A4g utilization >= .90 in >= 2 RTT-20 cells
  LONG_RTT_STATIC      in every RTT-200 cell |pi_mean(final) - pi_mean(step 0)| <= .05 and the A0 queue stays
                       within +-25% of step 0 (zero_sum rho=0 gave no usable signal at long RTT)
  GO_LONG_TRAINING     A0_good in >= 8 of 10 cells, gap <= .10 in >= 8 of 10, no monitor flag in the last third
  PARTIAL              anything else; the per-checkpoint trend decides whether to extend
Optional rows with checkpoint "old" (the historical moving checkpoint, arms A0/A4g) are compared with the final
checkpoint per cell; the comparison is reported and never used in a decision.
"""
import argparse
import csv
import json
import pathlib
import sys


def load_rows(path):
    with open(str(path), newline='') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get('window') != 'trim10':
            continue
        x = dict(r)
        for k in ('utilization', 'queue_mean_ms', 'queue_p95_ms', 'queue_p99_ms', 'drop_fraction', 'reward_mean',
                  'action_no_noise_mean', 'throughput_term_mean'):
            if x.get(k) not in (None, ''):
                x[k] = float(x[k])
        out.append(x)
    return out


def index(rows):
    table = {}
    for r in rows:
        table.setdefault((str(r['checkpoint']), r['arm'], r['condition']), []).append(r)
    return table


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), float)]
    return sum(vals) / len(vals) if vals else None


def good(run_rows, cubic_rows, tail_guard=False):
    if not run_rows or not cubic_rows:
        return None
    base=bool(mean(run_rows, 'utilization') >= .90 and mean(run_rows, 'queue_mean_ms') <= mean(cubic_rows, 'queue_mean_ms')
                and mean(run_rows, 'drop_fraction') <= max(.01, mean(cubic_rows, 'drop_fraction')))
    if tail_guard:
        for key in ['queue_p95_ms','queue_p99_ms']:
            value,reference=mean(run_rows,key),mean(cubic_rows,key)
            if value is None or reference is None or value>1.2*reference+1.0:return False
    return base


def is_alias(r):
    score = r.get('throughput_term_mean') if isinstance(r.get('throughput_term_mean'), float) else r.get('reward_mean')
    return bool(r['utilization'] < .6 and isinstance(score, float) and score >= .7)


def adjudicate(rows, registration, monitor):
    cells = registration['evaluation']['cells']
    t = index(rows)
    checkpoints = sorted({k[0] for k in t if k[0] not in ('reference', 'old')}, key=lambda c: int(c))
    final, step0 = checkpoints[-1], checkpoints[0]
    r20 = [c for c in cells if '-R20-' in c]
    r200 = [c for c in cells if '-R200-' in c]
    per_cell = {}
    for c in cells:
        cubic = t.get(('reference', 'CUBIC', c), [])
        a0, a4 = t.get((final, 'A0', c), []), t.get((final, 'A4g', c), [])
        per_cell[c] = dict(A0_util=mean(a0, 'utilization'), A4g_util=mean(a4, 'utilization'),
                           A0_queue=mean(a0, 'queue_mean_ms'), A4g_queue=mean(a4, 'queue_mean_ms'),
                           CUBIC_queue=mean(cubic, 'queue_mean_ms'), A0_good=good(a0, cubic,registration.get('tail_guard',False)), A4g_good=good(a4, cubic,registration.get('tail_guard',False)),
                           gap=(mean(a4, 'utilization') - mean(a0, 'utilization')) if a0 and a4 else None)
    trend = []
    for cp in checkpoints:
        a0 = [r for c in cells for r in t.get((cp, 'A0', c), [])]
        ratios = [mean(t.get((cp, 'A0', c), []), 'queue_mean_ms') / max(mean(t.get(('reference', 'CUBIC', c), []), 'queue_mean_ms'), 1e-9)
                  for c in cells if t.get((cp, 'A0', c)) and t.get(('reference', 'CUBIC', c))]
        trend.append(dict(checkpoint=int(cp), A0_runs=len(a0), A0_util_mean=mean(a0, 'utilization'),
                          A0_util_min=min((r['utilization'] for r in a0), default=None),
                          A0_queue_over_CUBIC_mean=sum(ratios) / len(ratios) if ratios else None,
                          alias_runs=sum(is_alias(r) for r in a0)))
    static = []
    for c in r200:
        f, z = t.get((final, 'A0', c), []), t.get((step0, 'A0', c), [])
        if f and z:
            dq = mean(f, 'queue_mean_ms') / max(mean(z, 'queue_mean_ms'), 1e-9)
            static.append(abs(mean(f, 'action_no_noise_mean') - mean(z, 'action_no_noise_mean')) <= .05 and .75 <= dq <= 1.25)
    rules = monitor.get('rules', [])
    tail = rules[len(rules) * 2 // 3:]
    late_flag = any(r[x] != 'OK' for r in tail for x in ('S1', 'S2', 'S3'))
    count = lambda pred: sum(1 for c in cells if pred(per_cell[c]))
    long_static = bool(static) and all(static)
    flags = []
    if monitor.get('stop'):
        flags.append('COLLAPSE_REPRODUCED')
    if sum(1 for c in r20 if per_cell[c]['A4g_util'] is not None and per_cell[c]['A4g_util'] < .80) >= 2:
        flags.append('POLICY_LIMITED')
    if sum(1 for c in r20 if per_cell[c]['gap'] is not None and per_cell[c]['gap'] >= .15 and per_cell[c]['A4g_util'] >= .90) >= 2:
        flags.append('ANCHORING_LIMITED')
    if long_static:
        flags.append('LONG_RTT_STATIC')
    if not flags and count(lambda x: x['A0_good']) >= 8 and count(lambda x: x['gap'] is not None and x['gap'] <= .10) >= 8 and not late_flag:
        flags.append('GO_LONG_TRAINING')
    decision = flags[0] if flags else 'PARTIAL'
    nxt = dict(COLLAPSE_REPRODUCED='Audit the first saturated actor (critic action-gradient, replay support) before any new variable.',
               POLICY_LIMITED='Offline audit of critic direction on RTT-20 states; then one variable (actor saturation protection or delay term).',
               ANCHORING_LIMITED='Frozen E2b on the final checkpoint (A0 vs observation-ended anchoring vs A4g), then one training arm with that takeover and connection resets.',
               LONG_RTT_STATIC='Zero-update check of a persistent offset (rho>0, matched stationary deviation) before long training.',
               GO_LONG_TRAINING='Long training with the same pipeline: 3 seeds x 2.4M transitions, same monitor and evaluation.',
               PARTIAL='Read the trend; extend to 2.4M only if A0 queue/CUBIC falls monotonically at stable utilization.')[decision]
    old = {}
    for c in cells:
        for arm in ('A0', 'A4g'):
            o, f = t.get(('old', arm, c), []), t.get((final, arm, c), [])
            if o and f:
                old.setdefault(c, {})[arm] = dict(util_new_minus_old=mean(f, 'utilization') - mean(o, 'utilization'),
                                                  queue_new_minus_old_ms=mean(f, 'queue_mean_ms') - mean(o, 'queue_mean_ms'))
    return dict(status='ADJUDICATED', decision=decision, flags=flags, next=nxt, final_checkpoint=int(final),
                LONG_RTT_STATIC=long_static, monitor_stop=bool(monitor.get('stop')), monitor_late_flag=late_flag,
                per_cell=per_cell, trend=trend, versus_historical_checkpoint=old)


def _synthetic_rows(case):
    cells = ['G12-R20-q2', 'G24-R20-q2', 'G48-R20-q2', 'G72-R20-q2', 'G48-R50-q2', 'G24-R100-q2', 'G72-R100-q2',
             'G12-R200-q2', 'G48-R200-q2', 'G72-R200-q2']
    rows = []
    for c in cells:
        rows.append(dict(checkpoint='reference', arm='CUBIC', condition=c, window='trim10', utilization=1.0,
                         queue_mean_ms=30.0, drop_fraction=0.002, reward_mean=0.5, action_no_noise_mean=0.0))
        for cp in ('0', '400000', '1200000', 'old'):
            final = cp == '1200000'
            low = '-R20-' in c and final
            a0u = 0.4 if (low and case in ('anchoring', 'policy')) else 0.97
            a4u = 0.95 if case == 'anchoring' else (0.5 if (low and case == 'policy') else 0.97)
            learns = case != 'static' and final and '-R200-' in c     # long-RTT policy moved away from step 0
            for arm, u in (('A0', a0u), ('A4g', a4u)):
                rows.append(dict(checkpoint=cp, arm=arm, condition=c, window='trim10',
                                 utilization=(0.9 if cp == 'old' else u), queue_mean_ms=(6.0 if learns else 10.0),
                                 drop_fraction=0.0, reward_mean=0.95,
                                 action_no_noise_mean=(-0.3 if '-R20-' in c else (-0.1 if learns else 0.0))))
    return rows, cells


def self_test(tmp):
    out = {}
    for case, expect in (('anchoring', 'ANCHORING_LIMITED'), ('policy', 'POLICY_LIMITED'), ('good', 'GO_LONG_TRAINING'),
                         ('static', 'LONG_RTT_STATIC')):
        rows, cells = _synthetic_rows(case)
        path = pathlib.Path(tmp) / ('%s.csv' % case)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        v = adjudicate(load_rows(path), dict(evaluation=dict(cells=cells)), dict(stop=False, rules=[]))
        assert v['decision'] == expect, (case, v['decision'], v['flags'], v['per_cell'])
        assert abs(v['versus_historical_checkpoint']['G12-R200-q2']['A0']['util_new_minus_old'] - 0.07) < 1e-9
        out[case] = v['flags']
    rows, cells = _synthetic_rows('good')
    v = adjudicate(rows, dict(evaluation=dict(cells=cells)), dict(stop=True, rules=[]))
    assert v['decision'] == 'COLLAPSE_REPRODUCED' and v['trend'][0]['alias_runs'] == 0
    out['collapse'] = v['flags']
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evaluations')
    p.add_argument('--registration')
    p.add_argument('--monitor')
    p.add_argument('--self-test', metavar='TMPDIR')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test(a.self_test)), indent=1))
        return 0
    v = adjudicate(load_rows(a.evaluations), json.loads(pathlib.Path(a.registration).read_text()),
                   json.loads(pathlib.Path(a.monitor).read_text()))
    print(json.dumps(v, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
