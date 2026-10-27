#!/usr/bin/env python3
"""Build the registration for the single-variable exploration-process check from the a969f67 plan.

Only the exploration process changes.  Every zero_sum job is the a969f67 small_uniform job for the
same condition and seed with exploration_process='zero_sum_offset' (rho=0, per-step variance matched
to Gaussian .2) plus new identities.  Same network initialization, reward, window mapping, verified
IPC, conditions and 60 s exposure; optimizer still structurally disabled.  For the two conditions
whose BDP+queue exceeds the end-host in-flight ceiling (~2070 packets) the plan also re-runs the
upstream iid process as same-day anchors and takes read-only `ss` snapshots in both arms.

    python3 make_exploration_registration.py --deadline-epoch 1793500000 --code-commit <sha> --out registration.json
"""
import argparse
import copy
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve()
BASE = HERE.parents[2] / 'evidence/scalar-ipc-init-20261027/registration.json'
CEILING_PACKETS = 2070          # observed in-flight ceiling: 3.12 MB wire = 2074 x 1504 B in four independent runs
IDENTITY_KEYS = {'run_id', 'model_id', 'stop_file', 'code_commit', 'absolute_deadline_epoch'}
ARM_KEYS = {'exploration_process', 'exploration_offset_rho'}
MEASUREMENT_KEYS = {'ss_snapshot_seconds'}
SS_SECONDS = [15, 30, 45]


def capped(condition):
    bdp = condition['capacity_mbps'] * 1e6 / 8 * condition['base_rtt_ms'] / 1000 / condition.get('packet_bytes', 1504)
    return bdp + condition['queue_packets'] > CEILING_PACKETS


def derive(base_job, arm, stage, deadline, commit, seconds=None):
    job = copy.deepcopy(base_job)
    c = job['conditions'][0]
    rid = 'explorecheck-%s-%s-s%d-%s' % (stage, c['condition'], job['seed'], 'zero_sum' if arm == 'zero_sum_offset' else 'iid_anchor')
    job.update(run_id=rid, model_id=rid, stop_file='${EXPERIMENT_ROOT}/audit/%s.stop' % rid, code_commit=commit,
               absolute_deadline_epoch=deadline, exploration_process=arm)
    if arm == 'zero_sum_offset':
        job['exploration_offset_rho'] = 0.0
    if capped(c):
        job['ss_snapshot_seconds'] = list(SS_SECONDS)
    if seconds is not None:
        job['probe_seconds'] = seconds
    changed = {k for k in set(job) | set(base_job) if job.get(k) != base_job.get(k)}
    allowed = IDENTITY_KEYS | ARM_KEYS | MEASUREMENT_KEYS | ({'probe_seconds'} if seconds is not None else set())
    assert changed <= allowed, ('more than the exploration process changed', sorted(changed - allowed))
    assert job['scalar_window_initialization'] == 'small_uniform' and job['diagnostic_no_optimizer'] is True
    assert job['exploration_std'] == 0.2 and job['verified_ipc_v2'] is True
    return job, base_job['run_id']


def build(base, deadline, commit, same_day_all=False):
    small = [j for j in base['formal'] if j['scalar_window_initialization'] == 'small_uniform']
    assert len(small) == 16, len(small)
    rng = random.Random(base['randomization_seed'] + 1)
    blocks, reference = [], {}
    for j in sorted(small, key=lambda x: (x['conditions'][0]['condition'], x['seed'])):
        block = [derive(j, 'zero_sum_offset', 'formal', deadline, commit)]
        if same_day_all or capped(j['conditions'][0]):
            block.append(derive(j, 'iid', 'formal', deadline, commit))
        rng.shuffle(block)
        blocks.append(block)
    rng.shuffle(blocks)
    formal = []
    for block in blocks:
        for job, ref in block:
            formal.append(job)
            reference[job['run_id']] = ref
    pick = {(j['conditions'][0]['condition'], j['seed']): j for j in small}
    smoke = [derive(pick[('G24-R20-q2', 3)], 'zero_sum_offset', 'smoke', deadline, commit, 12)[0],
             derive(pick[('G72-R200-q2', 4)], 'zero_sum_offset', 'smoke', deadline, commit, 12)[0]]
    capped_conditions = sorted({j['conditions'][0]['condition'] for j in small if capped(j['conditions'][0])})
    return dict(
        deadline_epoch=deadline, smoke=smoke, formal=formal,
        purpose='Single-variable test of the cumulative multiplicative effect of training exploration. Actual one-learner/'
                'one-actor path, optimizer structurally disabled, small_uniform initialization, verified IPC v2. No trained-policy claim.',
        one_variable='Exploration process only: iid Gaussian .2 (a969f67 reference and same-day anchors) versus zero_sum_offset '
                     '(rho=0, sigma_o=.2/sqrt(2), per-step executed perturbation std .2, same NumPy stream). Architecture, reward, '
                     'int(100*4**a) window mapping, native integer multiplier, MIN_CWND, conditions, initialization and exposure unchanged. '
                     'No window cap, no buffer change, no startup change.',
        window_definition=base['window_definition'],
        gates=base['gates'],
        predictions=dict(
            P1_gates='All 16 zero_sum runs pass both registered gates (no window >=1e6; boundary-weighted time above 4*(BDP+queue) <1%).',
            P1b_mechanism='All 16 zero_sum runs: max_t |sum(a_exec - pi)| < 0.75 action units (about 5 sigma_o); all 4 iid anchors > 2.0.',
            P2_loss='Window-binding conditions (12 pairs, all except %s): full-window drop fraction lower than the paired a969f67 '
                    'small_uniform iid run in at least 10 of 12 pairs.' % ', '.join(capped_conditions),
            P3_capped='Descriptive, no pass/fail: in %s report in-flight p95, binding/unanchored fractions, queue and utilization for '
                      'zero_sum and same-day iid anchors. If in-flight is pinned near 3.0 MB payload, delay there is set by end-host '
                      'buffers and is not attributable to the policy or to exploration.' % ', '.join(capped_conditions),
            P4_quantizer='All 20 runs: per-step percent truncation (ln int(100*4**a)/100 - a*ln4) between -0.0065 and -0.0035, '
                         'i.e. the action definition itself contributes about -0.5% per command in both arms.',
            anchors='At least 3 of 4 iid anchors reproduce a window >=1e5 (same seeds, same initialization, same noise draws as a969f67).'),
        decision_rules=dict(
            adopt='P1 and P1b hold: zero_sum_offset becomes the exploration process for the next limited training; low utilization '
                  'is reported and never counted as success.',
            capped_only_failure='Failures only in capped conditions with positive command drift (deterministic + rounding > 0): '
                                'classify as end-host-buffer problem, run the separate environment check, do not tune exploration.',
            uncapped_failure='Failures in window-binding conditions: inspect between-command native growth (PRR/ssthresh recovery path); '
                             'next single variable is an explicit registered cwnd validation for RL writes, never a silent cap.'),
        reference=dict(evidence='evidence/scalar-ipc-init-20261027/formal-results.json', commit='a969f67', paired_run_id=reference),
        capped_conditions=capped_conditions, ceiling_packets=CEILING_PACKETS,
        same_day_all=bool(same_day_all),
        low_window_and_packet_metrics=base['low_window_and_packet_metrics'],
        additional_training='NOT_AUTHORIZED_BY_THIS_PLAN; no buffer change, E2b or training automatically',
        selected_conditions=base['selected_conditions'], initialization_seeds=base['initialization_seeds'],
        randomization_seed=base['randomization_seed'] + 1,
        limitation='Two initialization seeds; the a969f67 reference ran on an earlier day, so capped conditions also have same-day '
                   'anchors. Duration gate is a boundary-weighted approximation. Exploration safety only, not learning quality.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', type=pathlib.Path, default=BASE)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--code-commit', required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--same-day-all',action='store_true',help='Pair all16 condition/seeds on the same day and host,32 runs')
    a = p.parse_args()
    plan = build(json.loads(a.base.read_text()), a.deadline_epoch, a.code_commit,a.same_day_all)
    a.out.write_text(json.dumps(plan, indent=1, allow_nan=False))
    print(json.dumps(dict(out=str(a.out), smoke=len(plan['smoke']), formal=len(plan['formal']),
                          capped=plan['capped_conditions'])))


if __name__ == '__main__':
    main()
