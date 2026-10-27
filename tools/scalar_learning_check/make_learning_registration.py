#!/usr/bin/env python3
"""Registration for the limited single-action Orca learning check (L1). Writes JSON only; starts nothing.

Training = the historical moving template plus exactly the components already validated without
optimization (verified IPC v2, small_uniform head, zero_sum_offset rho=0) and a finite budget.
Reward, architecture, int(100*4**a) window mapping, 24 conditions, one connection per actor and
immediate takeover stay as in the historical runs. Host TCP buffer maxima are raised to 64 MiB
(only BDP+queue > ~2070 packets is affected: G42/G48/G72-R200-q2). Deployment keys stay placeholders.

    python3 make_learning_registration.py --code-commit <sha> --out learning-registration.json
"""
import argparse
import copy
import json
import pathlib

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
TEMPLATE = ROOT / 'templates/scalar-orca-moving-train.json'
VALIDATED = ROOT / 'evidence/scalar-exploration-20261027/registration-master.json'
CELLS = ['G12-R20-q2', 'G24-R20-q2', 'G48-R20-q2', 'G72-R20-q2', 'G48-R50-q2', 'G24-R100-q2', 'G72-R100-q2',
         'G12-R200-q2', 'G48-R200-q2', 'G72-R200-q2']
CHECKPOINTS = [0, 200000, 400000, 800000, 1200000]
A4G_AT = [400000, 1200000]
VALIDATED_KEYS = ('verified_ipc_v2', 'scalar_window_initialization', 'exploration_process', 'exploration_offset_rho',
                  'exploration_std', 'require_normal_cubic_start', 'orca_actor_loss_moving_stats', 'backend',
                  'target_bn_consistent', 'independent_critic_adam_v34', 'reward_mode', 'LOSS_TYPE', 'update_ratio',
                  'warmup', 'memsize', 'sampler', 'control_period_ms')
DEPLOYMENT = ('actor_hosts', 'learner_endpoint', 'actor_port_base', 'stop_file', 'absolute_deadline_epoch', 'timeout_s')


def validated_job():
    plan = json.loads(VALIDATED.read_text())
    for job in plan['formal']:
        if job.get('exploration_process') == 'zero_sum_offset':
            return job
    raise ValueError('no validated zero_sum job')


def diff(a, b, skip=('conditions',)):
    return sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))


def build(commit, seed=3):
    template = json.loads(TEMPLATE.read_text())
    zero = validated_job()
    cfg = copy.deepcopy(template)
    run = 'learncheck-L1-orca_moving-s%d' % seed
    cfg.update(run_id=run, model_id=run, seed=seed, code_commit=commit,
               verified_ipc_v2=True, server_binary='@FRAMEWORK@/native/orca-server-verified-ipc',
               scalar_window_initialization='small_uniform', exploration_process='zero_sum_offset',
               exploration_offset_rho=0.0, checkpoint_interval=100000, sample_budget=1200000,
               model_source='Fresh random single-action run for a limited learning check; not a baseline release.')
    for key in DEPLOYMENT:
        cfg[key] = 'SET_BY_DEPLOYMENT'
    for key in VALIDATED_KEYS:
        assert cfg[key] == zero[key], ('training differs from the validated zero-update job', key, cfg[key], zero[key])
    names = [c['condition'] for c in cfg['conditions']]
    assert len(names) == 24 and set(CELLS) <= set(names)
    changed = diff(template, cfg)
    allowed = {'run_id', 'model_id', 'seed', 'code_commit', 'verified_ipc_v2', 'server_binary',
               'scalar_window_initialization', 'exploration_process', 'exploration_offset_rho', 'checkpoint_interval',
               'sample_budget', 'model_source'} | set(DEPLOYMENT)
    assert set(changed) <= allowed, sorted(set(changed) - allowed)
    jobs = []
    for cp in CHECKPOINTS:
        for cell in CELLS:
            # A0 startup anchoring is bimodal (E2), so the final checkpoint gets two A0 repeats.
            for repeat in range(2 if cp == CHECKPOINTS[-1] else 1):
                jobs.append(dict(checkpoint=cp, arm='A0', condition=cell, repeat=repeat))
            if cp in A4G_AT:
                jobs.append(dict(checkpoint=cp, arm='A4g', condition=cell, repeat=0))
    jobs += [dict(checkpoint='reference', arm='CUBIC', condition=cell) for cell in CELLS]
    jobs += [dict(checkpoint='old', arm=arm, condition=cell) for cell in CELLS for arm in ('A0', 'A4g')]
    return dict(
        identity='Limited learning check L1, one seed. Diagnostic only; no model release, no paper claim.',
        training_config=cfg,
        differences_from_historical_moving_template=changed,
        differences_from_validated_zero_update_job=diff(zero, cfg),
        environment=dict(
            sysctl={'net.ipv4.tcp_rmem': '4096 87380 67108864', 'net.ipv4.tcp_wmem': '4096 32768 67108864',
                    'net.core.rmem_max': '67108864', 'net.core.wmem_max': '67108864'},
            scope='All actor hosts before the learner starts; receipts (window_binding.py --host-receipt) before and after; '
                  'the same setting for every evaluation and CUBIC reference of this check.',
            effect='Only conditions with BDP+queue > ~2070 packets change (G42/G48/G72-R200-q2); others never reach the old cap.'),
        monitor=dict(tool='tools/scalar_learning_check/learning_monitor.py', window_s=30,
                     rules=dict(S1='actors with |pi|>=.95 for >=50% of a window: flag >=3, stop >=12',
                                S2='actors with cwnd<=10 for >=50% of a window: flag >=3, stop >=12',
                                S3='any cwnd>=1e6: stop'),
                     cadence='Every 30 s on synced actor states.csv when practical, otherwise post hoc; a STOP ends training and keeps all evidence.'),
        evaluation=dict(cells=CELLS, checkpoints=CHECKPOINTS, A4g_checkpoints=A4G_AT, seconds=60,
                        repeats='1; the final checkpoint runs A0 twice (repeats are averaged per cell)',
                        policy='deterministic, verified IPC v2, normal CUBIC start, same host buffers',
                        arms=dict(A0='immediate takeover (training protocol)',
                                  A4g='diagnostic only: normalizer floored at 0.963 x capacity (goodput), never a deployable result',
                                  CUBIC='reference, once per cell',
                                  old='historical moving checkpoint (twotrack orca_moving step-3557457) under A0 and A4g in the '
                                      'same environment; comparison only, never part of a decision'),
                        jobs=jobs, job_count=len(jobs),
                        record='states/control/ipc logs; window_binding.py (binding_rtt, ss_mode); trim10 and full windows'),
        predictions=dict(
            P_L1='No STOP in 1.2M transitions; |pi|>=.95 below 10% of every actor-window; no cwnd>=1e6. Weak test: the historical '
                 'moving arm never saturated (the early -1 collapse was the control arm with batch-statistics actor loss).',
            P_L2='Descriptive: RTT-20 actors learn negative pi while cwnd<ssthresh (slow-start compensation); utilization stays >=.9 in training windows.',
            P_L3='Anchoring at evaluation replicates E2: at the final checkpoint A4g - A0 utilization >= .15 in at least 2 of the 4 RTT-20 cells.',
            P_L4='Long RTT barely learns under rho=0: every RTT-200 cell keeps |pi(final)-pi(step 0)| <= .05 and A0 queue within +-25%.',
            P_L5='Clean replay helps where the old run exploded: at the final checkpoint A4g utilization >= .90 with queue <= CUBIC in '
                 'G24-R100, G72-R100 and the three RTT-200 cells (reported against the historical checkpoint, not a decision).'),
        decision_rules='tools/scalar_learning_check/adjudicate_learning.py: COLLAPSE_REPRODUCED > POLICY_LIMITED > '
                       'ANCHORING_LIMITED > LONG_RTT_STATIC > GO_LONG_TRAINING > PARTIAL (all triggered flags reported).',
        budget=dict(training='about 20 min wall at ~990 transitions/s; ~120k critic iterations (historical collapse by 217k transitions)',
                    evaluation='%d frozen 60 s runs; about 40-45 min on four actor hosts in parallel' % len(jobs)),
        not_in_this_check=['connection resets', 'anchored takeover in training', 'reward or normalizer change',
                           'exploration rho>0', 'architecture/window mapping change', 'MTP or dual action'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--code-commit', required=True)
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    plan = build(a.code_commit, a.seed)
    a.out.write_text(json.dumps(plan, indent=1, allow_nan=False))
    print(json.dumps(dict(out=str(a.out), changed_from_template=plan['differences_from_historical_moving_template'],
                          evaluation_jobs=plan['evaluation']['job_count'])))


if __name__ == '__main__':
    main()
