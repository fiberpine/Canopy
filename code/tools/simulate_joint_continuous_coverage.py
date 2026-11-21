"""Replay-coverage simulation for the continuous v4 calibration gate (no TF, no TCP).

Drives the REAL frameworks/joint_healthy/replay.py with production memsize,
actor count, reconnect interval and the v4 calibration behavior distribution:
probability `period_probe_probability` of a stratified continuous proposal
20*8**((k+u)/5), otherwise the declared initial policy period (fresh RTT ->
20.5ms; empty RTT -> clip(minRTT,40,120)). The empty-RTT fraction and minRTT
classes are explicit assumptions, not measurements. States are zeros: this
checks coverage/holdout bookkeeping only, never values or TCP behavior.

--bucketing nearest reproduces the pre-fix nearest-anchor buckets of 59cf69a.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def simulate(seconds=900., empty_fraction=.02, bucketing='code', seed=7, report_every=60.):
    import numpy as np
    from frameworks.joint_healthy import replay as replay_module
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.core import period_action, PERIODS
    cfg = configuration('continuous-v4')
    if bucketing == 'nearest':
        replay_module.period_bucket = lambda _cfg, ms: min(range(5), key=lambda i: abs(math.log(float(ms) / PERIODS[i])))
    elif bucketing != 'code':
        raise ValueError('bucketing must be code or nearest')
    rp = replay_module.Replay(cfg); rng = np.random.RandomState(seed)
    actors = int(cfg['num_actors']); minimum_rtt = np.array([20., 50., 100., 200.] * (actors // 4), float)
    zeros = np.zeros(82, np.float32)
    clock = np.zeros(actors); sequence = np.ones(actors, int); episode = np.zeros(actors, int); started = np.zeros(actors)

    def planned(actor):
        if rng.random_sample() < cfg['period_probe_probability']:
            return 20. * 8. ** ((rng.randint(5) + rng.random_sample()) / 5.)
        return 20.5 if rng.random_sample() >= empty_fraction else min(120., max(40., minimum_rtt[actor]))

    reports = []; next_report = report_every
    while clock.min() < seconds:
        actor = int(np.argmin(clock)); ms = planned(actor); dt = ms / 1000.
        terminal = float(clock[actor] + dt - started[actor] >= cfg['connection_reset_seconds'])
        rp.append([zeros, [0., period_action(ms)], [0.], zeros, [terminal], [dt], [.995 ** (dt / .02)]],
                  actor, episode[actor], sequence[actor], teacher_action=0.)
        clock[actor] += dt; sequence[actor] += 1
        if terminal: episode[actor] += 1; sequence[actor] = 1; started[actor] = clock[actor]
        if clock.min() >= next_report:
            reports.append(dict(physical_seconds_per_actor=next_report,
                                minimum_retained_rows_per_actor_by_bucket=rp.coverage.min(0).tolist(),
                                holdout_rows_by_bucket=[len(x) for x in rp.holdout],
                                short_return_blocks_by_bucket=[len(x) for x in rp.short_returns]))
            next_report += report_every
    required = int(cfg['calibration_rows_per_actor_period'])
    final = reports[-1]['minimum_retained_rows_per_actor_by_bucket']
    return dict(status='COVERAGE_REACHABLE' if min(final) >= required else 'COVERAGE_UNREACHABLE',
                bucketing=bucketing, required_rows_per_actor_bucket=required, memsize=cfg['memsize'],
                actors=actors, calibration_probe_probability=cfg['period_probe_probability'],
                assumed_empty_rtt_fraction=empty_fraction, assumed_minimum_rtt_classes_ms=[20, 50, 100, 200],
                simulated_physical_seconds_per_actor=seconds, appended_rows=int(rp.credit.inserted), reports=reports,
                limits='Real replay bookkeeping with an assumed behavior mix; no TF, values, TCP or measured RTT frames.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--empty-fraction', type=float, default=.02)
    parser.add_argument('--bucketing', default='code', choices=('code', 'nearest'))
    args = parser.parse_args()
    result = simulate(args.seconds, args.empty_fraction, args.bucketing)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], final=result['reports'][-1]), indent=1))
