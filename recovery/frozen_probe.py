"""Single host, one frozen flow at a time, compact evidence only."""
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', ORCA_PYTHON=sys.executable)

import numpy as np
from frameworks.baseline.mahimahi_metrics import mm
from frameworks.joint_healthy.audit import audit_flow
from frameworks.joint_healthy.configuration import checked_authorization, runtime_preflight
from frameworks.joint_healthy.core import validate
from frameworks.joint_healthy.flow import dump, run_flow
from frameworks.joint_healthy.metrics import queue_tails


def prepared_config(slot, deadline):
    path = ROOT/'templates/joint-healthy/joint-continuous-v4.json'
    cfg = json.loads(path.read_text().replace('@PACKAGE@', str(ROOT)))
    cfg.update(run_id=slot['id'], deadline_epoch=deadline, stop_reserve_s=15)
    return validate(cfg)


def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def candidate(slot, deadline, stopped):
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    cfg = prepared_config(slot, deadline)
    override = slot.get('period_ms')
    seconds = slot['seconds']
    callbacks = []

    def capture(row, episode, sequence):
        callbacks.append((sequence, float(row[2][0]), float(row[5][0]), float(row[6][0])))

    with tf.Graph().as_default():
        tf.set_random_seed(cfg['seed'])
        model = Model(cfg, False)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,
                                              inter_op_parallelism_threads=2)) as session:
            session.run(tf.global_variables_initializer())
            model.initialize_baseline(session, cfg['baseline_checkpoint'])
            before = model.digest(session)

            def action(state):
                feed = {model.s: state[None, :]}
                operation = model.action
                if override is not None:
                    feed[model.diagnostic_ms] = np.array([[override]], np.float32)
                    operation = model.diagnostic_action
                return session.run(operation, feed)[0], dict(explored=False, diagnostic_override_ms=override)

            result = run_flow(cfg, cfg['conditions'][slot['scenario']], action, stopped,
                              transition=capture, frozen_seconds=seconds)
            if before != model.digest(session):
                raise ValueError('Frozen parameters changed')
    path = Path(result['folder'])
    proof = audit_flow(path, seconds)
    feedback = [json.loads(line) for line in (path/'feedback.jsonl').read_text().splitlines()]
    decisions = [json.loads(line) for line in (path/'decisions.jsonl').read_text().splitlines()]
    if len(feedback) != len(callbacks)+1 or len(callbacks) != result['transitions']:
        raise ValueError('Incomplete reward ledger')
    reward_errors, discount_errors = [], []
    for i, (seq, reward, dt, discount) in enumerate(callbacks):
        sample = feedback[i+1]
        expected_discount = .995 ** (sample['actual_dt_s']/.02)
        expected_reward = sample['original_utility']*(1.-expected_discount)/.005-.01
        if seq != i+1 or dt != sample['actual_dt_s'] or feedback[i]['state_id'] != decisions[i]['state_id']:
            raise ValueError('Reward and decision identity mismatch')
        reward_errors.append(abs(reward-expected_reward))
        discount_errors.append(abs(discount-expected_discount))
    _, offset, _ = mm(path/'down.log', seconds)
    steady = [(row, decisions[i]) for i, row in enumerate(feedback[:-1])
              if 10 <= (decisions[i]['time_us']/1000-offset)/1000 < seconds]
    if not steady:
        raise ValueError('Missing steady state')
    planned = [d['planned_ms'] for d in decisions]
    if override is not None and max(abs(x-override) for x in planned) > .001:
        raise ValueError('Override was not applied')
    if max(reward_errors)>1e-4 or max(discount_errors)>1e-7:
        raise ValueError('Actual-time reward/discount mismatch')
    for window in proof['windows']:
        selected = [d for d in decisions if window['start_s'] <=
                    (d['time_us']/1000-offset)/1000 < window['end_s']]
        window['policy_calls_per_s'] = len(selected)/(window['end_s']-window['start_s'])
    return dict(windows=proof['windows'], proof=proof, path=result['folder'], policy_sha256=before,
                planned_ms=stats(planned), actual_ms=stats([v[2]*1000 for v in callbacks]),
                cwnd=stats([r['raw'][5] for r, d in steady]),
                ssthresh=stats([r['raw'][9] for r, d in steady]),
                cwnd_below_ssthresh_fraction=float(np.mean([r['raw'][5]<r['raw'][9] for r, d in steady])),
                window_multiplier=stats([4.**d['action'][0] for r,d in steady]),
                reward_records=len(callbacks), max_reward_error=max(reward_errors),
                max_discount_error=max(discount_errors), handoff_s=proof['handoff_s'])


def reference(slot, deadline, stopped, folder):
    from frameworks.baseline.frozen_startup_contract import verify
    seconds = slot['seconds']
    command = [sys.executable,'-B',str(ROOT/'run.py'),'evaluate','--variant','baseline',
               '--run-id',slot['id'],'--scenario',str(slot['baseline_scenario']),'--seconds',str(seconds),
               '--port','23100','--seed','3']
    with (folder/(slot['id']+'.stdout')).open('x') as out, (folder/(slot['id']+'.stderr')).open('x') as err:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=out, stderr=err, start_new_session=True)
        cutoff = min(deadline-15, time.time()+seconds+35)
        while process.poll() is None:
            if stopped.wait(.25) or time.time() >= cutoff:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                raise RuntimeError('Reference reached its time limit')
        if process.returncode:
            raise RuntimeError('Reference process failed')
    path = ROOT/'frameworks/baseline/logs'/slot['id']
    proof = verify(path, seconds, 10)
    windows, offset, _ = mm(path/'down.log', seconds)
    tails = queue_tails(path/'down.log', seconds)
    with (path/'control.csv').open() as stream:
        actions = list(csv.DictReader(stream))
    for window in windows:
        assert window['complete_window']
        window.update(tails[window['window']])
        calls = sum(window['start_s'] <= (float(row['time_us'])/1000-offset)/1000 < window['end_s']
                    for row in actions)
        window['policy_calls_per_s'] = calls/(window['end_s']-window['start_s'])
    return dict(windows=windows, proof=proof, path=str(path))


def main(config_path, phase):
    plan = json.loads(Path(config_path).read_text())
    auth = checked_authorization(ROOT/'online-control'/plan['authorization'])
    runtime_preflight()
    folder = ROOT/'online-control'/(plan['batch']+'-'+phase)
    folder.mkdir(exist_ok=False)
    deadline = auth['deadline_epoch']-plan['cleanup_reserve_s']
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    results = []
    for slot in plan[phase]:
        if stopped.is_set() or time.time()+slot['seconds']+35>=deadline:
            break
        dump(folder/'current.json', slot)
        try:
            info = (reference(slot, deadline, stopped, folder) if slot['group']=='baseline'
                    else candidate(slot, deadline, stopped))
            results.append(dict(slot=slot, status='COMPLETED', **info))
        except Exception:
            results.append(dict(slot=slot, status='FAILED', error=traceback.format_exc()))
        dump(folder/'results.json', results)
        if results[-1]['status'] != 'COMPLETED':
            break
    dump(folder/'finished.json', dict(count=len(results), planned=len(plan[phase]),
         status='COMPLETED' if len(results)==len(plan[phase]) and all(r['status']=='COMPLETED' for r in results)
         else 'PARTIAL_OR_FAILED'))


if __name__ == '__main__':
    main(*sys.argv[1:])
