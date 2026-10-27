"""Finite, sequential frozen evaluations on one host. No training entrypoints."""
import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
FRAME = ROOT/'framework'
OLD = pathlib.Path(os.environ['CANOPY_E2_FRAMEWORK']).resolve()
sys.path.insert(0, str(OLD))
from mahimahi_metrics import mm
from resource_cost import summarize


def put(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def audit(folder, cfg):
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'COMPLETED', result['reason']
    assert result['policy_sha256_before'] == result['policy_sha256_after']
    assert all(v == 0 for k,v in result['counters'].items() if k != 'valid')
    actions = list(csv.DictReader((folder/'e2-actions.csv').open()))
    raw = list(csv.DictReader((folder/'raw.csv').open()))
    assert len(actions) == len(raw) and len(raw) >= 5
    assert [r['state_id'] for r in raw] == [r['state_id'] for r in actions]
    for row in actions:
        predicted = float(row['policy_prediction'])
        executed = float(row['behavior_action'])
        assert math.isfinite(predicted) and math.isfinite(executed)
        assert executed == (predicted if row['released'] == 'True' else 0.)
    released = next((a for a in actions if a['released'] == 'True'), None)
    assert released is not None
    if cfg['e2_arm'] in ('A1','A2','A3'):
        assert float(released['feedback_elapsed_s']) >= cfg['e2_warmup_s']
        event = json.loads((folder/'handover.json').read_text())
        if cfg['e2_arm'] in ('A2','A3'):
            hist = json.loads(released['history70'])
            assert all(v == 0 for v in hist[:63])
        if cfg['e2_arm'] == 'A2':
            assert event['env_peak_after_reset'] == 0
            assert abs(float(released['max_bw'])-event['raw'][1]) <= 1e-5
    if cfg['e2_arm'] == 'A4':
        assert min(float(a['max_bw']) for a in actions) >= cfg['capacity_mbps']*1e6/8
    controls = list(csv.DictReader((folder/'control.csv').open()))
    for row in controls:
        assert row['state_id'] == row['action_id'] and int(row['return_code']) == 0
        assert int(row['cwnd_requested']) == min(2147483647,max(4,int(row['percent'])*int(row['cwnd_before'])//100))
    # Kernel may legitimately change ssthresh via HyStart: record, do not hard-code INT_MAX.
    proof = dict(status='PASS_REAL_FROZEN_HANDOVER',
                 raw_feedbacks=len(raw), actions=len(actions), control_records=len(controls),
                 first_raw=raw[0], first_released=released,
                 ssthresh_values=sorted(set(int(float(r['ssthresh'])) for r in raw)),
                 protected_policy_sha256=result['policy_sha256_before'])
    put(folder/'e2-contract.json', proof)
    windows, epoch_ms, _ = mm(folder/'down.log', cfg['seconds'])
    states = list(csv.DictReader((folder/'states.csv').open()))
    for w in windows:
        start, end = w['start_s'], w['end_s']
        selected = [s for s in states if start <= float(s['time_us'])/1e6-epoch_ms/1000 < end]
        for key in ('reward','cwnd','max_bw','action_no_noise'):
            w[key+'_mean'] = sum(float(s[key]) for s in selected)/len(selected) if selected else None
        w.update(run_id=cfg['run_id'], arm=cfg['e2_arm'], condition=cfg['condition'],
                 repeat=cfg['e2_repeat'], host=os.uname().nodename,
                 measurement='Packet-log clock; full includes startup; trim10 fixed10..60s')
    cost = summarize(folder, result)
    output = dict(cfg=cfg, result=result, windows=windows, resources=cost, contract=proof)
    put(folder/'e2-metrics.json', output)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pilot','formal'], required=True)
    args = ap.parse_args()
    plan = json.loads((ROOT/'plan.json').read_text())
    jobs = plan[args.stage]
    status_file = ROOT/(args.stage+'-status.json')
    assert not status_file.exists(), 'Stage already started; do not repeat runs'
    done = []
    current = None
    started = time.time()
    try:
        for cfg in jobs:
            assert time.time()+cfg['seconds']+45 < plan['deadline_epoch'], 'Budget exhausted'
            assert not (ROOT/'STOP').exists(), 'Explicit STOP marker'
            assert pathlib.Path('/proc/sys/net/ipv4/tcp_no_metrics_save').read_text().strip() == '1'
            if __import__('shutil').disk_usage(ROOT).free < 6*2**30:
                raise RuntimeError('Disk below6GiB')
            current = cfg['run_id']
            put(status_file, dict(stage='RUNNING', epoch=time.time(), pid=os.getpid(),
                                  current=current, completed=len(done), total=len(jobs)))
            cfg_path = ROOT/'audit'/(current+'.json')
            put(cfg_path, cfg)
            flush = subprocess.run(['sudo','-n','/sbin/ip','tcp_metrics','flush','all'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            assert flush.returncode == 0, flush.stderr.decode()
            with (ROOT/'audit'/(current+'.stdout')).open('w') as out:
                proc = subprocess.Popen([sys.executable,str(ROOT/'handover_runner.py'),'--config',str(cfg_path)],
                                        stdout=out, stderr=subprocess.STDOUT)
                try:
                    rc = proc.wait(timeout=cfg['seconds']+75)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired: proc.kill();proc.wait()
                    raise
            assert rc == 0, ('Evaluator failed', rc, current)
            item = audit(FRAME/'logs'/current, cfg)
            done.append(item)
            put(ROOT/(args.stage+'-results.json'), done)
        put(status_file, dict(stage='COMPLETE', epoch=time.time(), pid=os.getpid(),
                              completed=len(done), total=len(jobs), wall_s=time.time()-started))
    except Exception:
        import traceback
        put(status_file, dict(stage='FAILED', epoch=time.time(), pid=os.getpid(), current=current,
                              completed=len(done), total=len(jobs), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
