"""Post-run identity/timing/action checks; cannot affect the live controller."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from .core import period_ms, wire
from .flow import dump


def audit_flow(folder, frozen_seconds=None):
    folder = Path(folder)
    result = json.loads((folder / 'result.json').read_text())
    cfg = json.loads((folder / 'config.json').read_text())['cfg']
    if result['status'] != 'COMPLETED': raise ValueError('Incomplete TCP flow')
    def csv_rows(name):
        with (folder / name).open() as stream: return list(csv.DictReader(stream))
    decisions = [json.loads(x) for x in (folder / 'decisions.jsonl').read_text().splitlines()]
    feedback = [json.loads(x) for x in (folder / 'feedback.jsonl').read_text().splitlines()]
    controls, timings, boundaries = csv_rows('control.csv'), csv_rows('timing.csv'), csv_rows('boundaries.csv')
    if not decisions or len(controls) != len(timings): raise ValueError('Missing native execution records')
    applied = [(c, t) for c, t in zip(controls, timings) if int(t['apply']) == 1]
    noop = [i for i, timing in enumerate(timings) if int(timing['apply']) == 0]
    if len(noop) > 1 or (noop and noop[0] != len(timings) - 1): raise ValueError('Unexpected no-op inside the policy sequence')
    if len(applied) != len(decisions): raise ValueError('Policy-call/executed-command mismatch')
    for (control, timing), decision in zip(applied, decisions):
        if int(control['state_id']) != decision['state_id'] or control['state_id'] != control['action_id'] or control['state_id'] != timing['state_id']:
            raise ValueError('Decision/state ID binding mismatch')
        # TF1 tanh may overshoot an endpoint by one float32 ULP. The sender
        # already validates then clips; audit the exact serialized command.
        percent = int(wire(decision['state_id'],decision['action']).split()[1])
        requested = min(2147483647, max(4, percent * int(control['cwnd_before']) // 100))
        if percent != int(control['percent']) or requested != int(control['cwnd_requested']) or int(control['return_code']) != 0:
            raise ValueError('Native integer action mapping/application error')
        if abs(int(timing['planned_wait_us']) - period_ms(decision['action'][1]) * 1000) > .501:
            raise ValueError('Native scheduled period mismatch')
    if len(boundaries) < len(feedback): raise ValueError('Missing real observation endpoint records')
    normalization_peak=0.
    for index, item in enumerate(feedback):
        if int(boundaries[index]['state_id']) != item['state_id']: raise ValueError('Feedback/native boundary mismatch')
        if bool(item['fresh_rtt']) != (float(item['raw'][2]) > 0): raise ValueError('Fabricated RTT freshness')
        if abs(float(item['actual_dt_s']) * 1e6 - int(boundaries[index]['dt_us'])) > .11:
            raise ValueError('Physical interval does not match the native clock')
        if index and item['state_id'] != (feedback[index - 1]['state_id'] + 1) % 1000:
            raise ValueError('Feedback ID gap')
        if index and cfg['clock_mode'] == 'scheduled' and int(boundaries[index]['kernel_queries']) - int(boundaries[index - 1]['kernel_queries']) != 1:
            raise ValueError('Unexpected socket telemetry read between scheduled boundaries')
        if cfg.get('rate_mode')=='acked_legacy_peak':
            boundary=boundaries[index];raw=item['raw'];dt=int(boundary['dt_us'])/1e6
            if not math.isclose(float(raw[1]),int(boundary['delta_bytes'])/dt,rel_tol=2e-6,abs_tol=1e-6):
                raise ValueError('ACK numerator does not match its real counter interval')
            estimator=float(boundary['delivery_sample_mean'])
            if not math.isclose(float(raw[4]),estimator,rel_tol=1e-7,abs_tol=1e-5):
                raise ValueError('Legacy peak estimator is not bound to the same native snapshot')
            normalization_peak=max(normalization_peak,estimator)
            utility=(float(raw[1])-5*float(raw[7]))/normalization_peak*min(1.,1.25*float(raw[14])/float(raw[8]))
            if not math.isclose(item['max_bw'],normalization_peak,rel_tol=1e-7) or not math.isclose(item['original_utility'],utility,rel_tol=1e-7,abs_tol=1e-7):
                raise ValueError('Reward/state normalizer does not match declared legacy estimator peak')
    lines = [l for l in (folder / 'server.log').read_text().splitlines() if 'ORCA_STARTUP_HANDOFF ' in l]
    if len(lines) != 1: raise ValueError('Missing/duplicate natural handoff')
    startup = dict(x.split('=', 1) for x in lines[0].split('ORCA_STARTUP_HANDOFF ', 1)[1].split())
    if startup['mode'] != 'natural-cubic' or startup['scheme'] != 'cubic' or startup['assist_writes'] != '0' or int(startup['ssthresh']) >= int(startup['cwnd']):
        raise ValueError('Natural CUBIC startup contract changed')
    proof = dict(status='PASS_EXECUTION_CONTRACT_ONLY', controls=len(applied), feedback=len(feedback), startup=startup,
                 no_hidden_periodic_rl_observations=cfg['clock_mode'] == 'scheduled', mean_empty_rtt_fraction=float(np.mean([not x['fresh_rtt'] for x in feedback])),
                 performance_health_claim=False)
    if frozen_seconds:
        from frameworks.baseline.mahimahi_metrics import mm
        from .metrics import queue_tails, cpu_summary
        windows, offset, bins = mm(folder / 'down.log', frozen_seconds)
        tails = queue_tails(folder / 'down.log', frozen_seconds)
        for window in windows: window.update(tails[window['window']])
        if not all(w['complete_window'] for w in windows): raise ValueError('Incomplete packet-clock window')
        handoff = (float(controls[0]['time_us']) / 1000. - offset) / 1000.
        if not 0 <= handoff <= 10: raise ValueError('Policy handoff later than10 packet seconds')
        proof.update(windows=windows, handoff_s=handoff, cpu=cpu_summary(folder, result))
    dump(folder / 'execution-audit.json', proof)
    return proof
