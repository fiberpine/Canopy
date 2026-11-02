"""Frozen-only diagnostic oracle; never a deployable learned policy.

Override only at existing decision boundaries, without preempting an interval.
Mahimahi log epoch is used, not controller launch time or stale config segments.
"""
import csv
import hashlib
import json
import math
import time


def select(time_s, predicted_ms, specification):
    chosen = None
    for interval in specification.get('windows', []):
        if interval['start_s'] <= time_s < interval['end_s']:
            if chosen is not None:
                raise ValueError('overlapping diagnostic windows')
            chosen = float(interval['mtp_ms'])
    return (predicted_ms, False) if chosen is None else (chosen, True)


def encoded_choice(original_action,chosen_ms,cfg):
    import joint_mtp
    predicted_ms=joint_mtp.mtp_ms(original_action,cfg)
    # Preserve exact float32 output if no timing intervention is requested.
    return original_action if chosen_ms==predicted_ms else joint_mtp.mtp_action(chosen_ms,cfg)


class Probe:
    def __init__(self, cfg, run):
        self.spec = cfg.get('diagnostic_local_mtp', {'windows': []})
        self.overrides = self.calls = 0
        trace = run.parents[1]/'configs/traces'/cfg['trace']
        actual = hashlib.sha256(trace.read_bytes()).hexdigest()
        if actual != cfg['trace_sha256'] or self.spec.get('trace_sha256', actual) != actual:
            raise ValueError('diagnostic event source trace hash differs')
        windows = sorted(self.spec.get('windows', []), key=lambda x:x['start_s'])
        for i,w in enumerate(windows):
            if not (math.isfinite(w['mtp_ms']) and 20<=w['mtp_ms']<=160 and
                    10<=w['start_s']<w['end_s']<=60):
                raise ValueError('invalid diagnostic window')
            if i and windows[i-1]['end_s']>w['start_s']:
                raise ValueError('overlapping diagnostic windows')
        base = init = None
        with (run/'down.log').open() as f:
            for unused in range(32):
                line = f.readline()
                if line.startswith('# base timestamp:'):base=float(line.split(':')[1])
                if line.startswith('# init timestamp:'):init=float(line.split(':')[1])
                if base is not None and init is not None:break
        if init is None or base is None:
            raise ValueError('Mahimahi clock headers unavailable after first feedback')
        self.epoch_ms = init+base
        self.run = run
        self.file = (run/'diagnostic-mtp-choices.csv').open('w', buffering=262144)
        self.writer = csv.writer(self.file)
        self.writer.writerow(['inference','wall_epoch_s','trace_time_s','predicted_mtp_ms',
            'executed_mtp_ms','overridden'])

    def choose(self,predicted_ms):
        now = time.time()
        trace_s = (now*1000-self.epoch_ms)/1000
        if not -2<=trace_s<=65:raise ValueError('Unexpected trace/epoch clock offset')
        chosen,overridden = select(trace_s,predicted_ms,self.spec)
        self.calls += 1
        self.overrides += int(overridden)
        self.writer.writerow([self.calls,now,trace_s,predicted_ms,chosen,int(overridden)])
        return chosen

    def close(self):
        self.file.close()
        (self.run/'diagnostic-mtp-summary.json').write_text(json.dumps(dict(
            schema='frozen-local-mtp-oracle-v11',policy_calls=self.calls,
            override_calls=self.overrides,trace_epoch_ms=self.epoch_ms,
            specification=self.spec,preemption=False,
            training_allowed=False,window_weights_unchanged=True,
            semantics='Override executed timing only at an existing decision boundary; raw no-noise prediction retained separately; oracle labels diagnostic only'),indent=2))
