"""Frozen scalar evaluator with auditable, isolated startup interventions.

Original evaluator is copied byte-for-byte into a fresh evaluation workspace.
Three exact substitutions add hooks; original files/checkpoints remain untouched.
"""
import csv
import os
import hashlib
import json
import pathlib
import sys
import time
import types

from e2_support import Handover

ROOT = pathlib.Path(__file__).resolve().parent
FRAME = ROOT/'framework'
OLD = pathlib.Path(os.environ['CANOPY_E2_FRAMEWORK']).resolve()
cfg_path = pathlib.Path(sys.argv[sys.argv.index('--config')+1])
cfg = json.loads(cfg_path.read_text())
assert cfg['method'] == 'orca' and cfg['no_training'] is True
assert cfg['seconds'] <= 60 and cfg['e2_arm'] in ('A0','A1','A2','A3','A4')
sys.path.insert(0, str(OLD))
source = (FRAME/'evaluate.py').read_text()


def replace_once(old, new):
    global source
    assert source.count(old) == 1, ('evaluator source differs', old)
    source = source.replace(old, new)


replace_once('self.last_id=rid;self.last_raw=vals;return value+b\'\\0\'',
             'self.last_id=rid;self.last_raw=vals;on_raw_feedback(self);return value+b\'\\0\'')
replace_once('s=state(env.reset());hist=np.zeros(70,np.float32);hist[-7:]=s',
             'handover.env=env;s=state(env.reset());hist=handover.history(None,s)')
replace_once('h1=np.concatenate((hist[7:],s1)).astype(np.float32)',
             'h1=handover.history(hist,s1)')
module = types.ModuleType('evaluate')
module.__file__ = str(FRAME/'evaluate.py')
sys.modules['evaluate'] = module
exec(compile(source, module.__file__, 'exec'), module.__dict__)
handover = Handover(cfg['e2_arm'], cfg['e2_warmup_s'], cfg['capacity_mbps']*1e6/8)
folder = FRAME/'logs'/cfg['run_id']
# Base main owns creation of folder and must still fail if a run ID exists.
raw_file = decision_file = None
raw_writer = decision_writer = None
raw_count = decisions = 0
latest_id = None


def on_raw_feedback(reader):
    global raw_file, raw_writer, raw_count, latest_id
    if raw_file is None:
        raw_file = (folder/'raw.csv').open('w', buffering=262144)
        raw_writer = csv.writer(raw_file)
        raw_writer.writerow(['epoch_s','feedback_elapsed_s','state_id','released',
                             'avg_urtt_ms','throughput_sample_mean_Bps','rtt_samples','actual_dt_s',
                             'target','cwnd','pacing_Bps','loss_Bps','srtt_ms','ssthresh',
                             'packets_out','retrans_out','max_packets_out','mss','min_rtt_ms'])
    now = time.monotonic()
    previous = handover.released
    handover.boundary(now, handover.env)
    if previous != handover.released:
        event = dict(handover.event, epoch=time.time(), state_id=reader.last_id,
                     raw=reader.last_raw.tolist(), env_peak_after_reset=handover.env.max_bw)
        (folder/'handover.json').write_text(json.dumps(event, indent=2))
    raw_writer.writerow([time.time(), now-handover.first_time, reader.last_id,
                         handover.released]+reader.last_raw.tolist())
    latest_id = reader.last_id
    raw_count += 1


original_model = module.model


def model(c):
    a, sess, wrapper, params, predict, digest, variables = original_model(c)

    def action(history):
        global decision_file, decision_writer, decisions
        prediction = predict(history)
        executed = handover.action(prediction)
        if decision_file is None:
            decision_file = (folder/'e2-actions.csv').open('w', buffering=262144)
            decision_writer = csv.writer(decision_file)
            decision_writer.writerow(['epoch_s','feedback_elapsed_s','state_id','released',
                                      'policy_prediction','behavior_action','max_bw','history70'])
        decision_writer.writerow([time.time(), time.monotonic()-handover.first_time,
                                  latest_id, handover.released, prediction, executed,
                                  handover.env.max_bw, json.dumps(history.tolist())])
        decisions += 1
        return executed

    return a, sess, wrapper, params, action, digest, variables


module.model = model
module.handover = handover
module.on_raw_feedback = on_raw_feedback
try:
    module.main()
finally:
    for f in (raw_file, decision_file):
        if f:
            f.close()
    if folder.exists():
        (folder/'e2-provenance.json').write_text(json.dumps(dict(
            arm=cfg['e2_arm'], patched_evaluator_sha256=hashlib.sha256(source.encode()).hexdigest(),
            original_evaluator_sha256=hashlib.sha256((FRAME/'evaluate.py').read_bytes()).hexdigest(),
            raw_feedbacks=raw_count, decisions=decisions, released=handover.released,
            actor_updates=0, optimizer_updates=0,
            history_reset='Zero63 + current7, before first released policy query',
            reference_reset='A2 zero before original wrapper normalizes boundary feedback',
            warmup='A1/A2/A3 execute action0 (100 percent) at real20ms boundaries; shadow actor queried and logged',
            oracle='A4 only: nominal configured capacity floor changes actor normalization; diagnostic only'), indent=2))
