"""Finite one-learner/one-actor zero-optimizer checks through actual training entrypoints."""
import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import traceback

ROOT=pathlib.Path(os.environ['CANOPY_IPC_DIAGNOSTIC_ROOT']).resolve()
FRAME=ROOT/'frameworks/canopy_single'
PY=sys.executable


def put(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def ipc_proof(folder):
    py=list(csv.DictReader((folder/'ipc-python.csv').open()))
    native=list(csv.DictReader((folder/'ipc-native.csv').open()))
    matched=0;unmatched=[]
    for sender,receiver in [(native,py),(py,native)]:
        sent={(int(r['kind']),int(r['sequence'])):r for r in sender if r['direction']=='sent'}
        received={(int(r['kind']),int(r['sequence'])):r for r in receiver if r['direction']=='received'}
        for key,r in received.items():
            assert key in sent,('unpublished received frame',key)
            assert r['crc']==sent[key]['crc'] and r['payload']==sent[key]['payload'],('echo mismatch',key)
            matched+=1
        unmatched+=list(set(sent)-set(received))
    # Finite stop may leave one final state/action at either end, never a mismatched accepted frame.
    assert len(unmatched)<=2,unmatched
    stats=json.loads((folder/'ipc-python-stats.json').read_text())
    assert stats['incoming']['rejected']==stats['outgoing']['rejected']==0
    assert 'IPC_V2_REJECT' not in (folder/'server.log').read_text()
    return dict(status='PASS_BOTH_DIRECTIONS',matched_frames=matched,unaccepted_terminal_frames=unmatched,
                received_crc_or_sequence_failures=0,action_state_binding='Native enforces every accepted action')


def tensor_proof(cp,final):
    import tensorflow as tf
    import numpy as np
    before=tf.train.NewCheckpointReader(str(cp/'step-0'))
    after=tf.train.NewCheckpointReader(str(cp/('step-%d'%final)))
    names=before.get_variable_to_shape_map();assert names==after.get_variable_to_shape_map()
    changed=[]
    for name in names:
        a,b=before.get_tensor(name),after.get_tensor(name)
        if not np.array_equal(a,b):changed.append(name)
        assert np.isfinite(b).all(),name
    assert not changed,changed
    return dict(status='PASS_ALL_TENSORS_UNCHANGED',variables=len(names),changed=changed)


def measure(cfg):
    folder=FRAME/'logs'/(cfg['run_id']+'-actor0')
    result=json.loads((folder/'result.json').read_text())
    learner=json.loads((FRAME/'logs'/cfg['run_id']/'result.json').read_text())
    assert result['status']==learner['status']=='COMPLETED',(result,learner)
    assert all(v==0 for v in learner['counters'].values())
    records=list(csv.DictReader((folder/'states.csv').open()))
    controls=list(csv.DictReader((folder/'control.csv').open()))
    seconds=sum(float(r['delta_t']) for r in records)
    assert seconds>=cfg['probe_seconds']-.5,(seconds,cfg['probe_seconds'])
    c=cfg['conditions'][0]
    bdp=c['capacity_mbps']*1e6/8*(c['base_rtt_ms']/1000)/c.get('packet_bytes',1504)
    bound=4*(bdp+c['queue_packets'])
    observed=max(float(r['cwnd']) for r in records)
    requested=max(int(r['cwnd_requested']) for r in controls)
    mass=sum(float(r['delta_t']) for r in records if float(r['cwnd'])>bound)
    lowmass=sum(float(r['delta_t']) for r in records if float(r['cwnd'])<=10)
    actions=[float(r['action_no_noise']) for r in records]
    out=dict(run_id=cfg['run_id'],mode=cfg['scalar_window_initialization'],seed=cfg['seed'],condition=c['condition'],
             physical_exposure_s=seconds,transitions=learner['inserted'],optimizer_counters=learner['counters'],
             observed_max_cwnd=observed,requested_max_cwnd=requested,bdp_packets=bdp,queue_packets=c['queue_packets'],
             registered_window_bound=bound,boundary_weighted_seconds_above_bound=mass,
             boundary_weighted_fraction_above_bound=mass/seconds,low_window_fraction=lowmass/seconds,
             deterministic_action_mean=sum(actions)/len(actions),deterministic_action_min=min(actions),
             deterministic_action_max=max(actions),first32_no_noise_actions=actions[:32],
             million_window_pass=max(observed,requested)<1e6,four_bdp_queue_time_pass=mass/seconds<.01,
             mean_reward=sum(float(r['reward']) for r in records)/len(records),
             ipc=ipc_proof(folder),tensors=tensor_proof(FRAME/'checkpoints'/cfg['model_id'],learner['inserted']),
             limitation='Boundary actual_dt weighting approximates duration; no hidden intermediate TCP samples. Raw and requested maxima both retained. No learned-policy quality/CPU claim.')
    # Full packet quality retained even though this is an initialization/exploration diagnostic.
    sys.path.insert(0,str(FRAME));from mahimahi_metrics import mm
    windows,_,_=mm(folder/'down.log',cfg['probe_seconds']);out['packet_windows']=windows
    # Additive, read-only diagnostics (exploration-process check): arm label, window binding from the
    # verified state payloads, and the log-window decomposition. A diagnostic failure never hides a gate.
    out['exploration_process']=cfg.get('exploration_process','iid');out['exploration_offset_rho']=cfg.get('exploration_offset_rho')
    try:
        import window_binding
        out['window_diagnostics']=window_binding.analyze(folder)
        json.dumps(out['window_diagnostics'],allow_nan=False)   # a NaN must not break the stage's strict put()
    except Exception as e:
        out['window_diagnostics']=dict(status='FAILED',reason=repr(e))
    return out


def run_one(cfg,deadline):
    rid=cfg['run_id'];path=ROOT/'configs'/(rid+'.json');put(path,cfg)
    children=[]
    try:
        for role in ['learner','actor']:
            f=(ROOT/'audit'/(rid+'-'+role+'.stdout')).open('w')
            p=subprocess.Popen([PY,'-B',str(FRAME/'distributed.py'),'--config',str(path),'--role',role],
                               stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p);f.close()
            marker=FRAME/'logs'/rid/'ready' if role=='learner' else FRAME/'logs'/(rid+'-actor0')/'flow-ready'
            until=min(deadline,time.time()+100)
            while not marker.exists():
                if any(p.poll() is not None for p in children) or time.time()>until:
                    raise RuntimeError('Failed startup '+role+'; inspect exact stdout')
                time.sleep(.1)
        started=float(marker.read_text());end=started+cfg['probe_seconds']
        marks=sorted(cfg.get('ss_snapshot_seconds',[]));snaps=[]
        while time.time()<end:
            if any(p.poll() is not None for p in children):raise RuntimeError('Early worker exit')
            assert time.time()<deadline
            if marks and time.time()>=started+marks[0]:
                # Optional read-only server socket snapshot: tells receive-window from send-buffer limits.
                marks.pop(0)
                try:
                    import window_binding
                    snaps.append(dict(at_s=time.time()-started,**window_binding.ss_snapshot(cfg.get('actor_port_base',48200))))
                except Exception as e:snaps.append(dict(at_s=time.time()-started,status='FAILED',reason=repr(e)))
            time.sleep(.2)
        if snaps:put(ROOT/'audit'/(rid+'-ss.json'),snaps)
        pathlib.Path(cfg['stop_file']).write_text('FINITE_ZERO_OPTIMIZER_EXPOSURE_COMPLETE')
        for p in children:
            rc=p.wait(timeout=120);assert rc==0,('Worker failed',rc)
        return measure(cfg)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','formal'],required=True)
    args=ap.parse_args();plan=json.loads((ROOT/'registration.json').read_text());jobs=plan[args.stage]
    status=ROOT/(args.stage+'-status.json');assert not status.exists(),'Never repeat a started stage'
    if args.stage=='formal':assert json.loads((ROOT/'smoke-status.json').read_text())['stage']=='COMPLETE'
    try:
        import window_binding
        put(ROOT/(args.stage+'-host-receipt.json'),window_binding.host_receipt())   # TCP buffer sysctls, read-only
    except Exception as e:
        put(ROOT/(args.stage+'-host-receipt.json'),dict(status='FAILED',reason=repr(e)))
    results=[]
    try:
        for cfg in jobs:
            assert time.time()+cfg['probe_seconds']+130<plan['deadline_epoch'],'Finite budget exhausted'
            assert not (ROOT/'STOP').exists(),'Explicit STOP'
            put(status,dict(stage='RUNNING',epoch=time.time(),pid=os.getpid(),current=cfg['run_id'],completed=len(results),total=len(jobs)))
            subprocess.run(['sudo','-n','/sbin/ip','tcp_metrics','flush','all'],check=True,timeout=10)
            results.append(run_one(cfg,plan['deadline_epoch']))
            put(ROOT/(args.stage+'-results.json'),results)
        put(status,dict(stage='COMPLETE',epoch=time.time(),completed=len(results),total=len(jobs)))
    except Exception:
        put(status,dict(stage='FAILED',epoch=time.time(),completed=len(results),total=len(jobs),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
