"""Frozen deployment-mode driver: no Python per-step log serialization/I/O."""
import json,math,pathlib,time
import numpy as np
CAPTURES={}


def drive(env,reader,shm_w,act,cfg,stop,state_writer,run,transition_hook=None):
    if transition_hook is not None or not cfg.get('no_training') or not cfg.get('capture_raw_state'):raise ValueError('Frozen deployment measurement only')
    import joint_mtp as j
    from decision_clock_v21 import validate,wire,planned_us
    from observation_noise_v14 import Observation
    from timing_context_v17 import Context
    from feedback_context_v31 import Feedback
    validate(cfg);observation=Observation(cfg);context=Context(cfg);feedback=Feedback(cfg)
    extract=lambda x:x['state'] if isinstance(x,dict) else x
    history=np.zeros(70,np.float32);history[-7:]=observation.apply(extract(env.reset()));raw=reader.last_raw
    context.update(raw,observation.last_multiplier);gaps=np.zeros(10,np.float32);gaps[-1]=float(raw[3]);feedback.observe(raw)
    state=j.encode_state(history,0.,20.,cfg,context,gaps,feedback)
    initial_id=reader.last_id;old_id=initial_id;full=[]
    CAPTURES[str(run)]=dict(initial_state=state.copy(),initial_raw=raw.copy(),initial_id=initial_id,rows=full)
    started=time.monotonic();physical=0.;calls=0;rows=[];reward_integral=0.
    def infer(s):
        nonlocal calls
        start=time.perf_counter();action=np.asarray(act(s),np.float32).reshape(2);elapsed=(time.perf_counter()-start)*1000
        if not np.isfinite(action).all() or np.max(np.abs(action))>1.00001:raise ValueError('Invalid frozen action')
        calls+=1;return action,np.asarray(getattr(act,'no_noise',action)).copy(),elapsed
    def send(action):
        reader.expected_interval_s=planned_us(action,cfg)/1e6 if action is not None else .02
        shm_w.write(wire(reader.last_id,action,cfg))
    action,nn,inference=infer(state);send(action)
    while not stop.is_set():
        try:info,delay,u,ok=env.get_state(evaluation=True)
        except Exception as exc:
            from evaluate import EndOfRun
            if isinstance(exc,EndOfRun):
                requested=stop.is_set() or time.time()>=cfg['absolute_deadline_epoch'] or (cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists())
                timed=cfg.get('seconds',0) and reader.server.poll()==0 and time.time()-reader.server.created>=cfg['seconds']-.5
                if requested or timed:break
            raise
        raw=reader.last_raw;dt=float(raw[3]);sample=observation.apply(extract(info))
        if not ok or not math.isfinite(dt) or dt<=0 or not np.isfinite(sample).all():raise ValueError('Invalid boundary')
        physical+=dt;history=np.concatenate((history[7:],sample)).astype(np.float32);gaps=np.r_[gaps[1:],np.float32(dt)].astype(np.float32)
        context.update(raw,observation.last_multiplier);feedback.observe(raw,action);plan=j.mtp_ms(action[1],cfg)
        previous_state=state
        state=j.encode_state(history,dt,plan,cfg,context,gaps,feedback)
        rows.append((dt,float(nn[0]),plan,inference,float(u)))
        full.append((previous_state,action.copy(),state,raw.copy(),float(u),float(inference),old_id,reader.last_id))
        old_id=reader.last_id
        reward_integral+=float(u)*dt
        end=time.time()>=cfg['absolute_deadline_epoch'] or time.monotonic()-started>=cfg['seconds'] or (cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists())
        if end:send(None);break
        action,nn,inference=infer(state);send(action)
    a=np.asarray(rows,float)
    summary=dict(schema='common-period-deployment-no-step-logs-v1',decisions=len(rows),observations=len(rows),policy_inferences=calls,physical_exposure_s=physical,python_step_logs=False,unobserved_tail_dropped=True,
                 cwnd_low_saturation=float(np.mean(a[:,1]<=-.99)) if len(rows) else None,cwnd_high_saturation=float(np.mean(a[:,1]>=.99)) if len(rows) else None,
                 cwnd_abs_gt_099=float(np.mean(np.abs(a[:,1])>.99)) if len(rows) else None,
                 planned_mtp_ms_mean=float(np.mean(a[:,2])) if len(rows) else None,inference_ms_mean=float(np.mean(a[:,3])) if len(rows) else None,
                 time_weighted_raw_utility=reward_integral/max(1e-9,physical))
    (run/'joint-summary.json').write_text(json.dumps(summary,indent=2));return len(rows)

def install(evaluate):
    import joint_mtp
    original=joint_mtp.drive
    def select(env,reader,shm_w,act,cfg,stop,state_writer,run,transition_hook=None):
        if cfg.get('deployment_no_step_logs'):return drive(env,reader,shm_w,act,cfg,stop,state_writer,run,transition_hook)
        return original(env,reader,shm_w,act,cfg,stop,state_writer,run,transition_hook)
    joint_mtp.drive=select


def flush_captures():
    """Called only after evaluate.main has stopped native and released IPC."""
    results=[]
    for name,c in list(CAPTURES.items()):
        run=pathlib.Path(name);rows=c['rows'];path=run/'captured-boundaries.npz'
        if path.exists():raise ValueError('Never overwrite captured data')
        arrays=dict(initial_state=c['initial_state'],initial_raw=c['initial_raw'],initial_id=np.asarray(c['initial_id']),s0=np.asarray([x[0] for x in rows],np.float32).reshape(-1,149),action=np.asarray([x[1] for x in rows],np.float32).reshape(-1,2),s1=np.asarray([x[2] for x in rows],np.float32).reshape(-1,149),raw_next=np.asarray([x[3] for x in rows],np.float64).reshape(-1,15),utility=np.asarray([x[4] for x in rows],np.float64),inference_ms=np.asarray([x[5] for x in rows],np.float64),state_id=np.asarray([x[6] for x in rows],np.int64),next_state_id=np.asarray([x[7] for x in rows],np.int64))
        np.savez_compressed(str(path),**arrays);results.append(dict(run=name,rows=len(rows),file=str(path)))
    CAPTURES.clear();return results
