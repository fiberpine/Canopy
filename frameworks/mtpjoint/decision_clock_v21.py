"""V21: a learned decision duration is also the next RL observation interval.
No intermediate RL observations; TCP/CUBIC ACK processing remains live.
"""
import math,json,csv,time,pathlib
import numpy as np
SCHEMA='joint86-decision-observation-clock-v21'
def enabled(cfg):return cfg.get('decision_observation_clock',False)
def validate(cfg):
 if not enabled(cfg):return
 from timing_context_v17 import state_dim
 assert cfg.get('joint_state_dim')==state_dim(cfg) and cfg.get('timing_context_features')
 from feedback_context_v31 import validate as validate_feedback
 validate_feedback(cfg)
 if cfg.get('time_statistics_v27'):assert cfg.get('time_statistics_mode') in ('rates','masked_control')
 assert cfg.get('timing_policy_mode') in ('learned','fixed')
 if cfg['timing_policy_mode']=='learned':assert cfg.get('fixed_mtp_ms') is None
 else:assert cfg.get('fixed_period_train_window') and cfg.get('fixed_mtp_ms') in (20,40,80,120,160)
 assert cfg.get('window_policy_mode')=='staged_joint'
 assert bool(cfg.get('initialize_joint_from_scratch')) != bool(cfg.get('matching_clock_prior_checkpoint'))
 assert not any(cfg.get(k) for k in ['staged_joint_checkpoint','warm_start_window_checkpoint','frozen_window_checkpoint','preemptive_mtp','diagnostic_local_mtp','startup_probe_seconds'])
 assert 0<=float(cfg.get('wide_period_probe_probability',0))<=.2
 assert cfg.get('reward_sampling_semantics')=='observed_interval_orca_density'
 assert cfg.get('time_head_nonlinearity')=='softsign' and cfg.get('control_period_ms',20)==20
 # 20ms below is only the discount/update reference unit, not a hidden sampler.
def planned_us(action,cfg):
 from joint_mtp import mtp_ms
 return int(round(mtp_ms(float(action[1]),cfg)*1000))
def wire(state_id,action,cfg):
 if action is None:return ('%d 100 0 20000\0'%state_id).encode('ascii')
 a=np.asarray(action);assert a.shape==(2,) and np.isfinite(a).all() and np.max(np.abs(a))<=1.00001
 return ('%d %d 1 %d\0'%(state_id,int(100*4**float(np.clip(a[0],-1,1))),planned_us(a,cfg))).encode('ascii')
def fresh_initialize(agent,cfg):
 import tensorflow as tf
 if cfg.get('feedback_context_v31'):
  from feedback_context_v31 import initialize
  return initialize(agent,cfg)
 if cfg.get('time_statistics_v27'):
  from time_statistics_v27 import initialize
  return initialize(agent,cfg)
 from duration_head_v16 import inverse
 from joint_mtp import mtp_action
 variables={v.op.name:v for v in tf.global_variables()};p='abstract_actor/abstract_actor/'
 if cfg.get('matching_clock_prior_checkpoint'):
  source=cfg['matching_clock_prior_checkpoint'];reader=tf.train.NewCheckpointReader(source)
  assert reader.get_variable_to_shape_map()[p+'fc1/w'][0]==86 and int(reader.get_tensor('duration_head_schema'))==1
  meta=json.loads(pathlib.Path(source+'.counters.json').read_text())['cfg'];assert meta.get('decision_observation_clock') and meta.get('reward_sampling_semantics')==cfg['reward_sampling_semantics'] and meta['gamma']==cfg['gamma'] and meta['decision_cost']==cfg['decision_cost']
  names=[n for n in variables if '/Adam' not in n and not n.startswith(('beta1_power','beta2_power')) and n not in ('global_step','timing_learning_enabled')]
  slots=[v for n,v in variables.items() if n not in names];before=agent.sess.run(slots)
  for n in names:assert n in reader.get_variable_to_shape_map() and variables[n].shape.as_list()==list(reader.get_tensor(n).shape),n
  agent.sess.run([tf.assign(variables[n],reader.get_tensor(n)) for n in names])
  for x,y in zip(before,agent.sess.run(slots)):np.testing.assert_array_equal(x,y)
  for n,x in zip(names,agent.sess.run([variables[n] for n in names])):np.testing.assert_array_equal(x,reader.get_tensor(n))
  reset=cfg.get('reset_matching_time_head',False)
  if reset:
   from duration_head_v16 import inverse
   from joint_mtp import mtp_action
   for prefix in [p,'target_actor/']:
    ww,bb=variables[prefix+'fc1_1/w'],variables[prefix+'fc1_1/b'];wv,bv=agent.sess.run([ww,bb]);wv[:,1]=0.;bv[1]=inverse(mtp_action(cfg['timing_init_ms'],cfg),cfg);agent.sess.run([tf.assign(ww,wv),tf.assign(bb,bv)])
   for n,x in zip(names,agent.sess.run([variables[n] for n in names])):
    if n.endswith(('fc1_1/w','fc1_1/b')) and n.startswith(('abstract_actor/','target_actor/')):np.testing.assert_array_equal(x[...,0],reader.get_tensor(n)[...,0])
    else:np.testing.assert_array_equal(x,reader.get_tensor(n))
  return dict(source_checkpoint=source,same_observation_schema=True,copied_source_model_exact_before_time_reset=True,window_critics_bn_unchanged=True,time_head_reset_to80ms=reset,target_time_head_also_reset=reset,fresh_optimizers_replay_counters=True,actor_gate_recalibrated=True,scratch_training=False,complete_resume=False,schema=SCHEMA)

 w=variables[p+'fc1_1/w'];b=variables[p+'fc1_1/b'];wv,bv=agent.sess.run([w,b]);wv[:,1]=0.;bv[1]=inverse(mtp_action(cfg['timing_init_ms'],cfg),cfg)
 agent.sess.run([tf.assign(w,wv),tf.assign(b,bv)]);agent.init_target()
 return dict(source_checkpoint=None,scratch_training=True,window_and_critics_random=True,time_head_initialized_ms=cfg['timing_init_ms'],fresh_optimizers_replay=True,schema=SCHEMA,complete_resume=False)
def drive(env,reader,shm_w,act,cfg,stop,state_writer,run,transition_hook=None):
 import joint_mtp as j
 from observation_noise_v14 import Observation
 from timing_context_v17 import Context
 validate(cfg);observation=Observation(cfg);context=Context(cfg)
 from reward_reference_v26 import load as load_reference,density
 reference=load_reference(cfg)
 (run/'utility-normalization.json').write_text(json.dumps(reference if reference is not None else dict(mode='observed_peak'),indent=2))
 extract=lambda x:x['state'] if isinstance(x,dict) else x
 history=np.zeros(70,np.float32);history[-7:]=observation.apply(extract(env.reset()));context.update(reader.last_raw,observation.last_multiplier)
 gaps=np.zeros(10,np.float32);gaps[-1]=float(reader.last_raw[3])
 from feedback_context_v31 import Feedback
 feedback=Feedback(cfg);feedback.observe(reader.last_raw)
 state=j.encode_state(history,0.,20.,cfg,context,gaps,feedback)
 (run/'initial-context.json').write_text(json.dumps(dict(context.snapshot(),state_id=reader.last_id,history=history.tolist(),frame_intervals_s=gaps.tolist(),schema=SCHEMA,feedback_context=feedback.snapshot() if feedback.enabled else None)))
 observations=inferences=decisions=0;physical=0.;started=time.monotonic();deadline=cfg.get('absolute_deadline_epoch',float('inf'))
 def infer(s):
  nonlocal inferences
  t=time.perf_counter();a=np.asarray(act(s),np.float32).reshape(2);ms=(time.perf_counter()-t)*1000
  assert np.isfinite(a).all() and np.max(np.abs(a))<=1.00001;inferences+=1
  return a,np.asarray(getattr(act,'no_noise',a)).copy(),ms,getattr(act,'behavior_source','frozen-policy')
 def send(a):
  reader.expected_interval_s=planned_us(a,cfg)/1e6 if a is not None else .02
  shm_w.write(wire(reader.last_id,a,cfg))
 action,nn,inference_ms,behavior=infer(state);send(action)
 with (run/'decisions.csv').open('w',buffering=262144) as f,(run/'policy-samples.jsonl').open('w',buffering=262144) as sf,(run/'feedback-context.csv').open('w',buffering=262144) as cf:
  cw=csv.writer(cf);cw.writerow(['decision','actual_dt_s','executed_window','action_valid','fresh_rtt','age_lower_log','age_upper_log','inflight_logratio','retrans_logratio','ssthresh_logratio','rtt_samples','cwnd'])
  w=csv.writer(f);w.writerow(['decision','elapsed_s','state_id','cwnd_action','mtp_action','cwnd_no_noise','mtp_no_noise','planned_mtp_ms','actual_dt_ms','duration_overrun_ms','observation_ticks','reward','raw_utility_integral','discount','inference_ms','truncated','terminal','behavior_source'])
  last_flush=started
  while not stop.is_set():
   try:info,delay,utility,ok=env.get_state(evaluation=True)
   except Exception as e:
    from evaluate import EndOfRun
    if isinstance(e,EndOfRun):
     requested=stop.is_set() or time.time()>=deadline or (cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists())
     timed=cfg.get('seconds',0) and reader.server.poll()==0 and time.time()-reader.server.created>=cfg['seconds']-.5
     if requested or timed:break
    raise
   sample=observation.apply(extract(info));raw=reader.last_raw;dt=float(raw[3])
   if not ok or sample.shape!=(7,) or not np.isfinite(sample).all() or not math.isfinite(dt) or dt<=0:raise ValueError('Invalid decision-clock feedback')
   history=np.concatenate((history[7:],sample)).astype(np.float32);gaps=np.concatenate((gaps[1:],[dt])).astype(np.float32);context.update(raw,observation.last_multiplier)
   utility=density(utility,env.max_bw,reference)
   interval=j.Interval(cfg.get('gamma',.995),.02,cfg.get('decision_cost',0));interval.add(float(utility),dt)
   physical+=dt;observations+=1;decisions+=1;plan=j.mtp_ms(action[1],cfg)
   feedback.observe(raw,action)
   next_state=j.encode_state(history,dt,plan,cfg,context,gaps,feedback)
   is_stop=bool(time.time()>=deadline or (cfg.get('seconds',0) and time.monotonic()-started>=cfg['seconds']) or (cfg.get('episode_control_seconds',0) and time.monotonic()-started>=cfg['episode_control_seconds']) or (cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists()) or (cfg.get('physical_seconds_per_actor') and physical>=cfg['physical_seconds_per_actor']) or (cfg.get('training_steps') and decisions>=cfg['training_steps']))
   old_action,old_nn,old_ms,old_behavior=action.copy(),nn.copy(),inference_ms,behavior
   if not is_stop:action,nn,inference_ms,behavior=infer(next_state);send(action)
   else:send(None)
   # Send next action before logging and asynchronous replay enqueue.
   if decisions%100==1:sf.write(json.dumps(dict(decision=decisions,state=state.tolist(),action=old_action.tolist(),no_noise=old_nn.tolist(),next_state=next_state.tolist()))+'\n')
   w.writerow([decisions,time.monotonic()-started,reader.last_id,old_action[0],old_action[1],old_nn[0],old_nn[1],plan,dt*1000,dt*1000-plan,1,interval.reward,interval.undiscounted,interval.discount,old_ms,int(is_stop and dt*1000<plan),0,old_behavior])
   if feedback.enabled:cw.writerow([decisions,dt]+feedback.record())
   if transition_hook:transition_hook(state.copy(),old_action.copy(),interval.reward,next_state.copy(),dt,interval.discount)
   state=next_state
   state_writer.writerow([time.time()*1e6,time.monotonic()-started,observations,reader.last_id,old_action[0],old_nn[0],int(100*4**float(old_action[0])),utility,raw[1]/env.max_bw,5*raw[7]/env.max_bw,sample[-1],env.max_bw,raw[5],dt,raw[8],raw[14],json.dumps(sample.tolist()),json.dumps(context.features().tolist())])
   if is_stop:break
   if time.monotonic()-last_flush>5:f.flush();last_flush=time.monotonic()
 (run/'feedback-reader.json').write_text(json.dumps(dict(snapshot_attempts=reader.snapshot_attempts,delivered=reader.delivered,paced_wait_s=reader.paced_wait_s)))
 (run/'joint-summary.json').write_text(json.dumps(dict(schema=SCHEMA,state_schema=SCHEMA,decisions=decisions,observations=observations,policy_inferences=inferences,physical_exposure_s=physical,unobserved_tail_dropped=True,final_observed_partial_transition=False,reward_sampling_semantics=cfg['reward_sampling_semantics'],scheduling='Native waits commanded microseconds after applying each action; only then reads TCP. Actual sample-to-sample duration includes measured dispatch latency.',terminal_rule='Administrative truncations bootstrap real next state; no cross-connection samples'),indent=2))
 return observations
