"""Small Canopy TF1 parameter-server training. Official model and raw Orca reward.
Actors use shared learner variables and a shared TF FIFO, as the official launcher.
Flow control never waits for replay credits or learner optimization.
"""
import pathlib,os,sys,time,json,threading,queue,random,csv,hashlib,traceback
R=pathlib.Path(__file__).resolve().parent
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(R/'code/runtime/_deps'));sys.path.insert(0,str(R/'code/canopy/rl-module'))
import numpy as np,tensorflow as tf,psutil
import agent_v2
from agent_v2 import Agent
from utils_v2 import ReplayBuffer
import update_credit
import staged_timing

def setup(cfg,role,index):
 assert cfg['warmup']>=(512 if cfg.get('portable_smoke_training') else max(10000,20*512))
 joint=cfg.get('joint_control',False)
 if joint:
  import joint_mtp
  joint_mtp.validate(cfg)
 sd,ad=(72,2) if joint else (70,1)
 if cfg.get('update_credit_basis','decision')!='decision':
  assert joint and cfg['update_credit_basis']=='physical_exposure'
  assert 0<cfg.get('updates_per_base20ms',.08)<=1
 assert cfg.get('reward_mode','raw')=='raw','This baseline implements raw Orca reward only; constraint-reward training requires a separate verified backend.'
 if cfg.get('backend')=='moving_stats':
  import moving_stats_backend
  moving_stats_backend.install(agent_v2)
 if cfg.get('target_bn_consistent'):
  import consistent_target_bn
  consistent_target_bn.install(agent_v2)
 if staged_timing.enabled(cfg):
  import frozen_window,trainable_timing
  if trainable_timing.enabled(cfg):trainable_timing.install(agent_v2,cfg)
  else:frozen_window.install(agent_v2,cfg)
 random.seed(cfg['seed']+index*1009);np.random.seed(cfg['seed']+index*1009);tf.set_random_seed(cfg['seed'])
 cluster=tf.train.ClusterSpec({'learner':[cfg['learner_endpoint']]} if cfg.get('direct_actor_session',False) else {'learner':[cfg['learner_endpoint']],'actor':['127.0.0.1:%d'%(47101+i) for i in range(cfg['num_actors'])]})
 direct=role=='actor' and cfg.get('direct_actor_session',False)
 server=None if direct else tf.train.Server(cluster,job_name=role,task_index=index,config=tf.ConfigProto(intra_op_parallelism_threads=cfg.get("tf_threads",2),inter_op_parallelism_threads=cfg.get("tf_threads",2)))
 session_target='grpc://'+cfg['learner_endpoint'] if direct else server.target
 with tf.device('/job:learner/task:0/cpu:0'):
  a=Agent(sd,ad,256,256,gamma=cfg.get('gamma',.995),batch_size=512,lr_a=.0001,lr_c=.001,tau=.001,mem_size=cfg['memsize'] if role=='learner' else 1,noise_type=3,noise_exp=50000,stddev=cfg.get('exploration_std',.2),PER=False,CDQ=True,LOSS_TYPE=cfg.get('LOSS_TYPE','MSE'),use_original=False,use_snt_model_wo_ibp=False,smdp=joint,fixed_mtp_action=joint_mtp.fixed_action(cfg) if joint else None)
  if cfg.get('target_bn_consistent'):consistent_target_bn.connect_statistics(a,.001)
  shapes=[[sd],[ad],[1],[sd],[1]]+([[1],[1]] if joint else [])
  fifo=tf.FIFOQueue(10000,[tf.float32]*len(shapes)+[tf.int32],shapes=shapes+[[]],shared_name='rp_buf')
  ph=[tf.placeholder(tf.float32,x) for x in shapes]+[tf.placeholder(tf.int32,[])]
  enqueue=fifo.enqueue(ph);nph=tf.placeholder(tf.int32,[]);dequeue=fifo.dequeue_many(nph);size=fifo.size()
  if role=='learner':a.build_learn()
 if direct:assert not any('/job:actor/' in op.device for op in tf.get_default_graph().get_operations()),'Direct actor session requires every model/queue operation to remain learner-placed'
 sess=tf.Session(session_target,config=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=cfg.get("tf_threads",2),inter_op_parallelism_threads=cfg.get("tf_threads",2)));a.assign_sess(sess)
 return a,sess,fifo,ph,enqueue,nph,dequeue,size

class Replay(ReplayBuffer):
 def __init__(self,cfg):
  self.joint=cfg.get('joint_control',False)
  self.warmup_physical_seconds=None
  self.period_bank=None
  if cfg.get('period_bank_enabled'):
   from period_coverage import PeriodBank
   assert cfg.get('period_bank_fraction')==.25,'V8 uses an explicit quarter-batch bank'
   self.period_bank=PeriodBank(cfg)
  fifo_capacity=cfg['memsize']-(self.period_bank.capacity if self.period_bank else 0)
  super().__init__(fifo_capacity,72 if self.joint else 70,2 if self.joint else 1,512);self.duration=np.zeros((self.size,1),np.float32);self.discounts=np.zeros((self.size,1),np.float32);self.actor_seconds=np.zeros(cfg['num_actors'],np.float64);self.lock=threading.RLock();self.inserted=0;self.birth=np.zeros(self.size,np.int64);self.cfg=cfg;self.ages=[];self.actor_ids=np.full(self.size,-1,np.int32);self.actor_inserted=np.zeros(cfg['num_actors'],np.int64);self.actor_slots=[[] for _ in range(cfg['num_actors'])];self.slot_position=np.full(self.size,-1,np.int32);self.sampled_by_actor=np.zeros(cfg['num_actors'],np.int64);self.sample_calls=0
  self.period_ids=np.full(self.size,-1,np.int32)
  self.retained_period_counts=np.zeros((cfg['num_actors'],len(cfg.get('timing_calibration_periods_ms',[20,40,80,120,160]))),np.int64)
 def append_many(self,data):
  with self.lock:
   for row,actor_id in zip(zip(*data[:-1]),data[-1]):
    values=row[:5]
    assert all(np.all(np.isfinite(v)) for v in values),'nonfinite transition'
    assert np.all(np.abs(values[1])<=1.00001)
    if self.joint:
     duration,discount=float(row[5][0]),float(row[6][0]);assert np.isfinite(duration) and duration>0 and np.isfinite(discount) and 0<discount<=1
     assert np.isclose(discount,self.cfg.get('gamma',.995)**(duration/.02),rtol=2e-5,atol=1e-8),'time/discount mismatch'
    actor_id=int(actor_id);assert 0<=actor_id<self.cfg['num_actors'];old=int(self.actor_ids[self.ptr])
    if old>=0:
     old_bin=int(self.period_ids[self.ptr])
     if old_bin>=0:self.retained_period_counts[old,old_bin]-=1
     pool=self.actor_slots[old];position=int(self.slot_position[self.ptr]);moved=pool[-1];pool[position]=moved;self.slot_position[moved]=position;pool.pop()
    period=staged_timing.period_bin(values[1][1],self.cfg) if staged_timing.enabled(self.cfg) else -1
    self.period_ids[self.ptr]=period
    if period>=0:self.retained_period_counts[actor_id,period]+=1
    self.slot_position[self.ptr]=len(self.actor_slots[actor_id]);self.actor_slots[actor_id].append(self.ptr);self.actor_ids[self.ptr]=actor_id;self.actor_inserted[actor_id]+=1
    if self.joint:
     self.duration[self.ptr]=duration;self.discounts[self.ptr]=discount;self.actor_seconds[actor_id]+=duration
    if self.period_bank:self.period_bank.store(actor_id,row,self.inserted)
    self.birth[self.ptr]=self.inserted;self.store(*values);self.inserted+=1
    # Optional matched warmup: both real sample coverage and measured exposure,
    # never wall time. Legacy configs retain their original sample threshold.
    minimum_physical=self.cfg.get('warmup_min_physical_seconds',0.)
    if self.joint and self.warmup_physical_seconds is None and self.inserted>=self.cfg.get('warmup',10240) and float(self.actor_seconds.sum())>=minimum_physical and (not minimum_physical or int(self.actor_inserted.min())>=self.cfg.get('minimum_enqueued_per_actor',512)):
     self.warmup_physical_seconds=float(self.actor_seconds.sum())
 def sample(self):
  with self.lock:
   n=self.length_buf;k=min(n,self.batch_size)
   if self.cfg['sampler']=='balanced_actor':
    actors=self.cfg['num_actors'];base,remainder=divmod(k,actors);extra={(self.sample_calls*remainder+j)%actors for j in range(remainder)};indices=[]
    for actor_id,pool in enumerate(self.actor_slots):
     quota=base+int(actor_id in extra);assert len(pool)>=quota,'insufficient retained real samples for actor%d'%actor_id;indices+=random.sample(pool,quota)
    idx=np.asarray(indices);self.sample_calls+=1
   elif self.cfg['sampler']=='iid':idx=np.array(random.sample(range(n),k))
   else:idx=(np.arange(k)+int(n*random.random()))%n
   self.sampled_by_actor+=np.bincount(self.actor_ids[idx],minlength=self.cfg['num_actors'])
   births=self.birth[idx].copy()
   arrays=self.peek_buffer()+([self.duration,self.discounts] if self.joint else [])
   batch=[x[idx].copy() for x in arrays]
   if self.period_bank:
    positions=np.arange(self.sample_calls%4,k,4)
    anchored,old_births=self.period_bank.draw_actors(self.actor_ids[idx[positions]])
    for target,value in zip(batch,anchored):target[positions]=value
    births[positions]=old_births
   self.ages=(self.inserted-births).tolist()
   return batch

def learner(cfg):
 run=R/'logs'/cfg['run_id'];run.mkdir(exist_ok=False);(run/'config.json').write_text(json.dumps(cfg,indent=2))
 a,sess,fifo,ph,enq,nph,deq,size=setup(cfg,'learner',0)
 sess.run(tf.global_variables_initializer());a.init_target();a.rp_buffer=Replay(cfg)
 save=tf.train.Saver(max_to_keep=None);cp=R/'checkpoints'/cfg['model_id'];cp.mkdir(exist_ok=False)
 if cfg.get('cold_init_checkpoint'):
  initial=cfg['cold_init_checkpoint'];assert initial.startswith(str(R/'checkpoints')) and pathlib.Path(initial).name=='step-0','Only an own random step-0 checkpoint can be used for a paired cold initialization'
  save.restore(sess,initial);assert int(sess.run(a.global_step))==0,'Cold initialization must not contain prior optimization';a.init_target()
 if cfg.get('warm_start_window_checkpoint'):
  import window_warm_start
  provenance=window_warm_start.initialize(a,cfg['warm_start_window_checkpoint'],cfg)
  (cp/'window-warm-start.json').write_text(json.dumps(provenance,indent=2))
 if staged_timing.enabled(cfg):
  if cfg.get('cold_init_checkpoint'):raise ValueError('Staged source provenance must be explicit; cold_init is scratch-only')
  import frozen_window,trainable_timing
  if trainable_timing.enabled(cfg):provenance=trainable_timing.initialize(a,cfg['staged_joint_checkpoint'],dict(cfg,fixed_mtp_ms=None) if cfg.get('fixed_period_train_window') else cfg);teacher_hash=None
  else:provenance=frozen_window.initialize(a,cfg['frozen_window_checkpoint'],cfg);teacher_hash=frozen_window.digest(a)
  (cp/'window-transfer.json').write_text(json.dumps(provenance,indent=2))
 counter={'critic_iterations':0,'critic_optimizer_ops':0,'actor_updates':0,'target_updates':0};stop=threading.Event();errors=[]
 def checkpoint(n):
  if staged_timing.enabled(cfg):
   if teacher_hash is not None:assert frozen_window.digest(a)==teacher_hash,'Frozen scalar teacher changed'
   (cp/('step-%d.timing-stage.json'%n)).write_text(json.dumps(dict(learning_enabled=bool(sess.run(a.timing_learning_enabled)),teacher_sha256=teacher_hash,retained_period_counts=a.rp_buffer.retained_period_counts.tolist()),indent=2))
  save.save(sess,str(cp/('step-%d'%n)),write_meta_graph=n==0)
  (cp/('step-%d.counters.json'%n)).write_text(json.dumps(dict(counter,inserted=a.rp_buffer.inserted,cfg=cfg,physical_seconds=float(a.rp_buffer.actor_seconds.sum()),warmup_physical_seconds=a.rp_buffer.warmup_physical_seconds,update_credit_basis=cfg.get('update_credit_basis','decision')),indent=2))
  (cp/('step-%d.variables.json'%n)).write_text(json.dumps(dict(global_step=int(sess.run(a.global_step)),variables=[v.name for v in tf.global_variables()],target_assign_ops_per_set=len(a.target_actor_update_op)+len(a.target_critic_update_op)+len(a.target_critic_update_op2),meaning='global_step counts optimizer operations, not environment samples; a target update is one set invocation including stock overlapping critic scopes'),indent=2))
  policy=[v for v in tf.global_variables() if v.name.startswith('abstract_actor/') and '/Adam' not in v.name]
  if staged_timing.enabled(cfg):policy=frozen_window.policy_variables(tf.global_variables())
  h=hashlib.sha256()
  for v,value in zip(policy,sess.run(policy)):h.update(v.name.encode());h.update(value.tobytes())
  (cp/('step-%d.policy.sha256'%n)).write_text(h.hexdigest())
 checkpoint(0);(run/'ready').write_text(str(os.getpid()))
 def receive():
  try:
   while not stop.is_set():
    n=int(sess.run(size))
    if n:a.rp_buffer.append_many(sess.run(deq,{nph:min(n,100)}))
    else:stop.wait(.01)
  except Exception:errors.append(traceback.format_exc());stop.set()
 t=threading.Thread(target=receive,daemon=True);t.start();start=time.monotonic();last=0
 f=(run/'training.csv').open('w',buffering=65536);w=csv.writer(f)
 w.writerow(['wall_s','inserted','replay_length','critic_iterations','critic_optimizer_ops','actor_updates','target_updates','whole_U','post_warmup_U','fifo_backlog','sample_age_mean','critic_loss','critic2_loss','actor_loss','q_mean','actor_grad','critic_grad','minimum_condition_inserted','active_conditions','minimum_condition_sampled','maximum_condition_sampled','physical_seconds','updates_per_base20ms','sample_duration_ms_mean','update_credit_limit','update_credit_basis','warmup_physical_seconds','timing_learning_enabled','period_bank_valid','period_bank_min_retained','period_bank_sampled_total','fifo_capacity'])
 with tf.device('/job:learner/task:0/cpu:0'):
  grads=[tf.global_norm(tf.gradients(a.a_loss,a.symbolic_actor.train_var())),tf.global_norm(tf.gradients(a.c_loss,a.critic.train_var()))];qmean=tf.reduce_mean(a.critic_out)
 if cfg.get('target_no_fetch'):
  target_no_fetch=tf.group(*(a.target_actor_update_op+a.target_critic_update_op+a.target_critic_update_op2),name='target_no_fetch_capacity')
 bn=[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'actor' in v.name and 'target' not in v.name and (not cfg.get('bn_relevant_only') or 'abstract_actor' in v.name)];diag=[float('nan')]*6
 stage_rows=[];timing_stage=False;timing_diagnostics=[];timing_flat_since=None
 status='RUNNING';reason='';next_checkpoint=cfg.get('checkpoint_interval',50000);producer_check=0;producers_done=False;produced=0
 try:
  while not stop.is_set():
   with a.rp_buffer.lock:
    n=a.rp_buffer.inserted;physical=float(a.rp_buffer.actor_seconds.sum());warmup_physical=a.rp_buffer.warmup_physical_seconds
   it=counter['critic_iterations'];limit=update_credit.limit(n,cfg['warmup'],cfg['update_ratio'],physical,warmup_physical,cfg.get('update_credit_basis','decision'),cfg.get('updates_per_base20ms',.08))
   if cfg.get('joint_control') and cfg.get('capacity_only'):limit=0
   if time.monotonic()-last>5:
    w.writerow([time.monotonic()-start,n,a.rp_buffer.length_buf,it,it*2,counter['actor_updates'],counter['target_updates'],it/max(1,n),it/max(1,n-cfg['warmup']),int(sess.run(size)),float(np.mean(a.rp_buffer.ages)) if a.rp_buffer.ages else 0]+diag+[int(a.rp_buffer.actor_inserted.min()),int(np.count_nonzero(a.rp_buffer.actor_inserted)),int(a.rp_buffer.sampled_by_actor.min()),int(a.rp_buffer.sampled_by_actor.max()),physical,it/max(1,physical/.02),float(np.mean(batch[5])*1000) if cfg.get('joint_control') and it else 0,limit,cfg.get('update_credit_basis','decision'),warmup_physical,bool(sess.run(a.timing_learning_enabled)) if staged_timing.enabled(cfg) else 'legacy']+([int(np.minimum(a.rp_buffer.period_bank.seen,a.rp_buffer.period_bank.per_bin).sum()),int(np.minimum(a.rp_buffer.period_bank.seen,a.rp_buffer.period_bank.per_bin).min()),int(a.rp_buffer.period_bank.sampled.sum()),a.rp_buffer.size] if a.rp_buffer.period_bank else [0,0,0,a.rp_buffer.size]));f.flush();last=time.monotonic()
   if time.monotonic()-producer_check>=1:
    producer_check=time.monotonic()
    producer_paths=[R/'logs'/(cfg['run_id']+'-actor%d'%i)/'producer.json' for i in range(cfg['num_actors'])]
    producers_done=all(p.exists() for p in producer_paths)
    if producers_done:produced=sum(json.loads(p.read_text())['enqueued'] for p in producer_paths)
   received_all=producers_done and n==produced
   if received_all and (cfg.get('capacity_only') or it>=limit):break
   if n>=next_checkpoint:
    with a.rp_buffer.lock:checkpoint(a.rp_buffer.inserted)
    next_checkpoint=((a.rp_buffer.inserted//cfg.get('checkpoint_interval',50000))+1)*cfg.get('checkpoint_interval',50000)
   if it>=limit:time.sleep(.01);continue
   if cfg.get('require_coverage_gate') and not (R/'logs'/run.name/'coverage-ready.json').exists():time.sleep(.02);continue
   if cfg.get('require_coverage_gate') and int(a.rp_buffer.actor_inserted.min())<cfg.get('minimum_enqueued_per_actor',512):time.sleep(.02);continue
   if cfg.get('require_coverage_gate') and not (run/'first-update-coverage.json').exists():
    (run/'first-update-coverage.json').write_text(json.dumps(dict(epoch=time.time(),inserted=n,per_actor_inserted=a.rp_buffer.actor_inserted.tolist(),minimum_required=cfg.get('minimum_enqueued_per_actor',512),definition='Exact actor-ID metadata carried with each successfully inserted FIFO transition. No actor action is blocked.')))
   batch=a.rp_buffer.sample();fd={a.s0:batch[0],a.action:batch[1],a.reward:batch[2],a.s1:batch[3],a.terminal:batch[4],a.is_training:True,a.is_concrete:True,a.s0_delta:np.zeros_like(batch[0])}
   if cfg.get('joint_control'):fd[a.transition_discount]=batch[6]
   train_actor=not cfg.get('warm_start_window_checkpoint') or it>=cfg.get('window_critic_calibration_updates',5000)
   if staged_timing.enabled(cfg):
    with a.rp_buffer.lock:period_coverage=a.rp_buffer.retained_period_counts.copy()
    stage=timing_stage
    if cfg['timing_policy_mode']=='learned' and not stage and staged_timing.calibration_ready(it,period_coverage,cfg):
     sess.run(a.enable_timing_learning);stage=True;timing_stage=True
     (run/'timing-learning-start.json').write_text(json.dumps(dict(epoch=time.time(),critic_iterations=it,inserted=n,retained_period_counts=period_coverage.tolist()),indent=2))
    if cfg.get('fixed_period_train_window') and not stage and it>=cfg.get('timing_critic_calibration_updates',5000):
     assert cfg['timing_policy_mode']=='fixed' and cfg.get('fixed_mtp_ms')==80
     sess.run(a.enable_timing_learning);stage=True;timing_stage=True
     (run/'window-learning-start.json').write_text(json.dumps(dict(epoch=time.time(),critic_iterations=it,inserted=n,meaning='Calibration opens cwnd only; MTP stays hard-masked80ms and has zero gradient'),indent=2))
    train_actor=(cfg['timing_policy_mode']=='learned' and stage) or (cfg.get('fixed_period_train_window') and stage)
   t0=time.perf_counter();sess.run(a.critic_train_op,fd);t1=time.perf_counter()
   if train_actor:sess.run([a.actor_train_op,bn],fd)
   elif not cfg.get('warm_start_window_checkpoint') and cfg.get('window_policy_mode')!='staged_joint':sess.run(bn,fd)
   t2=time.perf_counter()
   if cfg.get('target_no_fetch'):sess.run(target_no_fetch)
   else:a.target_update()
   t3=time.perf_counter()
   if cfg.get('profile_stages'):stage_rows.append((time.time(),n,it,(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000))
   for k,v in [('critic_iterations',1),('critic_optimizer_ops',2),('actor_updates',int(train_actor)),('target_updates',1)]:counter[k]+=v
   if counter['critic_iterations']%100==0:
    diag=list(map(float,sess.run([a.c_loss,a.c_loss2,a.a_loss,qmean]+grads,fd)))
    if not np.all(np.isfinite(diag)):raise RuntimeError('nonfinite losses/gradients')
    if staged_timing.enabled(cfg):
     logits,time_actions=sess.run([a._moving_logits[:,1],a._unmasked_time[:,0]],{a.s0:batch[0],a.is_training:False})
     timing_diagnostics.append(dict(critic_iterations=counter['critic_iterations'],actor_updates=counter['actor_updates'],learning_enabled=timing_stage,logit_abs_max=float(np.max(np.abs(logits))),logit_median=float(np.median(logits)),tanh_slope_median=float(np.median(1-time_actions**2)),exact_saturation_count=int(np.count_nonzero(np.abs(time_actions)==1)),time_action_std=float(np.std(time_actions)),time_min_boundary_fraction=float(np.mean(time_actions<=-.99)),time_max_boundary_fraction=float(np.mean(time_actions>=.99)),batch_size=len(time_actions),q_mean=diag[3],actor_grad=diag[4]))
     if cfg.get('timing_action_health_guard'):
      latest=timing_diagnostics[-1];latest['epoch']=time.time()
      from node_control import dump
      dump(run/'timing-action-health.json',latest)
      # Persistent lack of state response at one boundary is a failure even if
      # throughput is high. Training noise cannot be counted as policy diversity.
      flat=timing_stage and counter['actor_updates']>=cfg.get('timing_health_after_actor_updates',10000) and max(latest['time_min_boundary_fraction'],latest['time_max_boundary_fraction'])>=.98 and latest['time_action_std']<.02
      timing_flat_since=(timing_flat_since or time.monotonic()) if flat else None
      if timing_flat_since and time.monotonic()-timing_flat_since>=cfg.get('timing_health_flat_seconds',90):
       raise RuntimeError('TIMING_POLICY_PERSISTENT_BOUNDARY_NO_STATE_RESPONSE')
   if cfg.get('optimizer_sleep_ms',5)>0:time.sleep(cfg.get('optimizer_sleep_ms',5)/1000.)
  if errors:raise RuntimeError(errors[0])
  stop.set();t.join(10);checkpoint(a.rp_buffer.inserted)
  rp=a.rp_buffer
  if rp.period_bank:
   with rp.lock:
    rp.period_bank.save(cp/'period-bank.npz')
    (cp/'period-bank-coverage.json').write_text(json.dumps(dict(rp.period_bank.snapshot(),fifo_capacity=rp.size,configured_total_capacity=cfg['memsize'],sampling_fraction=.25),indent=2))
  with rp.lock:np.savez(str(cp/'replay.npz'),s0=rp.s0_buf[:rp.length_buf],a=rp.a_buf[:rp.length_buf],reward=rp.reward_buf[:rp.length_buf],s1=rp.s1_buf[:rp.length_buf],terminal=rp.terminal_buf[:rp.length_buf],birth=rp.birth[:rp.length_buf],actor_ids=rp.actor_ids[:rp.length_buf],actor_inserted=rp.actor_inserted,sampled_by_actor=rp.sampled_by_actor,sample_calls=rp.sample_calls,ptr=rp.ptr,full=rp.full,duration=rp.duration[:rp.length_buf],discount=rp.discounts[:rp.length_buf],actor_seconds=rp.actor_seconds,warmup_physical_seconds=np.float64(rp.warmup_physical_seconds if rp.warmup_physical_seconds is not None else -1.))
  (cp/'sample-coverage.json').write_text(json.dumps(dict(inserted_by_actor=rp.actor_inserted.tolist(),sampled_by_actor=rp.sampled_by_actor.tolist(),retained_by_actor=[len(x) for x in rp.actor_slots],sample_calls=rp.sample_calls,physical_seconds_by_actor=rp.actor_seconds.tolist(),schema='joint-cwnd-mtp-smdp-v1' if cfg.get('joint_control') else 'fixed-period-v1',definition='sampled_by_actor counts replay rows used in complete updates; this is not additional real environment experience'),indent=2))
  import pickle
  with (cp/'python-state.pkl').open('wb') as out:pickle.dump(dict(counters=counter,inserted=rp.inserted,numpy_random=np.random.get_state(),python_random=random.getstate(),period_bank_storage_rng=rp.period_bank.rng.get_state() if rp.period_bank else None,period_bank_sampling_rng=rp.period_bank.draw_rng.get_state() if rp.period_bank else None),out,protocol=4)
  status='COMPLETED'
 except Exception as e:status='FAILED';reason=str(e);(run/'failure.txt').write_text(traceback.format_exc())
 finally:
  stop.set();f.close()
  if staged_timing.enabled(cfg):(run/'timing-diagnostics.json').write_text(json.dumps(timing_diagnostics,indent=2))
  if cfg.get('profile_stages'):
   with (run/'optimizer_stages.csv').open('w') as stage_file:
    stage_writer=csv.writer(stage_file);stage_writer.writerow(['epoch','inserted','iteration','critic_ms','actor_bn_ms','target_ms']);stage_writer.writerows(stage_rows)
  if status=='COMPLETED':
   # Keep every worker alive until producer drains and final checkpoint is complete.
   # TF1 Session.close can hang when another cluster worker has already exited.
   deadline=time.monotonic()+60
   while not all((R/'logs'/(cfg['run_id']+'-actor%d'%i)/'producer.json').exists() for i in range(cfg['num_actors'])):
    if time.monotonic()>deadline:status='FAILED';reason='producer drain barrier timeout';break
    time.sleep(.1)
  (run/'result.json').write_text(json.dumps(dict(status=status,reason=reason,inserted=a.rp_buffer.inserted,counters=counter,wall_s=time.monotonic()-start),indent=2))
  (run/'finished').write_text(status)
  sys.stdout.flush();sys.stderr.flush()
  # All Python/file/flow work has finished. Process exit releases the old gRPC runtime.
  os._exit(0 if status=='COMPLETED' else 1)

def actor(cfg,index):
 import evaluate
 run_id=cfg['run_id']+'-actor%d'%index
 a,sess,fifo,ph,enq,nph,deq,size=setup(cfg,'actor',index)
 deadline=time.monotonic()+100
 while not (R/'logs'/cfg['run_id']/'ready').exists():
  if time.monotonic()>deadline:raise RuntimeError('learner never initialized shared parameters')
  time.sleep(.2)
 policy=[v for v in tf.global_variables() if v.name.startswith('abstract_actor/') and '/Adam' not in v.name]
 h=hashlib.sha256()
 for v,value in zip(policy,sess.run(policy)):h.update(v.name.encode());h.update(value.tobytes())
 initial_policy_hash=h.hexdigest()
 if staged_timing.enabled(cfg):
  import frozen_window
  teacher_hash=frozen_window.digest(a)
  schedule=staged_timing.Scheduler(cfg,index)
 local=queue.Queue(1024);done=threading.Event();errors=[];counts={'generated':0,'enqueued':0,'exploration_calls':0};seen=[]
 def writer():
  try:
   last_progress=0
   while not done.is_set() or not local.empty():
    try:data=local.get(timeout=.05)
    except queue.Empty:continue
    sess.run(enq,dict(zip(ph,data+[np.int32(index)])));counts['enqueued']+=1
    if time.monotonic()-last_progress>=5:
     from node_control import dump
     dump(R/'logs'/run_id/'ingress-progress.json',dict(counts,epoch=time.time()));last_progress=time.monotonic()
  except Exception:errors.append(traceback.format_exc());done.set()
 t=threading.Thread(target=writer,daemon=True);t.start()
 from envwrapper_v2 import TCP_Env_Wrapper
 original=evaluate.model
 def model_proxy(ecfg):
  def action(hist):
   ready=R/'logs'/run_id/'flow-ready'
   if not ready.exists():ready.write_text(str(time.time()))
   value=a.get_concrete_action(hist,False)[0]
   if cfg.get('joint_control'):
    x=np.asarray(value,np.float32).reshape(2);no_noise=x.copy()
    if staged_timing.enabled(cfg):
     if hist[-2]==0:schedule.reset_episode()
     time_action,stage=schedule.choose(hist,x[1],a.timing_stage_enabled,True)
     x[1]=time_action;no_noise[1]=time_action
     if stage=='learned' and not cfg.get('capacity_only'):
      if cfg.get('window_policy_mode')=='staged_joint':x=np.clip(x+np.random.normal(0,cfg.get('exploration_std',.2),size=2),-1,1)
      else:x[1]=np.clip(x[1]+np.random.normal(0,cfg.get('exploration_std',.2)),-1,1)
      counts['exploration_calls']+=1
     if cfg.get('fixed_period_train_window') and stage=='fixed' and a.timing_stage_enabled and not cfg.get('capacity_only'):
      x[0]=np.clip(x[0]+np.random.normal(0,cfg.get('exploration_std',.2)),-1,1);counts['exploration_calls']+=1
     action.no_noise=no_noise;return x
    if cfg.get('fixed_action') is not None:x[0]=float(cfg['fixed_action'])
    elif not cfg.get('capacity_only'):
     x=np.clip(x+np.random.normal(0,cfg.get('exploration_std',.2),size=2),-1,1);counts['exploration_calls']+=1
    action.no_noise=no_noise;return x
   x=float(np.asarray(value).reshape(-1)[0]);no_noise=x
   if cfg.get('fixed_action') is not None:
    # Capacity diagnostic only: still execute the real remote inference call.
    x=float(cfg['fixed_action']);no_noise=x
   else:
    noise=float(np.asarray(a.actor_noise(np.array([[x]]))).reshape(-1)[0]);counts['exploration_calls']+=1;x=float(np.clip(x+noise,-1,1))
   evaluate.latest_nn=no_noise;return x
  def digest():return 'LIVE_TRAINING_POLICY'
  params=json.loads((R/'code/canopy/params_base_eval.json').read_text())
  return a,sess,TCP_Env_Wrapper,params,action,digest,[]
 evaluate.model=model_proxy
 # The copied evaluator's hook receives the completed transition after next action is sent.
 def transition(h,a0,reward,h1,duration=None,discount=None):
  if errors:raise RuntimeError(errors[0])
  values=[h.copy(),np.asarray(a0,np.float32).reshape(-1),np.array([reward],np.float32),h1.copy(),np.array([0],np.float32)]
  if cfg.get('joint_control'):values += [np.array([duration],np.float32),np.array([discount],np.float32)]
  local.put_nowait(values);counts['generated']+=1
 evaluate.transition_hook=transition
 def finish():
  done.set();t.join(timeout=30)
  if t.is_alive():errors.append('writer failed to drain in 30s')
 evaluate.producer_finish=finish
 evaluate.distributed_session=True
 ecfg=dict(cfg['conditions'][index],method='canopy',run_id=run_id,seconds=0,training_steps=cfg['steps_per_actor'],port=cfg.get('actor_port_base',48200)+index,seed=cfg['seed'],actor_rng_seed=cfg['seed']+index*1009,actor_index=index,code_commit=cfg.get('code_commit'),reward_mode='raw',control_period_ms=20,history_frames=10,use_normalizer=False,stop_file=cfg.get('stop_file'))
 if cfg.get('server_binary'):ecfg['server_binary']=cfg['server_binary']
 if cfg.get('joint_control'):
  if cfg.get('clock_injected_delay_us'):raise ValueError('No clock fault injection during distributed collection')
  ecfg['training_mode']=not cfg.get('capacity_only',False)
  ecfg['native_clock_mode']='sample'
  for key in ['joint_control','backend','mtp_min_ms','mtp_max_ms','gamma','decision_cost','fixed_mtp_ms','absolute_deadline_epoch','capacity_only','fixed_action','physical_seconds_per_actor','startup_probe_seconds']:
   if key in cfg:ecfg[key]=cfg[key]
  for key in ['window_policy_mode','frozen_teacher_schema','timing_policy_mode','timing_init_ms','timing_calibration_periods_ms','timing_critic_calibration_updates','timing_min_retained_per_period','connection_reset_seconds','target_bn_consistent','window_teacher_planned_context_ms']:
   if key in cfg:ecfg[key]=cfg[key]
 path=R/'configs'/(run_id+'.json');path.write_text(json.dumps(ecfg,indent=2));sys.argv=['evaluate.py','--config',str(path)]
 if cfg.get('serialized_start'):
  release=R/'logs'/(cfg['run_id']+'-launch-actor%d'%index)
  while not release.exists():time.sleep(.05)
 try:evaluate.main()
 finally:
  done.set();t.join(timeout=30)
  if t.is_alive():errors.append('writer failed to drain in 30s')
  # evaluator closes the session only after the producer thread has been drained (hook).
  (R/'logs'/run_id/'producer.json').write_text(json.dumps(dict(counts,errors=errors,seed=cfg['seed']+index*1009,initial_policy_hash=initial_policy_hash),indent=2))
  if staged_timing.enabled(cfg):
   if cfg.get('window_policy_mode')=='frozen_scalar':
    assert frozen_window.digest(a)==teacher_hash,'Frozen teacher changed in actor'
    (R/'logs'/run_id/'teacher-policy.sha256').write_text(teacher_hash)
  import pickle
  with (R/'logs'/run_id/'actor-state.pkl').open('wb') as out:
   pickle.dump(dict(counts=counts,numpy_random=np.random.get_state(),python_random=random.getstate(),noise_state=dict(a.actor_noise.__dict__) if a.actor_noise else None),out,protocol=4)
 if errors:
  sys.stdout.flush();sys.stderr.flush();os._exit(1)
 deadline=time.monotonic()+cfg.get('shutdown_barrier_s',180)
 while not (R/'logs'/cfg['run_id']/'finished').exists():
  if time.monotonic()>deadline:
   (R/'logs'/run_id/'shutdown-failure.txt').write_text('Learner checkpoint completion barrier timeout');sys.stdout.flush();sys.stderr.flush();os._exit(1)
  time.sleep(.1)
 sys.stdout.flush();sys.stderr.flush();os._exit(0)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--role',choices=['learner','actor'],required=True);p.add_argument('--index',type=int,default=0);args=p.parse_args();cfg=json.loads(pathlib.Path(args.config).read_text())
 if args.role=='learner':learner(cfg)
 else:actor(cfg,args.index)
