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

def setup(cfg,role,index):
 assert cfg['warmup']>=(512 if cfg.get('portable_smoke_training') else max(10000,20*512))
 assert cfg.get('reward_mode','raw') in ['raw','raw-sym']
 assert cfg.get('policy_family') in ['orca','canopy']
 assert cfg.get('policy_family')!='orca' or cfg.get('reward_mode')=='raw'
 assert cfg.get('control_period_ms',20)==20 and not cfg.get('joint_control')
 import independent_critic_v34
 independent_critic_v34.install(agent_v2,cfg)
 if cfg.get('backend')=='moving_stats':
  import moving_stats_backend
  moving_stats_backend.install(agent_v2)
 if cfg.get('target_bn_consistent'):
  import consistent_target_bn
  consistent_target_bn.install(agent_v2)
 random.seed(cfg['seed']+index*1009);np.random.seed(cfg['seed']+index*1009);tf.set_random_seed(cfg['seed'])
 cluster=tf.train.ClusterSpec({'learner':[cfg['learner_endpoint']]} if cfg.get('direct_actor_session',False) else {'learner':[cfg['learner_endpoint']],'actor':['127.0.0.1:%d'%(47101+i) for i in range(cfg['num_actors'])]})
 direct=role=='actor' and cfg.get('direct_actor_session',False)
 server=None if direct else tf.train.Server(cluster,job_name=role,task_index=index,config=tf.ConfigProto(intra_op_parallelism_threads=cfg.get("tf_threads",2),inter_op_parallelism_threads=cfg.get("tf_threads",2)))
 session_target='grpc://'+cfg['learner_endpoint'] if direct else server.target
 with tf.device('/job:learner/task:0/cpu:0'):
  if cfg.get('policy_family')=='orca':
   from orca_backend import create
   a=create(R,cfg,role)
  else:
   a=Agent(70,1,256,256,gamma=.995,batch_size=512,lr_a=.0001,lr_c=.001,tau=.001,mem_size=cfg['memsize'] if role=='learner' else 1,noise_type=3,noise_exp=50000,stddev=cfg.get('exploration_std',.2),PER=False,CDQ=True,LOSS_TYPE=cfg.get('LOSS_TYPE','MSE'),use_original=False,use_snt_model_wo_ibp=False)
   if cfg.get('target_bn_consistent'):consistent_target_bn.connect_statistics(a,.001)
   from canopy_feedback import install_bounds
   install_bounds(a)
  fifo=tf.FIFOQueue(10000,[tf.float32]*5+[tf.int32],shapes=[[70],[1],[1],[70],[1],[]],shared_name='rp_buf')
  ph=[tf.placeholder(tf.float32,x) for x in [[70],[1],[1],[70],[1]]]+[tf.placeholder(tf.int32,[])]
  enqueue=fifo.enqueue(ph);nph=tf.placeholder(tf.int32,[]);dequeue=fifo.dequeue_many(nph);size=fifo.size()
  if role=='learner':a.build_learn()
 if direct:assert not any('/job:actor/' in op.device for op in tf.get_default_graph().get_operations()),'Direct actor session requires every model/queue operation to remain learner-placed'
 sess=tf.Session(session_target,config=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=cfg.get("tf_threads",2),inter_op_parallelism_threads=cfg.get("tf_threads",2)));a.assign_sess(sess)
 return a,sess,fifo,ph,enqueue,nph,dequeue,size

class Replay(ReplayBuffer):
 def __init__(self,cfg):
  super().__init__(cfg['memsize'],70,1,512);self.lock=threading.RLock();self.inserted=0;self.birth=np.zeros(self.size,np.int64);self.cfg=cfg;self.ages=[];self.actor_ids=np.full(self.size,-1,np.int32);self.actor_inserted=np.zeros(cfg['num_actors'],np.int64);self.actor_slots=[[] for _ in range(cfg['num_actors'])];self.slot_position=np.full(self.size,-1,np.int32);self.sampled_by_actor=np.zeros(cfg['num_actors'],np.int64);self.sample_calls=0
 def append_many(self,data):
  with self.lock:
   for values,actor_id in zip(zip(*data[:5]),data[5]):
    assert all(np.all(np.isfinite(v)) for v in values),'nonfinite transition'
    assert -1.00001<=float(values[1][0])<=1.00001
    actor_id=int(actor_id);assert 0<=actor_id<self.cfg['num_actors'];old=int(self.actor_ids[self.ptr])
    if old>=0:
     pool=self.actor_slots[old];position=int(self.slot_position[self.ptr]);moved=pool[-1];pool[position]=moved;self.slot_position[moved]=position;pool.pop()
    self.slot_position[self.ptr]=len(self.actor_slots[actor_id]);self.actor_slots[actor_id].append(self.ptr);self.actor_ids[self.ptr]=actor_id;self.actor_inserted[actor_id]+=1
    self.birth[self.ptr]=self.inserted;self.store(*values);self.inserted+=1
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
   self.ages=(self.inserted-self.birth[idx]).tolist()
   return [x[idx].copy() for x in self.peek_buffer()]

def learner(cfg):
 run=R/'logs'/cfg['run_id'];run.mkdir(exist_ok=False);(run/'config.json').write_text(json.dumps(cfg,indent=2))
 a,sess,fifo,ph,enq,nph,deq,size=setup(cfg,'learner',0)
 sess.run(tf.global_variables_initializer())
 from scalar_window_init import initialize
 initial_window=initialize(a,sess,cfg)
 a.init_target();a.rp_buffer=Replay(cfg)
 if initial_window:(run/'window-initialization.json').write_text(json.dumps(initial_window,indent=2))
 save=tf.train.Saver(max_to_keep=None);cp=R/'checkpoints'/cfg['model_id'];cp.mkdir(exist_ok=False)
 if cfg.get('cold_init_checkpoint'):
  initial=cfg['cold_init_checkpoint'];assert initial.startswith(str(R/'checkpoints')) and pathlib.Path(initial).name=='step-0','Only an own random step-0 checkpoint can be used for a paired cold initialization'
  save.restore(sess,initial);assert int(sess.run(a.global_step))==0,'Cold initialization must not contain prior optimization';a.init_target()
 counter={'critic_iterations':0,'critic_optimizer_ops':0,'actor_updates':0,'target_updates':0};stop=threading.Event();errors=[]
 def checkpoint(n):
  save.save(sess,str(cp/('step-%d'%n)),write_meta_graph=n==0)
  (cp/('step-%d.counters.json'%n)).write_text(json.dumps(dict(counter,inserted=a.rp_buffer.inserted,cfg=cfg),indent=2))
  (cp/('step-%d.variables.json'%n)).write_text(json.dumps(dict(global_step=int(sess.run(a.global_step)),variables=[v.name for v in tf.global_variables()],target_assign_ops_per_set=len(a.target_actor_update_op)+len(a.target_critic_update_op)+len(a.target_critic_update_op2),meaning='global_step counts optimizer operations, not environment samples; a target update is one set invocation including stock overlapping critic scopes'),indent=2))
  policy=[v for v in tf.global_variables() if v.name.startswith('actor/' if cfg.get('policy_family')=='orca' else 'abstract_actor/') and '/Adam' not in v.name]
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
 w.writerow(['wall_s','inserted','replay_length','critic_iterations','critic_optimizer_ops','actor_updates','target_updates','whole_U','post_warmup_U','fifo_backlog','sample_age_mean','critic_loss','critic2_loss','actor_loss','q_mean','actor_grad','critic_grad','minimum_condition_inserted','active_conditions','minimum_condition_sampled','maximum_condition_sampled'])
 with tf.device('/job:learner/task:0/cpu:0'):
  grads=[tf.global_norm(tf.gradients(a.a_loss,a.symbolic_actor.train_var())),tf.global_norm(tf.gradients(a.c_loss,a.critic.train_var()))];qmean=tf.reduce_mean(a.critic_out)
 if cfg.get('target_no_fetch'):
  target_no_fetch=tf.group(*(a.target_actor_update_op+a.target_critic_update_op+a.target_critic_update_op2),name='target_no_fetch_capacity')
 bn=[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'actor' in v.name and 'target' not in v.name and (cfg.get('policy_family')=='orca' or not cfg.get('bn_relevant_only') or 'abstract_actor' in v.name)];diag=[float('nan')]*6
 stage_rows=[]
 status='RUNNING';reason='';next_checkpoint=cfg.get('checkpoint_interval',50000);producer_check=0;producers_done=False;produced=0
 try:
  while not stop.is_set():
   n=a.rp_buffer.inserted;it=counter['critic_iterations'];limit=int(max(0,n-cfg['warmup'])*cfg['update_ratio'])
   if cfg.get('diagnostic_no_optimizer',False):limit=0
   if time.monotonic()-last>5:
    w.writerow([time.monotonic()-start,n,a.rp_buffer.length_buf,it,it*2,it,it,it/max(1,n),it/max(1,n-cfg['warmup']),int(sess.run(size)),float(np.mean(a.rp_buffer.ages)) if a.rp_buffer.ages else 0]+diag+[int(a.rp_buffer.actor_inserted.min()),int(np.count_nonzero(a.rp_buffer.actor_inserted)),int(a.rp_buffer.sampled_by_actor.min()),int(a.rp_buffer.sampled_by_actor.max())]);f.flush();last=time.monotonic()
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
   t0=time.perf_counter();sess.run(a.critic_train_op,fd);t1=time.perf_counter()
   sess.run([a.actor_train_op,bn],fd);t2=time.perf_counter()
   if cfg.get('target_no_fetch'):sess.run(target_no_fetch)
   else:a.target_update()
   t3=time.perf_counter()
   if cfg.get('profile_stages'):stage_rows.append((time.time(),n,it,(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000))
   for k,v in [('critic_iterations',1),('critic_optimizer_ops',2),('actor_updates',1),('target_updates',1)]:counter[k]+=v
   if counter['critic_iterations']%100==0:
    diag=list(map(float,sess.run([a.c_loss,a.c_loss2,a.a_loss,qmean]+grads,fd)))
    if not np.all(np.isfinite(diag)):raise RuntimeError('nonfinite losses/gradients')
   if cfg.get('optimizer_sleep_ms',5)>0:time.sleep(cfg.get('optimizer_sleep_ms',5)/1000.)
  if errors:raise RuntimeError(errors[0])
  stop.set();t.join(10);checkpoint(a.rp_buffer.inserted)
  rp=a.rp_buffer
  with rp.lock:np.savez(str(cp/'replay.npz'),s0=rp.s0_buf[:rp.length_buf],a=rp.a_buf[:rp.length_buf],reward=rp.reward_buf[:rp.length_buf],s1=rp.s1_buf[:rp.length_buf],terminal=rp.terminal_buf[:rp.length_buf],birth=rp.birth[:rp.length_buf],actor_ids=rp.actor_ids[:rp.length_buf],actor_inserted=rp.actor_inserted,sampled_by_actor=rp.sampled_by_actor,sample_calls=rp.sample_calls,ptr=rp.ptr,full=rp.full)
  (cp/'sample-coverage.json').write_text(json.dumps(dict(inserted_by_actor=rp.actor_inserted.tolist(),sampled_by_actor=rp.sampled_by_actor.tolist(),retained_by_actor=[len(x) for x in rp.actor_slots],sample_calls=rp.sample_calls,definition='sampled_by_actor counts replay rows used in complete updates; this is not additional real environment experience'),indent=2))
  import pickle
  with (cp/'python-state.pkl').open('wb') as out:pickle.dump(dict(counters=counter,inserted=rp.inserted,numpy_random=np.random.get_state(),python_random=random.getstate()),out,protocol=4)
  status='COMPLETED'
 except Exception as e:status='FAILED';reason=str(e);(run/'failure.txt').write_text(traceback.format_exc())
 finally:
  stop.set();f.close()
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
 policy=[v for v in tf.global_variables() if v.name.startswith('actor/' if cfg.get('policy_family')=='orca' else 'abstract_actor/') and '/Adam' not in v.name]
 h=hashlib.sha256()
 for v,value in zip(policy,sess.run(policy)):h.update(v.name.encode());h.update(value.tobytes())
 initial_policy_hash=h.hexdigest()
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
 if cfg.get('policy_family')=='orca':
  sys.path.insert(0,str(R/'code/orca/rl-module'))
  from envwrapper import TCP_Env_Wrapper
 else:from envwrapper_v2 import TCP_Env_Wrapper
 original=evaluate.model
 def model_proxy(ecfg):
  nonlocal feedback
  if cfg.get('reward_mode')=='raw-sym':
   from canopy_feedback import Feedback
   feedback=Feedback(a,cfg,R/'logs'/run_id)
   evaluate.reward_hook=feedback.reward
  def action(hist):
   ready=R/'logs'/run_id/'flow-ready'
   if not ready.exists():ready.write_text(str(time.time()))
   if feedback is not None:x=feedback.observe_action(hist)
   else:
    value=a.get_concrete_action(hist,False)[0];x=float(np.asarray(value).reshape(-1)[0])
   no_noise=x
   if cfg.get('fixed_action') is not None:
    # Capacity diagnostic only: still execute the real remote inference call.
    x=float(cfg['fixed_action']);no_noise=x
   else:
    noise=float(np.asarray(a.actor_noise(np.array([[x]]))).reshape(-1)[0]);counts['exploration_calls']+=1;x=float(np.clip(x+noise,-1,1))
   evaluate.latest_nn=no_noise
   return x
  def digest():return 'LIVE_TRAINING_POLICY'
  params=json.loads((R/'code'/('orca' if cfg.get('policy_family')=='orca' else 'canopy')/'params_base_eval.json').read_text())
  return a,sess,TCP_Env_Wrapper,params,action,digest,[]
 feedback=None
 evaluate.model=model_proxy
 # The copied evaluator's hook receives the completed transition after next action is sent.
 def transition(h,a0,reward,h1):
  if errors:raise RuntimeError(errors[0])
  local.put_nowait([h.copy(),np.array([a0],np.float32),np.array([reward],np.float32),h1.copy(),np.array([0],np.float32)]);counts['generated']+=1
 evaluate.transition_hook=transition
 def finish():
  done.set();t.join(timeout=30)
  if t.is_alive():errors.append('writer failed to drain in 30s')
 evaluate.producer_finish=finish
 evaluate.distributed_session=True
 ecfg=dict(cfg['conditions'][index],method='orca' if cfg.get('policy_family')=='orca' else 'canopy',run_id=run_id,seconds=0,training_steps=cfg['steps_per_actor'],port=cfg.get('actor_port_base',48200)+index,seed=cfg['seed'],actor_rng_seed=cfg['seed']+index*1009,actor_index=index,code_commit=cfg.get('code_commit'),reward_mode='raw',control_period_ms=20,history_frames=10,use_normalizer=False,stop_file=cfg.get('stop_file'))
 if cfg.get('server_binary'):ecfg['server_binary']=cfg['server_binary']
 if cfg.get('require_normal_cubic_start'):ecfg['require_normal_cubic_start']=True
 if cfg.get('verified_ipc_v2'):ecfg['verified_ipc_v2']=True
 path=R/'configs'/(run_id+'.json');path.write_text(json.dumps(ecfg,indent=2));sys.argv=['evaluate.py','--config',str(path)]
 if cfg.get('serialized_start'):
  release=R/'logs'/(cfg['run_id']+'-launch-actor%d'%index)
  while not release.exists():time.sleep(.05)
 try:evaluate.main()
 finally:
  if feedback is not None:feedback.close()
  done.set();t.join(timeout=30)
  if t.is_alive():errors.append('writer failed to drain in 30s')
  # evaluator closes the session only after the producer thread has been drained (hook).
  (R/'logs'/run_id/'producer.json').write_text(json.dumps(dict(counts,errors=errors,seed=cfg['seed']+index*1009,initial_policy_hash=initial_policy_hash),indent=2))
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
