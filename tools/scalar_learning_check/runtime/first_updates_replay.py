"""Audited TF1 factual-replay intervention; no TCP, checkpoint write, or online resume.

Each job is a FRESH process. Training uses the original critic/actor Adam and
target ops; lr ablations feed the existing actor optimizer learning-rate tensor.
Diagnostics use deterministic unsmoothed targets and NEVER fetch random smoothing
ops. Static replay/sampling/graph RNG differ from historical asynchronous training;
R0 is a mechanism test, not a bit-exact reconstruction of the online trajectory.
"""
import argparse,os,pathlib,sys,json,random,time,hashlib
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--arm',choices=['R0','R1','R2','R3','R4','R5','R6'],required=True)
ap.add_argument('--seed',type=int,default=3);ap.add_argument('--redraw-seed',type=int,default=101)
ap.add_argument('--record-every',type=int,default=1);ap.add_argument('--iterations',type=int)
ap.add_argument('--output',required=True);args=ap.parse_args()
SOURCE=pathlib.Path(os.environ['CANOPY_FIRST_SOURCE']);F=pathlib.Path(os.environ.get('CANOPY_FIRST_FRAMEWORK',str(SOURCE/'framework')))
OUTPUT=pathlib.Path(args.output);assert not OUTPUT.exists()
sys.path[:0]=[str(F),str(F/'code/runtime/_deps')]
import numpy as np,tensorflow as tf
cfg=json.loads((SOURCE/'registration.json').read_text())['training_config'];rid=cfg['run_id']
D=pathlib.Path(os.environ.get('CANOPY_FIRST_CHECKPOINT_DIR',str(F/'checkpoints'/rid)))
first=json.loads((SOURCE/'first-update-coverage.json').read_text())
FIRST_INSERTED=int(first['inserted']);tf.set_random_seed(cfg['seed'])
from orca_backend import create
a=create(F,cfg,'learner');a.build_learn();original_vars=tf.global_variables();saver=tf.train.Saver(original_vars)
bn_ops=[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'actor' in v.name and 'target' not in v.name];assert bn_ops
g1=tf.gradients(a.critic_out,a.action)[0];g2=tf.gradients(a.critic_out2,a.action)[0]
actor_grad=tf.global_norm(tf.gradients(a.a_loss,a.actor.train_var()))
raw1=a.target_critic.build(a.s1,a.target_actor_out);raw2=a.target_critic2.build(a.s1,a.target_actor_out)
raw_target=a.reward+cfg.get('gamma',.995)*(1-a.terminal)*tf.minimum(raw1,raw2)
det_mse1=tf.reduce_mean(tf.square(a.critic_out-raw_target));det_mse2=tf.reduce_mean(tf.square(a.critic_out2-raw_target))
sens=tf.gradients(raw1,a.s1)[0]
assert list(original_vars)==tf.global_variables(),'Diagnostic construction changed checkpoint schema'
actor_lr=a.actor_optimizer._lr_t
def ancestors(tensors):
 todo=[t.op if hasattr(t,'op') else t for t in tensors];seen=set()
 while todo:
  op=todo.pop()
  if op in seen:continue
  seen.add(op);todo.extend(t.op for t in op.inputs);todo.extend(op.control_inputs)
 return seen
probe_fetches=[a.actor_out,a.critic_out,g1,g2,actor_grad,sens,det_mse1,det_mse2]
random_ancestors=[op.name for op in ancestors(probe_fetches) if op.type in ['RandomStandardNormal','RandomUniform','RandomUniformInt','TruncatedNormal']]
assert not random_ancestors,random_ancestors
sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1));a.assign_sess(sess)
saver.restore(sess,str(D/'step-0'));reader=tf.train.NewCheckpointReader(str(D/'step-0'))
assert {v.op.name:list(v.shape) for v in original_vars}==reader.get_variable_to_shape_map()
assert all(np.array_equal(v,reader.get_tensor(t.op.name)) for t,v in zip(original_vars,sess.run(original_vars)))
if args.arm=='R1':
 rng=np.random.RandomState(args.redraw_seed);critics=[v for v in tf.trainable_variables() if v.op.name.startswith(('critic/','critic2/'))]
 assert len(critics)==12 and len({v.op.name for v in critics})==len(critics)
 for v in critics:
  shape=v.shape.as_list()
  if v.op.name.endswith('/kernel'):
   lim=np.sqrt(6./(shape[0]+shape[1]));value=rng.uniform(-lim,lim,shape).astype(np.float32)
  else:assert v.op.name.endswith('/bias');value=np.zeros(shape,np.float32)
  sess.run(tf.assign(v,value))
 sess.run([a.target_cri_init_op,a.target_cri_init_op2])
 for v in critics:
  target=next(t for t in original_vars if t.op.name=='target_'+v.op.name)
  assert np.array_equal(sess.run(v),sess.run(target))
 for v,value in zip(original_vars,sess.run(original_vars)):
  if not v.op.name.startswith(('critic/','critic2/','target_critic/','target_critic2/')):assert np.array_equal(value,reader.get_tensor(v.op.name)),v.name
replay=np.load(str(D/'replay.npz'));keep=np.flatnonzero(replay['birth']<FIRST_INSERTED);assert len(keep)==FIRST_INSERTED
assert np.array_equal(replay['birth'][keep],np.arange(FIRST_INSERTED))
data={k:replay[k][keep] for k in ['s0','a','reward','s1','terminal']};aid=replay['actor_ids'][keep]
pools=[list(np.flatnonzero(aid==i)) for i in range(24)];assert [len(p) for p in pools]==first['per_actor_inserted']
probe=np.concatenate([np.asarray(p)[np.linspace(0,len(p)-1,16,dtype=int)] for p in pools]);P0=data['s0'][probe];P1=data['s1'][probe];PID=aid[probe]
RTT=np.asarray([cfg['conditions'][i]['base_rtt_ms'] for i in PID]);rng=random.Random(args.seed);sample_calls=0
def batch():
 global sample_calls
 base,rem=divmod(512,24);extra={(sample_calls*rem+j)%24 for j in range(rem)};idx=[]
 for i,p in enumerate(pools):idx+=rng.sample(p,base+int(i in extra))
 sample_calls+=1;idx=np.asarray(idx)
 return {a.s0:data['s0'][idx],a.action:data['a'][idx],a.reward:data['reward'][idx],a.s1:data['s1'][idx],a.terminal:data['terminal'][idx],a.is_training:True,a.is_concrete:True,a.s0_delta:np.zeros_like(data['s0'][idx])}
lr={'R4':1e-5,'R5':1e-6,'R6':1e-6}.get(args.arm,1e-4);cal=3000 if args.arm in ['R3','R5'] else 0
count=args.iterations if args.iterations is not None else {'R0':300,'R1':150,'R2':300,'R3':3300,'R4':1000,'R5':4000,'R6':1000}[args.arm]
initial_pi=sess.run(a.actor_out,{a.s0:P0,a.is_training:False}).ravel();release_pi=None;records=[];started=time.time();release_check=None
actor_variables=a.actor.train_var();initial_actor=sess.run(actor_variables)
def record(it,fd):
 global release_pi
 pi=sess.run(a.actor_out,{a.s0:P0,a.is_training:False}).ravel()
 if it==cal:release_pi=pi.copy()
 q,d1,d2=sess.run([a.critic_out,g1,g2],{a.s0:P0,a.action:pi.reshape(-1,1),a.is_training:False})
 ss=sess.run(sens,{a.s1:P1,a.is_training:False})
 m1,m2,ag=sess.run([det_mse1,det_mse2,actor_grad],fd)
 reference=release_pi if release_pi is not None else initial_pi;delta=pi-reference
 return dict(iteration=it,pi_mean=float(pi.mean()),pi_sat=float(np.mean(abs(pi)>=.95)),pi_mean_abs_delta=float(np.mean(abs(delta))),
  pi_abs_mean_delta=float(abs(delta.mean())),pi_delta_p95=float(np.percentile(abs(delta),95)),pi_delta_max=float(max(abs(delta))),
  dq1=float(d1.mean()),dq1_pos=float(np.mean(d1>0)),dq2=float(d2.mean()),dq2_pos=float(np.mean(d2>0)),twin_agree=float(np.mean(np.sign(d1)==np.sign(d2))),q1=float(q.mean()),
  diagnostic_unsmoothed_TD_MSE1=float(m1),diagnostic_unsmoothed_TD_MSE2=float(m2),actor_grad=float(ag),
  target_total_dpacing=float(ss[:,64].mean()),target_total_dsamples=float(ss[:,66].mean()),
  pi_by_rtt={str(r):float(pi[RTT==r].mean()) for r in [20,50,100,200]},
  dq1_by_rtt={str(r):float(d1.ravel()[RTT==r].mean()) for r in [20,50,100,200]})
for it in range(count+1):
 fd=batch()
 if it==cal and cal:
  current=sess.run(actor_variables);assert all(np.array_equal(x,y) for x,y in zip(initial_actor,current))
  release_check=dict(actor_trainables_unchanged=True,actor_optimizer_steps=0,BN_stats_are_allowed_to_change=True)
 if it in [0,cal,count] or ((it<=100 or cal<=it<=cal+100) and it%args.record_every==0) or it%10==0:
  row=record(it,fd);assert all(np.isfinite(v) for v in row.values() if isinstance(v,float));records.append(row)
 if it==count:break
 sess.run(a.critic_train_op,fd)
 if it>=cal:
  fd[actor_lr]=lr;sess.run([a.actor_train_op]+([] if args.arm=='R2' else bn_ops),fd)
 elif args.arm!='R2':sess.run(bn_ops,fd)
 a.target_update()
final_values=sess.run(original_vars);h=hashlib.sha256()
for v,value in zip(original_vars,final_values):assert np.isfinite(value).all();h.update(v.op.name.encode());h.update(value.tobytes())
post=[r for r in records if r['iteration']>=cal]
first_cross=lambda key,t:next((r['iteration']-cal for r in post if r[key]>=t),None)
summary=dict(updates_to_abs_mean_pi_point9=first_cross('pi_abs_mean_delta',.9),updates_to_dpi_point1=first_cross('pi_abs_mean_delta',.1),
 updates_to_dpi_point5=first_cross('pi_abs_mean_delta',.5),updates_to_mean_absolute_delta_point1=first_cross('pi_mean_abs_delta',.1),
 final=records[-1],final_parameter_sha256=h.hexdigest(),release_check=release_check)
out=dict(status='OFFLINE_FACTUAL_REPLAY_COMPLETE',arm=args.arm,sampler_seed=args.seed,TF_seed=cfg['seed'],redraw_seed=args.redraw_seed if args.arm=='R1' else None,
 replay_rows=len(keep),probe_states=len(probe),actor_lr=lr,critic_only_updates=cal,iterations=count,wall_s=time.time()-started,
 model_variables=len(original_vars),diagnostic_random_ancestors=random_ancestors,source_init_verified_all_tensors=True,summary=summary,records=records,
 optimizer_updates_in_memory_only=True,new_TCP_runs=0,source_checkpoints_written=0,
 limitations='Static pre-update factual replay, fresh local graph/RNG and sampling; not historical bit-exact async replay or new TCP performance. TD residual is deterministic unsmoothed diagnostic, not a true-value label. BN evolves during critic-only. Repeated sampler seeds are not independent training seeds.')
OUTPUT.parent.mkdir(parents=True,exist_ok=True);OUTPUT.write_text(json.dumps(out,indent=1,allow_nan=False));sess.close()
print(json.dumps(dict(output=str(OUTPUT),arm=args.arm,seed=args.seed,wall_s=out['wall_s'],summary=summary)),flush=True)
