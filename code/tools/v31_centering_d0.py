"""Real-TF1 offline centering screen on preserved factual V31 TCP data.
Not V35 replication: fixed healthy policy/BN; explicit legacy or V34 Adam; no actor update.
"""
import pathlib,os,sys,json,time,hashlib,argparse
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root',type=pathlib.Path,required=True)
parser.add_argument('--factual-root',type=pathlib.Path,required=True)
parser.add_argument('--stability-only',action='store_true')
parser.add_argument('--save-critic-diagnostics',action='store_true',help='Save critic tensors only for factual holdout queries; not a policy checkpoint')
parser.add_argument('--optimizer',choices=['legacy','independent_v34'],default='legacy')
parser.add_argument('--target-order',choices=['stock','serialized_legacy'],default='stock')
parser.add_argument('--allow-failed-null-descriptive-screen',action='store_true',help='Record descriptive results only; never releases an actor')
args=parser.parse_args()
R=args.experiment_root.resolve();O=args.factual_root.resolve();F=R/'frameworks/mtpjoint';sys.path[:0]=[str(R),str(F)]
assert (R/'audit').is_dir() and (R/'configs').is_dir(), 'Use an independently prepared experiment directory'
assert not (R/'audit'/('d0-stability.json' if args.stability_only else 'd0-result.json')).exists(), 'Use a new result directory; never overwrite existing evidence'

from reward_centering_audit_v38 import mass,estimate,center
import run as entry,distributed as d,compatible_calibration_v37 as compatible
tf=d.tf;original_server,original_device=tf.train.Server,tf.device
class LocalServer:
 target=''
 def __init__(self,*a,**kw):pass
tf.train.Server=LocalServer;tf.device=lambda n:original_device('/cpu:0' if isinstance(n,str) and n.startswith('/job:learner/') else n)
_,cfg=entry.template('mtpjoint','train');cfg.update(tf_threads=1,compatible_v31_checkpoint=str(R/'models/mtpjoint/model'),matching_clock_prior_checkpoint=None,seed=1307,independent_critic_adam_v34=args.optimizer=='independent_v34')
a,sess,*unused=d.setup(cfg,'learner',0);tf.train.Server,tf.device=original_server,original_device
sess.run(tf.global_variables_initializer());a.init_target();variables=tf.global_variables();fresh=sess.run(variables)
compatible.initialize(a,cfg)
critic_vars=[v for v in variables if v.op.name.startswith(('critic/','critic2/','target_critic/','target_critic2/')) and not compatible.optimizer_name(v.op.name)]
fresh_map={v.op.name:x for v,x in zip(variables,fresh)}
sess.run([tf.assign(v,fresh_map[v.op.name]) for v in critic_vars])
# Targets evaluate the same healthy deterministic actor plus explicit fixed TD3
# smoothing draws. The time forward is active throughout, never forced80ms.
current={v.op.name:v for v in variables}
for v in variables:
 if v.op.name.startswith('target_actor/'):
  suffix=v.op.name.split('/bn',1)
  if len(suffix)==2:
   source=next(z for z in variables if z.op.name.startswith('abstract_actor/') and z.op.name.endswith('/bn'+suffix[1]))
  else:
   name=v.op.name.replace('target_actor/target_actor/','abstract_actor/abstract_actor/')
   if name not in current:continue
   source=current[name]
  sess.run(tf.assign(v,source))
snapshot=compatible.policy_snapshot(a);base=sess.run(variables)
def restore(values):sess.run([tf.assign(v,x) for v,x in zip(variables,values)])
def query(states,actions,target=False):
 out=[]
 for start in range(0,len(states),512):
  q=sess.run([a.target_critic_actor_out,a.target_critic_actor_out2] if target else [a.critic_out,a.critic_out2],
   {a.s0:states[start:start+512],a.action:actions[start:start+512],a.is_training:False}) if not target else None
  if target:raise RuntimeError('Use explicit target policy query graph')
  out.append(np.minimum(q[0],q[1]).reshape(-1))
 return np.concatenate(out)
trials=json.loads((O/'audit/trials.json').read_text());analysis=json.loads((O/'audit/value-analysis.json').read_text())
rows=[]
for trial in trials:
 j=trial['job'];z=np.load(str(O/'factual-data'/(j['run_id']+'.npz')));n=len(z['a']);elapsed=np.r_[0.,np.cumsum(z['duration'].reshape(-1))]
 for i in range(n-1):
  # Factual20s reference requires no future intervention in the window.
  if i<trial['probe']['decision']-1:continue
  end=int(np.searchsorted(elapsed,elapsed[i]+20))
  if end>=n:continue
  weights=np.r_[1.,np.cumprod(z['discount'][i:end].reshape(-1))]
  rows.append(dict(s=z['s0'][i],action=z['a'][i],r=z['reward'][i,0],d=z['discount'][i,0],dt=z['duration'][i,0],next_s=z['s1'][i],
   endpoint=z['s0'][end],endpoint_action=z['a'][end],ref_r=np.dot(weights[:-1],z['reward'][i:end,0]),ref_d=weights[-1],scene=j['scene'],phase=j['phase'],run=j['run_id']))
def stack(k):return np.array([x[k] for x in rows],np.float32)
states,actions,next_states=stack('s'),stack('action'),stack('next_s');reward,discount,duration=stack('r'),stack('d'),stack('dt')
scene=np.array([x['scene'] for x in rows]);phase=np.array([x['phase'] for x in rows]);train=phase==0;held=phase==1
baselines=estimate(reward[train],discount[train],duration[train],scene[train]);global_b=float(np.average([baselines[str(i)] for i in range(6)],weights=[duration[train&(scene==i)].sum() for i in range(6)]));shift=global_b/(1-cfg['gamma'])
source_reader=tf.train.NewCheckpointReader(cfg['compatible_v31_checkpoint'])
# Preserve a fixed factual reference in original units, with source critic tail.
cold=sess.run(critic_vars);sess.run([tf.assign(v,source_reader.get_tensor(v.op.name)) for v in critic_vars])
reference=stack('ref_r')+stack('ref_d')*query(stack('endpoint'),stack('endpoint_action'))
sess.run([tf.assign(v,x) for v,x in zip(critic_vars,cold)])
next_policy=sess.run(a._moving_action,{a.s0:next_states,a.is_training:False})
target_action=tf.placeholder(tf.float32,[None,2]);target_q1=a.target_critic.build(a.s1,target_action);target_q2=a.target_critic2.build(a.s1,target_action)
target_min=tf.minimum(target_q1,target_q2)
cv=[v for v in tf.trainable_variables() if v.op.name.startswith(('critic/','critic2/'))]
gradients=tf.gradients(a.c_loss+a.c_loss2,cv);gradients=[g for g in gradients if g is not None]
window_policy=tf.concat([a._moving_action[:,:1],tf.stop_gradient(a._moving_action[:,1:2])],axis=1)
window_loss=-tf.reduce_mean(a.critic.build(a.s0,window_policy))
actor_variables=[v for v in tf.trainable_variables() if v.op.name.startswith('abstract_actor/')]
window_gradient=[g for g in tf.gradients(window_loss,actor_variables) if g is not None]
biases=[v for v in critic_vars if v.op.name in [x+'/dense/bias' for x in ['critic','critic2','target_critic','target_critic2']]];assert len(biases)==4
def shifted(values,amount):
 restore(values);sess.run([tf.assign_sub(v,tf.ones_like(v)*np.float32(amount)) for v in biases]);return sess.run(variables)
def target_update():
 # Keep stock V31 target operations, including its legacy overlapping scopes.
 ops=a.target_critic_update_op+a.target_critic_update_op2
 if args.target_order=='stock':sess.run(ops)
 else:
  for op in ops:sess.run(op)
def feed(ix,mode,rng):
 noisy=np.clip(next_policy[ix]+np.clip(rng.normal(0,.1,(len(ix),2)),-.2,.2),-1,1).astype(np.float32)
 q=sess.run(target_min,{a.s1:next_states[ix],target_action:noisy,a.is_training:False}).reshape(-1)
 r=reward[ix].astype(np.float64)
 if mode=='global':r=center(r,discount[ix],scene[ix],{str(i):global_b for i in range(6)})
 elif mode=='environment':r=center(r,discount[ix],scene[ix],baselines)
 y=(r+discount[ix]*q).astype(np.float32)[:,None]
 return {a.s0:states[ix],a.action:actions[ix],a.y:y,a.y2:y,a.is_training:False}
pools=[np.where(train&(scene==i))[0] for i in range(6)]
def batches(rng):return np.concatenate([rng.choice(p,86 if i<2 else 85,replace=True) for i,p in enumerate(pools)])
registration=dict(source='Preserved120 factual independent connections, six development scenes',train='phase0',heldout='phase1 independent connections/phase',
 baselines=baselines,global_baseline=global_b,baseline_source='Training rows only, physical-time weighted utility density, frozen before fitting',
 behavior='Healthy V31 deterministic window, single randomized timing action then healthy continuation; not V35 iid calibration distribution',
 architecture='149/256/256/2 actor unchanged; original twin critics',optimizer=('Original V31 shared Adam' if args.optimizer=='legacy' else 'V34 independent critic Adam, original loss/learning rates'),
 target_execution=args.target_order,optimizer_flag=cfg['independent_critic_adam_v34'],
 target='Frozen healthy actor function, explicit identical TD3 smoothing draws; critic Polyak .001, stock overlapping target ops retained',
 arms=['raw_zero_prior','raw_global_bias_prior','environment_centered_zero_prior'],screen_iterations=5000,
 null='Raw zero prior versus globally centered prior shifted down by bbar/(1-gamma)',
 tolerances=dict(single_td_max_abs=.0002,single_gradient_max_abs=.001,short100_q_max_abs=.02),
 limits=['Factual20s reference uses deterministic continuation whereas smoothed Bellman policy differs',
 'Overlapping rows are not independent futures','No actor optimization or network improvement claim','No converged Q truth or V35 checkpoint sequence exists',
 'Only one seed; this is a bounded mechanism screen, not confirmatory evidence'])
if args.stability_only:
 # Additional stability diagnostic after the preregistered100-step null failed.
 # This does not change the main three-arm experiment or optimizer configuration.
 checks={}
 beta=[v for v in variables if v.op.name.startswith(('beta1_power','beta2_power')) or v.op.name=='global_step']
 for schedule in ['stock_unordered','serialized_existing_operations_diagnostic_only']:
  traces=[]
  for label,mode,amount in [('rawA','raw',0.),('rawB','raw',0.),('centered','global',shift)]:
   initial=shifted(base,amount) if amount else base
   restore(initial);rng=np.random.RandomState(210);history=[]
   for k in range(100):
    ix=batches(rng);fd=feed(ix,mode,rng)
    if schedule=='stock_unordered':sess.run(a.critic_train_op,fd)
    else:
     for op in a.critic_train_op:sess.run(op,fd)
    if schedule=='stock_unordered':target_update()
    else:
     for op in a.target_critic_update_op+a.target_critic_update_op2:sess.run(op)
    if k in [0,9,99]:
     history.append(dict(iteration=k+1,q=(query(states[:64],actions[:64])+(shift if amount else 0)).tolist(),clock=[float(v) for v in sess.run(beta)]))
   traces.append(dict(label=label,history=history))
  comparisons={}
  for j in [1,2]:
   comparisons[traces[j]['label']]=[dict(iteration=a0['iteration'],q_max_abs=float(np.max(np.abs(np.array(a0['q'])-np.array(b0['q'])))),q_rms=float(np.sqrt(np.mean((np.array(a0['q'])-np.array(b0['q']))**2))),clock_A=a0['clock'],clock_B=b0['clock']) for a0,b0 in zip(traces[0]['history'],traces[j]['history'])]
  checks[schedule]=dict(comparisons=comparisons,traces=traces)
  (R/'audit/d0-stability.json').write_text(json.dumps(dict(status='RUNNING',checks=checks,registered_main_tolerance=.02),indent=2))
 compatible.assert_policy_frozen(a,snapshot)
 (R/'audit/d0-stability.json').write_text(json.dumps(dict(status='COMPLETED_DIAGNOSTIC',checks=checks,registered_main_tolerance=.02,actor_bn_unchanged=True,limits=['Serialized operations change execution semantics and are diagnostic only','No automatic relaxation of the main null tolerance']),indent=2))
 sess.close()
 sys.exit(0)
(R/'configs/d0-registration.json').write_text(json.dumps(registration,indent=2))
start=time.time();rng=np.random.RandomState(919);ix=batches(rng)
restore(base);fd=feed(ix,'raw',np.random.RandomState(11));raw_td,raw_grad=sess.run([a.td_error,gradients],fd)
down=shifted(base,shift);fdc=feed(ix,'global',np.random.RandomState(11));center_td,center_grad=sess.run([a.td_error,gradients],fdc)
td_error=float(np.max(np.abs(raw_td-center_td)));gradient_error=max(float(np.max(np.abs(x-y))) for x,y in zip(raw_grad,center_grad))
assert td_error<=.0002 and gradient_error<=.001,(td_error,gradient_error)
# Compare a short, deterministic replay trajectory; long chaotic equality is not required.
null=[]
for mode,initial in [('raw',base),('global',down)]:
 restore(initial);rng=np.random.RandomState(210);history=[]
 for k in range(100):
  ix=batches(rng);fd=feed(ix,mode,rng);sess.run(a.critic_train_op,fd);target_update()
  if k in [0,9,99]:history.append(query(states[:64],actions[:64])+(shift if mode=='global' else 0))
 null.append(np.array(history))
short_error=float(np.max(np.abs(null[0]-null[1])))
result=dict(status='RUNNING',registration=registration,tf_null=dict(td_max_abs=td_error,gradient_max_abs=gradient_error,short100_q_max_abs=short_error,
 short100_within_registered_tolerance=short_error<=.02),arms={},source_factual_reference='Observed20s deterministic-policy return + small fixed source tail; not exact smoothed-policy value')
result['eligible_for_online_release']=False
(R/'audit/d0-result.json').write_text(json.dumps(result,indent=2))
if not result['tf_null']['short100_within_registered_tolerance'] and not args.allow_failed_null_descriptive_screen:
 result['status']='BLOCKED_REGISTERED_NULL';(R/'audit/d0-result.json').write_text(json.dumps(result,indent=2))
 raise RuntimeError('Registered short null failed; no three-arm screen or actor release. Run --stability-only to audit A/A.')
query_states=np.array([x['probe']['state'] for x in trials],np.float32);query_actions=np.array([x['probe']['policy_action'] for x in trials],np.float32)
query_scene=np.array([x['job']['scene'] for x in trials]);query_phase=np.array([x['job']['phase'] for x in trials]);grid_actions=[]
from joint_mtp import mtp_action
def window_grad():
 return np.concatenate([x.reshape(-1) for x in sess.run(window_gradient,{a.s0:query_states,a.is_training:False})])
current_critic=sess.run(critic_vars);sess.run([tf.assign(v,source_reader.get_tensor(v.op.name)) for v in critic_vars]);source_window_gradient=window_grad();sess.run([tf.assign(v,x) for v,x in zip(critic_vars,current_critic)])
def diagnostics(mode):
 q=query(states[held],actions[held]);offset=np.array([baselines[str(i)]/(1-cfg['gamma']) for i in scene[held]]) if mode=='environment' else 0
 errors=q+offset-reference[held];by=[]
 for i in range(6):
  e=errors[scene[held]==i];by.append(dict(scene=i,count=len(e),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2)))))
 qs=[]
 for period in [20,40,80,120,160]:
  ac=query_actions.copy();ac[:,1]=mtp_action(period,cfg);qs.append(query(query_states,ac))
 qs=np.stack(qs,axis=1)
 g=window_grad();denom=np.linalg.norm(g)*np.linalg.norm(source_window_gradient)
 return dict(by_scene=by,scene_equal_rmse=float(np.mean([x['rmse'] for x in by])),period_q=qs.tolist(),window_actor_gradient_cosine_to_source=float(np.dot(g,source_window_gradient)/denom) if denom>1e-12 else None,
  long_minus_short_mean_by_scene=[float((qs[query_scene==i,-1]-qs[query_scene==i,0]).mean()) for i in range(6)])
for name,mode,initial in [('raw_zero_prior','raw',base),('raw_global_bias_prior','raw',shifted(base,-shift)),('environment_centered_zero_prior','environment',base)]:
 restore(initial);rng=np.random.RandomState(631);history=[]
 for k in range(5000):
  ix=batches(rng);fd=feed(ix,mode,rng);loss=sess.run([a.critic_train_op,a.c_loss,a.c_loss2],fd)[1:];assert np.isfinite(loss).all();target_update()
  if k+1 in [100,1000,5000]:
   compatible.assert_policy_frozen(a,snapshot);history.append(dict(iteration=k+1,loss=[float(x) for x in loss],heldout=diagnostics(mode)))
   result['arms'][name]=dict(history=history,actor_updates=0,critic_iterations=k+1,critic_optimizer_operations=2*(k+1));(R/'audit/d0-result.json').write_text(json.dumps(result,indent=2))
 compatible.assert_policy_frozen(a,snapshot)
 if args.save_critic_diagnostics:
  destination=R/'diagnostic-critics';destination.mkdir(exist_ok=True)
  path=destination/(name+'.npz');assert not path.exists()
  np.savez_compressed(str(path),**{v.op.name:value for v,value in zip(critic_vars,sess.run(critic_vars))})
  result['arms'][name]['diagnostic_critic_tensors']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,kind='Critics only; no actor, optimizer, replay or resume state')
result.update(status='COMPLETED_OFFLINE_SCREEN',elapsed_s=time.time()-start,actor_bn_unchanged=True,actor_updates=0,automatic_training_release=False)
(R/'audit/d0-result.json').write_text(json.dumps(result,indent=2));sess.close();print(json.dumps(dict(status=result['status'],null=result['tf_null'],elapsed_s=result['elapsed_s'])))
