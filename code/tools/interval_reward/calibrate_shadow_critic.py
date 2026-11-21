"""Bounded TF1 offline policy evaluation on exported, unchanged149 boundary data.
Both critics restart identically. Only reward changes. No actor/BN/TCP updates.
One-step TD uses each arm's own learned target and unchanged deterministic actor.
No old-objective MC tail, extrapolated counterfactual labels or automatic release.
"""
import argparse,hashlib,importlib.util,json,os,pathlib,sys,time,traceback
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=pathlib.Path,required=True)
 p.add_argument('--data',type=pathlib.Path,nargs='+',required=True);p.add_argument('--out',type=pathlib.Path,required=True)
 p.add_argument('--steps',type=int,default=2000);p.add_argument('--checkpoint');args=p.parse_args()
 assert 0<args.steps<=5000 and not args.out.exists()
 result=dict(status='NOT_RUN',actor_release=False,actor_updates=0,new_tcp_runs=0,
  raw_schema='v31-observed-interval-orca-density-srtt-v1',shadow_schema='shadow-v1-same-149-state-ack-sample-mean-rtt-density',
  target='One-step own-reward deterministic-policy TD. Each reward has its own target critic. No MC label or old raw critic tail.',
  limits=['Heldout Bellman fit does not establish trueQ or causal gradient quality','Forced-period logs are off-policy samples; unchanged149 finite history need not be Markov',
  'Shadow uses ACK-sample RTT mean, not time integral or delay quantile','No-fresh-RTT rows excluded identically from both arms'])
 args.out.parent.mkdir(parents=True,exist_ok=True)
 def write():args.out.write_text(json.dumps(result,indent=2))
 if importlib.util.find_spec('tensorflow') is None:
  result.update(status='BLOCKED_LOCAL_NO_TENSORFLOW');write();print(json.dumps(result));return 2
 try:
  root=args.package_root.resolve();framework=root/'frameworks/mtpjoint';sys.path[:0]=[str(root),str(framework)]
  import run as entry,distributed as d,compatible_calibration_v37 as compatible
  tf=d.tf;assert tf.__version__.startswith('1.')
  metas=[];sources=[]
  for folder in args.data:
   manifest=json.loads((folder/'manifest.json').read_text());assert manifest['status']=='PASS_EXPORTED' and not manifest['errors']
   for item in manifest['runs']:
    if 'npz' not in item:continue
    path=folder/pathlib.Path(item['npz']).name;assert sha(path)==item['npz_sha256']
    meta=json.loads((folder/(item['run_id']+'.json')).read_text());assert meta['state_max_abs_error']==0
    assert meta['raw_schema']==result['raw_schema'] and meta['shadow_schema']==result['shadow_schema']
    assert meta['state_dim']==149 and not meta['state_interface_changed']
    z=np.load(str(path));ix=np.where(z['valid_shadow'])[0]
    assert len(ix)>0 and all(np.isfinite(z[k][ix]).all() for k in z.files if k!='valid_shadow')
    sources.append(dict(meta=meta,data={k:z[k][ix] for k in z.files}));metas.append(meta)
  assert len(set(m['source_policy_sha'] for m in metas))==1
  train_runs=[i for i,m in enumerate(metas) if m['split']=='train'];held_runs=[i for i,m in enumerate(metas) if m['split']=='heldout']
  assert train_runs and held_runs and not set(metas[i]['run_id'] for i in train_runs)&set(metas[i]['run_id'] for i in held_runs)
  offsets=np.r_[0,np.cumsum([len(x['data']['a']) for x in sources])]
  arrays={k:np.concatenate([x['data'][k] for x in sources]) for k in sources[0]['data']}
  pools=[np.arange(offsets[i],offsets[i+1]) for i in train_runs];held=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in held_runs])
  # Fixed heldout rows, no training on heldout connections. All available rows used for measurements.
  held=held[np.linspace(0,len(held)-1,min(4096,len(held))).astype(int)]
  _,cfg=entry.template('mtpjoint','train');cfg.update(tf_threads=1,seed=1307,independent_critic_adam_v34=True,
    compatible_v31_checkpoint=args.checkpoint or str(root/'models/mtpjoint/model'),matching_clock_prior_checkpoint=None)
  assert all(m['gamma']==cfg['gamma'] and m['decision_cost']==cfg['decision_cost'] for m in metas)
  server,device=tf.train.Server,tf.device
  class LocalServer:
   target=''
   def __init__(self,*a,**kw):pass
  tf.train.Server=LocalServer;tf.device=lambda n:device('/cpu:0' if isinstance(n,str) and n.startswith('/job:learner/') else n)
  try:a,sess,*_=d.setup(cfg,'learner',0)
  finally:tf.train.Server,tf.device=server,device
  variables=tf.global_variables();sess.run(tf.global_variables_initializer());a.init_target()
  initial_random={v.op.name:x for v,x in zip(variables,sess.run(variables))}
  names=[v for v in variables if not compatible.optimizer_name(v.op.name) and v.op.name not in ['epoch','global_step']]
  canonical=tf.train.Saver(var_list={v.op.name:v for v in names});canonical.restore(sess,cfg['compatible_v31_checkpoint'])
  s=arrays['s0'][:512];ac=arrays['a'][:512]
  fd={a.s0:s,a.s1:s,a.action:ac,a.target_actor_policy:ac,a.is_training:False}
  probes=[a._moving_action,a.critic_out,a.critic_out2,a.target_actor_out,a.target_critic_actor_out,a.target_critic_actor_out2]
  expected=sess.run(probes,fd);proof=compatible.initialize(a,cfg);actual=sess.run(probes,fd)
  for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)
  # Fresh objective-specific critics: deliberately do not preserve raw-Q values as shadow priors.
  trainable=tf.trainable_variables();q1=[v for v in trainable if v.op.name.startswith('critic/')];q2=[v for v in trainable if v.op.name.startswith('critic2/')]
  allmap={v.op.name:v for v in variables};target_pairs=[]
  for target,source in [('target_critic/','critic/'),('target_critic2/','critic2/')]:
   for v in trainable:
    if v.op.name.startswith(target):target_pairs.append((v,allmap[source+v.op.name[len(target):]]))
  assert len(q1)==len(q2)==6 and len(target_pairs)==12
  sess.run([tf.assign(v,initial_random[v.op.name]) for v in q1+q2])
  sess.run([tf.assign(t,v) for t,v in target_pairs])
  target_action=tf.placeholder(tf.float32,[None,2],name='shadow_explicit_target_action')
  target_q=tf.minimum(a.target_critic.build(a.s1,target_action),a.target_critic2.build(a.s1,target_action))
  y=tf.placeholder(tf.float32,[None,1],name='shadow_same_objective_td_target')
  l1=tf.reduce_mean(tf.square(a.critic_out-y));l2=tf.reduce_mean(tf.square(a.critic_out2-y))
  opt1=tf.train.AdamOptimizer(.001,name='shadow_independent_q1');opt2=tf.train.AdamOptimizer(.001,name='shadow_independent_q2')
  op1=opt1.minimize(l1,var_list=q1);op2=opt2.minimize(l2,var_list=q2)
  polyak=[tf.assign(t,.001*v+.999*t) for t,v in target_pairs]
  oldnames={v.op.name for v in variables}
  newvars=[v for v in tf.global_variables() if v.op.name not in oldnames];sess.run(tf.variables_initializer(newvars))
  allvars=tf.global_variables();base=sess.run(allvars)
  frozen=[v for v in allvars if (v.op.name.startswith(('abstract_actor/','target_actor/')) or '/bn' in v.op.name) and not compatible.optimizer_name(v.op.name)]
  frozen_values=sess.run(frozen)
  def check_frozen():
   for v,x,z in zip(frozen,frozen_values,sess.run(frozen)):np.testing.assert_array_equal(x,z,err_msg=v.op.name)
   assert int(sess.run(a.global_step))==0,'Original optimizer counters must remain fresh and unused'
  def action(states):
   return np.concatenate([sess.run(a._moving_action,{a.s0:states[i:i+512],a.is_training:False}) for i in range(0,len(states),512)])
  next_action=action(arrays['s1'])
  def td(ix,reward_key):
   tail=sess.run(target_q,{a.s1:arrays['s1'][ix],target_action:next_action[ix],a.is_training:False}).reshape(-1)
   return arrays[reward_key][ix]+arrays['discount'][ix]*tail,tail
  def diagnostic(reward_key):
   allq=[];ally=[];tails=[]
   for lo in range(0,len(held),512):
    ix=held[lo:lo+512];yy,tail=td(ix,reward_key)
    q=sess.run([a.critic_out,a.critic_out2],{a.s0:arrays['s0'][ix],a.action:arrays['a'][ix],a.is_training:False})
    allq.append(np.stack(q).reshape(2,-1));ally.append(yy);tails.append(tail)
   qq=np.concatenate(allq,axis=1);yy=np.concatenate(ally);tail=np.concatenate(tails)
   return dict(n=len(yy),rmse_q1=float(np.sqrt(np.mean((qq[0]-yy)**2))),rmse_q2=float(np.sqrt(np.mean((qq[1]-yy)**2))),
    bias_q1=float(np.mean(qq[0]-yy)),q1_mean=float(qq[0].mean()),own_target_mean=float(yy.mean()),own_bootstrap_tail_mean=float(tail.mean()),
    own_bootstrap_discounted_mean=float(np.mean(yy-arrays[reward_key][held])),twin_gap=float(np.mean(np.abs(qq[0]-qq[1]))),true_q=False)
  result.update(status='RUNNING',full149_function_restore_exact=True,copied_source_tensors=len(proof['copied_names']),
    train_connections=[metas[i]['run_id'] for i in train_runs],heldout_connections=[metas[i]['run_id'] for i in held_runs],
    train_rows=sum(len(p) for p in pools),heldout_evaluated_rows=len(held),fresh_critic_initialization=True,fresh_adam=True,
    architecture='Unchanged149-state2-action healthy V31 actor; original twin MLP critics; exact-scope independent Adam, deterministic frozen actor target, no smoothing',
    source_checkpoint=cfg['compatible_v31_checkpoint'],sources=[{k:m[k] for k in ('run_id','split','npz_sha256','source_policy_sha')} for m in metas],arms={})
  write();started=time.time()
  for arm,key in [('raw','reward_raw'),('shadow','reward_shadow')]:
   sess.run([tf.assign(v,x) for v,x in zip(allvars,base)]);rng=np.random.RandomState(631);history=[]
   for step in range(args.steps):
    counts=[512//len(pools)+(i<512%len(pools)) for i in range(len(pools))]
    ix=np.concatenate([rng.choice(pool,n,replace=True) for pool,n in zip(pools,counts)])
    yy,_=td(ix,key);fd={a.s0:arrays['s0'][ix],a.action:arrays['a'][ix],y:yy.astype(np.float32)[:,None],a.is_training:False}
    sess.run(op1,fd);sess.run(op2,fd);sess.run(polyak)
    if step+1 in sorted(set([1,min(100,args.steps),min(1000,args.steps),args.steps])):
     check_frozen();history.append(dict(step=step+1,heldout=diagnostic(key)))
     result['arms'][arm]=dict(critic_iterations=step+1,critic_optimizer_operations=2*(step+1),actor_updates=0,history=history);write()
   check_frozen()
  result.update(status='COMPLETED_OFFLINE_TWO_REWARD_TD',elapsed_s=time.time()-started,actor_bn_frozen=True,actor_release=False)
  sess.close();write();print(json.dumps({k:result[k] for k in ('status','elapsed_s','actor_bn_frozen','actor_release')}));return 0
 except Exception:
  result.update(status='NO_GO',traceback=traceback.format_exc());write();raise
if __name__=='__main__':sys.exit(main())
