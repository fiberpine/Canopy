import os,pathlib,json,sys,time,hashlib,math
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',CUDA_VISIBLE_DEVICES='')
R=pathlib.Path(__file__).resolve().parent.parent;F=R/'framework';sys.path.insert(0,str(F));sys.path.insert(0,str(F/'code/runtime/_deps'))
import numpy as np
from capture_core import interval,finite_suffix,fixed_indices,PERIODS
from common_period_tf import Model,ONLINE
from common_period import encoded_period
from readonly_guard import ReadOnlySession
reg=json.loads((R/'registration.json').read_text());node=int(sys.argv[1]);rows=json.loads((R/'result.json').read_text());old=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');selection=json.loads((old/'evaluation/selected-model.json').read_text());prefix=selection['checkpoint'];meta=json.loads(pathlib.Path(prefix+'.metadata.json').read_text());source='/home/experiment/orca-release-current/models/mtpjoint/model';model=Model(source,meta['K'],meta['n0'],reward_config=meta['reward_config']);model.restore(prefix);tf=model.tf
with model.graph.as_default():variables=tf.global_variables();actor_vars=model.parameters(ONLINE)
def digest():
 h=hashlib.sha256()
 for v,x in zip(variables,model.session.run(variables)):h.update(v.op.name.encode());h.update(np.asarray(x).tobytes())
 return h.hexdigest()
before=digest();original=model.session;guard=ReadOnlySession(original);model.session=guard
with model.graph.as_default():
 q2=model.critic(model.s,tf.concat([model.mu,model.t],1),'critic2/');number=model.number(model.s,tf.concat([model.mu,model.t],1),'number/');qbar=(model.actor_q+q2)*.5;dq1=tf.gradients(model.actor_q,model.mu)[0];dqbar=tf.gradients(qbar,model.mu)[0]
 grads=tf.gradients(model.rl_loss,actor_vars);regs=tf.gradients(model.reg_loss,actor_vars);flat=lambda gs:tf.concat([tf.reshape(g,[-1]) for g in gs if g is not None],0);rl=flat(grads);reggrad=flat(regs);layers={key:flat([g for g,v in zip(grads,actor_vars) if g is not None and name in v.op.name]) for key,name in [('fc1','fc1/'),('fc2','fc2/'),('output','fc1_1/'),('period_input','period_w')]}
 def bn_variant(batch_second):
  v=model.v;x=tf.matmul(model.s,v[ONLINE+'fc1/w'])+v[ONLINE+'fc1/b'];mean,var=tf.nn.moments(x,[0]);x=tf.nn.leaky_relu(tf.nn.batch_normalization(x,mean,var,v[ONLINE+'bn1/beta'],None,.001));x=tf.matmul(x,v[ONLINE+'fc2/w'])+v[ONLINE+'fc2/b']+tf.matmul(model.t,v[ONLINE+'period_w']);mean,var=tf.nn.moments(x,[0]) if batch_second else (v[ONLINE+'bn2/moving_mean'],v[ONLINE+'bn2/moving_variance']);x=tf.nn.leaky_relu(tf.nn.batch_normalization(x,mean,var,v[ONLINE+'bn2/beta'],None,.001));z=(tf.matmul(x,v[ONLINE+'fc1_1/w'])+v[ONLINE+'fc1_1/b'])[:,:1];return tf.tanh(z)
 alt1=bn_variant(False);alt2=bn_variant(True)
def feed(s,t):
 a=np.zeros((len(s),2),np.float32);a[:,1]=encoded_period(t);return {model.s:s,model.action:a}
def values(s,t):
 values_=[guard.run([model.mu,model.z,model.actor_q,q2,number,dq1,dqbar],feed(row[None,:],t)) for row in s]
 return [np.concatenate([value[j] for value in values_],axis=0) for j in range(7)]
def cos(a,b):
 d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(np.dot(a,b)/d) if d else None
out=[]
try:
 for row in rows:
  if row['status']!='COMPLETED':continue
  if time.time()>reg['deadline_epoch']-600:break
  job=row['job'];d=F/'logs'/job['run'];z=np.load(str(d/'captured-boundaries.npz'),allow_pickle=False);s=z['s0'];sn=z['s1'];action=z['action'];raw=z['raw_next'];dt=raw[:,3];parts=[interval(x,reg['K']) for x in raw];reward=np.asarray([x['reward'] for x in parts]);t=job['period_ms'];elapsed=np.r_[0.,np.cumsum(dt[:-1])]
  mu,zvalue,q1v,q2v,nv,dqa,dqba=values(s,t);parity=float(np.max(abs(mu[:,0]-action[:,0])));assert parity<=reg['tolerances']['actor_abs'],(job['run'],parity)
  ids=set(fixed_indices(dt));cross=np.flatnonzero((abs(action[1:,0])>.99)!=(abs(action[:-1,0])>.99))[:4]+1
  for i in cross:
   for j in [i-1,i,i+1]:
    if 0<=j<len(s):ids.add(j)
  ids=sorted(ids);pre=np.concatenate((z['initial_raw'][None,:],raw[:-1]),axis=0);points=[]
  for i in ids:
   single=values(s[i:i+1],t);a=float(action[i,0]);diagnostic=float(single[0][0,0]);zz=float(single[1][0,0]);deriv=float(np.float32(1)-np.float32(diagnostic)*np.float32(diagnostic));points.append(dict(index=i,elapsed_s=float(elapsed[i]),window=a,diagnostic_window=diagnostic,forward_bitwise_equal=bool(np.float32(a)==np.float32(diagnostic)),z=zz,abs_gt_099=bool(abs(a)>.99),float32_exact_saturation=bool(abs(a)==1),diagnostic_exact_saturation=bool(abs(diagnostic)==1),float32_tanh_derivative=deriv,Q1_dQ_da=float(single[5][0,0]),Qmean_dQ_da=float(single[6][0,0]),RL_dloss_dz=-float(single[5][0,0])*deriv,regularizer_dloss_dz=.002*zz,integer_percent=int(100*4**float(np.clip(a,-1,1))),cwnd=float(pre[i,5]),in_flight=float(pre[i,10]),retrans_out=float(pre[i,11]),srtt_ms=float(pre[i,8]),avg_ACK_RTT_ms=float(pre[i,0]),rtt_count=float(pre[i,2]),min_rtt_ms=float(pre[i,14]),interval_utility=float(parts[i]['utility']),raw_R=float(reward[i]),centered_R=float(parts[i]['centered'])))
  # Every state queried under its executed fixed period; labels remain within run.
  suffix=[]
  for start_time in reg['S4']['start_grid_seconds']:
   i=int(np.searchsorted(elapsed,start_time));end=int(np.searchsorted(np.r_[0,np.cumsum(dt)],elapsed[i]+20,side='left'))-1
   if end>=len(s):continue
   tail=values(sn[end:end+1],t);label=finite_suffix(reward,dt,i,reg['K'],float((tail[2][0,0]+tail[3][0,0])*.5),float(tail[4][0,0]));pred=float((q1v[i,0]+q2v[i,0])*.5);suffix.append(dict(start_index=i,start_s=float(elapsed[i]),predicted_Qmean=pred,predicted_Q1=float(q1v[i,0]),predicted_Q2=float(q2v[i,0]),predicted_N=float(nv[i,0]),Q_label_minus_prediction=label['Q_label']-pred,N_label_minus_prediction=label['N_label']-float(nv[i,0]),tail_twin_half_difference=label['suffix_discount']*abs(float(tail[2][0,0]-tail[3][0,0]))*.5,**label))
  shared=s[np.linspace(0,len(s)-1,reg['S3']['gradient_shared_states_per_run']).astype(int)];vectors={};layer_vectors={key:{} for key in layers};gn=[];preference=[]
  for candidate in PERIODS:
   vs=guard.run([rl,reggrad,model.mu,qbar]+list(layers.values()),feed(shared,candidate));vectors[candidate]=vs[0];gn.append(dict(period_ms=candidate,RL_norm=float(np.linalg.norm(vs[0])),regularizer_norm=float(np.linalg.norm(vs[1])),window_mean=float(vs[2].mean()),Qmean=float(vs[3].mean())))
   preference.append(vs[3][:,0])
   for key,val in zip(layers,vs[4:]):layer_vectors[key][candidate]=val
  pairs=[dict(periods=[t1,t2],RL_cosine=cos(vectors[t1],vectors[t2]),by_layer={key:cos(v[t1],v[t2]) for key,v in layer_vectors.items()}) for j,t1 in enumerate(PERIODS) for t2 in PERIODS[j+1:]];winner=np.argmax(np.asarray(preference).T,axis=1)
  actual,ba,bb=guard.run([model.mu,alt1,alt2],feed(shared,t));bn=dict(batch1_moving2_action_abs_mean=float(np.mean(abs(ba-actual))),batch_both_action_abs_mean=float(np.mean(abs(bb-actual))))
  r=dict(run=job['run'],job=job,actor_forward_max_abs_error=parity,point_samples=points,all_run=dict(decisions=len(s),boundary_fraction=float(np.mean(abs(action[:,0])>.99)),exact_float32_saturation=float(np.mean(abs(action[:,0])==1)),diagnostic_batch_zero_RL_fraction=float(np.mean((1-mu*mu)*dqa==0)),utility_time_weighted=float(np.dot(np.asarray([p['utility'] for p in parts]),dt)/dt.sum()),raw_R_sum=float(reward.sum()),centered_R_sum=float(sum(p['centered'] for p in parts))),finite_returns=suffix,shared_state_gradient_norms=gn,shared_state_gradient_pairs=pairs,offpolicy_Qmean_argmax_counts={str(period):int(np.sum(winner==j)) for j,period in enumerate(PERIODS)},BN_hypothetical=bn,initial_state_sha256=hashlib.sha256(z['initial_state'].tobytes()).hexdigest(),instrumentation_pass=row['instrumentation_pass'],interpretation='Cross-period queries are off-policy on the same observed states; own-period finite returns are same-policy with frozen estimated tail. Overlapping suffixes are not independent runs. No bias correction or optimization.')
  out.append(r);(R/'analysis-progress.json').write_text(json.dumps(dict(status='RUNNING',runs=out,variables_before=before,optimizer_operations=0),indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
 after=digest();assert before==after
 result=dict(status='COMPLETED' if len(out)==sum(r['status']=='COMPLETED' for r in rows) else 'PARTIAL_BUDGET',runs=out,model_variables_unchanged=before==after,variables_before=before,variables_after=after,optimizer_operations=0,new_TCP_during_analysis=0,lambda_scan=False,ended_epoch=time.time());(R/'analysis-results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));print(json.dumps(dict(status=result['status'],runs=len(out),unchanged=result['model_variables_unchanged'])))
finally:original.close()
