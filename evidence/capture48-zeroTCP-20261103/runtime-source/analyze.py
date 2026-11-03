import os,pathlib,json,sys,time,hashlib,collections
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
R=pathlib.Path(__file__).resolve().parent;reg=json.loads((R/'registration.json').read_text());OLD=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');C=pathlib.Path(reg['capture_root']);sys.path.insert(0,str(C/'framework'));sys.path.insert(0,str(C/'code'));sys.path.insert(0,str(C/'framework/code/runtime/_deps'))
from common_period_tf import Model,ONLINE
from common_period import encoded_period
from readonly_guard import ReadOnlySession
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import sklearn
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
packs=[np.load(str(R/'exports'/('node%d.npz'%i)),allow_pickle=False) for i in range(5)];x=np.concatenate([p['s0'] for p in packs]);dt=np.concatenate([p['dt'] for p in packs]);rid=np.concatenate([p['run'] for p in packs]);raw=np.concatenate([p['pre_raw'] for p in packs]);elapsed=np.concatenate([p['elapsed'] for p in packs]);y=np.column_stack([np.concatenate([p['capacity_run'] for p in packs]),np.concatenate([p['capacity_recent'] for p in packs])]);node_results=[json.loads((R/'exports'/('node%d.json'%i)).read_text()) for i in range(5)];meta={m['job']['order']:m for r in node_results for m in r['metadata']};assert len(x)==63156 and len(meta)==48
sel=json.loads((OLD/'evaluation/selected-model.json').read_text());prefix=sel['checkpoint'];mp=pathlib.Path(prefix+'.replay.npz');replay_before=sha(mp);md=json.loads(pathlib.Path(prefix+'.metadata.json').read_text());m=Model('/home/experiment/orca-release-current/models/mtpjoint/model',md['K'],md['n0'],reward_config=md['reward_config']);m.restore(prefix);tf=m.tf
with m.graph.as_default():
 variables=tf.global_variables();q2mu=m.critic(m.s,tf.concat([m.mu,m.t],1),'critic2/');dqm=tf.gradients((m.actor_q+q2mu)*.5,m.mu)[0];dq1=tf.gradients(m.actor_q,m.mu)[0];de1=tf.gradients(m.q[0],m.action)[0];demean=tf.gradients((m.q[0]+m.q[1])*.5,m.action)[0]
def digest():
 h=hashlib.sha256()
 for v,a in zip(variables,m.session.run(variables)):h.update(v.op.name.encode());h.update(np.asarray(a).tobytes())
 return h.hexdigest()
before=digest();original=m.session;m.session=ReadOnlySession(original);gradient_points=[];gradient_runs=[]
try:
 for run_id,info in sorted(meta.items()):
  j=info['job']
  if j['period_ms']!=20:continue
  ids=np.flatnonzero(rid==run_id);grid=np.linspace(0,min(59.8,float(elapsed[ids][-1])),128);local=np.unique(np.minimum(len(ids)-1,np.searchsorted(elapsed[ids],grid)));indices=ids[local];summary=[]
  cap=np.load(str(C/'framework/logs'/j['run']/'captured-boundaries.npz'),allow_pickle=False) if j['node']==0 else None
  # Execution actions are reconstructed only for parity on learner-local captured runs.
  for idx,li in zip(indices,local):
   srtt=float(raw[idx,8]/raw[idx,14]);ack=float((raw[idx,0] if raw[idx,2]>0 else raw[idx,8])/raw[idx,14]);state_ratio=float(np.expm1(x[idx,73])/np.expm1(x[idx,72]));assert abs(srtt-state_ratio)<=1e-4,(j['run'],srtt,state_ratio)
   for t in reg['gradient_periods']:
    action=np.asarray([[0.,encoded_period(t)]],np.float32);feed={m.s:x[idx:idx+1],m.action:action};mu,z,q1,q2,g1,gm=m.session.run([m.mu,m.z,m.actor_q,q2mu,dq1,dqm],feed);actual=float(mu[0,0])
    if cap is not None and t==20:assert abs(actual-float(cap['action'][li,0]))<=1e-6
    # Hold the A20 executed-policy window fixed while changing critic period.
    ownfeed={m.s:x[idx:idx+1],m.action:np.asarray([[0.,encoded_period(20)]],np.float32)};a20=float(m.session.run(m.mu,ownfeed)[0,0]);fixed={m.s:x[idx:idx+1],m.action:np.asarray([[a20,encoded_period(t)]],np.float32)};e1,em=m.session.run([de1,demean],fixed)
    point=dict(run=j['run'],condition=j['condition'],repeat=j['repeat'],state_index=int(li),elapsed_s=float(elapsed[idx]),SRTT_over_min=srtt,ACK_over_min=ack,high_SRTT=srtt>=2,high_ACK=ack>=2,queried_period_ms=t,own_A20_window=a20,period_policy_window=actual,z=float(z[0,0]),Q1=float(q1[0,0]),Q2=float(q2[0,0]),Q1_gradient_at_period_policy=float(g1[0,0]),Qmean_gradient_at_period_policy=float(gm[0,0]),Q1_gradient_at_fixed_A20_window=float(e1[0,0]),Qmean_gradient_at_fixed_A20_window=float(em[0,0]))
    gradient_points.append(point);summary.append(point)
  for queue in ['all','high_SRTT','low_SRTT','high_ACK','low_ACK']:
   for t in reg['gradient_periods']:
    ps=[p for p in summary if p['queried_period_ms']==t and (queue=='all' or p['high_'+queue.split('_',1)[1]]==(queue.startswith('high_')))];gs=[p['Q1_gradient_at_period_policy'] for p in ps];gmv=[p['Qmean_gradient_at_period_policy'] for p in ps]
    gradient_runs.append(dict(run=j['run'],condition=j['condition'],repeat=j['repeat'],queue_stratum=queue,queried_period_ms=t,points=len(ps),Q1_mean=float(np.mean(gs)) if ps else None,Q1_positive_fraction=float(np.mean(np.asarray(gs)>0)) if ps else None,Qmean_mean=float(np.mean(gmv)) if ps else None,Qmean_positive_fraction=float(np.mean(np.asarray(gmv)>0)) if ps else None))
 after=digest();assert before==after
finally:original.close()
save('gradient-points.json',gradient_points);save('gradient-runs.json',gradient_runs)
data=np.load(str(mp),allow_pickle=False);rs=data['s0'];minimum=np.expm1(rs[:,72]);srtt=np.expm1(rs[:,73]);valid=minimum>0;ratio=np.divide(srtt,minimum,out=np.zeros_like(srtt),where=valid);coverage=[]
for actor in sorted(set(data['actor'].tolist())):
 for t in (20,40,80,120,160):
  mask=(data['actor']==actor)&(data['period_ms']==t);coverage.append(dict(actor=int(actor),period_ms=t,rows=int(mask.sum()),high_SRTT_rows=int(np.count_nonzero(mask&valid&(ratio>=2))),invalid_min_rtt=int(np.count_nonzero(mask&~valid))))
behavior=[]
if 'behavior' in data.files:
 for b in sorted(set(data['behavior'].tolist())):
  mask=(data['period_ms']==20)&(data['behavior']==b);behavior.append(dict(behavior=str(b),A20_rows=int(mask.sum()),A20_high_SRTT_rows=int(np.count_nonzero(mask&valid&(ratio>=2)))))
full=sum([n['full_training_coverage'] for n in node_results],[]);assert sum(v['rows'] for v in full)==711285;support=dict(full_training=full,retained_replay=coverage,retained_behavior=behavior,retained_rows=len(rs),replay_sha256_before=replay_before,replay_sha256_after=sha(mp));assert support['replay_sha256_before']==support['replay_sha256_after'];save('training-support.json',support)
modelproof=dict(variables_before=before,variables_after=after,model_variables_unchanged=before==after,Canopy_optimizer_operations=0,model_files_unchanged=all(sha(pathlib.Path(prefix+k))==v['sha256'] for k,v in reg['model_files'].items()),new_TCP=0,D_Q='NOT_RUN');assert modelproof['model_files_unchanged'];save('model-proof.json',modelproof)
# Independent supervised readouts. Hold out complete TCP runs or conditions.
condition=np.asarray([meta[int(i)]['job']['condition'] for i in rid]);repeat=np.asarray([meta[int(i)]['job']['repeat'] for i in rid]);w=dt.copy()
for i in set(rid.tolist()):mask=rid==i;w[mask]/=dt[mask].sum()
folds=[('repeat0_to1',repeat==0,repeat==1),('repeat1_to0',repeat==1,repeat==0)]+[('leave_condition_'+c,condition!=c,condition==c) for c in sorted(set(condition.tolist()))];probes=[]
def weighted(a,weight):return np.average(a,weights=weight,axis=0)
def scores(truth,pred,weight,baseline):
 mu=weighted(truth,weight);variance=weighted((truth-mu)**2,weight);mse=weighted((truth-pred)**2,weight);return dict(MAE_mbps=weighted(abs(truth-pred),weight).tolist(),RMSE_mbps=np.sqrt(mse).tolist(),R2=[float(1-e/v) if v>1e-12 else None for e,v in zip(mse,variance)],baseline_MAE_mbps=weighted(abs(truth-baseline),weight).tolist(),baseline_RMSE_mbps=np.sqrt(weighted((truth-baseline)**2,weight)).tolist())
for fold,train,test in folds:
 for features in reg['feature_sets']:
  cols=np.arange(149) if features=='full149' else np.r_[np.arange(72),np.arange(76,149)];xt=x[train][:,cols].astype(np.float64);xv=x[test][:,cols].astype(np.float64);wt=w[train]*train.sum()/w[train].sum();wv=w[test];yt=y[train];yv=y[test];mu=weighted(xt,wt);std=np.sqrt(weighted((xt-mu)**2,wt));std[std<1e-8]=1.;xt=(xt-mu)/std;xv=(xv-mu)/std;baseline=weighted(yt,wt)
  for kind in ['ridge','extra_trees']:
   if time.time()>reg['deadline_epoch']-600-60:save('probe-results.json',dict(status='PARTIAL_BUDGET',records=probes));raise RuntimeError('Finite probe budget reached; existing checks retained')
   model=Ridge(alpha=reg['probe_models']['ridge_alpha']) if kind=='ridge' else ExtraTreesRegressor(random_state=reg['seed'],**reg['probe_models']['extra_trees']);start=time.time();model.fit(xt,yt,sample_weight=wt);pred=model.predict(xv);row=dict(fold=fold,feature_set=features,model=kind,train_rows=int(train.sum()),test_rows=int(test.sum()),train_runs=len(set(rid[train].tolist())),test_runs=len(set(rid[test].tolist())),seconds=time.time()-start,**scores(yv,pred,wv,baseline));row['per_test_run']=[]
   for i in sorted(set(rid[test].tolist())):
    mask=rid[test]==i;row['per_test_run'].append(dict(run=meta[i]['job']['run'],condition=meta[i]['job']['condition'],mean_capacity_mbps=weighted(yv[mask],wv[mask]).tolist(),mean_prediction_mbps=weighted(pred[mask],wv[mask]).tolist(),MAE_mbps=weighted(abs(yv[mask]-pred[mask]),wv[mask]).tolist()))
   if kind=='extra_trees':ids=np.argsort(model.feature_importances_)[-15:][::-1];row['top_features']=[dict(index=int(cols[i]),importance=float(model.feature_importances_[i])) for i in ids]
   probes.append(row);save('probe-progress.json',dict(status='RUNNING',records=probes))
save('probe-results.json',dict(status='COMPLETED',records=probes,sklearn_version=sklearn.__version__,targets=reg['capacity_targets'],rows=len(x),run_count=48,condition_count=len(set(condition.tolist())),optimizer_operations_on_Canopy=0,small_probe_fit_only=True,ended_epoch=time.time()));print(json.dumps(dict(status='COMPLETED_ZERO_TCP_CHECKS',probe_records=len(probes),gradient_states=len(gradient_points)//3,training_rows=sum(v['rows'] for v in full),Canopy_unchanged=before==after)))
