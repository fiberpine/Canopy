
import pathlib,json,sys,hashlib,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',CUDA_VISIBLE_DEVICES='')
R=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');assert (R/'evaluation/result.json').exists();F=R/'A/framework';sys.path.insert(0,str(F));sys.path.insert(0,str(F/'code/runtime/_deps'))
import numpy as np
from common_period import PERIODS,encoded_period
from common_period_tf import Model
selection=json.loads((R/'evaluation/selected-model.json').read_text());prefix=selection['checkpoint'];meta=json.loads(pathlib.Path(prefix+'.metadata.json').read_text());cfg=json.loads((R/'A/config.json').read_text());m=Model(cfg['source_checkpoint'],meta['K'],meta['n0'],reward_config=cfg['common_reward_config']);m.restore(prefix)
with m.graph.as_default():variables=m.tf.global_variables()
def digest():
 h=hashlib.sha256()
 for v,x in zip(variables,m.session.run(variables)):h.update(v.op.name.encode());h.update(np.asarray(x).tobytes())
 return h.hexdigest()
before=digest();data=np.load(prefix+'.replay.npz',allow_pickle=False);rng=np.random.RandomState(20261102);indexes=[]
for actor in range(24):
 for period in PERIODS:
  choices=np.flatnonzero((data['actor']==actor)&(data['period_ms']==period));assert len(choices)>=32;indexes.extend(rng.choice(choices,32,replace=False).tolist())
idx=np.asarray(indexes);states=data['s0'][idx].astype(np.float32);nextstates=data['s1'][idx].astype(np.float32);actions=data['action'][idx].astype(np.float32);raw=data['reward'][idx].astype(np.float32).reshape(-1,1);gamma=data['discount'][idx].astype(np.float32).reshape(-1,1);boot=data['bootstrap'][idx].astype(np.float32).reshape(-1,1);actor_ids=data['actor'][idx];period_ids=data['period_ms'][idx]
centered=raw-np.float32(meta['K'])*(1-gamma)
terminal=data['kind'][idx]=='terminated';centered[terminal]=raw[terminal]-np.float32(meta['K'])
current=m.session.run(m.q+[m.n],{m.s:states,m.action:actions});next_a=actions.copy();next_a[:,0]=m.session.run(m.mu,{m.s:nextstates,m.action:actions})[:,0];nextq=m.session.run(m.q+[m.n],{m.s:nextstates,m.action:next_a})
# This online deterministic Bellman residual is only a descriptive cross-check;
# A was trained using smoothed/min target values and behavior includes noise.
residual=[centered+gamma*boot*q-nextval for q,nextval in zip(nextq[:2],current[:2])];n_res=1+gamma*boot*nextq[2]-current[2]
periods=[];all_q=[];all_n=[];all_a=[]
for period in PERIODS:
 a=np.zeros((len(states),2),np.float32);a[:,1]=encoded_period(period);a[:,0]=m.session.run(m.mu,{m.s:states,m.action:a})[:,0];q1,q2,n=m.session.run(m.q+[m.n],{m.s:states,m.action:a});q=(q1+q2)*.5
 periods.append(dict(period_ms=period,policy_action_mean=float(a[:,0].mean()),saturation=float(np.mean(abs(a[:,0])>.99)),Q_mean=float(q.mean()),Q_std=float(q.std()),Q1_Q2_abs_difference=float(np.mean(abs(q1-q2))),N_mean=float(n.mean()),N_min=float(n.min()),N_max=float(n.max())))
 all_q.append(q.ravel());all_n.append(n.ravel());all_a.append(a[:,0])
qs,ns,acts=np.asarray(all_q).T,np.asarray(all_n).T,np.asarray(all_a).T
preferences=[]
for penalty in [0,.005,.01,.02,.05,.1]:
 choices=np.argmax(qs-penalty*ns,axis=1);preferences.append(dict(penalty=penalty,proportion_by_period={str(t):float(np.mean(choices==i)) for i,t in enumerate(PERIODS)}))
out=dict(status='READ_ONLY_DIAGNOSIS',sample_states=len(states),sampling='32 retained real rows per training actor and executed period, independent of observed outputs; not development network-policy evaluation',periods=periods,preferences=preferences,conditional_action_range_mean=float(np.mean(np.ptp(acts,axis=1))),deterministic_online_Bellman_residuals=[dict(mean=float(r.mean()),rms=float(np.sqrt(np.mean(r*r)))) for r in residual],N_online_Bellman_residual=dict(mean=float(n_res.mean()),rms=float(np.sqrt(np.mean(n_res*n_res)))),variables_unchanged=before==digest(),optimizer_operations_executed=0,new_TCP_connections=0,limitation='Off-policy retained states and one-step residuals cannot identify true value bias, switching accuracy or selector performance; no fit correction and no B0 evaluation updates performed')
assert out['variables_unchanged'];(R/'A/learner/offline-value-diagnosis.json').write_text(json.dumps(out,indent=2));m.session.close();print(json.dumps(out))
