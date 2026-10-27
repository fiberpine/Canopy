"""Read-only local TF1 checkpoints/replay audit on learner, zero optimizer/TCP."""
import pathlib,sys,os,json,csv,time
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
R=pathlib.Path(os.environ['CANOPY_L1_ROOT']);F=R/'framework';sys.path[:0]=[str(F),str(F/'code/runtime/_deps')]
import numpy as np,tensorflow as tf
cfg=json.loads((R/'registration.json').read_text())['training_config'];rid=cfg['run_id'];D=F/'checkpoints'/rid
steps=sorted(int(p.name[5:-6]) for p in D.glob('step-*.index'));assert steps==[0,28513],steps
read=lambda folder,n:tf.train.NewCheckpointReader(str(folder/('step-%d'%n)))
start=read(D,0);end=read(D,max(steps));smoked=F/'checkpoints'/(rid+'-smoke');sn=sorted(int(p.name[5:-6]) for p in smoked.glob('step-*.index'));sm0=read(smoked,0);sm1=read(smoked,sn[-1])
shapes=start.get_variable_to_shape_map();assert shapes==end.get_variable_to_shape_map()==sm0.get_variable_to_shape_map()==sm1.get_variable_to_shape_map()
smoke_changed=[n for n in shapes if not np.array_equal(sm0.get_tensor(n),sm1.get_tensor(n))]
initial_differences=[n for n in shapes if not np.array_equal(start.get_tensor(n),sm0.get_tensor(n))]
assert not smoke_changed and not initial_differences
assert all(np.isfinite(end.get_tensor(n)).all() for n in shapes)
from orca_backend import create
a=create(F,cfg,'learner');a.build_learn()
grad1=tf.gradients(a.critic_out,a.action)[0];grad2=tf.gradients(a.critic_out2,a.action)[0]
vs=tf.global_variables();assert {v.op.name:list(v.shape) for v in vs}==shapes
sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1));a.assign_sess(sess)
saver=tf.train.Saver(vs);saver.restore(sess,str(D/'step-0'))
replay=np.load(str(D/'replay.npz'));ix=np.concatenate([np.linspace(*[int(z) for z in [np.flatnonzero(replay['actor_ids']==i)[0],np.flatnonzero(replay['actor_ids']==i)[-1]]],16,dtype=int) for i in range(24)])
# Explicitly sample inside each actor support, not between interleaved actor indices.
ix=np.concatenate([np.flatnonzero(replay['actor_ids']==i)[np.linspace(0,len(np.flatnonzero(replay['actor_ids']==i))-1,16,dtype=int)] for i in range(24)])
S=replay['s0'][ix];AID=replay['actor_ids'][ix]
def outputs():return sess.run(a.actor_out,{a.s0:S,a.is_training:False}).reshape(-1)
def gradient(p):
 q,g,g2=sess.run([a.critic_out,grad1,grad2],{a.s0:S,a.action:p.reshape(-1,1),a.is_training:False})
 return q.reshape(-1),g.reshape(-1),g2.reshape(-1)
initial=outputs();qi,gi,g2i=gradient(initial)
saver.restore(sess,str(D/('step-%d'%max(steps))));final=outputs();qf,gf,g2f=gradient(final)
bn=[v for v in vs if v.op.name.startswith('actor/') and ('moving_mean' in v.op.name or 'moving_variance' in v.op.name)]
for v in bn:sess.run(tf.assign(v,start.get_tensor(v.op.name)))
final_weights_initial_BN=outputs()
saver.restore(sess,str(D/'step-0'))
for v in bn:sess.run(tf.assign(v,end.get_tensor(v.op.name)))
initial_weights_final_BN=outputs()
def stat(x):return dict(mean=float(np.mean(x)),min=float(np.min(x)),p50=float(np.median(x)),max=float(np.max(x)),positive_fraction=float(np.mean(x>0)),saturated_fraction=float(np.mean(abs(x)>=.95)))
rows=[]
for i in range(24):
 m=AID==i;rows.append(dict(actor=i,condition=cfg['conditions'][i]['condition'],initial_pi=stat(initial[m]),final_pi=stat(final[m]),
 final_weights_initial_BN=stat(final_weights_initial_BN[m]),initial_weights_final_BN=stat(initial_weights_final_BN[m]),
 initial_Q=stat(qi[m]),final_Q=stat(qf[m]),initial_dQ_da=stat(gi[m]),final_dQ_da=stat(gf[m]),final_dQ2_da=stat(g2f[m])))
stages=list(csv.DictReader((F/'logs'/rid/'optimizer_stages.csv').open()));first_cov=json.loads((F/'logs'/rid/'first-update-coverage.json').read_text())
trace=list(csv.DictReader((F/'logs'/rid/'training.csv').open()))
out=dict(status='PASS_READ_ONLY_AUDIT',new_TCP_runs=0,optimizer_updates_in_audit=0,variables=len(shapes),smoke_steps=sn,smoke_changed=smoke_changed,
 initial_smoke_formal_differences=initial_differences,all_final_tensors_finite=True,first_update_coverage=first_cov,
 first_optimizer_epoch=float(stages[0]['epoch']) if stages else None,last_optimizer_epoch=float(stages[-1]['epoch']) if stages else None,
 checkpoints=steps,query_states=len(ix),query_sampling='16 factual replay states per actor spanning that actor stored transitions; post hoc descriptive',
 overall=dict(initial_pi=stat(initial),final_pi=stat(final),final_weights_initial_BN=stat(final_weights_initial_BN),initial_weights_final_BN=stat(initial_weights_final_BN),final_dQ_da=stat(gf)),
 per_actor=rows,training_curve=trace,limitations='Hybrid BN/weight swaps are same-input model sensitivity; not TCP counterfactuals. Critic gradients are not validated network gains. One seed, early safety stop.')
(R/'audit/tf-postmortem.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));sess.close()
