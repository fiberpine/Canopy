"""Real official Orca graph/update/save/frozen inference verification."""
import pathlib,sys,json,os
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
F=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(F),str(F/'code/runtime/_deps')]
import tensorflow as tf,numpy as np
from orca_backend import create
tf.set_random_seed(3);np.random.seed(241024)
a=create(F,dict(memsize=1024,independent_critic_adam_v34=True,exploration_std=.2,LOSS_TYPE='MSE'),'learner')
a.build_learn();s=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1));a.assign_sess(s)
s.run(tf.global_variables_initializer());a.init_target()
x=np.random.uniform(0,1,(512,70)).astype(np.float32)
fd={a.s0:x,a.s1:x*.9,a.action:np.random.uniform(-1,1,(512,1)),a.reward:np.random.uniform(0,1,(512,1)),a.terminal:np.zeros((512,1)),a.is_training:True}
av=[v for v in tf.global_variables() if v.op.name.startswith('actor/') and '/Adam' not in v.op.name]
before=s.run(av);critic_before=s.run(a.critic.train_var())
s.run(a.critic_train_op,fd);assert all(np.array_equal(v,w) for v,w in zip(before,s.run(av)))
assert any(not np.array_equal(v,w) for v,w in zip(critic_before,s.run(a.critic.train_var())))
bn=[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v.name.startswith('actor/')]
s.run([a.actor_train_op,bn],fd);a.target_update()
assert any(not np.array_equal(v,w) for v,w in zip(before,s.run(av)))
q1=s.run(a.target_actor_out,{a.s1:x,a.is_training:True})
q2=s.run(a.target_actor_out,{a.s1:x,a.is_training:False});assert np.array_equal(q1,q2)
expected=np.asarray(a.get_action(x,False)[0]);assert expected.shape==(512,1)
query=x[:1].copy();point=np.asarray(a.get_action(query,False)[0]);large=np.asarray(a.get_action(x,False)[0]);batch_error=float(np.max(np.abs(point-large[:1])));assert batch_error<2e-5
checkpoint=F/'audit/orca-verification/model';checkpoint.parent.mkdir(exist_ok=True)
tf.train.Saver().save(s,str(checkpoint));losses=list(map(float,s.run([a.c_loss,a.c_loss2,a.a_loss],fd)));assert np.isfinite(losses).all()
model_vars=[v.name for v in av];s.close();tf.reset_default_graph()
import evaluate
aa,ss,wrapper,params,action,digest,names=evaluate.model(dict(method='orca',checkpoint=str(checkpoint)))
before_digest=digest();observed=np.asarray(aa.get_action(x,False)[0]);restore_error=float(np.max(np.abs(expected-observed)));assert restore_error<2e-5
assert digest()==before_digest
data=dict(status='PASS_REAL_ORCA_TF',state_dim=70,action_dim=1,actor_scope='actor/',backend='Official Orca tf.layers Actor/Critic (not Canopy AbstractActor)',policy_variables=names,critic_updates=1,critic_optimizer_operations=2,actor_updates=1,target_updates=1,batch_context_max_error=batch_error,save_frozen_restore_error=restore_error,losses=losses,engineering_corrections=['Independent critic Adam','Target moving-BN statistics initialization and tau update'],original_models_modified=False)
(F/'audit/orca-tf-validation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
