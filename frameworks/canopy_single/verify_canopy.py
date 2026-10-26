"""Real TF graph, optimizer, interval, and upstream property parity checks."""
import os, pathlib, sys, json, hashlib, tempfile
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
F=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(F),str(F/'code/runtime/_deps'),str(F/'code/canopy/rl-module')]
import tensorflow as tf,numpy as np
import distributed,agent_v2,moving_stats_backend,consistent_target_bn,independent_critic_v34
from canopy_feedback import install_bounds
moving_stats_backend.install(agent_v2);consistent_target_bn.install(agent_v2)
independent_critic_v34.install(agent_v2,{'independent_critic_adam_v34':True})
tf.set_random_seed(3);np.random.seed(20261024)
a=agent_v2.Agent(70,1,256,256,gamma=.995,batch_size=512,lr_a=.0001,lr_c=.001,tau=.001,
 mem_size=1024,noise_type=3,noise_exp=50000,stddev=.2,PER=False,CDQ=True,LOSS_TYPE='MSE',use_original=False,use_snt_model_wo_ibp=False)
consistent_target_bn.connect_statistics(a,.001);install_bounds(a);a.build_learn()
s=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1));a.assign_sess(s)
s.run(tf.global_variables_initializer());a.init_target()
x=np.random.uniform(0,1,(512,70)).astype(np.float32);zero=np.zeros_like(x)
fd={a.s0:x,a.action:np.random.uniform(-1,1,(512,1)),a.reward:np.random.uniform(0,1,(512,1)),a.s1:x*.9,a.terminal:np.zeros((512,1)),a.is_training:True,a.is_concrete:True,a.s0_delta:zero}
actor_vars=a.symbolic_actor.train_var();before=s.run(actor_vars)
s.run(a.critic_train_op,fd);after=s.run(actor_vars);assert all(np.array_equal(v,w) for v,w in zip(before,after))
bn=[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'abstract_actor' in v.name and 'target' not in v.name]
s.run([a.actor_train_op,bn],fd);a.target_update();after=s.run(actor_vars)
assert any(not np.array_equal(v,w) for v,w in zip(before,after))
point=s.run(a._moving_action,{a.s0:x,a.is_training:False})
lo,hi=s.run([a.qc_lower,a.qc_upper],{a.qc_center:x,a.qc_radius:zero})
error=float(max(np.max(np.abs(point-lo)),np.max(np.abs(point-hi))))
assert error<2e-5,error
radius=np.full_like(x,.01);lo,hi=s.run([a.qc_lower,a.qc_upper],{a.qc_center:x,a.qc_radius:radius})
max_violation=0.
for k in range(20):
 query=x+np.random.uniform(-1,1,x.shape).astype(np.float32)*radius
 y=s.run(a._moving_action,{a.s0:query,a.is_training:False})
 max_violation=max(max_violation,float(np.max(lo-y)),float(np.max(y-hi)))
assert max_violation<2e-5,max_violation
losses=list(map(float,s.run([a.c_loss,a.c_loss2,a.a_loss],fd)));assert np.isfinite(losses).all()
# Preserve and expose upstream shallow-property construction, including its
# loss-center behavior; do not silently label it a full paper replication.
from symbolic_transitions import initialize_symbolic_spec_single_step_only_latency,update_symbolic_s_single_step,get_symbolic_actions
from constants import SymbolicComponent_bound
from constrained_reward import get_raw_and_constraint_reward,convert_reward_mode
spec=initialize_symbolic_spec_single_step_only_latency(.25,5,12)
symbolic=update_symbolic_s_single_step(x[0],spec,12)
class BoundAdapter:
 def get_symbolic_action(self,s,s_delta,use_noise=False):
  return a.sess.run([a.qc_lower,a.qc_upper],{a.qc_center:[s],a.qc_radius:[s_delta]})
_,union=get_symbolic_actions(symbolic,BoundAdapter(),12)
small=np.asarray(symbolic.small_c,np.float32);large=np.asarray(symbolic.large_c,np.float32)
lc,uc=s.run([a.qc_lower,a.qc_upper],{a.qc_center:np.concatenate([small,large]),a.qc_radius:np.asarray(symbolic.small_delta+symbolic.large_delta)})
batch_union=[float(lc[:5].min()),float(uc[:5].max()),float(lc[5:].min()),float(uc[5:].max())]
scalar_union=[float(getattr(union,n)) for n in ['small_l','small_u','large_l','large_u']]
parity_error=float(np.max(np.abs(np.array(batch_union)-scalar_union)));assert parity_error<2e-5,parity_error
values=get_raw_and_constraint_reward(x[0,-7:],.1,.1,.8,80,12,convert_reward_mode('raw-sym'),.25,5,25,.25,union,[x[0,-7:]],[0.],[80.],3)
assert abs(values[3]-(.75*.8+.25*values[2]))<1e-8
from canopy_feedback import Feedback
probe=Feedback(a,{'canopy_shallow_loss_centers':True},F/'audit')
probe.observe_action(x[0],float(point[0,0]))
sample=probe.reward(x[0],.1,float(point[0,0]),.8,x[1],np.array([0,0,0,0,0,80.]));probe.close()
import threading
concurrent_errors=[]
def updating():
 try:
  for _ in range(30):
   s.run(a.critic_train_op,fd);s.run([a.actor_train_op,bn],fd);a.target_update()
 except Exception as error:concurrent_errors.append(str(error))
t=threading.Thread(target=updating);t.start();concurrent_error=0.;interval_reversal=0.
for _ in range(100):
 xx=x[:32];pp,ll,hh=s.run([a.qc_point,a.qc_lower,a.qc_upper],{a.s0:xx,a.qc_center:xx,a.qc_radius:np.zeros_like(xx)})
 concurrent_error=max(concurrent_error,float(np.max(np.abs(pp-ll))),float(np.max(np.abs(pp-hh))))
 interval_reversal=max(interval_reversal,float(np.max(ll-hh)))
t.join(60);assert not t.is_alive() and not concurrent_errors,concurrent_errors
assert concurrent_error<2e-5 and interval_reversal<1e-6,(concurrent_error,interval_reversal)
assert np.isfinite(sample)
data={'status':'PASS_REAL_TF','state_dim':70,'action_dim':1,'mtp_ms':20,'critic_iterations':1,'critic_optimizer_operations':2,'actor_updates':1,'target_updates':1,'point_interval_max_error':error,'sampled_bound_max_violation':max_violation,'scalar_batch_bound_error':parity_error,'losses':losses,'qc_reward':float(values[2]),'combined_reward':float(values[3]),'upstream_loss_center':float(small[0,-5]),'intended_bad_loss_center':float(spec.small_c[0][-5]),'intended_good_loss_center':0.,'original_checkpoint_modified':False,'limitations':['Random-input interval containment is a numerical check, not a new formal proof.','Shallow loss centers explicitly corrected in adapter; not an unmodified official-artifact reproduction.']}
data.update(concurrent_point_bound_error=concurrent_error,concurrent_interval_reversal=interval_reversal,concurrent_optimizer_iterations=30,concurrent_queries=100,parameter_snapshot='Per-tensor copies shared by point and bounds; not a transactional optimizer-step snapshot')
(F/'audit/tf-validation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
