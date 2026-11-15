"""Read-only TF1 target wiring check on the actual saved full checkpoint."""
import os
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
R=Path(__file__).resolve().parents[2]
H=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from frameworks.joint_healthy.model import Model
from frameworks.joint_healthy.core import interval_return
cfg=json.loads((R/'models/joint-g4-cost0-seed3-20261115/evaluation.json').read_text().replace('@PACKAGE@',str(R).replace('\\','/')))
if len(sys.argv)!=3:raise SystemExit('Usage: TF1python reproduce_tf_target.py FULL_CHECKPOINT_PREFIX NEW_OUTPUT_JSON')
prefix=Path(sys.argv[1])
model=Model(cfg,training=True);rng=np.random.RandomState(20261116)
states=rng.uniform(0.,1.,(64,82)).astype(np.float32)
actions=rng.uniform(-1.,1.,(64,2)).astype(np.float32)
durations=np.tile(np.array([.02,.04,.08,.16],np.float64),16)[:,None]
utility=rng.uniform(0.,1.,(64,1))
values=[interval_return(float(u),float(t)) for u,t in zip(utility[:,0],durations[:,0])]
rewards=np.array([x[0] for x in values],np.float32)[:,None];discounts=np.array([x[1] for x in values],np.float32)[:,None]
physical=np.tile(np.array([0.,1.],np.float32),32)[:,None]
batch=[states,actions,rewards,states*.9,physical,durations,discounts]
def all_digest(sess):
 h=hashlib.sha256()
 for value in sess.run(tf.global_variables()):
  if hasattr(value,'tobytes'):h.update(value.tobytes())
  else:h.update(str(value).encode())
 return h.hexdigest()
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
 sess.run(tf.global_variables_initializer());model.restore(sess,str(prefix));before=all_digest(sess)
 feed=model.feed(batch);assert np.all(feed[model.terminal]==0)
 target,tail=sess.run([model.label,model.target_q],feed)
 expected=rewards+discounts*tail
 error=float(np.max(np.abs(target-expected)));assert error<1e-6
 # One graph execution yields both labels/tail so smoothing randomness is shared.
 c=.02;feed2=dict(feed);feed2[model.reward]=rewards-c
 target2,tail2=sess.run([model.label,model.target_q],feed2)
 cost_error=float(np.max(np.abs(target2-((rewards-c)+discounts*tail2))));assert cost_error<1e-6
 after=all_digest(sess);assert before==after
result=dict(status='PASS_ACTUAL_TF1_CHECKPOINT_SMDP_TARGET_WIRING',rows=64,
 target_max_absolute_error=error,once_per_decision_cost_target_max_error=cost_error,
 physical_boundary_rows=int(physical.sum()),bootstrap_terminal_nonzero=int(np.count_nonzero(feed[model.terminal])),
 all_saved_variables_unchanged=before==after,states='Synthetic fixed inputs, not real TCP trajectories',
 tcp_runs=0,ssh_connections=0,optimizer_updates=0,
 limitation='This checks graph wiring only; does not establish target value accuracy or duration-action ranking')
output=Path(sys.argv[2]);output.parent.mkdir(parents=True,exist_ok=True)
with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
