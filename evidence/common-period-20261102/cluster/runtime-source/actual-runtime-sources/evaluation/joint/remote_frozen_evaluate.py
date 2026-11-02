"""Frozen RPC inference diagnosis only. Same saved policy, no replay/optimization/noise."""
import pathlib,json,sys,os,hashlib
R=pathlib.Path(__file__).resolve().parent
import evaluate
from distributed import setup
cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
def model(c):
 import numpy as np,tensorflow as tf
 from envwrapper_v2 import TCP_Env_Wrapper
 settings=dict(seed=3,warmup=10240,reward_mode='raw',learner_endpoint=c['inference_endpoint'],direct_actor_session=True,num_actors=1,tf_threads=4,memsize=1,exploration_std=.2)
 a,sess,*unused=setup(settings,'actor',0)
 variables=[v for v in tf.global_variables() if v.name.startswith('abstract_actor/')]
 reader=tf.train.NewCheckpointReader(c['checkpoint'])
 for v,value in zip(variables,sess.run(variables)):assert np.array_equal(value,reader.get_tensor(v.op.name)),v.name
 def digest():
  h=hashlib.sha256()
  for v,value in zip(variables,sess.run(variables)):h.update(v.name.encode());h.update(value.tobytes())
  return h.hexdigest()
 def action(hist):return float(np.asarray(a.get_concrete_action(hist,False)[0]).reshape(-1)[0])
 params=json.loads((R/'code/canopy/params_base_eval.json').read_text())
 return a,sess,TCP_Env_Wrapper,params,action,digest,[v.name for v in variables]
evaluate.model=model;evaluate.distributed_session=True
evaluate.main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
