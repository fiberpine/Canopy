"""Explicit trainable70->72 window transfer; critics/Adam/time head not restored."""
import numpy as np,hashlib
from frozen_window import PREFIX,SHAPES
def initialize(agent,source,cfg):
 import tensorflow as tf
 assert cfg.get('joint_control') and cfg.get('fixed_mtp_ms')==20
 assert cfg.get('window_policy_mode')!='frozen_scalar' and not cfg.get('cold_init_checkpoint')
 variables={v.op.name:v for v in tf.global_variables()};reader=tf.train.NewCheckpointReader(source)
 critic=[v for v in tf.global_variables() if 'critic' in v.op.name];before=agent.sess.run(critic);ops=[];h=hashlib.sha256()
 for name,shape in sorted(SHAPES.items()):
  value=reader.get_tensor(PREFIX+name)
  assert value.shape==shape and np.all(np.isfinite(value)),name
  h.update(name.encode());h.update(value.tobytes())
  if name=='fc1/w':value=np.concatenate([value,np.zeros((2,256),np.float32)])
  if name=='fc1_1/w':value=np.concatenate([value,np.zeros((256,1),np.float32)],axis=1)
  if name=='fc1_1/b':value=np.array([value[0],0.],np.float32)
  assert tuple(variables[PREFIX+name].shape.as_list())==value.shape
  ops.append(tf.assign(variables[PREFIX+name],value))
 agent.sess.run(ops);agent.init_target()
 for x,y in zip(before,agent.sess.run(critic)):np.testing.assert_array_equal(x,y)
 copied=[]
 if cfg.get('warm_start_fixed20_critics'):
  assert cfg.get('decision_cost',0)==0 and cfg.get('gamma',.995)==.995,'Only original fixed20 utility/discount scale is supported'
  slots=[v for v in tf.global_variables() if '/Adam' in v.op.name or v.op.name.startswith(('beta1_power','beta2_power')) or v.op.name=='global_step'];saved_slots=agent.sess.run(slots);critic_ops=[]
  for scope in ['critic','critic2']:
   for layer in ['fc1','fc2','dense']:
    for suffix in ['kernel','bias']:
     name=scope+'/'+layer+'/'+suffix;value=reader.get_tensor(name)
     if layer=='fc1' and suffix=='kernel':assert value.shape==(70,256);value=np.concatenate([value,np.zeros((2,256),np.float32)])
     if layer=='fc2' and suffix=='kernel':assert value.shape==(257,256);value=np.concatenate([value,np.zeros((1,256),np.float32)])
     assert tuple(variables[name].shape.as_list())==value.shape,name
     critic_ops.append(tf.assign(variables[name],value));copied.append(name)
  agent.sess.run(critic_ops);agent.init_target()
  for x,y in zip(saved_slots,agent.sess.run(slots)):np.testing.assert_array_equal(x,y)
 return dict(source_checkpoint=source,source_tensor_sha256=h.hexdigest(),source_schema='scalar70',destination_schema='joint72-fixed20',window_trainable=True,extra_input_weights_initial_zero=True,time_head_initial_zero=True,critics_restored=bool(copied),critic_tensors=copied,optimizer_not_restored=True,scratch_training=False,complete_resume=False)
