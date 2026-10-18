"""Exact 70-input scalar teacher, checkpoint-embedded and excluded from optimization.

Only the timing student is trainable. Actor and Bellman target use the same teacher
without window exploration or target-window smoothing. No fallback/window clamp.
"""
import hashlib
import numpy as np
from staged_timing import enabled, validate
import joint_mtp

PREFIX='abstract_actor/abstract_actor/'
SHAPES={'fc1/w':(70,256),'fc1/b':(256,),'fc2/w':(256,256),'fc2/b':(256,),
 'fc1_1/w':(256,1),'fc1_1/b':(1,)}
for b in ['bn1','bn2']:
 for n in ['moving_mean','moving_variance','beta']:SHAPES[b+'/'+n]=(1,256)

def shapes(cfg):
 result=dict(SHAPES)
 if cfg.get('frozen_teacher_schema','scalar70')=='joint72':
  result['fc1/w']=(72,256);result['fc1_1/w']=(256,2);result['fc1_1/b']=(2,)
 return result

def forward(state,cfg):
 import tensorflow as tf
 with tf.variable_scope('frozen_window',reuse=tf.AUTO_REUSE):
  values={name:tf.get_variable(name,shape=shape,dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer()) for name,shape in sorted(shapes(cfg).items())}
 def linear(x,name):return tf.matmul(x,values[name+'/w'])+values[name+'/b']
 def bn(x,name):return tf.nn.batch_normalization(x,values[name+'/moving_mean'],values[name+'/moving_variance'],values[name+'/beta'],None,.001)
 width=shapes(cfg)['fc1/w'][0]
 inputs=state[:,:width]
 context=cfg.get('window_teacher_planned_context_ms')
 if context is not None:
  if width!=72 or float(context)!=20.:
   raise ValueError('Explicit fixed20 prior context correction requires joint72 teacher')
  # Teacher was trained only with previous planned-period=-1. Preserve its exact
  # learned fixed20 function instead of extrapolating an unidentifiable constant
  # input. Real measured duration/history remain unchanged. Student and critic
  # still receive true72 state; environment timing/actions/reward are untouched.
  inputs=tf.concat([inputs[:,:71],tf.ones_like(inputs[:,71:72])*-1.],axis=1)
 h=tf.nn.leaky_relu(bn(linear(inputs,'fc1'),'bn1'))
 h=tf.nn.leaky_relu(bn(linear(h,'fc2'),'bn2'))
 return tf.stop_gradient(tf.tanh(linear(h,'fc1_1'))[:,:1])

def policy_variables(variables):
 return [v for v in variables if (v.op.name.startswith('abstract_actor/') and '/Adam' not in v.op.name) or v.op.name.startswith('frozen_window/') or v.op.name in ('timing_learning_enabled','duration_head_schema','time_statistics_schema','interval_rate_schema')]

def install(module,cfg):
 if cfg.get('window_policy_mode')!='frozen_scalar':return
 validate(cfg)
 import tensorflow as tf
 cls=module.Agent
 if getattr(cls,'_frozen_window_installed',False):
  if cls._frozen_window_configuration!=cfg:raise ValueError('One explicit frozen-window configuration per process')
  return
 original_init=cls.__init__;original_target=cls.get_target_actor_policy;original_train=cls.build_actor_train_op
 def target(self):
  smoothed=original_target(self)
  self.timing_learning_enabled=tf.get_variable('timing_learning_enabled',shape=[],dtype=tf.bool,initializer=tf.constant_initializer(False),trainable=False)
  initial=joint_mtp.mtp_action(cfg.get('timing_init_ms',120),cfg)
  time=tf.cond(self.timing_learning_enabled,lambda:smoothed[:,1:2],lambda:tf.ones_like(smoothed[:,1:2])*initial)
  if cfg['timing_policy_mode']=='fixed':time=tf.ones_like(time)*joint_mtp.mtp_action(cfg['fixed_mtp_ms'],cfg)
  return tf.concat([forward(self.s1,cfg),time],axis=1)
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  initial=joint_mtp.mtp_action(cfg.get('timing_init_ms',120),cfg)
  self._unmasked_time=self._moving_action[:,1:2]
  constant=tf.ones([tf.shape(self.s0)[0],1],tf.float32)
  if cfg['timing_policy_mode']=='fixed':time=constant*joint_mtp.mtp_action(cfg['fixed_mtp_ms'],cfg)
  elif cfg['timing_policy_mode']=='hysteresis':time=constant*initial
  else:time=tf.cond(self.timing_learning_enabled,lambda:self._unmasked_time,lambda:constant*initial)
  self._moving_action=tf.concat([forward(self.s0,cfg),time],axis=1)
  self.enable_timing_learning=tf.assign(self.timing_learning_enabled,True)
 def action(self,state,use_noise=True):
  if use_noise:raise ValueError('Use explicit timing-only exploration; window must stay frozen')
  from agent_v2 import create_input_op_shape
  value,stage=self.sess.run([self._moving_action,self.timing_learning_enabled],{self.s0:create_input_op_shape(state,self.s0),self.is_training:False})
  self.timing_stage_enabled=bool(stage)
  return [value],[value]
 def train(self):
  if cfg['timing_policy_mode']=='learned':return original_train(self)
  self.a_loss=-tf.reduce_mean(self.critic.build(self.s0,self._moving_action))
  return tf.no_op(name='frozen_window_no_actor_optimizer')
 cls.__init__=init;cls.get_target_actor_policy=target;cls.get_concrete_action=action
 cls.build_actor_train_op=train
 cls._frozen_window_installed=True;cls._frozen_window_configuration=dict(cfg)

def initialize(agent,source,cfg):
 """New timing experiment: pretrained window/hidden features, random critics/Adam."""
 import tensorflow as tf
 reader=tf.train.NewCheckpointReader(str(source));variables={v.op.name:v for v in tf.global_variables()};ops=[];h=hashlib.sha256()
 for name,shape in sorted(shapes(cfg).items()):
  value=reader.get_tensor(PREFIX+name)
  if tuple(value.shape)!=shape or not np.all(np.isfinite(value)):raise ValueError('Incompatible scalar teacher '+name)
  h.update(name.encode());h.update(value.tobytes());ops.append(tf.assign(variables['frozen_window/'+name],value))
  # Initialize the timing representation from teacher; no teacher critic/Adam is copied.
  student=value
  if name=='fc1/w' and value.shape[0]==70:student=np.concatenate([value,np.zeros((2,256),np.float32)])
  if name=='fc1_1/w':student=np.concatenate([value[:,:1],np.zeros((256,1),np.float32)],axis=1)
  if name=='fc1_1/b':student=np.array([value[0],np.arctanh(joint_mtp.mtp_action(cfg.get('timing_init_ms',120),cfg))],np.float32)
  ops.append(tf.assign(variables[PREFIX+name],student))
 agent.sess.run(ops);agent.init_target()
 critic_source=cfg.get('timing_warm_start_critics_checkpoint');copied=[]
 if critic_source:
  if cfg.get('frozen_teacher_schema')!='joint72' or cfg.get('gamma',.995)!=.995:
   raise ValueError('Timing value warm start requires explicit joint72/original discount scale')
  # This is a value prior from fixed20, NOT a calibrated cross-period Q function.
  # Leave optimizer and clocks fresh; real multi-period coverage gates actor learning.
  slots=[v for v in tf.global_variables() if '/Adam' in v.op.name or v.op.name.startswith(('beta1_power','beta2_power')) or v.op.name=='global_step']
  before=agent.sess.run(slots);qreader=tf.train.NewCheckpointReader(str(critic_source));qops=[]
  for scope in ['critic','critic2']:
   value_map={scope+'/'+layer+'/'+suffix:qreader.get_tensor(scope+'/'+layer+'/'+suffix).copy() for layer in ['fc1','fc2','dense'] for suffix in ['kernel','bias']}
   if cfg.get('timing_canonical_fixed20_value_prior'):
    # Both fields were exactly -1 throughout donor training. Fold their
    # contributions into the biases, preserving its Q on all donor transitions.
    # New period preferences must be learned from real coverage, not arbitrary
    # extrapolation of an unidentifiable constant-input coefficient.
    for layer,row in [('fc1',71),('fc2',257)]:
     key=scope+'/'+layer+'/kernel';bias=scope+'/'+layer+'/bias'
     value_map[bias]-=value_map[key][row];value_map[key][row]=0.
   for layer in ['fc1','fc2','dense']:
    for suffix in ['kernel','bias']:
     name=scope+'/'+layer+'/'+suffix;value=value_map[name]
     if tuple(variables[name].shape.as_list())!=value.shape or not np.all(np.isfinite(value)):
      raise ValueError('Incompatible warm value tensor '+name)
     qops.append(tf.assign(variables[name],value));copied.append(name)
  agent.sess.run(qops);agent.init_target()
  for old,new in zip(before,agent.sess.run(slots)):np.testing.assert_array_equal(old,new)
 return dict(source_checkpoint=str(source),teacher_schema=cfg.get('frozen_teacher_schema','scalar70'),teacher_tensor_sha256=h.hexdigest(),teacher_frozen=True,window_teacher_planned_context_ms=cfg.get('window_teacher_planned_context_ms'),new_random_critics_and_optimizer=not bool(copied),optimizer_not_restored=True,critic_prior_checkpoint=str(critic_source) if copied else None,critic_tensors=copied,canonical_fixed20_value_prior=bool(cfg.get('timing_canonical_fixed20_value_prior')),value_prior_only_requires_period_calibration=bool(copied),initial_time_ms=cfg.get('timing_init_ms',120),scratch_training=False)

def digest(agent):
 import tensorflow as tf
 variables=sorted([v for v in tf.global_variables() if v.op.name.startswith('frozen_window/')],key=lambda v:v.name);h=hashlib.sha256()
 for v,x in zip(variables,agent.sess.run(variables)):h.update(v.name.encode());h.update(x.tobytes())
 return h.hexdigest()
