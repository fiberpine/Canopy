"""Explicit joint72 prior and coverage-gated joint cwnd/timing optimization.

Same DDPG/SMDP backend, moving BN, action mapping and real-sample update ledger.
Fixed20 donor's constant planned-input/action coefficients are folded into biases;
their fixed20 function is preserved, unsupported period gradients start at zero.
"""
import numpy as np
import joint_mtp
PREFIX='abstract_actor/abstract_actor/'
def enabled(cfg):return cfg.get('window_policy_mode')=='staged_joint'

def install(module,cfg):
 if not enabled(cfg):return
 import tensorflow as tf
 cls=module.Agent
 from duration_head_v16 import mode
 cls._duration_nonlinearity=mode(cfg)
 if getattr(cls,'_trainable_timing_installed',False):return
 original_init=cls.__init__;original_target=cls.get_target_actor_policy
 def target(self):
  policy=original_target(self)
  self.timing_learning_enabled=tf.get_variable('timing_learning_enabled',shape=[],dtype=tf.bool,initializer=tf.constant_initializer(False),trainable=False)
  initial=joint_mtp.mtp_action(cfg['timing_init_ms'],cfg)
  time=tf.cond(self.timing_learning_enabled,lambda:policy[:,1:2],lambda:tf.ones_like(policy[:,1:2])*initial)
  if cfg.get('fixed_mtp_ms') is not None:time=tf.ones_like(time)*joint_mtp.mtp_action(cfg['fixed_mtp_ms'],cfg)
  return tf.concat([policy[:,:1],time],axis=1)
 def init(self,*args,**kw):
  original_init(self,*args,**kw)
  if cfg.get('time_statistics_v27'):self.time_statistics_schema=tf.get_variable('time_statistics_schema',shape=[],dtype=tf.int32,initializer=tf.constant_initializer(1),trainable=False)
  if cfg.get('interval_rate_v28')=='acked_bytes':self.interval_rate_schema=tf.get_variable('interval_rate_schema',shape=[],dtype=tf.int32,initializer=tf.constant_initializer(1),trainable=False)
  if cfg.get('feedback_context_v31'):
   self.feedback_context_schema=tf.get_variable('feedback_context_schema',shape=[],dtype=tf.int32,initializer=tf.constant_initializer(1),trainable=False)
   self.feedback_context_mode_schema=tf.get_variable('feedback_context_mode_schema',shape=[],dtype=tf.int32,initializer=tf.constant_initializer(int(cfg.get('feedback_context_mode')=='feedback')),trainable=False)
  if cfg.get('scheduled_observation_v30'):self.timer_boundary_schema=tf.get_variable('timer_boundary_schema',shape=[],dtype=tf.int32,initializer=tf.constant_initializer(1),trainable=False)
  if mode(cfg)=='softsign':self.duration_head_schema=tf.get_variable('duration_head_schema',shape=[],dtype=tf.int32,initializer=tf.constant_initializer(1),trainable=False)
  self._unmasked_time=self._moving_action[:,1:2]
  initial=joint_mtp.mtp_action(cfg['timing_init_ms'],cfg)
  time=tf.cond(self.timing_learning_enabled,lambda:self._unmasked_time,lambda:tf.ones_like(self._unmasked_time)*initial)
  if cfg.get('fixed_mtp_ms') is not None:time=tf.ones_like(time)*joint_mtp.mtp_action(cfg['fixed_mtp_ms'],cfg)
  self._moving_action=tf.concat([self._moving_action[:,:1],time],axis=1)
  self.enable_timing_learning=tf.assign(self.timing_learning_enabled,True)
 def action(self,state,use_noise=True):
  if use_noise:raise ValueError('Use explicitly recorded exploration outside the joint policy')
  from agent_v2 import create_input_op_shape
  value,stage=self.sess.run([self._moving_action,self.timing_learning_enabled],{self.s0:create_input_op_shape(state,self.s0),self.is_training:False})
  self.timing_stage_enabled=bool(stage)
  return [value],[value]
 cls.__init__=init;cls.get_target_actor_policy=target;cls.get_concrete_action=action;cls._trainable_timing_installed=True

def initialize(agent,source,cfg):
 import tensorflow as tf
 assert enabled(cfg) and cfg.get('fixed_mtp_ms') is None and cfg.get('gamma',.995)==.995
 reader=tf.train.NewCheckpointReader(source);variables={v.op.name:v for v in tf.global_variables()}
 slots=[v for n,v in variables.items() if '/Adam' in n or n.startswith(('beta1_power','beta2_power')) or n=='global_step'];before=agent.sess.run(slots)
 names=['fc1/w','fc1/b','fc2/w','fc2/b','fc1_1/w','fc1_1/b']+[bn+'/'+s for bn in ['bn1','bn2'] for s in ['moving_mean','moving_variance','beta']]
 actor={PREFIX+n:reader.get_tensor(PREFIX+n).copy() for n in names}
 assert actor[PREFIX+'fc1/w'].shape==(72,256)
 source_ms=float(cfg.get('source_fixed_mtp_ms',20))
 constant=joint_mtp.mtp_action(source_ms,cfg)
 actor[PREFIX+'fc1/b']+=constant*actor[PREFIX+'fc1/w'][71];actor[PREFIX+'fc1/w'][71]=0.
 from timing_context_v17 import state_dim
 extra=state_dim(cfg)-72
 if extra:actor[PREFIX+'fc1/w']=np.pad(actor[PREFIX+'fc1/w'],((0,extra),(0,0)),mode='constant')
 actor[PREFIX+'fc1_1/w'][:,1]=0.
 from duration_head_v16 import inverse
 actor[PREFIX+'fc1_1/b'][1]=inverse(joint_mtp.mtp_action(cfg['timing_init_ms'],cfg),cfg)
 values=dict(actor)
 for scope in ['critic','critic2']:
  q={scope+'/'+l+'/'+s:reader.get_tensor(scope+'/'+l+'/'+s).copy() for l in ['fc1','fc2','dense'] for s in ['kernel','bias']}
  for l,row in [('fc1',71),('fc2',257)]:q[scope+'/'+l+'/bias']+=constant*q[scope+'/'+l+'/kernel'][row];q[scope+'/'+l+'/kernel'][row]=0.
  if extra:q[scope+'/fc1/kernel']=np.pad(q[scope+'/fc1/kernel'],((0,extra),(0,0)),mode='constant')
  values.update(q)
 for n,x in values.items():assert tuple(variables[n].shape.as_list())==x.shape and np.all(np.isfinite(x)),n
 agent.sess.run([tf.assign(variables[n],x) for n,x in values.items()]);agent.init_target()
 for x,y in zip(before,agent.sess.run(slots)):np.testing.assert_array_equal(x,y)
 return dict(source_checkpoint=source,source_schema='joint72-fixed%d'%source_ms,source_fixed_mtp_ms=source_ms,time_head_nonlinearity=cfg.get('time_head_nonlinearity','tanh'),canonical_constant_planned_state=True,canonical_constant_time_action=True,initial_time_ms=cfg['timing_init_ms'],window_and_critics_from_trained_prior=True,window_trainable_after_calibration=True,optimizer_fresh=True,complete_resume=False,scratch_training=False)
