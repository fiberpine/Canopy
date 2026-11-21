"""Optional, explicit normalization repair; stock behavior remains the default.
Target actions use moving statistics like deployed IBP point actions, and the
target's statistics follow the online actor with the same tau as target weights.
No reward, action mapping, architecture, learning rate or update schedule change.
"""
import tensorflow as tf,sonnet as snt
def install(module):
 if getattr(module,'_consistent_target_bn_installed',False):return
 original=module.AbstractActor._build
 def build(self,s,is_training):
  if self.name!='target_actor':return original(self,s,is_training)
  with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
   h=snt.Linear(self.h1_shape,use_bias=True,name='fc1')(s)
   h=snt.BatchNorm(name='bn1')(h,is_training=False,test_local_stats=False);h=tf.nn.leaky_relu(h)
   h=snt.Linear(self.h2_shape,use_bias=True,name='fc2')(h)
   h=snt.BatchNorm(name='bn2')(h,is_training=False,test_local_stats=False);h=tf.nn.leaky_relu(h)
   from duration_head_v16 import activate
   logits=snt.Linear(self.a_dim,use_bias=True,name='fc1')(h)
   return tf.multiply(activate(logits,getattr(module.Agent,'_duration_nonlinearity','tanh')),self.action_scale)
 module.AbstractActor._build=build
 module._consistent_target_bn_installed=True
def connect_statistics(agent,tau):
 variables={v.op.name:v for v in tf.global_variables()}
 targets=[v for name,v in variables.items() if name.startswith('target_actor/') and ('moving_mean' in name or 'moving_variance' in name)]
 assert len(targets)==4,'Unexpected target BN variable layout'
 for v in targets:
  suffix='/bn'+v.op.name.split('/bn',1)[1]
  matches=[source for name,source in variables.items() if name.startswith('abstract_actor/') and name.endswith(suffix)]
  assert len(matches)==1,('Unexpected online BN variable layout',v.op.name)
  source=matches[0]
  agent.target_act_init_op.append(tf.assign(v,source))
  agent.target_actor_update_op.append(tf.assign(v,tau*source+(1-tau)*v))
