"""Preserve Canopy zero-width IBP semantics with explicit moving-statistics forward.
No reward/action/target/critic semantics change; checkpoint variable names unchanged.
"""
def forward(agent):
 import tensorflow as tf
 variables={v.op.name:v for v in tf.global_variables()};prefix='abstract_actor/abstract_actor/'
 def linear(x,name):return tf.matmul(x,variables[prefix+name+'/w'])+variables[prefix+name+'/b']
 def bn(x,name):return tf.nn.batch_normalization(x,variables[prefix+name+'/moving_mean'],variables[prefix+name+'/moving_variance'],variables[prefix+name+'/beta'],None,.001)
 agent._moving_logits=linear(tf.nn.leaky_relu(bn(linear(tf.nn.leaky_relu(bn(linear(agent.s0,'fc1'),'bn1')),'fc2'),'bn2')),'fc1_1')
 return tf.nn.tanh(agent._moving_logits)
def install(module):
 import tensorflow as tf
 cls=module.Agent
 if getattr(cls,'_moving_statistics_backend_installed',False):return
 original_init=cls.__init__;original_action=cls.get_concrete_action;original_train=cls.build_actor_train_op
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  assert not self.use_original and not self.use_snt_model_wo_ibp
  before=list(tf.global_variables());self._moving_action=forward(self);assert before==list(tf.global_variables())
  if getattr(self,'fixed_mtp_action',None) is not None:
   self._moving_action=tf.concat([self._moving_action[:,:1],tf.ones_like(self._moving_action[:,1:2])*self.fixed_mtp_action],axis=1)
 def train(self):
  self.a_loss=-tf.reduce_mean(self.critic.build(self.s0,self._moving_action))
  return self.actor_optimizer.minimize(self.a_loss,var_list=self.symbolic_actor.train_var(),global_step=self.global_step)
 def action(self,state,use_noise=True):
  if use_noise:return original_action(self,state,use_noise)
  from agent_v2 import create_input_op_shape
  out=self.sess.run([self._moving_action],{self.s0:create_input_op_shape(state,self.s0),self.is_training:False})
  return out,out
 cls.__init__=init;cls.build_actor_train_op=train;cls.get_concrete_action=action;cls._moving_statistics_backend_installed=True
