"""Frozen same-schema factorial intervention, never a trained innovation policy."""
import hashlib,json,pathlib
import numpy as np

def compose(final,source,window_source,time_source,cfg):
 f=np.asarray(final,np.float32).reshape(2);s=np.asarray(source,np.float32).reshape(2)
 assert window_source in ('source','final') and time_source in ('80','final')
 assert np.isfinite(f).all() and np.isfinite(s).all()
 y=f.copy()
 if window_source=='source':y[0]=s[0]
 if time_source=='80':
  from joint_mtp import mtp_action
  y[1]=mtp_action(80,cfg)
 return y

class FrozenSource:
 def __init__(self,sess,checkpoint):
  import tensorflow as tf
  self.sess=sess;reader=tf.train.NewCheckpointReader(checkpoint);p='abstract_actor/abstract_actor/'
  assert reader.get_variable_to_shape_map()[p+'fc1/w'][0]==86
  assert int(reader.get_tensor('duration_head_schema'))==1
  meta=json.loads(pathlib.Path(checkpoint+'.counters.json').read_text())['cfg']
  assert meta.get('decision_observation_clock') and meta.get('reward_sampling_semantics')=='observed_interval_orca_density'
  names=[l+'/'+s for l in ['fc1','fc2','fc1_1'] for s in ['w','b']]+[l+'/'+s for l in ['bn1','bn2'] for s in ['moving_mean','moving_variance','beta']]
  self.sha=hashlib.sha256();values={}
  for n in names:
   value=reader.get_tensor(p+n);self.sha.update(n.encode());self.sha.update(value.tobytes());values[n]=tf.constant(value)
  self.sha=self.sha.hexdigest()
  with tf.name_scope('v25_frozen_source_no_variables'):
   self.x=tf.placeholder(tf.float32,[None,86])
   def linear(x,n):return tf.matmul(x,values[n+'/w'])+values[n+'/b']
   def bn(x,n):return tf.nn.batch_normalization(x,values[n+'/moving_mean'],values[n+'/moving_variance'],values[n+'/beta'],None,.001)
   x=tf.nn.leaky_relu(bn(linear(self.x,'fc1'),'bn1'));x=tf.nn.leaky_relu(bn(linear(x,'fc2'),'bn2'));z=linear(x,'fc1_1')
   self.y=tf.concat([tf.nn.tanh(z[:,:1]),tf.nn.softsign(z[:,1:2])],axis=1)
 def predict(self,x):return self.sess.run(self.y,{self.x:np.asarray(x,np.float32).reshape(-1,86)})

def wrap(act,sess,cfg,digest):
 if not cfg.get('diagnostic_factorial_v25'):return act,digest
 assert cfg.get('no_training') is True and not cfg.get('training_mode')
 assert cfg.get('decision_observation_clock') and cfg.get('joint_state_dim')==86
 assert cfg.get('timing_policy_mode')=='learned' and cfg.get('fixed_mtp_ms') is None
 ws=cfg['diagnostic_window_source'];ts=cfg['diagnostic_time_source']
 source=FrozenSource(sess,cfg['diagnostic_source_checkpoint'])
 def action(history):
  final=act(history);prior=source.predict(history)[0]
  y=compose(final,prior,ws,ts,cfg);action.no_noise=y.copy()
  return y
 # Each condition deliberately computes BOTH networks, including unselected heads.
 # Report diagnostic2forwards/decision; never claim CPU savings from this tool.
 def hash_all():return hashlib.sha256((digest()+source.sha).encode()).hexdigest()
 return action,hash_all
