"""Boundary-only action attribution and honest RTT freshness bounds.

The old96 prefix is preserved. New fields never claim exact RTT sample time:
when cnt>0 its last sample is somewhere inside the observed interval, hence
age in [0, dt]. Empty intervals increase both bounds by measured dt.
No additional socket reads, clock ticks, future trace or capacity inputs.
"""
import math
import numpy as np

EXTRA_DIM=53
def enabled(cfg):return bool(cfg.get('feedback_context_v31'))
def mode(cfg):
 m=cfg.get('feedback_context_mode','feedback')
 if m not in ('feedback','masked_control'):raise ValueError('Explicit feedback-context mode required')
 return m
def validate(cfg):
 if not enabled(cfg):return
 assert cfg.get('scheduled_observation_v30') and cfg.get('time_statistics_v27') and cfg.get('interval_rate_v28')=='acked_bytes'
 assert cfg.get('time_statistics_mode')=='rates' and cfg.get('joint_state_dim')==149
 mode(cfg)

class Feedback:
 def __init__(self,cfg):
  self.enabled=enabled(cfg);self.mode=mode(cfg);self.rows=np.zeros((10,5),np.float32);self.current=np.zeros(3,np.float32);self.lower=self.upper=0.;self.initialized=False
 def observe(self,raw,action=None):
  if not self.enabled:return
  r=np.asarray(raw,dtype=float);assert r.shape==(15,) and np.isfinite(r).all()
  dt,cnt,cwnd=float(r[3]),float(r[2]),float(r[5]);assert dt>0 and cnt>=0 and cwnd>0
  fresh=cnt>0
  if not self.initialized and not fresh:raise ValueError('RTT bootstrap must be real and fresh')
  if fresh:self.lower,self.upper=0.,dt
  else:self.lower+=dt;self.upper+=dt
  executed=0.;valid=action is not None
  if valid:
   a=np.asarray(action);assert a.shape==(2,) and np.isfinite(a).all() and abs(float(a[0]))<=1.00001
   percent=int(100*4**float(np.clip(a[0],-1,1)))
   executed=math.log(percent/100.)/math.log(4.)
  row=[executed,float(valid),float(fresh),math.log1p(self.lower/.02),math.log1p(self.upper/.02)]
  self.rows=np.concatenate((self.rows[1:],np.asarray([row],np.float32)),axis=0)
  self.current=np.asarray([math.log1p(max(0.,r[10])/cwnd),math.log1p(max(0.,r[11])/cwnd),min(8.,math.log1p(max(0.,r[9])/cwnd))],np.float32)
  self.initialized=True;self.raw=r.copy();self.last_row=np.asarray(row,np.float32)
 def features(self):
  if not self.enabled:return np.zeros(0,np.float32)
  assert self.initialized
  # Column-major history:10 executed actions,10 action-valid,10 fresh RTT,
  #10 lower age bounds,10 upper age bounds; then3 actual TCP snapshot ratios.
  y=np.concatenate((self.rows.T.reshape(-1),self.current)).astype(np.float32)
  assert y.shape==(EXTRA_DIM,) and np.isfinite(y).all()
  return y if self.mode=='feedback' else np.zeros_like(y)
 def record(self):return self.last_row.tolist()+self.current.tolist()+[float(self.raw[2]),float(self.raw[5])]
 def snapshot(self):return dict(mode=self.mode,rows=self.rows.tolist(),current=self.current.tolist(),age_lower_s=self.lower,age_upper_s=self.upper,raw=self.raw.tolist(),features=self.features().tolist(),age_definition='Bounds from RTT sample count over the actual last interval; not exact sample age',source='Only real scheduled observation boundaries')

def initialize(agent,cfg):
 """Explicit96->149 zero extension with fresh optimizer/replay/counters."""
 import tensorflow as tf,json,pathlib
 validate(cfg);source=cfg['matching_clock_prior_checkpoint'];rd=tf.train.NewCheckpointReader(source);sh=rd.get_variable_to_shape_map();meta=json.loads(pathlib.Path(source+'.counters.json').read_text())['cfg']
 assert sh['abstract_actor/abstract_actor/fc1/w'][0]==96
 assert int(rd.get_tensor('interval_rate_schema'))==1 and int(rd.get_tensor('time_statistics_schema'))==1
 assert meta.get('decision_observation_clock') and meta.get('interval_rate_v28')=='acked_bytes' and meta.get('time_statistics_mode')=='rates'
 assert meta['gamma']==cfg['gamma'] and meta['decision_cost']==cfg['decision_cost'] and meta.get('utility_normalization','observed_peak')==cfg.get('utility_normalization','observed_peak')
 v={x.op.name:x for x in tf.global_variables()}
 markers={'global_step','timing_learning_enabled','time_statistics_schema','interval_rate_schema','actor_gradient_schema','timer_boundary_schema','feedback_context_schema','feedback_context_mode_schema'}
 names=[n for n in v if '/Adam' not in n and not n.startswith(('beta1_power','beta2_power')) and n not in markers]
 slots=[x for n,x in v.items() if n not in names];before=agent.sess.run(slots);values={}
 for n in names:
  x=rd.get_tensor(n);shape=tuple(v[n].shape.as_list())
  if x.ndim==2 and x.shape==(96,256) and shape==(149,256):
   assert n.endswith(('fc1/w','fc1/kernel'));x=np.pad(x,((0,EXTRA_DIM),(0,0)),mode='constant')
  assert x.shape==shape,(n,x.shape,shape);values[n]=x
 agent.sess.run([tf.assign(v[n],x) for n,x in values.items()])
 for n,x in values.items():np.testing.assert_array_equal(agent.sess.run(v[n]),x)
 for x,y in zip(before,agent.sess.run(slots)):np.testing.assert_array_equal(x,y)
 return dict(source_checkpoint=source,source_state_dim=96,new_state_dim=149,old96_prefix_preserved=True,extra_weights_zero=True,window_critic_bn_targets_copied=True,optimizer_replay_counters_fresh=True,complete_resume=False,scratch_training=False,timer_interface_migrated_explicitly=True,remaining_limits=['old raw RTT count remains in96 prefix','observed peak normalization unchanged','nonlinear interval reward approximation unchanged'])
