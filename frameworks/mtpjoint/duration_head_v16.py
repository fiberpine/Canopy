"""Explicit time-head nonlinearity and checkpoint schema; no physical-range change."""
import math
def mode(cfg):
 value=cfg.get('time_head_nonlinearity','tanh')
 if value not in ('tanh','softsign'):raise ValueError('Explicit supported time head required')
 return value
def inverse(value,cfg):
 if not -1<float(value)<1:raise ValueError('Time initialization must be interior')
 return float(value)/(1-abs(float(value))) if mode(cfg)=='softsign' else math.atanh(float(value))
def activate(logits,kind='tanh'):
 import tensorflow as tf
 if kind=='tanh':return tf.nn.tanh(logits)
 if kind!='softsign' or logits.shape.as_list()[-1]!=2:raise ValueError('Softsign is only for joint2 time column')
 return tf.concat([tf.nn.tanh(logits[:,:1]),tf.nn.softsign(logits[:,1:2])],axis=1)
def check_checkpoint(cfg):
 import tensorflow as tf
 reader=tf.train.NewCheckpointReader(cfg['checkpoint']);names=reader.get_variable_to_shape_map()
 actual=int(reader.get_tensor('duration_head_schema')) if 'duration_head_schema' in names else 0
 expected=int(mode(cfg)=='softsign')
 if actual!=expected:raise ValueError('Checkpoint time-head schema mismatch:stored%d configured%d'%(actual,expected))
 from timing_context_v17 import state_dim
 actual_dim=names['abstract_actor/abstract_actor/fc1/w'][0]
 if actual_dim!=state_dim(cfg):raise ValueError('Checkpoint state dimension mismatch:stored%d configured%d'%(actual_dim,state_dim(cfg)))

 # State96 cannot be mistaken for a merely padded old policy.
 actual_statistics=int(reader.get_tensor('time_statistics_schema')) if 'time_statistics_schema' in names else 0
 if actual_statistics!=int(bool(cfg.get('time_statistics_v27'))):raise ValueError('Checkpoint physical-time statistics schema mismatch')
 actual_rate=int(reader.get_tensor('interval_rate_schema')) if 'interval_rate_schema' in names else 0
 if actual_rate!=int(cfg.get('interval_rate_v28')=='acked_bytes'):raise ValueError('Checkpoint native interval-rate measurement mismatch')
 actual_feedback=int(reader.get_tensor('feedback_context_schema')) if 'feedback_context_schema' in names else 0
 if actual_feedback!=int(bool(cfg.get('feedback_context_v31'))):raise ValueError('Checkpoint feedback-context schema mismatch')
 if actual_feedback:
  actual_mode=int(reader.get_tensor('feedback_context_mode_schema'))
  if actual_mode!=int(cfg.get('feedback_context_mode')=='feedback'):raise ValueError('Checkpoint feedback-context feature mode mismatch')
 actual_boundary=int(reader.get_tensor('timer_boundary_schema')) if 'timer_boundary_schema' in names else 0
 if actual_boundary!=int(bool(cfg.get('scheduled_observation_v30'))):raise ValueError('Checkpoint timer-boundary observation semantics mismatch')
