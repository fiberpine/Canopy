"""Pure scheduling and calibration rules; no extra network state or oracle capacity."""
import math
import numpy as np
from joint_mtp import mtp_action, bounds

MODES=('fixed','hysteresis','learned')

def enabled(cfg):return cfg.get('window_policy_mode') in ('frozen_scalar','staged_joint')

def validate(cfg):
 if not enabled(cfg):return
 if not cfg.get('joint_control') or cfg.get('backend')!='moving_stats' or not cfg.get('target_bn_consistent'):
  raise ValueError('Frozen scalar requires joint/moving_stats/consistent target BN')
 if cfg.get('timing_policy_mode') not in MODES:raise ValueError('Explicit timing policy mode required')
 if cfg.get('frozen_teacher_schema','scalar70') not in ('scalar70','joint72'):raise ValueError('Explicit supported frozen teacher schema required')
 if cfg.get('fixed_action') is not None and not cfg.get('capacity_only'):raise ValueError('Do not override frozen window during training')
 lo,hi=bounds(cfg)
 if not lo<float(cfg.get('timing_init_ms',120))<hi:raise ValueError('Initialize timing strictly inside bounds')
 if cfg['timing_policy_mode']=='fixed' and cfg.get('fixed_mtp_ms') is None:raise ValueError('Fixed mode requires fixed_mtp_ms')
 if cfg['timing_policy_mode']!='fixed' and cfg.get('fixed_mtp_ms') is not None:raise ValueError('Do not mask a learned/rule head with fixed_mtp_ms')
 if cfg['timing_policy_mode']=='learned':
  periods=cfg.get('timing_calibration_periods_ms',[20,40,80,120,160])
  if len(set(periods))<3 or any(not lo<=float(x)<=hi for x in periods):raise ValueError('At least three distinct supported calibration periods')
  if cfg.get('timing_critic_calibration_updates',5000)<1 or cfg.get('timing_min_retained_per_period',128)<1:raise ValueError('Explicit positive critic/coverage calibration gates required')

def period_bin(action,cfg):
 periods=cfg.get('timing_calibration_periods_ms',[20,40,80,120,160])
 normalized=np.array([mtp_action(float(x),cfg) for x in periods])
 i=int(np.argmin(np.abs(normalized-float(action))))
 return i if abs(normalized[i]-float(action))<1e-5 else -1

def calibration_ready(updates,retained,cfg):
 return int(updates)>=cfg.get('timing_critic_calibration_updates',5000) and np.asarray(retained).min()>=cfg.get('timing_min_retained_per_period',128)

class Scheduler:
 """Per-actor state, reset only hysteresis on TCP reset; coverage cursor survives."""
 def __init__(self,cfg,index=0):
  self.cfg=cfg;self.cursor=index;self.reset_episode()
 def reset_episode(self):self.period=float(self.cfg.get('timing_init_ms',120));self.stable=0
 def choose(self,state,policy_time,learning_enabled,training):
  mode=self.cfg['timing_policy_mode']
  if mode=='fixed':return mtp_action(self.cfg['fixed_mtp_ms'],self.cfg),'fixed'
  if mode=='learned':
   if training and not learning_enabled:
    p=self.cfg.get('timing_calibration_periods_ms',[20,40,80,120,160]);ms=p[self.cursor%len(p)];self.cursor+=1
    return mtp_action(ms,self.cfg),'critic-calibration'
   return float(policy_time),'learned' if learning_enabled else 'initial-policy'
  s=np.asarray(state,dtype=float)
  if s.shape!=(72,) or not np.all(np.isfinite(s)):raise ValueError('Invalid scheduler state')
  frame=s[63:70];lo,hi=bounds(self.cfg)
  # Existing Orca features: observed_peak-normalized throughput/loss and minRTT/sRTT.
  # These are local observations, not true utilization or trace capacity.
  pressure=frame[2]>.01 or frame[5]<.8 or frame[0]<.9
  if pressure:self.period=max(lo,self.period/2);self.stable=0
  else:
   self.stable+=1
   if self.stable>=3:self.period=min(hi,self.period*1.5);self.stable=0
  return mtp_action(self.period,self.cfg),'hysteresis'

