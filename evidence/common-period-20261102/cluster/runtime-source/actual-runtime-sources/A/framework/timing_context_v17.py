"""Local measured timescale and physical-time EWMA; no trace/capacity oracle."""
import math,numpy as np
NAMES=['log1p_min_rtt_over20ms','log1p_observed_srtt_over20ms','throughput_log_ewma2s_std','throughput_log_minus_ewma2s']
def enabled(cfg):return cfg.get('timing_context_features',False)
def state_dim(cfg):
 n=(96 if cfg.get('time_statistics_v27') else 86) if cfg.get('decision_observation_clock') else (76 if enabled(cfg) else 72)
 if cfg.get('feedback_context_v31'):n+=53
 if cfg.get('joint_state_dim',n)!=n:raise ValueError('Explicit joint state schema mismatch')
 return n
class Context:
 def __init__(self,cfg):
  self.enabled=enabled(cfg);self.tau=float(cfg.get('timing_context_tau_s',2.));self.initialized=False;self.rate_initialized=False;self.mu=self.second=0.;self.value=np.zeros(4,np.float32)
  if not math.isfinite(self.tau) or not .2<=self.tau<=10:raise ValueError('Bounded measured-time context horizon required')
 def update(self,raw,inverse_rtt_multiplier=1.):
  if not self.enabled:return
  raw=np.asarray(raw);tp=float(raw[1]);dt=float(raw[3]);srtt=float(raw[8]);minimum=float(raw[14]);m=float(inverse_rtt_multiplier)
  if not all(math.isfinite(x) for x in [tp,dt,srtt,minimum,m]) or tp<0 or dt<=0 or srtt<=0 or minimum<=0 or m<=0:raise ValueError('Invalid actual local context telemetry')
  x=math.log1p(tp)
  self.last=dict(throughput=tp,delta_t=dt,srtt_ms=srtt,min_rtt_ms=minimum,inverse_rtt_multiplier=m)
  self.initialized=True
  if not self.rate_initialized and tp>0:self.mu=x;self.second=x*x;self.rate_initialized=True
  elif self.rate_initialized:
   decay=math.exp(-dt/self.tau);self.mu=decay*self.mu+(1-decay)*x;self.second=decay*self.second+(1-decay)*x*x
  # Same inverseRTT observation perturbation also applies to observed absolute
  # srtt. Additional features must not secretly undo the registered noise stress.
  self.value=np.array([math.log1p(minimum/20.),math.log1p(srtt/m/20.),min(4.,math.sqrt(max(0.,self.second-self.mu*self.mu))) if self.rate_initialized else 0.,max(-4.,min(4.,x-self.mu)) if self.rate_initialized else 0.],np.float32)
 def features(self):
  if self.enabled and not self.initialized:raise ValueError('Context requires an actual initial feedback')
  return self.value.copy() if self.enabled else np.zeros(0,np.float32)
 def snapshot(self):return dict(mu=self.mu,second=self.second,rate_initialized=self.rate_initialized,features=self.features().tolist(),last=self.last,tau_s=self.tau)
