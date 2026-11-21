"""Optional local policy-observation stress. Never changes TCP, utility or dt."""
import numpy as np

class Observation:
 def __init__(self,cfg):
  self.amplitude=float(cfg.get('policy_inverse_rtt_noise',0.))
  if not 0<=self.amplitude<=.05:raise ValueError('Registered bounded inverseRTT noise only')
  # A private generator must not consume exploration or replay RNG streams.
  self.rng=np.random.RandomState(int(cfg.get('observation_noise_seed',cfg.get('actor_rng_seed',cfg.get('seed',3))))%2**32)
 def apply(self,frame):
  value=np.array(frame,dtype=np.float32,copy=True)
  if value.shape!=(7,) or not np.all(np.isfinite(value)):raise ValueError('Invalid observation')
  if self.amplitude:value[5]*=1.+self.rng.uniform(-self.amplitude,self.amplitude)
  return value
