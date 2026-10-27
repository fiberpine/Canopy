"""Opt-in exploration processes for the multiplicative single-action window.

Every executed action multiplies cwnd by 4**a, so the exploration part of
log(cwnd) is the SUM of all executed perturbations.

iid (upstream, default)
    a_t = clip(pi_t + e_t),  e_t ~ N(0, sigma^2) i.i.d.
    The noise part of log4(cwnd) is a random walk: std sigma*sqrt(t).  When
    nothing restores the window (receive-window/send-buffer limited flows, or
    the detection delay of a long RTT) it drifts without bound.

zero_sum_offset (this module)
    Draw a target offset o_t = rho*o_{t-1} + sqrt(1-rho^2)*sigma_o*z_t and
    execute a_t = clip(pi_t + o_t - C_{t-1}), C_t = C_{t-1} + (a_t - pi_t).
    Without clipping C_t == o_t: the ideal exploration log-impulse is stationary.
    This is not a bound on actual cwnd relative to a separately simulated policy:
    native CUBIC, integer quantization and state-dependent pi follow different paths.
    With rho=0 and match_step_variance=True, sigma_o = sigma/sqrt(2(1-rho)),
    so the per-step executed perturbation a_t - pi_t = o_t - o_{t-1} has the
    same stationary marginal variance sigma^2 as iid (first step/clipping excepted).
    The lag-one correlation is different by design. One normal is drawn per call from
    the same NumPy global stream as the upstream G_Noise, so a given actor
    seed sees the identical z sequence under both processes.

Architecture, reward, the 4**a window mapping, the deterministic policy and the
clip range are unchanged.  Reset at every new TCP connection.
"""
import math

import numpy as np


class ZeroSumOffset(object):
    def __init__(self, sigma=0.2, rho=0.0, match_step_variance=True, rng=None):
        if not 0.0 <= rho < 1.0:
            raise ValueError('rho must be in [0, 1)')
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.sigma_o = self.sigma / math.sqrt(2.0 * (1.0 - self.rho)) if match_step_variance else self.sigma
        self.rng = rng if rng is not None else np.random
        self.reset()

    def reset(self):
        self.offset = 0.0       # o_{t-1}
        self.applied = 0.0      # C_{t-1}: cumulative executed-minus-deterministic action
        self.calls = 0
        self.clipped = 0

    def __call__(self, deterministic):
        z = float(self.rng.randn())
        self.offset = self.rho * self.offset + math.sqrt(1.0 - self.rho ** 2) * self.sigma_o * z
        wanted = deterministic + self.offset - self.applied
        executed = float(np.clip(wanted, -1.0, 1.0))
        self.clipped += executed != wanted
        self.applied += executed - deterministic
        self.calls += 1
        return executed

    def state(self):
        return dict(process='zero_sum_offset', sigma=self.sigma, rho=self.rho, sigma_o=self.sigma_o,
                    offset=self.offset, applied=self.applied, calls=self.calls, clipped=self.clipped)


def build(cfg):
    """Return None for the upstream iid process, else a ZeroSumOffset instance."""
    mode = cfg.get('exploration_process', 'iid')
    if mode == 'iid':
        return None
    if mode != 'zero_sum_offset':
        raise ValueError('Unknown exploration_process ' + str(mode))
    return ZeroSumOffset(sigma=cfg.get('exploration_std', .2), rho=cfg.get('exploration_offset_rho', 0.0))
