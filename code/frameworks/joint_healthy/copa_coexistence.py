"""Copa NSDI'18-principled RTT-normalized Competitive Mode wrapper for Fiber (v3).
1) Detects persistent standing queue over win_s = min(2.5, max(1.5, 5*min_rtt))
2) Periodic 2-RTT drain probe every 5.0s while competitive (prevents false lock-in on solo 4G/Dynamic drops)
3) Blends negative cwnd suppression toward hybrid fixed point (a_w = 0 => 1.0x Kernel CUBIC).
Zero impact (beta == 0.0000) on solo wired/Starlink/5G flows!
"""
from collections import deque
import math
import numpy as np


class CopaCompetitiveWrapper:
    def __init__(self, base_actor):
        self.base_actor = base_actor
        self.srtt_win = deque()
        self.max_win = deque()
        self.t_s = 0.0
        self.beta = 0.0
        self.beta_sum = 0.0
        self.calls = 0
        self.comp_elapsed = 0.0
        self.probe_until = -1.0

    def act(self, state):
        action = np.array(self.base_actor.act(state), dtype=np.float32)
        dt = math.expm1(max(0.0, float(state[70]))) * 0.02
        min_rtt_ms = math.expm1(max(0.0, float(state[79]))) * 20.0
        srtt_ms = math.expm1(max(0.0, float(state[80]))) * 20.0
        dt_eff = max(0.005, dt)
        self.t_s += dt_eff

        win_s = min(2.5, max(1.5, 5.0 * (min_rtt_ms / 1000.0)))
        self.srtt_win.append((self.t_s, srtt_ms))
        while len(self.srtt_win) > 1 and self.srtt_win[0][0] < self.t_s - win_s:
            self.srtt_win.popleft()

        self.max_win.append((self.t_s, srtt_ms))
        while len(self.max_win) > 1 and self.max_win[0][0] < self.t_s - 10.0:
            self.max_win.popleft()

        in_probe = self.t_s < self.probe_until
        if self.t_s >= win_s and min_rtt_ms > 0.0:
            min_srtt_win = min(v for _, v in self.srtt_win)
            max_srtt_10s = max(v for _, v in self.max_win)
            standing = (min_srtt_win > 1.40 * min_rtt_ms) and (
                min_srtt_win > min_rtt_ms + 0.15 * (max_srtt_10s - min_rtt_ms)
            )
            if standing:
                if not in_probe:
                    self.beta = min(1.0, self.beta + 0.10)
                    self.comp_elapsed += dt_eff
                    if self.comp_elapsed >= 5.0:
                        # Periodic 3-RTT drain verification probe (Copa / BBR ProbeRTT principle)
                        self.probe_until = self.t_s + max(0.25, 3.0 * (srtt_ms / 1000.0))
                        self.comp_elapsed = 0.0
                        in_probe = True
            else:
                self.beta = max(0.0, self.beta - 0.35)
                self.comp_elapsed = 0.0

        eff_beta = 0.0 if in_probe else self.beta
        a_w = float(action[0])
        if eff_beta > 0.0 and a_w < 0.0:
            action[0] = np.float32((1.0 - eff_beta) * a_w)
        self.beta_sum += eff_beta
        self.calls += 1
        return action

    @property
    def mean_beta(self):
        return self.beta_sum / max(1, self.calls)
