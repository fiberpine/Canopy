"""Explicit handover state machine; no TF, sockets, sampling or training here."""
import numpy as np


class Handover:
    def __init__(self, arm, warmup_s, capacity_bytes_s=0.):
        assert arm in ('A0', 'A1', 'A2', 'A3', 'A4')
        assert warmup_s >= 0
        assert arm != 'A4' or capacity_bytes_s > 0
        self.arm, self.warmup_s = arm, warmup_s
        self.capacity_bytes_s = capacity_bytes_s
        self.first_time = None
        self.released = arm in ('A0', 'A4')
        self.reset_history = False
        self.event = None

    def boundary(self, now, env):
        if self.first_time is None:
            self.first_time = now
        if self.arm == 'A4':
            env.max_bw = max(env.max_bw, self.capacity_bytes_s)
        if not self.released and now-self.first_time >= self.warmup_s:
            self.event = dict(elapsed_s=now-self.first_time, max_bw_before=env.max_bw,
                              reset_max_bw=self.arm == 'A2',
                              reset_history=self.arm in ('A2', 'A3'))
            if self.arm == 'A2':
                env.max_bw = 0.
            self.reset_history = self.arm in ('A2', 'A3')
            self.released = True

    def history(self, prior, state):
        if prior is None or self.reset_history:
            result = np.zeros(70, dtype=np.float32)
            result[-7:] = state
            self.reset_history = False
            return result
        return np.concatenate((prior[7:], state)).astype(np.float32)

    def action(self, prediction):
        return float(prediction) if self.released else 0.
