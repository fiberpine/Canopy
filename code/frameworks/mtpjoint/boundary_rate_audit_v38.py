"""Offline-only causal normalizer audit. Does not alter policy or reward inputs."""
from bisect import bisect_right
import math

class BoundaryRate:
    """Use actual boundary counters over at least max(160ms, current sRTT).

    No interpolation, extra observation, trace capacity, or future sample.
    Startup has no full-window estimate and must not seed the mature peak.
    """
    def __init__(self, minimum_s=.160):
        if minimum_s<=0:raise ValueError('Positive horizon required')
        self.minimum_s=float(minimum_s);self.boundaries=[];self.times=[];self.peak=0.;self.last=None
    def add(self,time_s,bytes_acked,srtt_s):
        t,b,r=float(time_s),int(bytes_acked),float(srtt_s)
        if not math.isfinite(t) or not math.isfinite(r) or r<=0 or b<0:raise ValueError('Invalid TCP boundary')
        if self.last is not None and (t<=self.last[0] or b<self.last[1]):raise ValueError('Counter or clock reset requires a new connection')
        self.last=(t,b);self.boundaries.append((t,b));self.times.append(t);cut=t-max(self.minimum_s,r)
        # Keep one boundary before the cutoff. Older history may be needed if
        # sRTT grows later; retain the complete small diagnostic connection.
        index=bisect_right(self.times,cut+1e-12)-1
        if index<0:return None
        before=self.boundaries[index];dt=t-before[0];rate=(b-before[1])/dt;self.peak=max(self.peak,rate)
        return dict(rate=rate,peak=self.peak,span_s=dt,start_s=before[0])
