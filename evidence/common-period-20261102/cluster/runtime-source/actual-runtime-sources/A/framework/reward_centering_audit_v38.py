"""Offline-only frozen density centering. Not installed in TCP or learner paths."""
import numpy as np

def mass(discount,gamma=.995):
    d=np.asarray(discount,np.float64)
    if not 0<gamma<1 or not np.isfinite(d).all() or np.any(d<=0) or np.any(d>=1):
        raise ValueError('Finite positive-duration discount required')
    return (1-d)/(1-gamma)

def center(reward,discount,environment_ids,baselines,gamma=.995):
    r=np.asarray(reward,np.float64).reshape(-1);d=np.asarray(discount,np.float64).reshape(-1)
    ids=np.asarray(environment_ids).reshape(-1)
    if not len(r)==len(d)==len(ids) or not np.isfinite(r).all():raise ValueError('Invalid factual rows')
    if any(str(x) not in baselines for x in ids):raise ValueError('Unknown environment; actor identity is not environment identity')
    b=np.array([baselines[str(x)] for x in ids],np.float64)
    if not np.isfinite(b).all():raise ValueError('Nonfinite frozen baseline')
    return r-b*mass(d,gamma)

def estimate(reward,discount,duration,environment_ids,decision_cost=.02,gamma=.995):
    r=np.asarray(reward,np.float64).reshape(-1);d=np.asarray(discount,np.float64).reshape(-1)
    t=np.asarray(duration,np.float64).reshape(-1);ids=np.asarray(environment_ids).reshape(-1)
    if not len(r)==len(d)==len(t)==len(ids) or np.any(t<=0) or not np.isfinite(t).all():raise ValueError('Invalid exposure')
    u=(r+decision_cost)/mass(d,gamma)
    return {str(e):float(np.average(u[ids==e],weights=t[ids==e])) for e in np.unique(ids)}
