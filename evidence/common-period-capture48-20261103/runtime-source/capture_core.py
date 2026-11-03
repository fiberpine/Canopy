import math,numpy as np
PERIODS=(20,40,80,120,160)
def interval(raw,K):
    raw=np.asarray(raw,float);dt=float(raw[3]);factor=min(1.,1.25*raw[14]/(raw[0] if raw[2]>0 else raw[8]));throughput=raw[1]/6000000.;loss=5*raw[7]/6000000.;g=.995**(dt/.02);mass=(1-g)/.005
    return dict(duration=dt,discount=g,throughput=throughput,delay_factor=factor,delay_reduction=throughput*(1-factor),loss=loss,utility=throughput*factor-loss,reward=(throughput*factor-loss)*mass,centered=(throughput*factor-loss)*mass-K*(1-g))
def window_checks(old,new,limits):
    checks=[]
    for key,abs_limit,relative in [('utilization',limits['utilization_absolute'],0),('queue_mean_ms',limits['mean_queue_absolute_ms'],limits['mean_queue_relative']),('queue_p95_ms',limits['p95_absolute_ms'],limits['p95_relative'])]:
        margin=max(abs_limit,relative*abs(old[key]));change=new[key]-old[key];checks.append(dict(metric=key,old=old[key],new=new[key],signed_change=change,absolute_margin=margin,passed=abs(change)<=margin))
    return checks
def finite_suffix(rewards,durations,start,K,tail_Q,tail_N,horizon=20.):
    elapsed=np.r_[0.,np.cumsum(durations)];end=int(np.searchsorted(elapsed,elapsed[start]+horizon,side='left'))-1
    if end>=len(durations):raise ValueError('Insufficient same-run suffix')
    gamma=.995**(np.asarray(durations)/.02);w=.995**((elapsed[start:end+1]-elapsed[start])/.02);product=.995**((elapsed[end+1]-elapsed[start])/.02);centered=np.asarray(rewards)[start:end+1]-K*(1-gamma[start:end+1])
    return dict(end=end,H=float(elapsed[end+1]-elapsed[start]),suffix_discount=float(product),Q_label=float(np.dot(w,centered)+product*tail_Q),Q_tail=float(product*tail_Q),N_label=float(w.sum()+product*tail_N))
def fixed_indices(durations):
    dt=np.asarray(durations,float);start=np.r_[0.,np.cumsum(dt[:-1])];ids=set()
    for lo,hi in [(0,10),(10,float(dt.sum()))]:
        for t in np.linspace(lo,max(lo,hi-1e-6),16):
            i=int(np.searchsorted(start,t));ids.add(min(len(dt)-1,i))
    return sorted(ids)
