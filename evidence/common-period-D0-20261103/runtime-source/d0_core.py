"""D0 pure statistics. No sockets, training, or checkpoint writing."""
import math,numpy as np
PERIODS=(20,40,80,120,160)
REQUIRED_RATE={'acked_bytes_per_s','lost_bytes','dt_us','rtt_count','avg_rtt_us','srtt_us_scaled8','min_rtt_us'}
def reward_parts(row,K):
    missing=REQUIRED_RATE-set(row)
    if missing:raise ValueError('Missing original fields: '+','.join(sorted(missing)))
    dt=float(row['dt_us'])/1e6;g=.995**(dt/.02);mass=(1-g)/.005
    rtt=float(row['avg_rtt_us'])/1000 if int(row['rtt_count']) else (int(row['srtt_us_scaled8'])>>3)/1000.
    minimum=float(row['min_rtt_us'])/1000
    if dt<=0 or rtt<=0 or minimum<=0:raise ValueError('Invalid measured duration/RTT')
    throughput=float(row['acked_bytes_per_s'])/6000000;factor=min(1.,1.25*minimum/rtt);loss=5*float(row['lost_bytes'])/dt/6000000
    return dict(dt=dt,gamma=g,throughput=throughput,delay_factor=factor,delay_reduction=throughput*(1-factor),loss=loss,utility=throughput*factor-loss,reward=(throughput*factor-loss)*mass,centered=(throughput*factor-loss)*mass-K*(1-g))
def summarize(values):
    a=np.asarray(values,dtype=np.float64)
    if not a.size:return dict(n=0,mean=None,p50=None,p95=None,p99=None,min=None,max=None)
    if not np.isfinite(a).all():raise ValueError('Nonfinite measurement')
    return dict(n=int(a.size),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),min=float(a.min()),max=float(a.max()))
def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel();d=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/d) if d>0 else None
def initial_state(snapshot):
    h=np.asarray(snapshot['history'],np.float32);dt=np.asarray(snapshot['frame_intervals_s'],np.float32);ctx=np.asarray(snapshot['features'],np.float32);feedback=np.asarray(snapshot['feedback_context']['features'],np.float32)
    if h.shape!=(70,) or dt.shape!=(10,) or ctx.shape!=(4,) or feedback.shape!=(53,):raise ValueError('Complete recorded startup components required')
    rates=np.zeros(10,np.float32);valid=dt>0;rates[valid]=h.reshape(10,7)[valid,3]*(.02/dt[valid])
    state=np.concatenate((h,[0.,-1.],ctx,np.log1p(dt/.02),rates,feedback)).astype(np.float32)
    if state.shape!=(149,) or not np.isfinite(state).all():raise ValueError('Invalid startup state')
    return state
def expected_actor_period(counts):
    counts=np.asarray(counts,dtype=np.float64)
    if counts.ndim!=2 or counts.shape[1]!=5 or np.any(counts.sum(axis=1)<=0):raise ValueError('Real actor strata required')
    return (counts/counts.sum(axis=1,keepdims=True)).mean(axis=0).tolist()
def suffix_label(reward,dt,start,tail,K,horizon=20):
    reward=np.asarray(reward,float);dt=np.asarray(dt,float);ends=np.cumsum(dt);origin=ends[start-1] if start else 0.;j=int(np.searchsorted(ends,origin+horizon))
    if j>=len(dt):raise ValueError('Insufficient same-run suffix')
    g=.995**(dt/.02);w=np.r_[1.,np.cumprod(g[start:j])];r=reward[start:j+1]-K*(1-g[start:j+1]);product=float(np.prod(g[start:j+1]));return dict(label=float(np.dot(w,r)+product*tail),tail=product*tail,H=float(dt[start:j+1].sum()))

class ReadOnlySession:
    def __init__(self,session):self.session=session;self.cache=set();self.calls=0
    def run(self,fetches,feed_dict=None,**kwargs):
        def flatten(value):
            if isinstance(value,(tuple,list)):
                for item in value:
                    for child in flatten(item):yield child
            else:yield value
        fs=list(flatten(fetches))
        for f in fs:
            op=getattr(f,'op',f);name=op.name
            if name not in self.cache:
                stack=[op];seen=set()
                while stack:
                    x=stack.pop()
                    if x.name in seen:continue
                    seen.add(x.name)
                    if any(z in x.type for z in ['Assign','Apply','Scatter','SaveV2','RestoreV2','Enqueue','Dequeue']):raise RuntimeError('D0 mutating fetch rejected: '+x.name+' '+x.type)
                    stack.extend([t.op for t in x.inputs]);stack.extend(x.control_inputs)
                self.cache.add(name)
        self.calls+=1;return self.session.run(fetches,feed_dict=feed_dict,**kwargs)
