"""Independent packet-log windows; run quantiles are never linearly pooled."""
import collections,pathlib
import numpy as np

def network(path,seconds=60,with_seconds=False):
    specs=[('full',0,seconds),('startup0_10',0,min(10,seconds))]
    if seconds>10:specs.append(('trim10',10,seconds))
    accum={n:collections.defaultdict(float) for n,_,_ in specs};delays={n:[] for n,_,_ in specs};base=None;last=0.
    bins = [dict(service_bytes=0, departure_bytes=0) for _ in range(seconds)] if with_seconds else None
    with pathlib.Path(path).open() as stream:
        for line in stream:
            if line.startswith('# base timestamp:'):base=float(line.split(':')[1]);continue
            if line.startswith('#') or base is None:continue
            fields=line.split()
            if len(fields)<3:continue
            t=(float(fields[0])-base)/1000.;last=max(last,t);sign=fields[1];size=int(fields[2])
            if bins is not None and 0 <= t < seconds:
                if sign == '#':
                    bins[int(t)]['service_bytes'] += size
                elif sign == '-':
                    bins[int(t)]['departure_bytes'] += size
            for name,start,end in specs:
                if not start<=t<end:continue
                a=accum[name]
                if sign=='#':a['capacity']+=size
                elif sign=='+':a['arrivals']+=1;a['arrival_bytes']+=size
                elif sign=='d':a['drops']+=size;a['drop_bytes']+=int(fields[3])
                elif sign=='-':a['departure']+=size;delays[name].append(float(fields[3]))
    result=[]
    for name,start,end in specs:
        a=accum[name];q=delays[name]
        result.append(dict(window=name,utilization=a['departure']/a['capacity'] if a['capacity'] else None,
            departure_mbps=a['departure']*8/(end-start)/1e6,capacity_mbps=a['capacity']*8/(end-start)/1e6,
            queue_mean_ms=float(np.mean(q)) if q else 0.,queue_p95_ms=float(np.percentile(q,95)) if q else 0.,queue_p99_ms=float(np.percentile(q,99)) if q else 0.,
            dropped_packets=a['drops'],arrival_packets=a['arrivals'],
            drop_fraction=a['drops']/a['arrivals'] if a['arrivals'] else 0.,
            dropped_bytes=a['drop_bytes'],arrival_bytes=a['arrival_bytes'],
            complete_window=last>=end-.01,last_log_s=last,start_s=start,end_s=end))
    return (result, bins) if with_seconds else result
