import pathlib,json,csv,hashlib,collections,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
def csvread(p):
    if not p.exists(): return []
    with p.open() as f: return list(csv.DictReader(f))
def csvwrite(p,rows,fields=None):
    if fields is None: fields=sorted(set().union(*(x.keys() for x in rows))) if rows else ['run_id','status']
    with p.open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def stat(values,kind='mean'):
    if not len(values): return float('nan')
    return float(np.mean(values)) if kind=='mean' else float(np.percentile(values,float(kind)))
def mm(p,seconds,retain_delays=True):
    base=None; init=None; windows={0:collections.defaultdict(float),10:collections.defaultdict(float)}
    delays={0:[],10:[]}; bins=collections.defaultdict(lambda:collections.defaultdict(float)); last=0
    with p.open() as f:
        for line in f:
            if line.startswith('# base timestamp:'): base=float(line.split(':')[1])
            elif line.startswith('# init timestamp:'): init=float(line.split(':')[1])
            elif line.startswith('#'): continue
            else:
                x=line.split()
                if len(x)<3 or base is None: continue
                t=float(x[0])-base; last=max(t,last); sign=x[1];size=int(x[2]);sec=int(t//1000)
                d=bins[sec]
                if sign=='#':d['capacity_bytes']+=size
                elif sign=='-':d['departure_bytes']+=size;d['queue_delay_sum']+=float(x[3]);d['departures']+=1
                for skip,w in windows.items():
                    if not (skip*1000<=t<seconds*1000):continue
                    if sign=='#':w['capacity_bytes']+=size
                    elif sign=='-':
                        w['departure_bytes']+=size;w['departures']+=1;w['queue_delay_sum']+=float(x[3])
                        if retain_delays:delays[skip].append(float(x[3]))
                    elif sign=='+':w['arrival_packets']+=1;w['arrival_bytes']+=size
                    elif sign=='d':w['dropped_packets']+=size;w['dropped_bytes']+=int(x[3])
    out=[]
    for skip,w in windows.items():
        sec=seconds-skip
        if sec<=0:continue
        out.append(dict(window='full' if skip==0 else 'trim10',start_s=skip,end_s=seconds,
            departure_mbps=w['departure_bytes']*8/sec/1e6,capacity_mbps=w['capacity_bytes']*8/sec/1e6,
            utilization=w['departure_bytes']/w['capacity_bytes'] if w['capacity_bytes'] else float('nan'),
            queue_mean_ms=w['queue_delay_sum']/max(1,w['departures']),queue_p95_ms=stat(delays[skip],'95'),
            dropped_packets=w['dropped_packets'],arrival_packets=w['arrival_packets'],
            drop_fraction=w['dropped_packets']/w['arrival_packets'] if w['arrival_packets'] else float('nan'),
            last_log_s=last/1000,complete_window=(last/1000>=seconds-0.01)))
    return out,init+base,bins
