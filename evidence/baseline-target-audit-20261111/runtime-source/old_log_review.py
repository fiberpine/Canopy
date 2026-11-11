"""Cloud-side old logs only. Packet clock, checkpoint anchor, no new flows."""
import csv
import json
from pathlib import Path

source=Path(SOURCE)
cfg=json.loads((source/'configs/healthy-base-seed3.json').read_text())
selected=('G48-R20','G72-R20','G12-R200','G48-R200')
result=[]
for index,condition in enumerate(cfg['conditions']):
    if cfg['actor_hosts'][index]!=NODE-1 or condition['condition'] not in selected:continue
    folder=source/'logs'/('healthy-base-seed3-actor%d'%index)
    with (folder/'states.csv').open() as stream:rows=list(csv.DictReader(stream))
    windows=[]
    for cp in CP_ANCHORS:
        end=min(cp['epoch'],float(rows[-1]['time_us'])/1e6)
        part=[row for row in rows if end-60<=float(row['time_us'])/1e6<=end]
        assert part
        weight=sum(float(row['delta_t']) for row in part)
        avg=lambda key:sum(float(row[key])*float(row['delta_t']) for row in part)/weight
        windows.append(dict(step=cp['step'],checkpoint_epoch=cp['epoch'],start_epoch=end-60,end_epoch=end,
            rows=len(part),exposure_seconds=weight,reward=avg('reward'),throughput_term=avg('throughput_term'),
            max_bw_capacity=avg('max_bw')/(condition['capacity_mbps']*1e6/8),
            capacity_bytes=0,departure_bytes=0,queue_sum=0,departures=0,arrivals=0,drops=0))
    init=None
    with (folder/'down.log').open() as stream:
        for line in stream:
            if line.startswith('# init timestamp:'):init=float(line.split(':')[1]);continue
            if line.startswith('#'):continue
            fields=line.split()
            if len(fields)<3 or init is None:continue
            epoch=(init+float(fields[0]))/1000
            for window in windows:
                if not window['start_epoch']<=epoch<window['end_epoch']:continue
                size=int(fields[2]);kind=fields[1]
                if kind=='#':window['capacity_bytes']+=size
                elif kind=='-':window['departure_bytes']+=size;window['queue_sum']+=float(fields[3]);window['departures']+=1
                elif kind=='+':window['arrivals']+=1
                elif kind=='d':window['drops']+=size
    for window in windows:
        window['utilization']=window['departure_bytes']/max(1,window['capacity_bytes'])
        window['mean_queue_ms']=window['queue_sum']/max(1,window['departures'])
        window['drop_fraction']=window['drops']/max(1,window['arrivals'])
    result.append(dict(node=NODE,actor=index,condition=condition['condition'],windows=windows,
        clock='Packet epoch=(init timestamp+event timestamp)/1000; training end clipped to last observed state',
        limitation='Checkpoint file mtime is an anchor; online weights change and exploration is active. Descriptive only.'))
print(json.dumps(result))
