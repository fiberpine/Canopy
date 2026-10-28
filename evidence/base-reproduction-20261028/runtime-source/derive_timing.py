
import pathlib,json,sys,csv,collections
R=pathlib.Path(ROOT);F=R/'ipc';sys.path[:0]=[str(F),str(F/'code/runtime/_deps')];import numpy as np
jobs=json.loads((R/'plans/ipc.json').read_text());out=[]
for job in jobs:
 if job['host_index']!=INDEX:continue
 cfg=job['config'];d=F/'logs'/cfg['run_id'];base=init=None
 with (d/'down.log').open() as stream:
  for line in stream:
   if line.startswith('# base timestamp:'):base=float(line.split(':')[1])
   elif line.startswith('# init timestamp:'):init=float(line.split(':')[1])
   elif not line.startswith('#'):break
 assert base is not None and init is not None
 epoch0=(base+init)/1000.;last={};cycles=[];missing=[]
 with (d/'control.csv.stages.csv').open() as stream:
  for row in csv.DictReader(stream):
   t=float(row['time_us'])/1e6;ev=row['event'];sid=int(row['state_id'])
   if ev=='sample_missing':missing.append(t)
   if ev=='action_applied' and 'state_published' in last and 'sample_poll_start' in last and 'cadence_start' in last:
    c=dict(epoch=t,elapsed_s=t-epoch0,state_id=sid,cadence_sleep_ms=(last['sample_poll_start']-last['cadence_start'])*1000,sample_wait_ms=(last['sample_ready']-last['sample_poll_start'])*1000,publish_ms=(last['state_published']-last['sample_ready'])*1000,policy_response_ms=(t-last['state_published'])*1000,cycle_ms=(t-last['cadence_start'])*1000)
    cycles.append(c)
   last[ev]=t
 stats={}
 for skip in [0,10]:
  rows=[r for r in cycles if skip<=r['elapsed_s']<60]
  def stat(v):return dict(n=len(v),mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),max=max(v)) if v else None
  fields=['cadence_sleep_ms','sample_wait_ms','publish_ms','policy_response_ms','cycle_ms']
  metrics={k:stat([r[k] for r in rows]) for k in fields};slow=[r for r in rows if r['cycle_ms']>50]
  stats['full' if skip==0 else 'trim10']=dict(metrics=metrics,missing_rtt_polls=sum(skip<=t-epoch0<60 for t in missing),cycle_over_50ms=len(slow),wait_sample_dominates_slow=sum(r['sample_wait_ms']>.5*r['cycle_ms'] for r in slow),policy_response_over_20ms=sum(r['policy_response_ms']>20 for r in rows),largest_cycles=sorted(rows,key=lambda r:r['cycle_ms'],reverse=True)[:5])
 out.append(dict(run_id=cfg['run_id'],condition=cfg['condition'],arm=job['tag'],host_index=INDEX,epoch0=epoch0,native_first_epoch=cycles[0]['epoch'] if cycles else None,windows=stats))
print(json.dumps(out))
