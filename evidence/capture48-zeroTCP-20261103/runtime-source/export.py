import os,time,pathlib,json,csv,hashlib,sys,collections
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
R=pathlib.Path(__file__).resolve().parent;reg=json.loads((R/'registration.json').read_text());node=int(sys.argv[1]);start=time.time();assert start<reg['deadline_epoch']-600
OLD=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');C=pathlib.Path(reg['capture_root']);P=pathlib.Path('/home/experiment/orca-fixed-scratch-20261024');protected=json.loads((P/'audit/protected-models-before.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert len(protected)==51 and all(sha(P/n)==v for n,v in protected.items())
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
jobs=json.loads((C/'result.json').read_text());arrays=collections.defaultdict(list);meta=[]
for result in jobs:
 assert result['status']=='COMPLETED';job=result['job'];d=C/'framework/logs'/job['run'];p=d/'captured-boundaries.npz';assert sha(p)==reg['source_capture_manifest'][job['run']]['sha256'];z=np.load(str(p),allow_pickle=False);n=len(z['s0']);pre=np.concatenate([z['initial_raw'][None,:],z['raw_next'][:-1]],axis=0);dt=z['raw_next'][:,3];clock=rows(d/'sample-clock.csv');assert len(clock)>=n
 base=None;init=None;stamps=[];sizes=[]
 with (d/'down.log').open() as f:
  for line in f:
   if line.startswith('# init timestamp:'):init=float(line.split(':')[1]);continue
   if line.startswith('# base timestamp:'):base=float(line.split(':')[1]);continue
   if line.startswith('#'):continue
   v=line.split()
   if len(v)>=3 and v[1]=='#':stamps.append(float(v[0]));sizes.append(float(v[2]))
 assert base is not None;stamps=np.asarray(stamps);prefix=np.r_[0.,np.cumsum(sizes)]
 assert init is not None;current_ms=np.asarray([float(c['sample_wall_us'])/1000.-init for c in clock[:n]]);relative=(current_ms-base)/1000.;assert relative.min()>=0 and relative.max()<=61,(job['run'],base,current_ms[:1].tolist(),relative.min(),relative.max())
 lower=np.maximum(base,current_ms-1000.);width=(current_ms-lower)/1000.;assert np.all(width>0)
 end=np.searchsorted(stamps,current_ms,side='right');begin=np.searchsorted(stamps,lower,side='right');capacity=(prefix[end]-prefix[begin])*8/1e6/width
 avgcap=float(prefix[np.searchsorted(stamps,base+60000.,side='left')]-prefix[np.searchsorted(stamps,base,side='left')])*8/60e6
 assert all(int(clock[i]['state_id'])==int(z['state_id'][i]) for i in range(n))
 for name,value in dict(s0=z['s0'],pre_raw=pre,dt=dt,elapsed=np.r_[0.,np.cumsum(dt[:-1])],capacity_recent=capacity,capacity_run=np.full(n,avgcap),run=np.full(n,job['order'],np.int16)).items():arrays[name].append(value)
 meta.append(dict(job=job,rows=n,capacity_mean_mbps=avgcap,relative_state_time_range=[float(relative.min()),float(relative.max())],capture_sha256=sha(p),capacity_log_sha256=sha(d/'down.log'),clock_sha256=sha(d/'sample-clock.csv')))
coverage=[];training_sources={}
for actorfile in sorted((OLD/'A/host').glob('actor*.result.json')):
 actor=json.loads(actorfile.read_text());index=actor['actor']
 for ep in actor['episodes']:
  if ep['kind']!='training':continue
  d=OLD/'A/framework/logs'/ep['run'];dec=rows(d/'decisions.csv');rate=rows(d/'interval-rate.csv');cfg=json.loads((d/'config.json').read_text());bins={};assert len(rate)>=len(dec)+1
  for i,x in enumerate(dec):
   before,after=rate[i],rate[i+1];assert int(before['sequence'])==i+1 and int(after['sequence'])==i+2;t=int(round(float(x['planned_mtp_ms'])));delta=float(after['dt_us'])/1e6;minimum=float(before['min_rtt_us']);srtt=float(before['srtt_us_scaled8'])/8.;ack=float(before['avg_rtt_us']) if float(before['rtt_count'])>0 else srtt;assert minimum>0
   b=bins.setdefault(t,dict(rows=0,seconds=0.,high_SRTT_rows=0,high_SRTT_seconds=0.,high_ACK_rows=0,high_ACK_seconds=0.));b['rows']+=1;b['seconds']+=delta
   for name,v in [('SRTT',srtt),('ACK',ack)]:
    if v/minimum>=2:b['high_'+name+'_rows']+=1;b['high_'+name+'_seconds']+=delta
  coverage.extend(dict(actor=index,condition=cfg['condition'],episode=ep['run'],period_ms=t,**b) for t,b in sorted(bins.items()));training_sources[ep['run']]=dict(rate_sha256=sha(d/'interval-rate.csv'),decisions_sha256=sha(d/'decisions.csv'))
assert len(protected)==51 and all(sha(P/n)==v for n,v in protected.items());out=R/'export.npz';assert not out.exists();np.savez_compressed(str(out),**{k:np.concatenate(v,axis=0) for k,v in arrays.items()});result=dict(node=node,hostname=os.uname().nodename,started_epoch=start,ended_epoch=time.time(),metadata=meta,rows=sum(x['rows'] for x in meta),full_training_coverage=coverage,training_source_sha256=training_sources,protected51_unchanged=True,export_bytes=out.stat().st_size,export_sha256=sha(out),new_TCP=0,Canopy_optimizer_operations=0);assert time.time()-start<reg['actor_reserved_cap_seconds_each'] if node else True;(R/'export-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(node=node,rows=result['rows'],export_bytes=result['export_bytes'],export_sha256=result['export_sha256'],seconds=time.time()-start)))
