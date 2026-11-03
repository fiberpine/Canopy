import concurrent.futures,json,pathlib,sys
E=pathlib.Path(__file__).parent;sys.path.insert(0,str(E.parent/'common-period-cluster-20261102'));from remote import run
reg=json.loads((E/'registration.json').read_text());hosts=['learner']+['canopy-actor%d'%i for i in range(1,5)]
def one(pair):
 n,h=pair;code='ROOT=%r\n'%reg['cloud_root']+r'''
import pathlib,csv,json,sys,hashlib
R=pathlib.Path(ROOT);sys.path.insert(0,str(R/'framework/code/runtime/_deps'));import numpy as np
reg=json.loads((R/'registration.json').read_text())
rows=json.loads((R/'result.json').read_text());out=[]
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
for row in rows:
 if row['status']!='COMPLETED':continue
 d=R/'framework/logs'/row['run'];z=np.load(str(d/'captured-boundaries.npz'),allow_pickle=False);raw=z['raw_next'];n=len(raw);rates=read(d/'interval-rate.csv');assert len(rates)>=n+1
 first=int(rates[0]['sample_mono_us']);last=int(rates[n]['sample_mono_us']);polls=[p for p in read(d/'kernel-polls.csv') if first<int(p['sample_mono_us'])<=last];rate=rates[1:n+1]
 dt_error=float(np.max(abs(raw[:,3]-np.asarray([float(r['dt_us'])/1e6 for r in rate]))));ack_error=float(np.max(abs(raw[:,1]-np.asarray([float(r['acked_bytes_per_s']) for r in rate]))));loss_error=float(np.max(abs(raw[:,7]*raw[:,3]-np.asarray([float(r['lost_bytes']) for r in rate]))))
 seq_errors=sum(int(r['state_id'])!=int(z['next_state_id'][i]) for i,r in enumerate(read(d/'sample-clock.csv')[1:n+1]));wait_errors=sum(int(p['configured_wait_us'])!=row['job']['period_ms']*1000 for p in polls)
 x=dict(run=row['run'],intervals=n,controlled_kernel_polls=len(polls),wait_errors=wait_errors,next_state_id_errors=seq_errors,dt_max_abs_error_s=dt_error,ACK_rate_max_abs_error_bytes_s=ack_error,loss_max_abs_error_bytes=loss_error,sha256={name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ['interval-rate.csv','kernel-polls.csv','sample-clock.csv','decision-timing.csv']})
 dt=raw[:,3];factor=np.minimum(1.,1.25*raw[:,14]/np.where(raw[:,2]>0,raw[:,0],raw[:,8]));throughput=raw[:,1]/6000000.;delay=throughput*(1-factor);loss=5*raw[:,7]/6000000.;utility=throughput-delay-loss;gamma=.995**(dt/.02);reward=utility*(1-gamma)/.005
 x['reward_components']=dict(time_weighted_throughput=float(np.dot(throughput,dt)/dt.sum()),time_weighted_delay_reduction=float(np.dot(delay,dt)/dt.sum()),time_weighted_loss_penalty=float(np.dot(loss,dt)/dt.sum()),time_weighted_utility=float(np.dot(utility,dt)/dt.sum()),utility_component_max_abs_error=float(np.max(abs(utility-z['utility']))),raw_R_sum=float(reward.sum()),centered_R_sum=float((reward-reg['K']*(1-gamma)).sum()))
 x['passed']=len(polls)==n and not wait_errors and not seq_errors and dt_error<=1e-7 and ack_error<=.001 and loss_error<=1e-4 and x['reward_components']['utility_component_max_abs_error']<=1e-10;out.append(x)
result=dict(passed=all(x['passed'] for x in out),runs=out);(R/'native-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 r=run(h,code,80);assert r['code']==0,r;x=json.loads(r['output']);(E/('native-audit-node%d.json'%n)).write_text(json.dumps(x,indent=2));return x
xs=list(concurrent.futures.ThreadPoolExecutor(5).map(one,enumerate(hosts)));flat=sum([x['runs'] for x in xs],[]);result=dict(passed=all(x['passed'] for x in xs),runs=flat);(E/'native-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['passed'],runs=len(flat),failed=[r for r in flat if not r['passed']])));assert result['passed']
