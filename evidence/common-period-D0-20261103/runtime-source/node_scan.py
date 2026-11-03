"""Parse only pre-existing run files. Executed in a new D0 evidence directory."""
import pathlib,json,csv,hashlib,time,collections,sys,os
from d0_core import PERIODS,REQUIRED_RATE,reward_parts,summarize,initial_state
OLD=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102')
def rows(path):
    if not path.exists():return [],[]
    with path.open() as f:
        r=csv.DictReader(f);return list(r.fieldnames or []),list(r)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def scan(reg,node):
    started=time.time();out=dict(node=node,hostname=os.uname().nodename,started_epoch=started,coverage=[],training=[],training_reward_audit=[],control_series=[],startup_samples=[],status='COMPLETED',new_TCP=0,optimizer_operations=0)
    K=reg['K'];mapping={}
    for a in (OLD/'A/host').glob('actor*.result.json'):
        actor=json.loads(a.read_text());index=actor['actor'];elapsed=0.;training_eps=[e['run'] for e in actor['episodes'] if e['kind']=='training'];startup_runs={training_eps[0],training_eps[-1]}
        for ep in actor['episodes']:
            if ep['kind']!='training':continue
            d=OLD/'A/framework/logs'/ep['run'];dc,dec=rows(d/'decisions.csv');rc,rate=rows(d/'interval-rate.csv');sc,states=rows(d/'states.csv');byseq={int(x['sequence']):x for x in rate};bins={};maxerr=0.;empty=0;nonzero=0
            for i,x in enumerate(dec):
                r=byseq[i+2];t=int(round(float(x['planned_mtp_ms'])));dt=float(r['dt_us'])/1e6;phase=min(5,int(elapsed//300));elapsed+=dt
                key=(t,phase);z=bins.setdefault(key,dict(count=0,physical_seconds=0.,empty_RTT=0));z['count']+=1;z['physical_seconds']+=dt;z['empty_RTT']+=int(r['rtt_count'])==0
                if REQUIRED_RATE<=set(r):
                    p=reward_parts(r,K);maxerr=max(maxerr,abs(p['reward']-float(x['reward'])))
            if ep['run'] in startup_runs and (d/'initial-context.json').exists():
                snapshot=json.loads((d/'initial-context.json').read_text());v=initial_state(snapshot);out['startup_samples'].append(dict(actor=index,run=ep['run'],period_ms=int(round(float(dec[0]['planned_mtp_ms']))),state=v.tolist(),source='Recorded training initial components; final-model query is not original online policy output'))
            for (t,phase),v in bins.items():out['training'].append(dict(actor=index,run=ep['run'],period_ms=t,actor_time_bin=phase,**v))
            out['training_reward_audit'].append(dict(actor=index,run=ep['run'],intervals=len(dec),reconstructible=REQUIRED_RATE<=set(rc),max_abs_error=maxerr if REQUIRED_RATE<=set(rc) else None,state_columns=sc,state_vector_dimension=(len(json.loads(states[0]['state'])) if states else None),source_sha256=digest(d/'interval-rate.csv')))
    for job in reg['jobs']:
        if job['host']!=node:continue
        if time.time()>reg['deadline_epoch']-600:out['status']='PARTIAL_BUDGET';break
        group=job['group'];F=OLD/'evaluation'/('scalar' if group in ['H0','official-orca'] else 'joint');d=F/'logs'/job['run'];rc,rate=rows(d/'interval-rate.csv');sc,state=rows(d/'states.csv');cc,control=rows(d/'control.csv');dc,dec=rows(d/'decisions.csv')
        real149=bool(state) and any(len(json.loads(x.get('state','[]')))==149 or len(json.loads(x.get('context_state','[]')))==149 for x in state[:2]);missing=sorted(REQUIRED_RATE-set(rc));full=(not missing and bool(rate));summary=json.loads((d/'joint-summary.json').read_text()) if (d/'joint-summary.json').exists() else None
        cv=dict(run=job['run'],group=group,condition=job['condition'],repeat=job['repeat'],host=node,rate_columns=rc,state_columns=sc,state_rows=len(state),state149_recorded=real149,decision_rows=len(dec),original_reward_reconstructible=full,missing_reward_fields=missing,reward_status='MEASURED' if full else 'NOT_IDENTIFIABLE',S3_state_status='MEASURED' if real149 else 'NOT_RECORDED',S4_status='FIELDS_PRESENT_NEEDS_POLICY_MATCH' if full and real149 else 'NOT_IDENTIFIABLE',actual_logged_summary=summary,source_sha256={name:digest(d/name) for name in ['config.json','joint-summary.json'] if (d/name).exists()})
        if full:
            parts=[reward_parts(r,K) for r in rate[1:]];cv['reconstructed']={k:summarize([p[k] for p in parts]) for k in ['throughput','delay_factor','delay_reduction','loss','utility','reward','centered']};cv['reconstructed']['physical_seconds']=sum(p['dt'] for p in parts)
        if 'percent' in cc:
            total=collections.Counter(int(c['percent']) for c in control if c.get('apply','1')=='1');cv['integer_percent_counts']=dict(total)
            if job['diagnostic_stratum']:
                # Native-only series is not a 149-dimensional policy state.
                series=[]
                for i,c in enumerate(control):
                    if c.get('apply','1')!='1':continue
                    v=dict(sequence=i,percent=int(c['percent']),cwnd_before=int(c['cwnd_before']),cwnd_requested=int(c['cwnd_requested']))
                    if i+1<len(control) and int(c['cwnd_requested'])>0:v['next_boundary_rebound']=int(control[i+1]['cwnd_before'])/int(c['cwnd_requested'])
                    if i+1<len(rate):
                        r=rate[i+1];v.update(dt_s=float(r['dt_us'])/1e6,acked_bytes_per_s=float(r['acked_bytes_per_s']),lost_bytes=int(r['lost_bytes']),rtt_count=int(r['rtt_count']))
                    series.append(v)
                cv['native_trajectory']=dict(available_points=len(series),return_fraction_25_percent=sum(v['percent']==25 for v in series)/max(1,len(series)),rebound=summarize([v['next_boundary_rebound'] for v in series if 'next_boundary_rebound'in v]),state_interpretation='Native control and rate only; not full state, exact float action, z, or Q gradient')
                # Preselected first8 and32 uniform temporal points; no output-based selection.
                ids=sorted(set(list(range(min(8,len(series))))+[int(i) for i in __import__('numpy').linspace(0,max(0,len(series)-1),32)]))
                out['control_series'].append(dict(run=job['run'],group=group,condition=job['condition'],points=[series[i] for i in ids],sampling='first8 plus32 uniform index positions, frozen before model output'))
        out['coverage'].append(cv)
    out['ended_epoch']=time.time();return out
if __name__=='__main__':
    root=pathlib.Path(__file__).resolve().parent.parent;reg=json.loads((root/'registration.json').read_text());node=int(sys.argv[1]);result=scan(reg,node);(root/('node%d-results.json'%node)).write_text(json.dumps(result,indent=2));print(json.dumps(dict(node=node,status=result['status'],frozen_runs=len(result['coverage']),training_episodes=len(result['training_reward_audit']),ended_epoch=result['ended_epoch'])))
