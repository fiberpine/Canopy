import pathlib,json,csv,hashlib,numpy as np
from capture_core import interval,window_checks
def audit_run(d,job,reg):
    result=json.loads((d/'result.json').read_text());out=dict(run=d.name,job=job,status=result['status'],reason=result.get('reason'),network=[],capture_checks={})
    if result['status']!='COMPLETED':return out
    with np.load(str(d/'captured-boundaries.npz'),allow_pickle=False) as z:cap={k:z[k] for k in z.files}
    n=len(cap['s0']);summary=json.loads((d/'joint-summary.json').read_text());parts=[interval(r,reg['K']) for r in cap['raw_next']];u=np.array([p['utility'] for p in parts]);dt=cap['raw_next'][:,3]
    c=dict(rows=n,state_shape=list(cap['s0'].shape),raw_shape=list(cap['raw_next'].shape),finite=all(np.isfinite(x).all() for x in cap.values()),summary_count_equal=n==summary['decisions'],same_connection_state_chain=bool(np.array_equal(cap['s1'][:-1],cap['s0'][1:])),sequence_gaps=int(np.count_nonzero((cap['state_id']+1)%1000!=cap['next_state_id'])),utility_max_abs_error=float(np.max(abs(u-cap['utility']))),physical_seconds=float(dt.sum()))
    out['capture_checks']=c;assert n>0 and c['finite'] and c['summary_count_equal'] and c['same_connection_state_chain'] and not c['sequence_gaps'] and c['utility_max_abs_error']<=reg['tolerances']['utility_abs'],c
    assert result['policy_sha256_before']==result['policy_sha256_after'];out['policy_unchanged']=True
    with (d/'control.csv').open() as f:control=list(csv.DictReader(f))
    applied=[r for r in control if r.get('apply','1')=='1'];assert len(applied)>=n
    errors=0
    for i,r in enumerate(applied[:n]):
        a=float(cap['action'][i,0]);percent=int(100*4**float(np.clip(a,-1,1)));expected=min(2147483647,max(4,percent*int(r['cwnd_before'])//100));errors+=int(r['state_id'])!=int(cap['state_id'][i]) or int(r['action_id'])!=int(cap['state_id'][i]) or int(r['percent'])!=percent or int(r['cwnd_requested'])!=expected or int(r['return_code'])!=0
    c['native_mapping_errors']=int(errors);assert errors==0
    from common_period_metrics_final import network
    out['network']=network(d/'down.log',60);out['instrumentation_checks']=[]
    for window in out['network']:
        assert window['complete_window'];old=next(x for x in job['old_network'] if x['window']==window['window']);checks=window_checks(old,window,reg['instrumentation_limits']);out['instrumentation_checks'].append(dict(window=window['window'],checks=checks,passed=all(x['passed'] for x in checks)))
    out['instrumentation_pass']=all(x['passed'] for x in out['instrumentation_checks']);out['summary']=summary
    out['file_integrity']={name:dict(bytes=(d/name).stat().st_size,sha256=hashlib.sha256((d/name).read_bytes()).hexdigest()) for name in ['config.json','result.json','captured-boundaries.npz','control.csv','joint-summary.json']}
    return out
