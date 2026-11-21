"""Read existing V38 boundary logs. No socket, SSH, training or framework writes.

Export the unchanged149-state transitions with two explicitly labelled rewards.
ACK-sample mean RTT is a measurement alternative, not a time/packet tail integral.
"""
import argparse, csv, hashlib, json, math, pathlib, re
import numpy as np

RAW='v31-observed-interval-orca-density-srtt-v1'
SHADOW='shadow-v1-same-149-state-ack-sample-mean-rtt-density'

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mass(dt,gamma):return -np.expm1(np.log(gamma)*np.asarray(dt)/.02)/(1-gamma)
def stats(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean=float(x.mean()),min=float(x.min()),max=float(x.max())) if len(x) else None
def pack(history,gaps,context,feedback,dt,planned):
    # Match joint_mtp.encode_state's float32 casts and operation order.
    out=np.concatenate((history,[math.log1p(max(0.,dt)*1000/20.),2*math.log(planned/20.)/math.log(8.)-1],context)).astype(np.float32)
    out=np.concatenate((out,np.log1p(gaps/.02))).astype(np.float32)
    hh=history.reshape(10,7);rate=np.zeros(10,np.float32);valid=gaps>0
    rate[valid]=hh[valid,3]*(.02/gaps[valid])
    out=np.concatenate((out,rate)).astype(np.float32)
    out=np.concatenate((out,feedback)).astype(np.float32)
    assert out.shape==(149,) and np.isfinite(out).all()
    return out

def export_run(run,out):
    result=json.loads((run/'result.json').read_text());cfg=result['cfg']
    assert result['status']=='COMPLETED'
    assert result['policy_sha256_before']==result['policy_sha256_after']
    assert cfg['joint_state_dim']==149 and cfg['interval_rate_v28']=='acked_bytes'
    assert cfg['feedback_context_mode']=='feedback' and cfg['time_statistics_mode']=='rates'
    assert cfg['reward_sampling_semantics']=='observed_interval_orca_density'
    assert cfg.get('utility_normalization','observed_peak')=='observed_peak'
    rates=read(run/'interval-rate.csv');states=read(run/'states.csv')
    if not rates or 'avg_rtt_us' not in rates[0]:return dict(run=run.name,status='SKIPPED_NO_AVG_RTT')
    decisions={int(x['decision']):x for x in read(run/'decisions.csv')}
    feedback={int(x['decision']):x for x in read(run/'feedback-context.csv')}
    native={int(x['sequence']):x for x in rates}
    initial=json.loads((run/'initial-context.json').read_text())
    history=np.asarray(initial['history'],np.float32);gaps=np.asarray(initial['frame_intervals_s'],np.float32)
    fr=np.asarray(initial['feedback_context']['rows'],np.float32)
    current=np.asarray(initial['feedback_context']['current'],np.float32)
    previous=pack(history,gaps,np.asarray(initial['features'],np.float32),np.r_[fr.T.reshape(-1),current],0.,20.)
    sparse={int(x['decision']):x for x in map(json.loads,(run/'policy-samples.jsonl').read_text().splitlines())}
    rows=[];state_errors=[];raw_errors=[];native_errors=[];seen_sparse=0
    for ss in states:
        n=int(ss['transition']);d=decisions[n];f=feedback[n];rr=native[n+1]
        assert n==len(rows)+1,'Noncontiguous connection; never bridge missing rows'
        dt=float(ss['delta_t']);planned=float(d['planned_mtp_ms']);gamma=float(cfg['gamma']);cost=float(cfg['decision_cost'])
        assert abs(dt-float(d['actual_dt_ms'])/1000)<1e-10
        sample=np.asarray(json.loads(ss['state']),np.float32)
        history=np.concatenate((history[7:],sample)).astype(np.float32)
        gaps=np.concatenate((gaps[1:],[dt])).astype(np.float32)
        last=np.asarray([float(f[k]) for k in ['executed_window','action_valid','fresh_rtt','age_lower_log','age_upper_log']],np.float32)
        fr=np.concatenate((fr[1:],last[None]),axis=0)
        current=np.asarray([float(f[k]) for k in ['inflight_logratio','retrans_logratio','ssthresh_logratio']],np.float32)
        nxt=pack(history,gaps,np.asarray(json.loads(ss['context_state']),np.float32),np.r_[fr.T.reshape(-1),current],dt,planned)
        a=np.asarray([float(d['cwnd_action']),float(d['mtp_action'])],np.float32)
        if n in sparse:
            p=sparse[n];seen_sparse+=1
            state_errors.extend([float(np.max(np.abs(previous-np.asarray(p['state'],np.float32)))),float(np.max(np.abs(nxt-np.asarray(p['next_state'],np.float32))))])
            np.testing.assert_array_equal(a,np.asarray(p['action'],np.float32))
        srtt=float(ss['srtt_ms']);minimum=float(ss['min_rtt_ms']);mean=float(rr['avg_rtt_us'])/1000
        base=float(ss['throughput_term'])-float(ss['loss_term'])
        margin=float(cfg.get('delay_margin_coef',1.25))
        u=base*min(1.,margin*minimum/srtt)
        raw_errors.append(u-float(ss['reward']))
        independent=(float(rr['acked_bytes_per_s'])-5*float(rr['lost_bytes'])/(float(rr['dt_us'])/1e6))/float(ss['max_bw'])
        native_errors.append(independent-base)
        valid=int(rr['rtt_count'])>0 and mean>0
        sh=base*min(1.,margin*minimum/mean) if valid else np.nan
        r=float(d['reward']);m=float(mass(dt,gamma));raw_errors.append(r-(u*m-cost))
        rows.append(dict(s0=previous,a=a,s1=nxt,duration=dt,discount=gamma**(dt/.02),reward_raw=r,
            reward_shadow=sh*m-cost,raw_density=u,shadow_density=sh,valid_shadow=valid,
            avg_rtt_ms=mean,srtt_ms=srtt,min_rtt_ms=minimum,rtt_count=int(rr['rtt_count'])))
        previous=nxt
    assert seen_sparse>=2 and max(state_errors)==0.,('149-state reconstruction not bitwise exact',stats(state_errors))
    assert max(abs(x) for x in raw_errors)<2e-5,stats(raw_errors)
    assert max(abs(x) for x in native_errors)<2e-5,stats(native_errors)
    assert all(abs(float(decisions[i+1]['discount'])-x['discount'])<1e-12 for i,x in enumerate(rows))
    repeat=int(re.search(r'-r(\d+)-',run.name).group(1));split='train' if repeat==0 else 'heldout'
    arrays={k:np.asarray([x[k] for x in rows]) for k in rows[0]}
    out.mkdir(parents=True,exist_ok=True);dest=out/(run.name+'.npz');assert not dest.exists()
    np.savez_compressed(str(dest),**arrays)
    valid=arrays['valid_shadow'];dt=arrays['duration'];diff=arrays['shadow_density'][valid]-arrays['raw_density'][valid]
    files=['result.json','initial-context.json','states.csv','decisions.csv','feedback-context.csv','policy-samples.jsonl','interval-rate.csv']
    meta=dict(run_id=run.name,condition=cfg['condition'],split=split,rows=len(rows),valid_shadow_rows=int(valid.sum()),
        excluded_no_fresh_rtt=int((~valid).sum()),npz=str(dest),npz_sha256=sha(dest),source_policy_sha=result['policy_sha256_before'],
        raw_schema=RAW,shadow_schema=SHADOW,gamma=gamma,decision_cost=cost,delay_margin_coef=margin,state_dim=149,action_dim=2,
        state_interface_changed=False,actor_updates=0,extra_reads=0,exact_state_checks=seen_sparse*2,state_max_abs_error=max(state_errors),
        original_reward_max_abs_error=max(abs(x) for x in raw_errors),native_base_max_abs_error=max(abs(x) for x in native_errors),
        raw_density_time_weighted=float(np.average(arrays['raw_density'][valid],weights=dt[valid])),
        shadow_density_time_weighted=float(np.average(arrays['shadow_density'][valid],weights=dt[valid])),
        density_difference_time_weighted=float(np.average(diff,weights=dt[valid])),density_difference=stats(diff),
        source_files={x:sha(run/x) for x in files},cfg=cfg,
        limitation='ACK-sample mean RTT, not physical-time mean RTT or exact utility integral; both retain observed_peak normalization. No-fresh rows excluded in both arms.')
    (out/(run.name+'.json')).write_text(json.dumps(meta,indent=2))
    return {k:v for k,v in meta.items() if k not in ('cfg','source_files')}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--logs',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--glob',default='telemetry-s*-r*-m*-audit');a=p.parse_args()
    result=dict(status='RUNNING',schema='shadow-rtt-offline-export-v1',actor_release=False,runs=[],errors=[])
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'manifest.json').exists()
    for d in sorted(a.logs.glob(a.glob)):
        if not d.is_dir():continue
        try:result['runs'].append(export_run(d,a.out))
        except Exception as e:result['errors'].append(dict(run=d.name,error=repr(e)))
    ok=[x for x in result['runs'] if x.get('npz')]
    result.update(status='PASS_EXPORTED' if ok and not result['errors'] else 'NO_GO',train_runs=sum(x['split']=='train' for x in ok),heldout_runs=sum(x['split']=='heldout' for x in ok),
        evaluation='Connection/repeat split; same source weights; forced-period observational data, not causal policy improvement',actor_release=False)
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('runs','errors')}))
if __name__=='__main__':main()
