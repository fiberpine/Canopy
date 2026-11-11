"""Recompute baseline training-scale, trajectory and weakness evidence offline."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import mean,median

def readcsv(text):return list(csv.DictReader(io.StringIO(text)))
def numeric(rows):
    return [{k:(float(v) if v and math.isfinite(float(v)) else None) for k,v in r.items()} for r in rows]
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def nearest(rows,n):return min(rows,key=lambda r:abs(r['inserted']-n))
def run(repo,out):
    load=lambda p:json.loads(p.read_bytes())
    current=repo/'evidence/baseline-target-retrain-20261111'
    facts=load(out/'h0-and-current-facts.json');cp=load(current/'training-complete.json')
    text=load(current/'learner-summary.json')['runs']['targetbn-base-seed3']['training.csv']
    (out/'current-training.csv').write_bytes(text.replace('\r\n','\n').encode())
    rows=numeric(readcsv(text));old=numeric(readcsv((repo/'evidence/baseline/metrics/training_metrics.csv').read_text(encoding='utf-8')))
    mon=numeric(readcsv((current/'training-progress.csv').read_text(encoding='utf-8')))
    oldmon=numeric(readcsv((repo/'evidence/baseline/metrics/training_timeline.csv').read_text(encoding='utf-8')))
    assert all(rows[i]['inserted']<=rows[i+1]['inserted'] for i in range(len(rows)-1))
    warmup=cp['config']['warmup'];u=cp['config']['update_ratio']
    valid=[r for r in rows if r['actor_grad'] is not None]
    debt=lambda r:max(0,max(0,r['inserted']-warmup)*u-r['critic_iterations'])
    for r in rows:
        assert r['critic_optimizer_ops']==2*r['critic_iterations']
        assert r['critic_iterations']==r['actor_updates']==r['target_updates']
    times=[]
    for n in (50000,100000,250000,500000,750000,1000000,1200000,1300000,1500000,1800000,2100000,2400000):
        r=nearest(rows,n);h=nearest(old,n)
        times.append(dict(target_samples=n,current={k:r[k] for k in ('inserted','wall_s','actor_grad','critic_grad','critic_loss','critic2_loss','q_mean','sample_age_mean','fifo_backlog')},h0={k:h[k] for k in ('inserted','wall_s','actor_grad','critic_grad','critic_loss','q_mean')}))
    write(out/'trajectory-milestones.json',times)
    final=cp['final'];n=final['inserted'];updates=final['counters']['critic_iterations']
    coverage=cp['coverage'];samples=coverage['sampled_by_actor'];inserted=coverage['inserted_by_actor']
    actors=[]
    for i,c in enumerate(cp['config']['conditions']):
        actors.append(dict(actor=i,condition=c['condition'],generated=inserted[i],sampled_rows=samples[i],replay_rows_per_new_transition=samples[i]/inserted[i],retained=coverage['retained_by_actor'][i]))
    write(out/'actor-information-balance.json',actors)
    confirms=load(current/'confirm-with-startup-retries.json');runs=[r for h in confirms for r in h['rows']]
    assert len(runs)==72 and all(r['status']=='COMPLETED' for r in runs)
    trim=lambda r:next(w for w in r['windows'] if w['window']=='trim10')
    health=load(current/'health-assessment.json');cells=[]
    for name,cell in sorted(health['cells'].items()):
        rs=[r for r in runs if r['job']['condition']['condition']==name];c=rs[0]['job']['condition'];v=cell['medians']
        service_ms=c['queue_bytes']*8/(c['capacity_mbps']*1e6)*1000
        cells.append(dict(condition=name,rtt=c['base_rtt_ms'],bandwidth=c['nominal_mbps'],utilization=v['utilization'],mean_queue_ms=v['queue_mean_ms'],p95_ms=v['queue_p95_ms'],drop=v['drop_fraction'],tail_utilization=cell['tail_utilization'],queue_capacity_ms=service_ms,mean_queue_fraction_of_buffer=v['queue_mean_ms']/service_ms,
            inferred_flight_bytes=c['capacity_mbps']*125*(c['base_rtt_ms']+v['queue_mean_ms']),
            catastrophic_repeats=sum(trim(r)['utilization']<.5 for r in rs),util_min=min(trim(r)['utilization'] for r in rs),util_max=max(trim(r)['utilization'] for r in rs),checks=cell['checks']))
    groups={str(rtt):{k:mean(mean(trim(r)[k] for r in runs if r['job']['condition']['condition']==c['condition']) for c in cp['config']['conditions'] if c['base_rtt_ms']==rtt) for k in ('utilization','queue_mean_ms','queue_p95_ms','drop_fraction')} for rtt in (20,50,100,200)}
    write(out/'performance-cells.json',cells)
    finite_numerical_faults=[dict(row=i,key=k) for i,r in enumerate(rows) for k in ('critic_loss','critic2_loss','actor_grad','critic_grad') if r['critic_iterations']>0 and r[k] is None]
    oldfinal=facts['h0']['result']['final']
    gate=load(current/'learner-summary.json')['runs']['targetbn-base-seed3']['first-update-coverage.json']
    inferred=[c['inferred_flight_bytes'] for c in cells if c['drop']==0 and c['utilization']>.99 and c['rtt']>=100]
    result=dict(status='OFFLINE_ANALYSIS_NO_CLOUD_NO_TF',
        scale=dict(actors_current=24,actors_official=256,actor_ratio=24/256,nominal_official_training_decisions_per_round=256*50000,nominal_official_rounds_unknown=True,current_transitions=n,fraction_of_one_nominal_official_round=n/(256*50000),current_mean_transitions_per_actor=n/24,current_min_transitions_per_actor=min(inserted),current_max_transitions_per_actor=max(inserted),
            official_replay_capacity_from_launcher=55000*16,current_replay=150000,official_slots_per_actor=55000*16/256,current_mean_slots_per_actor=150000/24,
            current_critic_iterations=updates,current_replay_row_draws=updates*512,replay_rows_per_transition=updates*512/n,
            current_learner_minutes=final['wall_s']/60,h0_learner_minutes=oldfinal['wall_s']/60,h0_transitions=oldfinal['inserted'],h0_updates=oldfinal['counters']['critic_iterations'],
            current_final_replay_total_fraction=150000/n,official_replay_fraction_of_one_nominal_round=880000/(256*50000),
            scale_match_13M_at_observed_average_hours=(256*50000)/(n/final['wall_s'])/3600,
            theoretical_24actor_realtime20ms_max_samples_per_s=1200,actual_current_average_samples_per_s=n/final['wall_s'],
            warning='Nominal50k epochs are not verified successful insertions. Learner duration includes startup/drain; fixed-rate projections are arithmetic, not convergence or fee estimates. Official learner update count/UTD unavailable.'),
        training=dict(learner_log_rows=len(rows),actor_monitor_rows=len(mon),actor_monitor_first_inserted=mon[0]['inserted'],actor_monitor_first_is_not_training_start=True,
            fifo_max_logged=max(r['fifo_backlog'] for r in rows),max_credit_debt_logged=max(debt(r) for r in rows),max_credit_debt_after100k=max(debt(r) for r in rows if r['inserted']>=100000),
            first_update_gate_inserted=gate['inserted'],gate_credit_available=math.floor((gate['inserted']-warmup)*u),first_gate_min_actor_inserted=min(gate['per_actor_inserted']),
            final_debt=math.floor((n-warmup)*u)-updates,finite_numerical_faults=finite_numerical_faults,
            final_actor_grad=valid[-1]['actor_grad'],h0_final_logged_actor_grad=[r for r in old if r['actor_grad'] is not None][-1]['actor_grad'],
            actor_grad_max=max(r['actor_grad'] for r in valid),actor_grad_min=min(r['actor_grad'] for r in valid),
            first_logged_actor_grad_below_1e_minus6=next((r['inserted'] for r in valid if r['actor_grad']<1e-6),None),
            fraction_logged_points_actor_grad_below_1e_minus6=sum(r['actor_grad']<1e-6 for r in valid)/len(valid),
            all_min_observed=[r['inserted'] for r in mon if r['min_saturation_actors']==24],all_max_observed=[r['inserted'] for r in mon if r['max_saturation_actors']==24],
            monitor_last=mon[-1],h0_low_window_max_logged=max(r['low_window_actors'] for r in oldmon),h0_low_window_final_logged=oldmon[-1]['low_window_actors']),
        performance=dict(overall=health['overall'],failed_check_counts={k:sum(not c['checks'][k] for c in cells) for k in cells[0]['checks']},by_rtt_condition_equal_repeat_means=groups,
            catastrophic_runs=sum(c['catastrophic_repeats'] for c in cells),catastrophic_conditions=[c['condition'] for c in cells if c['catastrophic_repeats']],
            maximum_logged_control_period_ms=max(r['control_period_ms']['max'] for r in runs),worst_run_control_p95_ms=max(r['control_period_ms']['p95'] for r in runs),
            low_utilization_median_conditions=[c['condition'] for c in cells if c['utilization']<.9],zero_drop_high_util_R100plus_inferred_flight_bytes=dict(count=len(inferred),mean=mean(inferred),min=min(inferred),max=max(inferred),inference_only=True,not_socket_measurement=True)),
        config_differences={k:dict(h0=facts['h0']['config'][k],current=facts['current']['config'][k]) for k in facts['h0']['config'] if facts['h0']['config'][k]!=facts['current']['config'][k]},
        source_proof='upstream-proof.json',limitations=['No new TF queries, optimizer replay, TCP or cloud log reads.','Gradients are logged diagnostics, not parameter movements or ground-truth environment gradients.','Actor monitor begins744775; earlier actor saturation cannot be dated from it.','H0 and current are not matched one-factor causal runs.'])
    write(out/'analysis.json',result)
    print(json.dumps(dict(scale=result['scale'],training=result['training'],performance=result['performance'],config_differences=result['config_differences'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.repo,a.output or a.repo/'evidence/baseline-scale-audit-20261112')
