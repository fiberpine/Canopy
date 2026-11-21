"""Offline capacity evidence and explicit extrapolations; never contacts hosts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

def analyze(root,out):
    out.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_bytes())
    audit=root/'evidence/baseline/audit';metrics=root/'evidence/baseline/metrics'
    base=load(audit/'balanced-admission.json');sat=load(audit/'saturated-service.json')
    latest24=load(root/'evidence/baseline-target-retrain-20261111/admission-pass.json')
    assert latest24['status']=='PASS'
    with (metrics/'optimizer_stage_summary.csv').open(encoding='utf-8',newline='') as s:stages=list(csv.DictReader(s))
    selected=[r for r in stages if r['run_id'] in (base['run_id'],sat['run_id'],'formal24-grid24-seed3-balanced-u010')]
    kernel=[]
    for r in selected:
        t=sum(float(r[k]) for k in ('critic_ms_mean','actor_bn_ms_mean','target_ms_mean'))
        kernel.append(dict(run=r['run_id'],measured_mean_stage_sum_ms=t,inverse_stage_sum_per_s=1000/t,interpretation='Incomplete compute-time inverse, not measured end-to-end saturated service. Excludes sampling, orchestration and other work.'))
    per_actor=base['lambda_per_s']/24;u=.1;mu=sat['updates_per_s'];scenarios=[]
    conditions=load(root/'templates/baseline-train.json')['conditions']
    for n in (24,32,48,64,72,96,128,256):
        counts={c['condition']:sum(i%24==j for i in range(n)) for j,c in enumerate(conditions)}
        demand=per_actor*n*u
        scenarios.append(dict(actors=n,actors_per_host=n/4,projected_samples_per_s=per_actor*n,projected_required_complete_updates_per_s=demand,available_ms_per_complete_update=1000/demand,
            demand_over_old_48actor_saturated_service=demand/mu,condition_counts=counts,condition_weight_uniform=len(set(counts.values()))==1,
            projection_only=True,current_configuration_admission_passed=(True if n==24 else None)))
    process=[]
    with (metrics/'process_resource_metrics.csv').open(encoding='utf-8',newline='') as s:
        for r in csv.DictReader(s):
            if r['run_id'] in (base['run_id'],sat['run_id']) and r['role']=='learner' and r['cpu_percent_one_core']:
                process.append(dict(run=r['run_id'],one_core_pct=float(r['cpu_percent_one_core'])))
    result=dict(status='OFFLINE_CAPACITY_AUDIT_NO_CLOUD',current24_admission_proof='evidence/baseline-target-retrain-20261111/admission-pass.json',current24_neutral_saturated_service_measured=False,historical_24actor_admission=base,historical_48actor_saturation=sat,
        historical_stage_costs=kernel,historical_learner_process_cpu=[dict(run=run,mean_one_core_pct=mean(r['one_core_pct'] for r in process if r['run']==run)) for run in sorted({r['run'] for r in process})],
        projections=scenarios,projection_basis=dict(neutral_real_remote_inference_samples_per_actor_s=per_actor,update_ratio=u,old_saturated_service=mu,
            old_service_is_not_current_hard_ceiling=True,old_48actor_measured_update_ratio=.25,scenarios_use_current_ratio=.1,
            admission_service_not_saturated=not base['saturated_service_rate_measured']),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (audit/'balanced-admission.json',audit/'saturated-service.json',metrics/'optimizer_stage_summary.csv',root/'frameworks/baseline/distributed.py',root/'run.py')},
        findings=['Low host-average CPU can coexist with a serial learner throughput bottleneck.','All policy inference and optimizer compute are placed on the learner CPU; actor nodes primarily run TCP/Mahimahi and client control.','Adding actor servers does not independently increase learner capacity.','32 actors is a capacity probe candidate, not an admitted training configuration.','48 actors at U=.1 was not certified by the old U=.25 failure or U=.03 success.','The CLI cyclically repeats24conditions:32actors doubles only8condition weights with actor-balanced replay;48actors duplicates the whole grid.','The latest run lacks a complete locally retained per-core CPU time series. Historical metrics are not substituted for it.'],
        experiment_server_connections=0,new_TCP=0,new_TF_queries=0,new_optimizer_operations=0,new_training_authorized=False)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(old_measured_mu=mu,projections=[{k:r[k] for k in ('actors','projected_samples_per_s','projected_required_complete_updates_per_s','condition_weight_uniform')} for r in scenarios])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args()
    analyze(a.repo,a.output or a.repo/'evidence/baseline-capacity-audit-20261112')
