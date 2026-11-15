"""Independent local SMDP checks and explicit approximation counterexamples."""
import csv,hashlib,json,math,sys,tempfile
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[2]
H=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from frameworks.joint_healthy.core import interval_return
from tools.audit_joint_healthy_reward import audit

gamma=.995;base=.02;u=1.;rows=[]
for milliseconds in (20,40,80,120,160):
 n=milliseconds//20;duration=milliseconds/1000.
 reward,discount=interval_return(u,duration)
 direct=sum(gamma**j*u for j in range(n))
 assert abs(reward-direct)<1e-12
 constant_value=reward/(1-discount)
 assert abs(constant_value-200)<1e-10
 with_cost,d2=interval_return(u,duration,.02)
 assert abs((reward-with_cost)-.02)<1e-12 and d2==discount
 rows.append(dict(duration_ms=milliseconds,reward=reward,direct_geometric_sum=direct,discount=discount,
  zero_cost_constant_utility_value=constant_value,
  cost002_constant_utility_value=with_cost/(1-discount)))

critics=[]
for estimate in (195.,200.,205.):
 targets=[]
 for milliseconds in (20,80,160):
  reward,discount=interval_return(u,milliseconds/1000.)
  targets.append(dict(duration_ms=milliseconds,target=reward+discount*estimate))
 critics.append(dict(continuation_value=estimate,targets=targets,
  selected_duration_ms=max(targets,key=lambda x:x['target'])['duration_ms'] if estimate!=200 else 'indifferent',
  interpretation='Illustration with a duration-independent biased continuation value, not the measured final critic'))

delay_plateau=[]
for minimum in (20.,200.):
 for queue in ((1.5,5.,16.,46.) if minimum==200 else (1.5,5.,6.,16.)):
  srtt=minimum+queue
  delay_plateau.append(dict(min_rtt_ms=minimum,illustrative_queue_ms=queue,srtt_ms=srtt,
   delay_factor=min(1.,1.25*minimum/srtt),not_actual_measured_srtt=True))
assert next(x for x in delay_plateau if x['min_rtt_ms']==200 and x['illustrative_queue_ms']==16)['delay_factor']==1
assert next(x for x in delay_plateau if x['min_rtt_ms']==200 and x['illustrative_queue_ms']==46)['delay_factor']==1

# Same synthetic bytes/loss/legacy reference; only endpoint delay aggregation changes.
# Not a TCP-consistent SRTT process: a counterexample to exact interval integration.
with tempfile.TemporaryDirectory() as td:
 p=Path(td);frames=[];boundaries=[]
 rtts=[20.]+[x for _ in range(16) for x in (20.,100.,100.,20.)]
 for index,srtt in enumerate(rtts):
  raw=[srtt,1000.,1.,.02,1000.,10.,1000.,0.,srtt,8.,10.,0.,10.,1500.,20.]
  frames.append(dict(state_id=index,raw=raw))
  boundaries.append(dict(state_id=index,dt_us=20000,delta_bytes=20,delivery_sample_mean=1000))
 (p/'feedback.jsonl').write_text('\n'.join(json.dumps(x) for x in frames))
 with (p/'boundaries.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(boundaries[0]));writer.writeheader();writer.writerows(boundaries)
 alias=audit(p/'feedback.jsonl',p/'boundaries.csv','fixed_legacy_reference')
 assert len({r['final_observed_peak'] for r in alias['results']})==1
 assert alias['results'][2]['relative_return_change_same_trajectory']>.5

checkpoint=json.loads((R/'models/joint-g4-cost0-seed3-20261115/endpoint.joint.json').read_bytes())
assert checkpoint['cfg']['decision_cost']==0
assert checkpoint['cfg']['time_limit_handling']=='bootstrap_last_observation'
result=dict(status='PASS_ARITHMETIC_WITH_EXPLICIT_MODELING_GAPS',
 runtime_commit='59076959520728f6e26182128c9cb347c00868b8',
 actual_run_cost=checkpoint['cfg']['decision_cost'],period_table=rows,
 illustrative_critic_bias=critics,illustrative_delay_plateau=delay_plateau,
 synthetic_fixed_peak_delay_aggregation_counterexample=alias,
 discount_e_folding_seconds=-base/math.log(gamma),discount_half_life_seconds=base*math.log(.5)/math.log(gamma),
 continuous_density_normalization_note=dict(discrete20ms_match=True,
  versus_density_u_per_base_second_scale=-math.log(gamma)/(1-gamma),
  interpretation='Global constant utility scaling when cost0, not a duration preference'),
 previous_real_replay_audit=json.loads((R/'evidence/joint-healthy-twohour-20261115/formal-joint-admission.json').read_bytes()),
 limitations=['The counterexample is synthetic, not a measured reward bias of the final model',
  'Internal reward/discount reconstruction is not validation against a full physical utility integral',
  'Critic target/ranking correctness and state sufficiency are not proved by these tests'],
 new_ssh_connections=0,new_tcp_runs=0,optimizer_updates=0,runtime_files_modified=False)
if len(sys.argv)>1:
 output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],cost=result['actual_run_cost'],constant_values=[x['zero_cost_constant_utility_value'] for x in rows],
 fixed_peak_synthetic_alias_return_deltas=[x['relative_return_change_same_trajectory'] for x in alias['results']],
 e_folding_seconds=result['discount_e_folding_seconds'])))
