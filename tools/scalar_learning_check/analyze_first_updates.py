"""Summarize recorded factual-replay interventions; no true-Q or TCP inference."""
import pathlib,json,csv
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]/'evidence/scalar-first-updates-20261027');p.add_argument('--out',type=pathlib.Path,required=True);args=p.parse_args();L=args.evidence;OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
jobs={p.stem:json.loads(p.read_text()) for p in (L/'collected').glob('*/audit/R*.json')}
assert len(jobs)==16,len(jobs)
rows=[]
for name,z in sorted(jobs.items()):
 rec=z['records'];offset=z['critic_only_updates'];post=[r for r in rec if r['iteration']>=offset]
 def first(key,thr):return next((r['iteration']-offset for r in post if r[key]>=thr),None)
 flip=next((r['iteration'] for r in rec if r['dq1']>0),None)
 rows.append(dict(job=name,arm=z['arm'],sampler_seed=z['sampler_seed'],critic_redraw_seed=z['redraw_seed'],actor_lr=z['actor_lr'],critic_calibration_updates=offset,
  first_abs_mean_policy_ge_point9=next((r['iteration']-offset for r in post if abs(r['pi_mean'])>=.9),None),
  first_fraction_saturated_gt_point1=next((r['iteration']-offset for r in post if r['pi_sat']>.1),None),
  first_abs_mean_shift_ge_point1=first('pi_abs_mean_delta',.1),first_Q1_mean_gradient_positive=flip,
  peak_fraction_saturated=max(r['pi_sat'] for r in post),max_abs_mean_policy=max(abs(r['pi_mean']) for r in post),
  final_pi=post[-1]['pi_mean'],final_sat=post[-1]['pi_sat'],final_mean_abs_shift=post[-1]['pi_mean_abs_delta'],final_abs_mean_shift=post[-1]['pi_abs_mean_delta'],
  final_abs_shift_p95=post[-1]['pi_delta_p95'],final_abs_shift_max=post[-1]['pi_delta_max'],wall_s=z['wall_s']))
 rows[-1]['max_RTT_group_mean_shift']=max(abs(r['pi_by_rtt'][k]-post[0]['pi_by_rtt'][k]) for r in post for k in post[0]['pi_by_rtt'])
with (OUT/'arm-summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
full={r['iteration']:r for r in jobs['R0-s3']['records']};thin={r['iteration']:r for r in jobs['R0-s3-thin']['records']}
observer=dict(initial_metrics_equal=full[0]==thin[0],bitwise_final_parameters_equal=jobs['R0-s3']['summary']['final_parameter_sha256']==jobs['R0-s3-thin']['summary']['final_parameter_sha256'],
 maximum_common_grid_pi_mean_difference=max(abs(full[i]['pi_mean']-thin[i]['pi_mean']) for i in thin),
 interpretation='Diagnostics have no Random ancestors or update fetches. Different diagnostic cadence is not bit-exact; original actor+BN grouped operation has no enforced ordering. Scheduling/numerical explanation remains a hypothesis; do not claim exact historical replay.')
out=dict(status='OFFLINE_DIAGNOSIS_COMPLETE',arms=rows,observer_check=observer,
 supported=['Static factual pre-update data suffices for rapid policy saturation across original sampler repeats; no post-explosion data is required for this mechanism test.',
 'Frozen BN does not prevent rapid saturation in this setting; that does not prove BN and initialization are irrelevant.',
 'Critic-only3000 plus original actorLR still leads to extensive per-state saturation, despite high twin gradient agreement.',
 'lr1e-5 alone delays but does not eliminate saturation; lr1e-6 alone breaches original mean-shift0.1 in all3 sampler repetitions.',
 'critic-only3000 plus lr1e-6 meets the proposed mean-shift and saturation checks for1000 actor updates in3 sampler repeats.'],
 not_proven=['Network throughput/delay/CPU improvement','First1000 online critic gradients are random','Target feature sensitivity is the cause of gradient sign','A necessary10x time-scale theorem','Calibration eliminates normalization/reward aliasing','New baseline readiness','BN globally irrelevant'],
 next='At most one finite online candidate with calibration and slower actor, explicitly two training controls, after defining state-local early guards and policy-at-stop capture. No online candidate launched in this round.',
 limitations='Sampler seeds are not independent training seeds. Static source replay cannot test closed-loop performance. R0/R1/R2 have different snapshot horizons; compare their full histories. Legacy summary updates_to_abs_mean_pi_point9 actually measures absolute mean SHIFT; use the derived CSV for absolute policy means.')
(OUT/'final-analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=1))
