"""Recompute one candidate normalizer from saved logs; never runs a controller."""
import pathlib,json,csv,sys,argparse,math
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--logs-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.root/'frameworks/mtpjoint'))
from boundary_rate_audit_v38 import BoundaryRate
assert not args.output.exists(),'Refuse to overwrite evidence'
reports=[]
for D in sorted(args.logs_root.glob('telemetry-*')):
 if not (D/'result.json').exists() or json.loads((D/'result.json').read_text()).get('status')!='COMPLETED':continue
 rates=list(csv.DictReader((D/'interval-rate.csv').open()));states=list(csv.DictReader((D/'states.csv').open()));byseq={int(x['transition'])+1:x for x in states};smooth=BoundaryRate();first=rates[0];origin=float(first['previous_mono_us'])/1e6
 smooth.add(0,int(first['previous_bytes_acked']),.02);valid=[];fallback=0;reward_errors=[]
 for x in rates:
  seq=int(x['sequence']);s=byseq.get(seq);rtt=float(s['srtt_ms'])/1000 if s else .02
  t=float(x['sample_mono_us'])/1e6-origin;est=smooth.add(t,int(x['bytes_acked']),max(rtt,.000001))
  if est is None:fallback+=1;continue
  if s:
   bw=float(s['max_bw']);q=float(x['acked_bytes_per_s'])-5*float(x['lost_bytes'])/(float(x['dt_us'])/1e6);delay=min(1,1.25*float(s['min_rtt_ms'])/float(s['srtt_ms']));old=q/bw*delay;new=q/est['peak']*delay if est['peak']>0 else None
   reward_errors.append(old-float(s['reward']));valid.append(dict(t=t,original_peak_mbps=bw*8/1e6,smooth_peak_mbps=est['peak']*8/1e6,smooth_rate_mbps=est['rate']*8/1e6,span_ms=est['span_s']*1000,raw_utility=old,candidate_utility=new,dt_us=float(x['dt_us'])))
 mature=[x for x in valid if x['t']>=1];condition=json.loads((D/'result.json').read_text())['cfg']['condition']
 reports.append(dict(run_id=D.name,condition=condition,valid=len(valid),startup_without_full_window=fallback,raw_reward_reconstruction_max_abs=max(map(abs,reward_errors)),
  final_original_peak_mbps=valid[-1]['original_peak_mbps'],final_candidate_peak_mbps=valid[-1]['smooth_peak_mbps'],max_candidate_rate_mbps=max(x['smooth_rate_mbps'] for x in valid),
  minimum_span_ms=min(x['span_ms'] for x in valid),candidate_to_original_peak_median=float(np.median([x['smooth_peak_mbps']/x['original_peak_mbps'] for x in mature])),
  time_weighted_candidate_minus_raw_utility=float(np.average([x['candidate_utility']-x['raw_utility'] for x in mature],weights=[x['dt_us'] for x in mature])),
  candidate_peak_relative_error_vs24=abs(valid[-1]['smooth_peak_mbps']/24-1) if 'constant24' in condition else None))
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(dict(scope='Offline factual replay; changed normalization only, not counterfactual network outcomes',minimum_horizon_ms=160,uses_future_capacity=False,new_tcp_reads=0,startup='No mature estimate before full horizon; deployment startup semantics NOT_IMPLEMENTED',limits=['Cumulative ACK bursts remain in the numerator','No new native delivered counter','Running maximum still retains historical high capacity','State distribution changes, so old checkpoint cannot silently use this normalizer'],runs=reports),indent=2));print(json.dumps(dict(runs=len(reports),output=str(args.output))))
