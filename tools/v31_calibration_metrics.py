#!/usr/bin/env python3
"""Offline frozen-run metrics; consumes existing logs, performs no TCP reads."""
import argparse,csv,json,math,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frameworks/mtpjoint'))
from mahimahi_metrics import mm
from resource_cost import summarize

def rows(path):
 with path.open() as f:return list(csv.DictReader(f))

def collect(directory):
 result=json.loads((directory/'result.json').read_text());cfg=result['cfg']
 if result['status']!='COMPLETED' or cfg.get('training_mode') or cfg.get('seconds',0)<60:
  raise ValueError('Requires a completed frozen60-second evaluation; not training or a partial run')
 decisions=rows(directory/'decisions.csv');feedback=rows(directory/'feedback-context.csv')
 network,base,bins=mm(directory/'down.log',cfg['seconds'])
 quality=[]
 for metric in network:
  start=metric['start_s']
  chosen=[i for i,x in enumerate(decisions) if float(x['elapsed_s'])-float(x['actual_dt_ms'])/1000>=start and float(x['elapsed_s'])<=cfg['seconds']]
  selected=[decisions[i] for i in chosen];seconds=sum(float(x['actual_dt_ms'])/1000 for x in selected)
  impulses=[math.log(int(100*4**float(x['cwnd_action']))/100.) for x in selected]
  periods=[float(x['planned_mtp_ms']) for x in selected]
  fresh=[float(feedback[i]['fresh_rtt']) for i in chosen]
  metric.update(observed_exposure_s=seconds,
    integer_command_signed_log_impulse_per_s=sum(impulses)/seconds if seconds else None,
    integer_command_absolute_log_impulse_per_s=sum(abs(x) for x in impulses)/seconds if seconds else None,
    mtp_std_ms=float(np.std(periods)),window_action_std=float(np.std([float(x['cwnd_no_noise']) for x in selected])),
    mtp_low_boundary_fraction=float(np.mean([float(x['mtp_no_noise'])<=-.99 for x in selected])),
    mtp_high_boundary_fraction=float(np.mean([float(x['mtp_no_noise'])>=.99 for x in selected])),
    window_low_boundary_fraction=float(np.mean([float(x['cwnd_no_noise'])<=-.99 for x in selected])),
    window_high_boundary_fraction=float(np.mean([float(x['cwnd_no_noise'])>=.99 for x in selected])),
    fresh_rtt_boundary_fraction=float(np.mean(fresh)),
    fresh_rtt_time_fraction=sum(float(decisions[i]['actual_dt_ms'])*float(feedback[i]['fresh_rtt']) for i in chosen)/(seconds*1000) if seconds else None,
    action_health_scope='Frozen no-noise outputs only; interval-aligned controlled window, not pooled across scenarios')
  quality.append(metric)
 spans=[]
 for line in (directory/'policy-samples.jsonl').read_text().splitlines():
  state=np.asarray(json.loads(line)['state']);assert state.shape==(149,)
  spans.append(float((.02*np.expm1(state[76:86])).sum()))
 return dict(status='PASS_OFFLINE_METRIC_EXTRACTION',source=str(directory),windows=quality,
   cpu=summarize(directory,result),history_span_s=dict(mean=float(np.mean(spans)),p95=float(np.percentile(spans,95))),
   limitations=['Command impulse proxy excludes CUBIC ACK evolution and integer/minimum-window clipping effects',
     'Controlled decision windows and Mahimahi log windows have distinct origins, as in original implementation; both are labeled',
     'CPU retains existing monitor endpoints, separate from full/trim10 windows; includes simulator, not pure inference'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 x=collect(pathlib.Path(a.run));pathlib.Path(a.output).write_text(json.dumps(x,indent=2));print(json.dumps(x))
if __name__=='__main__':main()
