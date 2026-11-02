"""Evaluate frozen A engineering margins without revising their registered values."""
import collections,csv,json,pathlib,statistics
E=pathlib.Path(__file__).parent;rows=json.loads((E/'evaluation-metrics.json').read_text());complete=len(rows)==264 and all(x['status']=='COMPLETED' for x in rows)
valid=[x for x in rows if x['status']=='COMPLETED'];buckets=collections.defaultdict(list)
for r in valid:buckets[r['job']['group'],r['job']['condition']].append(r)
metrics=['utilization','queue_mean_ms','queue_p95_ms','queue_p99_ms','drop_fraction'];table=[];aggregates={};conditions=sorted({r['job']['condition'] for r in valid});groups=sorted({r['job']['group'] for r in valid})
for window in ['full','startup0_10','trim10']:
 for (group,condition),rs in sorted(buckets.items()):
  networks=[next(w for w in r['network'] if w['window']==window) for r in rs]
  out=dict(window=window,group=group,condition=condition,repeats=len(rs),base_rtt_ms=rs[0]['job']['base_rtt_ms'])
  for m in metrics:out[m]=statistics.mean(n[m] for n in networks);out[m+'_repeat_spread']=max(n[m] for n in networks)-min(n[m] for n in networks)
  for key in ['policy_calls_per_exposure_s','python_cpu_percent_one_core','all_owned_cpu_percent_one_core','native_children_cpu_percent_one_core','inference_ms_mean']:
   values=[r['resource_cost'].get(key) for r in rs];values=[v for v in values if v is not None]
   if key=='inference_ms_mean' and not values:values=[r['policy']['inference_ms_mean'] for r in rs if r.get('policy') and r['policy'].get('inference_ms_mean') is not None]
   out[key]=statistics.mean(values) if values else None
  policies=[r['policy'] for r in rs if r.get('policy')]
  out['saturation']=sum(p['cwnd_abs_gt_099']*p['decisions'] for p in policies)/sum(p['decisions'] for p in policies) if policies else None
  table.append(out)
 for group in groups:
  points=[r for r in table if r['window']==window and r['group']==group];aggregates[window,group]={m:statistics.mean(x[m] for x in points) for m in metrics}
  for m in ['policy_calls_per_exposure_s','python_cpu_percent_one_core','all_owned_cpu_percent_one_core','native_children_cpu_percent_one_core','inference_ms_mean']:
   values=[x[m] for x in points if x[m] is not None];aggregates[window,group][m]=statistics.mean(values) if values else None
  aggregates[window,group]['conditions']=len(points)
checks=[]
def add(name,window,ok,actual,limit,**kwargs):checks.append(dict(name=name,window=window,passed=bool(ok),actual=actual,limit=limit,**kwargs))
if complete:
 for (group,condition),rs in buckets.items():
  if group.startswith('A'):
   v=next(x['saturation'] for x in table if x['window']=='full' and x['group']==group and x['condition']==condition);add('action_health','full',v<.05,v,.05,group=group,condition=condition)
 # Report both full and trim10. Startup is always reported separately; it has no
 # invented extra cutoff in A. These are the original numerical margins.
 for window in ['full','trim10']:
  for t in [80,120,160]:
   a,b=aggregates[window,'A%d'%t],aggregates[window,'fixed%d'%t]
   add('long_mean_utilization',window,a['utilization']>=b['utilization']-.01,a['utilization']-b['utilization'],-.01,period_ms=t)
   for m in ['queue_mean_ms','queue_p95_ms']:add('long_'+m,window,a[m]<=1.25*b[m],a[m]/b[m] if b[m] else None,1.25,period_ms=t)
   for c in conditions:
    a=next(x for x in table if x['window']==window and x['group']=='A%d'%t and x['condition']==c);b=next(x for x in table if x['window']==window and x['group']=='fixed%d'%t and x['condition']==c)
    add('long_each_utilization',window,a['utilization']>=b['utilization']-.05,a['utilization']-b['utilization'],-.05,period_ms=t,condition=c)
  ref=min(['H0','official-orca'],key=lambda g:aggregates[window,g]['utilization'])
  for t in [20,40]:
   a,b=aggregates[window,'A%d'%t],aggregates[window,ref]
   add('short_mean_utilization',window,a['utilization']>=b['utilization']-.02,a['utilization']-b['utilization'],-.02,period_ms=t,reference=ref)
   add('short_queue_p95_ms',window,a['queue_p95_ms']<=1.5*b['queue_p95_ms'],a['queue_p95_ms']/b['queue_p95_ms'] if b['queue_p95_ms'] else None,1.5,period_ms=t,reference=ref)
   for a in [x for x in table if x['window']==window and x['group']=='A%d'%t and x['base_rtt_ms']==100]:add('short_each_R100_utilization',window,a['utilization']>=.9,a['utilization'],.9,period_ms=t,condition=a['condition'])
window_pass={w:all(c['passed'] for c in checks if c['window']==w and c['name']!='action_health') if complete else None for w in ['full','trim10']}
health=all(c['passed'] for c in checks if c['name']=='action_health') if complete else None
status='INCOMPLETE'
if complete:
 if not health or not any(window_pass.values()):status='FAIL_A'
 elif all(window_pass.values()):status='PASS_A'
 else:status='PARTIAL_A_WINDOW_AMBIGUITY'
result=dict(status=status,window_pass=window_pass,action_health_pass=health,completed=len(valid),planned=264,checks=checks,aggregates=[dict(window=k[0],group=k[1],**v) for k,v in aggregates.items()],failed_checks=[c for c in checks if not c['passed']],quantile_definition='Condition-equal mean of independent run packet quantiles; never pooled or linearly merged packet distributions',decision_rule='Frozen A numerical margins are reported independently for full and trim10; registration did not explicitly choose a primary A window. Agreement permits PASS/FAIL, disagreement is PARTIAL and cannot authorize B0. Startup is reported without a new cutoff. External old-reward comparisons are feasibility screens, not causal attribution.')
(E/'A-gates.json').write_text(json.dumps(result,indent=2));(E/'A-condition-metrics.json').write_text(json.dumps(table,indent=2))
if table:
 with (E/'A-condition-metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print(json.dumps(dict(status=result['status'],completed=len(valid),failed_checks=len(result['failed_checks']),trim10=[x for x in result['aggregates'] if x['window']=='trim10'])))
