"""Compute registered paired summaries from collected small result JSON files."""
import csv
import json
import pathlib
import statistics

REPO=pathlib.Path(__file__).resolve().parents[2]


def analyze(records, registration):
    assert len(records)==72 and len({r['cfg']['run_id'] for r in records})==72
    assert all(r['contract']['status']=='PASS_REAL_FROZEN_HANDOVER' for r in records)
    groups={}
    per_run=[]
    for r in records:
        c=r['cfg']
        assert r['result']['policy_sha256_before']==r['result']['policy_sha256_after']
        for w in r['windows']:
            assert w['complete_window']
            row=dict(w)
            row.update({k:v for k,v in r['resources'].items() if isinstance(v,(int,float,bool,str))})
            row['cpu_scope']='Same25..50s run CPU copied across windows, not independent observations'
            per_run.append(row)
            groups.setdefault((c['condition'],c['e2_repeat'],w['window']),{})[c['e2_arm']]=row
    differences=[]
    for (condition,repeat,window), arms in groups.items():
        assert set(arms)=={'A0','A1','A2','A4'}
        assert len({a['host'] for a in arms.values()})==1
        for candidate in ('A1','A2','A4'):
            before, after=arms['A0'],arms[candidate]
            differences.append(dict(condition=condition,repeat=repeat,window=window,
                                    candidate=candidate,host=before['host'],
                                    utilization_delta=after['utilization']-before['utilization'],
                                    queue_mean_delta_ms=after['queue_mean_ms']-before['queue_mean_ms']))
    means=[]
    keys=['utilization','departure_mbps','queue_mean_ms','queue_p95_ms','queue_p99_ms',
          'drop_fraction','reward_mean','max_bw_mean','cwnd_mean',
          'python_cpu_percent_one_core','all_owned_cpu_percent_one_core']
    for condition in registration['weak']+registration['normal']:
        for arm in ('A0','A1','A2','A4'):
            for window in ('full','trim10'):
                rs=[r for r in per_run if r['condition']==condition and r['arm']==arm and r['window']==window]
                assert len(rs)==3
                means.append(dict(condition=condition,arm=arm,window=window,repeats=3,
                                  training_seeds=1,**{k:statistics.mean(r[k] for r in rs) for k in keys}))
    primary=[d for d in differences if d['condition'] in registration['weak'] and d['candidate']=='A1' and d['window']=='trim10']
    gain=statistics.mean(d['utilization_delta'] for d in primary)
    positive={c:sum(d['utilization_delta']>0 for d in primary if d['condition']==c) for c in registration['weak']}
    guard=[]
    for c in registration['normal']:
        a=next(m for m in means if m['condition']==c and m['arm']=='A0' and m['window']=='trim10')
        b=next(m for m in means if m['condition']==c and m['arm']=='A1' and m['window']=='trim10')
        guard.append(dict(condition=c,utilization_pass=b['utilization']>=a['utilization']-.03,
                          queue_mean_pass=b['queue_mean_ms']<=1.2*a['queue_mean_ms']+1))
    support=gain>=.15 and min(positive.values())>=2
    decision='SUPPORT_STARTUP_HYPOTHESIS' if support else ('NO_SUPPORT_FOR_PRIMARY_STARTUP_HYPOTHESIS' if gain<=.05 else 'INCONCLUSIVE_NO_AUTOMATIC_EXTENSION')
    result=dict(status='COMPLETE_FROZEN_DIAGNOSTIC',decision=decision,
                primary_gain_percentage_points=100*gain,weak_positive_pairs=positive,
                normal_guard=guard,normal_guard_all_pass=all(x['utilization_pass'] and x['queue_mean_pass'] for x in guard),
                formal_runs=72,training_runs=0,training_seed=3,freeze_repeats=3,
                limitations=['Paired initial configuration and host, not identical TCP random trajectories.',
                             'A2 retains full TCP/CUBIC/RTT connection state, not only cwnd.',
                             'A4 changes normalized policy input using privileged configured capacity; not deployable evidence.',
                             'Support for startup mechanism is not acceptance of a baseline or innovation.',
                             'Development constants only; no new dynamic-network or multi-training-seed evidence.'])
    return result,means,per_run,differences


def write_csv(path, rows):
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=pathlib.Path,default=REPO/'evidence/e2-handover-20261027')
    ap.add_argument('--output',type=pathlib.Path,default=REPO/'diagnostics/e2-handover-20261027')
    args=ap.parse_args();root=args.evidence;output=args.output
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for p in (root/'runs').glob('*/formal-results.json'):
        records.extend(json.loads(p.read_text(encoding='utf-8')))
    decision,means,runs,differences=analyze(records,json.loads((root/'registration.json').read_text()))
    decision.update(json.loads((root/'audit-notes.json').read_text()))
    (output/'e2-decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    write_csv(output/'e2-summary.csv',means)
    write_csv(output/'e2-results.csv',runs)
    write_csv(output/'e2-paired-differences.csv',differences)
    print(json.dumps(decision))
