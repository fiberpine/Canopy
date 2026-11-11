"""Frozen endpoint only. All72 registered runs, absolute unchanged health gates."""
import csv
import json
from pathlib import Path
from statistics import mean,median

HERE=Path(__file__).resolve().parent
METRICS=('utilization','queue_mean_ms','queue_p95_ms','queue_p99_ms','drop_fraction')


def assess(input_name='confirm-latest.json',output_name='health-assessment'):
    registration=json.loads((HERE/'health-registration.json').read_bytes());g=registration['gates']
    hosts=json.loads((HERE/input_name).read_bytes())
    rows=[row for host in hosts for row in host['rows']]
    planned=json.loads((HERE/'confirm-plan.json').read_bytes())
    assert len(planned)==72 and len({r['run'] for r in planned})==72
    allowed={r['run']:r for r in planned}
    identities=[r['job'].get('original_run',r['job']['run']) for r in rows]
    assert len(identities)==len(set(identities)) and set(identities)<=set(allowed)
    for r,identity in zip(rows,identities):
        assert r['job']['condition']==allowed[identity]['condition']
        assert r['job']['repeat']==allowed[identity]['repeat']
        assert r['job']['checkpoint']==allowed[identity]['checkpoint']
        assert r['job']['model_sha256']==allowed[identity]['model_sha256']
    expected={c['condition'] for c in registration['conditions']}
    grouped={name:[] for name in expected}
    for row in rows:grouped[row['job']['condition']['condition']].append(row)
    complete=all(h['finished'] for h in hosts) and len(rows)==72 and all(len(v)==3 and {r['job']['repeat'] for r in v}=={0,1,2} and all(r['status']=='COMPLETED' for r in v) for v in grouped.values())
    if not complete:
        out=dict(status='INCOMPLETE_OR_EXECUTION_FAILED',runs=len(rows),errors=[r for r in rows if r['status']!='COMPLETED'])
    else:
        cells={};csvrows=[]
        for name in sorted(expected):
            runs=grouped[name];rtt=runs[0]['job']['condition']['base_rtt_ms']
            trim=lambda r:next(w for w in r['windows'] if w['window']=='trim10')
            values={key:median(trim(r)[key] for r in runs) for key in METRICS}
            avg={key:mean(trim(r)[key] for r in runs) for key in METRICS}
            tail=median(r['tail_utilization'] for r in runs);post=median(r['post_handoff_drop_fraction'] for r in runs)
            checks=dict(utilization=values['utilization']>=g['cell_utilization'],
                        mean_queue=values['queue_mean_ms']<=g['queue_rtt_factor']*rtt+g['queue_margin_ms'],
                        p95=values['queue_p95_ms']<=g['p95_rtt_factor']*rtt+g['p95_margin_ms'],
                        trim_drop=values['drop_fraction']<=g['trim_drop'],tail=tail>=g['tail_utilization'],
                        no_catastrophe=all(trim(r)['utilization']>=g['catastrophe_utilization'] for r in runs),
                        post_handoff_drop=post<=g['post_handoff_drop'])
            cells[name]=dict(medians=values,repeat_means=avg,tail_utilization=tail,post_handoff_drop=post,checks=checks,pass_all=all(checks.values()))
            csvrows.append(dict(condition=name,**values,tail_utilization=tail,post_handoff_drop=post,pass_all=all(checks.values())))
        overall={key:mean(c['repeat_means'][key] for c in cells.values()) for key in METRICS}
        good=overall['utilization']>=g['overall_utilization'] and overall['queue_mean_ms']<=g['overall_queue_ms']
        failures=[name for name,c in cells.items() if not c['pass_all']]
        catastrophe=any(not c['checks']['no_catastrophe'] for c in cells.values())
        status='PASS' if good and not failures else ('PARTIAL' if good and len(failures)<=2 and not catastrophe else 'FAIL')
        out=dict(status=status,overall=overall,cells=cells,failed_conditions=failures,catastrophe=catastrophe,runs=72,
                 scope='One training seed,24trained5BDP conditions,three fresh evaluation connections each; not multi-seed or shallow/dynamic validation')
        with (HERE/'condition-metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    (HERE/(output_name+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('cells','errors')}))


if __name__=='__main__':
    import sys
    assess(*sys.argv[1:])
