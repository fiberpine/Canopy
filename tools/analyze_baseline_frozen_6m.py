"""Describe the retained frozen runs; missing repeats are never imputed."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean,median

METRICS=('utilization','queue_mean_ms','queue_p95_ms','queue_p99_ms','drop_fraction')


def analyze(folder):
    load=lambda name:json.loads((folder/name).read_bytes())
    registration=load('health-registration.json');g=registration['gates']
    plan=load('frozen-final-plan.json');allowed={j['run']:j for j in plan}
    hosts=load('frozen-final-latest.json');rows=[r for h in hosts for r in h['rows']]
    assert len({r['job']['run'] for r in rows})==len(rows)
    assert all(r['job']==allowed[r['job']['run']] for r in rows)
    assert all(r['status']=='COMPLETED' for r in rows)
    assert all(r['valid']>0 and all(w['complete_window'] for w in r['windows']) for r in rows)
    trim=lambda r:next(w for w in r['windows'] if w['window']=='trim10')
    groups={group:[r for r in rows if r['job']['group']==group] for group in ('repaired-final6m','H0-reference')}
    cells={};table=[];catastrophes=[]
    for condition in registration['conditions']:
        name=condition['condition'];rr=[r for r in groups['repaired-final6m'] if r['job']['condition']['condition']==name]
        assert len(rr) in (2,3)
        med={k:median(trim(r)[k] for r in rr) for k in METRICS}
        avg={k:mean(trim(r)[k] for r in rr) for k in METRICS}
        rtt=condition['base_rtt_ms']
        checks=dict(utilization=med['utilization']>=g['cell_utilization'],
            mean_queue=med['queue_mean_ms']<=g['queue_rtt_factor']*rtt+g['queue_margin_ms'],
            p95=med['queue_p95_ms']<=g['p95_rtt_factor']*rtt+g['p95_margin_ms'],
            trim_drop=med['drop_fraction']<=g['trim_drop'],
            tail=median(r['tail_utilization'] for r in rr)>=g['tail_utilization'],
            no_catastrophe=all(trim(r)['utilization']>=g['catastrophe_utilization'] for r in rr),
            post_handoff_drop=median(r['post_handoff_drop_fraction'] for r in rr)<=g['post_handoff_drop'])
        complete=len(rr)==3 and {r['job']['repeat'] for r in rr}=={0,1,2}
        cells[name]=dict(valid_repeats=len(rr),complete_three_repeats=complete,
            available_repeat_means=avg,available_repeat_medians=med,
            minimum_run_utilization=min(trim(r)['utilization'] for r in rr),
            maximum_run_utilization=max(trim(r)['utilization'] for r in rr),
            registered_checks_on_available_repeats=checks,three_repeat_pass=all(checks.values()) if complete else None)
        for r in rr:
            if trim(r)['utilization']<g['catastrophe_utilization']:
                catastrophes.append(dict(run=r['job']['run'],condition=name,utilization=trim(r)['utilization']))
        table.append(dict(condition=name,valid_repeats=len(rr),complete_three_repeats=complete,
                          **avg,minimum_run_utilization=cells[name]['minimum_run_utilization'],
                          maximum_run_utilization=cells[name]['maximum_run_utilization'],
                          three_repeat_pass=all(checks.values()) if complete else 'INCOMPLETE'))
    lower=sum(trim(r)['queue_mean_ms'] for r in groups['repaired-final6m'])/72
    failed=[name for name,cell in cells.items() if cell['complete_three_repeats'] and cell['three_repeat_pass'] is False]
    missing=[j for j in plan if j['run'] not in {r['job']['run'] for r in rows}]
    assert len(groups['H0-reference'])==24
    overview={group:{k:mean(trim(r)[k] for r in rr) for k in METRICS} for group,rr in groups.items()}
    available_equal={k:mean(cell['available_repeat_means'][k] for cell in cells.values()) for k in METRICS}
    impossible_to_pass=lower>g['overall_queue_ms'] or len(failed)>2 or bool(catastrophes)
    result=dict(status='PERFORMANCE_FAILURE_ALREADY_FORCED_BUT_CONFIRMATION_INCOMPLETE' if impossible_to_pass else 'INCOMPLETE_NO_FINAL_PERFORMANCE_VERDICT',
        complete_runs=len(rows),candidate_valid_runs=len(groups['repaired-final6m']),H0_valid_runs=24,
        missing_jobs=missing,candidate_available_condition_equal=available_equal,
        candidate_available_run_equal=overview['repaired-final6m'],H0_single_repeat_condition_equal=overview['H0-reference'],
        cells=cells,complete_cells=len(cells)-len(missing),failed_complete_cells=failed,catastrophic_completed_runs=catastrophes,
        queue_lower_bound_even_if_missing_run_queue_zero=lower,
        overall_queue_gate=g['overall_queue_ms'],full_confirmation_cannot_pass=impossible_to_pass,
        interpretation=['95 completed runs are not 96 completed runs.',
            'Candidate available-condition mean uses two repeats in the incomplete cell and three elsewhere; not the planned72-run estimate.',
            'The nonnegative queue lower bound assigns zero only for a mathematical bound, never as an observed run or a plotted/imputed metric.',
            'Completed three-repeat cells use unchanged registered gates; incomplete cell is not certified.',
            'H0 has one repeat per condition: no stability, causal repair effect, or CPU-benefit conclusion.'])
    return result,table


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    result,table=analyze(args.folder)
    (args.folder/'observed-performance-assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (args.folder/'available-condition-metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','missing_jobs','catastrophic_completed_runs','interpretation')}))
