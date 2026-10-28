"""Recompute frozen-run aggregates without treating truncated runs as success.

Inputs are small worker result records, not checkpoint or packet downloads.
Every condition receives equal weight after averaging its registered repeats.
IPC timing counts describe the instrumented short arms, not a CPU benchmark.
"""
import argparse
import collections
import json
import math
import pathlib
import statistics

ZERO_COUNTERS=('critic_iterations','critic_optimizer_ops','actor_updates',
               'target_updates','exploration_calls')
METRICS=('utilization','departure_mbps','queue_mean_ms','queue_p95_ms',
         'queue_p99_ms','drop_fraction','control_p50_ms','control_p95_ms',
         'control_p99_ms')


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_runs(records,expected_conditions,repeats):
    """Fail closed on failed, duplicate, short, drifting, or incomplete cells."""
    expected=set(expected_conditions)
    if not expected or len(expected)!=len(expected_conditions) or repeats<1:
        raise ValueError('nonempty unique conditions and positive repeat count required')
    seen=set();cells=collections.Counter();flat=[]
    for record in records:
        result=record['result'];cfg=result['cfg'];run=cfg['run_id']
        if run in seen:raise ValueError('duplicate run: '+run)
        seen.add(run)
        if result['status']!='COMPLETED' or not finite(result['wall_s']) or result['wall_s']<cfg['seconds']:
            raise ValueError('failed or truncated run: '+run)
        if not result['policy_sha256_before'] or result['policy_sha256_before']!=result['policy_sha256_after']:
            raise ValueError('policy changed: '+run)
        if any(result['counters'][key]!=0 for key in ZERO_COUNTERS):
            raise ValueError('not frozen: '+run)
        condition=cfg['condition']
        if condition not in expected:raise ValueError('unexpected condition: '+condition)
        windows=record['windows']
        if len(windows)!=2 or {w['window'] for w in windows}!={'full','trim10'}:
            raise ValueError('missing/duplicate window: '+run)
        for w in windows:
            if not w['complete_window'] or w['end_s']!=cfg['seconds'] or any(not finite(w[k]) for k in METRICS):
                raise ValueError('incomplete or nonfinite window: '+run)
            flat.append(dict(w,tag=record['tag'],repeat=record['repeat'],host_index=record['host_index']))
        cells[condition]+=1
    if any(cells[c]!=repeats for c in expected):
        raise ValueError('missing or excess repeats: '+str(dict(cells)))
    return flat


def aggregate(records,conditions,repeats):
    flat=validate_runs(records,conditions,repeats)
    result={}
    for window in ('full','trim10'):
        rows=[w for w in flat if w['window']==window];per_condition=[]
        for condition in conditions:
            group=[r for r in rows if r['condition']==condition]
            per_condition.append(dict(condition=condition,**{
                k:statistics.mean(r[k] for r in group) for k in METRICS}))
        mean={k:statistics.mean(r[k] for r in per_condition) for k in METRICS}
        result[window]=dict(condition_equal=mean,per_condition=per_condition,
                            runs=len(rows),conditions=len(conditions),repeats=repeats)
    m=result['trim10']['condition_equal']
    result['registered_coarse_pass']=m['utilization']>=.95 and .7*32.32<=m['queue_mean_ms']<=1.3*32.32
    result['scope']='One training seed; coarse recipe-reproduction criterion, not all-scenario health, paper reproduction or dual-action/CPU advantage.'
    return result


def timing_summary(records):
    """Count observed stages using the same registered trim10 time window."""
    seen=set();groups=collections.defaultdict(list)
    for r in records:
        if r['run_id'] in seen:raise ValueError('duplicate timing run')
        seen.add(r['run_id']);groups[(r['condition'],r['arm'])].append(r)
    out=[]
    for (condition,arm),rows in sorted(groups.items()):
        windows=[r['windows']['trim10'] for r in rows]
        slow=sum(w['cycle_over_50ms'] for w in windows)
        explained=sum(w['wait_sample_dominates_slow'] for w in windows)
        if not 0<=explained<=slow:raise ValueError('invalid timing counts')
        out.append(dict(condition=condition,arm=arm,repeats=len(rows),
                        cycle_over_50ms=slow,sample_wait_majority=explained,
                        missing_rtt_polls=sum(w['missing_rtt_polls'] for w in windows),
                        max_policy_response_p99_ms=max(w['metrics']['policy_response_ms']['p99'] for w in windows),
                        sample_wait_p99_ms=[w['metrics']['sample_wait_ms']['p99'] for w in windows]))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',required=True)
    parser.add_argument('--conditions',required=True,help='JSON list of registered condition names')
    parser.add_argument('--repeats',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();rows=json.loads(pathlib.Path(args.records).read_text())
    conditions=json.loads(pathlib.Path(args.conditions).read_text())
    tags=sorted({r['tag'] for r in rows})
    result={tag:aggregate([r for r in rows if r['tag']==tag],conditions,args.repeats) for tag in tags}
    path=pathlib.Path(args.output)
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':main()
