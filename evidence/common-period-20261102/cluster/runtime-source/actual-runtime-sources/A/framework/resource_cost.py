"""CPU deltas for the prespecified steady monitoring interval, not model FLOPs."""
import json

def monitored_processes(python_process,native_process,run):
    """Native launcher backgrounds Mahimahi, so it is reparented outside its tree."""
    import psutil
    selected={python_process.pid:python_process,native_process.pid:native_process}
    for p in psutil.process_iter(['pid','cmdline']):
        if any(str(run)+'/' in arg for arg in (p.info.get('cmdline') or [])):
            selected[p.pid]=p
    for p in list(selected.values()):
        try:
            for child in p.children(recursive=True):selected[child.pid]=child
        except psutil.Error:pass
    return [python_process]+[p for pid,p in sorted(selected.items()) if pid!=python_process.pid]

def cpu_delta(samples):
    if len(samples)<3:return dict(cpu_measurement_available=False,reason='Need monitoring samples near 0,25,50 seconds')
    origin=samples[0]['time'];steady=[x for x in samples if 20<=x['time']-origin<=55]
    if len(steady)<2:return dict(cpu_measurement_available=False,reason='Insufficient prespecified steady samples')
    first,last=steady[0],steady[-1];elapsed=last['time']-first['time']
    if elapsed<=0:return dict(cpu_measurement_available=False,reason='Nonpositive time interval')
    left={p['pid']:p for p in first['processes']};right={p['pid']:p for p in last['processes']};common=set(left)&set(right);python_pid=first['processes'][0]['pid']
    work={pid:sum(max(0,right[pid]['cpu_times'].get(k,0)-left[pid]['cpu_times'].get(k,0)) for k in ['user','system']) for pid in common}
    inventory=first.get('resource_inventory','legacy Python/native direct tree; orphaned Mahimahi not captured')
    result=dict(cpu_measurement_available=True,resource_inventory=inventory,cpu_monitor_start_s=first['time']-origin,cpu_monitor_end_s=last['time']-origin,cpu_interval_s=elapsed,python_cpu_percent_one_core=100*work.get(python_pid,0)/elapsed,all_owned_cpu_percent_one_core=100*sum(work.values())/elapsed,native_children_cpu_percent_one_core=100*sum(v for p,v in work.items() if p!=python_pid)/elapsed,common_processes=len(common),unmatched_processes=len(set(left)^set(right)),definition='User+system only, summed over persistent inventoried run-owned PIDs; children_user/system excluded to avoid double counting. One core=100%. Inventory field states whether orphaned Mahimahi was captured; not CPU attributable solely to inference.')
    if 'cpu' in last:
        cpu=last['cpu']
        result['host_cpu_busy_percent_all_cores']=sum(cpu.get(k,0) for k in ['user','nice','system','irq','softirq'])-cpu.get('guest',0)-cpu.get('guest_nice',0)
        result['host_cpu_steal_percent']=cpu.get('steal',0)
        result['host_cpu_scope']='Whole host average across cores for final monitoring interval; includes asynchronous kernel/network and controller overhead. Excludes idle, iowait, steal and duplicate guest.'
    return result

def summarize(folder,result):
    path=folder/'resources.json';cost=cpu_delta(json.loads(path.read_text()) if path.exists() else [])
    summary=folder/'joint-summary.json'
    if summary.exists():
        j=json.loads(summary.read_text());calls=j['policy_inferences'];seconds=j['physical_exposure_s'];cost['policy_calls_source']='Recorded joint inference counter'
    else:
        calls=0 if result['cfg']['method']=='cubic' else result['counters']['valid']+1+int(bool(result['cfg'].get('old_eval_handoff')))
        import csv
        states=folder/'states.csv';seconds=sum(float(x['delta_t']) for x in csv.DictReader(states.open())) if states.exists() else result['cfg']['seconds'];cost['policy_calls_source']='Initial action plus valid feedback actions; no exploration or implicit fallback'
    cost.update(policy_inferences=calls,controlled_exposure_s=seconds,policy_calls_per_exposure_s=calls/seconds if seconds else None)
    return cost
