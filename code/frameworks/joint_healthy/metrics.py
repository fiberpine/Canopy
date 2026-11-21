"""Read-only packet tail and sampled CPU accounting; never part of the policy."""
import json
from pathlib import Path
import numpy as np


def queue_tails(path, seconds):
    base = None; delays = {0: [], 10: []}
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith('# base timestamp:'): base = float(line.split(':')[1])
            elif line.startswith('#'): continue
            else:
                fields = line.split()
                if base is None or len(fields) < 4 or fields[1] != '-': continue
                timestamp = (float(fields[0]) - base) / 1000.
                for skip, values in delays.items():
                    if skip <= timestamp < seconds: values.append(float(fields[3]))
    return {('full' if skip == 0 else 'trim10'): dict(
        queue_p99_ms=float(np.percentile(values, 99)) if values else None,
        departure_samples=len(values)) for skip, values in delays.items() if seconds > skip}


def cpu_summary(folder, result):
    rows = json.loads((Path(folder) / 'resources.json').read_text())
    if len(rows) < 2: return dict(status='INSUFFICIENT_SAMPLING')
    span = float(rows[-1]['elapsed_s']) - float(rows[0]['elapsed_s'])
    if span <= 0: raise ValueError('Resource samples are not ordered')
    seen = {}
    for row in rows:
        for p in row['processes']:
            key = (int(p['pid']), p.get('created'))
            cpu = float(p['cpu']['user']) + float(p['cpu']['system'])
            if key not in seen: seen[key] = [cpu, cpu, p['name']]
            else: seen[key][1] = cpu
    python = sum(max(0., b-a) for (pid, _), (a,b,_) in seen.items() if pid == result['python_pid'])
    native = sum(max(0., b-a) for (pid, _), (a,b,_) in seen.items() if pid == result['native_pid'])
    total = sum(max(0., b-a) for a,b,_ in seen.values())
    cores = int(result['logical_cpus'])
    fields = ('user','nice','system','idle','iowait','irq','softirq','steal')
    delta = {f: float(rows[-1]['machine_cpu'].get(f,0))-float(rows[0]['machine_cpu'].get(f,0)) for f in fields}
    machine_total = sum(delta.values())
    return dict(status='SAMPLED_CPU_ACCOUNTING_ONLY', sample_start_s=rows[0]['elapsed_s'],
        sample_end_s=rows[-1]['elapsed_s'], sample_span_s=span, logical_cpus=cores,
        python_cpu_seconds=python, native_cpu_seconds=native, observed_run_tree_cpu_seconds=total,
        python_core_equivalents=python/span, observed_run_tree_core_equivalents=total/span,
        observed_run_tree_all_core_fraction=total/span/cores,
        machine_busy_fraction=(machine_total-delta['idle']-delta['iowait'])/machine_total if machine_total > 0 else None,
        unique_process_identities=len(seen), includes_mahimahi=False,
        mahimahi_coverage='NOT_VERIFIED_DETACHED_EMULATOR_MAY_BE_MISSING',
        excludes_children_user_system_to_avoid_double_count=True,
        sampling_may_miss_short_lived_processes_and_final_cpu=True,
        sample_window_is_not_packet_trim10=True,
        not_a_training_cluster_CPU_measurement=True)
