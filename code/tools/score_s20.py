"""Apply the approved S20 health contract to audited frozen connections."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean


def identity(job):
    condition = job['condition']
    return (job['order'], job['host'], job['group'], condition['condition'],
            condition['trace_sha256'], job['repeat'])


def recovery(bins, event, thresholds):
    """Return recovery time at the end of a continuous high-utilization streak."""
    if len(bins) != 60:
        raise ValueError('Recovery requires 60 measured one-second bins')
    if any(not math.isfinite(row[k]) or row[k] < 0
           for row in bins for k in ('service_bytes', 'departure_bytes')):
        raise ValueError('Invalid recovery packet counts')
    start = 0 if event is None else event['second']
    horizon = thresholds['startup_seconds'] if event is None else thresholds['post_rise_seconds']
    if event is not None:
        # Reject trace/clock drift rather than moving the event to suit the result.
        for index, expected in enumerate(event['expected_service_bytes'], start-2):
            actual = bins[index]['service_bytes']
            tolerance = max(thresholds['phase_tolerance_bytes'],
                            thresholds['phase_tolerance_fraction'] * expected)
            if abs(actual-expected) > tolerance:
                raise ValueError('Logged service does not match preregistered rise')
    streak = 0
    for offset, row in enumerate(bins[start:start+horizon], 1):
        service = row['service_bytes']
        good = service > 0 and row['departure_bytes'] >= thresholds['recovery_utilization'] * service
        streak = streak + 1 if good else 0
        if streak == thresholds['consecutive_seconds']:
            return offset
    return None


def score(rows, registration):
    planned = sorted(identity(j) for j in registration['jobs'])
    actual = sorted(identity(r['job']) for r in rows)
    if len(planned) != 116 or planned != actual or any(
            r['status'] != 'COMPLETED' or r.get('frozen_audit_pass') is not True for r in rows):
        return dict(status='PARTIAL', reason='Incomplete or unaudited 116-run manifest')
    thresholds = registration['health_thresholds']
    conditions = {j['condition']['condition']: j['condition'] for j in registration['jobs']}
    cells = defaultdict(list)
    checks, events, mixed = [], [], []
    for r in rows:
        job = r['job']
        name = job['condition']['condition']
        group = job['group']
        network = {w['window']: w for w in r['network']}
        if set(network) != {'full', 'trim10', 'startup0_10'} or any(
                not w['complete_window'] for w in network.values()):
            return dict(status='PARTIAL', reason='Incomplete packet-log window')
        for w in network.values():
            for key in ('utilization', 'queue_mean_ms', 'queue_p95_ms', 'queue_p99_ms', 'drop_fraction'):
                if w.get(key) is None or not math.isfinite(w[key]) or w[key] < 0:
                    return dict(status='PARTIAL', reason='Invalid network metric')
        cells[group, name].append(dict(r, windows=network))
        event = registration['recovery_events'].get(name)
        try:
            startup = recovery(r.get('second_bins', []), None, thresholds)
            rise = recovery(r.get('second_bins', []), event, thresholds) if event else None
        except ValueError as error:
            return dict(status='PARTIAL', reason=str(error), run_order=job['order'])
        events.append(dict(group=group, condition=name, repeat=job['repeat'],
                           startup_seconds=startup, rise_required=event is not None,
                           rise_seconds=rise))

    def add(metric, scope, window, actual, limit, lower=False):
        checks.append(dict(metric=metric, condition=scope, window=window, actual=actual,
                           limit=limit, passed=actual >= limit if lower else actual <= limit))

    for name, condition in conditions.items():
        runs = cells['S20', name]
        policies = [r['policy'] for r in runs]
        decisions = sum(p['decisions'] for p in policies)
        if decisions <= 0:
            return dict(status='PARTIAL', reason='Missing S20 action-health measurements')
        edge = sum(p['cwnd_abs_gt_099'] * p['decisions'] for p in policies) / decisions
        checks.append(dict(metric='edge_action_fraction', condition=name, window='full',
                           actual=edge, limit=thresholds['edge_fraction_strict_lt'],
                           passed=edge < thresholds['edge_fraction_strict_lt']))
        startup = [r['windows']['startup0_10']['utilization'] for r in runs]
        add('startup_utilization', name, 'startup', mean(startup),
            thresholds['startup_mean_utilization'], lower=True)
        if min(startup) < thresholds['startup_min_repeat_utilization']:
            mixed.append(dict(condition=name, reason='Low startup utilization in a repetition'))
        matching = [e for e in events if e['group'] == 'S20' and e['condition'] == name]
        for metric in ('startup_seconds', 'rise_seconds'):
            if metric == 'rise_seconds' and name not in registration['recovery_events']:
                continue
            passed = sum(e[metric] is not None for e in matching)
            add(metric, name, 'startup', passed, len(matching), lower=True)
            if 0 < passed < len(matching):
                mixed.append(dict(condition=name, reason='Mixed ' + metric))

    for window in ('full', 'trim10'):
        per_condition = {}
        for name, condition in conditions.items():
            def values(group, key):
                return [r['windows'][window][key] for r in cells[group, name]]

            keys = ('utilization', 'queue_mean_ms', 'queue_p95_ms', 'queue_p99_ms', 'drop_fraction')
            s, h = [{k: mean(values(g, k)) for k in keys} for g in ('S20', 'H0')]
            per_condition[name] = s, h
            floor = max(thresholds['min_condition_utilization'],
                        h['utilization'] - thresholds['h0_condition_utilization_margin'])
            add('utilization', name, window, s['utilization'], floor, lower=True)
            if condition['queue_bdp'] >= 5:
                add('deep_queue_mean_ms', name, window, s['queue_mean_ms'],
                    thresholds['deep_h0_queue_factor'] * h['queue_mean_ms'])
            if condition['queue_bdp'] <= .5:
                for key, limit in thresholds['shallow_absolute'].items():
                    add(key, name, window, s[key], limit)
                    if s[key] <= limit < max(values('S20', key)):
                        mixed.append(dict(condition=name, window=window, reason='Individual ' + key))
            else:
                limit = min(thresholds['tail_h0_factor'] * h['queue_p99_ms'],
                            thresholds['tail_rtt_factor'] * condition['base_rtt_ms'])
                add('queue_p99_ms', name, window, s['queue_p99_ms'], limit)
            for group in ('S20', 'H0'):
                util = values(group, 'utilization')
                queue = values(group, 'queue_mean_ms')
                if (max(util)-min(util) > thresholds['repeat_utilization_spread'] or
                        max(queue)-min(queue) > max(thresholds['repeat_queue_spread_ms'],
                                                    thresholds['repeat_queue_spread_fraction'] * mean(queue))):
                    mixed.append(dict(condition=name, window=window, group=group, reason='Repeat spread'))
        x = [pair[0] for pair in per_condition.values()]
        y = [pair[1] for pair in per_condition.values()]
        add('overall_utilization', 'all', window, mean(r['utilization'] for r in x),
            mean(r['utilization'] for r in y) - thresholds['h0_overall_utilization_margin'], lower=True)
        for key in ('queue_mean_ms', 'queue_p95_ms'):
            add('overall_' + key, 'all', window, mean(r[key] for r in x),
                thresholds['h0_overall_queue_factor'] * mean(r[key] for r in y))

    def passes(window):
        return all(c['passed'] for c in checks if c['window'] == window)

    # Action health is shared by both decision windows, matching original A.
    health = all(c['passed'] for c in checks if c['metric'] == 'edge_action_fraction')
    window_pass = {w: passes(w) and health for w in ('full', 'trim10')}
    hard_recovery_failure = any(c['window'] == 'startup' and not c['passed'] and (
        c['metric'] == 'startup_utilization' or c['actual'] == 0) for c in checks)
    if hard_recovery_failure or not any(window_pass.values()):
        status = 'FAIL_S20'
    elif not all(window_pass.values()) or mixed or not passes('startup'):
        status = 'PARTIAL'
    else:
        status = 'PASS_S20'
    return dict(status=status, windows=window_pass, checks=checks, recovery=events, repeat_flags=mixed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.runs.read_text()), json.loads(args.registration.read_text())),
                     indent=2, allow_nan=False))
