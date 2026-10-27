#!/usr/bin/env python3
"""Adjudicate the registered exploration-process check without TF/TCP or servers.

Inputs: the new formal-results.json (probe_driver formal stage), the registration produced by
make_exploration_registration.py, and the a969f67 reference formal-results.json.  Applies the
pre-registered predictions and decision rules verbatim; prints one JSON verdict.
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve()
REFERENCE = HERE.parents[2] / 'evidence/scalar-ipc-init-20261027/formal-results.json'


def window(r, name):
    return [w for w in r['packet_windows'] if w['window'] == name][0]


def diag(r):
    d = r.get('window_diagnostics') or {}
    return d if 'decomposition' in d else None


def row(r, ref=None):
    d = diag(r)
    out = dict(run_id=r['run_id'], arm=r.get('exploration_process', 'iid'), condition=r['condition'], seed=r['seed'],
               max_cwnd=max(r['observed_max_cwnd'], r['requested_max_cwnd']),
               fraction_above_bound=r['boundary_weighted_fraction_above_bound'],
               gates_pass=bool(r['million_window_pass'] and r['four_bdp_queue_time_pass']),
               deterministic_action_mean=r['deterministic_action_mean'])
    for name in ('full', 'trim10'):
        w = window(r, name)
        out[name] = {k: w[k] for k in ('utilization', 'drop_fraction', 'queue_mean_ms', 'queue_p95_ms', 'queue_p99_ms')}
    if d:
        b = d['binding'].get('trim10', {})
        out['binding_trim10'] = {k: b.get(k) for k in ('binding_fraction', 'unanchored_fraction', 'inflight_bytes_p95', 'cwnd_over_packets_out_p50')}
        dec = d['decomposition']
        out['per_step'] = dec['per_step']
        out['max_abs_cumulative_exploration'] = dec['max_abs_cumulative_exploration']
        out['command_drift_per_step'] = dec['per_step']['deterministic'] + dec['per_step']['integer_rounding']
    if ref is not None:
        out['reference'] = dict(run_id=ref['run_id'], max_cwnd=max(ref['observed_max_cwnd'], ref['requested_max_cwnd']),
                                fraction_above_bound=ref['boundary_weighted_fraction_above_bound'],
                                full_drop_fraction=window(ref, 'full')['drop_fraction'],
                                trim10_queue_mean_ms=window(ref, 'trim10')['queue_mean_ms'],
                                trim10_utilization=window(ref, 'trim10')['utilization'])
    return out


def analyze(results, plan, reference):
    ref = {r['run_id']: r for r in reference}
    pairs = plan['reference']['paired_run_id']
    assert sorted(r['run_id'] for r in results) == sorted(j['run_id'] for j in plan['formal']), 'incomplete or extra runs'
    capped = set(plan['capped_conditions'])
    rows = [row(r, ref.get(pairs[r['run_id']])) for r in results]
    if plan.get('same_day_all'):
        daily={(r['condition'],r['seed']):r for r in results if r.get('exploration_process')=='iid'}
        for x in rows:
            if x['arm']=='zero_sum_offset':
                rr=daily[(x['condition'],x['seed'])]
                x['reference']=dict(run_id=rr['run_id'],full_drop_fraction=window(rr,'full')['drop_fraction'],source='SAME_DAY_SAME_HOST')
    z = [x for x in rows if x['arm'] == 'zero_sum_offset']
    iid = [x for x in rows if x['arm'] == 'iid']
    anchors = [x for x in iid if x['condition'] in capped]
    assert len(z) == 16 and len(anchors) == 2 * len(capped), (len(z), len(anchors))
    assert len(iid)==(16 if plan.get('same_day_all') else len(anchors))
    evaluable = all('per_step' in x for x in rows)
    verdict = dict(P1_gates=all(x['gates_pass'] for x in z))
    if evaluable:
        verdict['P1b_mechanism'] = (all(x['max_abs_cumulative_exploration'] < .75 for x in z) and
                                    all(x['max_abs_cumulative_exploration'] > 2.0 for x in anchors))
        verdict['P4_quantizer'] = all(-0.0065 < x['per_step']['percent_truncation'] < -0.0035 for x in rows)
    else:
        verdict['P1b_mechanism'] = verdict['P4_quantizer'] = 'NOT_EVALUABLE_WINDOW_DIAGNOSTICS_MISSING'
    binding_pairs = [x for x in z if x['condition'] not in capped]
    lower = sum(x['full']['drop_fraction'] < x['reference']['full_drop_fraction'] for x in binding_pairs)
    verdict['P2_loss'] = dict(pass_=lower >= 10, lower_drop_pairs=lower, pairs=len(binding_pairs))
    verdict['anchors_reproduce'] = dict(pass_=sum(x['max_cwnd'] >= 1e5 for x in anchors) >= 3,
                                        at_least_1e5=sum(x['max_cwnd'] >= 1e5 for x in anchors), anchors=len(anchors))
    failures = [x for x in z if not x['gates_pass']]
    if verdict['P1_gates'] and verdict['P1b_mechanism'] is True:
        decision = 'ADOPT_CANDIDATE_FOR_REVIEW_ONLY_NO_AUTOMATIC_TRAINING (separate environment decision still required for capped conditions)'
    elif failures and all(x['condition'] in capped and x.get('command_drift_per_step', 0) > 0 for x in failures):
        decision = 'CAPPED_ONLY_FAILURE: end-host limitation hypothesis; independent socket and environment verification required'
    elif failures:
        decision = 'UNCAPPED_FAILURE: inspect PRR/ssthresh recovery path; next variable is explicit registered cwnd validation'
    else:
        decision = 'GATES_PASS_BUT_MECHANISM_NOT_CONFIRMED: inspect window diagnostics before adoption'
    return dict(status='ADJUDICATED', decision=decision, verdict=verdict, capped_conditions=sorted(capped),
                failures=[x['run_id'] for x in failures], rows=rows,
                note='Utilization/delay are reported for every run; low utilization is never counted as success.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=pathlib.Path, required=True)
    p.add_argument('--registration', type=pathlib.Path, required=True)
    p.add_argument('--reference', type=pathlib.Path, default=REFERENCE)
    a = p.parse_args()
    print(json.dumps(analyze(json.loads(a.results.read_text()), json.loads(a.registration.read_text()),
                             json.loads(a.reference.read_text())), indent=1))


if __name__ == '__main__':
    main()
