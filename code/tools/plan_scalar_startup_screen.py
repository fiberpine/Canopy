"""Prepare a 40-run paired frozen-policy screen. Does not connect or execute."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/baseline-online-bn-repair-20261112'


def plan():
    registration = json.loads((EVIDENCE / 'frozen-plan-registration.json').read_text())
    conditions = {j['condition']['condition']: j['condition'] for j in registration['jobs']}
    model_hashes = {}
    for group in ('onlinebn-final6m', 'H0-reference'):
        # The registered candidate received its model hashes at final export.
        rows = json.loads((EVIDENCE / 'frozen-final-latest.json').read_text())
        matches = [r['job']['model_sha256'] for host in rows for r in host['rows'] if r['job']['group'] == group]
        assert matches and all(x == matches[0] for x in matches)
        model_hashes[group] = matches[0]
    specifications = [('onlinebn-final6m', 'G12-R200', 6), ('onlinebn-final6m', 'G12-R100', 3)]
    specifications += [('onlinebn-final6m', name, 2) for name in ('G72-R20', 'G48-R200', 'G24-R50', 'G36-R100')]
    specifications += [('H0-reference', 'G12-R200', 3)]
    pairs = [(group, condition, repeat) for group, condition, count in specifications for repeat in range(count)]
    rng = random.Random(20261113)
    rng.shuffle(pairs)
    jobs = []
    for number, (group, condition, repeat) in enumerate(pairs):
        modes = ['legacy', 'first-feedback']
        if rng.randrange(2):
            modes.reverse()
        for mode in modes:
            jobs.append(dict(pair=number, host_slot=number % 5, group=group,
                             repeat=repeat, startup=mode, condition=conditions[condition],
                             model_sha256=model_hashes[group]))
    assert len(jobs) == 40 and Counter(j['host_slot'] for j in jobs) == {k: 8 for k in range(5)}
    return dict(status='OFFLINE_DRAFT_NOT_AUTHORIZED_OR_EXECUTED', seed=20261113,
                slots_per_host=1, order='per-host plan order; both pair members on the same host',
                jobs=jobs, frozen_model='step-6013196; no optimizer, exploration or checkpoint selection',
                suggested_budget=dict(minutes=75, five_host_hours=6.25, cleanup_minutes=10, authorized=False),
                execution=dict(no_training=True, no_exploration=True, control_period_ms=20,
                               matched_audit=True, condition_coverage_change=False),
                screen_gates=dict(all_40_canonical_complete=True,
                                  legacy_G12_R200_catastrophes_at_least=2,
                                  first_feedback_G12_R200_catastrophes=0,
                                  catastrophe_utilization_below=0.5,
                                  candidate_all_cell_medians_pass_original_gates=True,
                                  per_condition_first_10s_paired_utilization_median_difference_at_least=-0.05,
                                  source_model_action_cadence_provenance_pass=True,
                                  real_inflight_audit_required=True,
                                  failure_to_reproduce_legacy='INCONCLUSIVE',
                                  incomplete_or_unqualified_execution_error='INCOMPLETE'),
                retry_policy=dict(maximum=2, only_zero_valid_and_zero_control=True,
                                  unchanged_model_and_owned_cleanup_required=True,
                                  preserve_original_and_canonical_mapping=True, performance_retry=False),
                conditional_confirmation=dict(startup='first-feedback', candidate_repeats=3,
                                              reference_repeats=1, conditions=24, total_runs=96,
                                              independent_from_screen=True,
                                              keep_original_legacy_FAIL=True,
                                              no_automatic_training=True),
                source_registration_sha256=hashlib.sha256((EVIDENCE / 'frozen-plan-registration.json').read_bytes()).hexdigest(),
                unbound=['new explicit budget/deadline', 'fresh host identity mapping',
                         'built native hashes and matched audit setting', 'metric/helper hashes',
                         'absolute traffic-clock and first-10-second capture protocol'],
                original_health_gates=registration['health_gates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON file; otherwise print only')
    args = parser.parse_args()
    data = json.dumps(plan(), indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as destination:
            destination.write(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
