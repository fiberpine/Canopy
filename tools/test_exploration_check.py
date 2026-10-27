"""Stdlib/NumPy tests for the exploration-check tooling (no TF, no TCP, no servers)."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/scalar_ipc_diagnostic'))
import analyze_exploration as ae  # noqa: E402
import make_exploration_registration as mr  # noqa: E402
import window_binding as wb  # noqa: E402

BASE = json.loads((ROOT / 'evidence/scalar-ipc-init-20261027/registration.json').read_text())
REFERENCE = json.loads((ROOT / 'evidence/scalar-ipc-init-20261027/formal-results.json').read_text())


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.plan = mr.build(BASE, 1793500000.0, 'TEST')

    def test_counts_and_capped_conditions(self):
        self.assertEqual(len(self.plan['smoke']), 2)
        self.assertTrue(all(j['probe_seconds'] == 12 for j in self.plan['smoke']))
        self.assertEqual(len(self.plan['formal']), 20)
        self.assertEqual(self.plan['capped_conditions'], ['G48-R200-q2', 'G72-R200-q2'])
        ids = [j['run_id'] for j in self.plan['smoke'] + self.plan['formal']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_same_day_pairs_all_conditions(self):
        plan=mr.build(BASE,1793500000.0,'TEST',same_day_all=True)
        self.assertEqual(len(plan['formal']),32)
        for condition,seed in {(j['conditions'][0]['condition'],j['seed']) for j in plan['formal']}:
            pair=[j for j in plan['formal'] if j['conditions'][0]['condition']==condition and j['seed']==seed]
            self.assertEqual({j['exploration_process'] for j in pair},{'iid','zero_sum_offset'})

    def test_only_the_exploration_process_changes(self):
        base = {j['run_id']: j for j in BASE['formal']}
        allowed = mr.IDENTITY_KEYS | mr.ARM_KEYS | mr.MEASUREMENT_KEYS
        for job in self.plan['formal']:
            ref = base[self.plan['reference']['paired_run_id'][job['run_id']]]
            changed = {k for k in set(job) | set(ref) if job.get(k) != ref.get(k)}
            self.assertTrue(changed <= allowed, (job['run_id'], changed - allowed))
            for key in ('seed', 'conditions', 'scalar_window_initialization', 'exploration_std', 'reward_mode',
                        'verified_ipc_v2', 'diagnostic_no_optimizer', 'probe_seconds', 'server_binary'):
                self.assertEqual(job[key], ref[key], key)
            if job['exploration_process'] == 'zero_sum_offset':
                self.assertEqual(job['exploration_offset_rho'], 0.0)
            else:
                self.assertEqual(job['exploration_process'], 'iid')
                self.assertIn(job['conditions'][0]['condition'], self.plan['capped_conditions'])


class WindowBindingTest(unittest.TestCase):
    def test_self_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = wb.self_test(tmp)
        self.assertEqual(r['capped']['trim10']['unanchored_fraction'], 1.0)
        self.assertEqual(r['binding']['trim10']['binding_fraction'], 1.0)

    def test_pending_final_command_does_not_enter_completed_impulse(self):
        import csv
        import math
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)/'run'
            wb._write_run(root,[100.0]*6,[100.0]*6,[100.0]*6,[2147483647]*6,[0.0]*6,[0.0]*6)
            before=wb.command_decomposition(root)
            with (root/'control.csv').open('a',newline='') as f:
                csv.writer(f).writerow([999999999,7,7,400,100,400,0,1])
            after=wb.command_decomposition(root)
            self.assertEqual(after['pending_commands'],1)
            self.assertEqual(after['command_log_growth'],before['command_log_growth'])


def synthetic_results(plan, fail=None, drift=-0.01):
    ref = {r['run_id']: r for r in REFERENCE}
    out = []
    for job in plan['formal']:
        r = copy.deepcopy(ref[plan['reference']['paired_run_id'][job['run_id']]])
        zero = job['exploration_process'] == 'zero_sum_offset'
        r.update(run_id=job['run_id'], exploration_process=job['exploration_process'])
        if zero:
            r.update(observed_max_cwnd=1000, requested_max_cwnd=1000, boundary_weighted_fraction_above_bound=0.0,
                     million_window_pass=True, four_bdp_queue_time_pass=True)
            for w in r['packet_windows']:
                w['drop_fraction'] *= 0.3
        cum = 0.5 if zero else 9.0
        r['window_diagnostics'] = dict(binding=dict(trim10=dict(binding_fraction=.5, unanchored_fraction=.1)),
                                       decomposition=dict(per_step=dict(deterministic=drift, exploration=0.0,
                                                                        percent_truncation=-0.005, integer_rounding=-0.007),
                                                          max_abs_cumulative_exploration=cum))
        if fail and job['run_id'] in fail:
            r.update(observed_max_cwnd=5e7, million_window_pass=False, four_bdp_queue_time_pass=False)
        out.append(r)
    return out


class AnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.plan = mr.build(BASE, 1793500000.0, 'TEST')

    def test_adopt_path(self):
        v = ae.analyze(synthetic_results(self.plan), self.plan, REFERENCE)
        self.assertTrue(v['decision'].startswith('ADOPT'), v['decision'])
        self.assertTrue(v['verdict']['P1_gates'] and v['verdict']['P1b_mechanism'] and v['verdict']['P4_quantizer'])
        self.assertEqual(v['verdict']['P2_loss']['pairs'], 12)

    def test_capped_only_failure_with_positive_drift(self):
        fail = {'explorecheck-formal-G72-R200-q2-s4-zero_sum'}
        v = ae.analyze(synthetic_results(self.plan, fail, drift=0.02), self.plan, REFERENCE)
        self.assertTrue(v['decision'].startswith('CAPPED_ONLY_FAILURE'), v['decision'])

    def test_uncapped_failure(self):
        fail = {'explorecheck-formal-G12-R200-q2-s3-zero_sum'}
        v = ae.analyze(synthetic_results(self.plan, fail), self.plan, REFERENCE)
        self.assertTrue(v['decision'].startswith('UNCAPPED_FAILURE'), v['decision'])


if __name__ == '__main__':
    unittest.main()
