"""Stdlib tests for the limited-learning-check tooling (no TF, no TCP, no servers)."""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/scalar_learning_check'))
sys.path.insert(0, str(ROOT / 'tools/scalar_ipc_diagnostic'))
import adjudicate_learning as al  # noqa: E402
import learning_monitor as lm  # noqa: E402
import make_learning_registration as mlr  # noqa: E402
import window_binding as wb  # noqa: E402


class MonitorTest(unittest.TestCase):
    def test_self_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = lm.self_test(tmp)
        self.assertEqual(r['first_trigger_s'], {'S3_STOP': 40.0})


class AdjudicatorTest(unittest.TestCase):
    def test_missing_or_bad_tail_blocks_long_training(self):
        rows,cells=al._synthetic_rows('good')
        plan=dict(evaluation=dict(cells=cells),tail_guard=True)
        self.assertNotEqual(al.adjudicate(rows,plan,dict(stop=False,rules=[]))['decision'],'GO_LONG_TRAINING')
        for row in rows:row.update(queue_p95_ms=40.0,queue_p99_ms=60.0)
        self.assertEqual(al.adjudicate(rows,plan,dict(stop=False,rules=[]))['decision'],'GO_LONG_TRAINING')
        for row in rows:
            if row['checkpoint']=='1200000':row['queue_p99_ms']=1000.0
        self.assertNotEqual(al.adjudicate(rows,plan,dict(stop=False,rules=[]))['decision'],'GO_LONG_TRAINING')

    def test_all_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = al.self_test(tmp)
        self.assertEqual({k: v[0] for k, v in r.items()},
                         dict(anchoring='ANCHORING_LIMITED', policy='POLICY_LIMITED', good='GO_LONG_TRAINING',
                              static='LONG_RTT_STATIC', collapse='COLLAPSE_REPRODUCED'))


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.plan = mlr.build('TEST')

    def test_training_changes_only_validated_components_and_budget(self):
        cfg = self.plan['training_config']
        self.assertEqual(cfg['exploration_process'], 'zero_sum_offset')
        self.assertEqual(cfg['exploration_offset_rho'], 0.0)
        self.assertEqual(cfg['scalar_window_initialization'], 'small_uniform')
        self.assertTrue(cfg['verified_ipc_v2'] and cfg['orca_actor_loss_moving_stats'])
        self.assertEqual((cfg['sample_budget'], cfg['checkpoint_interval']), (1200000, 100000))
        self.assertEqual(len(cfg['conditions']), 24)
        for key in ('reward_mode', 'update_ratio', 'warmup', 'memsize', 'sampler', 'exploration_std', 'LOSS_TYPE'):
            self.assertNotIn(key, self.plan['differences_from_historical_moving_template'])

    def test_evaluation_jobs(self):
        jobs = self.plan['evaluation']['jobs']
        self.assertEqual(len(jobs), 110)
        self.assertEqual(sum(j['checkpoint'] == 1200000 and j['arm'] == 'A0' for j in jobs), 20)
        a4g = {j['checkpoint'] for j in jobs if j['arm'] == 'A4g' and j['checkpoint'] != 'old'}
        self.assertEqual(a4g, {400000, 1200000})
        self.assertEqual(sum(j['arm'] == 'CUBIC' for j in jobs), 10)
        self.assertEqual(sum(j['checkpoint'] == 'old' for j in jobs), 20)


class WindowBindingTest(unittest.TestCase):
    def test_rtt_proxy_separates_jitter_from_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = wb.self_test(tmp)
        self.assertEqual(r['capped']['trim10']['binding_rtt_fraction'], 0.0)
        self.assertLess(r['jitter']['trim10']['binding_fraction'], 0.6)
        self.assertGreater(r['jitter']['trim10']['binding_rtt_fraction'], 0.9)


class ActorSpeedTest(unittest.TestCase):
    """Synthetic constant-gradient illustration, not universal scale independence."""
    def test_speed_is_independent_of_gradient_magnitude(self):
        import actor_speed_sim as sim
        big = sim.first(sim.run(c=0.02, seed=3, steps=120), .9)
        tiny = sim.first(sim.run(c=0.001, seed=3, steps=120), .9)
        frozen = sim.first(sim.run(c=0.02, seed=3, steps=120, bn_mode='frozen'), .9)
        self.assertTrue(55 <= big <= 110 and 55 <= tiny <= 110 and 55 <= frozen <= 110, (big, tiny, frozen))
        self.assertLessEqual(abs(big - tiny), 15)

if __name__ == '__main__':
    unittest.main()
