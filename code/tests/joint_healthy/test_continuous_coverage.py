"""Coverage bookkeeping for the continuous v4 calibration gate (no TF, no TCP)."""
import math
import sys
import unittest
from pathlib import Path
import numpy as np
from frameworks.joint_healthy.configuration import configuration
from frameworks.joint_healthy.core import PERIODS, period_bucket, period_action, interval_return
from frameworks.joint_healthy.replay import Replay

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))


def nearest(ms): return min(range(5), key=lambda i: abs(math.log(ms / PERIODS[i])))


def proposal_mass(bucket_of):
    """Probability that the stratified log-uniform proposal lands in each bucket."""
    fractions = (np.arange(200000) + .5) / 200000.
    buckets = np.array([bucket_of(20. * 8. ** f) for f in fractions])
    return np.bincount(buckets, minlength=5) / len(buckets)


class ContinuousCoverage(unittest.TestCase):
    def test_legacy_profiles_keep_nearest_anchor_buckets(self):
        cfg = configuration('g4')
        for ms in (20., 29.9, 31., 56., 57., 97., 99., 138., 139., 160.):
            self.assertEqual(period_bucket(cfg, ms), nearest(ms))

    def test_continuous_buckets_are_the_proposal_strata(self):
        cfg = configuration('continuous-v4')
        edges = [20. * 8. ** (k / 5.) for k in range(6)]
        for k in range(5):
            self.assertEqual(period_bucket(cfg, edges[k] * 1.0001), k)
            self.assertEqual(period_bucket(cfg, edges[k + 1] * .9999), k)
        self.assertEqual(period_bucket(cfg, 20.), 0); self.assertEqual(period_bucket(cfg, 160.), 4)
        self.assertEqual(period_bucket(cfg, 20.5), 0)  # fresh-RTT initial policy period

    def test_registered_coverage_is_reachable_only_with_strata(self):
        """Steady-state retained rows per actor and bucket, from probes alone.

        Rows retained per actor = memsize/actors; the optimizer pool excludes the
        8/257 rotated sequence blocks and every 17th birth. At 59cf69a the 160ms
        nearest-anchor bucket received ~6.9% of probes: ~57-69 rows < 128 forever.
        """
        cfg = configuration('continuous-v4')
        retained = cfg['memsize'] / cfg['num_actors'] * (1 - 8. / 257) * (1 - 1. / 17)
        probes = retained * cfg['period_probe_probability']
        strata = probes * proposal_mass(lambda ms: period_bucket(cfg, ms))
        anchors = probes * proposal_mass(nearest)
        required = cfg['calibration_rows_per_actor_period']
        self.assertTrue(np.all(strata >= 1.5 * required), strata)
        self.assertLess(anchors[4], required)
        np.testing.assert_allclose(proposal_mass(lambda ms: period_bucket(cfg, ms)), .2, atol=1e-3)

    def test_replay_counts_strata_for_continuous_rows(self):
        cfg = configuration('continuous-v4'); cfg.update(num_actors=1, memsize=64)
        rp = Replay(cfg); zeros = np.zeros(82, np.float32)
        for sequence, ms in enumerate((24.6, 37.3, 56.6, 85.7, 130., 150.), 1):
            dt = ms / 1000.
            rp.append([zeros, [0., period_action(ms)], [0.], zeros, [0.], [dt], [interval_return(1., dt)[1]]],
                      0, 0, sequence, teacher_action=0.)
        held = int(sum(len(x) for x in rp.holdout))
        self.assertEqual(int(rp.coverage.sum()) + held, 6)
        self.assertEqual(int(rp.coverage[0, 4] + sum(1 for x in rp.holdout[4])), 2)

    def test_pipeline_fixture_schedule_satisfies_its_own_gate(self):
        from verify_joint_healthy_pipeline import conditioned_schedule
        cfg = configuration('continuous-v4')
        periods, schedule, starts = conditioned_schedule(cfg, 5, 300)
        for actor in range(5):
            self.assertEqual(sorted({period_bucket(cfg, ms) for ms in schedule[actor]}), list(range(5)))
        self.assertEqual(sorted(period_bucket(cfg, schedule[a][starts[a] - 1]) for a in range(5)), list(range(5)))
        with self.assertRaises(AssertionError):  # the 59cf69a fixture length cannot reach every held-out block
            conditioned_schedule(cfg, 5, 100)


class PeriodResidualLearningRate(unittest.TestCase):
    def test_bounded_and_bound_to_the_conditioned_architecture(self):
        from frameworks.joint_healthy.core import validate
        from frameworks.joint_healthy import identity
        cfg = configuration('continuous-v4')
        self.assertEqual(cfg['critic_period_residual_lr'], 1e-4); validate(cfg)
        for value in (0., -1e-4, float('nan'), 2e-3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(dict(cfg, critic_period_residual_lr=value))
        validate(dict(cfg, critic_period_residual_lr=None,critic_common_calibration_updates=0))
        with self.assertRaises(ValueError): validate(dict(configuration('g4'), critic_period_residual_lr=1e-4))
        self.assertNotEqual(identity.continuous_contract(cfg),
                            identity.continuous_contract(dict(cfg, critic_period_residual_lr=1e-3)))

    def test_common_calibration_precedes_actor_and_is_admitted_explicitly(self):
        from frameworks.joint_healthy.core import validate
        from frameworks.joint_healthy import identity
        cfg=configuration('continuous-v4')
        self.assertEqual(cfg['critic_common_calibration_updates'],1000)
        for value in (-1,1.5,True,cfg['calibration_updates']):
            with self.subTest(value=value),self.assertRaises(ValueError):validate(dict(cfg,critic_common_calibration_updates=value))
        with self.assertRaises(ValueError):validate(dict(cfg,critic_period_residual_lr=None))
        self.assertNotEqual(identity.continuous_contract(cfg),identity.continuous_contract(dict(cfg,critic_common_calibration_updates=0)))


if __name__ == '__main__': unittest.main()
