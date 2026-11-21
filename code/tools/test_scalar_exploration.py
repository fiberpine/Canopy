"""NumPy-only tests for the opt-in zero-sum offset exploration process (no TF/TCP)."""
import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'frameworks/canopy_single'))
import scalar_exploration as se  # noqa: E402


class ZeroSumOffsetTest(unittest.TestCase):
    def test_offset_is_stationary_and_step_variance_matches_iid(self):
        rng = np.random.RandomState(3)
        x = se.ZeroSumOffset(sigma=0.2, rng=rng)
        steps, cum = [], []
        for _ in range(200000):
            a = x(0.0)
            steps.append(a)
            cum.append(x.applied)
        steps, cum = np.asarray(steps), np.asarray(cum)
        self.assertAlmostEqual(steps.std(), 0.2, delta=0.003)               # same marginal as iid N(0, .2^2)
        self.assertAlmostEqual(cum.std(), 0.2 / math.sqrt(2), delta=0.003)  # bounded offset, not sqrt(t)
        self.assertLess(np.abs(cum).max(), 0.2 / math.sqrt(2) * 6)

    def test_iid_reference_random_walk_grows(self):
        rng = np.random.RandomState(3)
        walk = np.cumsum(0.2 * rng.randn(200000))
        self.assertGreater(np.abs(walk).max(), 20 * 0.2 / math.sqrt(2) * 6)  # orders larger than the offset

    def test_without_clipping_cumulative_equals_offset(self):
        x = se.ZeroSumOffset(sigma=0.2, rng=np.random.RandomState(5))
        for _ in range(1000):
            x(0.1)
            self.assertAlmostEqual(x.applied, x.offset, places=12)

    def test_clipping_respected_and_recovers(self):
        x = se.ZeroSumOffset(sigma=0.2, rng=np.random.RandomState(7))
        for t in range(5000):
            det = 0.999 if t < 2500 else 0.0
            a = x(det)
            self.assertTrue(-1.0 <= a <= 1.0)
        self.assertLess(abs(x.applied - x.offset), 1e-9)                    # back on target once unclipped
        self.assertGreater(x.clipped, 0)

    def test_same_draw_stream_as_upstream_gaussian(self):
        a, b = np.random.RandomState(11), np.random.RandomState(11)
        x = se.ZeroSumOffset(sigma=0.2, rng=a)
        x(0.0)
        self.assertAlmostEqual(x.offset, 0.2 / math.sqrt(2) * b.randn(), places=12)

    def test_scalar_and_size_one_draws_share_the_stream(self):
        # Upstream G_Noise draws np.random.randn(1); the zero-sum process draws randn(): same sequence.
        a, b = np.random.RandomState(13), np.random.RandomState(13)
        self.assertEqual([a.randn() for _ in range(1001)], [float(b.randn(1)[0]) for _ in range(1001)])

    def test_reset_starts_a_fresh_connection(self):
        x = se.ZeroSumOffset(sigma=0.2, rng=np.random.RandomState(17))
        for _ in range(100):
            x(0.0)
        x.reset()
        self.assertEqual((x.offset, x.applied, x.calls, x.clipped), (0.0, 0.0, 0, 0))

    def test_build_modes(self):
        self.assertIsNone(se.build({}))
        self.assertIsInstance(se.build({'exploration_process': 'zero_sum_offset'}), se.ZeroSumOffset)
        with self.assertRaises(ValueError):
            se.build({'exploration_process': 'other'})


if __name__ == '__main__':
    unittest.main()
