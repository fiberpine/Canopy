import types
import unittest
import numpy as np
from e2_support import Handover


class Tests(unittest.TestCase):
    def test_a0_preserves_history_and_actions(self):
        h = Handover('A0', 10)
        env = types.SimpleNamespace(max_bw=123.)
        h.boundary(100, env)
        self.assertEqual(h.action(-.4), -.4)
        np.testing.assert_array_equal(h.history(np.arange(70), np.ones(7)), np.r_[np.arange(7, 70), np.ones(7)])
        self.assertEqual(env.max_bw, 123.)

    def test_a1_keeps_peak_and_history_at_release(self):
        h = Handover('A1', 10)
        env = types.SimpleNamespace(max_bw=123.)
        h.boundary(100, env)
        self.assertEqual(h.action(.3), 0.)
        h.boundary(109.999, env)
        self.assertFalse(h.released)
        h.boundary(110, env)
        self.assertEqual(env.max_bw, 123.)
        self.assertFalse(h.reset_history)
        self.assertEqual(h.action(.3), .3)

    def test_a2_resets_before_boundary_normalization_once(self):
        h = Handover('A2', 10)
        env = types.SimpleNamespace(max_bw=123.)
        h.boundary(100, env)
        h.boundary(110, env)
        self.assertEqual(env.max_bw, 0.)
        env.max_bw = 45.  # Actual wrapper subsequently processes current raw feedback.
        np.testing.assert_array_equal(h.history(np.ones(70), np.arange(7)), np.r_[np.zeros(63), np.arange(7)])
        h.boundary(111, env)
        self.assertEqual(env.max_bw, 45.)
        self.assertFalse(h.reset_history)

    def test_a3_preserves_peak(self):
        h = Handover('A3', 0)
        env = types.SimpleNamespace(max_bw=123.)
        h.boundary(100, env)
        self.assertEqual(env.max_bw, 123.)
        self.assertTrue(h.reset_history)

    def test_a4_floor_not_ceiling(self):
        h = Handover('A4', 10, 3e6)
        env = types.SimpleNamespace(max_bw=0.)
        h.boundary(100, env)
        self.assertEqual(env.max_bw, 3e6)
        env.max_bw = 4e6
        h.boundary(101, env)
        self.assertEqual(env.max_bw, 4e6)


if __name__ == '__main__':
    unittest.main()
