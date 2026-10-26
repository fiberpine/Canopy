#!/usr/bin/env python3
"""Numerical tests only; no simulated result is labeled real TCP evidence."""
import unittest
import numpy as np
from v31_factual_targets import factual_targets


def trajectory(n=100):
    return dict(s0=np.arange(n)[:, None], s1=np.arange(1, n+1)[:, None],
        a=np.zeros((n, 2)), birth=np.arange(n), actor_ids=np.zeros(n, int),
        reward=np.ones((n, 1)), duration=np.ones((n, 1)),
        discount=np.full((n, 1), .9), terminal=np.zeros((n, 1)))


class Tests(unittest.TestCase):
    def test_discount_and_embargo(self):
        rows, audit = factual_targets(trajectory(), 5)
        for row in rows:
            self.assertAlmostEqual(row['observed_return'], (1-.9**5)/.1)
            self.assertAlmostEqual(row['tail_discount'], .9**5)
            self.assertGreaterEqual(row['horizon_s'], 5)
        train = [x for x in rows if x['partition'] == 'train']
        held = [x for x in rows if x['partition'] == 'heldout']
        self.assertGreaterEqual(min(x['start_s'] for x in held) -
                                max(x['end_s'] for x in train), 5)
        self.assertEqual(audit['discontinuities'], 0)

    def test_connection_boundary(self):
        data = trajectory(); data['s0'][50] = 1000
        rows, audit = factual_targets(data, 5)
        self.assertGreater(audit['discontinuities'], 0)
        self.assertFalse(any(x['index'] < 50 <= x['endpoint'] for x in rows))

    def test_invalid_duration(self):
        data = trajectory(); data['duration'][0] = 0
        with self.assertRaises(ValueError): factual_targets(data)

    def test_actual_duration(self):
        data = trajectory(); data['duration'][:] = 1.3
        rows, _ = factual_targets(data, 5)
        self.assertTrue(all(abs(x['horizon_s']-5.2) < 1e-10 for x in rows))


if __name__ == '__main__': unittest.main()
