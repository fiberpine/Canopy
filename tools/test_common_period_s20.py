import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'frameworks/mtpjoint'))
from common_period import Credit, Exploration, REWARD, Replay, calibration, discount, encoded_period
from score_s20 import recovery


class S20Contracts(unittest.TestCase):
    def test_fixed_period_with_window_exploration(self):
        seen = set()
        exploration = Exploration(3, [20])
        actions = [exploration.choose(t, lambda p: seen.add(p) or .1)[0]
                   for t in np.arange(0, 120., .02)]
        self.assertEqual(seen, {20})
        self.assertEqual({a[1] for a in actions}, {encoded_period(20)})
        self.assertGreater(np.std(np.asarray(actions)[:, 0]), .1)

    def test_calibration_only_observes_s20(self):
        dt = .022
        gamma = discount(dt)
        duration = [dt] * 2000
        block = dict(connection='one', end_connection='one', duration=duration,
                     reward=[.4*(1-gamma)/.005]*len(duration), period_ms=20)
        result = calibration([block], [20])
        self.assertAlmostEqual(result['K'], 80.)
        self.assertAlmostEqual(result['n0']['20'], 1/(1-gamma))
        self.assertAlmostEqual(result['n0']['160'], 1/(1-discount(.16)))
        with self.assertRaises(ValueError):
            calibration([dict(block, period_ms=40)], [20])

    def test_wrong_period_rejected_before_replay_credit(self):
        replay = Replay('test', active_periods=[20])
        state = np.zeros(149)
        row = dict(actor=0, connection='one', next_connection='one', s0=state, s1=state,
                   action=[.1, encoded_period(40)], reward=.4, duration=.04,
                   discount=discount(.04), period_ms=40, snapshot=1, kind='continuing',
                   behavior='random_segment', apply=1, reward_version=REWARD, reward_checksum='test')
        with self.assertRaises(ValueError):
            replay.append(row)
        self.assertEqual(replay.inserted, 0)
        self.assertEqual(replay.seconds, 0)

    def test_actor_gate_needs_all_actors_at_the_active_period(self):
        replay = Replay('test', active_periods=[20])
        replay.inserted = 100000
        replay.by_actor[:] = 4000
        replay.seconds = 4000.
        replay.warmup_seconds = 1800.
        credit = Credit()
        credit.critic = 5001
        replay.retained_coverage[:, 0] = 128
        self.assertTrue(credit.next(replay))
        replay.retained_coverage[23, 0] = 127
        self.assertFalse(credit.next(replay))

    def test_recovery_rejects_wrong_trace_phase_and_bursty_progress(self):
        gates = dict(startup_seconds=10, post_rise_seconds=5, consecutive_seconds=3,
                     recovery_utilization=.85, phase_tolerance_bytes=2, phase_tolerance_fraction=.03)
        bins = [dict(service_bytes=100, departure_bytes=100) for _ in range(60)]
        event = dict(second=20, expected_service_bytes=[100]*7)
        self.assertEqual(recovery(bins, event, gates), 3)
        bins[21]['departure_bytes'] = 50
        bins[23]['departure_bytes'] = 50
        self.assertIsNone(recovery(bins, event, gates))
        bins[19]['service_bytes'] = 50
        with self.assertRaises(ValueError):
            recovery(bins, event, gates)


if __name__ == '__main__':
    unittest.main()
