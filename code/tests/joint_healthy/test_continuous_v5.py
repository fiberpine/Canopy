"""Contract, N-step replay, and backward-compatibility tests for continuous v5."""
import unittest
from unittest import mock
import numpy as np
from frameworks.joint_healthy.configuration import configuration
from frameworks.joint_healthy.continuous import schema, is_v5
from frameworks.joint_healthy.core import validate, period_action, interval_return
from frameworks.joint_healthy.replay import Replay
from frameworks.joint_healthy import identity


class ContinuousV5Contract(unittest.TestCase):
    def test_v5_schema_isolated_and_contract_validated(self):
        v4 = configuration('continuous-v4')
        v5 = configuration('continuous-v5')
        self.assertFalse(is_v5(v4))
        self.assertTrue(is_v5(v5))
        self.assertNotEqual(schema(v4), schema(v5))
        with self.assertRaises(ValueError):
            validate(dict(v5, schema=v4['schema']))
        with self.assertRaises(ValueError):
            validate(dict(v5, conditioned_window_actor_lr=0.))
        with self.assertRaises(ValueError):
            validate(dict(v5, time_barrier_weight=-0.01))
        with self.assertRaises(ValueError):
            validate(dict(v4, critic_period_zero_anchor=True, actor_architecture='independent_v2', actor_recipe='legacy_joint'))

    def test_v5_admission_rejects_v4_receipt(self):
        v5 = configuration('continuous-v5')
        impl = {'fixture': True}
        receipt = dict(
            status='PASS_PREREGISTERED_CONTINUOUS_V4_INITIALIZATION',
            implementation=impl,
            contract=identity.continuous_contract(v5),
            registration='fixture', evidence='fixture',
            offline_graph_pass=True, offline_small_advantage_pass=True,
            initial_real_input_parity_pass=True, same_decision_exploration_pass=True,
            short_tcp_mechanism_pass=True, initial_frozen_grid_health_pass=True,
        )
        with mock.patch.object(identity, 'implementation_identity', return_value=impl):
            with self.assertRaises(ValueError):
                identity.check_admission(receipt, v5)
            receipt['status'] = 'PASS_PREREGISTERED_CONTINUOUS_V5_INITIALIZATION'
            identity.check_admission(receipt, v5)

    def test_nstep_replay_respects_episode_boundaries_and_ring_wrap(self):
        cfg = configuration('continuous-v5')
        cfg.update(num_actors=1, memsize=10, batch_size=1, warmup=1, minimum_enqueued_per_actor=1, warmup_physical_seconds=0.01)
        rp = Replay(cfg)
        zeros = np.zeros(82, np.float32)
        act = np.array([0., period_action(20.)], np.float32)
        r, d = interval_return(1., 0.02, 0.01, 0.995)
        # Episode 0 has 4 transitions (sequence 1..4), then terminal at 4 (bootstrap_last_observation => non-absorbing).
        for seq in range(1, 5):
            s0 = zeros.copy(); s0[0] = float(seq)
            s1 = zeros.copy(); s1[0] = float(seq + 10)
            term = 1.0 if seq == 4 else 0.0
            rp.append([s0, act, [r], s1, [term], [0.02], [d]], 0, 0, seq, teacher_action=0.)
        # Also append episode 1 seq 1 so episode 0 terminal MUST NOT cross into episode 1!
        rp.append([zeros + 99., act, [r], zeros + 199., [0.], [0.02], [d]], 0, 1, 1, teacher_action=0.)
        # Force deterministic sample of index 1 (seq=2, since birth=1 is held out by birth%17? Wait: birth=1 is not %17==0, position=0 for actor 0 ep 0 seq 1!)
        # Actually index 1 (seq 2) is in pool! Sample n_step=4 from index 1 -> walks seq 2 -> seq 3 -> seq 4 (3 steps, stops at ep 0 terminal!).
        rp.pools[0].values = [1]
        rp.pools[0].positions = {1: 0}
        batch = rp.sample(n_step=4)
        expected_r = r + d * r + (d ** 2) * r
        self.assertAlmostEqual(float(batch[2][0, 0]), expected_r, places=5)
        self.assertAlmostEqual(float(batch[6][0, 0]), d ** 3, places=5)
        self.assertEqual(float(batch[3][0, 0]), 14.0)  # s1 of seq 4, NOT episode 1!

    def test_nstep_replay_never_traverses_heldout_slots(self):
        cfg = configuration('continuous-v5')
        cfg.update(num_actors=1, memsize=64, batch_size=1, warmup=1, minimum_enqueued_per_actor=1, warmup_physical_seconds=0.01)
        rp = Replay(cfg)
        zeros = np.zeros(82, np.float32)
        act = np.array([0., period_action(20.)], np.float32)
        r, d = interval_return(1., 0.02, 0.01, 0.995)
        for seq in range(1, 22):
            s0 = zeros.copy(); s0[0] = float(seq)
            s1 = zeros.copy(); s1[0] = float(seq + 100)
            rp.append([s0, act, [r], s1, [0.], [0.02], [d]], 0, 0, seq, teacher_action=0.)
        # birth=17 is index 16 (seq 17) and is held out (actor_ids[16] == -1).
        self.assertEqual(int(rp.actor_ids[16]), -1)
        self.assertEqual(int(rp.next_in_episode[15]), -1)
        self.assertEqual(int(rp.prev_in_episode[17]), -1)
        # Starting at index 14 (seq 15), n_step=5 must walk only seq 15 -> seq 16 (2 steps) and stop before seq 17!
        rp.pools[0].values = [14]
        rp.pools[0].positions = {14: 0}
        batch = rp.sample(n_step=5)
        self.assertAlmostEqual(float(batch[2][0, 0]), r + d * r, places=5)
        self.assertAlmostEqual(float(batch[6][0, 0]), d ** 2, places=5)
        self.assertEqual(float(batch[3][0, 0]), 116.0)


if __name__ == '__main__':
    unittest.main()
