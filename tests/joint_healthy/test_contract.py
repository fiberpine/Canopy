"""Offline invariants: actual source parity, temporal credit, replay and gates."""
import ast
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from frameworks.joint_healthy.configuration import configuration, checked_authorization
from frameworks.joint_healthy.core import (Observation, PhysicalCredit, PERIODS, validate, wire,
    period_ms, period_action, softsign_inverse, interval_return, bellman, CALIBRATION, FORWARD_ONLY, JOINT)
from frameworks.joint_healthy.core import calibration_period
from frameworks.joint_healthy.replay import Replay, CalibrationGate
from frameworks.joint_healthy.model import migrate_array, WINDOW


def raw(rate=1000., count=10., dt=.02):
    return np.array([25., rate, count, dt, 0., 10., 1100., 0., 25., 8., 10., 0., 10., 1500., 20.])


def row(period=20., terminal=0., dt=None):
    dt = float(period) / 1000 if dt is None else dt
    reward, discount = interval_return(.9, dt)
    return [np.zeros(82), np.array([0., period_action(period)]), np.array([reward]), np.ones(82),
            np.array([terminal]), np.array([dt]), np.array([discount])]


class NumericalContract(unittest.TestCase):
    def test_joint_calibration_can_use_its_validated40ms_initializer(self):
        cfg=dict(configuration(),calibration_main_ms=40.,period_probe_probability=0.)
        validate(cfg);rng=np.random.RandomState(7)
        for phase in (CALIBRATION,FORWARD_ONLY):
            self.assertEqual(calibration_period(cfg,phase,rng),(40.,False))
        default=dict(configuration(),period_probe_probability=0.)
        self.assertEqual(calibration_period(default,CALIBRATION,rng),(20.,False))
        with self.assertRaises(ValueError):validate(dict(cfg,calibration_main_ms=42.))

    def test_ack_numerator_uses_declared_legacy_estimator_peak(self):
        o=Observation('acked_legacy_peak')
        first=raw();first[4]=1000.;o.observe(first)
        burst=raw(rate=2000.);burst[4]=1000.
        state,utility,meta=o.observe(burst)
        self.assertEqual(meta['max_bw'],1000.)
        self.assertEqual(float(state[63]),2.)
        self.assertEqual(utility,2.)
        self.assertEqual(meta['normalization_source'],'acked_legacy_peak')
        missing=raw();missing[4]=0.
        with self.assertRaises(ValueError):Observation('acked_legacy_peak').observe(missing)
        self.assertEqual(validate(configuration('g3'))['utility_normalization'],'legacy_delivery_peak')
        with self.assertRaises(ValueError):validate(dict(configuration('g3'),utility_normalization='observed_peak'))

    def test_fixed_controls_never_probe_a_different_period(self):
        rng = np.random.RandomState(31)
        for period in (20, 80, 160):
            cfg = configuration('fixed%d' % period)
            for phase in (CALIBRATION, FORWARD_ONLY):
                for _ in range(100): self.assertEqual(calibration_period(cfg, phase, rng), (period, False))

    def test_original_reward_and_frame_source_parity(self):
        source = ROOT / 'frameworks/baseline/code/canopy/rl-module/envwrapper_v2.py'
        node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'TCP_Env_Wrapper')
        method = next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == 'get_state')
        module = ast.Module(body=[method], type_ignores=[]) if sys.version_info >= (3, 8) else ast.Module(body=[method])
        namespace = dict(np=np, sleep=lambda *_: None)
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        class Reference: pass
        env = Reference(); env.params = Reference(); env.params.dict = dict(input_dim=15, delay_margin_coef=1.25)
        env.prev_rid = -1; env.use_normalizer = False; env.local_counter = 0
        env.max_bw = env.max_cwnd = env.max_smp = 0.; env.min_del = 9999999.; env.rohit_randomness_exp = False
        observe = Observation()
        for index, data in enumerate((raw(), raw(800., 0., .08), raw(1200., 8., .04))):
            text = ('%d ' % index + ' '.join(str(x) for x in data) + '\0').encode()
            env.shrmem_r = type('Memory', (), {'read': lambda self, text=text: text})()
            old_state, delay, old_reward, valid = namespace['get_state'](env)
            new, reward, meta = observe.observe(data, 40.)
            np.testing.assert_array_equal(new[:70].reshape(10, 7)[-1], np.asarray(old_state['state'], np.float32))
            self.assertEqual(reward, old_reward)

    def test_zero_sample_is_not_fresh_and_age_is_bounded(self):
        o = Observation(); o.observe(raw())
        state, reward, meta = o.observe(raw(0., 0., .16), 160.)
        self.assertFalse(meta['fresh_rtt']); self.assertEqual(reward, 0.)
        self.assertAlmostEqual(meta['rtt_age_lower_s'], .16); self.assertAlmostEqual(meta['rtt_age_upper_s'], .18)
        self.assertEqual(state[71], 1.)

    def test_bootstrap_cannot_invent_rtt_or_bandwidth(self):
        with self.assertRaises(ValueError): Observation().observe(raw(count=0))
        with self.assertRaises(ValueError): Observation().observe(raw(rate=0))

    def test_actual_gap_is_preserved_instead_of_planned_time(self):
        state, _, _ = Observation().observe(raw(dt=.087123), 40.)
        self.assertAlmostEqual(float(state[70]), math.log1p(.087123 / .02), places=6)
        self.assertAlmostEqual(float(state[78]), math.log1p(.087123 / .02), places=6)
        self.assertAlmostEqual(float(state[79]),math.log1p(20./20.),places=6)
        self.assertAlmostEqual(float(state[80]),math.log1p(25./20.),places=6)
        self.assertAlmostEqual(float(state[81]),math.log1p(.087123/.02),places=6)

    def test_feedback_prior_calibration_follows_actual_frozen_policy(self):
        from frameworks.joint_healthy.core import feedback_initial_period
        cfg=dict(configuration('g4'),period_probe_probability=0.)
        rng=np.random.RandomState(3)
        for fresh,minimum,expected in ((True,200,20.5),(False,20,40),(False,100,100),(False,200,120)):
            self.assertEqual(feedback_initial_period(fresh,minimum),expected)
            for phase in (CALIBRATION,FORWARD_ONLY):
                self.assertEqual(calibration_period(cfg,phase,rng,expected),(expected,False))
        with self.assertRaises(ValueError):calibration_period(cfg,CALIBRATION,rng)
        with self.assertRaises(ValueError):validate(dict(cfg,calibration_main_mode='constant'))

    def test_period_roundtrip_and_finite_initialization(self):
        for p in PERIODS: self.assertAlmostEqual(period_ms(period_action(p)), p)
        self.assertAlmostEqual(softsign_inverse(period_action(40)), -.5)
        for a in (-1., 1.):
            with self.assertRaises(ValueError): softsign_inverse(a)

    def test_wire_strict_state_binding_and_integer_mapping(self):
        self.assertEqual(wire(999, [0., -1.]), b'999 100 1 20000\0')
        self.assertEqual(wire(0, [-1., 1.]), b'0 25 1 160000\0')
        self.assertEqual(wire(0, [.5, 0.], False).split()[1:3], [b'100', b'0'])
        for state, action in ((1000, [0., 0.]), (-1, [0., 0.]), (0, [float('nan'), 0.]), (0, [0.]), (0, [0., 2.])):
            with self.assertRaises(ValueError): wire(state, action)

    def test_base_reward_and_constant_utility_fixed_point(self):
        reward, discount = interval_return(.7, .02)
        self.assertAlmostEqual(reward, .7); self.assertEqual(discount, .995)
        for dt in (.02, .04, .16, .087123):
            reward, discount = interval_return(.7, dt)
            self.assertAlmostEqual(reward + discount * (.7 / .005), .7 / .005)

    def test_discounted_split_merge_and_call_cost_count(self):
        a, da = interval_return(.8, .027); b, db = interval_return(.8, .083)
        whole, dw = interval_return(.8, .11)
        self.assertAlmostEqual(a + da*b, whole); self.assertAlmostEqual(da*db, dw)
        a, da = interval_return(.8, .027, .02); b, _ = interval_return(.8, .083, .02)
        whole, _ = interval_return(.8, .11, .02)
        self.assertAlmostEqual(a + da*b - whole, -.02*da)

    def test_terminal_cannot_bootstrap_another_connection(self):
        self.assertEqual(float(bellman(3., .99, 1., 100., 200.)), 3.)

    def test_migration_only_permits_declared_zero_extensions(self):
        x = np.arange(70*256).reshape(70, 256)
        y = migrate_array(WINDOW+'/fc1/w', x, (82, 256))
        np.testing.assert_array_equal(y[:70], x); self.assertFalse(y[70:].any())
        q = migrate_array('critic/fc2/kernel', np.ones((257, 256)), (258, 256))
        self.assertFalse(q[-1].any())
        with self.assertRaises(ValueError): migrate_array(WINDOW+'/fc2/w', x, (82, 256))

    def test_every_profile_keeps_repairs_and_properties_disabled(self):
        for name in ('joint', 'g0', 'g1', 'g2', 'fixed20', 'fixed80', 'fixed160'):
            cfg = configuration(name); self.assertFalse(cfg['property_reward']); self.assertEqual(cfg['decision_cost'], 0.)
            self.assertEqual(validate(cfg), cfg)
        for key, value in (('unique_online_bn_updates', False), ('property_reward', True), ('updates_per_base20ms', .08), ('gamma', .99)):
            with self.assertRaises(ValueError): validate(dict(configuration(), **{key: value}))

    def test_no_expired_or_implicit_tcp_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'auth.json'; p.write_text(json.dumps(dict(allow_tcp=True, purpose='old', start_epoch=1, deadline_epoch=2)))
            with self.assertRaises(ValueError): checked_authorization(p)
            p.write_text('{}')
            with self.assertRaises(ValueError): checked_authorization(p)
            p.write_text(json.dumps(dict(allow_tcp=True,purpose='invalid',start_epoch=1,deadline_epoch=float('inf'))))
            with self.assertRaises(ValueError): checked_authorization(p)


class RealReplayAndGate(unittest.TestCase):
    def cfg(self, **kw):
        return dict(configuration(), **dict(dict(num_actors=2, memsize=64, batch_size=8, warmup=4,
            minimum_enqueued_per_actor=2, warmup_physical_seconds=.08, calibration_rows_per_actor_period=2,
            minimum_holdout_per_period=2, calibration_updates=5, forward_calibration_updates=3), **kw))

    def test_credit_opens_after_actual_warmup_not_retroactively(self):
        c = PhysicalCredit(2, 4, 2, .08)
        for i in range(4): c.insert(i%2, .02)
        self.assertEqual(c.limit, 0); self.assertEqual(c.warmup_inserted, 4)
        c.insert(0, .16); self.assertEqual(c.limit, 0)
        c.insert(1, .04); self.assertEqual(c.limit, 1)
        for _ in range(100): self.assertEqual(c.limit, 1)

    def test_missing_actor_never_earns_training_credit(self):
        c = PhysicalCredit(2, 4, 2, .08)
        for _ in range(100): c.insert(0, .16)
        self.assertEqual(c.limit, 0); self.assertIsNone(c.warmup_seconds)

    def test_holdout_never_enters_optimizer_and_eviction_is_consistent(self):
        rp = Replay(self.cfg())
        for sequence in range(1, 100):
            for actor in range(2): rp.append(row(PERIODS[sequence%5]), actor, 0, sequence)
        self.assertEqual(sum(map(len, rp.pools)), sum(rp.actor_ids >= 0))
        self.assertEqual(int(rp.coverage.sum()), sum(map(len, rp.pools)))
        for _ in range(5): rp.sample()
        self.assertEqual(rp.sampled.tolist(), [20, 20])
        self.assertTrue(all(len(x) > 0 for x in rp.holdout))

    def test_balanced_remainder_rotates(self):
        rp = Replay(self.cfg(num_actors=3, memsize=90, batch_size=8))
        for sequence in range(1, 21):
            for actor in range(3): rp.append(row(), actor, 0, sequence)
        for _ in range(3): rp.sample()
        self.assertEqual(rp.sampled.tolist(), [8, 8, 8])

    def test_rejects_gaps_duplicate_or_cross_connection_rows(self):
        rp = Replay(self.cfg()); rp.append(row(), 0, 0, 1)
        for episode, sequence in ((0, 1), (0, 3), (1, 1)):
            with self.assertRaises(ValueError): rp.append(row(), 0, episode, sequence)
        self.assertEqual(rp.credit.inserted, 1)

    def test_terminal_episode_reset_is_explicit(self):
        rp = Replay(self.cfg()); rp.append(row(terminal=1), 0, 0, 1); rp.append(row(), 0, 1, 1)
        self.assertEqual(rp.credit.inserted, 2)

    def test_replay_discount_checks_actual_not_planned_duration(self):
        data = row(20., dt=.08); data[6][0] = .995
        with self.assertRaises(ValueError): Replay(self.cfg()).append(data, 0, 0, 1)

    def test_critic_counter_does_not_override_coverage_or_quality(self):
        gate = CalibrationGate(self.cfg()); coverage = np.ones((2, 5), int)*2
        report = [dict(period=i, rows=3, normalized_td_rmse=.01) for i in range(5)]
        self.assertFalse(gate.assess(100, np.zeros((2, 5)), report)['admitted'])
        bad = [dict(x, normalized_td_rmse=2.) for x in report]
        self.assertFalse(gate.assess(100, coverage, bad)['admitted'])
        self.assertEqual(gate.phase, CALIBRATION)
        self.assertFalse(gate.assess(100, coverage, report)['admitted'])
        short=[dict(period=i,blocks=4,normalized_return_rmse=.01) for i in range(5)]
        self.assertTrue(gate.assess(100, coverage, report, short)['admitted']); self.assertEqual(gate.phase, FORWARD_ONLY)
        self.assertFalse(gate.assess(102, coverage, report, short)['admitted'])
        self.assertTrue(gate.assess(103, coverage, report, short)['admitted']); self.assertEqual(gate.phase, JOINT)

    def test_short_returns_are_contiguous_heldout_and_terminal_aware(self):
        replay=Replay(self.cfg())
        for i in range(1,9): replay.append(row(),0,0,i)
        self.assertEqual(len(replay.pools[0]),0)
        first,last,reward,continuation=replay.short_return_batch(0)
        expected,discount=interval_return(.9,.16)
        self.assertAlmostEqual(float(reward[0,0]),expected,places=5)
        self.assertAlmostEqual(float(continuation[0,0]),discount,places=6)
        replay=Replay(self.cfg())
        replay.append(row(),0,0,1);replay.append(row(terminal=1),0,0,2)
        self.assertEqual(float(replay.short_return_batch(0)[3][0,0]),0.)


if __name__ == '__main__': unittest.main()
