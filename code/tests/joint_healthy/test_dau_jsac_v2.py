import tempfile
import unittest
from pathlib import Path
import numpy as np


class TestCanopyDAUJSACv2(unittest.TestCase):
    def test_v2_profile_clean_mainstream_conditions(self):
        from frameworks.joint_healthy.configuration import configuration
        cfg = configuration('canopy-dau-jsac-v2')
        self.assertEqual(len(cfg['conditions']), 24)
        self.assertEqual(len(cfg['eval_conditions']), 24)
        self.assertTrue(cfg['obs_norm_enabled'])
        self.assertTrue(cfg['pid_lagrangian'])
        self.assertAlmostEqual(cfg['peak_window_s'], 10.0)
        # Ensure zero held-out/validation/test or blackout bus trace leaked into training
        for c in cfg['conditions']:
            t = c['trace']
            self.assertNotIn('validation', t)
            self.assertNotIn('test', t)
            self.assertNotIn('heldout', t)
            self.assertNotEqual(t, 'trace-3109898-bus')
            self.assertGreaterEqual(c['queue_packets'], 20)

    def test_sliding_window_peak_recovery(self):
        from frameworks.joint_healthy.core import Observation
        obs_static = Observation('acked_legacy_peak', peak_window_s=0.0)
        obs_win = Observation('acked_legacy_peak', peak_window_s=10.0)
        # 5s at 48 Mbps (rate=48e6/8), then 15s at 12 Mbps (rate=12e6/8)
        def make_raw(rate, dt=0.05):
            x = np.zeros(15, dtype=np.float64)
            x[0] = 5.0; x[1] = rate; x[2] = 10.0; x[3] = dt; x[4] = rate
            x[5] = 50.0; x[6] = rate; x[7] = 0.0; x[8] = 50.0; x[14] = 40.0
            return x
        for _ in range(100):  # 5s at 48
            obs_static.observe(make_raw(6e6, 0.05))
            obs_win.observe(make_raw(6e6, 0.05))
        for _ in range(300):  # 15s at 12
            _, u_st, m_st = obs_static.observe(make_raw(1.5e6, 0.05))
            _, u_wn, m_wn = obs_win.observe(make_raw(1.5e6, 0.05))
        self.assertAlmostEqual(m_st['max_bw'], 6e6)
        self.assertAlmostEqual(m_wn['max_bw'], 1.5e6)
        self.assertGreater(u_wn, 0.95)
        self.assertLess(u_st, 0.30)

    def test_v2_ibp_soundness_pid_and_numpy_equivalence(self):
        import tensorflow as tf
        from frameworks.joint_healthy.configuration import configuration
        from frameworks.joint_healthy.dau_jsac import CanopyDAUJSACModel, NumpyDAUJSACActor

        cfg = configuration('canopy-dau-jsac-v2')
        rng = np.random.RandomState(42)
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)
            model = CanopyDAUJSACModel(cfg, training=True)
            with tf.Session(graph=g) as sess:
                sess.run(tf.global_variables_initializer())
                # Initialize & update obs_norm
                states = rng.randn(64, 82).astype(np.float32) * 2.5 + 0.5
                model.init_obs_norm_from_states(sess, states)
                sess.run(model.update_obs_norm, {model.s: states})

                # 1. Verify 100% IBP soundness through affine RSNorm + joint actor
                lo = states[:16] - 0.15
                hi = states[:16] + 0.15
                w_lo, w_hi, t_lo, t_hi = sess.run(
                    [model.ibp_a_w_lo, model.ibp_a_w_hi, model.ibp_a_t_lo, model.ibp_a_t_hi],
                    {model.ibp_lo_in: lo, model.ibp_hi_in: hi},
                )
                for _ in range(25):
                    u = rng.uniform(0.0, 1.0, size=lo.shape).astype(np.float32)
                    pts = lo + u * (hi - lo)
                    act = sess.run(model.action, {model.s: pts})
                    self.assertTrue(np.all(act[:, 0:1] >= w_lo - 1e-5))
                    self.assertTrue(np.all(act[:, 0:1] <= w_hi + 1e-5))
                    self.assertTrue(np.all(act[:, 1:2] >= t_lo - 1e-5))
                    self.assertTrue(np.all(act[:, 1:2] <= t_hi + 1e-5))

                # 2. Verify PID-Lagrangian update runs cleanly & stays bounded
                sess.run(model.dual_update, {model.s: states})
                nus = sess.run([model.nu_cwnd, model.nu_mtp, model.nu_joint])
                for nu in nus:
                    self.assertTrue(np.isfinite(nu) and 0.0 <= nu <= cfg['dual_max'])

                # 3. Save checkpoint & verify NumpyDAUJSACActor exact equivalence (< 1e-5)
                with tempfile.TemporaryDirectory() as td:
                    prefix = str(Path(td) / 'step-123')
                    model.save(sess, prefix)
                    tf_acts = sess.run(model.action, {model.s: states[:32]})
                    np_actor = NumpyDAUJSACActor(prefix)
                    np_acts = np.array([np_actor.act(states[i]) for i in range(32)], dtype=np.float32)
                    max_diff = float(np.max(np.abs(tf_acts - np_acts)))
                    self.assertLess(max_diff, 1e-5)


if __name__ == '__main__':
    unittest.main()
