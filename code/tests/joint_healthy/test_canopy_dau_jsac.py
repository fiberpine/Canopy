"""Unit, IBP soundness, gradient isolation, and save/restore tests for Canopy-DAU-JSAC."""
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

from frameworks.joint_healthy.dau_jsac import (
    CanopyDAUJSACModel,
    build_canopy_symbolic_boxes,
    default_dau_jsac_config,
)


class CanopyDAUJSACTests(unittest.TestCase):
    def test_joint_ibp_soundness_over_4000_concrete_samples(self):
        import tensorflow as tf
        cfg = default_dau_jsac_config(seed=3, canopy_k_sub=3)
        rng = np.random.RandomState(42)
        base_states = rng.uniform(0.1, 0.9, size=(8, 82)).astype(np.float32)
        base_states[:, 65] = 0.0
        base_states[:, 68] = rng.uniform(0.3, 0.85, size=8).astype(np.float32)
        base_states[:, 69] = np.minimum(1.0, 1.25 * base_states[:, 68])
        base_states[:, 79] = np.log1p(rng.choice([20.0, 50.0, 100.0, 200.0], size=8) / 20.0).astype(np.float32)

        with tf.Graph().as_default():
            tf.set_random_seed(3)
            m = CanopyDAUJSACModel(cfg, training=False)
            boxes = build_canopy_symbolic_boxes(tf, m.s, k_sub=3)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # Perturb mu kernels so network outputs non-trivial nonlinear functions
                for v in m.actor_vars:
                    if v.op.name.endswith('/mu/kernel'):
                        sess.run(tf.assign(v, rng.randn(*v.shape.as_list()).astype(np.float32) * 0.2))
                for fam in ('local', 'bad_delay', 'bad_loss', 'good'):
                    lo_in, hi_in = sess.run(boxes[fam], {m.s: base_states})
                    self.assertTrue(np.all(lo_in <= hi_in + 1e-6))
                    w_lo, w_hi, t_lo, t_hi = sess.run(
                        [m.ibp_a_w_lo, m.ibp_a_w_hi, m.ibp_a_t_lo, m.ibp_a_t_hi],
                        {m.ibp_lo_in: lo_in, m.ibp_hi_in: hi_in},
                    )
                    self.assertTrue(np.all(w_lo <= w_hi + 1e-6))
                    self.assertTrue(np.all(t_lo <= t_hi + 1e-6))
                    # Sample 1000 concrete points inside the boxes and verify 100% inclusion
                    idx = rng.randint(0, len(lo_in), size=1000)
                    u = rng.uniform(0.0, 1.0, size=(1000, 82)).astype(np.float32)
                    pts = lo_in[idx] + u * (hi_in[idx] - lo_in[idx])
                    act = sess.run(m.action, {m.s: pts})
                    w_viol = np.sum((act[:, 0] < w_lo[idx, 0] - 1e-5) | (act[:, 0] > w_hi[idx, 0] + 1e-5))
                    t_viol = np.sum((act[:, 1] < t_lo[idx, 0] - 1e-5) | (act[:, 1] > t_hi[idx, 0] + 1e-5))
                    self.assertEqual(int(w_viol), 0, f'Window IBP unsound on {fam}: {w_viol}')
                    self.assertEqual(int(t_viol), 0, f'Time IBP unsound on {fam}: {t_viol}')

    def test_dau_gradients_primal_dual_and_save_restore(self):
        import tensorflow as tf
        cfg = default_dau_jsac_config(seed=3, batch_size=32, ensemble_size=4)
        rng = np.random.RandomState(7)
        s = rng.uniform(0.1, 0.9, size=(32, 82)).astype(np.float32)
        s[:, 79] = np.log1p(20.0 / 20.0)
        s1 = rng.uniform(0.1, 0.9, size=(32, 82)).astype(np.float32)
        sn = rng.uniform(0.1, 0.9, size=(32, 82)).astype(np.float32)
        act = rng.uniform(-0.8, 0.8, size=(32, 2)).astype(np.float32)
        r1 = rng.uniform(0.5, 1.5, size=(32, 1)).astype(np.float32)
        retn = rng.uniform(4.0, 10.0, size=(32, 1)).astype(np.float32)
        d1 = np.full((32, 1), 0.995, np.float32)
        dn = np.full((32, 1), 0.995 ** 8, np.float32)
        tm = np.zeros((32, 1), np.float32)

        with tf.Graph().as_default():
            tf.set_random_seed(3)
            m = CanopyDAUJSACModel(cfg, training=True)
            # V(s) must have ZERO gradient wrt actor_vars
            v_actor_grads = [g for g in tf.gradients(m.v_loss, m.actor_vars) if g is not None]
            self.assertEqual(len(v_actor_grads), 0)
            # IBP penalties must have non-zero gradient capability wrt actor_vars
            ibp_grads = [g for g in tf.gradients(m.v_cwnd + m.v_mtp + m.v_joint, m.actor_vars) if g is not None]
            self.assertGreater(len(ibp_grads), 0)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(m.target_init)
                feed = {
                    m.s: s, m.executed: act, m.r1: r1, m.s1: s1, m.term1: tm, m.disc1: d1,
                    m.ret_n: retn, m.s_n: sn, m.term_n: tm, m.disc_n: dn,
                }
                for _ in range(5):
                    vl, al = sess.run([m.v_loss, m.adv_loss, m.v_step, m.critic_step], feed)[:2]
                    sess.run([m.actor_step, m.alpha_step, m.dual_update, m.target_update], feed)
                    self.assertTrue(np.isfinite(vl) and np.isfinite(al))

                # Check warmstart + save/restore bitwise parity
                ws_info = m.warmstart_from_teacher(sess, s, np.zeros((32, 1), np.float32), init_period_ms=40.0, steps=40, batch_size=32)
                self.assertLess(ws_info['window_mae'], 0.15)
                pred_before = sess.run(m.action, {m.s: s})
                dig_before = m.digest(sess)
                with tempfile.TemporaryDirectory(prefix='dau-jsac-') as td:
                    prefix = str(Path(td) / 'ckpt')
                    m.save(sess, prefix, counters={'steps': 5})
                    sess.run(tf.global_variables_initializer())
                    m.restore(sess, prefix)
                    pred_after = sess.run(m.action, {m.s: s})
                    self.assertEqual(float(np.max(np.abs(pred_before - pred_after))), 0.0)
                    self.assertEqual(dig_before, m.digest(sess))


if __name__ == '__main__':
    unittest.main()
