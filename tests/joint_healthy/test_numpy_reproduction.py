"""The TF-free reproduction used for the 2026-11-17 critic diagnosis stays correct."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'joint_numpy_reproduction'))
import tfckpt  # noqa: E402
import critic_np as C  # noqa: E402

CHECKPOINT = str(ROOT / 'models/healthy-baseline-seed3/training/step-6013196')
PROBES = ROOT / 'evidence/joint-cost-repair-20261116/real-frozen-probes.npz'


class NumpyReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt = tfckpt.read(CHECKPOINT)
        with np.load(str(PROBES)) as p: cls.states, cls.actions = p['states'], p['actions']

    def test_checkpoint_reader_reproduces_recorded_baseline_actions(self):
        t, p = self.ckpt, 'abstract_actor/abstract_actor/'
        self.assertEqual(t[p + 'fc1/w'].shape, (70, 256)); self.assertEqual(t['critic/fc2/kernel'].shape, (257, 256))
        relu = lambda x: np.where(x > 0, x, .2 * x)
        bn = lambda x, n: (x - t[p + n + '/moving_mean']) / np.sqrt(t[p + n + '/moving_variance'] + 1e-3) + t[p + n + '/beta']
        h = relu(bn(self.states[:, :70] @ t[p + 'fc1/w'] + t[p + 'fc1/b'], 'bn1'))
        h = relu(bn(h @ t[p + 'fc2/w'] + t[p + 'fc2/b'], 'bn2'))
        window = np.tanh(h @ t[p + 'fc1_1/w'] + t[p + 'fc1_1/b'])[:, 0]
        # Rows produced by the unchanged baseline window (G4 initialization) match to float32 precision.
        self.assertGreaterEqual(int(np.sum(np.abs(window - self.actions[:, 0]) < 1e-4)), 529)

    def test_manual_critic_gradients_match_finite_differences(self):
        rng = np.random.RandomState(0)
        critic = C.Critic(self.ckpt, 'critic', rng)
        for k in critic.p: critic.p[k] = critic.p[k] + rng.normal(0, .01, critic.p[k].shape)
        x = self.states[:16].astype(np.float64)
        a = np.column_stack([rng.uniform(-.5, .5, 16), rng.uniform(-1, 1, 16)]); labels = rng.normal(168, 1, (16, 1))
        def loss():
            q, _, _ = critic.forward(x, a); return float(np.mean((q - labels) ** 2))
        q, _, _ = critic.forward(x, a); grads = critic.backward(2 * (q - labels) / 16)
        for name in ('W1', 'W2', 'b3', 'W4', 'W5', 'W6'):
            index = tuple(rng.randint(s) for s in critic.p[name].shape); old = critic.p[name][index]
            critic.p[name][index] = old + 1e-6; up = loss(); critic.p[name][index] = old - 1e-6; down = loss()
            critic.p[name][index] = old
            numeric = (up - down) / 2e-6
            self.assertLess(abs(numeric - grads[name][index]), 1e-3 * max(1., abs(numeric)), name)

    def test_initial_residual_is_zero_and_cost_prior_is_applied(self):
        critic = C.Critic(self.ckpt, 'critic2', np.random.RandomState(1))
        x = self.states[:8].astype(np.float64); a = np.column_stack([np.zeros(8), np.linspace(-1, 1, 8)])
        q, base, residual = critic.forward(x, a)
        np.testing.assert_array_equal(residual, 0.)
        self.assertAlmostEqual(float(self.ckpt['critic2/dense/bias'][0]) - float(critic.p['b3'][0]),
                               .01 / (1 - .995 ** (.0216 / .02)), places=6)


if __name__ == '__main__': unittest.main()
