"""NumPy re-implementation of the v4 joint critic for the known-advantage task.

Mirrors frameworks/joint_healthy/model.py::_critic + continuous.critic_residual at
commit 59cf69a: migrated backbone (fc1 82x256 zero-padded rows, fc2 258x256 with
the time input held at -1), dense bias minus the 21.6ms cost prior, plus the
zero-initialised physical-period residual. Adam matches tf.train.AdamOptimizer.
This is an independent reproduction, not the TF graph itself.
"""
import numpy as np
import tfckpt

GAMMA, COST = .995, .01
ALPHA = .2  # tf.nn.leaky_relu default


def lrelu(x): return np.where(x > 0, x, ALPHA * x)
def dlrelu(x): return np.where(x > 0, 1., ALPHA)


def glorot(rng, fan_in, fan_out):
    limit = np.sqrt(6. / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def features(window, clock, kind):
    if kind == 'v4':      # exactly the committed feature set
        return np.column_stack([window, clock, clock ** 2, window ** 2, window * clock])
    raise ValueError(kind)


class Critic:
    def __init__(self, ckpt, scope, rng, residual_mode='v4', feature_kind='v4', cost_reference_s=.0216):
        self.mode, self.kind = residual_mode, feature_kind
        W1 = np.zeros((82, 256)); W1[:70] = ckpt[scope + '/fc1/kernel']
        W2 = np.zeros((258, 256)); W2[:257] = ckpt[scope + '/fc2/kernel']
        b3 = ckpt[scope + '/dense/bias'].astype(np.float64) - COST / (1. - GAMMA ** (cost_reference_s / .02))
        p = dict(W1=W1, b1=ckpt[scope + '/fc1/bias'].astype(np.float64), W2=W2,
                 b2=ckpt[scope + '/fc2/bias'].astype(np.float64),
                 W3=ckpt[scope + '/dense/kernel'].astype(np.float64), b3=b3)
        nf = 5
        p.update(W4=glorot(rng, 82, 64), b4=np.zeros(64), W5=glorot(rng, 64 + nf, 64), b5=np.zeros(64),
                 W6=np.zeros((64 + nf, 1)), b6=np.zeros(1))
        self.p = {k: np.array(v, dtype=np.float64) for k, v in p.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------------- forward ----------------
    def _residual(self, X, window, clock):
        p = self.p
        a4 = X @ p['W4'] + p['b4']; g4 = lrelu(a4)
        S = np.sum(g4 * g4, 1, keepdims=True); r = np.sqrt(np.maximum(S, 1e-12)); z = g4 / r
        f = features(window, clock, self.kind)
        in5 = np.hstack([z, f]); a5 = in5 @ p['W5'] + p['b5']; h5 = lrelu(a5)
        in6 = np.hstack([h5, f]); q = in6 @ p['W6'] + p['b6']
        return q, (a4, g4, r, z, in5, a5, in6)

    def forward(self, X, action):
        p = self.p
        window = action[:, :1]
        duration = 20. * 8. ** ((action[:, 1:2] + 1.) / 2.)
        clock = (duration - 20.) / 140.
        a1 = X @ p['W1'] + p['b1']; h1 = lrelu(a1)
        in2 = np.hstack([h1, window, -np.ones_like(window)])
        a2 = in2 @ p['W2'] + p['b2']; h2 = lrelu(a2)
        qb = h2 @ p['W3'] + p['b3']
        qr, cache = self._residual(X, window, clock)
        cache_ref = None
        if self.mode == 'advantage':  # residual relative to the same window at 20ms
            qref, cache_ref = self._residual(X, window, np.zeros_like(clock))
            qr = qr - qref
        self.cache = (X, a1, in2, a2, h2, cache, cache_ref)
        return qb + qr, qb, qr

    # ---------------- backward ----------------
    def _residual_backward(self, X, cache, dq, g):
        p = self.p
        a4, g4, r, z, in5, a5, in6 = cache
        g['W6'] += in6.T @ dq; g['b6'] += dq.sum(0)
        dh5 = dq @ p['W6'][:64].T; da5 = dh5 * dlrelu(a5)
        g['W5'] += in5.T @ da5; g['b5'] += da5.sum(0)
        dz = da5 @ p['W5'][:64].T
        dg = (dz - z * np.sum(z * dz, 1, keepdims=True)) / r
        da4 = dg * dlrelu(a4)
        g['W4'] += X.T @ da4; g['b4'] += da4.sum(0)

    def backward(self, dq):
        p = self.p
        X, a1, in2, a2, h2, cache, cache_ref = self.cache
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g['W3'] += h2.T @ dq; g['b3'] += dq.sum(0)
        da2 = (dq @ p['W3'].T) * dlrelu(a2)
        g['W2'] += in2.T @ da2; g['b2'] += da2.sum(0)
        da1 = (da2 @ p['W2'][:256].T) * dlrelu(a1)
        g['W1'] += X.T @ da1; g['b1'] += da1.sum(0)
        self._residual_backward(X, cache, dq, g)
        if cache_ref is not None: self._residual_backward(X, cache_ref, -dq, g)
        return g

    def adam(self, g, lr, lr_residual=None, b1=.9, b2=.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            rate = lr_residual if (lr_residual is not None and k in ('W4', 'b4', 'W5', 'b5', 'W6', 'b6')) else lr
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] * g[k]
            step = rate * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
            self.p[k] -= step * self.m[k] / (np.sqrt(self.v[k]) + eps)

    def step(self, X, action, label, lr=1e-3, lr_residual=None):
        q, _, _ = self.forward(X, action)
        dq = 2. * (q - label) / len(X)
        self.adam(self.backward(dq), lr, lr_residual)
        return float(np.mean((q - label) ** 2))


def encode(ms): return 2. * np.log(ms / 20.) / np.log(8.) - 1.


def task(states, sign, seed_np=100):
    """The committed task: 2 states x 256 log-uniform durations, full batch."""
    rng = np.random.RandomState(seed_np)
    t = np.tile(np.exp(rng.uniform(np.log(20.), np.log(160.), 256)), 2)
    ss = np.repeat(states, 256, axis=0)
    base = np.repeat(np.array([168., 169.]), 256)
    a = np.column_stack([np.zeros(len(t)), encode(t)])
    d = GAMMA ** (t / 20.)
    adv = sign * COST * (GAMMA - d) / (1 - GAMMA)
    return ss, a, (base + adv)[:, None], rng


def truth(sign, queries=np.array([20., 40., 80., 160.])):
    return sign * COST * (GAMMA - GAMMA ** (queries / 20.)) / (1 - GAMMA)
