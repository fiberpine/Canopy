"""How many Adam steps does the Orca actor need to saturate when dQ/da has a consistent sign? (NumPy only)

NumPy re-implementation of the actor: fc1(256)-BN(no scale)-lrelu-fc2(256)-BN-lrelu-dense(1)-tanh,
TF1 defaults (glorot_uniform kernels, zero bias, BN beta 0, moving mean 0 / var 1, eps 1e-3, momentum .99,
leaky_relu alpha .2), small_uniform output head U(+-3e-3). Actor loss uses MOVING statistics (moving arm);
moving stats are updated from the training-mode batch statistics after every step (as in distributed.py).
dQ/da is a constant c for every state (worst case: perfectly consistent direction, any magnitude).
State features are synthetic but in the observed ranges (thr/max_bw, pacing/max_bw, loss, samples/cwnd, dt,
minRTT/sRTT, delay metric; 10-frame history). The result is an order-of-magnitude statement about Adam on this
architecture, not a reproduction of a particular run.

    python3 actor_speed_sim.py            # six configurations x seeds 3/4/5, about two minutes
"""
import numpy as np

def states(rng, n):
    rtt = rng.choice([20, 50, 100, 200], size=n)
    spc = {20: .8, 50: .35, 100: .18, 200: .09}
    f = np.zeros((n, 7))
    f[:, 0] = np.clip(rng.normal(.95, .05, n), 0, 1.2)            # thr/max_bw
    f[:, 1] = np.clip(rng.normal(1.3, .2, n), 0, 10)               # pacing/max_bw
    f[:, 2] = np.clip(rng.normal(.01, .02, n), 0, None)            # 5*loss/max_bw
    f[:, 3] = np.array([spc[r] for r in rtt]) * rng.lognormal(0, .2, n)   # samples/cwnd
    f[:, 4] = rng.normal(.02, .002, n)                              # delta_t (s)
    f[:, 5] = np.clip(rng.normal(.65, .1, n), .2, 1)                # minRTT/sRTT
    f[:, 6] = np.minimum(1, 1.25 * f[:, 5])                         # delay metric
    hist = np.repeat(f[:, None, :], 10, axis=1) + rng.normal(0, .02, (n, 10, 7))
    return hist.reshape(n, 70)

def glorot(rng, fan_in, fan_out):
    lim = np.sqrt(6. / (fan_in + fan_out)); return rng.uniform(-lim, lim, (fan_in, fan_out))

def lrelu(x): return np.where(x > 0, x, .2 * x)
def dlrelu(x): return np.where(x > 0, 1., .2)

class Adam:
    def __init__(self, shapes, lr=1e-4): self.m=[np.zeros(s) for s in shapes]; self.v=[np.zeros(s) for s in shapes]; self.t=0; self.lr=lr
    def step(self, params, grads):
        self.t += 1; b1, b2, eps = .9, .999, 1e-8
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)     # TF1 AdamOptimizer form
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = b1 * m + (1 - b1) * g; v[:] = b2 * v + (1 - b2) * g * g; p -= lr_t * m / (np.sqrt(v) + eps)

def run(c=+0.02, bn_mode='update', calibrate=False, seed=3, steps=400, lr=1e-4):
    rng = np.random.RandomState(seed)
    W1, b1, be1 = glorot(rng, 70, 256), np.zeros(256), np.zeros(256)
    W2, b2, be2 = glorot(rng, 256, 256), np.zeros(256), np.zeros(256)
    W3, b3 = rng.uniform(-.003, .003, (256, 1)), rng.uniform(-.003, .003, 1)
    mu1, var1, mu2, var2 = np.zeros(256), np.ones(256), np.zeros(256), np.ones(256)
    params = [W1, b1, be1, W2, b2, be2, W3, b3]
    opt = Adam([p.shape for p in params], lr)
    probe = states(np.random.RandomState(99), 384)
    def batch_stats(s):
        x1 = s @ W1 + b1; m1, v1 = x1.mean(0), x1.var(0)
        h1 = lrelu((x1 - m1) / np.sqrt(v1 + 1e-3) + be1)
        x2 = h1 @ W2 + b2; return m1, v1, x2.mean(0), x2.var(0)
    if calibrate:
        mu1, var1, mu2, var2 = batch_stats(states(rng, 4096))
    def forward(s):
        x1 = s @ W1 + b1; n1 = (x1 - mu1) / np.sqrt(var1 + 1e-3) + be1; h1 = lrelu(n1)
        x2 = h1 @ W2 + b2; n2 = (x2 - mu2) / np.sqrt(var2 + 1e-3) + be2; h2 = lrelu(n2)
        z = h2 @ W3 + b3; return x1, n1, h1, x2, n2, h2, z
    traj = []
    for t in range(steps + 1):
        pi = np.tanh(forward(probe)[-1]).ravel()
        traj.append((t, pi.mean(), np.mean(np.abs(pi) >= .95)))
        if t == steps: break
        s = states(rng, 512)
        x1, n1, h1, x2, n2, h2, z = forward(s)
        dz = (-c * (1 - np.tanh(z) ** 2)) / len(s)                 # dL/dz with L = -mean Q(s, pi(s))
        gW3 = h2.T @ dz; gb3 = dz.sum(0)
        dh2 = dz @ W3.T; dn2 = dh2 * dlrelu(n2); gbe2 = dn2.sum(0); dx2 = dn2 / np.sqrt(var2 + 1e-3)
        gW2 = h1.T @ dx2; gb2 = dx2.sum(0)
        dh1 = dx2 @ W2.T; dn1 = dh1 * dlrelu(n1); gbe1 = dn1.sum(0); dx1 = dn1 / np.sqrt(var1 + 1e-3)
        gW1 = s.T @ dx1; gb1 = dx1.sum(0)
        opt.step(params, [gW1, gb1, gbe1, gW2, gb2, gbe2, gW3, gb3])
        if bn_mode == 'update':
            m1, v1, m2, v2 = batch_stats(s)
            mu1 += .01 * (m1 - mu1); var1 += .01 * (v1 - var1); mu2 += .01 * (m2 - mu2); var2 += .01 * (v2 - var2)
    return traj

def first(traj, thr):
    for t, m, sat in traj:
        if abs(m) >= thr: return t
    return None

if __name__ == '__main__':
    for label, kw in [('moving BN updated (as L1)', dict(bn_mode='update')),
                      ('moving BN frozen at init', dict(bn_mode='frozen')),
                      ('BN pre-calibrated then updated', dict(bn_mode='update', calibrate=True)),
                      ('as L1, dQ/da=-0.02', dict(bn_mode='update', c=-0.02)),
                      ('as L1, tiny dQ/da=+0.001', dict(bn_mode='update', c=0.001)),
                      ('as L1, actor lr 1e-5', dict(bn_mode='update', lr=1e-5, steps=1500))]:
        res = [run(seed=s, **kw) for s in (3, 4, 5)]
        print('%-34s steps to |mean pi|>=.5: %-16s >=.9: %-16s' % (label, [first(r, .5) for r in res], [first(r, .9) for r in res]))
