#!/usr/bin/env python3
"""Model-free check of the critic's action direction from the randomized zero_sum exploration (NumPy only).

Without clipping, gaps or policy mismatch, zero_sum_offset (rho=0) has action
pi_t + o_t - o_{t-1}; its exploration-only cumulative log4 impulse is o_t.
This is not the actual cwnd deviation from a counterfactual connection: CUBIC,
rounding, clipping and later state-dependent policy outputs alter that trajectory.
If randomized innovations are correctly identified, regression measures their
reduced-form response under the observed compensation/feedback protocol.
State classes must be measured before the current innovation is drawn. Neither
the persistent-action effect nor the critic gradient is identified by this alone.

What it measures and what it does not:
  * beta_k, and the cumulative response sum_{k<=J} beta_k (estimated as one regression of the summed rewards,
    with series intercepts and Bartlett HAC standard errors for correlated scores);
  * 'discounted_persistent' extrapolates to a persistent shift (what dQ/da at a=pi means for a multiplicative
    action followed by a near-constant pi) by linear superposition; it ignores CUBIC's restoring dynamics, so
    only its sign is meant to be compared with the critic's.

Inputs (policy frozen: zero-update smoke, or rows before the first optimizer update):
  states.csv rows: row t holds the executed action, the deterministic action and the reward of the same
  transition and, in delay_term, the delay factor of the state after it (= before action t+1). o_t is the
  cumulative sum of (action - action_no_noise) from the connection start; bounded |o_t| (about 0.5-0.6
  for sigma=.2) confirms the reconstruction.
  --series-npz: arrays o, r, rtt, cls (1 penalized, 0 free, -1 unknown; state before the action) and
  series_id, e.g. written from the pre-update replay by runtime/critic_horizon_replay.py --job E1R.

    python3 exploration_response.py --config train.json --root <framework> --run <run_id> [--max-lag 40]
    python3 exploration_response.py --series-npz e1r-series.npz [--max-lag 40]
    python3 exploration_response.py --self-test
Python 3.6+.
"""
import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

GAMMA = .995
LAGS = (0, 1, 3, 10, 20, 40)
FREE, PENALIZED, UNKNOWN = 0, 1, -1


def offsets(actions, deterministic):
    return np.cumsum(np.asarray(actions, float) - np.asarray(deterministic, float))


def _stack(series, max_lag, cls_value=None):
    xs, ys, segments = [], [], []
    used = 0
    for item in series:
        o, r = np.asarray(item[0], float), np.asarray(item[1], float)
        n = len(o) - max_lag
        if n <= 0:
            continue
        if cls_value is None:
            idx = np.arange(n)
        else:
            idx = np.flatnonzero(np.asarray(item[2])[:n] == cls_value)
        if not len(idx):
            continue
        xs.append(o[idx])
        ys.append(np.stack([r[idx + k] for k in range(max_lag + 1)], axis=1))
        segments.append((used, n, idx))
        used += len(idx)
    if not xs:
        return None, None, []
    return np.concatenate(xs), np.concatenate(ys), segments


def _ols(x, Y, segments=None, hac_lags=0):
    """Series intercepts and Bartlett HAC scores; never join separate connections.

    Class-filtered rows keep their original time coordinates (holes get zero scores).
    Serially correlated outcomes do not by themselves justify iid OLS standard errors.
    """
    if segments is None:
        segments = [(0, len(x), np.arange(len(x)))]
    xc, Yc = np.zeros_like(x), np.zeros_like(Y)
    for start, n, idx in segments:
        sel = slice(start, start + len(idx))
        xc[sel] = x[sel] - x[sel].mean()
        Yc[sel] = Y[sel] - Y[sel].mean(axis=0)
    xx = float(xc @ xc)
    if xx <= 1e-16:
        raise ValueError('No within-series offset variance')
    b = xc @ Yc / xx
    resid = Yc - np.outer(xc, b)
    meat = np.zeros(Y.shape[1])
    for start, n, idx in segments:
        sel = slice(start, start + len(idx))
        scores = np.zeros((n, Y.shape[1]))
        scores[idx] = xc[sel, None] * resid[sel]
        meat += np.sum(scores * scores, axis=0)
        for lag in range(1, min(hac_lags, n - 1) + 1):
            meat += 2 * (1 - lag / float(hac_lags + 1)) * np.sum(scores[lag:] * scores[:-lag], axis=0)
    se = np.sqrt(np.maximum(meat, 0)) / xx
    return b, se


VISIBLE_LAMBDAS = (2, 4, 6, 8, 10, 15, 20, 25, 30, 40)


def survival(lam, m_max):
    """Pr[N >= m], N ~ Poisson(lam), m = 0..m_max (same as value_horizon.survival; inlined so this file runs alone)."""
    m = np.arange(m_max + 1)
    if lam <= 0:
        return (m == 0).astype(float)
    pmf = np.exp(-lam + m * np.log(lam) - np.array([math.lgamma(k + 1) for k in m]))
    return np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]]), 0.0, 1.0)


def visible_weights(lam, max_lag, gamma=GAMMA):
    """Exploratory Poisson weighting of measured innovation responses; not a neural-visibility bound.

    one-step:   sum_k beta_k S(k)                       (S = Pr[Poisson(lam) >= k])
    persistent: sum_j gamma^j S(j) cum_j = sum_k beta_k sum_{j>=k} gamma^j S(j)   (linear-superposition heuristic)
    Rewards beyond max_lag are dropped. S(max_lag + 1) is Poisson tail mass,
    not a bound on omitted reward/response error without an additional magnitude bound.
    """
    S = survival(lam, max_lag + 1)
    one = S[:max_lag + 1]
    disc = gamma ** np.arange(max_lag + 1) * one
    per = np.cumsum(disc[::-1])[::-1]
    return one, per, float(S[max_lag + 1])


def response(series, max_lag=40, gamma=GAMMA, cls_value=None, lambdas=VISIBLE_LAMBDAS):
    """series: (o, r) or (o, r, cls) arrays per connection segment. Returns None if no rows qualify."""
    x, Y, segments = _stack(series, max_lag, cls_value)
    if x is None or len(x) < 10:
        return None
    hac_lags = max(1, 2 * max_lag)
    beta, se = _ols(x, Y, segments, hac_lags)
    lags = [J for J in LAGS if J <= max_lag]
    cum_b, cum_se = _ols(x, np.cumsum(Y, axis=1)[:, lags], segments, hac_lags)
    W = np.zeros((max_lag + 1, len(lags)))
    for c, J in enumerate(lags):
        for k in range(J + 1):
            W[k, c] = sum(gamma ** j for j in range(k, J + 1))   # r_{t+k} counts in every cum_j with k <= j <= J
    per_b, per_se = _ols(x, Y @ W, segments, hac_lags)
    vis = [visible_weights(lam, max_lag, gamma) for lam in lambdas]
    v1_b, v1_se = _ols(x, Y @ np.stack([v[0] for v in vis], axis=1), segments, hac_lags)
    vp_b, vp_se = _ols(x, Y @ np.stack([v[1] for v in vis], axis=1), segments, hac_lags)
    key = lambda lam: ('%g' % lam)
    return dict(samples=int(len(x)), offset_sd=float(np.std(x)), beta=[float(v) for v in beta], se=[float(v) for v in se],
                cumulative={str(J): float(v) for J, v in zip(lags, cum_b)},
                cumulative_se={str(J): float(v) for J, v in zip(lags, cum_se)},
                discounted_persistent={str(J): float(v) for J, v in zip(lags, per_b)},
                discounted_persistent_se={str(J): float(v) for J, v in zip(lags, per_se)},
                visible_onestep={key(l): float(v) for l, v in zip(lambdas, v1_b)},
                visible_onestep_se={key(l): float(v) for l, v in zip(lambdas, v1_se)},
                visible_persistent={key(l): float(v) for l, v in zip(lambdas, vp_b)},
                visible_persistent_se={key(l): float(v) for l, v in zip(lambdas, vp_se)},
                visible_truncated_mass={key(l): v[2] for l, v in zip(lambdas, vis)},
                standard_errors='Bartlett HAC with connection/segment fixed intercepts', hac_lags=hac_lags,
                interpretation='Exploration-innovation response under the observed feedback/compensation policy. '
                               'Persistent-action extrapolation is heuristic, not an identified Q gradient. '
                               'visible_* are Poisson-weighted descriptive response summaries. Fitted lambda does '
                               'not identify a neural critic horizon or exclude within-class gradient learning. '
                               'The truncated Poisson mass is not a response-error bound.')


def split_response(series, max_lag=40):
    counts = {}
    for item in series:
        if len(item) > 2:
            for name, v in (('free', FREE), ('penalized', PENALIZED), ('unknown', UNKNOWN)):
                counts[name] = counts.get(name, 0) + int(np.sum(np.asarray(item[2]) == v))
    out = dict(all=response(series, max_lag), class_counts=counts)
    if counts:
        out['free'] = response(series, max_lag, cls_value=FREE)
        out['penalized'] = response(series, max_lag, cls_value=PENALIZED)
    return out


def read(path):
    with open(str(path), newline='') as f:
        rows = list(csv.DictReader(f))
    a = [float(r['action']) for r in rows]
    p = [float(r['action_no_noise']) for r in rows]
    rew = np.asarray([float(r['reward']) for r in rows])
    cls = np.full(len(rows), UNKNOWN)
    if rows and 'delay_term' in rows[0]:
        d = np.asarray([float(r['delay_term']) for r in rows])
        cls[1:] = np.where(d[:-1] < .999, PENALIZED, FREE)   # state before action t = state after row t-1
    latent=[r.get('exploration_offset','') for r in rows]
    if latent and all(v!='' for v in latent):
        o=np.asarray(latent,float)
        if not np.all(np.isfinite(o)):raise ValueError('Nonfinite logged exploration innovation')
    else:o=offsets(a,p)
    return o, rew, cls


def analyze(root, run, config, max_lag=40):
    groups, recon = {}, []
    for i, cond in enumerate(config['conditions']):
        path = pathlib.Path(root) / 'logs' / ('%s-actor%d' % (run, i)) / 'states.csv'
        if not path.exists():
            continue
        o, r, cls = read(path)
        recon.append(float(np.max(np.abs(o))))
        groups.setdefault(int(cond['base_rtt_ms']), []).append((o, r, cls))
    out = {str(k): split_response(v, max_lag) for k, v in sorted(groups.items())}
    return dict(run=run, actors=sum(len(v) for v in groups.values()), max_abs_offset=max(recon),
                offset_reconstruction_ok=bool(max(recon) < 1.0), by_rtt=out)


def analyze_npz(path, max_lag=40):
    z = np.load(str(path))
    groups = {}
    for sid in np.unique(z['series_id']):
        m = z['series_id'] == sid
        rtt = int(np.unique(z['rtt'][m])[0])
        groups.setdefault(rtt, []).append((z['o'][m], z['r'][m], z['cls'][m]))
    max_abs = float(np.max(np.abs(z['o'])))
    return dict(source=str(path), series=int(len(np.unique(z['series_id']))), rows=int(len(z['o'])), max_abs_offset=max_abs,
                offset_reconstruction_ok=bool(max_abs < 1.0),
                randomized_innovation_identified=bool(z['innovation_identified']) if 'innovation_identified' in z else None,
                identification_note='Bounded cumulative action residuals alone do not establish randomization. '
                                    'Check frozen policy identity, clipping, gaps, rho=0 and pre-action classification.',
                by_rtt={str(k): split_response(v, max_lag) for k, v in sorted(groups.items())})


def self_test():
    rng = np.random.RandomState(4)
    series, split = [], []
    for _ in range(6):
        n = 2000
        o = 0.2 / np.sqrt(2) * rng.randn(n)
        r = 0.8 + 0.02 * rng.randn(n)
        r = r + np.convolve(0.01 * rng.randn(n), np.ones(20) / 20, 'same')   # autocorrelated nuisance
        r[1:] += 0.05 * o[:-1]          # short-term gain one step later
        for k in range(10, 20):
            r[k:] -= 0.02 * o[:-k]      # delayed cost from lag 10 to 19
        pi = rng.uniform(-.01, .01, n)
        a = pi + o - np.concatenate([[0.0], o[:-1]])
        series.append((offsets(a, pi), r))
        cls = rng.randint(0, 2, n)      # state-dependent sign: gain in free states, cost in penalized states
        r2 = 0.7 + 0.02 * rng.randn(n)
        r2[1:] += np.where(cls[:-1] == FREE, 0.05, -0.05) * o[:-1]
        split.append((offsets(a, pi), r2, cls))
    res = response(series, 40)
    assert max(float(np.max(np.abs(o))) for o, _ in series) < 1.0      # reconstructed offsets stay bounded
    assert abs(res['beta'][1] - 0.05) < 0.01 and abs(res['beta'][15] + 0.02) < 0.01, res['beta'][:20]
    assert res['cumulative']['3'] > 0 and res['cumulative']['40'] < 0, res['cumulative']
    assert abs(res['cumulative']['40'] - sum(res['beta'])) < 1e-9          # sum of betas == beta of the sum
    sr = split_response(split, 10)
    assert abs(sr['free']['beta'][1] - .05) < .01 and abs(sr['penalized']['beta'][1] + .05) < .01, (sr['free']['beta'][1], sr['penalized']['beta'][1])
    # visible responses are exact linear functionals of the betas (same regression, summed outcomes)
    beta, cum = np.array(res['beta']), np.cumsum(res['beta'])
    for lam in (2, 15):
        S = survival(lam, 40)
        assert abs(res['visible_onestep']['%g' % lam] - float(beta @ S)) < 1e-9
        assert abs(res['visible_persistent']['%g' % lam] - float((GAMMA ** np.arange(41) * S) @ cum)) < 1e-9
    # a short horizon sees the gain at lag 1 but not the cost at lags 10-19; a long one sees both
    assert res['visible_onestep']['2'] > 0 and res['visible_onestep']['20'] < 0, res['visible_onestep']
    return dict(beta1=res['beta'][1], beta15=res['beta'][15], cumulative=res['cumulative'],
                split_beta1=dict(free=sr['free']['beta'][1], penalized=sr['penalized']['beta'][1]),
                visible_onestep={k: res['visible_onestep'][k] for k in ('2', '20')})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root')
    p.add_argument('--run')
    p.add_argument('--config')
    p.add_argument('--series-npz')
    p.add_argument('--max-lag', type=int, default=40)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test()), indent=1))
        return 0
    if a.series_npz:
        print(json.dumps(analyze_npz(a.series_npz, a.max_lag), indent=1))
        return 0
    config = json.loads(pathlib.Path(a.config).read_text())
    config = config.get('training_config', config)
    print(json.dumps(analyze(a.root, a.run, config, a.max_lag), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
