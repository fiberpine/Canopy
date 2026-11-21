#!/usr/bin/env python3
"""How far ahead the critic can see after t soft-target updates, checked against recorded Q trajectories (NumPy).

Ideal model only. Suppose a fixed-policy tabular critic tracks its TD target exactly,
Q_c = r + gamma * P Q_T, with zero initial target, and the target moves by soft updates,
Q_T <- (1 - tau) Q_T + tau Q_c. Then, exactly,

    Q_c(t) = sum_m (gamma P)^m r * Pr[Binomial(t, tau) >= m]   (~ Poisson(tau*t) for small tau)

because each bootstrap generation needs one target 'renewal'. So:
  * the reward m steps ahead enters with weight Pr[N >= m], N ~ Poisson(tau*t): after t updates the critic sees
    about tau*t steps (tau=.001: 3 steps after 3,000 updates, 30 steps after 30,000);
  * from a near-zero start the value grows by about gamma*tau*r_mean per update (time constant
    1/(tau*(1-gamma)) = 200,000 updates for the converged scale r_mean/(1-gamma));
  * a window change whose cost appears after L steps (about one smoothed RTT) cannot reach dQ/da before
    tau*t ~ L. With 2-BDP queues the queued sRTT is about twice the base RTT for every group, so the sign
    change seen at RTT-20 should repeat at a time proportional to base RTT / tau.
These identities do not establish a neural critic's information horizon or gradient
reliability. Shared features, off-policy data, changing BN/policy, twin clipping and
overlapping target assignments violate the ideal assumptions. A fitted lambda or
normalized Q slope is a descriptive scalar fit, not measured propagation efficiency.

    python3 value_horizon.py --evidence evidence/scalar-first-updates-20261027 [--reward-mean 0.706]
    python3 value_horizon.py --self-test
Python 3.6+.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

GAMMA = .995
TAU = .001


def survival(lam, m_max):
    """Pr[N >= m] for N ~ Poisson(lam), m = 0..m_max."""
    m = np.arange(m_max + 1)
    if lam <= 0:
        return (m == 0).astype(float)
    logpmf = -lam + m * math.log(lam) - np.array([math.lgamma(k + 1) for k in m])
    pmf = np.exp(logpmf)
    return np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]]), 0.0, 1.0)


def survival_binomial(t, tau, m_max):
    """Pr[N >= m] for N ~ Binomial(t, tau), m = 0..m_max (exact counterpart of survival)."""
    m = np.arange(m_max + 1)
    logpmf = np.full(m_max + 1, -np.inf)
    for k in m:
        if k <= t:
            logpmf[k] = (math.lgamma(t + 1) - math.lgamma(k + 1) - math.lgamma(t - k + 1)
                         + k * math.log(tau) + (t - k) * math.log1p(-tau))
    pmf = np.exp(logpmf)
    return np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]]), 0.0, 1.0)


def weighted_return(future_rewards, lam, gamma=GAMMA):
    """future_rewards: (n, M) with column m = r_{t+m}. Returns sum_m gamma^m r_{t+m} Pr[N >= m] per row."""
    R = np.atleast_2d(np.asarray(future_rewards, float))
    M = R.shape[1]
    w = gamma ** np.arange(M) * survival(lam, M - 1)
    return R @ w


def fit_lambda(q_mean, future_rewards, gamma=GAMMA, q0=0.0):
    """Descriptive mean-value fit; constant q0 term is heuristic, not the nonzero-target identity."""
    R = np.atleast_2d(np.asarray(future_rewards, float))
    lo, hi = 0.0, float(R.shape[1] - 1)
    f = lambda lam: float(np.mean(weighted_return(R, lam, gamma))) + gamma * q0 - q_mean
    if f(lo) >= 0:
        return lo
    if f(hi) <= 0:
        return None          # the horizon exceeds the recorded future rewards
    for _ in range(60):
        mid = .5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return .5 * (lo + hi)


def ideal_q(t, r_mean, tau=TAU, gamma=GAMMA, q0=0.0):
    """Scalar (state-aggregated) version of the model, starting from target value q0."""
    decay = (1 - tau * (1 - gamma)) ** t
    return r_mean + gamma * (r_mean / (1 - gamma) * (1 - decay) + q0 * decay)


def predicted_sign_change(base_rtt_ms, ref_rtt_ms=20, ref_updates=2000, tau=TAU, ref_tau=TAU):
    """Scale the observed RTT-20 sign change by base RTT and 1/tau (2-BDP queues: queued sRTT ~ 2x base)."""
    return ref_updates * (base_rtt_ms / ref_rtt_ms) * (ref_tau / tau)


def slope(records, lo, hi):
    pts = [(r['iteration'], r['q1']) for r in records if lo <= r['iteration'] <= hi]
    x, y = np.array(pts, float).T
    b = float(np.polyfit(x, y, 1)[0])
    return b, len(pts)


def first_nonpositive(records, rtt, after=500, until=None):
    for r in records:
        if r['iteration'] >= after and (until is None or r['iteration'] <= until) and r['dq1_by_rtt'][str(rtt)] <= 0:
            return r['iteration']
    return None


def analyze_run(z, reward_mean=None, tau=TAU, gamma=GAMMA):
    rec, cal = z['records'], z['critic_only_updates']
    at = {r['iteration']: r for r in rec}
    b, n = slope(rec, 1000, cal)
    q0 = at[0]['q1']
    out = dict(arm=z['arm'], sampler_seed=z['sampler_seed'], critic_only_updates=cal, slope_points=n,
               q1_slope_per_update=b, implied_reward_mean=b / (gamma * tau),
               q1_at={str(k): at[k]['q1'] for k in (0, 1000, 2000, cal, z['iterations']) if k in at},
               rtt20_first_nonpositive_dq1=first_nonpositive(rec, 20, until=cal),
               dq1_by_rtt_at_release=at[cal]['dq1_by_rtt'], dq1_by_rtt_at_end=rec[-1]['dq1_by_rtt'])
    if reward_mean is not None:
        out['ideal_slope_per_update'] = gamma * tau * reward_mean
        out['tracking_efficiency'] = b / (gamma * tau * reward_mean)
        out['ideal_q1_at_release'] = ideal_q(cal, reward_mean, tau, gamma, q0)
    return out


def analyze_evidence(root, reward_mean=None):
    runs, seen = [], set()
    for path in sorted(pathlib.Path(root).glob('collected/*/audit/R*.json')):
        z = json.loads(path.read_text())
        if not z.get('critic_only_updates'):
            continue
        key = (z['sampler_seed'], z.get('redraw_seed'))
        row = analyze_run(z, reward_mean)
        row['job'] = path.stem
        row['duplicate_of_earlier_critic_phase'] = key in seen
        seen.add(key)
        runs.append(row)
    independent = [r for r in runs if not r['duplicate_of_earlier_critic_phase']]
    summary = dict(independent_critic_phases=len(independent),
                   implied_reward_mean_range=[min(r['implied_reward_mean'] for r in independent),
                                              max(r['implied_reward_mean'] for r in independent)],
                   rtt20_first_nonpositive=[r['rtt20_first_nonpositive_dq1'] for r in independent],
                   predicted_sign_change_tau001={str(k): predicted_sign_change(k) for k in (20, 50, 100, 200)},
                   predicted_sign_change_tau005={str(k): predicted_sign_change(k, tau=.005) for k in (20, 50, 100, 200)},
                   poisson_weight_at_lag={str(lam): {str(m): float(survival(lam, 40)[m]) for m in (1, 2, 3, 5, 10, 20, 30)}
                                          for lam in (1, 2, 3, 4, 10, 20, 30)})
    return dict(runs=runs, summary=summary,
                limitations='Slope uses the probe-state mean at the deterministic action; the reward mean of the replay is '
                            'not in the published evidence (H1 records it). The sign-change scaling assumes the queued sRTT '
                            'is proportional to base RTT; free (unqueued) states need not change sign at all.')


def self_test():
    # 1) exact identity on a deterministic chain with perfect tracking
    tau, gamma, t, K = .01, .99, 400, 60
    rng = np.random.RandomState(0)
    r = rng.uniform(0, 1, K)
    QT = np.zeros(K + 1)
    for _ in range(t):
        Qc = r + gamma * QT[1:]
        QT[:K] = (1 - tau) * QT[:K] + tau * Qc
    Qc = r + gamma * QT[1:]
    w = gamma ** np.arange(K) * survival_binomial(t, tau, K - 1)
    exact = float(np.dot(w, r))
    assert abs(Qc[0] - exact) < 1e-9, (Qc[0], exact)
    approx = float(weighted_return(r.reshape(1, -1), tau * t, gamma)[0])
    assert abs(approx - exact) / exact < .02, (approx, exact)
    # 2) value growth ~ gamma*tau*r_mean per update and lambda recovery
    assert abs(ideal_q(3000, .7) - .7 * (1 + .995 * 3)) < .02
    R = np.full((5, 200), .7)
    lam = fit_lambda(float(np.mean(weighted_return(R, 7.5))), R)
    assert abs(lam - 7.5) < 1e-6, lam
    # 3) scaling of the sign change
    assert predicted_sign_change(200) == 20000 and predicted_sign_change(200, tau=.005) == 4000
    return dict(chain_exact=exact, poisson_approx=approx, lambda_recovered=lam)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evidence')
    p.add_argument('--reward-mean', type=float)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test()), indent=1))
        return 0
    print(json.dumps(analyze_evidence(a.evidence, a.reward_mean), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
