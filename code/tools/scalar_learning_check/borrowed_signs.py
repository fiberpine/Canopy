#!/usr/bin/env python3
"""Exploratory model-weight and sign comparisons; these do not identify gradient provenance.

The ideal fixed-operator model weights lag k by Pr[Poisson(lambda) >= k]
(value_horizon.py). This tool compares that model with selected lag statistics.
This ideal weighting is not a bound on a neural critic's class-specific gradient.
Fitted lambda describes a pooled Q mean, and statistical onset is not a zero-effect
boundary. Legacy labels 'borrowed', 'not_visible' and 'grounded' are retained only
to reproduce the proposal. They do not establish cross-class transfer or Q truth.

Per RTT x delay class (penalized/free):
  E1:     per-lag beta/se -> significant lags (|z| >= z_lag), onset (first), strongest lag (max |z|);
          reference sign = sign of the visible persistent response at the critic's lambda when exploration_response
          wrote visible_* with standard errors and |z| >= z_ref; otherwise the static sign of the strongest
          cumulative response over J in {3,10,20,40} with |z| >= z_ref (0 = not significant).
  Critic: per record the class-mean dQ1/da and dQ2/da, twin agreement, lambda (fitted; tau*t when not fitted).
Per record x cell, coverage = Pr[Poisson(lambda) >= onset] and status:
  unverified            reference sign 0
  not_visible_agree/... coverage < .05: own consequence not yet visible -> a definite sign is borrowed
  partial_agree/...     .05 <= coverage < .5
  grounded_agree/...    coverage >= .5
Gate audit (a proposed rule; passing is NOT authorization or an online-safety certificate): first record with
lambda >= the largest strongest-lag among gate cells (significant lags and a static reference sign) and, for the
last K records, dQ1 and dQ2 signs equal to the reference in every gate cell with twin agreement >= .9 where recorded.

    python3 borrowed_signs.py --critic H1.json --e1 E1R-response.json [--k 5] [--z-lag 3] [--z-ref 2.5]
    python3 borrowed_signs.py --self-test
NumPy only; reads JSON. Python 3.6+.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

RTTS = ('20', '50', '100', '200')
CLASSES = ('penalized', 'free')
GAMMA = .995


def survival(lam, m_max):
    m = np.arange(m_max + 1)
    if lam is None or lam <= 0:
        return (m == 0).astype(float)
    pmf = np.exp(-lam + m * math.log(lam) - np.array([math.lgamma(k + 1) for k in m]))
    return np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]]), 0.0, 1.0)


def coverage(lam, lag):
    if lag is None:
        return None
    return float(survival(lam, lag)[lag])


def lag_structure(cell, z_lag):
    beta, se = np.asarray(cell['beta'], float), np.asarray(cell['se'], float)
    z = np.where(se > 0, beta / np.where(se > 0, se, 1), 0.0)
    sig = [int(k) for k in np.flatnonzero(np.abs(z) >= z_lag)]
    strongest = int(sig[int(np.argmax(np.abs(z[sig])))]) if sig else None
    return dict(significant_lags=sig, onset=sig[0] if sig else None, strongest_lag=strongest,
                strongest_beta=float(beta[strongest]) if sig else None, strongest_z=float(z[strongest]) if sig else None)


def static_reference(cell, z_ref):
    best = None
    for J in ('3', '10', '20', '40'):
        if J in cell.get('cumulative', {}) and cell['cumulative_se'][J] > 0:
            z = cell['cumulative'][J] / cell['cumulative_se'][J]
            if best is None or abs(z) > abs(best[1]):
                best = (J, z)
    if best is None:
        return 0, dict(source='none')
    return (int(np.sign(best[1])) if abs(best[1]) >= z_ref else 0), dict(source='strongest_cumulative', J=best[0], z=best[1])


def reference(cell, lam, z_ref):
    vp, vse = cell.get('visible_persistent'), cell.get('visible_persistent_se')
    if vp and vse and lam is not None:
        k = min(vp, key=lambda s: abs(float(s) - lam))
        z = vp[k] / vse[k] if vse[k] > 0 else 0.0
        return (int(np.sign(vp[k])) if abs(z) >= z_ref else 0), dict(source='visible_persistent', lambda_key=float(k), z=z)
    return static_reference(cell, z_ref)


def own_visible_point(cell, lam, gamma=GAMMA):
    """Point estimate (no SE) of the persistent response visible at lam, from the per-lag betas."""
    beta = np.asarray(cell['beta'], float)
    S = survival(lam, len(beta) - 1)
    return float((gamma ** np.arange(len(beta)) * S) @ np.cumsum(beta))


def sign(v):
    return None if v is None else int(np.sign(v))


def runs(seq):
    out = []
    for u, s in seq:
        if out and out[-1][0] == s:
            out[-1][2] = u
        else:
            out.append([s, u, u])
    return [dict(status=s, first_update=a, last_update=b) for s, a, b in out]


def analyze(critic, e1, k=5, z_lag=3.0, z_ref=2.5, tau=None):
    if k < 1:
        raise ValueError('k must be positive')
    result = critic.get('result', critic)
    records = result['records']
    updates = [rec['update'] for rec in records]
    if not all(b > a for a, b in zip(updates, updates[1:])):
        raise ValueError('Updates must be strictly increasing; repeated records are not new checks')
    tau = tau if tau is not None else result.get('tau', .001)
    by_rtt = e1['by_rtt']
    cells, per_cell = {}, {}
    for r in RTTS:
        for c in CLASSES:
            cell = (by_rtt.get(r) or {}).get(c)
            if not cell:
                continue
            ls = lag_structure(cell, z_lag)
            ref0, why0 = static_reference(cell, z_ref)
            cells[(r, c)] = dict(cell=cell, lags=ls, static_ref=ref0, static_ref_detail=why0)
            per_cell[(r, c)] = []
    gate_cells = [key for key, v in cells.items() if v['lags']['significant_lags'] and v['static_ref'] != 0]
    required = max((cells[key]['lags']['strongest_lag'] for key in gate_cells), default=None)
    rows, ok_hist, gate = [], [], None
    for rec in records:
        lam = rec.get('fitted_lambda')
        lam_source = 'fitted'
        fitted_valid = lam is not None and np.isfinite(lam) and lam >= 0
        if not fitted_valid:
            lam, lam_source = tau * rec['update'], 'nominal_tau_times_updates_NOT_A_BOUND'
        row = dict(update=rec['update'], lam=lam, lam_source=lam_source, cells={})
        ok = fitted_valid and required is not None and lam >= required
        for key, v in cells.items():
            r, c = key
            d1 = (rec['dq1_by_rtt_class'].get(r) or {}).get(c)
            d2 = (rec.get('dq2_by_rtt_class', {}).get(r) or {}).get(c)
            tw = (rec.get('twin_agree_by_rtt_class', {}).get(r) or {}).get(c)
            s1, s2 = sign(d1['mean'] if d1 else None), sign(d2['mean'] if d2 else None)
            ref, why = reference(v['cell'], lam, z_ref)
            cov = coverage(lam, v['lags']['onset'])
            if s1 is None:
                status = 'missing'
            elif ref == 0:
                status = 'unverified'
            else:
                agree = 'agree' if s1 == ref else 'disagree'
                stage = 'not_visible' if (cov is not None and cov < .05) else ('grounded' if (cov is not None and cov >= .5) else 'partial')
                status = '%s_%s' % (stage, agree)
            point = own_visible_point(v['cell'], lam)
            row['cells']['%s-%s' % key] = dict(dq1=d1['mean'] if d1 else None, dq2=d2['mean'] if d2 else None,
                                               twin=(tw or {}).get('mean') if isinstance(tw, dict) else tw,
                                               reference_sign=ref, reference=why, onset_coverage=cov,
                                               gate_reference_sign=v['static_ref'],
                                               own_visible_point=point, own_visible_point_sign=sign(point), status=status)
            per_cell[key].append((rec['update'], status))
            if key in gate_cells:
                twin = row['cells']['%s-%s' % key]['twin']
                # Missing or nonfinite agreement is a failed check, never a pass.
                ok = ok and s1 == v['static_ref'] and s2 == v['static_ref']
                ok = ok and twin is not None and np.isfinite(twin) and .9 <= twin <= 1
        row['literal_proposal_check_pass'] = bool(ok)
        row['fitted_lambda_valid'] = bool(fitted_valid)
        ok_hist.append(ok)
        if gate is None and len(ok_hist) >= k and all(ok_hist[-k:]):
            gate = rec['update']
        rows.append(row)
    summary = {}
    for key, v in cells.items():
        summary['%s-%s' % key] = dict(lag_structure=v['lags'], static_reference_sign=v['static_ref'],
                                      static_reference=v['static_ref_detail'], gate_cell=key in gate_cells,
                                      status_runs=runs(per_cell[key]))
    return dict(tau=tau, k=k, z_lag=z_lag, z_ref=z_ref, gate_cells=['%s-%s' % key for key in gate_cells],
                required_lambda=required, proposed_gate_first_pass=gate, cells=summary, records=rows,
                gate_reference_mode='Static cumulative reference selected over3/10/20/40 on the same evidence',
                gate_is_exploratory=True, deployment_ready=False, online_release_authorized=False,
                unverified_cells=[('%s-%s' % key) for key, v in cells.items() if v['static_ref'] == 0],
                note='Legacy visibility/grounding labels are ideal-model classifications, not identified '
                     'gradient provenance. Lag/horizon selection is retrospective without multiplicity '
                     'correction. Innovation responses and persistent-action Q gradients differ. '
                     'A literal gate pass is neither a necessary safety theorem nor an online release token.')


def self_test():
    # Two classes share features. Class A's cost appears at lag 2, class B's own gain at lag 20.
    lags = 41
    def cell(beta_lags, se=.01):
        beta = np.zeros(lags)
        for kk, b in beta_lags.items():
            beta[kk] = b
        cum = np.cumsum(beta)
        return dict(beta=beta.tolist(), se=[se] * lags,
                    cumulative={J: float(cum[int(J)]) for J in ('3', '10', '20', '40')},
                    cumulative_se={J: se * math.sqrt(int(J) + 1) for J in ('3', '10', '20', '40')})
    e1 = dict(by_rtt={'20': dict(penalized=cell({2: -.2})), '200': dict(free=cell({20: .3, 21: .3}))})
    recs = []
    for u in range(0, 41000, 1000):
        lam = .9 * .001 * u
        borrowed = -1.0 if u >= 3000 else .5          # class B follows class A at first ...
        own = 1.0 if lam >= 20 else 0.0               # ... until its own gain becomes visible
        b = borrowed + 3 * own
        mk = lambda a, bb: {'20': {'penalized': {'mean': a}}, '200': {'free': {'mean': bb}}}
        recs.append(dict(update=u, fitted_lambda=lam if u else None, dq1_by_rtt_class=mk(-1.0 if u >= 2000 else .5, b),
                         dq2_by_rtt_class=mk(-1.0 if u >= 2000 else .5, b),
                         twin_agree_by_rtt_class=mk(1.0, 1.0)))
    out = analyze(dict(result=dict(tau=.001, records=recs)), e1, k=3)
    runs_b = [x['status'] for x in out['cells']['200-free']['status_runs']]
    # B is right by luck, then wrong (borrowed from A), and only right on its own once its lag-20 gain is visible
    assert runs_b[:2] == ['not_visible_agree', 'not_visible_disagree'] and runs_b[-1] == 'grounded_agree', runs_b
    assert [x['status'] for x in out['cells']['20-penalized']['status_runs']][-1] == 'grounded_agree'
    assert out['required_lambda'] == 20, out['required_lambda']
    first = out['proposed_gate_first_pass']
    lam_first = next(r['lam'] for r in out['records'] if r['update'] == first)
    assert first == 25000 and lam_first >= out['required_lambda'], (first, lam_first)
    return dict(gate=first, required_lambda=out['required_lambda'], free200_runs=runs_b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--critic')
    p.add_argument('--e1')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--z-lag', type=float, default=3.0)
    p.add_argument('--z-ref', type=float, default=2.5)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test()), indent=1))
        return 0
    critic = json.loads(pathlib.Path(a.critic).read_text())
    e1 = json.loads(pathlib.Path(a.e1).read_text())
    print(json.dumps(analyze(critic, e1, a.k, a.z_lag, a.z_ref), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
