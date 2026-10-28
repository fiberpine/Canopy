#!/usr/bin/env python3
"""Where did the slow actor move, and is the largest movement learned or extrapolated? (NumPy; reads JSON only.)

Input: first_updates_replay.py v3 outputs (records at release and at the end carry per_state arrays: delta, dq1,
rtt, actor, newest-frame features) and, optionally, the per-actor pre-update row counts (to turn the probe index
into the transition ordinal). Probe k of each actor is linspace(0, n-1, 16)[k]; k = 0 is the first transition of
the connection, whose 70-dim history is zero except the newest frame and whose pacing/max_bw is inflated by the
observed-peak normalizer (clipped at 10).

Reported per run:
  first-transition stratum: mean shift, linear fit of shift on pacing/max_bw (slope, intercept, correlation), the
    spread of the shift across RTT groups, and the critic's release gradient at those states (does the critic ask
    for the move?). A near-perfect linear dependence on one input that ignores RTT, without critic support, is the
    descriptive pattern consistent with input amplification. It does not exclude
    changing critic gradients, BN effects or beneficial startup compensation;
  remaining probes by RTT x delay class (mean shift, release and final dQ1/da, correlation with pacing/max_bw);
  nonfirst probes with pacing/max_bw > 2 and their ordinals (no density-based support claim);
  top-5% composition with and without first transitions.

    python3 startup_extrapolation.py --s-json S-R5-s3.json [S-R5-s4.json ...] [--per-actor input-proof.json]
    python3 startup_extrapolation.py --self-test
Python 3.6+.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

PACING, DELAY = 1, 6           # indices in the 7 newest-frame features


def _corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze_run(z, per_actor=None, release=None):
    recs = {r['iteration']: r for r in z['records'] if 'per_state' in r}
    end = max(recs)
    release = release if release is not None else (z.get('critic_only_updates') or min(recs))
    fin, rel = recs[end]['per_state'], recs.get(release, recs[end])['per_state']
    F = np.asarray(fin['features'], float)
    delta, dq_rel, dq_fin = np.asarray(fin['delta']), np.asarray(rel['dq1']), np.asarray(fin['dq1'])
    rtt, actor = np.asarray(fin['rtt']), np.asarray(fin['actor'])
    k = np.zeros(len(actor), int)
    for a in np.unique(actor):
        idx = np.flatnonzero(actor == a)
        k[idx] = np.arange(len(idx))
    ordinal = None
    if per_actor is not None:
        ordinal = np.asarray([np.linspace(0, per_actor[a] - 1, 16, dtype=int)[kk] for a, kk in zip(actor, k)])
    pacing, pen = F[:, PACING], F[:, DELAY] < .999
    first = k == 0
    fit = np.polyfit(pacing[first], delta[first], 1) if first.sum() >= 3 else [None, None]
    first_by_rtt = {str(r): float(delta[first & (rtt == r)].mean()) for r in np.unique(rtt) if np.any(first & (rtt == r))}
    out = dict(final_update=int(end), release_update=int(release), probes=int(len(delta)),
               first_transition=dict(n=int(first.sum()), mean_shift=float(delta[first].mean()),
                                     pacing_range=[float(pacing[first].min()), float(pacing[first].max())],
                                     shift_on_pacing_slope=None if fit[0] is None else float(fit[0]),
                                     shift_on_pacing_intercept=None if fit[1] is None else float(fit[1]),
                                     shift_pacing_corr=_corr(pacing[first], delta[first]),
                                     mean_shift_by_rtt=first_by_rtt,
                                     release_dq1_mean=float(dq_rel[first].mean()),
                                     release_dq1_positive=float(np.mean(dq_rel[first] > 0)),
                                     share_of_total_abs_shift=float(np.abs(delta[first]).sum() / max(np.abs(delta).sum(), 1e-12))),
               remaining={})
    rest = ~first
    for r in np.unique(rtt):
        cell = {}
        for name, m in (('penalized', rest & (rtt == r) & pen), ('free', rest & (rtt == r) & ~pen)):
            cell[name] = None if not m.any() else dict(n=int(m.sum()), mean_shift=float(delta[m].mean()),
                                                       release_dq1=float(dq_rel[m].mean()), final_dq1=float(dq_fin[m].mean()))
        m = rest & (rtt == r)
        cell['shift_pacing_corr'] = _corr(pacing[m], delta[m])
        out['remaining'][str(r)] = cell
    early = rest & (pacing > 2)
    out['in_distribution_startup'] = dict(n=int(early.sum()), mean_shift=float(delta[early].mean()) if early.any() else None,
                                          ordinals=None if ordinal is None else sorted(int(v) for v in ordinal[early]))
    n_top = max(1, int(math.ceil(.05 * len(delta))))
    top = np.argsort(-np.abs(delta))[:n_top]
    top_rest = np.flatnonzero(rest)[np.argsort(-np.abs(delta[rest]))[:max(1, int(math.ceil(.05 * rest.sum())))]]
    neg_short_pen = lambda idx: int(np.sum((delta[idx] < 0) & pen[idx] & np.isin(rtt[idx], (20, 50))))
    out['top5pct'] = dict(n=int(n_top), first_transitions=int(first[top].sum()), negative_R20_R50_penalized=neg_short_pen(top))
    out['top5pct_without_first'] = dict(n=int(len(top_rest)), negative_R20_R50_penalized=neg_short_pen(top_rest),
                                        negative_R20_any_class=int(np.sum((delta[top_rest] < 0) & (rtt[top_rest] == 20))))
    out['interpretation'] = ('Descriptive associations on legacy first_updates_replay probes; first probe per actor '
                             'is a first transition only for that original full-start probe design. '
                             'Correlation and a release-time gradient do not identify the cause of later policy changes. '
                             'Neither helpful startup compensation nor harmful extrapolation is proved.')
    return out


def analyze(paths, per_actor=None):
    runs = {pathlib.Path(p).stem: analyze_run(json.loads(pathlib.Path(p).read_text()), per_actor) for p in paths}
    corrs = [r['first_transition']['shift_pacing_corr'] for r in runs.values()]
    return dict(runs=runs, first_transition_shift_pacing_corr=corrs,
                note='Fixed historical probes; associations, not proof of beneficial or harmful network behavior.')


def self_test():
    rng = np.random.RandomState(0)
    actors, rtts = np.repeat(np.arange(24), 16), np.repeat(np.tile([20, 50, 100, 200], 6), 16)
    F = rng.uniform(0, 1, (384, 7))
    F[:, PACING] = rng.uniform(1, 1.6, 384)
    F[:, DELAY] = np.where(rng.rand(384) < .7, .6, 1.0)
    k = np.tile(np.arange(16), 24)
    F[k == 0, PACING] = rng.uniform(4, 10, 24)
    F[k == 0, DELAY] = 1.0
    w = -.03                                           # one global direction, amplified by the input magnitude
    delta = w * F[:, PACING] + .01 * rng.randn(384)
    delta[(k > 0) & (rtts == 20) & (F[:, DELAY] < .999)] -= .08
    dq_rel = rng.randn(384) * .05
    dq_rel[k == 0] = .05                               # the critic does not ask for the first-state move
    ps = lambda d: dict(delta=d.tolist(), dq1=dq_rel.tolist(), rtt=rtts.tolist(), actor=actors.tolist(), features=F.tolist())
    z = dict(critic_only_updates=3000, records=[dict(iteration=3000, per_state=ps(np.zeros(384))), dict(iteration=4000, per_state=ps(delta))])
    out = analyze_run(z, per_actor=[1000] * 24)
    ft = out['first_transition']
    assert ft['shift_pacing_corr'] < -.95 and ft['release_dq1_positive'] == 1.0, ft
    assert out['top5pct']['first_transitions'] >= 15, out['top5pct']
    assert out['remaining']['20']['penalized']['mean_shift'] < out['remaining']['200']['penalized']['mean_shift']
    return dict(first_transition=ft, top5pct=out['top5pct'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--s-json', nargs='*')
    p.add_argument('--per-actor')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test()), indent=1))
        return 0
    per = None
    if a.per_actor:
        x = json.loads(pathlib.Path(a.per_actor).read_text())
        per = x.get('per_actor') or x.get('per_actor_inserted')
    print(json.dumps(analyze(a.s_json, per), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
