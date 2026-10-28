#!/usr/bin/env python3
"""Can the observation tell RTT groups apart? (NumPy only; no TF.)

If states of different base RTTs were indistinguishable from the 70-dim observation, the critic would be forced to
share its action gradient across RTTs (state aliasing), and more data could not fix the long-RTT sign. If they are
distinguishable, cross-RTT transfer is a property of learning (shared representation, bootstrapping), which own
data and training can in principle override.

Classifier: shrinkage LDA on standardized features, cross-validated by holding out whole actors (fold k holds out
the k-th actor of every RTT group, so no held-out condition is seen in training). Reported: balanced accuracy for
the 4 RTT groups, within each delay class (newest-frame delay factor < 1 = penalized), for R20 vs R200 and for
adjacent pairs, and single-feature accuracies on the newest frame. kNN composition: for each RTT x class, the share
of the k nearest neighbours from OTHER actors that belong to the same RTT group (1 = no local aliasing).

    python3 rtt_separability.py --s-json S-R5-s3.json          # 384 published probes, newest frame only
    python3 rtt_separability.py --replay replay.npz --config registration.json [--max-rows 30000]
    python3 rtt_separability.py --self-test
"""
import argparse
import json
import pathlib
import sys

import numpy as np

NAMES = ['thr/max_bw', 'pacing/max_bw', '5loss/max_bw', 'samples/cwnd', 'dt', 'minRTT/sRTT', 'delay']


def folds_by_actor(actor, rtt):
    if any(len(np.unique(rtt[actor == a])) != 1 for a in np.unique(actor)):
        raise ValueError('Actor identifiers must uniquely identify a single RTT condition')
    fold = np.zeros(len(actor), int)
    for r in np.unique(rtt):
        for rank, a in enumerate(sorted(np.unique(actor[rtt == r]))):
            fold[actor == a] = rank
    return fold


def lda_fit(X, y, shrink=1e-3):
    classes = np.unique(y)
    mu = np.stack([X[y == c].mean(axis=0) for c in classes])
    Xc = X - mu[np.searchsorted(classes, y)]
    cov = Xc.T @ Xc / max(len(X) - len(classes), 1)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)
    W = mu @ inv
    b = -.5 * np.sum(W * mu, axis=1) + np.log(np.array([np.mean(y == c) for c in classes]))
    return classes, W, b


def lda_predict(model, X):
    classes, W, b = model
    return classes[np.argmax(X @ W.T + b, axis=1)]


def balanced_accuracy(y, p):
    if not len(y):
        return None
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def cv_predictions(X, y, fold):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    pred = np.empty_like(y)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        if len(np.unique(y[tr])) < len(np.unique(y)):
            return None
        m, s = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
        model = lda_fit((X[tr] - m) / s, y[tr])
        pred[te] = lda_predict(model, (X[te] - m) / s)
    return pred


def cv_accuracy(X, y, fold):
    pred = cv_predictions(X, y, fold)
    return None if pred is None else balanced_accuracy(y, pred)


def cv_details(X, y, actor, fold):
    pred = cv_predictions(X, y, fold)
    if pred is None:
        return dict(status='INSUFFICIENT_CLASSES_OR_TRAINING_SUPPORT')
    groups = sorted(np.unique(y))
    return dict(status='COMPLETE', balanced_accuracy=balanced_accuracy(y,pred),
                confusion={str(int(a)):{str(int(b)):int(np.sum((y==a)&(pred==b))) for b in groups} for a in groups},
                per_actor={str(int(a)):dict(rows=int(np.sum(actor==a)),rtt=int(y[actor==a][0]),
                                           accuracy=float(np.mean(pred[actor==a]==y[actor==a]))) for a in np.unique(actor)},
                per_fold={str(int(k)):dict(rows=int(np.sum(fold==k)),balanced_accuracy=balanced_accuracy(y[fold==k],pred[fold==k])) for k in np.unique(fold)})


def knn_same_group(X, y, actor, k=10, max_queries=2000, seed=0):
    if k < 1:
        raise ValueError('k must be positive')
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    rng = np.random.RandomState(seed)
    q = np.arange(len(X)) if len(X) <= max_queries else rng.choice(len(X), max_queries, replace=False)
    share = np.empty(len(q))
    for n, i in enumerate(q):
        d = np.sum((Z - Z[i]) ** 2, axis=1)
        d[actor == actor[i]] = np.inf                      # neighbours from other connections only
        eligible = np.flatnonzero(actor != actor[i])
        count = min(k,len(eligible))
        if count == 0:
            share[n] = np.nan
            continue
        nn = eligible[np.argpartition(d[eligible],count-1)[:count]]
        share[n] = np.mean(y[nn] == y[i])
    return q, share


def analyze(X, rtt, actor, delay, newest=None, k=10):
    if not len(X) or not np.isfinite(X).all():
        raise ValueError('Finite nonempty observation matrix required')
    fold = folds_by_actor(actor, rtt)
    pen = delay < .999
    out = dict(rows=int(len(X)), features=int(X.shape[1]), four_way=cv_accuracy(X, rtt, fold), by_class={}, pairs={})
    for name, sel in (('penalized', pen), ('free', ~pen)):
        out['by_class'][name] = dict(rows=int(sel.sum()), four_way=cv_accuracy(X[sel], rtt[sel], fold[sel]))
    groups = sorted(np.unique(rtt))
    for a, b in [(20, 200)] + list(zip(groups[:-1], groups[1:])):
        sel = np.isin(rtt, (a, b))
        out['pairs']['%d-%d' % (a, b)] = cv_accuracy(X[sel], rtt[sel], fold[sel])
    out['cross_validation'] = cv_details(X,rtt,actor,fold)
    pair = np.isin(rtt,(20,200))
    out['r20_r200_cross_validation'] = cv_details(X[pair],rtt[pair],actor[pair],fold[pair])
    if newest is not None:
        out['single_feature_four_way'] = {NAMES[j]: cv_accuracy(newest[:, [j]], rtt, fold) for j in range(newest.shape[1])}
        out['newest_frame_four_way'] = cv_accuracy(newest, rtt, fold)
    q, share = knn_same_group(X, rtt, actor, k)
    out['knn_same_rtt_share'] = {}
    for r in groups:
        for name, sel in (('penalized', pen), ('free', ~pen)):
            m = (rtt[q] == r) & sel[q] & np.isfinite(share)
            out['knn_same_rtt_share']['%d-%s' % (r, name)] = None if not m.any() else float(share[m].mean())
    out['note'] = ('Retrospective linear readout with train-fold-only scaling and held-out actors. '
                   'High aggregate RTT accuracy does not exclude local aliasing, hidden state or critic representation failures; '
                   'low accuracy does not prove absent information. kNN is descriptive and metric-dependent. '
                   'No Q reliability, network performance or online-release decision is identified.')
    return out


def from_s_json(path):
    z = json.loads(pathlib.Path(path).read_text())
    rec = next(r for r in z['records'] if 'per_state' in r)
    ps = rec['per_state']
    F = np.asarray(ps['features'], float)
    rtt, actor = np.asarray(ps['rtt']), np.asarray(ps['actor'])
    k = np.zeros(len(actor), int)
    for a in np.unique(actor):
        idx = np.flatnonzero(actor == a)
        k[idx] = np.arange(len(idx))
    keep = k > 0                                            # drop zero-padded connection starts
    return F[keep], rtt[keep], actor[keep], F[keep, 6], F[keep]


def from_replay(path, config, max_rows=30000, seed=0):
    z = np.load(str(path))
    cfg = json.loads(pathlib.Path(config).read_text())
    cfg = cfg.get('training_config', cfg)
    s0, actor = z['s0'].astype(float), np.asarray(z['actor_ids']).astype(int)
    if max_rows < 1 or np.any((actor < 0)|(actor >= len(cfg['conditions']))):
        raise ValueError('Invalid row budget or actor metadata')
    padded = np.any(np.all(s0[:, :63].reshape(-1, 9, 7) == 0, axis=2), axis=1)
    idx = np.flatnonzero(~padded)
    if len(idx) > max_rows:
        idx = np.sort(np.random.RandomState(seed).choice(idx, max_rows, replace=False))
    rtt = np.asarray([cfg['conditions'][a]['base_rtt_ms'] for a in actor[idx]])
    return s0[idx], rtt, actor[idx], s0[idx, 69], s0[idx, 63:70]


def self_test():
    rng = np.random.RandomState(1)
    n_act, per = 24, 120
    actor = np.repeat(np.arange(n_act), per)
    rtt = np.repeat(np.tile([20, 50, 100, 200], 6), per)
    newest = rng.randn(len(actor), 7)
    newest[:, 3] = 20.0 / (2 * rtt) * np.exp(.15 * rng.randn(len(actor)))   # samples/cwnd ~ 20 ms / sRTT
    newest[:, 6] = np.where(rng.rand(len(actor)) < .7, .6, 1.0)
    out = analyze(newest, rtt, actor, newest[:, 6], newest)
    assert out['pairs']['20-200'] > .99 and out['four_way'] > .9, out
    assert out['single_feature_four_way']['samples/cwnd'] > .9 and out['single_feature_four_way']['thr/max_bw'] < .4
    blind = newest.copy()
    blind[:, 3] = rng.randn(len(actor))
    out2 = analyze(blind, rtt, actor, blind[:, 6], blind)
    assert out2['four_way'] < .4, out2['four_way']
    return dict(informative=dict(four_way=out['four_way'], r20_r200=out['pairs']['20-200']), uninformative=out2['four_way'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--s-json')
    p.add_argument('--replay')
    p.add_argument('--config')
    p.add_argument('--max-rows', type=int, default=30000)
    p.add_argument('--newest-only', action='store_true')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test()), indent=1))
        return 0
    if a.s_json:
        X, rtt, actor, delay, newest = from_s_json(a.s_json)
    else:
        X, rtt, actor, delay, newest = from_replay(a.replay, a.config, a.max_rows)
        if a.newest_only:
            X = newest
    print(json.dumps(analyze(X, rtt, actor, delay, newest, a.k), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
