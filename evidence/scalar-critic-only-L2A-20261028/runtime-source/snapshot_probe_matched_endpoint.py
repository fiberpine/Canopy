"""Probe saved learner snapshots of an online critic-only run (TF1 learner runtime; read-only; self-contained).

For the proposed online critic-only stage (L2-A): the learner saves checkpoints every few thousand critic updates;
this script restores each one in a newly constructed learner graph and records, on one fixed probe set drawn from the run's
final replay, the same per-class quantities as critic_horizon_replay.py, so that borrowed_signs.py and
exploration_response.py (states.csv of the same run) can decide whether each RTT x delay-class gradient has its own
support before any actor update. Nothing is trained or written to checkpoints.

Probe set (from the final replay, birth order per actor, contiguous segments only). The replay is a ring buffer
(memsize 300,000 in L1), so early rows may be gone; startup rows are therefore recognized from the state itself:
  startup: rows whose history still contains an all-zero older frame (the first 9 transitions of a connection),
           reported separately and never gated (present only if not evicted);
  steady:  per actor up to 8 'penalized' (newest-frame delay factor < 1) and 8 'free' rows without zero frames and
           with a complete 200-step future, evenly spaced.
Per snapshot: Q1 at the behavior action; fitted lambda (Poisson-weighted behavior return + discounted first-snapshot
value) and lambda/(tau*t); dQ1/da and dQ2/da at pi and at the first snapshot's pi (separates actor-BN drift);
twin sign agreement; all by RTT x class, plus the startup stratum.

Inputs: CANOPY_PROBE_SOURCE (registration.json with training_config), CANOPY_PROBE_FRAMEWORK (scalar framework with
orca_backend), --replay (final replay.npz), --snapshots (checkpoint prefixes, e.g. .../step-20000).
--update-map JSON {snapshot basename: complete critic updates} is mandatory. Include
the true update-0 snapshot. Prefixes must be explicit; shell globbing checkpoint
sidecar files does not produce valid TensorFlow checkpoint prefixes.
    python3 snapshot_probe.py --replay replay.npz --snapshots ckpt/step-0 ckpt/step-412288 --update-map updates.json --output probe.json
"""
import argparse, json, math, os, pathlib, re, sys, time
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--replay', required=True)
ap.add_argument('--snapshots', nargs='+', required=True)
ap.add_argument('--update-map', required=True)
ap.add_argument('--per-class', type=int, default=8)
ap.add_argument('--output', required=True)
args = ap.parse_args()
SOURCE = pathlib.Path(os.environ['CANOPY_PROBE_SOURCE'])
F = pathlib.Path(os.environ.get('CANOPY_PROBE_FRAMEWORK', str(SOURCE / 'framework')))
OUTPUT = pathlib.Path(args.output)
assert not OUTPUT.exists()
sys.path[:0] = [str(F), str(F / 'code/runtime/_deps')]
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

cfg = json.loads((SOURCE / 'registration.json').read_text())['training_config']
GAMMA, FUTURE, TAU = float(cfg.get('gamma', .995)), 200, float(cfg.get('tau', .001))
tf.set_random_seed(cfg['seed'])
from orca_backend import create  # noqa: E402
a = create(F, cfg, 'learner')
a.build_learn()
original_vars = tf.global_variables()
saver = tf.train.Saver(original_vars)
g1 = tf.gradients(a.critic_out, a.action)[0]
g2 = tf.gradients(a.critic_out2, a.action)[0]
assert list(original_vars) == tf.global_variables(), 'diagnostics must not add variables'
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
a.assign_sess(sess)

replay = np.load(args.replay)
order = np.argsort(replay['birth'], kind='stable')
data = {k: replay[k][order] for k in ['s0', 'a', 'reward', 's1', 'terminal']}
aid = replay['actor_ids'][order]
n_actors = len(cfg['conditions'])
RTT_OF = [cfg['conditions'][i]['base_rtt_ms'] for i in range(n_actors)]
assert np.all((aid >= 0) & (aid < n_actors)), 'Invalid actor IDs in replay'
assert len(np.unique(replay['birth'])) == len(order), 'Duplicate replay birth IDs'
assert args.per_class > 0


def survival(lam, m_max):
    m = np.arange(m_max + 1)
    if lam <= 0:
        return (m == 0).astype(float)
    pmf = np.exp(-lam + m * math.log(lam) - np.array([math.lgamma(k + 1) for k in m]))
    return np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]]), 0.0, 1.0)


def weighted_return(R, lam):
    return R @ (GAMMA ** np.arange(R.shape[1]) * survival(lam, R.shape[1] - 1))


def fit_lambda(q_mean, R, q0):
    f = lambda lam: float(np.mean(weighted_return(R, lam))) + GAMMA * q0 - q_mean
    lo, hi = 0.0, float(R.shape[1] - 1)
    if f(lo) >= 0:
        return lo
    if f(hi) <= 0:
        return None
    for _ in range(60):
        mid = .5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return .5 * (lo + hi)


PADDED = np.any(np.all(data['s0'][:, :63].reshape(-1, 9, 7) == 0, axis=2), axis=1)   # an all-zero older frame
steady, startup, FR, breaks = [], [], [], 0
for i in range(n_actors):
    rows = np.flatnonzero(aid == i)
    if not len(rows):
        continue
    startup += list(rows[PADDED[rows]])
    cut = [j + 1 for j in range(len(rows) - 1)
           if data['terminal'][rows[j]].any()
           or not np.array_equal(data['s1'][rows[j]], data['s0'][rows[j + 1]])]
    breaks += len(cut)
    edges = [0] + cut + [len(rows)]
    seg = max((rows[edges[k]:edges[k + 1]] for k in range(len(edges) - 1)), key=len)
    r = data['reward'][seg].ravel()
    ok = np.arange(len(seg))
    ok = ok[(~PADDED[seg][ok]) & (ok + FUTURE < len(seg))]
    pen = data['s0'][seg][:, 69] < .999
    for cls in (True, False):
        cand = ok[pen[ok] == cls]
        if len(cand):
            pick = cand[np.linspace(0, len(cand) - 1, min(args.per_class, len(cand)), dtype=int)]
            steady += list(seg[pick])
            FR += [r[p:p + FUTURE] for p in pick]
steady, startup, FR = np.asarray(steady, dtype=np.int64), np.asarray(startup, dtype=np.int64), np.asarray(FR)
assert len(steady), 'No steady probes with complete contiguous future'
P0, PA = data['s0'][steady], data['a'][steady]
RTT = np.asarray([RTT_OF[i] for i in aid[steady]])
PEN = P0[:, 69] < .999
S0 = data['s0'][startup] if len(startup) else np.zeros((0, data['s0'].shape[1]), np.float32)
CLASSES = dict(penalized=PEN, free=~PEN)
probe_classes = {str(r): {c: int(np.sum((RTT == r) & m)) for c, m in CLASSES.items()} for r in sorted(set(RTT_OF))}


def by_class(d):
    out = {}
    for r in sorted(set(RTT_OF)):
        out[str(r)] = {}
        for c, m in CLASSES.items():
            sel = (RTT == r) & m
            out[str(r)][c] = dict(mean=float(d[sel].mean()), positive=float(np.mean(d[sel] > 0)), n=int(sel.sum())) if np.any(sel) else None
    return out


def updates_of(prefix, mapping):
    name = pathlib.Path(prefix).name
    if mapping is None or name not in mapping:
        raise ValueError('Explicit complete critic-update map required; checkpoint names count experience')
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Critic updates must be nonnegative integers')
    return value, 'explicit_critic_update_map'


mapping = json.loads(pathlib.Path(args.update_map).read_text()) if args.update_map else None
snaps = sorted(args.snapshots, key=lambda p: updates_of(p, mapping)[0])
counts = [updates_of(p, mapping)[0] for p in snaps]
assert counts and counts[0] == 0, 'Include true step-0 snapshot to anchor baseline Q and actor'
assert all(b > a for a, b in zip(counts, counts[1:])), 'Duplicate or unordered snapshot updates'
records, fixed_pi, fixed_startup_pi, q0 = [], None, None, None
actor_slots = [a.actor_optimizer.get_slot(v, name) for v in a.actor.train_var()
               for name in a.actor_optimizer.get_slot_names()]
actor_frozen_vars = a.actor.train_var() + actor_slots + list(a.actor_optimizer._get_beta_accumulators())
assert all(v is not None for v in actor_frozen_vars)
actor_baseline = None
started = time.time()
for prefix in snaps:
    reader = tf.train.NewCheckpointReader(prefix)
    assert {v.op.name: list(v.shape) for v in original_vars} == reader.get_variable_to_shape_map(), prefix
    saver.restore(sess, prefix)
    restored = sess.run(original_vars)
    assert all(np.array_equal(value, reader.get_tensor(v.op.name))
               for v, value in zip(original_vars, restored)), 'Restored tensor differs from checkpoint'
    t, basis = updates_of(prefix, mapping)
    assert int(sess.run(a.global_step)) == 2 * t, 'Counter inconsistent with critic-only independent Adam ledger'
    frozen = sess.run(actor_frozen_vars)
    if actor_baseline is None:
        actor_baseline = frozen
    else:
        assert all(np.array_equal(x, y) for x, y in zip(actor_baseline, frozen)), 'Actor weights/Adam state changed'
    pi = sess.run(a.actor_out, {a.s0: P0, a.is_training: False})
    pi_s = sess.run(a.actor_out, {a.s0: S0, a.is_training: False}) if len(S0) else np.zeros((0, 1))
    if fixed_pi is None:
        fixed_pi, fixed_startup_pi = pi.copy(), pi_s.copy()
    q_beh = sess.run(a.critic_out, {a.s0: P0, a.action: PA, a.is_training: False}).ravel()
    if q0 is None:
        q0 = float(q_beh.mean())
    d1, d2 = [x.ravel() for x in sess.run([g1, g2], {a.s0: P0, a.action: pi, a.is_training: False})]
    fd1 = sess.run(g1, {a.s0: P0, a.action: fixed_pi, a.is_training: False}).ravel()
    sd1 = sess.run(g1, {a.s0: S0, a.action: pi_s, a.is_training: False}).ravel() if len(S0) else np.zeros(0)
    lam = fit_lambda(float(q_beh.mean()), FR, q0)
    records.append(dict(update=t, update_basis=basis, snapshot=pathlib.Path(prefix).name,
                        q1_behavior_mean=float(q_beh.mean()), fitted_lambda=lam, model_lambda=TAU * t,
                        propagation_efficiency=(lam / (TAU * t)) if (lam is not None and t > 0) else None,
                        dq1_by_rtt_class=by_class(d1), dq2_by_rtt_class=by_class(d2),
                        fixed_initial_action_dq1_by_rtt_class=by_class(fd1),
                        twin_agree_by_rtt_class=by_class((np.sign(d1) == np.sign(d2)).astype(float)),
                        policy_max_shift_from_first=float(np.max(np.abs(pi - fixed_pi))),
                        startup=None if not len(S0) else dict(
                            n=int(len(startup)), pi_mean=float(pi_s.mean()), dq1_mean=float(sd1.mean()),
                            dq1_positive=float(np.mean(sd1 > 0)),
                            pi_max_shift_from_first=float(np.max(np.abs(pi_s - fixed_startup_pi))))))
out = dict(status='OFFLINE_SNAPSHOT_PROBE_COMPLETE', replay_rows=int(len(order)), contiguity_breaks=breaks,
           steady_probes=int(len(steady)), startup_probes=int(len(startup)), wall_s=time.time() - started,
           result=dict(tau=TAU, records=records, probe_classes=probe_classes,
                       probe_birth=replay['birth'][order][steady].tolist(), probe_actor=aid[steady].tolist(),
                       actor_weights_and_optimizer_unchanged=True, counter_ledger_verified=True,
                       replay_reward_mean=float(np.mean(data['reward'])), probe_future_reward_mean=float(FR.mean())),
           new_TCP_runs=0, source_checkpoints_written=0,
           limitations='Snapshots of one run on one probe set. lambda is a descriptive scalar fit, not an identified '
                       'horizon. Actor BN may evolve. Final-replay probes can be outside an early snapshot\'s '
                       'training support; they are retrospective held-out queries, not a historical online gate. '
                       'Zero older frames are a startup proxy, not proof of episode identity. '
                       'This prototype still requires a real TF1 validation before use.')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(out, indent=1, allow_nan=False))
sess.close()
print(json.dumps(dict(output=str(OUTPUT), snapshots=len(records), wall_s=out['wall_s']), indent=1), flush=True)
