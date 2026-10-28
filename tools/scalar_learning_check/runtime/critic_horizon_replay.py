"""Zero-TCP checks of what the critic knows after N updates (TF1 learner runtime; fresh process per job; self-contained).

Hypothesis under test (tools/scalar_learning_check/value_horizon.py has the derivation and the fit to the
published R3/R5 records): with soft targets the reward m steps ahead enters the critic with weight
Pr[N >= m], N ~ Poisson(tau*t). At tau=.001 the 3,000-update critic sees about 3 steps (60 ms). The
recorded Q1 grew linearly at 5.9-6.2e-4 per update (= gamma*tau*r_mean for r_mean 0.59-0.62), and the
RTT-20 dQ1/da first turned non-positive at 1,820-1,980 updates. With 2-BDP queues the queued sRTT is about
twice the base RTT in every group, so in queued ('penalized': sRTT > 1.25 minRTT) states the same sign
change should appear near 5k (RTT-50), 10k (RTT-100) and 20k (RTT-200) updates at tau=.001, and five times
earlier at tau=.005. In unqueued ('free') states a positive dQ1/da can be correct and need not change sign.

Jobs (same environment variables, inputs and sampler as first_updates_replay.py; the actor is never trained):
  H1   critic-only at the original tau=.001 for 40,000 updates (actor BN statistics update as in R3/R5)
  H2   critic-only at tau=.005 for 10,000 updates (critic target ops rebuilt; actor/BN unchanged)
  E1R  no training: pi(s0) of the step-0 actor on the pre-update rows, zero_sum offsets o_t reconstructed per
       actor, written as <output>.npz for tools/scalar_learning_check/exploration_response.py --series-npz
Recorded for H1/H2 at 0, 100, 300 and every 1,000 updates on the probe states (16 per actor, each with a
complete 200-step future in the replay): Q1 at the behavior action and at pi; the Poisson-weighted return
predicted by the model, the fitted lambda and the propagation efficiency lambda/(tau*t); the H-step return
(H in 1..200) closest to Q1; dQ1/da and dQ2/da by RTT and by RTT x delay class; twin sign agreement.
Nothing is written to checkpoints. Output: one JSON per job (plus the .npz for E1R).
"""
import argparse, json, math, os, pathlib, random, sys, time
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--job', choices=['H1', 'H2', 'E1R'], required=True)
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--updates', type=int)
ap.add_argument('--output', required=True)
args = ap.parse_args()
SOURCE = pathlib.Path(os.environ['CANOPY_FIRST_SOURCE'])
F = pathlib.Path(os.environ.get('CANOPY_FIRST_FRAMEWORK', str(SOURCE / 'framework')))
OUTPUT = pathlib.Path(args.output)
assert not OUTPUT.exists()
sys.path[:0] = [str(F), str(F / 'code/runtime/_deps')]
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

cfg = json.loads((SOURCE / 'registration.json').read_text())['training_config']
rid = cfg['run_id']
D = pathlib.Path(os.environ.get('CANOPY_FIRST_CHECKPOINT_DIR', str(F / 'checkpoints' / rid)))
first = json.loads((SOURCE / 'first-update-coverage.json').read_text())
FIRST_INSERTED = int(first['inserted'])
GAMMA = .995
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200]
FUTURE = 200
tf.set_random_seed(cfg['seed'])
from orca_backend import create  # noqa: E402
a = create(F, cfg, 'learner')
a.build_learn()
original_vars = tf.global_variables()
saver = tf.train.Saver(original_vars)
bn_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'actor' in v.name and 'target' not in v.name]
assert bn_ops
g1 = tf.gradients(a.critic_out, a.action)[0]
g2 = tf.gradients(a.critic_out2, a.action)[0]
tau = {'H1': .001, 'H2': .005}.get(args.job)
if tau is not None and tau != .001:
    pairs = list(zip(a.target_critic.train_var(), a.critic.train_var())) + list(zip(a.target_critic2.train_var(), a.critic2.train_var()))
    target_ops = [tf.assign(t, tau * v + (1 - tau) * t) for t, v in pairs] + list(a.target_actor_update_op)
else:
    target_ops = None
assert list(original_vars) == tf.global_variables(), 'diagnostics must not add variables'
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
a.assign_sess(sess)
saver.restore(sess, str(D / 'step-0'))
reader = tf.train.NewCheckpointReader(str(D / 'step-0'))
assert {v.op.name: list(v.shape) for v in original_vars} == reader.get_variable_to_shape_map()
assert all(np.array_equal(value, reader.get_tensor(v.op.name))
           for v, value in zip(original_vars, sess.run(original_vars)))

replay = np.load(str(D / 'replay.npz'))
keep = np.flatnonzero(replay['birth'] < FIRST_INSERTED)
assert len(keep) == FIRST_INSERTED and np.array_equal(replay['birth'][keep], np.arange(FIRST_INSERTED))
data = {k: replay[k][keep] for k in ['s0', 'a', 'reward', 's1', 'terminal']}
aid = replay['actor_ids'][keep]
pools = [np.flatnonzero(aid == i) for i in range(24)]
assert [len(p) for p in pools] == first['per_actor_inserted']
RTT_OF = [cfg['conditions'][i]['base_rtt_ms'] for i in range(24)]


# Same functions as tools/scalar_learning_check/value_horizon.py (tested there); inlined so this file runs alone.
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


def contiguous_runs(rows):
    """Split one actor's birth-ordered rows wherever s1(t) != s0(t+1)."""
    breaks = [j + 1 for j in range(len(rows) - 1)
              if data['terminal'][rows[j]].any()
              or not np.array_equal(data['s1'][rows[j]], data['s0'][rows[j + 1]])]
    edges = [0] + breaks + [len(rows)]
    return [rows[edges[k]:edges[k + 1]] for k in range(len(edges) - 1)], len(breaks)


runs, gaps = [], 0
for i in range(24):
    parts, n = contiguous_runs(pools[i])
    runs.append(parts)
    gaps += n


def exploration_series():
    """E1R: o_t = cumsum(a_t - pi(s0_t)) per contiguous segment; delay class of s0 (state before the action)."""
    assert cfg.get('exploration_process') == 'zero_sum_offset'
    assert float(cfg.get('exploration_offset_rho', 0.0)) == 0.0
    pi = np.concatenate([sess.run(a.actor_out, {a.s0: data['s0'][k:k + 4096], a.is_training: False}).ravel()
                         for k in range(0, FIRST_INSERTED, 4096)])
    cols = dict(o=[], r=[], rtt=[], cls=[], series_id=[], actor=[], a=[], pi=[])
    for i in range(24):
        for s, rows in enumerate(runs[i]):
            act = data['a'][rows].ravel()
            o = np.cumsum(act - pi[rows])
            delay = data['s0'][rows][:, 69]
            cols['o'].append(o)
            cols['r'].append(data['reward'][rows].ravel())
            cols['rtt'].append(np.full(len(rows), RTT_OF[i]))
            cols['cls'].append(np.where(delay < .999, 1, 0))
            cols['series_id'].append(np.full(len(rows), 1000 * i + s))
            cols['actor'].append(np.full(len(rows), i))
            cols['a'].append(act)
            cols['pi'].append(pi[rows])
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    npz = OUTPUT.with_suffix('.npz')
    assert not npz.exists()
    npz.parent.mkdir(parents=True, exist_ok=True)
    arrays['innovation_identified'] = np.asarray(gaps == 0 and not np.any(np.abs(arrays['a']) >= .999))
    np.savez_compressed(str(npz), **arrays)
    sigma = float(cfg.get('exploration_std', .2))
    first_offsets = [float(arrays['o'][arrays['series_id'] == 1000 * i][0]) for i in range(24)]
    clipped = bool(np.any(np.abs(arrays['a']) >= .999))
    return dict(npz=str(npz), rows=int(len(arrays['o'])), segments=int(len(np.unique(arrays['series_id']))),
                contiguity_breaks=gaps, max_abs_offset=float(np.max(np.abs(arrays['o']))),
                offset_sd=float(np.std(arrays['o'])), expected_offset_sd=sigma / math.sqrt(2),
                clipped_fraction=float(np.mean(np.abs(arrays['a']) >= .999)),
                penalized_fraction_by_rtt={str(r): float(np.mean(arrays['cls'][arrays['rtt'] == r])) for r in (20, 50, 100, 200)},
                first_offset_sd=float(np.std(first_offsets)),
                randomized_offset_identified=bool(gaps == 0 and not clipped),
                note='Without clipping or gaps this is the randomized innovation offset, not an actual cwnd '
                     'counterfactual. With clipping/gaps reconstruction alone does not identify innovations. '
                     'Randomized response includes compensation, CUBIC and future feedback; it is not dQ/da.')


def future_rewards(rows):
    r = data['reward'][rows].ravel()
    n = len(r) - FUTURE + 1
    return np.stack([r[k:k + n] for k in range(FUTURE)], axis=1) if n > 0 else np.zeros((0, FUTURE))


# Probe: 16 rows per actor whose 200-step future is in the same contiguous run.
probe, FR = [], []
for i in range(24):
    longest = max(runs[i], key=len)
    fr = future_rewards(longest)
    assert len(fr) >= 16, ('actor', i, 'needs 215 contiguous rows', len(longest))
    pick = np.linspace(0, len(fr) - 1, 16, dtype=int)
    probe += list(longest[pick])
    FR.append(fr[pick])
probe = np.asarray(probe)
FR = np.concatenate(FR)
P0, PA = data['s0'][probe], data['a'][probe]
RTT = np.asarray([RTT_OF[i] for i in aid[probe]])
NEWEST = P0[:, 63:70]          # thr/max_bw, pacing/max_bw, 5*loss/max_bw, samples/cwnd, dt, minRTT/sRTT, delay factor
PEN = NEWEST[:, 6] < .999
SRTT_STEPS = RTT / np.maximum(NEWEST[:, 5], 1e-3) / 20.0
returns = {H: FR[:, :H] @ (GAMMA ** np.arange(H)) for H in HORIZONS}
CLASSES = dict(penalized=PEN, free=~PEN)
probe_classes = {str(r): {c: dict(n=int(np.sum((RTT == r) & m)),
                                  median_srtt_steps=float(np.median(SRTT_STEPS[(RTT == r) & m])) if np.any((RTT == r) & m) else None)
                          for c, m in CLASSES.items()} for r in (20, 50, 100, 200)}
Q0 = [None]
FIXED_PI = sess.run(a.actor_out, {a.s0: P0, a.is_training: False})
PROGRESS = OUTPUT.with_suffix('.progress.jsonl')
assert not PROGRESS.exists()


def by_class(d):
    out = {}
    for r in (20, 50, 100, 200):
        out[str(r)] = {}
        for c, m in CLASSES.items():
            sel = (RTT == r) & m
            out[str(r)][c] = dict(mean=float(d[sel].mean()), positive=float(np.mean(d[sel] > 0))) if np.any(sel) else None
    return out


def critic_record(t):
    pi = sess.run(a.actor_out, {a.s0: P0, a.is_training: False})
    q_beh = sess.run(a.critic_out, {a.s0: P0, a.action: PA, a.is_training: False}).ravel()
    q_pi, d1, d2 = sess.run([a.critic_out, g1, g2], {a.s0: P0, a.action: pi, a.is_training: False})
    fixed_d1 = sess.run(g1, {a.s0: P0, a.action: FIXED_PI, a.is_training: False}).ravel()
    d1, d2 = d1.ravel(), d2.ravel()
    if Q0[0] is None:
        Q0[0] = float(q_beh.mean())   # the target starts as a copy of this critic
    gap = {H: float(np.mean(np.abs(q_beh - returns[H]))) for H in HORIZONS}
    corr = {}
    for H in HORIZONS:
        c = float(np.corrcoef(q_beh, returns[H])[0, 1]) if np.std(returns[H]) > 0 and np.std(q_beh) > 0 else float('nan')
        corr[H] = c if np.isfinite(c) else None
    lam_model = tau * t
    lam_fit = fit_lambda(float(q_beh.mean()), FR, Q0[0])
    return dict(update=t, q1_behavior_mean=float(q_beh.mean()), q1_pi_mean=float(np.mean(q_pi)),
                model_lambda=lam_model, model_q1=float(np.mean(weighted_return(FR, lam_model))) + GAMMA * Q0[0],
                fitted_lambda=lam_fit, propagation_efficiency=(lam_fit / lam_model) if (lam_fit is not None and t > 0) else None,
                best_matching_horizon=min(HORIZONS, key=lambda H: gap[H]), mean_abs_gap_by_H={str(H): gap[H] for H in HORIZONS},
                corr_by_H={str(H): corr[H] for H in HORIZONS},
                dq1_by_rtt={str(r): float(d1[RTT == r].mean()) for r in (20, 50, 100, 200)},
                dq2_by_rtt={str(r): float(d2[RTT == r].mean()) for r in (20, 50, 100, 200)},
                dq1_by_rtt_class=by_class(d1), dq2_by_rtt_class=by_class(d2),
                fixed_initial_action_dq1_by_rtt_class=by_class(fixed_d1),
                policy_max_shift_from_initial=float(np.max(np.abs(pi - FIXED_PI))),
                twin_agree_by_rtt_class=by_class((np.sign(d1) == np.sign(d2)).astype(float)),
                twin_agree=float(np.mean(np.sign(d1) == np.sign(d2))))


def critic_job(updates):
    rng, calls = random.Random(args.seed), [0]

    def batch():
        base, rem = divmod(512, 24)
        extra = {(calls[0] * rem + j) % 24 for j in range(rem)}
        idx = []
        for i, p in enumerate(pools):
            idx += rng.sample(list(p), base + int(i in extra))
        calls[0] += 1
        idx = np.asarray(idx)
        return {a.s0: data['s0'][idx], a.action: data['a'][idx], a.reward: data['reward'][idx], a.s1: data['s1'][idx],
                a.terminal: data['terminal'][idx], a.is_training: True, a.is_concrete: True,
                a.s0_delta: np.zeros_like(data['s0'][idx])}
    actor_before = sess.run(a.actor.train_var())
    records = []
    def record(t):
        row = critic_record(t)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, float)), row
        with PROGRESS.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        records.append(row)
    record(0)
    for t in range(1, updates + 1):
        fd = batch()
        sess.run(a.critic_train_op, fd)
        sess.run(bn_ops, fd)
        if target_ops is None:
            a.target_update()
        else:
            sess.run(target_ops)
        if t % 1000 == 0 or t in (100, 300, updates):
            record(t)
    assert all(np.array_equal(x, y) for x, y in zip(actor_before, sess.run(a.actor.train_var()))), 'actor must stay frozen'
    sign_change = {}
    for r in (20, 50, 100, 200):
        for c in CLASSES:
            hist = [(rec['update'], (rec['dq1_by_rtt_class'][str(r)][c] or {}).get('mean')) for rec in records if rec['update'] >= 500]
            sign_change['%d-%s' % (r, c)] = next((u for u, v in hist if v is not None and v <= 0), None)
    predicted = {str(r): 2000.0 * r / 20 * (.001 / tau) for r in (20, 50, 100, 200)}
    return dict(tau=tau, updates=updates, records=records, first_nonpositive_dq1_after_500=sign_change,
                predicted_penalized_sign_change=predicted, probe_classes=probe_classes,
                actor_trainables_unchanged=True, actor_optimizer_updates=0,
                critic_complete_updates=updates, critic_optimizer_operations=2 * updates,
                target_set_updates=updates, actor_BN_updates=updates,
                target_actor_tau=.001, target_actor_BN_tau=.001,
                target_critic_assignments=(len(a.target_critic_update_op) + len(a.target_critic_update_op2)),
                historical_overlapping_critic_target_scopes_retained=True,
                replay_reward_mean=float(np.mean(data['reward'])), probe_future_reward_mean=float(FR.mean()))


started = time.time()
if args.job == 'E1R':
    result = exploration_series()
else:
    result = critic_job(args.updates or {'H1': 40000, 'H2': 10000}[args.job])
out = dict(status='OFFLINE_COMPLETE', job=args.job, seed=args.seed, replay_rows=int(len(keep)), contiguity_breaks=gaps,
           probe_states=int(len(probe)), wall_s=time.time() - started, result=result, new_TCP_runs=0,
           source_init_verified_all_tensors=True, model_variables=len(original_vars),
           source_checkpoints_written=0,
           limitations='Static 27,436-row replay reused many times (about 750 draws per row after 40k updates; '
                       'overfitting possible beyond ~10k). H-step and Poisson-weighted returns follow the behavior '
                       'policy; the critic estimates the deterministic policy. Sign changes are compared per RTT x '
                       'delay class; the 2-BDP scaling of the predicted times is approximate (+-50%).')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(out, indent=1, allow_nan=False))
sess.close()
print(json.dumps(dict(output=str(OUTPUT), job=args.job, wall_s=out['wall_s']), indent=1), flush=True)
