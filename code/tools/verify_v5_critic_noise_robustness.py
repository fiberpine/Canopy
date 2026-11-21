"""Compare v4 unanchored vs v5 zero-anchored critic under realistic 85%-at-20ms noisy replay."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_critic_stress(cfg, states_file, updates=10000, noise_std=0.50, seed=3):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model

    with np.load(str(states_file)) as p:
        source = p['states']
    min_rtt = np.expm1(source[:, 79]) * 20.0
    # 16 real TCP states: 4 per RTT tier (<30, 30-70, 70-130, >=130ms).
    rng_pick = np.random.RandomState(42)
    picks = []
    for lo, hi in [(0, 30), (30, 70), (70, 130), (130, 300)]:
        idx = np.flatnonzero((min_rtt >= lo) & (min_rtt < hi))
        picks.extend(rng_pick.choice(idx, 4, replace=False).tolist())
    states = source[picks].copy()
    states[:, 66] = 1.0
    state_rtt = np.expm1(states[:, 79]) * 20.0
    # Steady high-RTT states (>80ms) have positive period advantage; low-RTT states (<50ms) negative.
    state_signs = np.where(state_rtt >= 80.0, 1.0, np.where(state_rtt <= 50.0, -1.0, 0.5))

    gamma, cost = 0.995, 0.01
    minimum_margin = cost * (gamma - gamma ** 2) / (1.0 - gamma)
    threshold = minimum_margin / 4.0

    # Build realistic online replay dataset: 85% around 20.5ms, 15% log-uniform [20, 160]ms.
    rng = np.random.RandomState(100 + seed)
    n_per_state = 256
    n_total = len(states) * n_per_state
    ss = np.repeat(states, n_per_state, axis=0)
    signs = np.repeat(state_signs, n_per_state)
    rtts = np.repeat(state_rtt, n_per_state)

    is_probe = rng.random_sample(n_total) < 0.15
    t = np.where(
        is_probe,
        np.exp(rng.uniform(np.log(20.0), np.log(160.0), n_total)),
        np.clip(20.5 + rng.normal(0.0, 0.25, n_total), 20.0, 22.0),
    )
    a_win = np.clip(rng.normal(0.0, 0.15, n_total), -1.0, 1.0)
    a_time = 2.0 * np.log(t / 20.0) / np.log(8.0) - 1.0
    actions = np.column_stack([a_win, a_time]).astype(np.float32)

    base_q = 150.0 + 0.1 * rtts - 1.5 * (a_win ** 2)
    d = gamma ** (t / 20.0)
    adv = signs * cost * (gamma - d) / (1.0 - gamma)
    # Realistic base TD noise (~0.5 std) on all samples, matching observed real TCP TD RMSE.
    td_noise = rng.normal(0.0, noise_std, n_total)
    labels = (base_q + adv + td_noise)[:, None].astype(np.float32)

    queries = np.array([20.0, 40.0, 80.0, 160.0], dtype=np.float32)
    qs = np.repeat(states, 4, axis=0)
    qt = np.tile(queries, len(states))
    qa = np.column_stack([np.zeros(len(qt)), 2.0 * np.log(qt / 20.0) / np.log(8.0) - 1.0]).astype(np.float32)
    truth = (state_signs[:, None] * cost * (gamma - gamma ** (queries[None, :] / 20.0)) / (1.0 - gamma))

    session_cfg = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
    late_start = updates // 2
    trajectory = []
    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        model = Model(cfg, True)
        with tf.Session(config=session_cfg) as sess:
            sess.run(tf.global_variables_initializer())
            model.initialize_baseline(sess, cfg['baseline_checkpoint'])
            for i in range(updates + 1):
                if i in (0, 1000, 3000, 5000, updates) or (i >= late_start and i % 500 == 0):
                    q1, q2 = sess.run([model.q1, model.q2], {model.s: qs, model.executed: qa})
                    maes = []
                    ranks = []
                    for q in (q1, q2):
                        q = q.reshape(len(states), 4)
                        pred_adv = q - q[:, :1]
                        maes.append(float(np.mean(np.abs(pred_adv - truth)[:, 1:])))
                        ranks.append(float(np.mean(np.sign(pred_adv[:, 1:]) == np.sign(truth[:, 1:]))))
                    trajectory.append(dict(updates=i, advantage_mae=maes, ranking=ranks))
                if i < updates:
                    at = rng.choice(n_total, int(cfg['batch_size']), replace=False)
                    sess.run(model.critic_step, {model.s: ss[at], model.executed: actions[at], model.label: labels[at]})

    late_maes = np.array([x['advantage_mae'] for x in trajectory if x['updates'] >= late_start])
    late_ranks = np.array([x['ranking'] for x in trajectory if x['updates'] >= late_start])
    end = trajectory[-1]
    return dict(
        endpoint_mae=end['advantage_mae'],
        endpoint_ranking=end['ranking'],
        late_mean_mae=late_maes.mean(axis=0).tolist(),
        late_max_mae=late_maes.max(axis=0).tolist(),
        late_mean_ranking=late_ranks.mean(axis=0).tolist(),
        late_fraction_within_gate=(late_maes <= threshold).mean(axis=0).tolist(),
        pass_endpoint=all(m <= threshold for m in end['advantage_mae']) and all(r >= 0.95 for r in end['ranking']),
        trajectory=trajectory,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v4-config', required=True)
    p.add_argument('--v5-config', required=True)
    p.add_argument('--states', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--noise-std', type=float, default=0.50)
    args = p.parse_args()

    def load_cfg(path):
        return json.loads(Path(path).read_text().replace('@PACKAGE@', str(ROOT).replace('\\', '/')))

    cfg_v4 = load_cfg(args.v4_config)
    cfg_v5 = load_cfg(args.v5_config)
    res_v4 = run_critic_stress(cfg_v4, args.states, updates=args.updates, noise_std=args.noise_std)
    res_v5 = run_critic_stress(cfg_v5, args.states, updates=args.updates, noise_std=args.noise_std)

    threshold = 0.01 * (0.995 - 0.995 ** 2) / 0.005 / 4.0
    out = dict(
        task='Realistic 85%%-at-20.5ms + 15%% probe distribution, 16 real TCP states, TD noise_std=%.2f' % args.noise_std,
        threshold_mae=threshold,
        v4_unanchored=res_v4,
        v5_zero_anchored=res_v5,
    )
    Path(args.output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(
        threshold=threshold,
        v4_end_mae=res_v4['endpoint_mae'], v4_end_rank=res_v4['endpoint_ranking'], v4_late_mean_mae=res_v4['late_mean_mae'], v4_pass=res_v4['pass_endpoint'],
        v5_end_mae=res_v5['endpoint_mae'], v5_end_rank=res_v5['endpoint_ranking'], v5_late_mean_mae=res_v5['late_mean_mae'], v5_pass=res_v5['pass_endpoint'],
    ), indent=2))


if __name__ == '__main__':
    main()
