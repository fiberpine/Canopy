"""Closed-loop coupled Actor-Critic SMDP verification (no oracle Q).

Uses real 82-dim TCP probe states, frozen step-6013196 initialization,
real Replay + PhysicalCredit + CalibrationGate, and genuine SMDP Bellman
bootstrapping from the learned Target Critic. Compares v4 vs v5 head-to-head.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_real_states(states_file, cfg):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model

    with np.load(str(states_file)) as d:
        raw_states = d['states'].copy()
    assert np.all(raw_states[:, 66] > 0.0)  # all real probes already have fresh=1 (20.5ms init trap)
    with tf.Graph().as_default():
        m = Model(cfg, False)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer())
            m.initialize_baseline(sess, cfg['baseline_checkpoint'])
            teacher = sess.run(m.base_window_action, {m.s: raw_states})[:, 0]

    min_rtt = np.expm1(raw_states[:, 79]) * 20.0
    mask = np.abs(teacher) <= 0.65
    steady_idx = np.flatnonzero((min_rtt >= 80.0) & mask)
    volatile_idx = np.flatnonzero((min_rtt <= 55.0) & mask)
    rng = np.random.RandomState(2026)
    s_pick = rng.choice(steady_idx, 64, replace=False)
    v_pick = rng.choice(volatile_idx, 64, replace=False)
    train_idx = np.concatenate([s_pick[:32], v_pick[:32]])
    eval_idx = np.concatenate([s_pick[32:], v_pick[32:]])
    return dict(
        train_states=raw_states[train_idx],
        train_teacher=teacher[train_idx],
        train_steady=np.array([True] * 32 + [False] * 32),
        eval_states=raw_states[eval_idx],
        eval_teacher=teacher[eval_idx],
        eval_steady=np.array([True] * 32 + [False] * 32),
    )


def smdp_physics(state_steady, teacher_action, exec_action, rng, cost=0.01, gamma=0.995):
    """Coupled window-period SMDP environment calibrated to observed TCP behavior."""
    from frameworks.joint_healthy.core import interval_return, period_ms
    a_c = float(exec_action[0])
    t_ms = period_ms(float(exec_action[1]))
    dt = max(0.020, (t_ms + float(rng.uniform(0.0, 0.5))) / 1000.0)
    clock_log = math.log(t_ms / 20.0) / math.log(8.0)
    clock_lin = (t_ms - 20.0) / 140.0
    a_opt = float(max(-0.95, min(0.95, teacher_action - 0.22 * clock_log)))
    win_penalty = 1.6 * (a_c - a_opt) ** 2
    int_coeff = 0.0035 if state_steady else 0.12
    int_penalty = int_coeff * (clock_lin ** 2)
    u_clean = 0.995 - win_penalty - int_penalty
    u_noisy = float(max(0.0, min(1.05, u_clean + rng.normal(0.0, 0.015))))
    reward, discount = interval_return(u_noisy, dt, cost, gamma)
    return u_clean, reward, discount, dt, t_ms, a_opt


def evaluate_frozen(sess, model, states, teachers, is_steady):
    import numpy as np
    from frameworks.joint_healthy.core import period_ms

    pure = sess.run(model.action, {model.s: states})
    periods = np.array([period_ms(x) for x in pure[:, 1]])
    clocks_log = np.log(periods / 20.0) / np.log(8.0)
    clocks_lin = (periods - 20.0) / 140.0
    a_opt = np.clip(teachers - 0.22 * clocks_log, -0.95, 0.95)
    win_err = np.abs(pure[:, 0] - a_opt)
    int_coeff = np.where(is_steady, 0.0035, 0.12)
    utilities = 0.995 - 1.6 * (pure[:, 0] - a_opt) ** 2 - int_coeff * (clocks_lin ** 2)

    # Check conditioned window tracking across fixed diagnostic periods [20, 40, 80, 120, 160].
    cond_errors = {}
    for ms in (20.0, 40.0, 80.0, 120.0, 160.0):
        diag = sess.run(model.diagnostic_action, {
            model.s: states,
            model.diagnostic_ms: np.full((len(states), 1), ms, np.float32),
        })
        target = np.clip(teachers - 0.22 * math.log(ms / 20.0) / math.log(8.0), -0.95, 0.95)
        cond_errors[str(int(ms))] = float(np.mean(np.abs(diag[:, 0] - target)))

    calls_per_s = 1000.0 / periods
    steady_mask = is_steady
    volatile_mask = ~is_steady
    return dict(
        mean_utility=float(np.mean(utilities)),
        steady_utility=float(np.mean(utilities[steady_mask])),
        volatile_utility=float(np.mean(utilities[volatile_mask])),
        mean_calls_per_s=float(np.mean(calls_per_s)),
        call_reduction_vs_20ms_pct=float((50.0 - np.mean(calls_per_s)) / 50.0 * 100.0),
        steady_period_ms_mean=float(np.mean(periods[steady_mask])),
        steady_period_ms_p10_p90=np.percentile(periods[steady_mask], [10, 90]).tolist(),
        volatile_period_ms_mean=float(np.mean(periods[volatile_mask])),
        volatile_period_ms_p10_p90=np.percentile(periods[volatile_mask], [10, 90]).tolist(),
        adaptive_gap_ms=float(np.mean(periods[steady_mask]) - np.mean(periods[volatile_mask])),
        mean_window_err=float(np.mean(win_err)),
        cond_window_mae_by_ms=cond_errors,
    )


def run_closed_loop(name, cfg, data, joint_updates=6000, seed=3):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    from frameworks.joint_healthy.replay import Replay, CalibrationGate
    from frameworks.joint_healthy.core import CALIBRATION, FORWARD_ONLY, JOINT
    from frameworks.joint_healthy import continuous

    cfg = dict(cfg)
    cfg.update(
        seed=seed,
        num_actors=8,
        memsize=40000,
        batch_size=256,
        warmup=1024,
        minimum_enqueued_per_actor=64,
        warmup_physical_seconds=40.0,
        calibration_updates=1500,
        forward_calibration_updates=1500,
        calibration_check_updates=500,
        calibration_rows_per_actor_period=16,
        minimum_holdout_per_period=8,
        minimum_short_return_blocks=2,
        critic_common_calibration_updates=min(500, int(cfg.get('critic_common_calibration_updates', 0))),
        time_limit_handling='bootstrap_last_observation',
        holdout_layout='episode_rotated_blocks_v1',
        release_joint=True,
    )

    train_states = data['train_states']
    train_teachers = data['train_teacher']
    train_steady = data['train_steady']
    n_states = len(train_states)
    episode_len = 40

    replay = Replay(cfg)
    gate = CalibrationGate(cfg)
    rng_env = np.random.RandomState(seed + 77)
    actor_rngs = [np.random.RandomState(seed + i * 1009) for i in range(cfg['num_actors'])]
    actor_pos = [i * (n_states // cfg['num_actors']) for i in range(cfg['num_actors'])]
    actor_ep = [0] * cfg['num_actors']
    actor_seq = [0] * cfg['num_actors']

    pre_adapt_cond = bool(cfg.get('pre_adapt_conditioned_window_in_forward_phase', False))
    trajectory = []
    t0 = time.monotonic()

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        model = Model(cfg, training=True)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer())
            model.initialize_baseline(sess, cfg['baseline_checkpoint'])
            bn_vars = [v for v in model.base_window_vars if v.op.name.endswith(('/moving_mean', '/moving_variance'))]
            bn_initial = sess.run(bn_vars)
            frozen_base_hash = model.digest(sess, base_only=True)

            def step_actor(actor_id):
                idx = actor_pos[actor_id] % n_states
                s0 = train_states[idx]
                draws = continuous.exploration_inputs(cfg, actor_rngs[actor_id])
                feed = {model.s: s0[None, :]}
                feed.update((model.behavior_inputs[k], v) for k, v in draws.items())
                exec_action = sess.run(model.behavior_action, feed)[0]
                _, reward, discount, dt, _, _ = smdp_physics(
                    bool(train_steady[idx]), float(train_teachers[idx]), exec_action, rng_env,
                    cost=float(cfg['decision_cost']), gamma=float(cfg['gamma'])
                )
                next_idx = (idx + 1) % n_states
                s1 = train_states[next_idx]
                actor_pos[actor_id] = next_idx
                actor_seq[actor_id] += 1
                ep = actor_ep[actor_id]
                seq = actor_seq[actor_id]
                terminal = (seq >= episode_len)
                teacher_val = float(train_teachers[idx]) if model.teacher_output is not None else None
                row = [
                    s0.copy(), exec_action.astype(np.float32),
                    np.array([reward], np.float32), s1.copy(),
                    np.array([float(terminal)], np.float32), np.array([dt], np.float64),
                    np.array([discount], np.float32),
                ]
                replay.append(row, actor_id, ep, seq, teacher_action=teacher_val)
                if terminal:
                    actor_ep[actor_id] += 1
                    actor_seq[actor_id] = 0

            # Warmup replay
            while replay.credit.warmup_seconds is None or replay.credit.limit < 1:
                for a in range(cfg['num_actors']):
                    step_actor(a)

            trajectory.append(dict(
                phase=gate.phase, critic_iterations=0, actor_updates=0,
                eval=evaluate_frozen(sess, model, data['eval_states'], data['eval_teacher'], data['eval_steady']),
            ))

            critic_iters = 0
            actor_updates = 0
            cond_pre_updates = 0
            phase_transitions = {}
            max_total_critic = cfg['calibration_updates'] + cfg['forward_calibration_updates'] + joint_updates + 6000

            while actor_updates < joint_updates and critic_iters < max_total_critic:
                while replay.credit.limit <= critic_iters:
                    for a in range(cfg['num_actors']):
                        step_actor(a)
                n_step = int(cfg.get('common_calibration_n_step', 1)) if critic_iters < model.common_calibration_updates else 1
                batch = replay.sample(n_step=n_step)
                feed = model.feed(batch)
                sess.run(model.critic_step, feed)
                if gate.phase == JOINT:
                    sess.run([model.actor_step, model.window_update_ops], feed)
                    actor_updates += 1
                elif gate.phase == FORWARD_ONLY and pre_adapt_cond and getattr(model, 'cond_window_step', None) is not None:
                    if critic_iters >= model.common_calibration_updates:
                        sess.run(model.cond_window_step, feed)
                        cond_pre_updates += 1
                sess.run(model.target_update)
                critic_iters += 1

                if critic_iters % cfg['calibration_check_updates'] == 0:
                    if gate.phase < JOINT:
                        if gate.phase == FORWARD_ONLY and pre_adapt_cond:
                            assert model.digest(sess, base_only=True) == frozen_base_hash
                        batches = [replay.holdout_batch(i) for i in range(5)]
                        short_batches = [replay.short_return_batch(i) for i in range(5)]
                        reports = []
                        for period, heldout in enumerate(batches):
                            if heldout is None:
                                continue
                            q1, q2, target = sess.run([model.q1, model.q2, model.label], model.feed(heldout))
                            scale = max(1., float(np.sqrt(np.mean(target ** 2))))
                            abs_err = [float(np.sqrt(np.mean((q - target) ** 2))) for q in (q1, q2)]
                            errs = [e / scale for e in abs_err]
                            reports.append(dict(period=period, rows=len(q1), normalized_td_rmse=max(errs), twin_errors=errs))
                        short_reports = []
                        for period, block in enumerate(short_batches):
                            if block is None:
                                continue
                            first, last, rewards, continuation = block
                            q1, q2 = sess.run([model.q1, model.q2], model.feed(first))
                            tail = sess.run(model.target_q, model.feed(last))
                            target = rewards + continuation * tail
                            scale = max(1., float(np.sqrt(np.mean(target ** 2))))
                            errs = [float(np.sqrt(np.mean((q - target) ** 2)) / scale) for q in (q1, q2)]
                            short_reports.append(dict(period=period, blocks=len(q1), normalized_return_rmse=max(errs), twin_errors=errs))
                        prev_phase = gate.phase
                        decision = gate.assess(critic_iters, replay.coverage.copy(), reports, short_reports)
                        if gate.phase != prev_phase:
                            phase_transitions[f'{prev_phase}->{gate.phase}'] = critic_iters
                            sess.run(tf.assign(model.phase, gate.phase))
                    if critic_iters % 1500 == 0 or actor_updates == joint_updates:
                        trajectory.append(dict(
                            phase=gate.phase, critic_iterations=critic_iters, actor_updates=actor_updates,
                            eval=evaluate_frozen(sess, model, data['eval_states'], data['eval_teacher'], data['eval_steady']),
                        ))

            bn_final = sess.run(bn_vars)
            bn_frozen = all(np.array_equal(a, b) for a, b in zip(bn_initial, bn_final))
            final_eval = evaluate_frozen(sess, model, data['eval_states'], data['eval_teacher'], data['eval_steady'])
            final_train = evaluate_frozen(sess, model, data['train_states'], data['train_teacher'], data['train_steady'])

    return dict(
        name=name,
        wall_s=round(time.monotonic() - t0, 2),
        final_phase=gate.phase,
        phase_transitions=phase_transitions,
        critic_iterations=critic_iters,
        cond_pre_updates=cond_pre_updates,
        actor_updates=actor_updates,
        bn_bitwise_frozen=bn_frozen,
        eval_unseen=final_eval,
        eval_train=final_train,
        trajectory=trajectory,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v4-config', required=True)
    p.add_argument('--v5-config', required=True)
    p.add_argument('--states', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--joint-updates', type=int, default=6000)
    p.add_argument('--arms', default='v4_baseline,v5_full')
    args = p.parse_args()

    def load_cfg(path):
        return json.loads(Path(path).read_text().replace('@PACKAGE@', str(ROOT).replace('\\', '/')))

    cfg_v4 = load_cfg(args.v4_config)
    cfg_v5 = load_cfg(args.v5_config)
    data = select_real_states(args.states, cfg_v5)

    all_arms = {
        'v4_baseline': cfg_v4,
        'v5_full': cfg_v5,
        'v5_no_zero_anchor': dict(cfg_v5, critic_period_zero_anchor=False),
        'v5_no_pre_adapt': dict(cfg_v5, pre_adapt_conditioned_window_in_forward_phase=False),
        'v5_no_stopgrad_envelope': dict(cfg_v5, envelope_stop_gradient_duration=False),
        'v5_no_barrier': dict(cfg_v5, time_barrier_weight=0.0),
        'v5_no_nstep': dict(cfg_v5, common_calibration_n_step=1, calibration_critic_tau=None),
    }
    selected = [x.strip() for x in args.arms.split(',') if x.strip()]
    arms = [(name, all_arms[name]) for name in selected]

    results = {}
    for name, cfg in arms:
        res = run_closed_loop(name, cfg, data, joint_updates=args.joint_updates)
        results[name] = res
        e = res['eval_unseen']
        print(
            f"[{name}] phase={res['final_phase']} util={e['mean_utility']:.5f} "
            f"(steady={e['steady_utility']:.5f}, vol={e['volatile_utility']:.5f}) "
            f"calls={e['mean_calls_per_s']:.2f}/s (-{e['call_reduction_vs_20ms_pct']:.1f}%) "
            f"T_steady={e['steady_period_ms_mean']:.1f}ms T_vol={e['volatile_period_ms_mean']:.1f}ms "
            f"gap={e['adaptive_gap_ms']:.1f}ms cond160_err={e['cond_window_mae_by_ms']['160']:.4f}",
            flush=True,
        )

    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
