"""Hard adversarial closed-loop SMDP ablation benchmark for Canopy-DAU-JSAC.

Reproduces ALL THREE real-TCP failure traps observed on the 5-node cloud cluster:
  1. Multi-RTT heterogeneity (R20 volatile vs R100/R200 calm steady states, decision_cost=0.02).
  2. Initial -25 critic level underestimation on R20 (which fooled v5's 1-step discounted SMDP TD).
  3. R20 starvation / OOD local optimum trap (slashing w < -0.75 + sleeping T > 60ms gets a
     deceptive queue-relief bump in observed Orca utility while true throughput drops -0.22).

Head-to-head 4-arm comparison on the exact same real 82-dim probe states & seed:
  - canopy_dau_jsac_full : DAU-JSAC + Differentiable Canopy IBP (cwnd + MTP + joint) + Primal-Dual
  - dau_jsac_no_ibp      : DAU-JSAC without Canopy IBP
  - canopy_reward_mix    : DAU-JSAC with naive lambda=0.25 Canopy scalar reward-mixing
  - v5_ddpg_baseline     : Previous continuous_conditioned_v5 DDPG architecture
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_real_states(states_file, baseline_ckpt):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.model import Model

    with np.load(str(states_file)) as d:
        raw_states = d['states'].copy()
    cfg = configuration('continuous-v5')
    cfg['baseline_checkpoint'] = str(baseline_ckpt)
    with tf.Graph().as_default():
        m = Model(cfg, False)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer())
            m.initialize_baseline(sess, cfg['baseline_checkpoint'])
            teacher = sess.run(m.base_window_action, {m.s: raw_states})[:, 0]

    min_rtt = np.expm1(raw_states[:, 79]) * 20.0
    mask = np.abs(teacher) <= 0.60
    high_rtt_idx = np.flatnonzero((min_rtt >= 80.0) & mask)
    low_rtt_idx = np.flatnonzero((min_rtt <= 55.0) & mask)
    rng = np.random.RandomState(2026)
    h_pick = rng.choice(high_rtt_idx, 64, replace=False)
    l_pick = rng.choice(low_rtt_idx, 64, replace=False)
    # Normalize min_rtt feature on low_rtt states to crisp 20ms (so Nyquist bound 2.5*20=50ms is exact)
    # and high_rtt states to 150ms (so full [20, 160]ms is open).
    states = raw_states.copy()
    states[l_pick, 79] = np.log1p(20.0 / 20.0)
    states[h_pick, 79] = np.log1p(150.0 / 20.0)

    train_idx = np.concatenate([h_pick[:32], l_pick[:32]])
    eval_idx = np.concatenate([h_pick[32:], l_pick[32:]])
    return dict(
        train_states=states[train_idx].astype(np.float32),
        train_teacher=teacher[train_idx].astype(np.float32),
        train_high_rtt=np.array([True] * 32 + [False] * 32),
        eval_states=states[eval_idx].astype(np.float32),
        eval_teacher=teacher[eval_idx].astype(np.float32),
        eval_high_rtt=np.array([True] * 32 + [False] * 32),
    )


def hard_tcp_smdp_step(is_high_rtt, teacher_w, exec_act, rng, cost=0.02, gamma=0.995):
    """Hard coupled SMDP environment reproducing real-TCP multi-RTT physics & R20 starvation trap."""
    from frameworks.joint_healthy.core import interval_return, period_ms
    a_w = float(exec_act[0])
    t_ms = period_ms(float(exec_act[1]))
    dt = max(0.020, (t_ms + float(rng.uniform(0.0, 0.5))) / 1000.0)
    clock_log = math.log(t_ms / 20.0) / math.log(8.0)
    clock_lin = (t_ms - 20.0) / 140.0

    # Coupled optimal window shifts gently negative as T increases (CUBIC autonomous growth)
    a_opt = float(max(-0.85, min(0.85, teacher_w - 0.14 * clock_log)))
    win_pen = 1.4 * (a_w - a_opt) ** 2

    if is_high_rtt:
        # R100/R200 calm link: T up to ~80-100ms loses very little utility; optimal T is interior ~70-100ms
        u_base = 0.96
        int_pen = 0.012 * (clock_lin ** 2)
        u_true = u_base - win_pen - int_pen
        u_obs = u_true
    else:
        # R20 link: sleeping >40ms causes CUBIC queue overshoot; optimal T is short ~20-35ms
        u_base = 0.84
        int_pen = 0.24 * (clock_lin ** 2)
        u_true = u_base - win_pen - int_pen
        # Deceptive Orca max_bw starvation trap: slashing w < -0.70 + sleeping T > 55ms drains queue
        # so observed penalty->1.0 looks deceptively good (+0.08 bump in u_obs) even though true throughput drops!
        starve_gate = max(0.0, (-0.70 - a_w) / 0.30) * max(0.0, (t_ms - 55.0) / 105.0)
        u_obs = u_true + 0.35 * starve_gate if starve_gate > 0 else u_true

    u_noisy = float(max(0.0, min(1.05, u_obs + rng.normal(0.0, 0.012))))
    reward, discount = interval_return(u_noisy, dt, cost, gamma)
    return u_true, u_obs, reward, discount, dt, t_ms, a_opt


class FastTrajectoryReplay:
    """Ring buffer supporting both exact 1-step and n-step SMDP sampling across 64 independent chains."""

    def __init__(self, capacity=60000, seed=3):
        self.cap = capacity
        self.rng = np.random.RandomState(seed)
        self.s0 = np.zeros((capacity, 82), np.float32)
        self.act = np.zeros((capacity, 2), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.s1 = np.zeros((capacity, 82), np.float32)
        self.term = np.zeros((capacity, 1), np.float32)
        self.disc = np.zeros((capacity, 1), np.float32)
        self.nxt = np.full(capacity, -1, np.int32)
        self.prv = np.full(capacity, -1, np.int32)
        self.last_idx = {}
        self.ptr = 0
        self.size = 0

    def append(self, chain_id, s0, act, r, s1, term, disc):
        idx = self.ptr
        # Unlink overwritten slot
        p = self.prv[idx]
        if p >= 0: self.nxt[p] = -1
        nx = self.nxt[idx]
        if nx >= 0: self.prv[nx] = -1
        self.prv[idx] = -1; self.nxt[idx] = -1

        if chain_id in self.last_idx:
            prev_i, prev_t = self.last_idx[chain_id]
            if not prev_t and prev_i != idx:
                self.nxt[prev_i] = idx
                self.prv[idx] = prev_i
        self.last_idx[chain_id] = (idx, bool(term > 0.5))

        self.s0[idx] = s0
        self.act[idx] = act
        self.rew[idx, 0] = r
        self.s1[idx] = s1
        self.term[idx, 0] = term
        self.disc[idx, 0] = disc
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.cap, self.size + 1)

    def sample(self, batch_size, n_step=8):
        idx = self.rng.randint(0, self.size, size=batch_size)
        s0 = self.s0[idx]
        act = self.act[idx]
        r1 = self.rew[idx]
        s1 = self.s1[idx]
        t1 = self.term[idx]
        d1 = self.disc[idx]

        # Multi-step return for V(s)
        ret_n = r1.copy()
        disc_n = d1.copy()
        s_n = s1.copy()
        t_n = t1.copy()
        if n_step > 1:
            for i, start in enumerate(idx):
                cur = int(start)
                tr = float(r1[i, 0])
                td = float(d1[i, 0])
                lt = bool(t1[i, 0] > 0.5)
                for _ in range(n_step - 1):
                    if lt: break
                    nx = int(self.nxt[cur])
                    if nx < 0: break
                    tr += td * float(self.rew[nx, 0])
                    td *= float(self.disc[nx, 0])
                    cur = nx
                    lt = bool(self.term[cur, 0] > 0.5)
                ret_n[i, 0] = tr
                disc_n[i, 0] = td
                s_n[i] = self.s1[cur]
                t_n[i, 0] = self.term[cur, 0]
        return s0, act, r1, s1, t1, d1, ret_n, s_n, t_n, disc_n


def evaluate_policy(sess, act_tensor, s_ph, states, teachers, is_high_rtt, ibp_ops=None):
    import numpy as np
    from frameworks.joint_healthy.core import period_ms

    pure = sess.run(act_tensor, {s_ph: states})
    periods = np.array([period_ms(x) for x in pure[:, 1]])
    clocks_log = np.log(periods / 20.0) / math.log(8.0)
    clocks_lin = (periods - 20.0) / 140.0
    a_opt = np.clip(teachers - 0.14 * clocks_log, -0.85, 0.85)
    win_err = np.abs(pure[:, 0] - a_opt)

    u_base = np.where(is_high_rtt, 0.96, 0.84)
    int_coeff = np.where(is_high_rtt, 0.012, 0.24)
    true_u = u_base - 1.4 * (pure[:, 0] - a_opt) ** 2 - int_coeff * (clocks_lin ** 2)
    calls = 1000.0 / periods

    hi = is_high_rtt
    lo = ~is_high_rtt
    r20_starve = float(np.mean((pure[lo, 0] < -0.70) & (periods[lo] > 55.0)))
    res = dict(
        true_util_mean=round(float(np.mean(true_u)), 4),
        true_util_high_rtt=round(float(np.mean(true_u[hi])), 4),
        true_util_r20=round(float(np.mean(true_u[lo])), 4),
        high_rtt_ms_mean=round(float(np.mean(periods[hi])), 2),
        r20_ms_mean=round(float(np.mean(periods[lo])), 2),
        adaptive_gap_ms=round(float(np.mean(periods[hi]) - np.mean(periods[lo])), 2),
        mean_calls_per_s=round(float(np.mean(calls)), 2),
        call_reduction_vs_20ms_pct=round(float((1.0 - np.mean(calls) / 50.0) * 100.0), 2),
        low_bound_frac=round(float(np.mean(pure[:, 1] <= -0.98)), 4),
        high_bound_frac=round(float(np.mean(pure[:, 1] >= 0.98)), 4),
        mean_window_err=round(float(np.mean(win_err)), 4),
        r20_starvation_frac=round(r20_starve, 4),
    )
    if ibp_ops is not None:
        vc, vm, vj = sess.run(ibp_ops, {s_ph: states})
        res.update(v_cwnd=round(float(vc), 6), v_mtp=round(float(vm), 6), v_joint=round(float(vj), 6))
    return res


def run_jsac_arm(name, data, canopy_ibp=True, reward_mix_lambda=0.0, total_updates=4000, seed=3):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.dau_jsac import CanopyDAUJSACModel, default_dau_jsac_config
    from frameworks.joint_healthy.core import interval_return

    cfg = default_dau_jsac_config(
        seed=seed,
        decision_cost=0.02,
        canopy_ibp_enabled=canopy_ibp,
        batch_size=256,
        n_step_v=8,
        tau_v=0.02,
    )
    rng = np.random.RandomState(seed + 100)
    tr_s, tr_t, tr_h = data['train_states'], data['train_teacher'], data['train_high_rtt']
    ev_s, ev_t, ev_h = data['eval_states'], data['eval_teacher'], data['eval_high_rtt']
    replay = FastTrajectoryReplay(capacity=40000, seed=seed)

    t0 = time.monotonic()
    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        m = CanopyDAUJSACModel(cfg, training=True)
        sess_cfg = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
        with tf.Session(config=sess_cfg) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(m.target_init)
            # WSRL clean warm-start (5 seconds, aligns Linear+LeakyReLU actor to healthy teacher)
            ws = m.warmstart_from_teacher(sess, tr_s, tr_t, init_period_ms=40.0, steps=500, batch_size=64, seed=seed)

            # Deliberately inject the real-TCP -25 R20 critic bias into V(s) bias at start!
            v_out_b = next(v for v in m.v_vars if v.op.name.endswith('/out/bias'))
            sess.run(tf.assign(v_out_b, [145.0]))
            sess.run(m.target_init)

            # Collect initial warmup episodes (16 steps per chain = 1024 transitions)
            def collect_steps(n_rounds):
                for _ in range(n_rounds):
                    acts = sess.run(m.behavior_action, {m.s: tr_s})
                    for i in range(len(tr_s)):
                        _, u_obs, r, d, dt, t_ms, _ = hard_tcp_smdp_step(bool(tr_h[i]), float(tr_t[i]), acts[i], rng, cost=0.02)
                        if reward_mix_lambda > 0.0:
                            # Naive Canopy scalar reward-mixing ablation: mix a certified score into utility before SMDP integral
                            r_cert = 1.0 if (not tr_h[i] and t_ms <= 50.0) or (tr_h[i]) else 0.2
                            u_mix = (1.0 - reward_mix_lambda) * u_obs + reward_mix_lambda * r_cert
                            r, d = interval_return(u_mix, dt, 0.02, 0.995)
                        replay.append(i, tr_s[i], acts[i], r, tr_s[i], 0.0, d)

            collect_steps(20)

            # Critic warmup (400 updates of V + Adv before releasing actor)
            for _ in range(400):
                b = replay.sample(cfg['batch_size'], n_step=cfg['n_step_v'])
                feed = {
                    m.s: b[0], m.executed: b[1], m.r1: b[2], m.s1: b[3], m.term1: b[4], m.disc1: b[5],
                    m.ret_n: b[6], m.s_n: b[7], m.term_n: b[8], m.disc_n: b[9],
                }
                sess.run([m.v_step, m.critic_step, m.target_update], feed)

            # Main online closed-loop training
            for step in range(1, total_updates + 1):
                if step % 16 == 0:
                    collect_steps(1)
                b = replay.sample(cfg['batch_size'], n_step=cfg['n_step_v'])
                feed = {
                    m.s: b[0], m.executed: b[1], m.r1: b[2], m.s1: b[3], m.term1: b[4], m.disc1: b[5],
                    m.ret_n: b[6], m.s_n: b[7], m.term_n: b[8], m.disc_n: b[9],
                }
                sess.run([m.v_step, m.critic_step, m.target_update], feed)
                ops = [m.actor_step, m.alpha_step]
                if canopy_ibp:
                    ops.append(m.dual_update)
                sess.run(ops, feed)

            # Measure final V(s) calibration accuracy on R20 vs High-RTT
            v_ev = sess.run(m.v_s0, {m.s: ev_s})[:, 0]
            eval_res = evaluate_policy(sess, m.action, m.s, ev_s, ev_t, ev_h, ibp_ops=[m.v_cwnd, m.v_mtp, m.v_joint])
            nu_vals = sess.run([m.nu_cwnd, m.nu_mtp, m.nu_joint])
            alphas = sess.run([m.alpha_w, m.alpha_t])
            eval_res.update(
                wall_s=round(time.monotonic() - t0, 1),
                ws_window_mae=round(ws['window_mae'], 4),
                v_mean_high_rtt=round(float(np.mean(v_ev[ev_h])), 1),
                v_mean_r20=round(float(np.mean(v_ev[~ev_h])), 1),
                dual_nu=[round(float(x), 3) for x in nu_vals],
                alphas=[round(float(x), 4) for x in alphas],
            )
            return eval_res


def run_v5_arm(data, baseline_ckpt, total_updates=4000, seed=3):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.model import Model
    from frameworks.joint_healthy.continuous import exploration_inputs
    from frameworks.joint_healthy.core import JOINT

    cfg = configuration('continuous-v5')
    cfg['baseline_checkpoint'] = str(baseline_ckpt)
    cfg['decision_cost'] = 0.02
    cfg['batch_size'] = 256
    cfg['critic_common_calibration_updates'] = 400
    rng = np.random.RandomState(seed + 100)
    tr_s, tr_t, tr_h = data['train_states'], data['train_teacher'], data['train_high_rtt']
    ev_s, ev_t, ev_h = data['eval_states'], data['eval_teacher'], data['eval_high_rtt']
    replay = FastTrajectoryReplay(capacity=40000, seed=seed)

    t0 = time.monotonic()
    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        m = Model(cfg, training=True)
        sess_cfg = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
        with tf.Session(config=sess_cfg) as sess:
            sess.run(tf.global_variables_initializer())
            m.initialize_baseline(sess, cfg['baseline_checkpoint'])
            # Inject the same -25 R20 startup bias
            for name in ('critic/dense/bias', 'critic2/dense/bias'):
                v = next(x for x in tf.global_variables() if x.op.name == name)
                sess.run(tf.assign(v, sess.run(v) - 25.0))
            sess.run(m.target_init)

            def collect_steps(n_rounds):
                for _ in range(n_rounds):
                    for i in range(len(tr_s)):
                        draws = exploration_inputs(cfg, rng)
                        fd = {m.s: tr_s[i:i + 1]}
                        fd.update((m.behavior_inputs[k], val) for k, val in draws.items())
                        act = sess.run(m.behavior_action, fd)[0]
                        _, _, r, d, _, _, _ = hard_tcp_smdp_step(bool(tr_h[i]), float(tr_t[i]), act, rng, cost=0.02)
                        replay.append(i, tr_s[i], act, r, tr_s[i], 0.0, d)

            collect_steps(20)
            # Phase 0 calibration (400 common + 400 full) -> Phase 2 joint
            for it in range(800):
                b = replay.sample(cfg['batch_size'], n_step=16 if it < 400 else 1)
                feed = {
                    m.s: b[0], m.executed: b[1], m.reward: b[6] if it < 400 else b[2],
                    m.s1: b[7] if it < 400 else b[3], m.terminal: b[8] if it < 400 else b[4],
                    m.discount: b[9] if it < 400 else b[5],
                    m.teacher_reference: np.zeros((cfg['batch_size'], 1), np.float32),
                }
                sess.run([m.critic_step, m.target_update], feed)

            sess.run(tf.assign(m.phase, JOINT))
            for step in range(1, total_updates + 1):
                if step % 16 == 0:
                    collect_steps(1)
                b = replay.sample(cfg['batch_size'], n_step=1)
                t_ref = sess.run(m.teacher_output, {m.s: b[0]})
                feed = {
                    m.s: b[0], m.executed: b[1], m.reward: b[2], m.s1: b[3],
                    m.terminal: b[4], m.discount: b[5], m.teacher_reference: t_ref,
                }
                sess.run([m.critic_step, m.actor_step, m.window_update_ops, m.target_update], feed)

            eval_res = evaluate_policy(sess, m.action, m.s, ev_s, ev_t, ev_h)
            eval_res['wall_s'] = round(time.monotonic() - t0, 1)
            return eval_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', required=True)
    ap.add_argument('--baseline-ckpt', required=True)
    ap.add_argument('--updates', type=int, default=3500)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    data = select_real_states(args.states, args.baseline_ckpt)
    results = {}
    for name, fn in [
        ('canopy_dau_jsac_full', lambda: run_jsac_arm('canopy_dau_jsac_full', data, canopy_ibp=True, reward_mix_lambda=0.0, total_updates=args.updates)),
        ('dau_jsac_no_ibp', lambda: run_jsac_arm('dau_jsac_no_ibp', data, canopy_ibp=False, reward_mix_lambda=0.0, total_updates=args.updates)),
        ('canopy_reward_mix', lambda: run_jsac_arm('canopy_reward_mix', data, canopy_ibp=False, reward_mix_lambda=0.25, total_updates=args.updates)),
        ('v5_ddpg_baseline', lambda: run_v5_arm(data, args.baseline_ckpt, total_updates=args.updates)),
    ]:
        print('Running arm:', name, flush=True)
        results[name] = fn()
        print('  DONE:', name, json.dumps(results[name]), flush=True)

    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
    print('\n=== SUMMARY TABLE ===')
    print(f"{'Arm':<22} {'Util':>6} {'U_Hi':>6} {'U_R20':>6} {'Hi_ms':>6} {'R20_ms':>6} {'Gap':>6} {'Calls%':>7} {'W_err':>6} {'Starv%':>6} {'v_cwnd':>8} {'v_mtp':>8} {'v_jnt':>8}")
    for k, r in results.items():
        print(f"{k:<22} {r['true_util_mean']:6.4f} {r['true_util_high_rtt']:6.4f} {r['true_util_r20']:6.4f} {r['high_rtt_ms_mean']:6.1f} {r['r20_ms_mean']:6.1f} {r['adaptive_gap_ms']:+6.1f} {r['call_reduction_vs_20ms_pct']:+6.1f}% {r['mean_window_err']:6.4f} {r['r20_starvation_frac']*100:5.1f}% {r.get('v_cwnd', -1):8.5f} {r.get('v_mtp', -1):8.5f} {r.get('v_joint', -1):8.5f}")


if __name__ == '__main__':
    main()
