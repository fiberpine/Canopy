"""First-principles Canopy-DAU-JSAC: Value-Advantage Decomposed Joint Soft Actor-Critic
with Differentiable Canopy IBP Formal Constraints and Primal-Dual Multipliers.

Zero defensive patches:
  1. Unified factorized stochastic policy pi(w, tau | s) = pi(tau | s) * pi(w | s, tau)
     with pure Linear+LeakyReLU+Tanh mean heads for 100% sound, tight IBP.
  2. Value-Advantage decomposition (DAU, ICML'19 / Continuous SAC, NeurIPS'25):
     - LayerNorm V(s) trained via multi-step SMDP returns (eliminates level-slope coupling).
     - LayerNorm K-ensemble centered advantage networks A_k(s, w, tau) trained on exact
       1-step SMDP excess returns (eliminates 10,000:1 scale mismatch & OOD critic spikes).
  3. Differentiable Canopy IBP constraints over symbolic boxes (cwnd safety/liveness,
     MTP congestion reactivity + Nyquist RTT stability, and joint effort-duration coupling)
     enforced via Primal-Dual Lagrange multipliers (zero runtime overhead, zero interior bias).
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCHEMA_DAU_JSAC = 'canopy-dau-jsac-v1'
ARCH_DAU_JSAC = 'canopy_dau_jsac_v1'


def _layer_norm(tf, x, scope, eps=1e-5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = int(x.shape[-1])
        gamma = tf.get_variable('gamma', (1, dim), initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', (1, dim), initializer=tf.zeros_initializer())
    mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
    return gamma * (x - mean) * tf.rsqrt(var + eps) + beta


def _linear(tf, x, scope, width, init_scale=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        in_dim = int(x.shape[-1])
        if init_scale is not None:
            init = tf.random_uniform_initializer(-init_scale, init_scale)
        else:
            init = tf.glorot_uniform_initializer()
        w = tf.get_variable('kernel', (in_dim, width), initializer=init)
        b = tf.get_variable('bias', (width,), initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b


def _ibp_linear(tf, lo, hi, scope):
    with tf.variable_scope(scope, reuse=True):
        w = tf.get_variable('kernel')
        b = tf.get_variable('bias')
    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    c_out = tf.matmul(center, w) + b
    r_out = tf.matmul(radius, tf.abs(w))
    return c_out - r_out, c_out + r_out


def _ibp_leaky_relu(tf, lo, hi, alpha=0.2):
    return tf.nn.leaky_relu(lo, alpha=alpha), tf.nn.leaky_relu(hi, alpha=alpha)


def _ibp_tanh(tf, lo, hi):
    return tf.tanh(lo), tf.tanh(hi)


def _encoded_to_ms(tf, a_tau):
    return 20.0 * tf.pow(8.0, 0.5 * (a_tau + 1.0))


def _gaussian_logp(tf, z, mu, log_std):
    std = tf.exp(log_std)
    return -0.5 * (tf.square((z - mu) / (std + 1e-8)) + 2.0 * log_std + math.log(2.0 * math.pi))


def _squashed_logp(tf, z, mu, log_std, a):
    return _gaussian_logp(tf, z, mu, log_std) - tf.log(tf.maximum(1e-6, 1.0 - tf.square(a)))


def _replace_col(tf, x, col, vals):
    """Replace column `col` in 2D tensor `x` [N, D] with `vals` [N, 1]."""
    return tf.concat([x[:, :col], vals, x[:, col + 1:]], axis=1)


def build_canopy_symbolic_boxes(tf, states, k_sub=3):
    """Construct vectorized Canopy symbolic input boxes [lo, hi] directly from 82D states.

    Latest frame is indices 63..69:
      63: rate/max_bw, 64: pacing/max_bw, 65: 5*loss/max_bw,
      66: count/cwnd,  67: dt,            68: min_rtt/srtt,  69: penalty=min(1, 1.25*min/srtt)
    Index 79 is log1p(min_rtt_ms / 20.0).
    """
    n = tf.shape(states)[0]

    # 1. Local robustness box B_local(s): +/-2% perturbation on min/srtt (and consistent penalty)
    min_srtt = states[:, 68:69]
    lo_ms = tf.clip_by_value(min_srtt - 0.02, 0.01, 1.0)
    hi_ms = tf.clip_by_value(min_srtt + 0.02, 0.01, 1.0)
    lo_pen = tf.minimum(1.0, 1.25 * lo_ms)
    hi_pen = tf.minimum(1.0, 1.25 * hi_ms)
    loc_lo = _replace_col(tf, _replace_col(tf, states, 68, lo_ms), 69, lo_pen)
    loc_hi = _replace_col(tf, _replace_col(tf, states, 68, hi_ms), 69, hi_pen)

    # Helper to split an interval [a, b] into k_sub sub-intervals along batch axis
    def split_boxes(base_lo, base_hi, col, a, b, also_penalty=False):
        lo_list, hi_list = [], []
        width = (b - a) / float(k_sub)
        for i in range(k_sub):
            v_lo = tf.fill([n, 1], float(a + i * width))
            v_hi = tf.fill([n, 1], float(a + (i + 1) * width))
            s_lo = _replace_col(tf, base_lo, col, v_lo)
            s_hi = _replace_col(tf, base_hi, col, v_hi)
            if also_penalty:
                s_lo = _replace_col(tf, s_lo, 69, tf.minimum(1.0, 1.25 * v_lo))
                s_hi = _replace_col(tf, s_hi, 69, tf.minimum(1.0, 1.25 * v_hi))
            lo_list.append(s_lo)
            hi_list.append(s_hi)
        return tf.concat(lo_list, axis=0), tf.concat(hi_list, axis=0)

    # 2. B_bad_delay: min/srtt in [0.05, 0.25] (Canopy Deep Buffer high-delay safety)
    bd_lo, bd_hi = split_boxes(states, states, 68, 0.05, 0.25, also_penalty=True)

    # 3. B_bad_loss: 5*loss/max_bw in [0.25, 1.0] (Canopy Shallow Buffer high-loss safety)
    bl_lo, bl_hi = split_boxes(states, states, 65, 0.25, 1.0, also_penalty=False)

    # 4. B_good: min/srtt in [0.75, 1.0] AND loss = 0.0 (Canopy low-delay liveness)
    zero_loss = tf.zeros([n, 1], dtype=tf.float32)
    base_good = _replace_col(tf, states, 65, zero_loss)
    gd_lo, gd_hi = split_boxes(base_good, base_good, 68, 0.75, 1.0, also_penalty=True)

    return dict(
        local=(loc_lo, loc_hi),
        bad_delay=(bd_lo, bd_hi),
        bad_loss=(bl_lo, bl_hi),
        good=(gd_lo, gd_hi),
        k_sub=k_sub,
    )


def default_dau_jsac_config(**overrides):
    cfg = dict(
        schema=SCHEMA_DAU_JSAC,
        actor_architecture=ARCH_DAU_JSAC,
        state_dim=82,
        action_dim=2,
        base_period_s=0.02,
        gamma=0.995,
        decision_cost=0.02,
        # Actor / Critic architecture
        time_hidden=128,
        window_hidden=256,
        critic_hidden=128,
        v_hidden=256,
        ensemble_size=3,
        dau_center_samples=2,
        # Optimization
        lr_actor=3e-4,
        lr_v=1e-3,
        lr_adv=1e-3,
        lr_alpha=3e-4,
        tau_v=0.01,
        n_step_v=8,
        # Dual entropy targets (1D squashed Gaussian each)
        target_entropy_w=-1.6,
        target_entropy_t=-1.8,
        init_alpha_w=0.03,
        init_alpha_t=0.03,
        max_alpha=0.3,
        # Canopy Differentiable IBP + Primal-Dual Lagrange Multipliers
        canopy_ibp_enabled=True,
        canopy_k_sub=2,
        canopy_ibp_batch=64,
        mtp_react_ms=40.0,
        mtp_nyquist_kappa=2.5,
        joint_effort_delta=0.20,
        cwnd_good_min=-0.50,
        # Primal-dual settings
        dual_lr=0.05,
        pid_lagrangian=False,
        dual_kp=1.0,
        dual_ki=0.05,
        dual_kd=0.25,
        dual_tol_cwnd=1e-4,
        dual_tol_mtp=1e-4,
        dual_tol_joint=1e-4,
        dual_init=1.0,
        dual_max=50.0,
        # IBP-exact running observation standardization (SimBa RSNorm)
        obs_norm_enabled=False,
        obs_norm_momentum=0.005,
        obs_norm_std_floor=0.20,
        # Training / seed
        seed=3,
        batch_size=512,
        tf_threads=4,
    )
    cfg.update(overrides)
    return cfg


class CanopyDAUJSACModel:
    """Self-contained TF1.14 Canopy-DAU-JSAC model."""

    def __init__(self, cfg, training=False):
        import tensorflow as tf
        self.tf = tf
        self.cfg = dict(cfg)
        self.training = training

        self.s = tf.placeholder(tf.float32, (None, 82), name='state')
        self.schema = tf.get_variable('dau_jsac_schema', initializer=cfg.get('schema', SCHEMA_DAU_JSAC),
                                      dtype=tf.string, trainable=False)
        self.obs_norm_enabled = bool(cfg.get('obs_norm_enabled', False))
        if self.obs_norm_enabled:
            with tf.variable_scope('jsac_actor/obs_norm', reuse=tf.AUTO_REUSE):
                self.obs_mean = tf.get_variable('mean', shape=(1, 82), initializer=tf.zeros_initializer(),
                                                dtype=tf.float32, trainable=False)
                self.obs_std = tf.get_variable('std', shape=(1, 82), initializer=tf.ones_initializer(),
                                               dtype=tf.float32, trainable=False)
            self._obs_mu_ph = tf.placeholder(tf.float32, (1, 82), name='obs_mu_ph')
            self._obs_sd_ph = tf.placeholder(tf.float32, (1, 82), name='obs_sd_ph')
            self.set_obs_norm = tf.group(
                tf.assign(self.obs_mean, self._obs_mu_ph),
                tf.assign(self.obs_std, tf.maximum(self._obs_sd_ph, float(cfg.get('obs_norm_std_floor', 0.20)))),
            )
            mom = float(cfg.get('obs_norm_momentum', 0.005))
            std_floor = float(cfg.get('obs_norm_std_floor', 0.20))
            b_mean, b_var = tf.nn.moments(self.s, axes=[0], keep_dims=True)
            b_std = tf.maximum(tf.sqrt(tf.maximum(b_var, 1e-8)), std_floor)
            self.update_obs_norm = tf.group(
                tf.assign(self.obs_mean, (1.0 - mom) * self.obs_mean + mom * b_mean),
                tf.assign(self.obs_std, tf.maximum((1.0 - mom) * self.obs_std + mom * b_std, std_floor)),
            )
        else:
            self.obs_mean = None
            self.obs_std = None
            self.set_obs_norm = tf.no_op()
            self.update_obs_norm = tf.no_op()

        s_in = self._norm_s(self.s)

        # --- 1. Unified Factorized Stochastic Actor ---
        mu_t, log_std_t = self._time_head(s_in)
        std_t = tf.exp(log_std_t)
        eps_t = tf.random_normal(tf.shape(mu_t))
        z_t = mu_t + std_t * eps_t
        a_t_sto = tf.tanh(z_t)
        logp_t = _squashed_logp(tf, z_t, mu_t, log_std_t, a_t_sto)

        mu_w_sto, log_std_w_sto = self._window_head(s_in, a_t_sto)
        std_w_sto = tf.exp(log_std_w_sto)
        eps_w = tf.random_normal(tf.shape(mu_w_sto))
        z_w = mu_w_sto + std_w_sto * eps_w
        a_w_sto = tf.tanh(z_w)
        logp_w = _squashed_logp(tf, z_w, mu_w_sto, log_std_w_sto, a_w_sto)

        # Deterministic deployment policy (mean of factorized distribution)
        self.det_a_t = tf.tanh(mu_t)
        mu_w_det, _ = self._window_head(s_in, self.det_a_t)
        self.det_a_w = tf.tanh(mu_w_det)
        self.action = tf.concat([self.det_a_w, self.det_a_t], axis=1)
        self.planned_ms = _encoded_to_ms(tf, self.det_a_t)
        self.behavior_action = tf.concat([a_w_sto, a_t_sto], axis=1)
        self.policy_std = tf.concat([std_w_sto, std_t], axis=1)

        # Diagnostic same-weight override: evaluate window mean at specified period_ms
        self.diagnostic_ms = tf.placeholder(tf.float32, (None, 1), name='diagnostic_ms')
        diag_a_t = 2.0 * tf.log(tf.clip_by_value(self.diagnostic_ms, 20.0, 160.0) / 20.0) / math.log(8.0) - 1.0
        diag_mu_w, _ = self._window_head(s_in, diag_a_t)
        self.diagnostic_action = tf.concat([tf.tanh(diag_mu_w), diag_a_t], axis=1)

        self.actor_vars = self._scope_vars('jsac_actor', trainable=True)
        self.online_vars = self._scope_vars('jsac_actor', trainable=False)

        # --- 2. Sound Joint IBP Verifier (built for both training and auditing) ---
        self.ibp_lo_in = tf.placeholder(tf.float32, (None, 82), name='ibp_lo_in')
        self.ibp_hi_in = tf.placeholder(tf.float32, (None, 82), name='ibp_hi_in')
        self.ibp_a_w_lo, self.ibp_a_w_hi, self.ibp_a_t_lo, self.ibp_a_t_hi = self._joint_ibp(self.ibp_lo_in, self.ibp_hi_in)
        self.ibp_ms_lo = _encoded_to_ms(tf, self.ibp_a_t_lo)
        self.ibp_ms_hi = _encoded_to_ms(tf, self.ibp_a_t_hi)

        if not training:
            return

        # --- 3. Training Placeholders ---
        # Exact 1-step SMDP transition for Advantage networks A_k
        self.executed = tf.placeholder(tf.float32, (None, 2), name='executed_action')
        self.r1 = tf.placeholder(tf.float32, (None, 1), name='reward_1step')
        self.s1 = tf.placeholder(tf.float32, (None, 82), name='next_state_1step')
        self.term1 = tf.placeholder(tf.float32, (None, 1), name='terminal_1step')
        self.disc1 = tf.placeholder(tf.float32, (None, 1), name='discount_1step')

        # Multi-step SMDP transition for State Value network V(s)
        self.ret_n = tf.placeholder(tf.float32, (None, 1), name='return_nstep')
        self.s_n = tf.placeholder(tf.float32, (None, 82), name='next_state_nstep')
        self.term_n = tf.placeholder(tf.float32, (None, 1), name='terminal_nstep')
        self.disc_n = tf.placeholder(tf.float32, (None, 1), name='discount_nstep')

        # Optional supervised WSRL warm-start placeholders
        self.ws_target_w = tf.placeholder(tf.float32, (None, 1), name='ws_target_w')
        self.ws_target_t = tf.placeholder(tf.float32, (None, 1), name='ws_target_t')

        self.global_step = tf.train.get_or_create_global_step()

        # --- 4. Value-Advantage Decomposed LayerNorm Ensemble Critic ---
        s1_in = self._norm_s(self.s1)
        sn_in = self._norm_s(self.s_n)
        self.v_s0 = self._v_net(s_in, 'jsac_v')
        self.target_v_s0 = self._v_net(s_in, 'jsac_target_v')
        self.target_v_s1 = self._v_net(s1_in, 'jsac_target_v')
        self.target_v_sn = self._v_net(sn_in, 'jsac_target_v')

        # Multi-step V(s) target -> fast, unbiased calibration across all RTT/BW regimes
        huber_d = float(cfg.get('critic_huber_delta', 0.0))
        def _reg_loss(err):
            if huber_d > 0.0:
                ae = tf.abs(err)
                return tf.reduce_mean(tf.where(ae <= huber_d, 0.5 * tf.square(err), huber_d * (ae - 0.5 * huber_d)))
            return tf.reduce_mean(tf.square(err))
        self.v_target = tf.stop_gradient(self.ret_n + self.disc_n * (1.0 - self.term_n) * self.target_v_sn)
        self.v_loss = _reg_loss(self.v_s0 - self.v_target)

        # Exact 1-step SMDP advantage target: R_1 + gamma_1 * V_target(s_1) - V_target(s_0)
        self.adv_target = tf.stop_gradient(self.r1 + self.disc1 * (1.0 - self.term1) * self.target_v_s1 - self.target_v_s0)

        # DAU centering baseline samples from current policy (stop_gradient for critic centering)
        m_samples = int(cfg.get('dau_center_samples', 4))
        s_rep = tf.tile(s_in, [m_samples, 1])
        mu_t_rep = tf.tile(mu_t, [m_samples, 1])
        std_t_rep = tf.tile(std_t, [m_samples, 1])
        a_t_rep = tf.tanh(mu_t_rep + std_t_rep * tf.random_normal(tf.shape(mu_t_rep)))
        mu_w_rep, log_std_w_rep = self._window_head(s_rep, a_t_rep)
        a_w_rep = tf.tanh(mu_w_rep + tf.exp(log_std_w_rep) * tf.random_normal(tf.shape(mu_w_rep)))
        act_rep_sg = tf.stop_gradient(tf.concat([a_w_rep, a_t_rep], axis=1))

        k_ens = int(cfg.get('ensemble_size', 4))
        self.adv_losses = []
        self.adv_exec_list = []
        self.adv_pi_list = []
        self.adv_vars_list = []
        b_size = tf.shape(self.s)[0]

        for k in range(k_ens):
            scope = 'jsac_adv_%d' % k
            raw_exec = self._adv_net(s_in, self.executed, scope)
            raw_rep = self._adv_net(s_rep, act_rep_sg, scope)
            center = tf.reduce_mean(tf.reshape(raw_rep, [m_samples, b_size, 1]), axis=0)
            centered_exec = raw_exec - center
            self.adv_exec_list.append(centered_exec)
            self.adv_losses.append(_reg_loss(centered_exec - self.adv_target))

            # Policy evaluation (center is independent of a_pi, so stop_gradient(center) is exact)
            raw_pi = self._adv_net(s_in, self.behavior_action, scope)
            self.adv_pi_list.append(raw_pi - tf.stop_gradient(center))
            self.adv_vars_list.append(self._scope_vars(scope, trainable=True))

        self.adv_loss = tf.add_n(self.adv_losses) / float(k_ens)
        self.min_adv_pi = tf.reduce_min(tf.concat(self.adv_pi_list, axis=1), axis=1, keep_dims=True)
        self.mean_adv_exec = tf.reduce_mean(tf.concat(self.adv_exec_list, axis=1), axis=1, keep_dims=True)

        # --- 5. Auto-tuned Dual Entropy Temperatures (alpha_w, alpha_t) ---
        self.log_alpha_w = tf.get_variable('log_alpha_w', initializer=math.log(float(cfg.get('init_alpha_w', 0.05))), dtype=tf.float32)
        self.log_alpha_t = tf.get_variable('log_alpha_t', initializer=math.log(float(cfg.get('init_alpha_t', 0.05))), dtype=tf.float32)
        max_alpha = float(cfg.get('max_alpha', 0.5))
        self.alpha_w = tf.clip_by_value(tf.exp(self.log_alpha_w), 1e-4, max_alpha)
        self.alpha_t = tf.clip_by_value(tf.exp(self.log_alpha_t), 1e-4, max_alpha)

        target_ent_w = float(cfg.get('target_entropy_w', -1.2))
        target_ent_t = float(cfg.get('target_entropy_t', -1.0))
        self.alpha_loss = (
            -tf.reduce_mean(self.log_alpha_w * tf.stop_gradient(logp_w + target_ent_w))
            - tf.reduce_mean(self.log_alpha_t * tf.stop_gradient(logp_t + target_ent_t))
        )
        self.entropy_w = -tf.reduce_mean(logp_w)
        self.entropy_t = -tf.reduce_mean(logp_t)

        # --- 6. Differentiable Canopy IBP Specification Losses & Primal-Dual Multipliers ---
        self._build_canopy_constraints()

        # --- 7. Unified Actor Objective (Zero 3-player game, Zero defensive patches) ---
        self.rl_actor_loss = tf.reduce_mean(
            -self.min_adv_pi + tf.stop_gradient(self.alpha_w) * logp_w + tf.stop_gradient(self.alpha_t) * logp_t
        )
        if cfg.get('canopy_ibp_enabled', True):
            self.ibp_penalty = (
                tf.stop_gradient(self.nu_cwnd) * self.v_cwnd
                + tf.stop_gradient(self.nu_mtp) * self.v_mtp
                + tf.stop_gradient(self.nu_joint) * self.v_joint
            )
            self.actor_loss = self.rl_actor_loss + self.ibp_penalty
        else:
            self.ibp_penalty = tf.constant(0.0, dtype=tf.float32)
            self.actor_loss = self.rl_actor_loss

        # --- 8. Optimizers & Target Updates ---
        self.v_vars = self._scope_vars('jsac_v', trainable=True)
        self.target_v_vars = self._scope_vars('jsac_target_v', trainable=False)
        v_all = self._scope_vars('jsac_v', trainable=False)
        assert len(v_all) == len(self.target_v_vars)
        self.target_init = [tf.assign(b, a) for a, b in zip(v_all, self.target_v_vars)]
        tau_v = float(cfg.get('tau_v', 0.01))
        self.target_update = [tf.assign(b, tau_v * a + (1.0 - tau_v) * b) for a, b in zip(v_all, self.target_v_vars)]

        self.v_opt = tf.train.AdamOptimizer(float(cfg.get('lr_v', 1e-3)), name='AdamJSACV')
        self.v_step = self.v_opt.minimize(self.v_loss, var_list=self.v_vars)

        self.adv_opts = [tf.train.AdamOptimizer(float(cfg.get('lr_adv', 1e-3)), name='AdamJSACAdv%d' % k) for k in range(k_ens)]
        adv_ops = [opt.minimize(loss, var_list=vs) for opt, loss, vs in zip(self.adv_opts, self.adv_losses, self.adv_vars_list)]
        with tf.control_dependencies(adv_ops):
            self.critic_step = tf.assign_add(self.global_step, 1)

        self.actor_opt = tf.train.AdamOptimizer(float(cfg.get('lr_actor', 3e-4)), name='AdamJSACActor')
        actor_grads = self.actor_opt.compute_gradients(self.actor_loss, var_list=self.actor_vars)
        actor_grads = [(tf.clip_by_norm(g, 10.0), v) for g, v in actor_grads if g is not None]
        self.actor_grad_norm = tf.global_norm([g for g, _ in actor_grads])
        self.actor_step = self.actor_opt.apply_gradients(actor_grads)

        self.alpha_opt = tf.train.AdamOptimizer(float(cfg.get('lr_alpha', 3e-4)), name='AdamJSACAlpha')
        self.alpha_step = self.alpha_opt.minimize(self.alpha_loss, var_list=[self.log_alpha_w, self.log_alpha_t])

        # Clean WSRL supervised warm-start op (aligns window mean to teacher, time mean to initial target)
        # Uses a separate optimizer so RL Adam moments start completely clean!
        self.ws_loss = (
            tf.reduce_mean(tf.square(self.det_a_w - self.ws_target_w))
            + 0.25 * tf.reduce_mean(tf.square(self.det_a_t - self.ws_target_t))
        )
        self.ws_opt = tf.train.AdamOptimizer(1e-3, name='AdamJSACWarmStart')
        self.ws_step = self.ws_opt.minimize(self.ws_loss, var_list=self.actor_vars)

    def _scope_vars(self, scope, trainable=False):
        tf = self.tf
        coll = tf.trainable_variables() if trainable else tf.global_variables()
        return sorted([v for v in coll if v.op.name.startswith(scope + '/') and '/Adam' not in v.op.name],
                      key=lambda v: v.op.name)

    def _time_head(self, s):
        tf = self.tf
        h_dim = int(self.cfg.get('time_hidden', 128))
        h = tf.nn.leaky_relu(_linear(tf, s, 'jsac_actor/time/fc1', h_dim), alpha=0.2)
        h = tf.nn.leaky_relu(_linear(tf, h, 'jsac_actor/time/fc2', h_dim), alpha=0.2)
        mu = _linear(tf, h, 'jsac_actor/time/mu', 1, init_scale=3e-3)
        log_std = tf.clip_by_value(_linear(tf, h, 'jsac_actor/time/log_std', 1, init_scale=3e-3) - 0.7, -4.0, 0.8)
        return mu, log_std

    def _window_head(self, s, a_t):
        tf = self.tf
        h_dim = int(self.cfg.get('window_hidden', 256))
        x = tf.concat([s, a_t], axis=1)
        h = tf.nn.leaky_relu(_linear(tf, x, 'jsac_actor/win/fc1', h_dim), alpha=0.2)
        h = tf.nn.leaky_relu(_linear(tf, h, 'jsac_actor/win/fc2', h_dim), alpha=0.2)
        mu = _linear(tf, h, 'jsac_actor/win/mu', 1, init_scale=3e-3)
        log_std = tf.clip_by_value(_linear(tf, h, 'jsac_actor/win/log_std', 1, init_scale=3e-3) - 1.2, -4.0, 0.5)
        return mu, log_std

    def _norm_s(self, s):
        if not self.obs_norm_enabled:
            return s
        return self.tf.clip_by_value((s - self.obs_mean) / self.obs_std, -10.0, 10.0)

    def _norm_s_ibp(self, s_lo, s_hi):
        if not self.obs_norm_enabled:
            return s_lo, s_hi
        # Exact monotonic affine IBP propagation (obs_std >= obs_norm_std_floor > 0)
        lo = self.tf.clip_by_value((s_lo - self.obs_mean) / self.obs_std, -10.0, 10.0)
        hi = self.tf.clip_by_value((s_hi - self.obs_mean) / self.obs_std, -10.0, 10.0)
        return lo, hi

    def init_obs_norm_from_states(self, session, states):
        """Initialize IBP-exact standardization from probe states without biasing 20ms-only clock dims."""
        if not self.obs_norm_enabled:
            return
        arr = np.asarray(states, dtype=np.float32)
        mu = np.mean(arr, axis=0, keepdims=True).astype(np.float32)
        sd = np.maximum(np.std(arr, axis=0, keepdims=True), float(self.cfg.get('obs_norm_std_floor', 0.20))).astype(np.float32)
        # Keep log-time / period-action dims (70..78) unshifted initially since probes are fixed-20ms
        mu[:, 70:79] = 0.0
        sd[:, 70:79] = 1.0
        session.run(self.set_obs_norm, {self._obs_mu_ph: mu, self._obs_sd_ph: sd})

    def _joint_ibp(self, s_lo, s_hi):
        """Sound joint IBP composition over the factorized deterministic mean actor."""
        tf = self.tf
        s_lo, s_hi = self._norm_s_ibp(s_lo, s_hi)
        # 1. Propagate state box through time mean network
        lo, hi = _ibp_linear(tf, s_lo, s_hi, 'jsac_actor/time/fc1')
        lo, hi = _ibp_leaky_relu(tf, lo, hi, alpha=0.2)
        lo, hi = _ibp_linear(tf, lo, hi, 'jsac_actor/time/fc2')
        lo, hi = _ibp_leaky_relu(tf, lo, hi, alpha=0.2)
        lo, hi = _ibp_linear(tf, lo, hi, 'jsac_actor/time/mu')
        a_t_lo, a_t_hi = _ibp_tanh(tf, lo, hi)

        # 2. Concatenate certified time box [a_t_lo, a_t_hi] with state box and propagate through window mean network
        w_in_lo = tf.concat([s_lo, a_t_lo], axis=1)
        w_in_hi = tf.concat([s_hi, a_t_hi], axis=1)
        lo, hi = _ibp_linear(tf, w_in_lo, w_in_hi, 'jsac_actor/win/fc1')
        lo, hi = _ibp_leaky_relu(tf, lo, hi, alpha=0.2)
        lo, hi = _ibp_linear(tf, lo, hi, 'jsac_actor/win/fc2')
        lo, hi = _ibp_leaky_relu(tf, lo, hi, alpha=0.2)
        lo, hi = _ibp_linear(tf, lo, hi, 'jsac_actor/win/mu')
        a_w_lo, a_w_hi = _ibp_tanh(tf, lo, hi)
        return a_w_lo, a_w_hi, a_t_lo, a_t_hi

    def _v_net(self, s, scope):
        tf = self.tf
        h_dim = int(self.cfg.get('v_hidden', 256))
        h = tf.nn.leaky_relu(_layer_norm(tf, _linear(tf, s, scope + '/fc1', h_dim), scope + '/ln1'), alpha=0.2)
        h = tf.nn.leaky_relu(_layer_norm(tf, _linear(tf, h, scope + '/fc2', h_dim), scope + '/ln2'), alpha=0.2)
        return _linear(tf, h, scope + '/out', 1, init_scale=3e-3)

    def _adv_net(self, s, a, scope):
        tf = self.tf
        h_dim = int(self.cfg.get('critic_hidden', 256))
        x = tf.concat([s, a], axis=1)
        h = tf.nn.leaky_relu(_layer_norm(tf, _linear(tf, x, scope + '/fc1', h_dim), scope + '/ln1'), alpha=0.2)
        h = tf.nn.leaky_relu(_layer_norm(tf, _linear(tf, h, scope + '/fc2', h_dim), scope + '/ln2'), alpha=0.2)
        return _linear(tf, h, scope + '/out', 1, init_scale=3e-3)

    def _build_canopy_constraints(self):
        tf, cfg = self.tf, self.cfg
        k_sub = int(cfg.get('canopy_k_sub', 2))
        ibp_cap = int(cfg.get('canopy_ibp_batch', 64))
        s_ibp = self.s[:ibp_cap]
        boxes = build_canopy_symbolic_boxes(tf, s_ibp, k_sub=k_sub)
        n = tf.shape(s_ibp)[0]

        # Batch all 4 box families into ONE single IBP forward pass for speed
        all_lo = tf.concat([boxes['local'][0], boxes['bad_delay'][0], boxes['bad_loss'][0], boxes['good'][0]], axis=0)
        all_hi = tf.concat([boxes['local'][1], boxes['bad_delay'][1], boxes['bad_loss'][1], boxes['good'][1]], axis=0)
        w_lo, w_hi, t_lo, t_hi = self._joint_ibp(all_lo, all_hi)
        ms_hi = _encoded_to_ms(tf, t_hi)

        # Slice back into the 4 families
        n_loc = n
        n_sub = k_sub * n
        sizes = [n_loc, n_sub, n_sub, n_sub]
        w_lo_loc, w_lo_bd, w_lo_bl, w_lo_gd = tf.split(w_lo, sizes, axis=0)
        w_hi_loc, w_hi_bd, w_hi_bl, w_hi_gd = tf.split(w_hi, sizes, axis=0)
        ms_hi_loc, ms_hi_bd, ms_hi_bl, _ = tf.split(ms_hi, sizes, axis=0)

        # Aggregate union bounds across k_sub sub-intervals per state
        w_hi_bd_max = tf.reduce_max(tf.reshape(w_hi_bd, [k_sub, n, 1]), axis=0)
        w_lo_gd_min = tf.reduce_min(tf.reshape(w_lo_gd, [k_sub, n, 1]), axis=0)

        # 1. CWND Canopy violation: safety (never expand on bad_delay/bad_loss) + monotonic liveness + anti-starvation
        cwnd_good_min = float(cfg.get('cwnd_good_min', -0.50))
        self.v_cwnd = (
            tf.reduce_mean(tf.square(tf.nn.relu(w_hi_bd)))
            + tf.reduce_mean(tf.square(tf.nn.relu(w_hi_bl)))
            + tf.reduce_mean(tf.square(tf.nn.relu(w_hi_bd_max - w_lo_gd_min)))
            + tf.reduce_mean(tf.square(tf.nn.relu(cwnd_good_min - w_lo_gd)))
        )

        # 2. MTP Canopy violation: M1 congestion reactivity + M2 Nyquist RTT stability
        react_ms = float(cfg.get('mtp_react_ms', 40.0))
        kappa = float(cfg.get('mtp_nyquist_kappa', 2.5))
        min_rtt_ms = tf.expm1(tf.clip_by_value(s_ibp[:, 79:80], 0.0, 20.0)) * 20.0
        tau_nyq = tf.clip_by_value(kappa * min_rtt_ms, 40.0, 160.0)
        self.v_mtp = (
            tf.reduce_mean(tf.square(tf.nn.relu((ms_hi_bd - react_ms) / 140.0)))
            + tf.reduce_mean(tf.square(tf.nn.relu((ms_hi_bl - react_ms) / 140.0)))
            + tf.reduce_mean(tf.square(tf.nn.relu((ms_hi_loc - tau_nyq) / 140.0)))
        )

        # 3. Joint (cwnd, MTP) Effort-Duration Coupling J1: |a_w|_max * (tau_hi - 20)/140 <= delta_joint
        delta_j = float(cfg.get('joint_effort_delta', 0.20))
        w_abs_max = tf.maximum(tf.abs(w_lo_loc), tf.abs(w_hi_loc))
        clock_hi_loc = (ms_hi_loc - 20.0) / 140.0
        self.v_joint = tf.reduce_mean(tf.square(tf.nn.relu(w_abs_max * clock_hi_loc - delta_j)))

        # Primal-Dual Lagrange Multipliers (standard integral or Stooke et al. ICML'20 PID-Lagrangian)
        d_init = float(cfg.get('dual_init', 1.0))
        d_max = float(cfg.get('dual_max', 50.0))
        d_lr = float(cfg.get('dual_lr', 0.05))
        self.nu_cwnd = tf.get_variable('dual_nu_cwnd', initializer=d_init, dtype=tf.float32, trainable=False)
        self.nu_mtp = tf.get_variable('dual_nu_mtp', initializer=d_init, dtype=tf.float32, trainable=False)
        self.nu_joint = tf.get_variable('dual_nu_joint', initializer=d_init, dtype=tf.float32, trainable=False)

        tol_c = float(cfg.get('dual_tol_cwnd', 1e-4))
        tol_m = float(cfg.get('dual_tol_mtp', 1e-4))
        tol_j = float(cfg.get('dual_tol_joint', 1e-4))
        if cfg.get('pid_lagrangian', False):
            kp = float(cfg.get('dual_kp', 1.0))
            ki = float(cfg.get('dual_ki', d_lr))
            kd = float(cfg.get('dual_kd', 0.25))
            pid_ops = []
            for name, nu_var, v_val, tol in (
                ('cwnd', self.nu_cwnd, self.v_cwnd, tol_c),
                ('mtp', self.nu_mtp, self.v_mtp, tol_m),
                ('joint', self.nu_joint, self.v_joint, tol_j),
            ):
                i_var = tf.get_variable('dual_pid_i_' + name, initializer=d_init, dtype=tf.float32, trainable=False)
                prev_var = tf.get_variable('dual_pid_prev_' + name, initializer=0.0, dtype=tf.float32, trainable=False)
                err = v_val - tol
                i_new = tf.clip_by_value(i_var + ki * err, 0.0, d_max)
                d_term = tf.nn.relu(v_val - prev_var)
                p_term = tf.nn.relu(err)
                nu_new = tf.clip_by_value(kp * p_term + i_new + kd * d_term, 0.0, d_max)
                with tf.control_dependencies([i_new, nu_new]):
                    upd_i = tf.assign(i_var, i_new)
                    upd_nu = tf.assign(nu_var, nu_new)
                    with tf.control_dependencies([upd_i, upd_nu]):
                        upd_prev = tf.assign(prev_var, v_val)
                pid_ops.append(upd_prev)
            self.dual_update = tf.group(*pid_ops)
        else:
            self.dual_update = tf.group(
                tf.assign(self.nu_cwnd, tf.clip_by_value(self.nu_cwnd + d_lr * (self.v_cwnd - tol_c), 0.0, d_max)),
                tf.assign(self.nu_mtp, tf.clip_by_value(self.nu_mtp + d_lr * (self.v_mtp - tol_m), 0.0, d_max)),
                tf.assign(self.nu_joint, tf.clip_by_value(self.nu_joint + d_lr * (self.v_joint - tol_j), 0.0, d_max)),
            )

    def warmstart_from_teacher(self, session, states, teacher_w, init_period_ms=40.0, steps=600, batch_size=128, seed=3):
        """Clean WSRL supervised initialization of the Linear+LeakyReLU actor (zero BN, zero permanent anchor)."""
        rng = np.random.RandomState(seed)
        n = len(states)
        t_enc = 2.0 * math.log(float(init_period_ms) / 20.0) / math.log(8.0) - 1.0
        target_w = np.asarray(teacher_w, dtype=np.float32).reshape(-1, 1)
        target_t = np.full((n, 1), t_enc, dtype=np.float32)
        last_loss = 0.0
        for _ in range(steps):
            idx = rng.randint(0, n, size=min(batch_size, n))
            _, last_loss = session.run([self.ws_step, self.ws_loss], {
                self.s: states[idx],
                self.ws_target_w: target_w[idx],
                self.ws_target_t: target_t[idx],
            })
        pred = session.run(self.action, {self.s: states})
        return dict(
            warmstart_steps=steps,
            final_loss=float(last_loss),
            window_mae=float(np.mean(np.abs(pred[:, :1] - target_w))),
            mean_init_ms=float(np.mean(20.0 * (8.0 ** ((pred[:, 1] + 1.0) / 2.0)))),
        )

    def digest(self, session):
        h = hashlib.sha256()
        for var, val in zip(self.online_vars, session.run(self.online_vars)):
            h.update(var.op.name.encode())
            h.update(np.ascontiguousarray(val).tobytes())
        return h.hexdigest()

    def save(self, session, prefix, counters=None):
        saver = self.tf.train.Saver(var_list=self.tf.global_variables() if self.training else self.online_vars + [self.schema])
        saver.save(session, str(prefix), write_meta_graph=False)
        Path(str(prefix) + '.dau_jsac.json').write_text(json.dumps(dict(
            schema=self.cfg.get('schema', SCHEMA_DAU_JSAC),
            cfg=self.cfg,
            counters=counters or {},
            state_dim=82,
            action_dim=2,
        ), indent=2) + '\n')

    def restore(self, session, prefix):
        rec = json.loads(Path(str(prefix) + '.dau_jsac.json').read_text())
        if rec['schema'] != self.cfg.get('schema', SCHEMA_DAU_JSAC):
            raise ValueError('DAU-JSAC checkpoint schema mismatch')
        selected = self.tf.global_variables() if self.training else self.online_vars + [self.schema]
        self.tf.train.Saver(var_list=selected).restore(session, str(prefix))
        return rec


class NumpyDAUJSACActor:
    """Zero-TensorFlow exact pure-NumPy deployment inference for Canopy-DAU-JSAC actors."""

    def __init__(self, prefix):
        import tensorflow as tf
        rec = json.loads(Path(str(prefix) + '.dau_jsac.json').read_text())
        self.cfg = rec['cfg']
        self.schema = rec['schema']
        reader = tf.train.NewCheckpointReader(str(prefix))
        self.obs_norm_enabled = bool(self.cfg.get('obs_norm_enabled', False))
        if self.obs_norm_enabled:
            self.obs_mean = reader.get_tensor('jsac_actor/obs_norm/mean').astype(np.float32)
            self.obs_std = reader.get_tensor('jsac_actor/obs_norm/std').astype(np.float32)
        else:
            self.obs_mean = None
            self.obs_std = None
        self.t_layers = [
            (reader.get_tensor('jsac_actor/time/%s/kernel' % n).astype(np.float32),
             reader.get_tensor('jsac_actor/time/%s/bias' % n).astype(np.float32))
            for n in ('fc1', 'fc2', 'mu')
        ]
        self.w_layers = [
            (reader.get_tensor('jsac_actor/win/%s/kernel' % n).astype(np.float32),
             reader.get_tensor('jsac_actor/win/%s/bias' % n).astype(np.float32))
            for n in ('fc1', 'fc2', 'mu')
        ]
        h = hashlib.sha256()
        for name in sorted(reader.get_variable_to_shape_map().keys()):
            if name.startswith('jsac_actor/'):
                h.update(name.encode())
                h.update(np.ascontiguousarray(reader.get_tensor(name)).tobytes())
        self._digest = h.hexdigest()

    def digest(self):
        return self._digest

    def act(self, state):
        s = np.asarray(state, dtype=np.float32).reshape(1, 82)
        if self.obs_norm_enabled:
            s = np.clip((s - self.obs_mean) / self.obs_std, -10.0, 10.0)
        h = s
        for w, b in self.t_layers[:2]:
            h = h.dot(w) + b
            h = np.where(h > 0.0, h, 0.2 * h)
        a_t = np.tanh(h.dot(self.t_layers[2][0]) + self.t_layers[2][1])
        h = np.concatenate([s, a_t], axis=1)
        for w, b in self.w_layers[:2]:
            h = h.dot(w) + b
            h = np.where(h > 0.0, h, 0.2 * h)
        a_w = np.tanh(h.dot(self.w_layers[2][0]) + self.w_layers[2][1])
        return np.array([float(a_w[0, 0]), float(a_t[0, 0])], dtype=np.float32)
