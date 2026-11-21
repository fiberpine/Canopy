"""Opt-in continuous duration and same-decision window conditioning.

The five coverage buckets are diagnostics, never a set of policy actions.
All exploration inputs are supplied by the actor's seeded NumPy generator.
"""
import math
import numpy as np

ARCHITECTURE = 'continuous_conditioned_v4'
ARCHITECTURE_V5 = 'continuous_conditioned_v5'
ARCHITECTURES = (ARCHITECTURE, ARCHITECTURE_V5)
SCHEMA = 'healthy-seed3-joint82-continuous-conditioned-v4'
SCHEMA_V5 = 'healthy-seed3-joint82-continuous-conditioned-v5'
CONDITION = 'conditioned_window'
TARGET_CONDITION = 'target_conditioned_window'


def enabled(cfg):
    return cfg.get('actor_architecture', 'independent_v2') in ARCHITECTURES


def is_v5(cfg):
    return cfg.get('actor_architecture', 'independent_v2') == ARCHITECTURE_V5


def schema(cfg):
    from .core import SCHEMA as legacy
    if is_v5(cfg): return SCHEMA_V5
    return SCHEMA if enabled(cfg) else legacy


def exploration_inputs(cfg, rng):
    """Local logit noise plus a stratified continuous log-duration proposal."""
    return dict(window_noise=float(rng.normal(0., cfg['exploration_std'])),
                time_noise=float(rng.normal(0., cfg['time_logit_exploration_std'])),
                probability=float(rng.random_sample()),
                global_fraction=(int(rng.randint(5)) + float(rng.random_sample())) / 5.)


def validate(cfg):
    if cfg.get('actor_architecture', 'independent_v2') not in ('independent_v2', ARCHITECTURE, ARCHITECTURE_V5):
        raise ValueError('Unknown actor architecture')
    if not enabled(cfg):
        if cfg.get('actor_recipe') in ARCHITECTURES or cfg.get('teacher_anchor_clock_weighting') == 'base20_only' \
                or cfg.get('critic_period_residual_lr') is not None or cfg.get('critic_common_calibration_updates',0) \
                or cfg.get('critic_period_zero_anchor', False) or cfg.get('envelope_stop_gradient_duration', False) \
                or cfg.get('pre_adapt_conditioned_window_in_forward_phase', False) \
                or cfg.get('conditioned_window_actor_lr') is not None or float(cfg.get('time_barrier_weight', 0.)) != 0. \
                or float(cfg.get('critic_period_residual_l2', 0.)) != 0. \
                or cfg.get('calibration_critic_tau') is not None \
                or int(cfg.get('common_calibration_n_step', 1)) != 1:
            raise ValueError('Conditioned training semantics require the conditioned architecture')
        return
    residual_lr = cfg.get('critic_period_residual_lr')
    if residual_lr is not None:
        residual_lr = float(residual_lr)
        if not math.isfinite(residual_lr) or not 0 < residual_lr <= float(cfg.get('lr_c', .001)):
            raise ValueError('Period-residual critic learning rate must be in (0, lr_c]')
    common = cfg.get('critic_common_calibration_updates',0)
    if (not isinstance(common,int) or isinstance(common,bool) or common<0
            or common>=int(cfg.get('calibration_updates',5000))):
        raise ValueError('Common-value calibration must be an integer below the actor calibration minimum')
    if common and residual_lr is None:
        raise ValueError('Common-value calibration requires a separate period-residual Adam')
    expected_recipe = ARCHITECTURE_V5 if is_v5(cfg) else ARCHITECTURE
    required = dict(actor_recipe=expected_recipe, time_parameterization='physical_sigmoid_v1',
                    teacher_anchor_clock_weighting='base20_only', window_bn_statistics='frozen_migrated',
                    time_initializer='feedback_coverage_v1', calibration_main_mode='policy_initializer',
                    time_feature_normalization='unit_l2_v1', clock_mode='scheduled',
                    rate_mode='acked_legacy_peak', utility_normalization='legacy_delivery_peak')
    for name, value in required.items():
        if cfg.get(name) != value: raise ValueError('Continuous contract mismatch: ' + name)
    if is_v5(cfg):
        cond_lr = float(cfg.get('conditioned_window_actor_lr', 0.))
        if not math.isfinite(cond_lr) or cond_lr <= 0:
            raise ValueError('v5 requires positive conditioned_window_actor_lr')
        barrier = float(cfg.get('time_barrier_weight', 0.))
        if not math.isfinite(barrier) or not 0. <= barrier <= 0.05:
            raise ValueError('v5 time_barrier_weight must be in [0, 0.05]')
        for bname in ('critic_period_zero_anchor', 'envelope_stop_gradient_duration', 'pre_adapt_conditioned_window_in_forward_phase'):
            if not isinstance(cfg.get(bname, False), bool):
                raise ValueError('Boolean v5 flag required: ' + bname)
        l2 = float(cfg.get('critic_period_residual_l2', 0.))
        if not math.isfinite(l2) or not 0. <= l2 <= 0.1:
            raise ValueError('Invalid critic_period_residual_l2')
        nstep = int(cfg.get('common_calibration_n_step', 1))
        if nstep < 1 or nstep > 64:
            raise ValueError('Invalid common_calibration_n_step')
        ctau = cfg.get('calibration_critic_tau')
        if ctau is not None and (not math.isfinite(float(ctau)) or not 0 < float(ctau) <= 1.):
            raise ValueError('Invalid calibration_critic_tau')
    if cfg.get('fixed_mtp_ms') is not None:
        raise ValueError('Continuous training has no fixed-period policy; use explicit diagnostic overrides')
    if not 0 < float(cfg.get('joint_period_probe_probability', 0)) <= .2:
        raise ValueError('Continuous training needs bounded full-range exploration')
    noise = float(cfg.get('time_logit_exploration_std', 0))
    if not math.isfinite(noise) or not 0 < noise <= 1:
        raise ValueError('Local continuous exploration scale must be in (0,1]')


def _window(model, base, features, duration, target):
    tf = model.tf
    scope = TARGET_CONDITION if target else CONDITION
    clock = tf.log(duration / 20.)
    hidden = tf.nn.leaky_relu(model._linear(tf.concat([features, clock], 1), scope + '/joint', 32))
    with tf.variable_scope(scope + '/output', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('kernel', (32, 1), initializer=tf.zeros_initializer())
        b = tf.get_variable('bias', (1,), initializer=tf.zeros_initializer())
    return tf.tanh(base + clock * (tf.matmul(hidden, w) + b))


def _duration(tf, logit):
    return 20. + 140. * tf.sigmoid(logit)


def _encoded(tf, duration):
    return 2. * tf.log(duration / 20.) / np.log(8.) - 1.


def critic_residual(model, state, action, scope):
    """Zero-initialized Q correction with explicit continuous physical features.

    Planned duration is an input, not a substitution for actual replay duration.
    The original migrated Q remains the initial function. No reward/capacity or
    analytic action-value oracle is inserted in this branch.
    """
    tf = model.tf
    scope += '/period_residual'
    z = tf.nn.leaky_relu(model._linear(state, scope + '/state', 64))
    z = tf.nn.l2_normalize(z, axis=1, epsilon=1e-12)
    duration = 20. * tf.pow(8., (action[:, 1:2]+1.)/2.)
    clock = (duration-20.)/140.
    window = action[:, :1]
    inputs = tf.concat([z, window, clock, tf.square(clock), tf.square(window), window*clock], 1)
    h = tf.nn.leaky_relu(model._linear(inputs, scope + '/joint', 64))
    # Linear physical features need not wait for state-dominated hidden ReLUs
    # to acquire breakpoints before learning a small time-dependent difference.
    inputs = tf.concat([h, window, clock, tf.square(clock), tf.square(window), window*clock], 1)
    with tf.variable_scope(scope + '/output', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('kernel', (int(inputs.shape[-1]), 1), initializer=tf.zeros_initializer())
        b = tf.get_variable('bias', (1,), initializer=tf.zeros_initializer())
    raw = tf.matmul(inputs, w) + b
    if model.cfg.get('critic_period_zero_anchor', False):
        return clock * raw
    return raw


def policy(model, state, target):
    tf, cfg = model.tf, model.cfg
    p = 'target_actor/target_actor' if target else 'abstract_actor/abstract_actor'
    h = tf.nn.leaky_relu(model._bn(model._linear(state, p + '/fc1', 256, True), p + '/bn1'))
    h = tf.nn.leaky_relu(model._bn(model._linear(h, p + '/fc2', 256, True), p + '/bn2'))
    base = model._linear(h, p + '/fc1_1', 1, True)
    c = TARGET_CONDITION if target else CONDITION
    features = tf.nn.leaky_relu(model._linear(state, c + '/state', 64))
    t = 'target_mtp_actor' if target else 'mtp_actor'
    z = tf.nn.leaky_relu(model._linear(state, t + '/fc1', 64))
    z = tf.nn.leaky_relu(model._linear(z, t + '/fc2', 64))
    z = tf.nn.l2_normalize(z, axis=1, epsilon=1e-12)
    with tf.variable_scope(t + '/output', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('kernel', (64, 1), initializer=tf.zeros_initializer())
        b = tf.get_variable('bias', (1,), initializer=tf.zeros_initializer())
    minimum = tf.expm1(tf.clip_by_value(state[:, 79:80], 0., 20.)) * 20.
    prior = tf.where(state[:, 66:67] > 0., tf.ones_like(minimum) * 20.5,
                     tf.clip_by_value(minimum, 40., 120.))
    fraction = (prior - 20.) / 140.
    logit = tf.log(fraction) - tf.log1p(-fraction) + tf.matmul(z, w) + b
    duration = _duration(tf, logit)
    cond_duration = tf.stop_gradient(duration) if cfg.get('envelope_stop_gradient_duration', False) else duration
    window = _window(model, base, features, cond_duration, target)
    if not target:
        model.base_window_action = tf.tanh(base)
        model.time_features, model.time_logit, model.planned_ms = z, logit, duration
        model.time_barrier_loss = tf.reduce_mean(tf.nn.softplus(-logit) + tf.nn.softplus(logit))
        model._condition_context = (base, features)
    return tf.concat([window, _encoded(tf, duration)], 1)


def conditioned_window_objective(model):
    """Train conditioned_window across both policy periods and continuous [20, 160]ms."""
    tf = model.tf
    base, features = model._condition_context
    sg_base = tf.stop_gradient(base)
    sg_duration = tf.stop_gradient(model.planned_ms)
    policy_win = _window(model, sg_base, features, sg_duration, False)
    policy_act = tf.concat([policy_win, _encoded(tf, sg_duration)], 1)
    probe_duration = 20. * tf.pow(8., tf.random_uniform(tf.shape(sg_duration), 0., 1.))
    probe_win = _window(model, sg_base, features, probe_duration, False)
    probe_act = tf.concat([probe_win, _encoded(tf, probe_duration)], 1)
    q_policy = model._critic(model.s, policy_act, 'critic')
    q_probe = model._critic(model.s, probe_act, 'critic')
    return -0.5 * (tf.reduce_mean(q_policy) + tf.reduce_mean(q_probe))


def behavior(model):
    """Reuse the online trunks in ONE session call, condition on executed T."""
    tf, cfg = model.tf, model.cfg
    base, features = model._condition_context
    model.behavior_inputs = {name: tf.placeholder(tf.float32, (), name='behavior_' + name)
                             for name in ('window_noise', 'time_noise', 'probability', 'global_fraction')}
    inputs = model.behavior_inputs
    joint = model.joint_enabled
    probability = tf.where(joint, float(cfg['joint_period_probe_probability']), float(cfg['period_probe_probability']))
    model.behavior_probe = inputs['probability'] < probability
    local = _duration(tf, model.time_logit + tf.where(joint, inputs['time_noise'], 0.))
    proposed = tf.ones_like(local) * (20. * tf.pow(8., inputs['global_fraction']))
    duration = tf.where(model.behavior_probe, proposed, local)
    window = _window(model, base, features, duration, False)
    window = tf.clip_by_value(window + tf.where(joint, inputs['window_noise'], 0.), -1., 1.)
    model.behavior_action = tf.concat([window, _encoded(tf, duration)], 1)
    # Read-only same-weight interventions; caller must record the override.
    model.diagnostic_ms = tf.placeholder(tf.float32, (None, 1), name='diagnostic_duration_ms')
    with tf.control_dependencies([tf.assert_greater_equal(model.diagnostic_ms, 20.),
                                  tf.assert_less_equal(model.diagnostic_ms, 160.)]):
        model.diagnostic_action = tf.concat([_window(model, base, features, model.diagnostic_ms, False),
                                            _encoded(tf, model.diagnostic_ms)], 1)


def value_diagnostics(model, session, batches):
    """Same saved inputs and two action sweeps; learned Q is NOT ground truth."""
    states = [batch[0][:8] for batch in batches if batch is not None]
    if not states: return dict(rows=0, value_ordering_verified=False)
    states = np.concatenate(states)
    n = len(states)
    repeated = np.tile(states, (5, 1))
    periods = np.repeat(np.array([20.,40.,80.,120.,160.],np.float32),n)[:,None]
    pure = session.run(model.action,{model.s:states})
    conditional = session.run(model.diagnostic_action,{model.s:repeated,model.diagnostic_ms:periods})
    unchanged_window = conditional.copy(); unchanged_window[:,0] = np.tile(pure[:,0],5)
    result = dict(rows=n, periods_ms=[20,40,80,120,160], value_ordering_verified=False,
                  sweep_is_learned_Q_not_counterfactual_truth=True,
                  planned_ms_mean=float(np.mean(20.*8.**((pure[:,1]+1.)/2.))),
                  time_low_boundary_fraction=float(np.mean(pure[:,1]<=-.99)),
                  time_high_boundary_fraction=float(np.mean(pure[:,1]>=.99)))
    for name,actions in (('conditioned_window',conditional),('unchanged_window',unchanged_window)):
        qs=session.run([model.q1,model.q2],{model.s:repeated,model.executed:actions})
        advantages=[(q.reshape(5,n)-q.reshape(5,n)[:1]).mean(axis=1) for q in qs]
        result[name]=dict(twin_mean_advantages=[x.tolist() for x in advantages],
            maximum_twin_advantage_disagreement=float(np.max(np.abs(advantages[0]-advantages[1]))))
    return result
