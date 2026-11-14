"""TF1 dual-branch actor and clipped twin Q, with explicit update ownership.

The migrated window uses the frozen Sonnet parameter names and its moving-BN
forward formula. No symbolic/QC graph is constructed. Four BN writers exist,
are fetched only with a joint window update, and never run during calibration.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import SCHEMA, STATE_DIM, CALIBRATION, FORWARD_ONLY, JOINT, period_action, softsign_inverse

WINDOW = 'abstract_actor/abstract_actor'
TARGET_WINDOW = 'target_actor/target_actor'


def migrate_array(name, value, shape):
    """Only the documented new state/time columns may be zero-extended."""
    value = np.asarray(value)
    if tuple(value.shape) == tuple(shape): return value.copy()
    if name == WINDOW + '/fc1/w' or name in ('critic/fc1/kernel', 'critic2/fc1/kernel'):
        if value.shape == (70, 256) and tuple(shape) == (82, 256):
            return np.pad(value, ((0, 12), (0, 0)), mode='constant')
    if name in ('critic/fc2/kernel', 'critic2/fc2/kernel'):
        if value.shape == (257, 256) and tuple(shape) == (258, 256):
            return np.pad(value, ((0, 1), (0, 0)), mode='constant')
    raise ValueError('Undeclared checkpoint shape migration: ' + name)


class Model:
    def __init__(self, cfg, training=False):
        import tensorflow as tf
        self.tf, self.cfg, self.training = tf, cfg, training
        self.s = tf.placeholder(tf.float32, (None, 82), name='state')
        self.phase = tf.get_variable('joint_phase', initializer=0, dtype=tf.int32, trainable=False)
        self.schema = tf.get_variable('joint_schema', initializer=SCHEMA, dtype=tf.string, trainable=False)
        self.rate_schema = tf.get_variable('joint_rate_schema', initializer=cfg['rate_mode'], dtype=tf.string, trainable=False)
        self.clock_schema = tf.get_variable('joint_clock_schema', initializer=cfg['clock_mode'], dtype=tf.string, trainable=False)
        self.joint_enabled = tf.equal(self.phase, JOINT)
        self.window_update_ops = []
        self.action = self._policy(self.s, False)
        self.window_vars = self._variables(WINDOW)
        self.time_vars = self._variables('mtp_actor')
        self.online_vars = self.window_vars + self.time_vars
        self.target_pairs = []
        if not training: return
        self.s1 = tf.placeholder(tf.float32, (None, 82), name='next_state')
        self.executed = tf.placeholder(tf.float32, (None, 2), name='executed_action')
        self.reward = tf.placeholder(tf.float32, (None, 1), name='interval_reward')
        self.discount = tf.placeholder(tf.float32, (None, 1), name='transition_discount')
        self.terminal = tf.placeholder(tf.float32, (None, 1), name='terminal')
        self.global_step = tf.train.get_or_create_global_step()
        next_action = self._policy(self.s1, True)
        self.target_action = next_action
        # Retain the scalar baseline's clipped target-window smoothing. The new
        # time coordinate is deliberately not smoothed (and fixed clocks stay fixed).
        noise = tf.clip_by_value(tf.random_normal(tf.shape(next_action[:, :1]), stddev=cfg['target_window_noise_std']),
                                 -cfg['target_window_noise_clip'], cfg['target_window_noise_clip'])
        self.target_policy = tf.concat([tf.clip_by_value(next_action[:, :1]+noise,-1.,1.), next_action[:, 1:]],axis=1)
        next_q = tf.minimum(self._critic(self.s1, self.target_policy, 'target_critic'),
                            self._critic(self.s1, self.target_policy, 'target_critic2'))
        self.target_q = next_q
        self.label = tf.stop_gradient(self.reward + self.discount * (1. - self.terminal) * next_q)
        self.q1 = self._critic(self.s, self.executed, 'critic')
        self.q2 = self._critic(self.s, self.executed, 'critic2')
        self.losses = [tf.reduce_mean(tf.square(self.label - q)) for q in (self.q1, self.q2)]
        self.actor_loss = -tf.reduce_mean(self._critic(self.s, self.action, 'critic'))
        self.actor_vars = self._variables(WINDOW, True) + self._variables('mtp_actor', True)
        self.window_grad = tf.global_norm(tf.gradients(self.actor_loss, self._variables(WINDOW, True)))
        self.time_grad = tf.global_norm(tf.gradients(self.actor_loss, self._variables('mtp_actor', True)))
        self.critic_optimizers = [tf.train.AdamOptimizer(cfg.get('lr_c', .001), name='AdamCritic%dJoint' % i) for i in (1, 2)]
        self.critic_parameters = [self._variables(n, True) for n in ('critic', 'critic2')]
        assert not set(self.critic_parameters[0]) & set(self.critic_parameters[1])
        updates = [optimizer.minimize(loss, var_list=variables)
                   for optimizer, loss, variables in zip(self.critic_optimizers, self.losses, self.critic_parameters)]
        with tf.control_dependencies(updates):
            self.critic_step = tf.assign_add(self.global_step, 2)
        self.actor_optimizer = tf.train.AdamOptimizer(cfg.get('lr_a', .0001), name='AdamJointActor')
        with tf.control_dependencies([tf.Assert(self.joint_enabled, [self.phase], name='actor_requires_joint_phase')]):
            self.actor_step = self.actor_optimizer.minimize(self.actor_loss, var_list=self.actor_vars, global_step=self.global_step)
        self._add_bn_writers()
        for source, target in ((WINDOW, TARGET_WINDOW), ('mtp_actor', 'target_mtp_actor'),
                               ('critic', 'target_critic'), ('critic2', 'target_critic2')):
            a = {v.op.name[len(source) + 1:]: v for v in self._variables(source)}
            b = {v.op.name[len(target) + 1:]: v for v in self._variables(target)}
            assert a and set(a) == set(b)
            for name in sorted(a):
                assert a[name].shape == b[name].shape
                self.target_pairs.append((a[name], b[name]))
        destinations = [b.op.name for _, b in self.target_pairs]
        assert len(destinations) == len(set(destinations)), 'Duplicate target destination'
        self.target_init = [tf.assign(b, a) for a, b in self.target_pairs]
        self.target_update = []
        for a, b in self.target_pairs:
            def update(a=a, b=b): return tf.assign(b, .001 * a + .999 * b)
            # Calibration must preserve actor/target weights AND statistics exactly.
            if a.op.name.startswith((WINDOW + '/', 'mtp_actor/')):
                self.target_update.append(tf.cond(self.joint_enabled, update, lambda b=b: tf.identity(b)))
            else: self.target_update.append(update())

    def _variables(self, scope, trainable=False):
        tf = self.tf
        collection = tf.trainable_variables() if trainable else tf.global_variables()
        return sorted([v for v in collection if v.op.name.startswith(scope + '/') and '/Adam' not in v.op.name], key=lambda v: v.op.name)

    def _linear(self, state, scope, width, sonnet=False):
        tf = self.tf
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w' if sonnet else 'kernel', (int(state.shape[-1]), width), initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable('b' if sonnet else 'bias', (width,), initializer=tf.zeros_initializer())
        return tf.matmul(state, w) + b

    def _bn(self, value, scope):
        tf = self.tf
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            mean = tf.get_variable('moving_mean', (1, 256), initializer=tf.zeros_initializer(), trainable=False)
            var = tf.get_variable('moving_variance', (1, 256), initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', (1, 256), initializer=tf.zeros_initializer())
        return tf.nn.batch_normalization(value, mean, var, beta, None, .001)

    def _policy(self, state, target):
        tf = self.tf; p = TARGET_WINDOW if target else WINDOW
        h = tf.nn.leaky_relu(self._bn(self._linear(state, p + '/fc1', 256, True), p + '/bn1'))
        h = tf.nn.leaky_relu(self._bn(self._linear(h, p + '/fc2', 256, True), p + '/bn2'))
        window = tf.tanh(self._linear(h, p + '/fc1_1', 1, True))
        t = 'target_mtp_actor' if target else 'mtp_actor'
        z = tf.nn.leaky_relu(self._linear(state, t + '/fc1', 64))
        z = tf.nn.leaky_relu(self._linear(z, t + '/fc2', 64))
        with tf.variable_scope(t + '/output', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('kernel', (64, 1), initializer=tf.zeros_initializer())
            coverage=self.cfg.get('time_initializer')=='feedback_coverage_v1'
            initial=0. if coverage else softsign_inverse(period_action(self.cfg.get('time_init_ms',40.)))
            b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(initial))
        residual=tf.matmul(z,w)+b
        if coverage:
            # Absolute minRTT lives at79 in the versioned state; no capacity,
            # trace identity, future signal, action mask or window fallback.
            minimum_ms=tf.expm1(tf.clip_by_value(state[:,79:80],0.,20.))*20.
            prior_ms=tf.where(state[:,66:67]>0.,tf.ones_like(minimum_ms)*20.5,
                              tf.clip_by_value(minimum_ms,40.,120.))
            prior_action=2.*tf.log(prior_ms/20.)/np.log(8.)-1.
            residual=residual+prior_action/(1.-tf.abs(prior_action))
        learned_time = tf.nn.softsign(residual)
        fixed = self.cfg.get('fixed_mtp_ms')
        if fixed is not None:
            timing = tf.ones_like(learned_time) * period_action(float(fixed))
        elif coverage:
            timing=learned_time
        else:
            timing = tf.cond(tf.greater_equal(self.phase, FORWARD_ONLY), lambda: learned_time,
                             lambda: tf.ones_like(learned_time) * -1.)
        return tf.concat([window, timing], axis=1)

    def _critic(self, state, action, scope):
        tf = self.tf
        h = tf.nn.leaky_relu(self._linear(state, scope + '/fc1', 256))
        h = tf.nn.leaky_relu(self._linear(tf.concat([h, action], axis=1), scope + '/fc2', 256))
        return self._linear(h, scope + '/dense', 1)

    def _add_bn_writers(self):
        tf = self.tf
        variables = {v.op.name: v for v in self.window_vars}
        x = self.s
        for fc, bn in (('fc1', 'bn1'), ('fc2', 'bn2')):
            y = tf.matmul(x, variables[WINDOW + '/' + fc + '/w']) + variables[WINDOW + '/' + fc + '/b']
            mean, variance = tf.nn.moments(y, [0], keep_dims=True)
            x = tf.nn.leaky_relu(tf.nn.batch_normalization(y, mean, variance, variables[WINDOW + '/' + bn + '/beta'], None, .001))
            for name, value in (('moving_mean', mean), ('moving_variance', variance)):
                v = variables[WINDOW + '/' + bn + '/' + name]
                # Sonnet1.34 uses assign_sub((old-value)*(1-decay)), zero_debias=False.
                def update(v=v, value=value): return tf.assign_sub(v, (v - value) * .001)
                self.window_update_ops.append(tf.cond(self.joint_enabled, update, lambda v=v: tf.identity(v)))
        assert len(self.window_update_ops) == 4

    def initialize_baseline(self, session, checkpoint, initialize_critics=True):
        tf = self.tf
        root = Path(__file__).resolve().parents[2]
        contract = json.loads((root / 'CURRENT_BASELINE.json').read_text())['models_and_natives']
        for suffix in ('.index', '.data-00000-of-00001'):
            name = 'models/healthy-baseline-seed3/training/step-6013196' + suffix
            if hashlib.sha256(Path(str(checkpoint) + suffix).read_bytes()).hexdigest() != contract[name]['sha256']:
                raise ValueError('Source tensor bytes do not match the frozen healthy seed3')
        rd = tf.train.NewCheckpointReader(str(checkpoint))
        names = rd.get_variable_to_shape_map()
        if names.get(WINDOW + '/fc1/w') != [70, 256] or names.get(WINDOW + '/fc1_1/w') != [256, 1]:
            raise ValueError('Only the declared70/1 baseline may initialize this schema')
        variables = list(self.window_vars)
        if self.training and initialize_critics: variables += self._variables('critic') + self._variables('critic2')
        assignments = []
        for var in variables:
            if var.op.name not in names: raise ValueError('Missing source tensor: ' + var.op.name)
            value = migrate_array(var.op.name, rd.get_tensor(var.op.name), var.shape.as_list())
            if not np.isfinite(value).all(): raise ValueError('Nonfinite source tensor')
            assignments.append(tf.assign(var, value))
        session.run(assignments)
        if self.training: session.run(self.target_init)
        return dict(source=str(checkpoint), source_state_dim=70, source_action_dim=1, state_dim=82,
                    new_inputs_zero=True, optimizer_replay_counters_fresh=True, complete_resume=False,
                    critics='approximate_initialization_needs_calibration' if self.training and initialize_critics else 'not_restored',
                    targets='hard_copy_of_migrated_online', copied=[v.op.name for v in variables])

    def feed(self, batch):
        return dict(zip((self.s, self.executed, self.reward, self.s1, self.terminal, self.discount),
                        (batch[0], batch[1], batch[2], batch[3], batch[4], batch[6])))

    def digest(self, session, window_only=False):
        h = hashlib.sha256()
        for var, value in zip(self.window_vars if window_only else self.online_vars,
                              session.run(self.window_vars if window_only else self.online_vars)):
            h.update(var.op.name.encode()); h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()

    def save_manifest(self, prefix, cfg, counters):
        Path(str(prefix) + '.joint.json').write_text(json.dumps(dict(schema=SCHEMA, cfg=cfg, counters=counters,
            state_dim=82, action_dim=2, no_complete_resume=True), indent=2) + '\n')

    def restore(self, session, prefix, allow_diagnostic=False):
        tf = self.tf
        record = json.loads(Path(str(prefix) + '.joint.json').read_text())
        if record['schema'] != SCHEMA: raise ValueError('Checkpoint schema mismatch')
        for key in ('rate_mode', 'clock_mode', 'utility_normalization', 'decision_cost', 'fixed_mtp_ms',
                    'time_initializer','calibration_main_mode'):
            if record['cfg'].get(key) != self.cfg.get(key): raise ValueError('Checkpoint semantics mismatch: ' + key)
        selected = tf.global_variables() if self.training else self.online_vars + [self.phase, self.schema, self.rate_schema, self.clock_schema]
        tf.train.Saver(var_list=selected).restore(session, str(prefix))
        if session.run(self.schema).decode() != SCHEMA: raise ValueError('Saved schema tensor mismatch')
        if session.run(self.rate_schema).decode() != self.cfg['rate_mode'] or session.run(self.clock_schema).decode() != self.cfg['clock_mode']:
            raise ValueError('Saved measurement schema tensor mismatch')
        if not allow_diagnostic and int(session.run(self.phase)) != JOINT:
            raise ValueError('Calibration-only checkpoint is not a trained joint endpoint')
        return record
