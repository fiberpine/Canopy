"""TF1 dual-branch actor and clipped twin Q, with explicit update ownership.

The migrated window uses the frozen Sonnet parameter names and its moving-BN
forward formula. No symbolic/QC graph is constructed. Four BN writers exist,
are fetched only with a joint window update, and never run during calibration.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import SCHEMA, STATE_DIM, CALIBRATION, FORWARD_ONLY, JOINT, period_action, softsign_inverse, bootstrap_terminal
from . import continuous

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
        self.schema = tf.get_variable('joint_schema', initializer=continuous.schema(cfg), dtype=tf.string, trainable=False)
        self.rate_schema = tf.get_variable('joint_rate_schema', initializer=cfg['rate_mode'], dtype=tf.string, trainable=False)
        self.clock_schema = tf.get_variable('joint_clock_schema', initializer=cfg['clock_mode'], dtype=tf.string, trainable=False)
        self.joint_enabled = tf.equal(self.phase, JOINT)
        self.window_update_ops = []
        self.action = self._policy(self.s, False)
        self.base_window_vars = self._variables(WINDOW)
        self.window_vars = self.base_window_vars + self._variables(continuous.CONDITION)
        self.time_vars = self._variables('mtp_actor')
        self.online_vars = self.window_vars + self.time_vars
        if continuous.enabled(cfg): continuous.behavior(self)
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
        self.policy_loss = -tf.reduce_mean(self._critic(self.s, self.action, 'critic'))
        self.teacher_output = None;self.teacher_reference = None
        self.anchor_loss = tf.constant(0.,tf.float32)
        if float(cfg.get('teacher_anchor_weight',0))>0:
            self.teacher_output=self._reference_window(self.s)
            self.teacher_reference=tf.placeholder(tf.float32,(None,1),name='same_state_frozen_teacher')
            anchored = self.base_window_action if continuous.enabled(cfg) else self.action[:,:1]
            difference=tf.abs(anchored-tf.stop_gradient(self.teacher_reference))
            delta=float(cfg.get('teacher_anchor_delta',.1))
            anchor=tf.where(difference<=delta,.5*tf.square(difference),delta*(difference-.5*delta))
            if cfg.get('teacher_anchor_clock_weighting','constant')=='base_over_planned_squared_stopgrad':
                planned_ms=20.*tf.pow(8.,(self.action[:,1:2]+1.)/2.)
                anchor=anchor*tf.stop_gradient(tf.square(20./planned_ms))
            self.anchor_loss=tf.reduce_mean(anchor)
        self.actor_loss = self.policy_loss+float(cfg.get('teacher_anchor_weight',0))*self.anchor_loss
        window_trainable = self._variables(WINDOW, True) + self._variables(continuous.CONDITION, True)
        self.actor_vars = window_trainable + self._variables('mtp_actor', True)
        self.window_grad = tf.global_norm(tf.gradients(self.actor_loss, window_trainable))
        self.time_grad = tf.global_norm(tf.gradients(self.actor_loss, self._variables('mtp_actor', True)))
        self.critic_optimizers = [tf.train.AdamOptimizer(cfg.get('lr_c', .001), name='AdamCritic%dJoint' % i) for i in (1, 2)]
        self.critic_parameters = [self._variables(n, True) for n in ('critic', 'critic2')]
        assert not set(self.critic_parameters[0]) & set(self.critic_parameters[1])
        residual_lr = cfg.get('critic_period_residual_lr') if continuous.enabled(cfg) else None
        if residual_lr is None:
            updates = [optimizer.minimize(loss, var_list=variables)
                       for optimizer, loss, variables in zip(self.critic_optimizers, self.losses, self.critic_parameters)]
            self.critic_adam_applies_per_step = 2
        else:
            # The migrated backbone keeps lr_c. The new zero-initialized period
            # residual gets its own Adam: with ~69 output weights each moving
            # ~lr per step, lr=1e-3 jitters the period advantage by ~1e-2,
            # larger than the 0.01-cost single-step margin it must resolve.
            # Gradients are computed once, from the SAME pre-update critic.
            self.critic_period_optimizers = [tf.train.AdamOptimizer(float(residual_lr), name='AdamCritic%dPeriod' % i) for i in (1, 2)]
            updates = []
            for main_optimizer, period_optimizer, loss, variables in zip(self.critic_optimizers, self.critic_period_optimizers,
                                                                         self.losses, self.critic_parameters):
                period = [v for v in variables if '/period_residual/' in v.op.name]
                main = [v for v in variables if '/period_residual/' not in v.op.name]
                assert period and main and len(period) + len(main) == len(variables)
                pairs = [(g, v) for g, v in zip(tf.gradients(loss, main + period), main + period) if g is not None]
                gradients = tf.tuple([g for g, _ in pairs])
                pairs = [(g, v) for g, (_, v) in zip(gradients, pairs)]
                main_pairs = [(g, v) for g, v in pairs if '/period_residual/' not in v.op.name]
                period_pairs = [(g, v) for g, v in pairs if '/period_residual/' in v.op.name]
                assert main_pairs and period_pairs, 'Both critic parameter groups need gradients'
                updates.append(main_optimizer.apply_gradients(main_pairs))
                updates.append(period_optimizer.apply_gradients(period_pairs))
            self.critic_adam_applies_per_step = 4
        with tf.control_dependencies(updates):
            self.critic_step = tf.assign_add(self.global_step, 2)
        self.actor_optimizer = tf.train.AdamOptimizer(cfg.get('lr_a', .0001), name='AdamJointActor')
        self.actor_optimizer_ops_per_step=1
        with tf.control_dependencies([tf.Assert(self.joint_enabled, [self.phase], name='actor_requires_joint_phase')]):
            if cfg.get('actor_recipe') in ('cost_preserve_v2', continuous.ARCHITECTURE):
                self.window_actor_optimizer=tf.train.AdamOptimizer(cfg['window_actor_lr'],name='AdamPreservedWindow')
                self.time_actor_optimizer=tf.train.AdamOptimizer(cfg['time_actor_lr'],name='AdamPreservedTime')
                window_gradients=self.window_actor_optimizer.compute_gradients(self.actor_loss,var_list=window_trainable)
                time_gradients=self.time_actor_optimizer.compute_gradients(self.actor_loss,var_list=self._variables('mtp_actor',True))
                effective_window=[(g,v) for g,v in window_gradients if g is not None]
                effective_time=[(g,v) for g,v in time_gradients if g is not None]
                if continuous.enabled(cfg):
                    # Both partial derivatives see the SAME pre-update joint
                    # policy, even though two Adam optimizers own the variables.
                    gradients=tf.tuple([g for g,_ in effective_window+effective_time])
                    split=len(effective_window)
                    effective_window=list(zip(gradients[:split],[v for _,v in effective_window]))
                    effective_time=list(zip(gradients[split:],[v for _,v in effective_time]))
                window_step=self.window_actor_optimizer.apply_gradients(effective_window)
                time_step=self.time_actor_optimizer.apply_gradients(effective_time) if effective_time else tf.no_op()
                self.actor_optimizer_ops_per_step=1+int(bool(effective_time))
                with tf.control_dependencies([window_step,time_step]):self.actor_step=tf.assign_add(self.global_step,1)
            else:
                self.actor_step = self.actor_optimizer.minimize(self.actor_loss, var_list=self.actor_vars, global_step=self.global_step)
        self._add_bn_writers()
        scopes = [(WINDOW, TARGET_WINDOW), ('mtp_actor', 'target_mtp_actor'),
                  ('critic', 'target_critic'), ('critic2', 'target_critic2')]
        if continuous.enabled(cfg): scopes.append((continuous.CONDITION, continuous.TARGET_CONDITION))
        for source, target in scopes:
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
            frozen_stat=(cfg.get('window_bn_statistics')=='frozen_migrated' and a.op.name.startswith(WINDOW+'/')
                         and a.op.name.endswith(('/moving_mean','/moving_variance')))
            if frozen_stat:
                self.target_update.append(tf.identity(b));continue
            # Calibration must preserve actor/target weights AND statistics exactly.
            if a.op.name.startswith((WINDOW + '/', 'mtp_actor/', continuous.CONDITION + '/')):
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
        if continuous.enabled(self.cfg): return continuous.policy(self, state, target)
        tf = self.tf; p = TARGET_WINDOW if target else WINDOW
        h = tf.nn.leaky_relu(self._bn(self._linear(state, p + '/fc1', 256, True), p + '/bn1'))
        h = tf.nn.leaky_relu(self._bn(self._linear(h, p + '/fc2', 256, True), p + '/bn2'))
        window = tf.tanh(self._linear(h, p + '/fc1_1', 1, True))
        t = 'target_mtp_actor' if target else 'mtp_actor'
        z = tf.nn.leaky_relu(self._linear(state, t + '/fc1', 64))
        z = tf.nn.leaky_relu(self._linear(z, t + '/fc2', 64))
        if self.cfg.get('time_feature_normalization','none')=='unit_l2_v1':
            z=tf.nn.l2_normalize(z,axis=1,epsilon=1e-12)
        if not target:self.time_features=z
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
        base_action = action
        if continuous.enabled(self.cfg):
            # Preserve a scalar-window Q backbone. Time dependence belongs to
            # the physical-duration branch, not two competing coordinate maps.
            base_action = tf.concat([action[:, :1], -tf.ones_like(action[:, 1:2])], 1)
        h = tf.nn.leaky_relu(self._linear(tf.concat([h, base_action], axis=1), scope + '/fc2', 256))
        value = self._linear(h, scope + '/dense', 1)
        if continuous.enabled(self.cfg): value += continuous.critic_residual(self, state, action, scope)
        return value

    def _reference_window(self,state):
        """Learner-only fixed source network. Its labels are cached once per real row."""
        tf=self.tf;reader=tf.train.NewCheckpointReader(self.cfg['baseline_checkpoint'])
        def tensor(name):return tf.constant(reader.get_tensor(WINDOW+'/'+name),dtype=tf.float32)
        value=state[:,:70]
        for fc,bn in (('fc1','bn1'),('fc2','bn2')):
            value=tf.matmul(value,tensor(fc+'/w'))+tensor(fc+'/b')
            value=tf.nn.batch_normalization(value,tensor(bn+'/moving_mean'),tensor(bn+'/moving_variance'),tensor(bn+'/beta'),None,.001)
            value=tf.nn.leaky_relu(value)
        return tf.stop_gradient(tf.tanh(tf.matmul(value,tensor('fc1_1/w'))+tensor('fc1_1/b')))

    def _add_bn_writers(self):
        tf = self.tf
        variables = {v.op.name: v for v in self.window_vars}
        if self.cfg.get('window_bn_statistics')=='frozen_migrated':
            self.window_update_ops=[tf.identity(variables[WINDOW+'/'+bn+'/'+name])
                                    for bn in ('bn1','bn2') for name in ('moving_mean','moving_variance')]
            return
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
        variables = list(self.base_window_vars)
        if self.training and initialize_critics:
            variables += [v for v in self._variables('critic') + self._variables('critic2')
                          if '/period_residual/' not in v.op.name]
        assignments = []
        for var in variables:
            if var.op.name not in names: raise ValueError('Missing source tensor: ' + var.op.name)
            value = migrate_array(var.op.name, rd.get_tensor(var.op.name), var.shape.as_list())
            reference=float(self.cfg.get('critic_initial_cost_reference_s',0.))
            if reference>0 and var.op.name in ('critic/dense/bias','critic2/dense/bias'):
                # Constant-period cost prior only; the actual reward/TD still use
                # each real duration and every later policy may change period.
                value=value-float(self.cfg['decision_cost'])/(1.-float(self.cfg['gamma'])**(reference/float(self.cfg['base_period_s'])))
            if not np.isfinite(value).all(): raise ValueError('Nonfinite source tensor')
            assignments.append(tf.assign(var, value))
        session.run(assignments)
        if self.training: session.run(self.target_init)
        return dict(source=str(checkpoint), source_state_dim=70, source_action_dim=1, state_dim=82,
                    critic_initial_cost_reference_s=float(self.cfg.get('critic_initial_cost_reference_s',0.)),
                    new_inputs_zero=True, optimizer_replay_counters_fresh=True, complete_resume=False,
                    critics='approximate_initialization_needs_calibration' if self.training and initialize_critics else 'not_restored',
                    targets='hard_copy_of_migrated_online', copied=[v.op.name for v in variables])

    def feed(self, batch):
        result=dict(zip((self.s, self.executed, self.reward, self.s1, self.terminal, self.discount),
                        (batch[0], batch[1], batch[2], batch[3], bootstrap_terminal(self.cfg, batch[4]), batch[6])))
        reference=getattr(self,'teacher_reference',None)
        if reference is not None and len(batch)>7:result[reference]=batch[7]
        return result

    def digest(self, session, window_only=False):
        h = hashlib.sha256()
        for var, value in zip(self.window_vars if window_only else self.online_vars,
                              session.run(self.window_vars if window_only else self.online_vars)):
            h.update(var.op.name.encode()); h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()

    def save_manifest(self, prefix, cfg, counters):
        Path(str(prefix) + '.joint.json').write_text(json.dumps(dict(schema=continuous.schema(cfg), cfg=cfg, counters=counters,
            state_dim=82, action_dim=2, no_complete_resume=True), indent=2) + '\n')

    def restore(self, session, prefix, allow_diagnostic=False):
        tf = self.tf
        record = json.loads(Path(str(prefix) + '.joint.json').read_text())
        if record['schema'] != continuous.schema(self.cfg): raise ValueError('Checkpoint schema mismatch')
        time_limit_default = 'physical_end_is_terminal_legacy'
        if record['cfg'].get('time_limit_handling', time_limit_default) != self.cfg.get('time_limit_handling', time_limit_default):
            raise ValueError('Checkpoint semantics mismatch: time_limit_handling')
        for key,default in (('actor_architecture','independent_v2'),('time_parameterization','softsign_log_v1'),
                            ('window_bn_statistics','adaptive_moving'),('actor_recipe','legacy_joint'),
                            ('teacher_anchor_weight',0.),('teacher_anchor_delta',.1),('teacher_anchor_clock_weighting','constant'),
                            ('time_feature_normalization','none'),('critic_initial_cost_reference_s',0.),
                            ('critic_period_residual_lr',None)):
            if record['cfg'].get(key,default)!=self.cfg.get(key,default):raise ValueError('Checkpoint semantics mismatch: '+key)
        if self.training and self.cfg.get('actor_recipe') in ('cost_preserve_v2', continuous.ARCHITECTURE):
            for key in ('window_actor_lr','time_actor_lr'):
                if record['cfg'].get(key)!=self.cfg.get(key):raise ValueError('Checkpoint learning-rate mismatch: '+key)
        for key in ('rate_mode', 'clock_mode', 'utility_normalization', 'decision_cost', 'fixed_mtp_ms',
                    'time_initializer','calibration_main_mode'):
            if record['cfg'].get(key) != self.cfg.get(key): raise ValueError('Checkpoint semantics mismatch: ' + key)
        selected = tf.global_variables() if self.training else self.online_vars + [self.phase, self.schema, self.rate_schema, self.clock_schema]
        tf.train.Saver(var_list=selected).restore(session, str(prefix))
        if session.run(self.schema).decode() != continuous.schema(self.cfg): raise ValueError('Saved schema tensor mismatch')
        if session.run(self.rate_schema).decode() != self.cfg['rate_mode'] or session.run(self.clock_schema).decode() != self.cfg['clock_mode']:
            raise ValueError('Saved measurement schema tensor mismatch')
        if not allow_diagnostic and int(session.run(self.phase)) != JOINT:
            raise ValueError('Calibration-only checkpoint is not a trained joint endpoint')
        return record
