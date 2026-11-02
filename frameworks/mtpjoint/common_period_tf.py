"""Isolated TF1 backend for the agreed common-period experiment.

Requires the original Linux TF1 environment. It is not installed into Agent or
the legacy distributed driver. All update paths are explicit and ordered.
"""
import json
import pathlib
import threading
import numpy as np
from common_period import PERIODS, encoded_period, SCHEMA, REWARD, checksum

ONLINE = 'abstract_actor/abstract_actor/'
TARGET = 'target_actor/'


class Model:
    def __init__(self, checkpoint, K, n0, seed=3, lambda_z=1e-3, reward_config=None):
        import tensorflow as tf
        self.tf = tf
        self.lock = threading.RLock()
        self.lambda_z = lambda_z
        self.metadata = dict(schema=SCHEMA, reward_version=REWARD, reward_config=reward_config or {},
                             reward_checksum=checksum(reward_config or {}), K=float(K), n0=n0,
                             complete_tcp_resume=False)
        self.graph = tf.Graph()
        self.pairs = []
        with self.graph.as_default():
            tf.set_random_seed(seed)
            reader = tf.train.NewCheckpointReader(checkpoint)
            shapes = reader.get_variable_to_shape_map()
            self.v = {}
            prefixes = (ONLINE, TARGET, 'critic/', 'critic2/', 'target_critic/', 'target_critic2/')
            for name in sorted(shapes):
                if name.startswith(prefixes) and '/Adam' not in name:
                    value = reader.get_tensor(name)
                    if name.endswith('/dense/bias'):
                        value = value - np.float32(K)
                    self.v[name] = tf.get_variable(name, initializer=value,
                        trainable=not name.startswith(('target_',)) and '/moving_' not in name)
            if self.v[ONLINE + 'fc1/w'].shape.as_list() != [149, 256]:
                raise ValueError('Expected verified V31 149-state checkpoint')
            for prefix in (ONLINE, TARGET):
                name = prefix + 'period_w'
                self.v[name] = tf.get_variable(name, [1, 256], initializer=tf.zeros_initializer(), trainable=prefix == ONLINE)
            self.s = tf.placeholder(tf.float32, [None, 149], 'state')
            self.sp = tf.placeholder(tf.float32, [None, 149], 'successor')
            self.action = tf.placeholder(tf.float32, [None, 2], 'executed_action')
            self.r = tf.placeholder(tf.float32, [None, 1], 'centered_reward')
            self.g = tf.placeholder(tf.float32, [None, 1], 'actual_discount')
            self.b = tf.placeholder(tf.float32, [None, 1], 'bootstrap')
            self.noise = tf.placeholder(tf.float32, [None, 1], 'paired_target_noise')
            self.do_actor = tf.placeholder(tf.bool, [], 'update_actor')
            self.t = self.action[:, 1:2]
            self.mu, self.z = self.actor(self.s, self.t, ONLINE)
            target_a, _ = self.actor(self.sp, self.t, TARGET)
            smooth = tf.clip_by_value(target_a + tf.clip_by_value(self.noise, -.2, .2), -1., 1.)
            next_action = tf.concat([smooth, self.t], 1)
            qnext = [self.critic(self.sp, next_action, p) for p in ('target_critic/', 'target_critic2/')]
            y = tf.stop_gradient(self.r + self.g * self.b * tf.minimum(*qnext))
            self.q = [self.critic(self.s, self.action, p) for p in ('critic/', 'critic2/')]
            self.losses = [tf.reduce_mean(tf.square(y - q)) for q in self.q]
            self.actor_q = self.critic(self.s, tf.concat([self.mu, self.t], 1), 'critic/')
            self.rl_loss = -tf.reduce_mean(self.actor_q)
            self.reg_loss = lambda_z * tf.reduce_mean(tf.square(self.z))
            av = self.parameters(ONLINE)
            self.rl_grad = tf.global_norm([g for g in tf.gradients(self.rl_loss, av) if g is not None])
            self.reg_grad = tf.global_norm([g for g in tf.gradients(self.reg_loss, av) if g is not None])
            prior = [n0[str(t)] for t in PERIODS]
            self.n_prior = tf.get_variable('discounted_count_prior', initializer=np.asarray(prior,np.float32), trainable=False)
            self.n = self.number(self.s, self.action, 'number/')
            nt = self.number(self.sp, tf.concat([target_a, self.t], 1), 'target_number/')
            self.n_loss = tf.reduce_mean(tf.square(tf.stop_gradient(1 + self.g * self.b * nt) - self.n))
            self.optimizers = [tf.train.AdamOptimizer(.001, name='Q1Adam'), tf.train.AdamOptimizer(.001, name='Q2Adam'),
                               tf.train.AdamOptimizer(.0001, name='ActorAdam'), tf.train.AdamOptimizer(.001, name='NAdam')]
            # Prepare slots before conditional/control-flow scopes.
            for opt, params in zip(self.optimizers, [self.parameters('critic/'), self.parameters('critic2/'), av, self.parameters('number/')]):
                opt._create_slots(params)
            self.make_pairs()
            # One shared target value per complete critic iteration.
            qops = [o.minimize(l, var_list=self.parameters(p)) for o, l, p in zip(self.optimizers[:2], self.losses, ('critic/', 'critic2/'))]
            with tf.control_dependencies(qops):
                nop = self.optimizers[3].minimize(self.n_loss, var_list=self.parameters('number/'))
            with tf.control_dependencies([nop]):
                actor_done = tf.cond(self.do_actor, self.actor_update, lambda: tf.constant(0))
            with tf.control_dependencies([actor_done]):
                bn_done = tf.cond(self.do_actor, self.update_bn, lambda: tf.constant(0))
            with tf.control_dependencies([bn_done]):
                self.step = tf.group(*[tf.assign(t, .001 * s + .999 * t) for s, t in self.pairs])
            # B0: own deterministic target per critic, same frozen online policy.
            deterministic, _ = self.actor(self.sp, self.t, ONLINE)
            det_action = tf.concat([deterministic, self.t], 1)
            det_targets = [self.critic(self.sp, det_action, p) for p in ('target_critic/', 'target_critic2/')]
            self.b0_losses = [tf.reduce_mean(tf.square(tf.stop_gradient(self.r + self.g * self.b * qt) - q)) for qt, q in zip(det_targets, self.q)]
            det_n = self.number(self.sp, det_action, 'target_number/')
            self.b0_n_loss = tf.reduce_mean(tf.square(tf.stop_gradient(1 + self.g * self.b * det_n) - self.n))
            b0ops = [o.minimize(l, var_list=self.parameters(p)) for o, l, p in zip(self.optimizers[:2], self.b0_losses, ('critic/', 'critic2/'))]
            b0ops.append(self.optimizers[3].minimize(self.b0_n_loss, var_list=self.parameters('number/')))
            with tf.control_dependencies(b0ops):
                self.b0_step = tf.group(*[tf.assign(t, .001 * s + .999 * t) for s, t in self.pairs if not s.op.name.startswith(ONLINE)])
            self.schema_marker = tf.get_variable('common_period_schema', initializer=SCHEMA, trainable=False, dtype=tf.string)
            self.reward_marker = tf.get_variable('reward_config_sha256', initializer=self.metadata['reward_checksum'], trainable=False, dtype=tf.string)
            self.K_marker = tf.get_variable('frozen_K', initializer=np.float64(K), trainable=False, dtype=tf.float64)
            self.snapshot_version = tf.get_variable('policy_snapshot_version', initializer=np.int64(0), trainable=False, dtype=tf.int64)
            self.bump_snapshot = tf.assign_add(self.snapshot_version, 1)
            self.init = tf.global_variables_initializer()
            self.init_n = tf.group(*[tf.assign(t, s) for s, t in self.pairs if s.op.name.startswith('number/')])
            # The finite experiment manifest, not Saver's implicit last-five
            # default, determines which diagnostic snapshots are retained.
            self.saver = tf.train.Saver(max_to_keep=0)
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        self.session.run(self.init)
        self.session.run(self.init_n)

    def parameters(self, prefix):
        return [v for v in self.tf.trainable_variables() if v.op.name.startswith(prefix)]

    def actor(self, state, t, prefix):
        tf, v = self.tf, self.v
        def linear(x, name):
            return tf.matmul(x, v[prefix + name + '/w']) + v[prefix + name + '/b']
        def bn(x, name):
            return tf.nn.batch_normalization(x, v[prefix + name + '/moving_mean'], v[prefix + name + '/moving_variance'], v[prefix + name + '/beta'], None, .001)
        h = tf.nn.leaky_relu(bn(linear(state, 'fc1'), 'bn1'))
        h = tf.nn.leaky_relu(bn(linear(h, 'fc2') + tf.matmul(t, v[prefix + 'period_w']), 'bn2'))
        # Preserve original two-column matmul; only window column is used.
        z = linear(h, 'fc1_1')[:, :1]
        return tf.tanh(z), z

    def critic(self, s, action, prefix):
        tf, v = self.tf, self.v
        def dense(x, name, activate=True):
            y = tf.matmul(x, v[prefix + name + '/kernel']) + v[prefix + name + '/bias']
            return tf.nn.leaky_relu(y) if activate else y
        h = dense(s, 'fc1')
        output = dense(dense(tf.concat([h, action], 1), 'fc2'), 'dense', False)
        with tf.variable_scope(prefix.rstrip('/'), reuse=tf.AUTO_REUSE):
            offset = tf.get_variable('period_calibration_bias', [5], initializer=tf.zeros_initializer(), trainable=False)
        codes = tf.constant([encoded_period(t) for t in PERIODS], tf.float32)
        idx = tf.argmin(tf.abs(action[:,1:2]-codes[None,:]),axis=1)
        return output + tf.gather(offset,idx)[:,None]

    def number(self, s, action, prefix):
        tf = self.tf
        with tf.variable_scope(prefix.rstrip('/'), reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(tf.concat([s, action], 1), 64, activation=tf.nn.leaky_relu, name='hidden')
            residual = tf.layers.dense(h, 1, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(), name='residual')
        codes = tf.constant([encoded_period(t) for t in PERIODS], tf.float32)
        idx = tf.argmin(tf.abs(action[:, 1:2] - codes[None, :]), axis=1)
        return residual + tf.gather(self.n_prior, idx)[:, None]

    def make_pairs(self):
        tf = self.tf
        variables = {v.op.name: v for v in tf.global_variables()}
        seen = set()
        for source, target in ((ONLINE, TARGET), ('critic/', 'target_critic/'), ('critic2/', 'target_critic2/'), ('number/', 'target_number/')):
            sources = {n[len(source):]: v for n, v in variables.items() if n.startswith(source) and not any('/' + o in n for o in ('Q1Adam', 'Q2Adam', 'ActorAdam', 'NAdam'))}
            targets = {n[len(target):]: v for n, v in variables.items() if n.startswith(target)}
            if set(sources) != set(targets):
                raise ValueError('Target full-name pairing mismatch')
            for name in sorted(sources):
                a, b = sources[name], targets[name]
                if b.op.name in seen or a.shape != b.shape:
                    raise ValueError('Duplicate target assignment or shape mismatch')
                seen.add(b.op.name)
                self.pairs.append((a, b))

    def actor_update(self):
        tf = self.tf
        # Build this forward inside the dependency scope, after critic updates.
        mu, z = self.actor(self.s, self.t, ONLINE)
        q = self.critic(self.s, tf.concat([mu, self.t], 1), 'critic/')
        loss = -tf.reduce_mean(q) + self.lambda_z * tf.reduce_mean(tf.square(z))
        with tf.control_dependencies([self.optimizers[2].minimize(loss, var_list=self.parameters(ONLINE))]):
            return tf.constant(1)

    def bn_updates(self):
        tf, v = self.tf, self.v
        x = self.s
        updates = []
        for layer, norm in (('fc1', 'bn1'), ('fc2', 'bn2')):
            x = tf.matmul(x, v[ONLINE + layer + '/w']) + v[ONLINE + layer + '/b']
            if layer == 'fc2':
                x += tf.matmul(self.t, v[ONLINE + 'period_w'])
            mean, variance = tf.nn.moments(x, [0])
            for suffix, value in (('moving_mean', mean), ('moving_variance', variance)):
                var = v[ONLINE + norm + '/' + suffix]
                updates.append(tf.assign(var, .999 * var + .001 * value))
            x = tf.nn.leaky_relu(tf.nn.batch_normalization(x, mean, variance, v[ONLINE + norm + '/beta'], None, .001))
        return updates

    def update_bn(self):
        with self.tf.control_dependencies(self.bn_updates()):
            return self.tf.constant(1)

    def snapshot(self, window_only=False):
        """Single session fetch: one coherent published policy and its version."""
        with self.graph.as_default():
            variables = [v for v in self.tf.global_variables() if
                         v.op.name.startswith((ONLINE, 'critic/', 'critic2/', 'number/'))
                         and not any('/' + opt in v.op.name for opt in ('Q1Adam','Q2Adam','ActorAdam','NAdam'))]
            if window_only:variables=[v for v in variables if v.op.name.startswith(ONLINE)]
        with self.lock:
            version = int(self.session.run(self.bump_snapshot))
            values = self.session.run(variables)
        return dict(version=version, schema=SCHEMA, reward_checksum=self.metadata['reward_checksum'],
                    values={v.op.name: x for v, x in zip(variables, values)})

    def initialize_centering(self, K, n0):
        """Only before learning, after isolated real calibration blocks."""
        if self.metadata['K']!=0 or self.metadata.get('centering_initialized'):
            raise ValueError('Centering is frozen once')
        tf=self.tf
        with self.graph.as_default():
            variables={v.op.name:v for v in tf.global_variables()}
            slots=[v for name,v in variables.items() if any('/'+o in name for o in ['Q1Adam','Q2Adam','ActorAdam','NAdam'])]
            if any(np.any(x!=0) for x in self.session.run(slots)):
                raise ValueError('Calibration cannot change after optimization')
            operations=[tf.assign_sub(self.v[p+'dense/bias'],np.asarray([K],np.float32)) for p in ['critic/','critic2/','target_critic/','target_critic2/']]
            operations += [tf.assign(self.K_marker,np.float64(K)),tf.assign(self.n_prior,np.asarray([n0[str(t)] for t in PERIODS],np.float32))]
        with self.lock:self.session.run(operations)
        self.metadata.update(K=float(K),n0=n0,centering_initialized=True)

    def train_batch(self, batch, noise, update_actor):
        fd = self.feed(batch)
        fd.update({self.noise:noise,self.do_actor:bool(update_actor)})
        with self.lock:
            self.session.run(self.step,fd)

    def diagnostics(self, states):
        """No random operations or updates: report each executed candidate."""
        result = {}
        for period in PERIODS:
            a = np.zeros((len(states),2),np.float32)
            a[:,1] = encoded_period(period)
            with self.lock:
                window,rl,reg = self.session.run([self.mu,self.rl_grad,self.reg_grad],{self.s:states,self.action:a})
            result[str(period)] = dict(rl_gradient_norm=float(rl),regularizer_gradient_norm=float(reg),
                                       saturation=float(np.mean(np.abs(window)>.99)))
        return result

    def save(self, prefix, ledger, replay_metadata):
        path = pathlib.Path(prefix)
        if pathlib.Path(str(path)+'.index').exists() or pathlib.Path(str(path)+'.metadata.json').exists():
            raise ValueError('Never overwrite an experiment checkpoint')
        saved = self.saver.save(self.session, str(path))
        meta = dict(self.metadata, ledger=ledger, replay=replay_metadata,
                    snapshot_version=int(self.session.run(self.snapshot_version)))
        pathlib.Path(saved+'.metadata.json').write_text(json.dumps(meta,sort_keys=True,indent=2))
        return saved

    def calibrate_bias_once(self, offsets, per_period=False):
        """One fit-set correction; never call on a confirmation set."""
        key = 'B0_period_bias' if per_period else 'A_global_bias'
        if key in self.metadata:raise ValueError('Bias correction cannot be iterated')
        tf=self.tf
        with self.graph.as_default():
            variables={v.op.name:v for v in tf.global_variables()}
            operations=[]
            for prefix in ['critic/','critic2/','target_critic/','target_critic2/']:
                name=prefix+('period_calibration_bias' if per_period else 'dense/bias')
                value=np.asarray(offsets,np.float32)
                if per_period and value.shape!=(5,):raise ValueError('Five frozen period offsets required')
                if not per_period:value=np.full(variables[name].shape.as_list(),float(value),np.float32)
                if not np.isfinite(value).all():raise ValueError('Non-finite bias correction')
                operations.append(tf.assign_add(variables[name],value))
        with self.lock:self.session.run(operations)
        self.metadata[key]=value.tolist()

    def restore(self, prefix):
        meta = json.loads(pathlib.Path(prefix+'.metadata.json').read_text())
        for key in ('schema','reward_version','reward_checksum','K','n0'):
            if meta[key] != self.metadata[key]:
                raise ValueError('Checkpoint contract mismatch: '+key)
        self.saver.restore(self.session, prefix)
        marker, reward, K = self.session.run([self.schema_marker,self.reward_marker,self.K_marker])
        if marker.decode()!=SCHEMA or reward.decode()!=self.metadata['reward_checksum'] or K!=self.metadata['K']:
            raise ValueError('Checkpoint embedded schema mismatch')
        for key in ('A_global_bias','B0_period_bias'):
            if key in meta:self.metadata[key]=meta[key]
            else:self.metadata.pop(key,None)
        return meta

    def feed(self, batch):
        return {self.s: batch['s0'], self.sp: batch['s1'], self.action: batch['action'],
                self.r: batch['reward'], self.g: batch['discount'], self.b: batch['bootstrap']}

    def choose(self, state, penalty):
        codes = np.asarray([encoded_period(t) for t in PERIODS], np.float32)
        states = np.repeat(np.asarray(state, np.float32)[None, :], 5, axis=0)
        actions = np.column_stack([np.zeros(5), codes]).astype(np.float32)
        windows = self.session.run(self.mu, {self.s: states, self.action: actions})[:, 0]
        actions[:, 0] = windows
        q1, q2, n = self.session.run(self.q + [self.n], {self.s: states, self.action: actions})
        scores = ((q1 + q2) * .5 - penalty * n).ravel()
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite selector value')
        i = int(np.argmax(scores))
        return actions[i], dict(period_ms=PERIODS[i], q1=q1.ravel().tolist(), q2=q2.ravel().tolist(), n=n.ravel().tolist())
