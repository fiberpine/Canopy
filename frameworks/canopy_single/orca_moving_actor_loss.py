"""Opt-in Orca actor gradient through the deployed moving-BN function.

No added variables, action mapping, target policy, or deployment changes.
This is a training ablation, not a performance improvement claim.
"""
FLAG = 'orca_actor_loss_moving_stats'

def forward(agent):
    import tensorflow as tf
    variables = {v.op.name: v for v in tf.global_variables()}
    prefix = 'actor/'
    def linear(x, name):
        return tf.matmul(x, variables[prefix + name + '/kernel']) + variables[prefix + name + '/bias']
    def bn(x, name):
        return tf.nn.batch_normalization(x,
            variables[prefix + name + '/moving_mean'],
            variables[prefix + name + '/moving_variance'],
            variables[prefix + name + '/beta'], None, .001)
    h = tf.nn.leaky_relu(bn(linear(agent.s0, 'fc1'), 'batch_normalization'))
    h = tf.nn.leaky_relu(bn(linear(h, 'fc2'), 'batch_normalization_1'))
    return tf.nn.tanh(linear(h, 'dense'))

def install(module, cfg):
    active = cfg.get(FLAG, False)
    if not isinstance(active, bool):
        raise ValueError(FLAG + ' must be an explicit boolean')
    cls = module.Agent
    cls._orca_actor_loss_moving_stats_enabled = active
    if not active or getattr(cls, '_orca_actor_loss_moving_stats_installed', False):
        return
    original = cls.build_actor_train_op
    def build(self):
        if not cls._orca_actor_loss_moving_stats_enabled:
            return original(self)
        import tensorflow as tf
        before = list(tf.global_variables())
        self._orca_deployed_action_for_loss = forward(self)
        self.a_loss = -tf.reduce_mean(self.critic.build(self.s0, self._orca_deployed_action_for_loss))
        assert before == list(tf.global_variables()), 'Moving actor forward must reuse the complete original function'
        return self.actor_optimizer.minimize(self.a_loss,
            var_list=self.actor.train_var(), global_step=self.global_step)
    cls.build_actor_train_op = build
    cls._orca_actor_loss_moving_stats_installed = True
