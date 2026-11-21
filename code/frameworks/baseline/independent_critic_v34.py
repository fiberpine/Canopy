"""Optional independent Adam clocks for the two existing clipped-Q critics.

No model, target-label, loss, actor, reward, replay, or update-credit change.
The two critic minimizers share one target tensor in one Session.run; their
independent Adam powers advance once each, then the shared operation counter
advances by two. The actor remains a separate, subsequent operation.
"""
import re

FLAG = 'independent_critic_adam_v34'
OPTIMIZER_NAMES = ('AdamCritic1V34', 'AdamCritic2V34')


def enabled(cfg):
    value = cfg.get(FLAG, False)
    if not isinstance(value, bool):
        raise ValueError(FLAG + ' must be an explicit boolean')
    return value


def _build(agent, configured_loss):
    import tensorflow as tf
    # Identical shared clipped-Q target and existing MSE/HUBER/PER reductions.
    target = tf.minimum(agent.y, agent.y2, name='v34_shared_critic_target')
    def loss(prediction):
        weights = agent.importance if configured_loss and agent.PER else 1.0
        if configured_loss and agent.LOSS_TYPE == 'HUBER':
            return tf.compat.v1.losses.huber_loss(target, prediction, weights=weights)
        if configured_loss and agent.LOSS_TYPE != 'MSE':
            raise KeyError(agent.LOSS_TYPE)
        return tf.reduce_mean(tf.square(target - prediction) * weights)
    agent.c_loss, agent.c_loss2 = loss(agent.critic_out), loss(agent.critic_out2)
    critics = (agent.critic, agent.critic2)
    parameters = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                  scope=re.escape(critic.name) + r'/') for critic in critics]
    if not all(parameters) or set(parameters[0]) & set(parameters[1]):
        raise ValueError('Each V34 critic needs a disjoint nonempty parameter scope')
    optimizers = [tf.train.AdamOptimizer(agent.lr_c, name=name)
                  for name in OPTIMIZER_NAMES]
    updates = [optimizer.minimize(objective, var_list=variables)
               for optimizer, objective, variables in zip(
                   optimizers, (agent.c_loss, agent.c_loss2), parameters)]
    finished = tf.group(*updates, name='v34_both_critic_minimizers_finished')
    with tf.control_dependencies([finished]):
        counted = tf.assign_add(agent.global_step, 2, use_locking=True,
                                name='v34_count_two_critic_operations')
    agent._v34_critic_target = target
    agent._v34_critic_parameters = parameters
    agent._v34_critic_optimizers = optimizers
    agent._v34_critic_minimize_ops = updates
    if not configured_loss:
        # Preserve the summaries emitted by the legacy alternate MSE entry.
        tf.summary.scalar('Loss/critic_loss', agent.c_loss)
        tf.summary.scalar('Loss/critic_loss_2', agent.c_loss2)
    return [counted]


def install(module, cfg):
    """Install before Agent.build_learn; disabled configurations are untouched."""
    active = enabled(cfg)
    cls = module.Agent
    cls._independent_critic_adam_v34_enabled = active
    if not active:
        return
    if getattr(cls, '_independent_critic_adam_v34_installed', False):
        return
    original_configured = cls.build_critic_train_op_huber
    original_mse = cls.build_critic_train_op
    def configured(self):
        # Stock build_learn calls this entry even when LOSS_TYPE is MSE.
        if not self.CDQ or not cls._independent_critic_adam_v34_enabled:
            return original_configured(self)
        return _build(self, configured_loss=True)
    def mse(self):
        if not self.CDQ or not cls._independent_critic_adam_v34_enabled:
            return original_mse(self)
        return _build(self, configured_loss=False)
    cls.build_critic_train_op_huber = configured
    cls.build_critic_train_op = mse
    cls._independent_critic_adam_v34_installed = True
