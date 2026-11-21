"""Pair each critic target once by exact relative variable name."""
def connect(agent, tau):
    import tensorflow as tf
    variables = {v.op.name: v for v in tf.trainable_variables()}
    for source_scope, target_scope, update_name, init_name in (
        ('critic', 'target_critic', 'target_critic_update_op', 'target_cri_init_op'),
        ('critic2', 'target_critic2', 'target_critic_update_op2', 'target_cri_init_op2')):
        sources = {name[len(source_scope)+1:]: v for name, v in variables.items()
                   if name.startswith(source_scope+'/')}
        targets = {name[len(target_scope)+1:]: v for name, v in variables.items()
                   if name.startswith(target_scope+'/')}
        assert sources and set(sources) == set(targets)
        assert all(sources[k].shape == targets[k].shape for k in sources)
        setattr(agent, update_name, [tf.assign(targets[k], tau*sources[k]+(1-tau)*targets[k]) for k in sorted(sources)])
        setattr(agent, init_name, [tf.assign(targets[k], sources[k]) for k in sorted(sources)])
