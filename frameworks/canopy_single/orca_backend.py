"""Official Orca Agent adapter for the common scalar distributed harness.

The TF-layers Orca actor/critics are used directly, not the Canopy IBP actor.
Target BN is explicitly tied to deployment moving statistics; independent
critic Adam is installed separately and shared as an engineering correction.
"""
import pathlib,sys


def create(framework,cfg,role):
    import tensorflow as tf
    sys.path.insert(0,str(pathlib.Path(framework)/'code/orca/rl-module'))
    import agent as module
    import independent_critic_v34
    independent_critic_v34.install(module,cfg)
    import orca_moving_actor_loss
    orca_moving_actor_loss.install(module,cfg)
    if not getattr(module.Actor,'_target_bn_explicit',False):
        original=module.Actor.build
        def build(self,state,is_training):
            return original(self,state,False if self.name=='target_actor' else is_training)
        module.Actor.build=build;module.Actor._target_bn_explicit=True
    a=module.Agent(70,1,256,256,gamma=.995,batch_size=512,lr_a=.0001,lr_c=.001,tau=.001,
        mem_size=cfg['memsize'] if role=='learner' else 1,noise_type=3,noise_exp=50000,
        stddev=cfg.get('exploration_std',.2),PER=False,CDQ=True,LOSS_TYPE=cfg.get('LOSS_TYPE','MSE'))
    # Adapter names/placeholders carry no model parameters and do not change
    # the upstream Orca loss or inference function.
    a.symbolic_actor=a.actor
    a.is_concrete=tf.placeholder_with_default(True,[],name='orca_harness_concrete')
    a.s0_delta=tf.placeholder_with_default(tf.zeros_like(a.s0),[None,70],name='orca_harness_unused_delta')
    def concrete(state,use_noise=False):
        out=a.get_action(state,use_noise);return out,out
    a.get_concrete_action=concrete
    variables={v.op.name:v for v in tf.global_variables()}
    moving=[v for n,v in variables.items() if n.startswith('target_actor/') and ('moving_mean' in n or 'moving_variance' in n)]
    assert len(moving)==4
    for v in moving:
        source=variables['actor/'+v.op.name[len('target_actor/'):]]
        a.target_act_init_op.append(tf.assign(v,source))
        a.target_actor_update_op.append(tf.assign(v,.001*source+.999*v))
    return a
