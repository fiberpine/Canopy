"""Explicit critic-only admission/ledger. No actor release or resume implementation."""
import hashlib
import numpy as np


def enabled(cfg):
    value=cfg.get('critic_only',False)
    if not isinstance(value,bool):raise ValueError('critic_only must be boolean')
    if value and (cfg.get('policy_family')!='orca' or cfg.get('critic_update_budget',0)<=0):
        raise ValueError('Explicit Orca critic-update budget required')
    return value


def fingerprint(values):
    h=hashlib.sha256()
    for x in values:h.update(x.tobytes())
    return h.hexdigest()


class FrozenActor:
    def __init__(self,a,sess):
        self.a,self.sess=a,sess
        self.vars=a.actor.train_var()+[a.actor_optimizer.get_slot(v,n) for v in a.actor.train_var() for n in a.actor_optimizer.get_slot_names()]+list(a.actor_optimizer._get_beta_accumulators())
        assert all(v is not None for v in self.vars)
        self.values=sess.run(self.vars)
    def verify(self,critic_updates):
        values=self.sess.run(self.vars)
        assert all(np.array_equal(x,y) for x,y in zip(self.values,values)),'Actor trainables/Adam changed'
        step=int(self.sess.run(self.a.global_step))
        assert step==2*critic_updates,('Optimizer ledger mismatch',step,critic_updates)
        return dict(actor_trainables_and_Adam_unchanged=True,actor_optimizer_updates=0,global_step=step,critic_complete_updates=critic_updates,sha256=fingerprint(values),BN_allowed_to_evolve=True)
