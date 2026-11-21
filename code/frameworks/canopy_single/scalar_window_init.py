"""Opt-in small output-layer initialization for FRESH single-action Orca only."""
import hashlib
import numpy as np


def initialize(agent,session,cfg):
    mode=cfg.get('scalar_window_initialization','upstream')
    if mode=='upstream':return None
    if mode!='small_uniform' or cfg.get('policy_family')!='orca':
        raise ValueError('Explicit single-action Orca small_uniform required')
    if cfg.get('cold_init_checkpoint'):
        raise ValueError('Never alter a restored checkpoint; fresh initialization only')
    import tensorflow as tf
    if int(session.run(agent.global_step))!=0:raise ValueError('Initialization after optimization forbidden')
    tensors={v.op.name:v for v in tf.global_variables()}
    names=['actor/dense/kernel','actor/dense/bias']
    assert tensors[names[0]].shape.as_list()==[256,1]
    assert tensors[names[1]].shape.as_list()==[1]
    # Separate RNG leaves replay/exploration Python/NumPy streams unchanged.
    rng=np.random.RandomState(cfg['seed']+7919)
    proof={}
    for name in names:
        v=tensors[name];old=session.run(v)
        value=rng.uniform(-.003,.003,size=v.shape.as_list()).astype(np.float32)
        session.run(tf.assign(v,value))
        proof[name]=dict(before_sha256=hashlib.sha256(old.tobytes()).hexdigest(),
                         after_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                         min=float(value.min()),max=float(value.max()))
    return dict(mode=mode,half_width=.003,seed=cfg['seed']+7919,variables=proof,
                target='Original init_target runs afterwards; no optimizer/replay restoration')
