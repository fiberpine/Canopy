"""Explicit per-frame physical-time sample rates; old86 prefix preserved.

The ten new fields are normalized RTT-sample rates, not ACK packet rates.
20ms is a unit reference only, never an observation timer.
"""
import numpy as np
SCHEMA=1
def enabled(cfg):return cfg.get('time_statistics_v27',False)
def features(history,gaps,cfg):
    h=np.asarray(history,np.float32).reshape(10,7);dt=np.asarray(gaps,np.float32)
    assert dt.shape==(10,) and np.isfinite(dt).all() and np.all(dt>=0)
    if cfg.get('time_statistics_mode','rates')=='masked_control':return np.zeros(10,np.float32)
    assert cfg.get('time_statistics_mode','rates')=='rates'
    y=np.zeros(10,np.float32);valid=dt>0
    y[valid]=h[valid,3]*(.02/dt[valid])
    assert np.isfinite(y).all() and np.all(y>=0)
    return y

def initialize(agent,cfg):
    """86->96 exact function extension, zero extra input weights. No teacher at run time."""
    import tensorflow as tf,json,pathlib
    source=cfg['matching_clock_prior_checkpoint'];reader=tf.train.NewCheckpointReader(source)
    p='abstract_actor/abstract_actor/';source_dim=reader.get_variable_to_shape_map()[p+'fc1/w'][0];assert source_dim in (86,96)
    meta=json.loads(pathlib.Path(source+'.counters.json').read_text())['cfg']
    assert meta.get('decision_observation_clock') and meta['gamma']==cfg['gamma'] and meta['decision_cost']==cfg['decision_cost']
    assert cfg.get('utility_normalization','observed_peak')==meta.get('utility_normalization','observed_peak'),'Keep reward fixed for state feature comparison'
    if source_dim==96:
        assert int(reader.get_tensor('time_statistics_schema'))==1
        assert meta.get('time_statistics_mode')==cfg.get('time_statistics_mode')
        assert int(reader.get_tensor('interval_rate_schema'))==int(cfg.get('interval_rate_v28')=='acked_bytes')
        assert meta.get('interval_rate_v28')==cfg.get('interval_rate_v28'),'Same96 migration must preserve measurement semantics'
    v={x.op.name:x for x in tf.global_variables()}
    names=[n for n in v if '/Adam' not in n and not n.startswith(('beta1_power','beta2_power')) and n not in ('global_step','timing_learning_enabled','time_statistics_schema','interval_rate_schema','actor_gradient_schema','timer_boundary_schema')]
    slots=[x for n,x in v.items() if n not in names];before=agent.sess.run(slots);values={}
    for n in names:
        x=reader.get_tensor(n)
        shape=tuple(v[n].shape.as_list())
        if x.ndim==2 and x.shape==(86,256) and shape==(96,256):
            assert n.endswith(('fc1/w','fc1/kernel')),n
            x=np.pad(x,((0,10),(0,0)),mode='constant')
        assert x.shape==tuple(v[n].shape.as_list()),(n,x.shape,v[n].shape)
        values[n]=x
    agent.sess.run([tf.assign(v[n],x) for n,x in values.items()])
    for n,x in values.items():np.testing.assert_array_equal(agent.sess.run(v[n]),x)
    for x,y in zip(before,agent.sess.run(slots)):np.testing.assert_array_equal(x,y)
    return dict(source_checkpoint=source,source_state_dim=source_dim,new_state_dim=96,extra_input_weights_zero=source_dim==86,
        source_function_preserved_before_learning=True,actor_critics_targets_bn_copied=True,
        optimizer_replay_counters_fresh=True,complete_resume=False,scratch_training=False,
        extra_feature='RTT_sample_count/cwnd *20ms/actual_frame_dt',teacher_inference=False)
