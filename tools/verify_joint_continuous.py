"""Actual TF1 graph, executed-duration coupling and chain-rule verification.

Uses saved real inputs, temporary checkpoints and synthetic updates only.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(config, states_file, output):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.model import Model
    from frameworks.joint_healthy.continuous import CONDITION, exploration_inputs
    from frameworks.joint_healthy.core import interval_return
    cfg = json.loads(Path(config).read_text().replace('@PACKAGE@', str(ROOT).replace('\\', '/')))
    with np.load(str(states_file)) as p: states = p['states'][:128].copy()
    # Include empty-feedback initializers in addition to actual fresh samples.
    states[::3, 66] = 0.
    prior = np.where(states[:, 66] > 0., 20.5, np.clip(np.expm1(states[:, 79])*20., 40., 120.))
    session_cfg = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
    with tf.Graph().as_default():
        tf.set_random_seed(3); model = Model(cfg, True)
        anchor_time = [g for g in tf.gradients(model.anchor_loss, model.time_vars) if g is not None]
        chain = tf.gradients(tf.reduce_sum(model.action[:, :1]), model.time_logit)[0]
        # Independent direct + indirect chain-rule decomposition of the REAL Q.
        q = model._critic(model.s, model.executed, 'critic')
        dq = tf.gradients(tf.reduce_sum(q), model.executed)[0]
        dwindow = tf.gradients(tf.reduce_sum(model.action[:, :1]), model.time_logit)[0]
        dtime = tf.gradients(tf.reduce_sum(model.action[:, 1:2]), model.time_logit)[0]
        full = tf.gradients(-model.policy_loss*tf.cast(tf.shape(model.s)[0], tf.float32), model.time_logit)[0]
        analytic = dq[:, :1]*dwindow + dq[:, 1:2]*dtime
        stats = [a for a in model.window_vars if a.op.name.endswith(('/moving_mean', '/moving_variance'))]
        online_names = {v.op.name for v in model.online_vars}
        actor_pairs = [(a,b) for a,b in model.target_pairs if a.op.name in online_names]
        saver = tf.train.Saver()
        with tf.Session(config=session_cfg) as sess:
            sess.run(tf.global_variables_initializer()); model.initialize_baseline(sess, cfg['baseline_checkpoint'])
            action, teacher, duration = sess.run([model.action, model.teacher_output, model.planned_ms], {model.s: states})
            parity = float(np.max(np.abs(action[:, :1]-teacher))); assert parity < 1e-5
            time_parity = float(np.max(np.abs(duration[:,0]-prior))); assert time_parity < .0001
            before = sess.run([a for pair in actor_pairs for a in pair]+stats)
            # Nonzero conditional output makes a broken execution order visible.
            bias = next(v for v in model.window_vars if v.op.name == CONDITION+'/output/bias')
            sess.run(tf.assign(bias, [.15])); sess.run(model.target_init)
            before = sess.run([a for pair in actor_pairs for a in pair]+stats)
            draws = exploration_inputs(cfg, np.random.RandomState(9)); draws.update(probability=0., global_fraction=.713)
            feed = {model.s: states}; feed.update((model.behavior_inputs[k], v) for k,v in draws.items())
            behavior, pure, probe = sess.run([model.behavior_action, model.action, model.behavior_probe], feed)
            proposed = 20.*8.**draws['global_fraction']
            expected = sess.run(model.diagnostic_action, {model.s: states, model.diagnostic_ms: np.full((len(states),1), proposed, np.float32)})
            probe_error = float(np.max(np.abs(behavior-expected))); assert probe and probe_error < 2e-6
            mismatch_visible = float(np.max(np.abs(behavior[:,0]-pure[:,0]))); assert mismatch_visible > .001
            base20, base = sess.run([model.diagnostic_action, model.base_window_action],
                {model.s: states, model.diagnostic_ms: np.full((len(states),1),20.,np.float32)})
            assert np.array_equal(base20[:,:1],base)
            durations = np.full((len(states),1),.08,np.float64)
            reward, disc = interval_return(.9,.08,cfg['decision_cost'])
            batch=[states, action, np.full((len(states),1),reward,np.float32), states*.99,
                   np.zeros((len(states),1),np.float32),durations,np.full((len(states),1),disc,np.float32),teacher]
            trainfeed=model.feed(batch)
            for _ in range(3): sess.run(model.critic_step,trainfeed);sess.run(model.target_update)
            assert all(np.array_equal(a,b) for a,b in zip(before,sess.run([a for pair in actor_pairs for a in pair]+stats)))
            try: sess.run(model.actor_step,trainfeed)
            except tf.errors.InvalidArgumentError: pass
            else: raise AssertionError('Calibration allowed an actor update')
            sess.run(tf.assign(model.phase,2))
            before_window=sess.run(model.window_vars);before_time=sess.run(model.time_vars);before_stats=sess.run(stats)
            for _ in range(20):
                sess.run(model.critic_step,trainfeed);sess.run(model.actor_step,trainfeed);sess.run(model.target_update)
            assert all(np.array_equal(a,b) for a,b in zip(before_stats,sess.run(stats)))
            assert any(not np.array_equal(a,b) for a,b in zip(before_window,sess.run(model.window_vars)))
            assert any(not np.array_equal(a,b) for a,b in zip(before_time,sess.run(model.time_vars)))
            assert all(np.count_nonzero(g)==0 for g in sess.run(anchor_time,trainfeed))
            actual=sess.run(model.action,{model.s:states})
            cf={model.s:states,model.executed:actual}
            full_value, analytic_value, chain_value=sess.run([full,analytic,chain],cf)
            chain_error=float(np.max(np.abs(full_value-analytic_value)))
            assert chain_error<1e-5 and np.max(np.abs(chain_value))>1e-6,chain_error
            # Verify equality after hard copy, including the new conditional vars.
            sess.run(model.target_init)
            assert np.array_equal(sess.run(model.action,{model.s:states}),sess.run(model.target_action,{model.s1:states}))
            export=sess.run(model.online_vars+[model.phase,model.schema,model.rate_schema,model.clock_schema])
            prediction=sess.run(model.action,{model.s:states})
            with tempfile.TemporaryDirectory(prefix='continuous-v4-') as td:
                prefix=str(Path(td)/'temporary'); saver.save(sess,prefix,write_meta_graph=False);model.save_manifest(prefix,cfg,{})
                sess.run(tf.global_variables_initializer());model.restore(sess,prefix)
                restore_error=float(np.max(np.abs(prediction-sess.run(model.action,{model.s:states}))));assert restore_error==0
                # Schema rejection occurs before the TensorFlow Saver reads tensors.
                record=json.loads(Path(prefix+'.joint.json').read_text());record['schema']=configuration('g4')['schema']
                Path(prefix+'.joint.json').write_text(json.dumps(record))
                try:model.restore(sess,prefix)
                except ValueError:pass
                else:raise AssertionError('Accepted incompatible old checkpoint')
    with tf.Graph().as_default():
        frozen=Model(cfg,False)
        selected=frozen.online_vars+[frozen.phase,frozen.schema,frozen.rate_schema,frozen.clock_schema]
        with tf.Session(config=session_cfg) as sess:
            sess.run(tf.global_variables_initializer());sess.run([tf.assign(v,x) for v,x in zip(selected,export)])
            export_error=float(np.max(np.abs(prediction-sess.run(frozen.action,{frozen.s:states}))));assert export_error==0
    result=dict(status='PASS_ACTUAL_TF1_CONTINUOUS_GRAPH_NOT_TCP',inputs=len(states),initial_window_max_error=parity,
        initial_period_max_error_ms=time_parity,same_decision_exploration_max_error=probe_error,
        old_wrong_window_pairing_detectable_difference=mismatch_visible,base20_structural_parity=True,
        calibration_actor_target_and_bn_bitwise_frozen=True,calibration_actor_update_rejected=True,
        joint_both_branches_update=True,teacher_loss_time_gradient_zero=True,online_and_target_bn_frozen=True,
        time_direct_plus_conditioned_chain_gradient_max_error=chain_error,
        target_includes_conditional_weights=True,full_restore_error=restore_error,inference_export_error=export_error,
        legacy_schema_rejected=True,real_tcp_runs=0)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--states',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();verify(args.config,args.states,args.output)
