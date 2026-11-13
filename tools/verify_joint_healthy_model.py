"""G0 numerical tests on the REAL selected checkpoint. No sockets or TCP runs."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')


def verify(output):
    import numpy as np
    import tensorflow as tf
    import sonnet as snt
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.model import Model, WINDOW, TARGET_WINDOW
    from frameworks.joint_healthy.core import interval_return, JOINT, FORWARD_ONLY
    cfg = configuration(); cfg.update(num_actors=1, lr_a=.0001)
    source = cfg['baseline_checkpoint']
    rng = np.random.RandomState(19)
    states = np.tile(np.array([1., 1.2, 0., 1., .02, .98, 1.], np.float32), (512, 10))
    states += rng.uniform(-.015, .015, states.shape).astype(np.float32)
    states = np.clip(states, 0., None)
    extension = rng.uniform(-1, 3, (512, 12)).astype(np.float32)
    joint_states = np.concatenate((states, extension), axis=1)
    actions = rng.uniform(-.25, .25, (512, 2)).astype(np.float32)
    # Use the actual unchanged source class for the independent scalar reference.
    file = ROOT / 'frameworks/baseline/code/canopy/rl-module/agent_v2.py'
    tree = ast.parse(file.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'AbstractActor')
    namespace = dict(tf=tf, snt=snt)
    module = ast.Module(body=[node])
    if hasattr(module, 'type_ignores'): module.type_ignores = []
    exec(compile(ast.fix_missing_locations(module), str(file), 'exec'), namespace)
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Critic')
    module = ast.Module(body=[node])
    if hasattr(module, 'type_ignores'): module.type_ignores = []
    exec(compile(ast.fix_missing_locations(module), str(file), 'exec'), namespace)
    with tf.Graph().as_default():
        s0 = tf.placeholder(tf.float32, [None, 70]); training = tf.placeholder(tf.bool)
        actor = namespace['AbstractActor'](70, 1, 256, 256)
        actor(s0, training)
        spec = importlib.util.spec_from_file_location('original_moving_backend', str(ROOT / 'frameworks/baseline/moving_stats_backend.py'))
        backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
        proxy = type('Reference', (), dict(s0=s0))()
        original_output = backend.forward(proxy)
        scalar_action = tf.placeholder(tf.float32, [None, 1])
        scalar_q = [namespace['Critic'](70, 1, 256, 256, name=name).build(s0, scalar_action) for name in ('critic', 'critic2')]
        reference_vars = sorted(tf.global_variables(), key=lambda v: v.op.name)
        reader = tf.train.NewCheckpointReader(source)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([tf.assign(v, reader.get_tensor(v.op.name)) for v in reference_vars])
            reference = sess.run(original_output, {s0: states, training: False})
            reference_q = sess.run(scalar_q, {s0: states, scalar_action: actions[:, :1]})
            sess.run(tf.get_collection(tf.GraphKeys.UPDATE_OPS), {s0: states, training: True})
            expected_ema = {v.op.name: value for v, value in zip(reference_vars, sess.run(reference_vars)) if '/moving_' in v.op.name}
    result = dict(status='RUNNING', tf_version=tf.__version__, sonnet_version=snt.__version__,
                  source_checkpoint='models/healthy-baseline-seed3/training/step-6013196',
                  test_rows=len(states), new_TCP=0, physical_experience=0, synthetic_optimizer_checks=True)
    with tempfile.TemporaryDirectory(prefix='joint-g0-') as temporary:
        prefix = str(Path(temporary) / 'verified')
        with tf.Graph().as_default():
            tf.set_random_seed(3); model = Model(cfg, training=True)
            with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as sess:
                sess.run(tf.global_variables_initializer())
                provenance = model.initialize_baseline(sess, source)
                prediction = sess.run(model.action, {model.s: joint_states})
                migrated_q = sess.run([model.q1, model.q2], {model.s: joint_states, model.executed: actions})
                q_error = max(float(np.max(np.abs(a-b))) for a,b in zip(reference_q, migrated_q))
                for a,b in zip(reference_q,migrated_q): np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-5)
                error = float(np.max(np.abs(prediction[:, :1] - reference)))
                assert error < 3e-6, ('Migrated window differs from frozen runtime', error)
                assert np.all(prediction[:, 1] == -1.)
                assert sess.run(model.global_step) == 0
                sess.run(tf.assign(model.phase, JOINT))
                sess.run(model.window_update_ops, {model.s: joint_states})
                ema_error = max(float(np.max(np.abs(sess.run(v) - expected_ema[v.op.name]))) for v in model.window_vars if v.op.name in expected_ema)
                assert ema_error < 3e-6, ('BN does not match one canonical Sonnet EMA', ema_error)
                model.initialize_baseline(sess, source)
                sess.run(tf.assign(model.phase, 0))
                initial_window = model.digest(sess, window_only=True)
                initial_target = sess.run(model._variables(TARGET_WINDOW))
                # Target queries must not depend on unrelated rows of the batch.
                one = sess.run(model.target_action, {model.s1: joint_states[:64]})
                many = sess.run(model.target_action, {model.s1: np.r_[joint_states[:1], joint_states[64:127]]})
                assert np.max(np.abs(one[0] - many[0])) < 3e-6
                pure, smoothed = sess.run([model.target_action,model.target_policy],{model.s1:joint_states[:64]})
                np.testing.assert_array_equal(pure[:,1],smoothed[:,1])
                assert np.max(np.abs(pure[:,0]-smoothed[:,0])) <= .200001
                duration = rng.choice([.02, .04, .08, .12, .16], 512)
                rewards, discounts = zip(*[interval_return(float(u), float(dt)) for u, dt in zip(rng.uniform(.7, 1., 512), duration)])
                batch = [joint_states, actions, np.asarray(rewards, np.float32)[:, None], joint_states * .999,
                         np.zeros((512, 1), np.float32), duration[:, None], np.asarray(discounts, np.float32)[:, None]]
                feed = model.feed(batch)
                for _ in range(12): sess.run(model.critic_step, feed); sess.run(model.target_update)
                assert initial_window == model.digest(sess, True)
                for x, y in zip(initial_target, sess.run(model._variables(TARGET_WINDOW))): np.testing.assert_array_equal(x, y)
                powers = [float(sess.run(x._get_beta_accumulators()[0])) for x in model.critic_optimizers]
                assert model.critic_optimizers[0]._get_beta_accumulators()[0] is not model.critic_optimizers[1]._get_beta_accumulators()[0]
                assert all(abs(x - .9 ** 13) < 3e-6 for x in powers), powers
                # Even an explicitly fetched BN writer cannot mutate during freeze.
                sess.run(model.window_update_ops, feed)
                assert initial_window == model.digest(sess, True)
                try: sess.run(model.actor_step, feed)
                except tf.errors.InvalidArgumentError: pass
                else: raise AssertionError('Calibration allowed actor optimizer')
                sess.run(tf.assign(model.phase, FORWARD_ONLY))
                forward = sess.run(model.action, {model.s: joint_states})
                period = 20 * 8 ** ((forward[:, 1] + 1) / 2)
                assert np.max(np.abs(period - 40)) < 1e-4
                for _ in range(3): sess.run(model.critic_step, feed); sess.run(model.target_update)
                assert initial_window == model.digest(sess, True)
                sess.run(tf.assign(model.phase, JOINT))
                gradients = sess.run([model.window_grad, model.time_grad], feed)
                assert all(np.isfinite(gradients)) and min(gradients) > 0, gradients
                before_time = sess.run(model.time_vars)
                sess.run([model.actor_step, model.window_update_ops], feed)
                sess.run(model.target_update)
                post_one=sess.run(model.target_action,{model.s1:joint_states[:64]})
                post_many=sess.run(model.target_action,{model.s1:np.r_[joint_states[:1],joint_states[64:127]]})
                post_context_error=float(np.max(np.abs(post_one[0]-post_many[0])))
                assert post_context_error<3e-6
                assert initial_window != model.digest(sess, True)
                assert any(not np.array_equal(a, b) for a, b in zip(before_time, sess.run(model.time_vars)))
                assert len(model.window_update_ops) == 4
                destinations = [b.op.name for _, b in model.target_pairs]
                assert len(destinations) == len(set(destinations))
                saver = tf.train.Saver(); saver.save(sess, prefix)
                model.save_manifest(prefix, cfg, dict(critic_iterations=15, actor_updates=1))
                expected = sess.run(model.action, {model.s: joint_states})
                expected_digest = model.digest(sess)
                result.update(window_max_error=error, migrated_critic_old_input_max_error=q_error, canonical_sonnet_ema_max_error=ema_error, four_unique_online_BN_writers=4,
                    target_pairs=len(destinations), target_batch_context_max_error=float(np.max(np.abs(one[0] - many[0]))),
                    critic_beta1_powers_after12=powers, frozen_window_and_target_bitwise=True,
                    independent_critic_optimizer_accumulators=True, post_joint_target_context_max_error=post_context_error,
                    baseline_target_window_smoothing_retained=True, time_target_not_smoothed=True,
                    actor_rejected_during_calibration=True, forward_only_window_bitwise=True,
                    joint_gradients=list(map(float, gradients)), both_action_branches_updated=True,
                    optimizer_counters_fresh=True, provenance=provenance)
        with tf.Graph().as_default():
            restored = Model(cfg, training=False)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer()); restored.restore(sess, prefix)
                np.testing.assert_array_equal(expected, sess.run(restored.action, {restored.s: joint_states}))
                assert expected_digest == restored.digest(sess)
                result['checkpoint_inference_roundtrip_bitwise'] = True
        with tf.Graph().as_default():
            wrong = Model(dict(cfg, rate_mode='legacy'), training=False)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try: wrong.restore(sess, prefix)
                except ValueError: result['incompatible_measurement_checkpoint_rejected'] = True
                else: raise AssertionError('Incompatible measurement schema accepted')
    fixed_checks = []
    for ms in (20, 80, 160):
        with tf.Graph().as_default():
            fixed = Model(configuration('fixed%d' % ms), training=True)
            with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as sess:
                sess.run(tf.global_variables_initializer()); fixed.initialize_baseline(sess, source)
                sess.run(tf.assign(fixed.phase, JOINT))
                before = sess.run(fixed.time_vars)
                timing, gradient = sess.run([fixed.action, fixed.time_grad], fixed.feed(batch))
                assert np.max(np.abs(20 * 8 ** ((timing[:, 1] + 1) / 2) - ms)) < .0001
                assert gradient == 0.
                sess.run(fixed.actor_step, fixed.feed(batch))
                for a, b in zip(before, sess.run(fixed.time_vars)): np.testing.assert_array_equal(a, b)
                fixed_checks.append(dict(period_ms=ms, time_gradient=0., time_weights_unchanged=True))
    result['independent_fixed_control_checks'] = fixed_checks
    result['provenance']['source'] = '@PACKAGE@/' + result['source_checkpoint']
    result['status'] = 'PASS_REAL_CHECKPOINT_G0_NUMERICAL_ONLY'
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', required=True)
    verify(ap.parse_args().output)
