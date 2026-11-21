"""TF1 graph, zero-anchor critic, envelope gradient, and Phase 1 pre-adaptation checks for v5."""
import argparse
import hashlib
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
    production_common = int(cfg.get('critic_common_calibration_updates', 0))
    if production_common:
        cfg['critic_common_calibration_updates'] = 2
    with np.load(str(states_file)) as p:
        states = p['states'][:128].copy()
    states[::3, 66] = 0.
    prior = np.where(states[:, 66] > 0., 20.5, np.clip(np.expm1(states[:, 79]) * 20., 40., 120.))
    session_cfg = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)

    with tf.Graph().as_default():
        tf.set_random_seed(3)
        model = Model(cfg, True)
        assert model.cond_window_step is not None
        anchor_time = [g for g in tf.gradients(model.anchor_loss, model.time_vars) if g is not None]
        anchor_cond = [g for g in tf.gradients(model.anchor_loss, model._variables(CONDITION, True)) if g is not None]
        chain = tf.gradients(tf.reduce_sum(model.action[:, :1]), model.time_logit)[0]

        q = model._critic(model.s, model.executed, 'critic')
        dq = tf.gradients(tf.reduce_sum(q), model.executed)[0]
        dtime = tf.gradients(tf.reduce_sum(model.action[:, 1:2]), model.time_logit)[0]
        time_direct_plus_barrier = tf.gradients(
            model.time_actor_loss * tf.cast(tf.shape(model.s)[0], tf.float32), model.time_logit
        )[0]
        expected_time_grad = -dq[:, 1:2] * dtime + float(cfg['time_barrier_weight']) * (
            2. * tf.sigmoid(model.time_logit) - 1.
        )

        # Zero-anchor property: residual and its parameter gradients at T=20ms must be exactly 0.
        period_vars = [v for v in model.critic_parameters[0] if '/period_residual/' in v.op.name]
        residual_at_exec = model._critic(model.s, model.executed, 'critic') - model._critic(
            model.s, tf.concat([model.executed[:, :1], -tf.ones_like(model.executed[:, 1:2])], 1), 'critic'
        )
        res_grads = tf.gradients(tf.reduce_sum(residual_at_exec), period_vars)

        stats = [a for a in model.window_vars if a.op.name.endswith(('/moving_mean', '/moving_variance'))]
        online_names = {v.op.name for v in model.online_vars}
        actor_pairs = [(a, b) for a, b in model.target_pairs if a.op.name in online_names]
        cond_pairs = [(a, b) for a, b in actor_pairs if a.op.name.startswith(CONDITION + '/')]
        non_cond_pairs = [(a, b) for a, b in actor_pairs if not a.op.name.startswith(CONDITION + '/')]
        saver = tf.train.Saver()

        with tf.Session(config=session_cfg) as sess:
            sess.run(tf.global_variables_initializer())
            model.initialize_baseline(sess, cfg['baseline_checkpoint'])

            action, teacher, duration = sess.run([model.action, model.teacher_output, model.planned_ms], {model.s: states})
            parity = float(np.max(np.abs(action[:, :1] - teacher)))
            assert parity < 1e-5, parity
            time_parity = float(np.max(np.abs(duration[:, 0] - prior)))
            assert time_parity < 1e-4, time_parity

            # Set non-zero period residual weights and verify exact zero at T=20ms.
            res_bias = next(v for v in period_vars if v.op.name.endswith('/output/bias'))
            sess.run(tf.assign(res_bias, [0.25]))
            act20 = np.column_stack([action[:, 0], -np.ones(len(states), np.float32)]).astype(np.float32)
            res20_val = sess.run(residual_at_exec, {model.s: states, model.executed: act20})
            res20_grads = sess.run(res_grads, {model.s: states, model.executed: act20})
            assert float(np.max(np.abs(res20_val))) == 0.0, res20_val
            assert all(float(np.max(np.abs(g))) == 0.0 for g in res20_grads), 'Nonzero period residual grad at 20ms'

            # Verify nonzero dQ/dT at T=20ms despite zero residual value!
            dq_at_20 = sess.run(dq, {model.s: states, model.executed: act20})
            assert float(np.min(np.abs(dq_at_20[:, 1]))) > 1e-4, dq_at_20[:, 1]
            sess.run(tf.assign(res_bias, [0.0]))
            sess.run(model.target_init)

            # Verify exploration pairs conditioned window with executed T.
            bias = next(v for v in model.window_vars if v.op.name == CONDITION + '/output/bias')
            sess.run(tf.assign(bias, [.15]))
            sess.run(model.target_init)
            before_all = sess.run([a for pair in actor_pairs for a in pair] + stats)

            draws = exploration_inputs(cfg, np.random.RandomState(9))
            draws.update(probability=0., global_fraction=.713)
            feed = {model.s: states}
            feed.update((model.behavior_inputs[k], v) for k, v in draws.items())
            behavior, pure, probe = sess.run([model.behavior_action, model.action, model.behavior_probe], feed)
            proposed = 20. * 8. ** draws['global_fraction']
            expected = sess.run(model.diagnostic_action, {model.s: states, model.diagnostic_ms: np.full((len(states), 1), proposed, np.float32)})
            probe_error = float(np.max(np.abs(behavior - expected)))
            assert probe and probe_error < 2e-6, probe_error

            durations = np.full((len(states), 1), .08, np.float64)
            reward, disc = interval_return(.9, .08, cfg['decision_cost'])
            batch = [states, action, np.full((len(states), 1), reward, np.float32), states * .99,
                     np.zeros((len(states), 1), np.float32), durations, np.full((len(states), 1), disc, np.float32), teacher]
            trainfeed = model.feed(batch)

            # Phase 0: critic common calibration then full critic; actor & cond_window rejected.
            for iteration in range(3):
                sess.run(model.critic_step, trainfeed)
                sess.run(model.target_update)
            assert all(np.array_equal(a, b) for a, b in zip(before_all, sess.run([a for pair in actor_pairs for a in pair] + stats)))
            for bad_op in (model.actor_step, model.cond_window_step):
                try:
                    sess.run(bad_op, trainfeed)
                except tf.errors.InvalidArgumentError:
                    pass
                else:
                    raise AssertionError('Phase 0 allowed actor/cond step')

            # Phase 1 (FORWARD_ONLY): cond_window_step allowed, base/time/BN frozen, actor_step rejected.
            sess.run(tf.assign(model.phase, 1))
            try:
                sess.run(model.actor_step, trainfeed)
            except tf.errors.InvalidArgumentError:
                pass
            else:
                raise AssertionError('Phase 1 allowed full actor_step')
            before_non_cond = sess.run([a for pair in non_cond_pairs for a in pair] + stats)
            before_cond = sess.run([a for pair in cond_pairs for a in pair])
            for _ in range(10):
                sess.run(model.critic_step, trainfeed)
                sess.run(model.cond_window_step, trainfeed)
                sess.run(model.target_update)
            assert all(np.array_equal(a, b) for a, b in zip(before_non_cond, sess.run([a for pair in non_cond_pairs for a in pair] + stats)))
            assert any(not np.array_equal(a, b) for a, b in zip(before_cond, sess.run([a for pair in cond_pairs for a in pair])))

            # Phase 2 (JOINT): all three branches update, BN frozen, envelope stop_gradient holds.
            sess.run(tf.assign(model.phase, 2))
            before_base = sess.run(model.base_window_vars)
            before_cond2 = sess.run(model._variables(CONDITION, True))
            before_time = sess.run(model.time_vars)
            before_stats = sess.run(stats)
            for _ in range(20):
                sess.run(model.critic_step, trainfeed)
                sess.run(model.actor_step, trainfeed)
                sess.run(model.target_update)
            assert all(np.array_equal(a, b) for a, b in zip(before_stats, sess.run(stats)))
            assert any(not np.array_equal(a, b) for a, b in zip(before_base, sess.run(model.base_window_vars)))
            assert any(not np.array_equal(a, b) for a, b in zip(before_cond2, sess.run(model._variables(CONDITION, True))))
            assert any(not np.array_equal(a, b) for a, b in zip(before_time, sess.run(model.time_vars)))
            assert all(np.count_nonzero(g) == 0 for g in sess.run(anchor_time, trainfeed))
            assert all(np.count_nonzero(g) == 0 for g in sess.run(anchor_cond, trainfeed))

            actual = sess.run(model.action, {model.s: states})
            cf = {model.s: states, model.executed: actual}
            chain_val, got_tg, exp_tg = sess.run([chain, time_direct_plus_barrier, expected_time_grad], cf)
            envelope_chain_zero = float(np.max(np.abs(chain_val)))
            assert envelope_chain_zero == 0.0, envelope_chain_zero
            time_grad_err = float(np.max(np.abs(got_tg - exp_tg)))
            assert time_grad_err < 1e-6, time_grad_err

            # Restore & schema rejection check
            prediction = sess.run(model.action, {model.s: states})
            with tempfile.TemporaryDirectory(prefix='continuous-v5-') as td:
                prefix = str(Path(td) / 'temporary')
                saver.save(sess, prefix, write_meta_graph=False)
                model.save_manifest(prefix, cfg, {})
                sess.run(tf.global_variables_initializer())
                model.restore(sess, prefix)
                restore_err = float(np.max(np.abs(prediction - sess.run(model.action, {model.s: states}))))
                assert restore_err == 0.0
                record = json.loads(Path(prefix + '.joint.json').read_text())
                record['schema'] = configuration('continuous-v4')['schema']
                Path(prefix + '.joint.json').write_text(json.dumps(record))
                try:
                    model.restore(sess, prefix)
                except ValueError:
                    pass
                else:
                    raise AssertionError('v5 accepted v4 schema')

    result = dict(
        status='PASS_ACTUAL_TF1_CONTINUOUS_V5_GRAPH',
        initial_window_max_error=parity,
        initial_period_max_error_ms=time_parity,
        zero_anchor_residual_at_20ms=True,
        zero_anchor_residual_grad_at_20ms=True,
        nonzero_dQ_dT_at_20ms=True,
        same_decision_exploration_max_error=probe_error,
        phase0_everything_frozen=True,
        phase1_cond_window_only_updates=True,
        phase2_all_three_branches_update=True,
        teacher_grad_zero_on_time_and_cond=True,
        envelope_stop_gradient_chain_max=envelope_chain_zero,
        time_direct_plus_barrier_grad_max_error=time_grad_err,
        restore_error=restore_err,
        v4_schema_rejected=True,
    )
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--states', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    verify(a.config, a.states, a.output)
