"""Temporary TF1 numerical validation; no TCP, distributed server, or formal run.

The real Agent and its migration are checked, plus small deterministic gradient
fixtures for optimizer recurrence and next-step save/restore. Only temporary
validation variables are optimized. No checkpoint/replay from training is edited.
"""
import os, sys, pathlib, json, tempfile, types, hashlib, time
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT/'frameworks/mtpjoint'
sys.path[:0] = [str(ROOT), str(R), str(R/'code/runtime/_deps'), str(R/'code/canopy/rl-module')]
import numpy as np
import importlib.util
if importlib.util.find_spec('tensorflow') is None:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=pathlib.Path(args.output);assert not output.exists(), 'Do not overwrite evidence'
    result=dict(status='BLOCKED_BY_ENVIRONMENT',reason='Preserved TensorFlow1/Sonnet runtime required',TCP_runs=0)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result));sys.exit(2)
import tensorflow as tf
import independent_critic_v34 as fix


def optimizer_values(session, optimizers, parameter_sets):
    tensors = []
    for optimizer, parameters in zip(optimizers, parameter_sets):
        tensors.extend(optimizer._get_beta_accumulators())
        for variable in parameters:
            tensors.extend([optimizer.get_slot(variable, 'm'), optimizer.get_slot(variable, 'v')])
    return session.run(tensors)


def miniature(threads, loss_type, per, alternate=False):
    with tf.Graph().as_default():
        class Mini:
            CDQ = True
            LOSS_TYPE = loss_type
            PER = per
            lr_c = .001
            def build_critic_train_op_huber(self): raise AssertionError('Patch not installed')
            def build_critic_train_op(self): raise AssertionError('Patch not installed')
        module = types.SimpleNamespace(Agent=Mini); fix.install(module, {fix.FLAG: True})
        a = Mini(); a.global_step = tf.train.get_or_create_global_step()
        a.critic = types.SimpleNamespace(name='critic'); a.critic2 = types.SimpleNamespace(name='critic2')
        initial = np.array([.3, -.8, 1.5], np.float32)
        with tf.variable_scope('critic'): a.critic_out = tf.get_variable('w', initializer=initial)
        with tf.variable_scope('critic2'): a.critic_out2 = tf.get_variable('w', initializer=initial * 1.3)
        a.y = tf.constant([.1, .2, -.3], tf.float32); a.y2 = a.y
        a.importance = tf.constant([1., .5, 2.], tf.float32)
        op = a.build_critic_train_op() if alternate else a.build_critic_train_op_huber()
        gradients = [tf.gradients(loss, variables)[0] for loss, variables in zip(
            (a.c_loss, a.c_loss2), a._v34_critic_parameters)]
        saver = tf.train.Saver(); global_variables = tf.global_variables()
        errors = dict(weight=0., first_moment=0., second_moment=0., powers=0.)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=threads,
                                              inter_op_parallelism_threads=threads)) as sess:
            sess.run(tf.global_variables_initializer())
            expected = [sess.run(a.critic_out).astype(np.float64), sess.run(a.critic_out2).astype(np.float64)]
            first = [np.zeros(3), np.zeros(3)]; second = [np.zeros(3), np.zeros(3)]
            # Check the original entry's weighted MSE/Huber reductions separately.
            losses = sess.run([a.c_loss, a.c_loss2])
            for value, estimate in zip(losses, expected):
                difference = estimate - np.array([.1, .2, -.3])
                element = (.5*np.minimum(abs(difference), 1.)**2 + np.maximum(abs(difference)-1., 0.)) if loss_type == 'HUBER' and not alternate else difference**2
                weights = np.array([1., .5, 2.]) if per and not alternate else 1.
                np.testing.assert_allclose(value, np.mean(element*weights), atol=2e-7, rtol=2e-6)
            for k in range(1, 101):
                gs = sess.run(gradients)
                for i, gradient in enumerate(gs):
                    gradient = gradient.astype(np.float64)
                    first[i] = .9*first[i] + .1*gradient
                    second[i] = .999*second[i] + .001*gradient**2
                    rate = .001*np.sqrt(1-.999**k)/(1-.9**k)
                    expected[i] -= rate*first[i]/(np.sqrt(second[i])+1e-8)
                sess.run(op)
                assert int(sess.run(a.global_step)) == 2*k
                for i, (optimizer, variables) in enumerate(zip(a._v34_critic_optimizers, a._v34_critic_parameters)):
                    value, m, v = sess.run([variables[0], optimizer.get_slot(variables[0], 'm'), optimizer.get_slot(variables[0], 'v')])
                    powers = sess.run(optimizer._get_beta_accumulators())
                    errors['weight'] = max(errors['weight'], float(np.max(abs(value-expected[i]))))
                    errors['first_moment'] = max(errors['first_moment'], float(np.max(abs(m-first[i]))))
                    errors['second_moment'] = max(errors['second_moment'], float(np.max(abs(v-second[i]))))
                    errors['powers'] = max(errors['powers'], float(np.max(abs(np.array(powers)-[.9**(k+1), .999**(k+1)]))))
                    np.testing.assert_allclose(value, expected[i], rtol=2e-5, atol=3e-6)
                    np.testing.assert_allclose(m, first[i], rtol=2e-5, atol=3e-6)
                    np.testing.assert_allclose(v, second[i], rtol=3e-5, atol=3e-6)
                    np.testing.assert_allclose(powers, [.9**(k+1), .999**(k+1)], rtol=3e-5, atol=3e-6)
            with tempfile.TemporaryDirectory(prefix='v34-mini-') as temp:
                cp = saver.save(sess, str(pathlib.Path(temp)/'model'), write_meta_graph=False)
                sess.run(op); next_values = sess.run(global_variables)
                saver.restore(sess, cp); sess.run(op); restored_next = sess.run(global_variables)
                for before, after in zip(next_values, restored_next): np.testing.assert_array_equal(before, after)
        return dict(threads=threads, loss=loss_type, PER=per, alternate_MSE_entry=alternate,
                    iterations=100, max_errors=errors, deterministic_next_step_restore_exact=True)


def actual_agent(cfg):
    import agent_v2, moving_stats_backend, consistent_target_bn, trainable_timing, decision_clock_v21
    from joint_mtp import validate
    validate(cfg); assert fix.enabled(cfg)
    moving_stats_backend.install(agent_v2); consistent_target_bn.install(agent_v2)
    trainable_timing.install(agent_v2, cfg); fix.install(agent_v2, cfg)
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.get('seed', 3))
        a = agent_v2.Agent(149, 2, 256, 256, gamma=cfg['gamma'], batch_size=512, lr_a=.0001,
            lr_c=.001, tau=.001, mem_size=1, CDQ=True, PER=False, LOSS_TYPE=cfg['LOSS_TYPE'],
            use_original=False, use_snt_model_wo_ibp=False, smdp=True)
        consistent_target_bn.connect_statistics(a, .001); a.build_learn()
        assert a.critic_optimizer._get_beta_accumulators() == (None, None), 'Unused stock Adam created powers'
        assert a.critic_optimizer.get_slot_names() == [], 'Unused stock Adam created slots'
        for i, optimizer in enumerate(a._v34_critic_optimizers):
            assert len(a._v34_critic_parameters[i]) == 6
            for variable in a._v34_critic_parameters[i]:
                for slot in ('m', 'v'):
                    assert '/Adam' in optimizer.get_slot(variable, slot).op.name
                    assert a._v34_critic_optimizers[1-i].get_slot(variable, slot) is None
        powers = [v for opt in a._v34_critic_optimizers for v in opt._get_beta_accumulators()]
        assert len(set(v.op.name for v in powers)) == 4
        saver = tf.train.Saver(); variables = tf.global_variables()
        opt_variables = [v for v in variables if '/Adam' in v.op.name or v.op.name.startswith(('beta1_power','beta2_power')) or v.op.name == 'global_step']
        rng = np.random.RandomState(34); s = rng.uniform(-.3, .7, (8,149)).astype(np.float32)
        fd = {a.s0:s, a.s1:s*.9, a.action:rng.uniform(-.5,.5,(8,2)).astype(np.float32),
              a.reward:np.ones((8,1),np.float32), a.terminal:np.zeros((8,1),np.float32),
              a.transition_discount:np.full((8,1),.995**4,np.float32), a.is_training:True,
              a.is_concrete:True, a.s0_delta:np.zeros_like(s)}
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
            a.assign_sess(sess); sess.run(tf.global_variables_initializer()); a.init_target()
            before = sess.run(opt_variables)
            import compatible_calibration_v37
            proof = compatible_calibration_v37.initialize(a, cfg)
            for x,y in zip(before, sess.run(opt_variables)): np.testing.assert_array_equal(x,y)
            assert int(sess.run(a.global_step)) == 0
            for opt, params in zip(a._v34_critic_optimizers, a._v34_critic_parameters):
                np.testing.assert_allclose(sess.run(opt._get_beta_accumulators()), [.9,.999], atol=1e-7)
                for var in params:
                    assert not np.count_nonzero(sess.run(opt.get_slot(var,'m')))
                    assert not np.count_nonzero(sess.run(opt.get_slot(var,'v')))
            sess.run(a.critic_train_op, fd); assert int(sess.run(a.global_step)) == 2
            critic_states = optimizer_values(sess, a._v34_critic_optimizers, a._v34_critic_parameters)
            sess.run(a.enable_timing_learning); sess.run(a.actor_train_op, fd)
            assert int(sess.run(a.global_step)) == 3
            for x,y in zip(critic_states, optimizer_values(sess,a._v34_critic_optimizers,a._v34_critic_parameters)): np.testing.assert_array_equal(x,y)
            opt_before_target = sess.run(opt_variables); a.target_update()
            for x,y in zip(opt_before_target,sess.run(opt_variables)): np.testing.assert_array_equal(x,y)
            with tempfile.TemporaryDirectory(prefix='v34-actual-') as temp:
                cp = saver.save(sess,str(pathlib.Path(temp)/'model'),write_meta_graph=False)
                all_saved = sess.run(variables); sess.run(a.critic_train_op,fd)
                saver.restore(sess,cp)
                for x,y in zip(all_saved,sess.run(variables)): np.testing.assert_array_equal(x,y)
        return dict(actual_Agent=True, migration=proof, unused_stock_Adam_variables=0,
                    disjoint_critic_parameters=[len(x) for x in a._v34_critic_parameters],
                    independent_power_variables=[v.op.name for v in powers], fresh_slots_verified=True,
                    global_step_after_critic_actor=3, actor_does_not_touch_critic_optimizer=True,
                    target_does_not_touch_any_optimizer=True, checkpoint_all_variables_restored_exact=True,
                    shared_target_tensor=a._v34_critic_target.name,
                    limitation='Actual target smoothing RNG is not checkpointed; next-step exactness is tested only by the deterministic miniature fixture, not asserted as full training resume.')


def main():
    started = time.monotonic()
    import argparse, run as entry
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=pathlib.Path(args.output);assert not output.exists(), 'Do not overwrite verification evidence'
    _,cfg=entry.template('mtpjoint','train')
    cfg.update(independent_critic_adam_v34=True,compatible_v31_checkpoint=str(ROOT/'models/mtpjoint/model'),matching_clock_prior_checkpoint=None)
    trials = [miniature(threads, loss, per) for threads in [1,2]
              for loss,per in [('MSE',False),('MSE',True),('HUBER',False),('HUBER',True)]]
    trials.append(miniature(2,'MSE',True,alternate=True))
    result = dict(status='PASS_INDEPENDENT_CRITIC_ADAM_TF1_V34', tensorflow=tf.__version__,
                  miniature=trials, actual=actual_agent(cfg), elapsed_s=time.monotonic()-started,
                  TCP_runs=0, formal_training_updates=0,
                  scope='Only disposable RAM optimizer fixtures/Agent, synthetic inputs, source-prior migration, and automatically removed temporary checkpoints. No distributed server or replay.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == '__main__': main()
