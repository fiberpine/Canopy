#!/usr/bin/env python3
"""One local TF1 optimizer smoke test; no TCP, RPC, actor or learner loop.

Uses the selected framework's distributed.setup and initialization helpers.
Only placement/transport is adapted: TF Server becomes an empty local target
and /job:learner placement becomes /cpu:0. No framework file is modified.
Synthetic batches and an explicitly opened timing gate test operation wiring,
not warmup coverage, real replay, convergence, performance or complete resume.
--checkpoint optionally checks delivered policy weights; training still starts
from the template's fresh initialization, restored before the optimizer test.
Run under the package's Python3.6/TF1 environment, once per framework/process.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys
import tempfile
import time
import traceback


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(distributed, agent, session, cfg, tf):
    """Same initialization order as distributed.learner; no learner I/O/loop."""
    session.run(tf.global_variables_initializer())
    agent.init_target()
    proof = {"mode": "random_baseline", "complete_resume": False}
    if cfg.get('cold_init_checkpoint'):
        raise ValueError('cold_init is unsupported here; use a fresh portable template')
    if cfg.get('warm_start_window_checkpoint'):
        proof = importlib.import_module('window_warm_start').initialize(
            agent, cfg['warm_start_window_checkpoint'], cfg)
    staged = hasattr(distributed, 'staged_timing') and distributed.staged_timing.enabled(cfg)
    if staged:
        trainable = importlib.import_module('trainable_timing')
        if cfg.get('initialize_joint_from_scratch') or cfg.get('matching_clock_prior_checkpoint'):
            proof = importlib.import_module('decision_clock_v21').fresh_initialize(agent, cfg)
        elif trainable.enabled(cfg):
            effective = dict(cfg, fixed_mtp_ms=None) if cfg.get('fixed_period_train_window') else cfg
            proof = trainable.initialize(agent, cfg['staged_joint_checkpoint'], effective)
        else:
            proof = importlib.import_module('frozen_window').initialize(
                agent, cfg['frozen_window_checkpoint'], cfg)
    elif cfg.get('joint_control') and not cfg.get('warm_start_window_checkpoint'):
        # Early joint models can be explicitly random. Do not invent a prior.
        require(not any(cfg.get(k) for k in ('staged_joint_checkpoint',
                'matching_clock_prior_checkpoint', 'frozen_window_checkpoint')),
                'Configured prior has no supported initialization branch')
        proof = {"mode": "random_joint", "complete_resume": False}
    return proof, staged


def verify(args, result):
    root = pathlib.Path(args.framework)
    require(root.is_absolute() and (root / 'distributed.py').is_file(),
            '--framework must be an absolute framework directory')
    config_path = pathlib.Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    result.update(framework=str(root.resolve()), config=str(config_path),
                  config_sha256=digest(config_path),
                  distributed_sha256=digest(root / 'distributed.py'))
    for key in ('matching_clock_prior_checkpoint', 'staged_joint_checkpoint',
                'frozen_window_checkpoint', 'warm_start_window_checkpoint'):
        if cfg.get(key):
            prefix = pathlib.Path(cfg[key])
            require(prefix.is_absolute() and pathlib.Path(str(prefix) + '.index').is_file(),
                    'Missing/localization failure for ' + key + ': ' + str(prefix))
    os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1')
    sys.path[:0] = [str(root), str(root / 'code/runtime/_deps'),
                    str(root / 'code/canopy/rl-module')]
    distributed = importlib.import_module('distributed')
    require(pathlib.Path(distributed.__file__).resolve() == (root / 'distributed.py').resolve(),
            'Wrong distributed module imported')
    tf, np = distributed.tf, distributed.np
    require(str(tf.__version__).startswith('1.'), 'Requires the preserved TF1 runtime')
    result['runtime'] = {'python': sys.version.split()[0], 'tensorflow': tf.__version__}
    real_server, real_device = tf.train.Server, tf.device
    transports = []

    class LocalServer(object):
        target = ''

        def __init__(self, *unused_args, **unused_kwargs):
            transports.append('TF Server suppressed; local Session target')

    def local_device(name):
        if isinstance(name, str) and name.startswith('/job:learner/'):
            name = '/cpu:0'
        return real_device(name)

    tf.reset_default_graph()
    tf.train.Server, tf.device = LocalServer, local_device
    session = None
    try:
        # role=learner exercises build_learn; no distributed.learner/actor runs.
        agent, session, fifo, ph, enq, nph, deq, size = distributed.setup(cfg, 'learner', 0)
    finally:
        tf.train.Server, tf.device = real_server, real_device
    require(session is not None and transports, 'Local-only setup adapter did not run')
    require(not any('/job:' in op.device for op in tf.get_default_graph().get_operations()),
            'Distributed placement escaped the local-only adapter')
    result['transport'] = transports
    try:
        proof, staged = initialize(distributed, agent, session, cfg, tf)
        result['fresh_initialization'] = proof
        sd, ad = int(agent.s0.shape.as_list()[-1]), int(agent.action.shape.as_list()[-1])
        require(ad == (2 if cfg.get('joint_control') else 1), 'Action dimension mismatch')
        if cfg.get('feedback_context_v31'):
            require((sd, ad) == (149, 2), 'V31 must be149-state/2-action')
        elif not cfg.get('joint_control'):
            require((sd, ad) == (70, 1), 'Baseline must be70-state/1-action')
        result['shape'] = {'state': sd, 'action': ad, 'batch': 512}
        rng = np.random.RandomState(831)
        s0 = rng.uniform(.05, .8, (512, sd)).astype(np.float32)
        s1 = rng.uniform(.05, .8, (512, sd)).astype(np.float32)
        action_tensor = getattr(agent, '_moving_action', agent.concrete_actor_out)
        action_feed = {agent.s0: s0, agent.is_training: False,
                       agent.is_concrete: True, agent.s0_delta: np.zeros_like(s0)}

        def actions():
            values = session.run(action_tensor, action_feed)
            require(values.shape == (512, ad) and np.isfinite(values).all()
                    and np.max(np.abs(values)) <= 1.00001, 'Invalid policy action output')
            return values

        variables = tf.global_variables()
        saver = tf.train.Saver(var_list=variables, max_to_keep=2)
        with tempfile.TemporaryDirectory(prefix='canopy-training-smoke-') as temp:
            fresh = saver.save(session, os.path.join(temp, 'fresh'), write_meta_graph=False)
            if args.checkpoint:
                cp = str(pathlib.Path(args.checkpoint).resolve())
                if cfg.get('joint_control'):
                    checker = root / 'duration_head_v16.py'
                    if checker.is_file():
                        importlib.import_module('duration_head_v16').check_checkpoint(dict(cfg, checkpoint=cp))
                policy = [v for v in variables if v.name.startswith('abstract_actor/')
                          and '/Adam' not in v.name]
                if staged:
                    policy = importlib.import_module('frozen_window').policy_variables(variables)
                require(policy, 'No policy variables for optional checkpoint check')
                reader = tf.train.NewCheckpointReader(cp)
                shapes = reader.get_variable_to_shape_map()
                for v in policy:
                    require(v.op.name in shapes and v.shape.as_list() == shapes[v.op.name],
                            'Delivered policy checkpoint shape mismatch: ' + v.op.name)
                tf.train.Saver(var_list=policy).restore(session, cp)
                delivered = actions()
                result['delivered_checkpoint'] = {'prefix': cp, 'policy_variables': len(policy),
                    'scope': 'Policy/BN only, then restored fresh initialization; not resume',
                    'action_min': float(delivered.min()), 'action_max': float(delivered.max())}
                saver.restore(session, fresh)
            # Explicitly test post-calibration optimizer wiring, without claiming
            # synthetic rows have passed actual warmup/period-coverage gates.
            if staged:
                require(hasattr(agent, 'enable_timing_learning'), 'Missing staged learning gate')
                session.run(agent.enable_timing_learning)
            result['timing_gate'] = 'Opened only for synthetic optimizer smoke' if staged else 'Not staged'
            values = actions()
            fd = dict(action_feed)
            fd.update({agent.s1: s1, agent.action: values,
                       agent.reward: rng.uniform(.1, .8, (512, 1)).astype(np.float32),
                       agent.terminal: np.zeros((512, 1), np.float32), agent.is_training: True})
            if cfg.get('joint_control'):
                require(hasattr(agent, 'transition_discount'), 'Missing SMDP discount input')
                duration = rng.uniform(.02, .16, (512, 1)).astype(np.float32)
                fd[agent.transition_discount] = (cfg.get('gamma', .995) ** (duration / .02)).astype(np.float32)
            critic_variables = [v for v in tf.trainable_variables()
                                if v.op.name.startswith(('critic/', 'critic2/'))]
            actor_variables = [v for v in tf.trainable_variables()
                               if v.op.name.startswith('abstract_actor/')]
            require(critic_variables and actor_variables, 'Missing actual training variables')
            before_q = session.run(critic_variables)
            before_actor = session.run(actor_variables)
            before_step = int(session.run(agent.global_step))
            bn = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                  if 'actor' in op.name and 'target' not in op.name
                  and (not cfg.get('bn_relevant_only') or 'abstract_actor' in op.name)]
            graph_ops = len(tf.get_default_graph().get_operations())
            require(isinstance(agent.critic_train_op, (list, tuple)) and len(agent.critic_train_op) == 2,
                    'Expected the actual two-critic optimizer operation list')
            session.run(agent.critic_train_op, fd)
            after_critic_step = int(session.run(agent.global_step))
            session.run([agent.actor_train_op, bn], fd)
            after_actor_step = int(session.run(agent.global_step))
            after_q, after_actor = session.run([critic_variables, actor_variables])
            for scope in ('critic/', 'critic2/'):
                require(any(not np.array_equal(x, y) for v, x, y in zip(critic_variables, before_q, after_q)
                            if v.op.name.startswith(scope)), scope + ' weights did not change')
            require(any(not np.array_equal(x, y) for x, y in zip(before_actor, after_actor)),
                    'Actor weights did not change')
            require(after_critic_step - before_step == 2 and after_actor_step - after_critic_step == 1,
                    'Optimizer global_step does not match two critic plus one actor operations')
            bn_pairs = []
            if cfg.get('target_bn_consistent'):
                for target in variables:
                    n = target.op.name
                    if n.startswith('target_actor/') and ('moving_mean' in n or 'moving_variance' in n):
                        suffix = '/bn' + n.split('/bn', 1)[1]
                        sources = [v for v in variables if v.op.name.startswith('abstract_actor/')
                                   and v.op.name.endswith(suffix)]
                        require(len(sources) == 1, 'Ambiguous target BN source: ' + n)
                        bn_pairs.append((sources[0], target))
                require(len(bn_pairs) == 4, 'Expected four target BN statistics')
            before_bn = [(session.run(s), session.run(t)) for s, t in bn_pairs]
            agent.target_update()
            errors = []
            for (source, target), (source_value, old_target) in zip(bn_pairs, before_bn):
                expected = .001 * source_value + .999 * old_target
                actual = session.run(target)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                errors.append(float(np.max(np.abs(actual - expected))))
            context_delta = None
            if cfg.get('target_bn_consistent'):
                changed = s1.copy()
                changed[1:] = rng.uniform(2., 8., changed[1:].shape)
                tfd = {agent.s1: s1, agent.is_training: False}
                first = session.run(agent.target_actor_out, tfd)[0]
                tfd[agent.s1] = changed
                second = session.run(agent.target_actor_out, tfd)[0]
                context_delta = float(np.max(np.abs(first - second)))
                np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
            losses = session.run([agent.c_loss, agent.c_loss2, agent.a_loss], fd)
            require(np.isfinite(losses).all(), 'Nonfinite training loss')
            require(len(tf.get_default_graph().get_operations()) == graph_ops,
                    'Graph grew during optimizer/target calls')
            snapshot = session.run(variables)
            require(all(np.isfinite(x).all() for x in snapshot), 'Nonfinite model/optimizer variable')
            output_before = actions()
            saved = saver.save(session, os.path.join(temp, 'updated'), write_meta_graph=True)
            # Demonstrate actual reload, not just the creation of files.
            session.run(tf.assign(actor_variables[0], np.zeros_like(session.run(actor_variables[0]))))
            saver.restore(session, saved)
            for v, expected, actual in zip(variables, snapshot, session.run(variables)):
                np.testing.assert_array_equal(actual, expected, err_msg=v.op.name)
            np.testing.assert_array_equal(actions(), output_before)
            result.update(status='PASS_OFFLINE_TRAINING_OPERATIONS',
                counters={'critic_iterations': 1, 'critic_optimizer_operations': 2,
                          'actor_updates': 1, 'target_updates': 1,
                          'global_step_before': before_step, 'global_step_after': after_actor_step},
                losses=[float(x) for x in losses], graph_operations=graph_ops,
                target_bn={'configured_consistent': bool(cfg.get('target_bn_consistent')),
                           'statistics_checked': len(bn_pairs),
                           'tau_max_error': max(errors) if errors else None,
                           'fixed_query_batch_context_delta': context_delta,
                           'legacy_mode_preserved': not bool(cfg.get('target_bn_consistent'))},
                temporary_checkpoint={'all_variables_exact_after_reload': True,
                                      'policy_outputs_exact_after_reload': True,
                                      'variables': len(variables), 'removed_after_test': True},
                action_range=[float(output_before.min()), float(output_before.max())])
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.time()
    result = {'status': 'FAILED', 'TCP_runs': 0, 'training_environment': 'synthetic finite batch',
              'network_transport': 'local TF session only', 'complete_resume': False,
              'scope': 'One complete optimizer/actor/target iteration; no real replay/coverage/convergence proof'}
    code = 0
    try:
        verify(args, result)
    except Exception as error:
        code = 1
        result.update(status='FAILED', error=str(error), traceback=traceback.format_exc()[-6000:])
    result['elapsed_seconds'] = time.time() - started
    out = pathlib.Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': str(out),
                      'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
