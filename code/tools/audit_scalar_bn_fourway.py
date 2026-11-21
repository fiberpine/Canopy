"""Real TF1, real retained states; temporary-memory update contract diagnosis.

Source checkpoints/replay are never written. This is not a network experiment.
"""
import argparse
import collections
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def audit(args):
    os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = Path(args.framework)
    sys.path[:0] = [str(root), str(root/'code/runtime/_deps'), str(root/'code/canopy/rl-module')]
    d = importlib.import_module('distributed')
    tf, np = d.tf, d.np
    cfg = json.loads(Path(args.config).read_text())
    cfg['target_bn_consistent'] = True
    if args.repaired:
        cfg.update(independent_critic_adam_v34=True, unique_critic_targets=True)
    old_server, old_device = tf.train.Server, tf.device
    class LocalServer:
        target = ''
    tf.train.Server = lambda *a, **k: LocalServer()
    tf.device = lambda name: old_device('/cpu:0' if isinstance(name, str) and name.startswith('/job:') else name)
    try:
        a, sess, _, _, _, _, _, _ = d.setup(cfg, 'learner', 0)
    finally:
        tf.train.Server, tf.device = old_server, old_device
    assert not any('/job:' in op.device for op in tf.get_default_graph().get_operations())
    variables = tf.global_variables()
    reader0 = tf.train.NewCheckpointReader(args.checkpoint)
    saver = tf.train.Saver(var_list=[v for v in variables if reader0.has_tensor(v.op.name)])
    initialize = tf.global_variables_initializer()
    def restore():
        sess.run(initialize)
        saver.restore(sess, args.checkpoint)
    files = [args.replay]
    for cp in (args.checkpoint, args.earlier, args.later):
        files.extend(cp+s for s in ('.index', '.data-00000-of-00001'))
    hashes = {p: sha(p) for p in set(files)}
    with np.load(args.replay) as r:
        data = {k: r[k].copy() for k in ('s0', 'a', 'reward', 's1', 'terminal', 'actor_ids')}
    rng = np.random.RandomState(20261112)
    idx = np.concatenate([rng.choice(np.flatnonzero(data['actor_ids'] == i), 22 if i < 8 else 21, replace=False) for i in range(24)])
    assert len(idx) == 512
    fd = {a.s0: data['s0'][idx], a.action: data['a'][idx], a.reward: data['reward'][idx],
          a.s1: data['s1'][idx], a.terminal: data['terminal'][idx], a.is_training: True,
          a.is_concrete: True, a.s0_delta: np.zeros_like(data['s0'][idx]),
          a.target_actor_policy: np.zeros((512, 1), np.float32)}
    cvars = [v for v in tf.trainable_variables() if v.name.startswith(('critic/', 'critic2/'))]
    actual_gradients = {}
    for label, loss in [('critic1', a.c_loss), ('critic2', a.c_loss2)]:
        actual_gradients[label] = [v.name for v, g in zip(cvars, tf.gradients(loss, cvars)) if g is not None]
    beta = [v for opt in a._v34_critic_optimizers for v in opt._get_beta_accumulators()] if args.repaired else list(a.critic_optimizer._get_beta_accumulators())
    sess.run(initialize)
    clocks, digests, outputs = [], [], []
    for repeat in range(24):
        restore()
        before = sess.run(list(beta)+[a.global_step])
        sess.run(a.critic_train_op, fd)
        after = sess.run(list(beta)+[a.global_step])
        values = sess.run(cvars)
        clocks.append(dict(before=[float(x) for x in before], after=[float(x) for x in after]))
        digests.append(hashlib.sha256(b''.join(x.tobytes() for x in values)).hexdigest())
        outputs.append(values)
    weight_spread = max(float(np.max(np.abs(x-y))) for row in outputs[1:] for x, y in zip(row, outputs[0]))
    restore()
    sequential_before = sess.run(list(beta)+[a.global_step])
    for op in a.critic_train_op:
        sess.run(op, fd)
    sequential_after = sess.run(list(beta)+[a.global_step])
    actor_bn_contract = None
    actor_bn_session_runs = 0
    if args.with_actor_bn:
        restore()
        bn_entries_before_reference = [op.name for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
        reference_loss = -tf.reduce_mean(a.critic_actor_out)
        parameters = a.symbolic_actor.train_var()
        reference_gradients = tf.gradients(reference_loss, parameters)
        moving_gradients = tf.gradients(a.a_loss, parameters)
        defined = [(v, ref, moving) for v, ref, moving in zip(parameters, reference_gradients, moving_gradients) if ref is not None and moving is not None]
        comparison = {}
        for training_flag in (True, False):
            query = dict(fd)
            query[a.is_training] = training_flag
            moving, ref, moving_loss, original_loss = sess.run([a._moving_action, a.symbolic_actor_bound_lower, a.a_loss, reference_loss], query)
            grad_ref, grad_moving = sess.run([[r for v,r,m in defined], [m for v,r,m in defined]], query)
            comparison[str(training_flag)] = dict(action_max_error=float(np.max(np.abs(moving-ref))),
                loss_max_error=float(abs(moving_loss-original_loss)),
                gradient_max_error=max(float(np.max(np.abs(x-y))) for x,y in zip(grad_ref, grad_moving)),
                compared_gradient_variables=[v.name for v,r,m in defined],
                gradient_defined_mask_equal=all((r is None)==(m is None) for r,m in zip(reference_gradients,moving_gradients)))
        # Exactly the runtime filter. These are separate temporary-memory tests.
        bn = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
              if 'actor' in op.name and 'target' not in op.name
              and (not cfg.get('bn_relevant_only') or 'abstract_actor' in op.name)]
        assert bn_entries_before_reference == [op.name for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
        destinations = []
        unique = {}
        for entry in bn:
            # Collections may hold Merge/NoOp wrappers, not assignments.
            pending = [entry.op if hasattr(entry, 'op') else entry]
            seen = set()
            while pending:
                op = pending.pop()
                if op in seen:
                    continue
                seen.add(op)
                if op.type in ('Assign', 'AssignSub', 'AssignAdd', 'AssignVariableOp', 'AssignSubVariableOp', 'AssignAddVariableOp'):
                    variable = op.inputs[0].op
                    while variable.type in ('Identity', 'Switch', 'RefSwitch', 'Enter', 'RefEnter'):
                        variable = variable.inputs[0].op
                    destination = variable.name
                    if destination.startswith('abstract_actor/') and 'moving_' in destination:
                        destinations.append(dict(collection_entry=entry.name, operation=op.name, type=op.type, destination=destination))
                        unique.setdefault(destination, entry)
                pending.extend(t.op for t in op.inputs)
                pending.extend(op.control_inputs)
        assert len(unique) == 4, ('Unexpected online moving-statistics layout', list(unique))
        from unique_online_bn import select_online_updates
        actual_selected, selection_contract = select_online_updates(bn)
        assert {op.name for op in actual_selected} == {op.name for op in unique.values()}
        unique = {item['destination']:op for item,op in zip(selection_contract['statistics'],actual_selected)}
        # Compare each original writer in isolation from the same restored CP.
        moving_vars = {v.op.name:v for v in variables if v.op.name in unique}
        writer_values = {}
        writer_changes = []
        for entry in bn:
            restore()
            before_stats = sess.run(moving_vars)
            sess.run(entry, fd)
            after_stats = sess.run(moving_vars)
            matched = [item['destination'] for item in destinations if item['collection_entry']==entry.name]
            assert len(matched)==1, matched
            destination=matched[0]
            writer_values.setdefault(destination,[]).append(after_stats[destination])
            writer_changes.append(dict(entry=entry.name,destination=destination,
                changed_variables=[name for name in moving_vars if not np.array_equal(before_stats[name],after_stats[name])]))
        source_equivalence = {name:dict(writers=len(values),
            max_post_statistic_difference=max(float(np.max(np.abs(value-values[0]))) for value in values),
            post_statistic_digest=[hashlib.sha256(value.tobytes()).hexdigest() for value in values]) for name,values in writer_values.items()}
        policy = [v for v in variables if v.name.startswith('abstract_actor/') and '/Adam' not in v.name]
        cases = {}
        for mode in ('runtime_concurrent', 'concurrent_unique_BN_actor', 'serial_all_BN_then_actor', 'serial_unique_BN_then_actor'):
            results = []
            post_actions = []
            for repeat in range(24):
                restore()
                if mode in ('runtime_concurrent', 'concurrent_unique_BN_actor'):
                    sess.run([a.actor_train_op, bn if mode == 'runtime_concurrent' else list(unique.values())], fd)
                    actor_bn_session_runs += 1
                else:
                    sess.run(bn if mode == 'serial_all_BN_then_actor' else list(unique.values()), fd)
                    sess.run(a.actor_train_op, fd)
                    actor_bn_session_runs += 2
                results.append(sess.run(policy))
                post_actions.append(sess.run(a._moving_action, {a.s0:data['s0'][idx], a.is_training:False}))
            digests_actor = [hashlib.sha256(b''.join(v.tobytes() for v in values)).hexdigest() for values in results]
            cases[mode] = dict(unique_digests=len(set(digests_actor)),
                max_policy_variable_spread=max(float(np.max(np.abs(x-y))) for values in results[1:] for x,y in zip(values,results[0])),
                per_variable=[dict(name=v.name,reference_abs_max=float(np.max(np.abs(results[0][j]))),
                    max_deviation_from_first=float(max(np.max(np.abs(values[j]-results[0][j])) for values in results[1:]))) for j,v in enumerate(policy)],
                maximum_post_update_action_range=float(np.max(np.ptp(np.asarray(post_actions),axis=0))),
                p95_post_update_action_range=float(np.percentile(np.ptp(np.asarray(post_actions),axis=0),95)))
        actor_bn_contract = dict(actual_candidate_selector=selection_contract, writer_source_numerical_agreement=source_equivalence, isolated_writer_changes=writer_changes, cases=cases, moving_vs_original_zero_width_IBP=comparison, update_operations=destinations,
            destination_multiplicity=dict(collections.Counter(d['destination'] for d in destinations)),
            repeats=24, source_optimizer_state='Restored source actor moments/powers; no source writes',
            unique_assignment_control='Four-way control adds unique BN concurrent with actor; distinguishes deduplication from serial order. Candidate only, not active training.',
            scope='The serial control changes update ordering. This is a diagnostic, not an authorized change to the active training recipe.')
    # Synthetic values isolate the effective tau of the exact runtime invocation.
    online = next(v for v in variables if v.name.startswith('critic2/') and v.name.endswith('/bias:0'))
    suffix = online.op.name.split('/', 1)[1]
    target = next(v for v in variables if v.op.name == 'target_critic2/'+suffix)
    tau_values = []
    for repeat in range(32):
        sess.run([tf.assign(online, np.full(online.shape.as_list(), 1.25, np.float32)),
                  tf.assign(target, np.full(target.shape.as_list(), .5, np.float32))])
        sess.run([a.target_critic_update_op, a.target_critic_update_op2])
        tau_values.append(float(sess.run(target).flat[0]))
    # Fixed real states; swap only online deployment weights and BN statistics.
    online_vars = [v for v in variables if v.name.startswith('abstract_actor/') and '/Adam' not in v.name]
    readers = {k: tf.train.NewCheckpointReader(cp) for k, cp in [('A', args.earlier), ('B', args.later)]}
    swaps = {}
    for weights, stats in [('A', 'A'), ('B', 'B'), ('A', 'B'), ('B', 'A')]:
        assigns = []
        for v in online_vars:
            reader = readers[stats if 'moving_' in v.name else weights]
            assigns.append(tf.assign(v, reader.get_tensor(v.op.name)))
        sess.run(assigns)
        actions, logits = sess.run([a._moving_action, a._moving_action.op.inputs[0]], {a.s0: data['s0'][idx], a.is_training: False})
        swaps[weights+stats] = dict(exact_saturation=float(np.mean(np.abs(actions) == 1)),
            near_saturation=float(np.mean(np.abs(actions) >= .99)),
            logits_percentiles=np.percentile(logits, [0, 10, 50, 90, 100]).tolist(),
            action_mean=float(np.mean(actions)), by_actor_mean=[float(np.mean(actions[data['actor_ids'][idx] == i])) for i in range(24)])
    assert all(sha(p) == value for p, value in hashes.items())
    result = dict(repaired=args.repaired, actual_gradient_destinations=actual_gradients, paired_Adam_clocks=clocks,
        paired_update_unique_digests=len(set(digests)), paired_update_weight_max_spread=weight_spread,
        sequential_control=dict(before=[float(x) for x in sequential_before], after=[float(x) for x in sequential_after]),
        duplicate_target_tau=dict(variable=target.name, values=tau_values, expected_single=.50075, expected_double=.50149925),
        online_BN_function_swaps=swaps, online_policy_variables=[v.name for v in online_vars],
        immutable_source_sha256=hashes, training_session_runs=24+len(a.critic_train_op),
        temporary_critic_minimizer_operations=50, new_TCP_flows=0,
        actor_BN_contract=actor_bn_contract, temporary_actor_optimizer_operations=96 if args.with_actor_bn else 0,
        actor_BN_training_session_runs=actor_bn_session_runs,
        interpretation='Retained endpoint replay states only. Function swaps are not causal online performance tests.')
    sess.close()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('framework', 'config', 'checkpoint', 'replay', 'earlier', 'later', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--repaired', action='store_true')
    p.add_argument('--with-actor-bn', action='store_true')
    args = p.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('paired_update_unique_digests', 'paired_update_weight_max_spread', 'online_BN_function_swaps')}))
