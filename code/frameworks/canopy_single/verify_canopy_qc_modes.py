"""Frozen TF1 comparison: same histories/checkpoint, two QC loss-center modes.

No TCP, optimization, replay insertion, or checkpoint writes. Optional replay
provides real history inputs only; raw reward/cwnd/history contexts below are
explicit synthetic probes, never claimed to be measured network rewards.
"""
import argparse
import hashlib
import json
import pathlib
import sys
from collections import deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--replay')
    parser.add_argument('--real-count', type=int, default=12)
    parser.add_argument('--output')
    args = parser.parse_args()
    root = pathlib.Path(args.framework).resolve()
    sys.path[:0] = [str(root), str(root / 'code/runtime/_deps'),
                    str(root / 'code/canopy/rl-module')]
    import numpy as np
    import tensorflow as tf
    import agent_v2
    import moving_stats_backend
    import consistent_target_bn
    from canopy_feedback import install_bounds
    from canopy_qc_modes import symbolic_arrays, selected_mode, install
    from symbolic_transitions import initialize_symbolic_spec_single_step_only_latency
    from constants import SymbolicComponent_bound
    import constrained_reward as rewards
    moving_stats_backend.install(agent_v2)
    consistent_target_bn.install(agent_v2)
    tf.set_random_seed(3)
    agent = agent_v2.Agent(70, 1, 256, 256, gamma=.995, batch_size=512,
                           lr_a=.0001, lr_c=.001, tau=.001, mem_size=1,
                           noise_type=5, PER=False, CDQ=True, LOSS_TYPE='MSE',
                           use_original=False, use_snt_model_wo_ibp=False)
    consistent_target_bn.connect_statistics(agent, .001)
    install_bounds(agent)
    assert not any('ApplyAdam' in op.type for op in tf.get_default_graph().get_operations())
    policy = [v for v in tf.global_variables() if v.op.name.startswith('abstract_actor/')]
    assert policy
    checkpoint = tf.train.NewCheckpointReader(args.checkpoint)
    for var in policy:
        assert checkpoint.has_tensor(var.op.name), var.op.name
        assert list(var.shape) == list(checkpoint.get_tensor(var.op.name).shape)
    spec = initialize_symbolic_spec_single_step_only_latency(.25, 5, 12)
    cases = []
    frame = np.asarray([.7, 1., 0., .1, .02, .995, 1.], np.float32)
    for loss in (0., .1, .6):
        history = np.tile(frame, 10)
        history[-5] = loss
        cases.append(('synthetic_current_loss_%s' % loss, history))
    cold = np.zeros(70, np.float32)
    cold[-7:] = frame
    cases.append(('synthetic_cold_history', cold))
    if args.replay:
        with np.load(args.replay) as replay:
            real = replay['s0']
            assert real.ndim == 2 and real.shape[1] == 70
            indices = np.linspace(0, len(real) - 1, min(args.real_count, len(real))).astype(int)
            cases += [('real_replay_row_%d' % i, np.asarray(real[i], np.float32).copy())
                      for i in indices]

    def reset_reward_queues(value=0.):
        rewards.reward_q = deque(maxlen=3)
        for name in ('low_delay_delta_cwnd_lower_q', 'low_delay_delta_cwnd_upper_q',
                     'high_delay_delta_cwnd_lower_q', 'high_delay_delta_cwnd_upper_q'):
            setattr(rewards, name, deque([value] * 3, maxlen=3))

    def reward_probe(history, action, bounds, previous_action):
        reset_reward_queues()
        values = rewards.get_raw_and_constraint_reward(
            state=history[-7:], action=action, action_before_noise=action,
            raw_reward=.8, current_tcp_cwnd=80., constraints_id=12,
            reward_mode=rewards.convert_reward_mode('raw-sym'), threshold=.25,
            x1=5, x2=25, lambda_=.25, union_a_symbolic=bounds,
            past_states=[history[-7:].copy()] * 3,
            past_actions=[previous_action] * 3, past_tcp_cwnds=[80.] * 3, k=3)
        assert abs(float(values[3]) - (.75 * .8 + .25 * float(values[2]))) < 1e-7
        return dict(qc=float(values[2]), combined=float(values[3]))

    rows = []
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                         inter_op_parallelism_threads=1)) as sess:
        agent.assign_sess(sess)
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(var_list=policy).restore(sess, args.checkpoint)

        def policy_digest():
            digest = hashlib.sha256()
            for var, value in zip(policy, sess.run(policy)):
                digest.update(var.op.name.encode())
                digest.update(value.tobytes())
            return digest.hexdigest()

        before = policy_digest()
        max_ibp_error = 0.
        for label, history in cases:
            item = dict(case=label, observed_normalized_loss=float(history[-5]), modes={})
            for mode in ('artifact', 'initializer'):
                centers, radii = symbolic_arrays(history, spec, mode)
                action, lo, hi = sess.run([agent.qc_point, agent.qc_lower, agent.qc_upper],
                    {agent.s0: [history], agent.qc_center: centers, agent.qc_radius: radii})
                assert np.all(np.isfinite(lo)) and np.all(lo <= hi + 1e-6)
                original_lower, original_upper = [], []
                for center, radius in zip(centers, radii):
                    ll, uu = agent.get_symbolic_action(center, radius)
                    original_lower.append(float(np.asarray(ll).reshape(-1)[0]))
                    original_upper.append(float(np.asarray(uu).reshape(-1)[0]))
                error = float(max(np.max(np.abs(lo.reshape(-1) - original_lower)),
                                  np.max(np.abs(hi.reshape(-1) - original_upper))))
                max_ibp_error = max(max_ibp_error, error)
                bounds = SymbolicComponent_bound(12, float(lo[:5].min()), float(hi[:5].max()),
                                                 float(lo[5:].min()), float(hi[5:].max()))
                a = float(action[0, 0])
                item['modes'][mode] = dict(
                    action=a, loss_intervals=np.stack([centers[:, -5] - radii[:, -5],
                                                       centers[:, -5] + radii[:, -5]], 1).tolist(),
                    inverse_rtt_intervals=np.stack([centers[:, -2] - radii[:, -2],
                                                       centers[:, -2] + radii[:, -2]], 1).tolist(),
                    component_lower=lo.reshape(-1).tolist(), component_upper=hi.reshape(-1).tolist(),
                    union=[bounds.small_l, bounds.small_u, bounds.large_l, bounds.large_u],
                    snapshot_vs_original_ibp_max_error=error,
                    rewards={str(previous): reward_probe(history, a, bounds, previous)
                             for previous in (-.2, 0., .2)})
            assert item['modes']['artifact']['action'] == item['modes']['initializer']['action']
            rows.append(item)
        after = policy_digest()
        assert before == after
    for mode in ('artifact', 'initializer'):
        assert selected_mode(dict(canopy_qc_loss_center_mode=mode,
                                  canopy_shallow_loss_centers=(mode == 'initializer'))) == mode
    parity_pass = max_ibp_error <= 2e-5
    result = dict(status='PASS_FROZEN_MODE_COMPARISON' if parity_pass else 'FAIL_BOUND_PARITY', checkpoint=args.checkpoint,
                  policy_sha_before=before, policy_sha_after=after,
                  cases=rows, max_snapshot_vs_original_ibp_error=max_ibp_error,
                  bound_parity_tolerance=2e-5, bound_parity_pass=parity_pass,
                  semantics=dict(artifact='Unmodified symbolic updater loss centers; moving snapshot bounds retained.',
                                 initializer='Distinct initializer loss centers; separate ablation.',
                                 reward_context='Synthetic raw reward=.8,cwnd=80,previous action=-.2/0/.2,prehistory delta queues=0.',
                                 limitation='No live optimizer concurrency, TCP, learning or performance claim.'))
    output = json.dumps(result, indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(output)
    print(output)
    if not parity_pass:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
