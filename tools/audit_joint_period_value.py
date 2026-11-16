"""Read-only decomposition of a saved joint model on existing real probes.

Python3.6/TF1 compatible. Does not open TCP, train, or change model files.
Stratified rows are descriptive mixed-behaviour observations, not paired
counterfactuals or an independent Monte Carlo value-validation set.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '1')


def summary(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return dict(mean=float(values.mean()), minimum=float(values.min()),
                p10_p50_p90=np.percentile(values, [10, 50, 90]).tolist(),
                maximum=float(values.max()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytical_cost_scales(cost, gamma, base):
    """Separate one changed decision from changing the entire future policy."""
    d0 = gamma
    fixed_short_value = 1. / (1. - gamma) - cost / (1. - d0)
    result = []
    for ms in (20, 40, 80, 120, 160):
        d = gamma ** (ms / 1000. / base)
        interval = (1. - d) / (1. - gamma)
        one_changed = interval - cost + d * fixed_short_value
        fixed_period = 1. / (1. - gamma) - cost / (1. - d)
        direct = cost * (d0 - d) / (1. - d0)
        assert math.isclose(one_changed - fixed_short_value, direct, abs_tol=1e-10)
        result.append(dict(period_ms=ms, discount=d,
                           one_changed_then20_advantage=direct,
                           entire_fixed_period_value=fixed_period,
                           ideal_calls_per_second=1000. / ms))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--baseline-checkpoint', required=True)
    ap.add_argument('--transitions', required=True)
    ap.add_argument('--frozen-probes', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.package)
    import numpy as np
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    from frameworks.joint_healthy.model import Model
    from frameworks.joint_healthy.core import period_action

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cfg['baseline_checkpoint'] = args.baseline_checkpoint
    with np.load(args.transitions) as data:
        rows = {k: data[k].copy() for k in data.files}
    with np.load(args.frozen_probes) as data:
        frozen = data['states'].copy()
    columns = ('s0', 'actions', 'rewards', 's1', 'terminal', 'duration', 'discount')
    batch = [rows[k] for k in columns]
    cost, gamma = float(cfg['decision_cost']), float(cfg['gamma'])
    duration = rows['duration'][:, 0].astype(np.float64)
    discount = np.exp(np.log(gamma) * duration / cfg['base_period_s'])
    integral_weight = -np.expm1(np.log(gamma) * duration / cfg['base_period_s']) / (1 - gamma)
    # Last seven prefix features describe the actually observed next interval.
    utility = ((rows['s1'][:, 63].astype(np.float64) - rows['s1'][:, 65]) * rows['s1'][:, 69])
    reconstructed_reward = utility * integral_weight - cost
    groups = np.rint(20 * 8 ** ((rows['actions'][:, 1].astype(np.float64) + 1) / 2)).astype(int)
    result = dict(scope='Read-only final checkpoint; stratified mixed behaviour, no causal/MC claim',
                  optimizer_updates=0, tcp_runs=0,
                  inputs={Path(p).name: sha(p) for p in (args.config, args.transitions, args.frozen_probes)},
                  reward_reconstruction_max_error=float(np.max(np.abs(reconstructed_reward - rows['rewards'][:, 0]))),
                  discount_reconstruction_max_error=float(np.max(np.abs(discount - rows['discount'][:, 0]))),
                  analytical_constant_utility_examples=analytical_cost_scales(cost, gamma, cfg['base_period_s']))
    with tf.Graph().as_default():
        tf.set_random_seed(116)
        model = Model(cfg, True)
        # Explicit deterministic target, without a new draw of window smoothing.
        target_deterministic = tf.minimum(
            model._critic(model.s1, model.target_action, 'target_critic'),
            model._critic(model.s1, model.target_action, 'target_critic2'))
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer())
            model.restore(sess, args.checkpoint)
            before = sess.run(tf.global_variables())
            feed = model.feed(batch)
            q1, q2, future = sess.run([model.q1, model.q2, target_deterministic], feed)
            # Physical reconnect markers are not necessarily Bellman terminals.
            effective_terminal = feed[model.terminal][:, 0]
            continuation = rows['discount'][:, 0] * (1 - effective_terminal) * future[:, 0]
            label = rows['rewards'][:, 0] + continuation
            result['boundary_semantics'] = dict(
                mode=cfg.get('time_limit_handling', 'physical_end_is_terminal_legacy'),
                physical_end_rows=int(np.sum(rows['terminal'])),
                effective_bellman_terminal_rows=int(np.sum(effective_terminal)))
            result['descriptive_by_executed_period'] = []
            for ms in (20, 40, 80, 120, 160):
                selected = groups == ms
                assert selected.sum() == 192
                result['descriptive_by_executed_period'].append(dict(
                    period_ms=ms, rows=int(selected.sum()),
                    actor_count=int(len(np.unique(rows['actor_ids'][selected]))),
                    actual_ms=summary(duration[selected] * 1000),
                    observed_utility_density=summary(utility[selected]),
                    actual_reward=summary(rows['rewards'][selected]),
                    deterministic_target_continuation=summary(continuation[selected]),
                    learned_q1=summary(q1[selected]), learned_q2=summary(q2[selected]),
                    deterministic_td_rmse_q1=float(np.sqrt(np.mean((q1[selected, 0] - label[selected]) ** 2))),
                    deterministic_td_rmse_q2=float(np.sqrt(np.mean((q2[selected, 0] - label[selected]) ** 2))),
                    executed_window=summary(rows['actions'][selected, 0]),
                    window_teacher_difference=summary(rows['actions'][selected, 0] - rows['teacher_actions'][selected, 0])))
            action = sess.run(model.action, {model.s: frozen})
            period = 20 * 8 ** ((action[:, 1].astype(np.float64) + 1) / 2)
            result['frozen_real_input_actions'] = dict(
                rows=len(frozen), window=summary(action[:, 0]), period_ms=summary(period),
                window_le_minus099=int(np.sum(action[:, 0] <= -.99)),
                window_ge_plus099=int(np.sum(action[:, 0] >= .99)),
                time_le_minus099=int(np.sum(action[:, 1] <= -.99)),
                time_exact_minus1=int(np.sum(action[:, 1] == -1)),
                period_le21_ms=int(np.sum(period <= 21)),
                median_time_softsign_derivative=float(np.median((1 - np.abs(action[:, 1])) ** 2)))
            predicted = {}
            for ms in (20, 40, 80, 120, 160):
                query = action.copy()
                query[:, 1] = period_action(ms)
                predicted[ms] = sess.run([model.q1, model.q2], {model.s: frozen, model.executed: query})
            result['same_input_window_critic_queries_not_truth'] = []
            for ms, values in sorted(predicted.items()):
                delta1 = values[0] - predicted[20][0]
                delta2 = values[1] - predicted[20][1]
                result['same_input_window_critic_queries_not_truth'].append(dict(
                    period_ms=ms, q1_delta_from20=summary(delta1), q2_delta_from20=summary(delta2),
                    twin_delta_sign_disagreement_fraction=float(np.mean((delta1 > 0) != (delta2 > 0)))))
            after = sess.run(tf.global_variables())
            assert all(np.array_equal(a, b) for a, b in zip(before, after))
            result['all_model_variables_unchanged'] = True
    result['limitations'] = [
        'Balanced192 rows/period intentionally changes sampling weights; not the retained replay population.',
        'Executed-window, state and behaviour policies differ across period strata; means do not estimate intervention effects.',
        'Deterministic target omits training-time target-window smoothing; TD error is not independent MC calibration.',
        'Actual duration is observed, not a differentiable known function of the planned time action.',
        'Analytical examples assume constant utility and immediate return to the20ms policy; not TCP predictions.',
        'Nonzero gradients and low TD errors cannot establish correct counterfactual duration ordering.']
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(rows=len(duration), output=args.output,
                         reward_error=result['reward_reconstruction_max_error'],
                         frozen=result['frozen_real_input_actions'])))


if __name__ == '__main__':
    main()
