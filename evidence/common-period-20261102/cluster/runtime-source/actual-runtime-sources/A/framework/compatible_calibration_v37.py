"""Same-schema V31 warm start and actor-update isolation; no control changes."""
import hashlib
import json
import pathlib
import numpy as np


def optimizer_name(name):
    return '/Adam' in name or name.startswith(('beta1_power', 'beta2_power'))


def validate_tensor(name, value, shape):
    if list(value.shape) != list(shape):
        raise ValueError('Checkpoint tensor shape mismatch: '+name)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite checkpoint tensor: '+name)


def validate_metadata(meta, cfg, markers):
    """Pure compatibility checks shared by TF initialization and local tests."""
    keys = ('joint_state_dim', 'gamma', 'decision_cost', 'time_head_nonlinearity',
            'feedback_context_mode', 'interval_rate_v28', 'time_statistics_mode',
            'reward_sampling_semantics', 'mtp_min_ms', 'mtp_max_ms',
            'decision_observation_clock', 'scheduled_observation_v30', 'target_bn_consistent')
    for key in keys:
        if meta.get(key) != cfg.get(key):
            raise ValueError('Incompatible checkpoint semantics: ' + key)
    required = {'time_statistics_schema': 1, 'interval_rate_schema': 1,
                'feedback_context_schema': 1, 'feedback_context_mode_schema': 1,
                'timer_boundary_schema': 1, 'duration_head_schema': 1,
                'timing_learning_enabled': True}
    for key, expected in required.items():
        if key not in markers or markers[key] != expected:
            raise ValueError('Invalid or missing checkpoint marker: ' + key)
    if cfg.get('joint_state_dim') != 149 or cfg.get('fixed_mtp_ms') is not None:
        raise ValueError('Compatible warm start requires healthy dynamic149-state V31')
    if (cfg.get('mtp_min_ms'), cfg.get('mtp_max_ms')) != (20, 160):
        raise ValueError('Preserve V31 action range and mapping')


def actor_gate(cfg, updates, retained, history):
    """Necessary factual consistency checks plus an explicit manual release.

    Stable self-consistency is never labeled true value accuracy. Thresholds
    are prospective configuration, not fitted to the stopped diagnostic run.
    """
    mode = cfg.get('actor_update_mode_v37', 'quality_gate')
    if mode not in ('quality_gate', 'legacy_count_gate'):
        raise ValueError('Explicit supported actor-update gate required')
    coverage = np.asarray(retained)
    if coverage.ndim != 2 or coverage.shape != (cfg['num_actors'], 5) or not np.isfinite(coverage).all():
        return False, 'missing_actor_period_coverage'
    if updates < cfg.get('timing_critic_calibration_updates', 5000):
        return False, 'minimum_critic_updates'
    if coverage.min() < cfg.get('timing_min_retained_per_period', 128):
        return False, 'minimum_actor_period_coverage'
    if mode == 'legacy_count_gate':
        return True, 'legacy_count_and_coverage'
    if not cfg.get('actor_release_after_calibration_v37', False):
        return False, 'explicit_release_not_configured'
    count = int(cfg.get('calibration_stable_diagnostics_v37', 3))
    if count < 2 or len(history) < count:
        return False, 'insufficient_stability_history'
    recent = history[-count:]
    for item in recent:
        fields = ('rmse_q1', 'rmse_q2', 'twin_q_gap_mean', 'actor_grad_norm',
                  'q1_mean', 'target_mean', 'period_rank_flip_fraction',
                  'window_single_boundary_fraction', 'mtp_single_boundary_fraction')
        if any(not np.isfinite(item.get(key, np.nan)) for key in fields):
            return False, 'nonfinite_value_diagnostic'
        if item.get('count', 0) < cfg.get('calibration_min_holdout_v37', 512):
            return False, 'insufficient_heldout_factual_rows'
        scale = max(1., abs(item['target_mean']))
        if max(item['rmse_q1'], item['rmse_q2'])/scale > cfg.get('calibration_max_normalized_rmse_v37', .25):
            return False, 'normalized_bellman_residual'
        if item['twin_q_gap_mean']/scale > cfg.get('calibration_max_normalized_twin_gap_v37', .1):
            return False, 'twin_critic_disagreement'
        if abs(item['q1_mean']-item['target_mean'])/scale > cfg.get('calibration_max_normalized_bias_v37', .1):
            return False, 'q_target_bias'
        if item['actor_grad_norm'] > cfg.get('calibration_max_actor_gradient_v37', 100.):
            return False, 'actor_gradient_limit'
        if item.get('window_single_boundary_fraction', 1.) >= .98 or item.get('mtp_single_boundary_fraction', 1.) >= .98:
            return False, 'policy_boundary_degeneracy'
        if item['period_rank_flip_fraction'] > cfg.get('calibration_max_rank_flip_v37', .25):
            return False, 'unstable_period_model_ranking'
        support = item.get('by_period', [])
        if len(support) != 5 or min(x['count'] for x in support) < cfg.get('calibration_min_holdout_per_period_v37', 16):
            return False, 'heldout_period_support'
    means = [x['target_mean'] for x in recent]
    if (max(means)-min(means))/max(1., abs(np.mean(means))) > cfg.get('calibration_max_target_drift_v37', .2):
        return False, 'target_drift'
    return True, 'explicit_release_and_necessary_consistency_checks'


def initialize(agent, cfg):
    import tensorflow as tf
    source = cfg['compatible_v31_checkpoint']
    reader = tf.train.NewCheckpointReader(source)
    meta = json.loads(pathlib.Path(source + '.counters.json').read_text())['cfg']
    markers = {n: reader.get_tensor(n).item() for n in reader.get_variable_to_shape_map()
               if n.endswith('_schema') or n == 'timing_learning_enabled'}
    validate_metadata(meta, cfg, markers)
    variables = {v.op.name: v for v in tf.global_variables()}
    excluded = [n for n in variables if optimizer_name(n) or n in ('global_step', 'epoch')]
    copied = [n for n in variables if n not in excluded]
    old_slots = agent.sess.run([variables[n] for n in excluded])
    assignments = []
    for n in copied:
        assert reader.has_tensor(n), ('missing full-network tensor', n)
        value = reader.get_tensor(n)
        validate_tensor(n, value, variables[n].shape.as_list())
        assignments.append(tf.assign(variables[n], value))
    agent.sess.run(assignments)
    h = hashlib.sha256()
    for n, value in zip(copied, agent.sess.run([variables[n] for n in copied])):
        np.testing.assert_array_equal(value, reader.get_tensor(n), err_msg=n)
        h.update(n.encode()); h.update(value.tobytes())
    for before, after in zip(old_slots, agent.sess.run([variables[n] for n in excluded])):
        np.testing.assert_array_equal(before, after)
    assert bool(agent.sess.run(agent.timing_learning_enabled))
    return dict(source_checkpoint=source, copied_names=copied, excluded_names=excluded,
                model_target_bn_schema_exact=True, copied_sha256=h.hexdigest(),
                optimizer_replay_counters_fresh=True, time_forward_active=True,
                complete_resume=False, scratch_training=False,
                optimizer_math=('V34 independent critic Adam, fresh clocks and slots' if cfg.get('independent_critic_adam_v34',False) else 'V31 legacy shared-Adam semantics retained in both arms'))


def policy_snapshot(agent):
    import tensorflow as tf
    variables = [v for v in tf.global_variables() if
                 v.op.name.startswith('abstract_actor/') and not optimizer_name(v.op.name)]
    return variables, agent.sess.run(variables)


def assert_policy_frozen(agent, snapshot):
    variables, values = snapshot
    for v, before, after in zip(variables, values, agent.sess.run(variables)):
        np.testing.assert_array_equal(before, after, err_msg='Frozen actor/BN drift: ' + v.op.name)


class Holdout:
    """Predetermined real rows excluded from both FIFO sampling and period bank.

    Counts and physical credits still include them. Diagnostics are necessary
    Bellman checks, not proof of causal period ranking or a sufficient unlock.
    """
    def __init__(self, cfg):
        self.rows = []
        self.maximum = int(cfg.get('calibration_holdout_rows', 1024))
        self.residual = None
        self.history = []
        self.previous_best = None

    def store(self, row, actor_id, birth):
        item = ([np.asarray(x).copy() for x in row], int(actor_id), int(birth))
        if len(self.rows) < self.maximum:
            self.rows.append(item)

    def tensors(self, agent):
        import tensorflow as tf
        if self.residual is None:
            y = tf.minimum(agent.y, agent.y2)
            self.residual = [agent.critic_out, agent.critic_out2, y,
                             tf.global_norm(tf.gradients(agent.a_loss, agent.symbolic_actor.train_var()))]

    def diagnose(self, agent, iteration):
        self.tensors(agent)
        if not self.rows:
            return dict(iteration=iteration, count=0, actor_unlock=False)
        rows = self.rows
        batch = [np.stack([item[0][i] for item in rows]) for i in range(7)]
        fd = {agent.s0: batch[0], agent.action: batch[1], agent.reward: batch[2],
              agent.s1: batch[3], agent.terminal: batch[4], agent.transition_discount: batch[6],
              agent.is_training: False, agent.is_concrete: True, agent.s0_delta: np.zeros_like(batch[0])}
        q1, q2, y, grad = agent.sess.run(self.residual, fd)
        policy = agent.sess.run(agent._moving_action, fd)
        periods = np.array([20, 40, 80, 120, 160])
        centers = 2*np.log(periods/20.)/np.log(8.)-1
        bins = np.abs(batch[1][:, 1:2]-centers).argmin(1)
        grid = []
        for center in centers:
            actual = batch[1].copy(); actual[:, 1] = center
            query = dict(fd); query[agent.action] = actual
            first, second = agent.sess.run(self.residual[:2], query)
            grid.append(np.minimum(first, second).reshape(-1))
        grid = np.stack(grid, axis=1)
        best = grid.argmax(axis=1)
        comparable = self.previous_best is not None and self.previous_best.shape == best.shape
        flips = float(np.mean(best != self.previous_best)) if comparable else float('nan')
        self.previous_best = best.copy()
        by_period = []
        for i, ms in enumerate(periods):
            ix = bins == i
            by_period.append(dict(period_ms=int(ms), count=int(ix.sum()),
                rmse_q1=float(np.sqrt(np.mean((q1[ix]-y[ix])**2))) if ix.any() else None,
                bias_q1=float(np.mean(q1[ix]-y[ix])) if ix.any() else None,
                q1_mean=float(np.mean(q1[ix])) if ix.any() else None,
                target_mean=float(np.mean(y[ix])) if ix.any() else None))
        result = dict(iteration=iteration, count=len(rows), by_period=by_period,
                      rmse_q1=float(np.sqrt(np.mean((q1-y)**2))),
                      rmse_q2=float(np.sqrt(np.mean((q2-y)**2))),
                      twin_q_gap_mean=float(np.mean(np.abs(q1-q2))), actor_grad_norm=float(grad),
                      q1_mean=float(np.mean(q1)), target_mean=float(np.mean(y)),
                      period_rank_flip_fraction=flips,
                      best_period_counts={str(ms):int(np.sum(best == i)) for i, ms in enumerate(periods)},
                      window_single_boundary_fraction=float(max(np.mean(policy[:,0]<=-.99), np.mean(policy[:,0]>=.99))),
                      mtp_single_boundary_fraction=float(max(np.mean(policy[:,1]<=-.99), np.mean(policy[:,1]>=.99))),
                      actor_unlock=False, reason='Diagnostic alone cannot authorize release or establish causal period ordering; consult explicit actor-gate-history',
                      no_holdout_optimization=True, target_policy_smoothing='Unchanged TF random smoothing; diagnostic repeated on same held-out rows')
        self.history.append(result)
        return result
