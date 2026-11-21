"""Explicit shallow loss-center modes; shared framework source is untouched.

artifact: execute the vendored updater without changing its loss centers.
initializer: use the distinct loss centers specified by the initializer.
Both use the existing snapshot/moving-statistics bound graph. Neither mode
claims that all other engineering adaptations equal the upstream artifact.
"""
import time
import numpy as np


def selected_mode(cfg):
    mode = cfg.get('canopy_qc_loss_center_mode')
    if mode not in ('artifact', 'initializer'):
        raise ValueError('Require explicit canopy_qc_loss_center_mode=artifact|initializer')
    expected = mode == 'initializer'
    if cfg.get('canopy_shallow_loss_centers') is not expected:
        raise ValueError('canopy_shallow_loss_centers must be %r for %s' % (expected, mode))
    for key, value in (('canopy_constraints_id', 12), ('canopy_partitions', 5),
                       ('canopy_history_k', 3), ('canopy_lambda', .25)):
        if key in cfg and cfg[key] != value:
            raise ValueError('This explicit shallow overlay requires %s=%r' % (key, value))
    return mode


def symbolic_arrays(history, spec, mode):
    from symbolic_transitions import update_symbolic_s_single_step
    symbolic = update_symbolic_s_single_step(history, spec, 12)
    if mode == 'initializer':
        for i in range(5):
            symbolic.small_c[i][-5] = spec.small_c[i][-5]
            symbolic.large_c[i][-5] = 0.
            symbolic.large_delta[i][-5] = 0.
    elif mode != 'artifact':
        raise ValueError(mode)
    centers = np.asarray(symbolic.small_c + symbolic.large_c, np.float32)
    radii = np.asarray(symbolic.small_delta + symbolic.large_delta, np.float32)
    if mode == 'artifact':
        assert np.all(centers[:, -5] == np.float32(history[-5]))
    else:
        assert np.allclose(centers[5:, -5], 0.)
        assert np.allclose(centers[:5, -5] - radii[:5, -5], [.25, .4, .55, .7, .85])
        assert np.allclose(centers[:5, -5] + radii[:5, -5], [.4, .55, .7, .85, 1.])
    return centers, radii


def install():
    import canopy_feedback
    base = canopy_feedback.Feedback
    if getattr(base, '_explicit_loss_center_modes', False):
        return

    class ExplicitModeFeedback(base):
        _explicit_loss_center_modes = True

        def __init__(self, agent, cfg, directory):
            self.loss_center_mode = selected_mode(cfg)
            super(ExplicitModeFeedback, self).__init__(agent, cfg, directory)

        def observe_action(self, history, no_noise=None):
            from constants import SymbolicComponent_bound
            started = time.perf_counter()
            centers, radii = symbolic_arrays(history, self.spec, self.loss_center_mode)
            point, lo, hi = self.agent.sess.run(
                [self.agent.qc_point, self.agent.qc_lower, self.agent.qc_upper],
                {self.agent.qc_center: centers, self.agent.qc_radius: radii,
                 self.agent.s0: [history]})
            actual = float(point[0, 0])
            if no_noise is not None:
                assert abs(actual - no_noise) < 2e-5
            assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
            assert np.all(lo <= hi + 1e-6), 'invalid QC interval'
            bounds = SymbolicComponent_bound(12, float(lo[:5].min()),
                                             float(hi[:5].max()), float(lo[5:].min()),
                                             float(hi[5:].max()))
            elapsed = (time.perf_counter() - started) * 1000
            self.pending.append((history.copy(), actual, bounds, elapsed))
            return actual

    canopy_feedback.Feedback = ExplicitModeFeedback
