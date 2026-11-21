"""Opt-in one-action timing experiment, followed by the unchanged frozen policy.

Consumes existing decision boundaries only. Never samples TCP or wakes the policy.
All file writes occur after control ends; these trials are not CPU benchmarks.
"""
import json
import numpy as np


class Probe:
    def __init__(self, cfg, run):
        self.cfg, self.run = cfg, run
        self.options = cfg['factual_mtp_probe_v37']
        if cfg.get('training_mode') or not cfg.get('no_training'):
            raise ValueError('One-shot factual probes require explicitly frozen evaluation')
        if cfg.get('fixed_mtp_ms') is not None or cfg.get('diagnostic_fixed_clock_ms') is not None:
            raise ValueError('One-shot probe cannot be combined with fixed-policy overrides')
        self.period = int(self.options['period_ms'])
        if self.period not in [20, 40, 80, 120, 160]:
            raise ValueError('Unsupported randomized period')
        self.window_delta = float(self.options.get('window_action_delta_v38', 0.))
        if not np.isfinite(self.window_delta) or abs(self.window_delta) > .100000001:
            raise ValueError('Window probe must be finite and bounded by0.1 action units')
        self.trigger = float(self.options.get('trigger_s', 8))
        self.horizons = list(self.options.get('horizons_s', [1, 4, 8, 20]))
        if self.trigger < 0 or not self.horizons or not all(0 < x <= 20 for x in self.horizons):
            raise ValueError('Invalid bounded factual horizons')
        self.event = None
        self.endpoints = {}
        self.rows = []

    def choose(self, state, policy_action, physical_s, decision):
        action = np.asarray(policy_action, np.float32).copy()
        if self.event is None and physical_s >= self.trigger:
            from joint_mtp import mtp_action
            action[1] = mtp_action(self.period, self.cfg)
            if self.window_delta:
                action[0] = np.clip(float(action[0])+self.window_delta, -1., 1.)
            self.event = dict(decision=int(decision), start_physical_s=float(physical_s),
                state=np.asarray(state).tolist(), policy_action=np.asarray(policy_action).tolist(),
                executed_action=action.tolist(), period_ms=self.period,
                requested_window_delta=self.window_delta,
                executed_window_delta=float(action[0])-float(policy_action[0]),
                randomization=self.options.get('randomization'),
                scope=('One real joint action perturbation; original multiply-window mapping and healthy continuation' if self.window_delta else 'One real MTP action; original window output; frozen healthy continuation'))
            return action, True
        if self.event is not None:
            elapsed = physical_s-self.event['start_physical_s']
            for horizon in self.horizons:
                if str(horizon) not in self.endpoints and elapsed >= horizon:
                    self.endpoints[str(horizon)] = dict(decision=int(decision),
                        physical_s=float(physical_s), actual_horizon_s=float(elapsed),
                        state=np.asarray(state).tolist(), action=action.tolist())
        return action, False

    def transition(self, state, action, reward, next_state, duration, discount):
        self.rows.append((state.copy(), action.copy(), float(reward), next_state.copy(),
                          float(duration), float(discount)))

    def finish(self):
        if self.event is None or len(self.endpoints) != len(set(map(str, self.horizons))):
            raise RuntimeError('Incomplete factual one-shot continuation')
        start = self.event['decision']-1
        if not np.array_equal(self.rows[start][1], np.asarray(self.event['executed_action'], np.float32)):
            raise AssertionError('Recorded intervention differs from executed transition')
        for endpoint in self.endpoints.values():
            end = endpoint['decision']-1
            product, total, duration = 1., 0., 0.
            for row in self.rows[start:end]:
                total += product*row[2]; product *= row[5]; duration += row[4]
            endpoint.update(observed_return=total, tail_discount=product, duration_s=duration)
            if abs(duration-endpoint['actual_horizon_s']) > 1e-6:
                raise AssertionError('Factual return time partition mismatch')
        self.event.update(endpoints=self.endpoints, total_transitions=len(self.rows),
            additional_tcp_reads=0, policy_optimizer_updates=0,
            limitations='Factual randomized action return; one noisy outcome, not exact counterfactual Q')
        np.savez_compressed(str(self.run/'factual-transitions.npz'),
            s0=np.asarray([x[0] for x in self.rows], np.float32),
            a=np.asarray([x[1] for x in self.rows], np.float32),
            reward=np.asarray([x[2] for x in self.rows], np.float32)[:, None],
            s1=np.asarray([x[3] for x in self.rows], np.float32),
            duration=np.asarray([x[4] for x in self.rows], np.float64)[:, None],
            discount=np.asarray([x[5] for x in self.rows], np.float64)[:, None])
        (self.run/'factual-probe.json').write_text(json.dumps(self.event, indent=2))
