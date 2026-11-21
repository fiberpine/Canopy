"""Single-action Canopy training feedback; no runtime policy replacement.

The upstream property/reward routines are vendored unchanged. Interval bounds
use the same moving-statistics function as the deployed concrete actor.
"""
import csv
from collections import deque
import numpy as np


def install_bounds(agent):
    import tensorflow as tf
    variables = {v.op.name: v for v in tf.global_variables()}
    prefix = 'abstract_actor/abstract_actor/'
    # Ref-variable buffers can change during concurrent gRPC optimizer runs.
    # Snapshot copies each required tensor once, for both point and bounds.
    # This is not a transactional snapshot of a complete optimizer step.
    variables = {name: tf.raw_ops.Snapshot(input=value, name='qc_copy_'+name.replace('/','_'))
                 for name,value in variables.items() if name.startswith(prefix)}
    agent.qc_center = tf.placeholder(tf.float32, [None, 70], name='qc_center')
    agent.qc_radius = tf.placeholder(tf.float32, [None, 70], name='qc_radius')
    low, high = agent.qc_center-agent.qc_radius, agent.qc_center+agent.qc_radius
    for layer, bn in [('fc1', 'bn1'), ('fc2', 'bn2'), ('fc1_1', None)]:
        weight, bias = variables[prefix+layer+'/w'], variables[prefix+layer+'/b']
        center, radius = (low+high)*.5, (high-low)*.5
        center = tf.matmul(center, weight)+bias
        radius = tf.matmul(radius, tf.abs(weight))
        low, high = center-radius, center+radius
        if bn:
            mean = variables[prefix+bn+'/moving_mean']
            variance = variables[prefix+bn+'/moving_variance']
            beta = variables[prefix+bn+'/beta']
            # Sonnet actor uses no trainable scale; epsilon and leaky slope
            # match moving_stats_backend.forward and upstream AbstractActor.
            inverse = tf.rsqrt(variance+.001)
            center = (center-mean)*inverse+beta
            radius = radius*inverse
            low = tf.nn.leaky_relu(center-radius)
            high = tf.nn.leaky_relu(center+radius)
        else:
            low, high = tf.tanh(low), tf.tanh(high)
    agent.qc_lower, agent.qc_upper = low, high
    point = agent.s0
    for layer,bn in [('fc1','bn1'),('fc2','bn2'),('fc1_1',None)]:
        point = tf.matmul(point,variables[prefix+layer+'/w'])+variables[prefix+layer+'/b']
        if bn:
            point = tf.nn.leaky_relu(tf.nn.batch_normalization(
                point,variables[prefix+bn+'/moving_mean'],variables[prefix+bn+'/moving_variance'],
                variables[prefix+bn+'/beta'],None,.001))
        else:point=tf.tanh(point)
    agent.qc_point=point


class Feedback:
    def __init__(self, agent, cfg, directory):
        from symbolic_transitions import initialize_symbolic_spec_single_step_only_latency
        self.agent, self.cfg = agent, cfg
        self.spec = initialize_symbolic_spec_single_step_only_latency(.25, 5, 12)
        self.pending = deque()
        self.states, self.actions, self.windows = [], [], []
        self.file = (directory/'canopy-rewards.csv').open('w', buffering=65536)
        self.writer = csv.writer(self.file)
        self.writer.writerow(['transition','raw_reward','qc_reward','combined_reward',
                              'action','no_noise','tcp_cwnd','qc_ms','state_match'])
        self.count = 0
        self.qc_ms = 0.

    def observe_action(self, history, no_noise=None):
        import time
        from symbolic_transitions import update_symbolic_s_single_step
        from constants import SymbolicComponent_bound
        started = time.perf_counter()
        symbolic = update_symbolic_s_single_step(history, self.spec, 12)
        # Upstream initializes these loss centers, but its generic updater
        # copies only inverse RTT. Apply the intended shallow preconditions.
        # The vendored implementation remains unchanged for parity auditing.
        assert self.cfg.get('canopy_shallow_loss_centers') is True
        for i in range(5):
            symbolic.small_c[i][-5] = self.spec.small_c[i][-5]
            symbolic.large_c[i][-5] = 0.
            symbolic.large_delta[i][-5] = 0.
        centers = np.asarray(symbolic.small_c+symbolic.large_c, np.float32)
        radii = np.asarray(symbolic.small_delta+symbolic.large_delta, np.float32)
        assert np.allclose(centers[5:, -5], 0.)
        assert np.allclose(radii[5:, -5], 0.)
        assert np.allclose(centers[:5, -5]-radii[:5, -5], [.25,.4,.55,.7,.85])
        assert np.allclose(centers[:5, -5]+radii[:5, -5], [.4,.55,.7,.85,1.])
        point, lo, hi = self.agent.sess.run([self.agent.qc_point,self.agent.qc_lower, self.agent.qc_upper],
                                    {self.agent.qc_center: centers,
                                     self.agent.qc_radius: radii,self.agent.s0:[history]})
        actual=float(point[0,0])
        if no_noise is not None:assert abs(actual-no_noise)<2e-5
        no_noise=actual
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert np.all(lo <= hi+1e-6), ('invalid QC interval',float(np.max(lo-hi)))
        bounds = SymbolicComponent_bound(12, float(lo[:5].min()), float(hi[:5].max()),
                                        float(lo[5:].min()), float(hi[5:].max()))
        elapsed = (time.perf_counter()-started)*1000
        self.pending.append((history.copy(), no_noise, bounds, elapsed))
        return actual

    def reward(self, history, action, no_noise, raw_reward, next_history, raw):
        from constrained_reward import get_raw_and_constraint_reward, convert_reward_mode
        h, expected_action, bounds, elapsed = self.pending.popleft()
        assert np.array_equal(h, history), 'QC/action state alignment failure'
        assert abs(expected_action-no_noise) < 1e-6
        values = get_raw_and_constraint_reward(
            state=history[-7:], action=action, action_before_noise=no_noise,
            raw_reward=raw_reward, current_tcp_cwnd=float(raw[5]), constraints_id=12,
            reward_mode=convert_reward_mode('raw-sym'), threshold=.25, x1=5, x2=25,
            lambda_=.25, union_a_symbolic=bounds, past_states=self.states,
            past_actions=self.actions, past_tcp_cwnds=self.windows, k=3)
        combined = float(values[3])
        assert np.isfinite(combined)
        assert abs(combined-(.75*raw_reward+.25*values[2])) < 1e-7
        self.count += 1
        self.writer.writerow([self.count, raw_reward, values[2], combined, action,
                              no_noise, raw[5], elapsed, True])
        self.states.append(history[-7:].copy())
        self.actions.append(action)
        self.windows.append(float(raw[5]))
        self.states, self.actions, self.windows = self.states[-10:], self.actions[-10:], self.windows[-10:]
        if self.count % 100 == 0:
            self.file.flush()
        return combined

    def close(self):
        self.file.close()
