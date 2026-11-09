"""Common-period SMDP contract. No socket I/O or TensorFlow import.

Replay keeps original rewards. Calibration never enters replay or issues credit.
This module deliberately does not enable the historical distributed entrypoint.
"""
import hashlib
import json
import math
from collections import deque
import numpy as np

PERIODS = (20, 40, 80, 120, 160)
SCHEMA = 'common-period-v1'
REWARD = 'throughput-times-delay-minus-loss-v1'


def training_periods(periods=PERIODS):
    periods = tuple(periods)
    if periods not in ((20,), PERIODS):
        raise ValueError('Expected S20 or the original A period set')
    return periods


def encoded_period(ms):
    if ms not in PERIODS:
        raise ValueError('Unknown candidate period')
    return 2 * math.log(ms / 20.) / math.log(8.) - 1


def discount(dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive actual duration required')
    return .995 ** (dt / .02)


def utility(throughput, loss_rate, denominator, avg_rtt, srtt, min_rtt, samples):
    vals = (throughput, loss_rate, denominator, avg_rtt, srtt, min_rtt, samples)
    if not all(math.isfinite(x) for x in vals) or min(throughput, loss_rate, samples) < 0:
        raise ValueError('Invalid boundary telemetry')
    rtt = avg_rtt if samples > 0 else srtt
    if denominator <= 0 or rtt <= 0 or min_rtt <= 0:
        raise ValueError('Positive normalization and RTT required')
    factor = min(1., 1.25 * min_rtt / rtt)
    return throughput / denominator * factor - 5 * loss_rate / denominator


def center(reward, gamma, kind, K):
    if kind not in ('terminated', 'truncated', 'continuing'):
        raise ValueError('Censored transitions cannot be sampled')
    return reward - K if kind == 'terminated' else reward - K * (1 - gamma)


def ending(kind, connection, next_connection, has_next):
    if kind not in ('continuing', 'terminated', 'truncated', 'censored'):
        raise ValueError('Unknown ending')
    if kind == 'censored':
        return False, False
    if kind == 'terminated':
        return True, False
    if not has_next:
        return False, False  # truncated without an observed successor is censored
    if connection != next_connection:
        raise ValueError('Cross-connection bootstrap rejected')
    return True, True


def calibration(blocks, active_periods=PERIODS):
    """Blocks contain interval rewards, actual durations, period and connection IDs."""
    mass = total_reward = 0.
    active_periods = training_periods(active_periods)
    gammas = {t: [] for t in active_periods}
    for block in blocks:
        if block['connection'] != block['end_connection']:
            raise ValueError('Calibration block crosses a connection')
        if sum(block['duration']) < 40.:
            raise ValueError('Calibration block shorter than 40 seconds')
        if len(block['duration']) != len(block['reward']):
            raise ValueError('Mismatched calibration arrays')
        if block['period_ms'] not in gammas:
            raise ValueError('Inactive calibration period')
        for dt, reward in zip(block['duration'], block['reward']):
            g = discount(dt)
            # R / discounted mass estimates the interval utility density;
            # weight that density by actual exposure, as specified by 4.1.
            total_reward += reward / ((1 - g) / .005) * dt
            mass += dt
            gammas[block['period_ms']].append(g)
    if mass <= 0 or not all(gammas.values()):
        raise ValueError('All active periods need calibration data')
    c = total_reward / mass
    n0 = {str(t): 1 / (1 - discount(t / 1000.)) for t in PERIODS}
    n0.update({str(t): 1 / (1 - float(np.mean(gs))) for t, gs in gammas.items()})
    return dict(c=c, K=c / .005, n0=n0)


def bias_correction(labels, predictions, suffix_discounts, weights=None):
    g, q, d = [np.asarray(x, dtype=np.float64) for x in (labels, predictions, suffix_discounts)]
    w = np.ones_like(g) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (g.shape == q.shape == d.shape == w.shape) or g.size == 0:
        raise ValueError('Calibration shape mismatch')
    if not all(np.isfinite(x).all() for x in (g, q, d, w)) or np.any(w < 0) or np.any((d < 0) | (d >= 1)) or w.sum() <= 0:
        raise ValueError('Invalid calibration weights or discounts')
    return float(np.sum(w * (g - q)) / np.sum(w * (1 - d)))


def return_label(rows, start, tail_value, K, horizon=20.):
    """Finite same-connection suffix plus explicitly supplied frozen tail."""
    total, product, elapsed = 0., 1., 0.
    connection = rows[start]['connection']
    period = rows[start]['period_ms']
    for row in rows[start:]:
        if row['connection'] != connection or row['period_ms'] != period:
            raise ValueError('Return label crosses connection or period')
        if row['kind'] not in ('continuing', 'truncated'):
            raise ValueError('No legal fixed-policy suffix')
        total += product * center(row['reward'], row['discount'], row['kind'], K)
        product *= row['discount']
        elapsed += row['duration']
        if elapsed >= horizon:
            return dict(label=total + product * tail_value, tail=product * tail_value,
                        suffix_discount=product, duration=elapsed)
    raise ValueError('Insufficient observed suffix')


class Replay:
    def __init__(self, reward_checksum, capacity=150000, active_periods=PERIODS):
        self.active_periods = training_periods(active_periods)
        self.rows = []
        self.capacity = capacity
        self.ptr = 0
        self.slots = [[] for _ in range(24)]
        self.slot_position = {}
        self.reward_checksum = reward_checksum
        self.inserted = 0
        self.seconds = 0.
        self.by_actor = np.zeros(24, dtype=np.int64)
        self.coverage = np.zeros((24, 5), dtype=np.int64)
        self.retained_coverage = np.zeros((24, 5), dtype=np.int64)
        self.sample_calls = 0
        self.sampled_by_actor = np.zeros(24, dtype=np.int64)
        self.warmup_seconds = None

    def append(self, row):
        if row.get('calibration'):
            raise ValueError('Calibration cannot enter training replay')
        if row.get('apply') != 1:
            raise ValueError('Every training action must apply the window multiplier')
        if row['reward_checksum'] != self.reward_checksum or row['reward_version'] != REWARD:
            raise ValueError('Reward schema mismatch')
        if not math.isfinite(float(row['reward'])):
            raise ValueError('Non-finite raw reward cannot issue experience credit')
        keep, bootstrap = ending(row['kind'], row['connection'], row.get('next_connection'), row.get('s1') is not None)
        if not keep:
            return False
        g = discount(row['duration'])
        if abs(g - row['discount']) > 1e-7:
            raise ValueError('Discount must use actual duration')
        if row['period_ms'] not in self.active_periods:
            raise ValueError('Inactive replay period')
        p = PERIODS.index(row['period_ms'])
        if abs(row['action'][1] - encoded_period(row['period_ms'])) > 1e-6:
            raise ValueError('Executed period and action mismatch')
        if abs(row['action'][0]) > 1 or not np.isfinite(row['action']).all():
            raise ValueError('Invalid window action')
        if row['behavior'] not in ('random_segment', 'neutral_segment', 'policy'):
            raise ValueError('Missing behavior attribution')
        for key in ('s0', 's1'):
            if key == 's1' and not bootstrap:
                continue
            if np.asarray(row[key]).shape != (149,) or not np.isfinite(row[key]).all():
                raise ValueError('149 real finite state values required')
        actor = int(row['actor'])
        if not 0 <= actor < 24 or int(row['snapshot']) < 0:
            raise ValueError('Actor or policy snapshot missing')
        saved = dict(row, bootstrap=bootstrap)
        for key in ('s0', 's1', 'action'):
            if saved.get(key) is not None:
                saved[key] = np.asarray(saved[key], np.float32).copy()
        # Terminal successor is a non-bootstrapped placeholder, never a reset state.
        if not bootstrap and saved.get('s1') is None:
            saved['s1'] = np.zeros(149, np.float32)
        index = self.ptr
        if len(self.rows) == self.capacity:
            old_actor = self.rows[index]['actor']
            self.retained_coverage[old_actor, PERIODS.index(self.rows[index]['period_ms'])] -= 1
            old_position = self.slot_position[index]
            last = self.slots[old_actor].pop()
            if last != index:
                self.slots[old_actor][old_position] = last
                self.slot_position[last] = old_position
            self.rows[index] = saved
        else:
            self.rows.append(saved)
        self.slot_position[index] = len(self.slots[actor])
        self.slots[actor].append(index)
        self.ptr = (index + 1) % self.capacity
        self.inserted += 1
        self.seconds += row['duration']
        self.by_actor[actor] += 1
        self.coverage[actor, p] += 1
        self.retained_coverage[actor, p] += 1
        if self.warmup_seconds is None and self.inserted >= 12288 and self.by_actor.min() >= 512 and self.seconds >= 1800:
            self.warmup_seconds = self.seconds
        return True

    def sample(self, rng, K, batch=512):
        # Balance actor contribution; never manufacture unavailable coverage.
        if not all(self.slots):
            raise ValueError('Every actor needs retained real experience')
        # Rotate the remainder, so cumulative actor contributions differ <=1.
        base, remainder = divmod(batch, 24)
        ids = np.repeat(np.arange(24), base)
        ids = np.r_[ids, (self.sample_calls * remainder + np.arange(remainder)) % 24]
        rng.shuffle(ids)
        self.sample_calls += 1
        self.sampled_by_actor += np.bincount(ids,minlength=24)
        chosen = [self.rows[self.slots[i][rng.randint(len(self.slots[i]))]] for i in ids]
        return dict(s0=np.asarray([r['s0'] for r in chosen], np.float32),
                    s1=np.asarray([r['s1'] for r in chosen], np.float32),
                    action=np.asarray([r['action'] for r in chosen], np.float32),
                    reward=np.asarray([[center(r['reward'], r['discount'], r['kind'], K)] for r in chosen], np.float32),
                    discount=np.asarray([[r['discount']] for r in chosen], np.float32),
                    bootstrap=np.asarray([[r['bootstrap']] for r in chosen], np.float32))


class Credit:
    def __init__(self):
        self.warmup_seconds = None
        self.critic = self.actor = self.n = self.target = self.b0 = 0

    def limit(self, replay):
        if self.warmup_seconds is None:
            if replay.inserted < 12288 or replay.by_actor.min() < 512 or replay.seconds < 1800:
                return 0
            self.warmup_seconds = replay.warmup_seconds if replay.warmup_seconds is not None else replay.seconds
        return int(math.floor(max(0, replay.seconds - self.warmup_seconds) / .02 * .08 + 1e-8))

    def next(self, replay):
        if self.critic >= self.limit(replay):
            raise ValueError('No real-experience credit')
        # The first 5000 updates calibrate critics only. N still learns each step.
        columns = [PERIODS.index(t) for t in replay.active_periods]
        return (self.critic >= 5000 and replay.retained_coverage[:, columns].min() >= 128
                and (self.critic + 1) % 2 == 0)

    def record(self, actor):
        self.critic += 1
        self.actor += int(actor)
        self.n += 1
        self.target += 1


class Exploration:
    def __init__(self, seed, active_periods=PERIODS):
        self.active_periods = training_periods(active_periods)
        self.rng = np.random.RandomState(seed)
        self.until = self.neutral_until = self.next_second = 0.
        self.period = 20

    def choose(self, elapsed, window):
        if elapsed >= self.until:
            self.period = int(self.rng.choice(self.active_periods))
            self.until = elapsed + self.rng.uniform(1, 5)
        while elapsed >= self.next_second:
            if self.next_second >= self.neutral_until and self.rng.rand() < .05:
                self.neutral_until = self.next_second + self.rng.uniform(.5, 1.)
            self.next_second += 1.
        neutral = elapsed < self.neutral_until
        value = window(self.period) if callable(window) else window
        a = 0. if neutral else float(np.clip(value + self.rng.normal(0, .2), -1, 1))
        return np.asarray([a, encoded_period(self.period)], np.float32), ('neutral_segment' if neutral else 'random_segment')


def checksum(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_execution(cfg):
    if cfg.get('decision_cost') != 0:
        raise ValueError('Reward v1 forbids per-transition cost; cost belongs only to B0 N')
    required = dict(scheduled_observation_v30=True,decision_observation_clock=True,
                    interval_rate_v28='acked_bytes',target_bn_consistent=True,joint_state_dim=149)
    for key,value in required.items():
        if cfg.get(key)!=value:raise ValueError('Wrong common-period execution contract: '+key)
    if cfg.get('gamma')!=.995:raise ValueError('Discount contract changed')
    reward=cfg['common_reward_config']
    if reward.get('denominator')!='global' or reward.get('scale_bps')!=6000000:
        raise ValueError('Use the frozen single global training scale')
    if any(cfg.get(k) for k in ('preemptive_mtp','diagnostic_local_mtp','startup_probe_seconds','legacy_protocol')):
        raise ValueError('No additional observation/early-wake/startup policy allowed')


class Hysteresis:
    """Existing local pressure thresholds, snapped to the five candidate periods.

The rule supplies only T; the same learned conditional actor supplies a.
Log-distance snapping avoids querying an untrained intermediate period.
"""
    def __init__(self):
        self.period=80
        self.stable=0

    def choose(self,state):
        state=np.asarray(state)
        if state.shape!=(149,) or not np.isfinite(state).all():raise ValueError('Invalid V31 state')
        frame=state[63:70]
        if frame[2]>.01 or frame[5]<.8 or frame[0]<.9:
            proposal=max(20,self.period/2)
            self.stable=0
        else:
            self.stable+=1
            proposal=self.period
            if self.stable>=3:
                proposal=min(160,self.period*1.5)
                self.stable=0
        self.period=min(PERIODS,key=lambda t:(abs(np.log(t/proposal)),t))
        return self.period
