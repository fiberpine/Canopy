"""Pure numerical contract. No TensorFlow, socket, clock or cloud side effects."""
import math
import re
import numpy as np

SCHEMA = 'healthy-seed3-joint82-v1'
BASE_COMMIT = 'c67717de585692bf6a79d1c0545ace92e66347d2'
STATE_DIM, ACTION_DIM, BASE_S = 82, 2, .02
PERIODS = (20., 40., 80., 120., 160.)
CALIBRATION, FORWARD_ONLY, JOINT = 0, 1, 2


def validate(cfg):
    required = dict(schema=SCHEMA, state_dim=82, action_dim=2,
                    reward_mode='raw',
                    reward_semantics='observed_interval_orca_density',
                    base_period_s=.02, gamma=.995, tau=.001, updates_per_base20ms=.1,
                    critic_loss='MSE', target_window_noise_std=.1, target_window_noise_clip=.2, target_time_noise_std=0.)
    for key, value in required.items():
        if cfg.get(key) != value: raise ValueError('Explicit contract mismatch: ' + key)
    for flag in ('target_bn_consistent', 'independent_critic_adam_v34',
                 'unique_critic_targets', 'unique_online_bn_updates'):
        if cfg.get(flag) is not True: raise ValueError('Required repair: ' + flag)
    if any(cfg.get(x) for x in ('property_reward', 'runtime_qc', 'fallback', 'capacity_oracle')):
        raise ValueError('Properties, QC, fallback and capacity oracle are disabled')
    if cfg.get('clock_mode') not in ('legacy_feedback', 'scheduled'):
        raise ValueError('Explicit observation clock required')
    if cfg.get('rate_mode') not in ('legacy', 'acked_bytes', 'acked_legacy_peak'):
        raise ValueError('Explicit rate measurement required')
    normalization = 'legacy_delivery_peak' if cfg['rate_mode'] == 'acked_legacy_peak' else 'observed_peak'
    if cfg.get('utility_normalization') != normalization: raise ValueError('Rate/normalizer contract mismatch')
    fixed = cfg.get('fixed_mtp_ms')
    if fixed is not None and float(fixed) not in PERIODS: raise ValueError('Unsupported fixed period')
    if cfg['clock_mode'] == 'legacy_feedback' and fixed != 20:
        raise ValueError('Compatibility clock only supports fixed20')
    if not 20 < float(cfg.get('time_init_ms', 40)) < 160:
        raise ValueError('Softsign initialization must be strictly inside the range')
    for name in ('decision_cost', 'physical_seconds_per_actor'):
        value = float(cfg.get(name, 0))
        if not math.isfinite(value) or value < 0: raise ValueError('Invalid ' + name)
    if int(cfg.get('num_actors', 0)) < 1: raise ValueError('Positive actor count required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', cfg.get('run_id', '')):
        raise ValueError('Run ID must remain a safe single directory component')
    if not 0 <= float(cfg.get('period_probe_probability', .1)) <= .2:
        raise ValueError('Calibration probes must be bounded')
    if cfg.get('connection_reset_seconds', 0) and cfg['connection_reset_seconds'] < 30:
        raise ValueError('Connection reset interval must be >=30s')
    return cfg


def period_ms(action):
    a = float(action)
    if not math.isfinite(a) or abs(a) > 1.00001: raise ValueError('Invalid time action')
    return 20. * 8. ** ((float(np.clip(a, -1., 1.)) + 1.) / 2.)


def period_action(ms):
    if not math.isfinite(float(ms)) or not 20 <= ms <= 160: raise ValueError('Invalid MTP')
    return 2. * math.log(float(ms) / 20.) / math.log(8.) - 1.


def softsign_inverse(action):
    if not -1 < action < 1: raise ValueError('No finite logit at a softsign endpoint')
    return float(action) / (1. - abs(float(action)))


def calibration_period(cfg, phase, rng):
    """An independently trained fixed-period control never explores its clock."""
    fixed = cfg.get('fixed_mtp_ms')
    if fixed is not None: return float(fixed), False
    nominal = 20. if phase == CALIBRATION else cfg['time_init_ms']
    if rng.random_sample() < cfg['period_probe_probability']:
        return float(rng.choice(PERIODS)), True
    return nominal, False


def wire(state_id, action, apply=True):
    a = np.asarray(action, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1.00001:
        raise ValueError('Two finite normalized actions required')
    if not isinstance(state_id, (int, np.integer)) or not 0 <= state_id < 1000:
        raise ValueError('Invalid state ID')
    percent = int(100 * 4 ** float(np.clip(a[0], -1., 1.))) if apply else 100
    us = int(round(period_ms(a[1]) * 1000))
    return ('%d %d %d %d\0' % (state_id, percent, int(apply), us)).encode('ascii')


def cwnd_target(action, cwnd):
    return min(2147483647, max(4, int(100 * 4 ** float(action)) * int(cwnd) // 100))


def interval_return(utility, duration, decision_cost=0., gamma=.995):
    if not all(math.isfinite(float(x)) for x in (utility, duration, decision_cost, gamma)):
        raise ValueError('Nonfinite SMDP input')
    if duration <= 0 or decision_cost < 0 or not 0 < gamma < 1:
        raise ValueError('Invalid SMDP input')
    log_discount = math.log(gamma) * duration / BASE_S
    discount = math.exp(log_discount)
    reward = float(utility) * (-math.expm1(log_discount)) / (1. - gamma) - decision_cost
    return reward, discount


def bellman(reward, discount, terminal, q1, q2):
    return np.asarray(reward) + np.asarray(discount) * (1. - np.asarray(terminal)) * np.minimum(q1, q2)


class Observation:
    """Original seven features, plus explicit history time; one call per boundary.

    Prefix throughput semantics are deliberately selected by the native mode.
    An empty RTT interval never becomes a fabricated fresh measurement.
    """
    def __init__(self, rate_mode='legacy'):
        if rate_mode not in ('legacy','acked_bytes','acked_legacy_peak'): raise ValueError('Observation rate schema')
        self.rate_mode = rate_mode
        self.max_bw = 0.
        self.history = np.zeros((10, 7), np.float32)
        self.gaps = np.zeros(10, np.float32)
        self.age_lower = self.age_upper = 0.
        self.frames = 0

    def observe(self, raw, previous_planned_ms=20.):
        x = np.asarray(raw, dtype=np.float64)
        if x.shape != (15,) or not np.isfinite(x).all(): raise ValueError('Invalid feedback shape/values')
        delay, rate, count, dt, _, cwnd, pacing, loss, srtt = x[:9]
        minimum = x[14]
        if min(dt, cwnd, srtt, minimum) <= 0 or min(rate, count, loss, pacing) < 0:
            raise ValueError('Invalid feedback units/ranges')
        if not self.frames and count <= 0: raise ValueError('Bootstrap needs a real RTT sample')
        normalization_rate = x[4] if self.rate_mode == 'acked_legacy_peak' else rate
        if normalization_rate < 0: raise ValueError('Negative normalization estimator')
        self.max_bw = max(self.max_bw, normalization_rate)
        if self.max_bw <= 0: raise ValueError('No observed bandwidth for reward normalization')
        penalty = min(1., 1.25 * minimum / srtt)
        utility = (rate - 5 * loss) / self.max_bw * penalty
        frame = np.array([rate / self.max_bw, min(10., pacing / self.max_bw),
                          5 * loss / self.max_bw, count / cwnd, dt, minimum / srtt, penalty], np.float32)
        self.history = np.concatenate((self.history[1:], frame[None, :]))
        self.gaps = np.concatenate((self.gaps[1:], np.array([dt], np.float32)))
        if count > 0: self.age_lower, self.age_upper = 0., dt
        else: self.age_lower += dt; self.age_upper += dt
        extra = np.r_[math.log1p(dt / BASE_S), period_action(previous_planned_ms), np.log1p(self.gaps / BASE_S)]
        state = np.r_[self.history.reshape(-1), extra].astype(np.float32)
        assert state.shape == (82,) and np.isfinite(state).all()
        self.frames += 1
        metadata = dict(fresh_rtt=bool(count > 0), rtt_age_lower_s=self.age_lower,
                        rtt_age_upper_s=self.age_upper, actual_dt_s=dt, max_bw=self.max_bw,
                        original_utility=utility, throughput_term=rate / self.max_bw,
                        loss_term=5 * loss / self.max_bw, delay_term=penalty,
                        normalization_sample=float(normalization_rate), normalization_source=self.rate_mode)
        return state, float(utility), metadata


class PhysicalCredit:
    """Only real INSERTED exposure earns credits, after a shared warm-up gate."""
    def __init__(self, actors, warmup=12288, each=512, minimum_seconds=245.76, ratio=.1):
        self.actors, self.warmup, self.each = actors, warmup, each
        self.minimum_seconds, self.ratio = minimum_seconds, ratio
        self.counts = np.zeros(actors, np.int64)
        self.actor_seconds = np.zeros(actors, np.float64)
        self.inserted = 0; self.seconds = 0.; self.compensation = 0.
        self.warmup_seconds = None; self.warmup_inserted = None

    def insert(self, actor, dt):
        if not 0 <= actor < self.actors or not math.isfinite(dt) or dt <= 0: raise ValueError('Invalid real exposure')
        self.inserted += 1; self.counts[actor] += 1
        self.actor_seconds[actor] += dt
        corrected = float(dt) - self.compensation
        total = self.seconds + corrected
        self.compensation = (total - self.seconds) - corrected; self.seconds = total
        if self.warmup_seconds is None and self.inserted >= self.warmup and self.counts.min() >= self.each and self.seconds + 1e-9 >= self.minimum_seconds:
            self.warmup_seconds, self.warmup_inserted = self.seconds, self.inserted

    @property
    def limit(self):
        if self.warmup_seconds is None: return 0
        return int(math.floor(max(0., self.seconds - self.warmup_seconds) * self.ratio / BASE_S + 1e-9))

    def record(self):
        return dict(inserted=self.inserted, physical_seconds=self.seconds, warmup_seconds=self.warmup_seconds,
                    warmup_inserted=self.warmup_inserted, per_actor=self.counts.tolist(), per_actor_seconds=self.actor_seconds.tolist(), update_limit=self.limit,
                    updates_per_base20ms=self.ratio)
