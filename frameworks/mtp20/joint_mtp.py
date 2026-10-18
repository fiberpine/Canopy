"""Joint cwnd/intervention timing; pure numerical core, Linux I/O imported lazily.

No repeated cwnd writes between policy decisions. CUBIC keeps evolving.
Reward is the existing Orca utility sampled at the unchanged base cadence.
"""
import math
import numpy as np

SCHEMA = 'joint-cwnd-mtp-smdp-v1'


def validate(cfg):
    if not cfg.get('joint_control'):
        raise ValueError('joint_control must be explicit')
    if cfg.get('reward_mode', 'raw') != 'raw':
        raise ValueError('Joint control preserves raw Orca reward')
    if not isinstance(cfg.get('target_bn_consistent', False), bool):
        raise ValueError('target_bn_consistent must be an explicit boolean')
    if cfg.get('legacy_protocol'):
        raise ValueError('Strict IDs required')
    if cfg.get('connection_reset_seconds'):
        if cfg.get('window_policy_mode') not in ('frozen_scalar','staged_joint') or not math.isfinite(float(cfg['connection_reset_seconds'])) or float(cfg['connection_reset_seconds'])<30:
            raise ValueError('Connection resets require explicit staged mode and >=30 seconds')
    import staged_timing
    staged_timing.validate(cfg)
    if cfg.get('backend') != 'moving_stats':
        raise ValueError('Joint V1 requires the verified moving_stats online backend')
    lo, hi = bounds(cfg)
    if not 20 <= lo < hi <= 1000 or cfg.get('control_period_ms', 20) != 20:
        raise ValueError('Keep the base observation cadence at 20ms; bounded MTP required')
    if not 0 < cfg.get('gamma', .995) < 1:
        raise ValueError('gamma must be strictly between zero and one')
    cost = cfg.get('decision_cost', 0.)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('decision_cost must be nonnegative and finite')
    fixed = cfg.get('fixed_mtp_ms')
    if fixed is not None and not lo <= fixed <= hi:
        raise ValueError('fixed MTP outside supported bounds')
    if cfg.get('fixed_action') is not None and not cfg.get('capacity_only'):
        raise ValueError('fixed cwnd action is a capacity diagnostic, not training')
    probe=float(cfg.get('startup_probe_seconds',0))
    if not math.isfinite(probe) or not 0<=probe<=10:
        raise ValueError('Explicit native CUBIC discovery phase must be bounded to 0..10 seconds')


def bounds(cfg):
    return float(cfg.get('mtp_min_ms', 20)), float(cfg.get('mtp_max_ms', 160))


def fixed_action(cfg):
    return mtp_action(cfg['fixed_mtp_ms'], cfg) if cfg.get('fixed_mtp_ms') is not None else None


def mtp_ms(a, cfg):
    lo, hi = bounds(cfg)
    if not math.isfinite(float(a)) or not -1.00001 <= a <= 1.00001:
        raise ValueError('invalid normalized MTP action')
    return lo * (hi / lo) ** ((float(np.clip(a, -1, 1)) + 1) / 2)


def mtp_action(ms, cfg):
    lo, hi = bounds(cfg)
    if not lo <= ms <= hi:
        raise ValueError('invalid MTP')
    return 2 * math.log(ms / lo) / math.log(hi / lo) - 1


def encode_state(history, actual_dt_s, previous_planned_ms, cfg):
    """70 base-clock history features + previous duration and planned-time features."""
    if len(history) != 70 or not np.all(np.isfinite(history)):
        raise ValueError('invalid base-clock history')
    lo, hi = bounds(cfg)
    duration_feature = math.log1p(max(0., actual_dt_s) * 1000 / lo)
    # Zero initial duration is an explicit reset; large scheduling overruns are retained.
    return np.concatenate((history, [duration_feature, mtp_action(previous_planned_ms, cfg)])).astype(np.float32)


class Interval:
    def __init__(self, gamma=.995, base_s=.02, decision_cost=0.):
        if not 0 < gamma < 1 or base_s <= 0 or decision_cost < 0:
            raise ValueError('invalid time/reward parameters')
        self.gamma, self.base_s = gamma, base_s
        self.elapsed = 0.
        self.reward = -decision_cost
        self.undiscounted = 0.
        self.ticks = 0

    @property
    def discount(self):
        return self.gamma ** (self.elapsed / self.base_s)

    def add(self, utility, dt_s):
        if not math.isfinite(utility) or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('invalid feedback duration or reward')
        # Piecewise constant utility integral, normalized so a single exact base
        # interval returns the original reward. Split/merge invariant for constant u.
        mass = -math.expm1(math.log(self.gamma) * dt_s / self.base_s) / (1 - self.gamma)
        self.reward += self.discount * utility * mass
        self.undiscounted += utility * dt_s / self.base_s
        self.elapsed += dt_s
        self.ticks += 1


def command(state_id, action=None):
    """Protocol: ID percent APPLY. APPLY=0 acknowledges observation without setsockopt."""
    if action is None:
        return ('%d 100 0\0' % state_id).encode('ascii')
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape != (2,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.00001):
        raise ValueError('joint action must contain two finite normalized values')
    percent = int(100 * 4 ** float(np.clip(action[0], -1, 1)))
    return ('%d %d 1\0' % (state_id, percent)).encode('ascii')


def bellman(reward, discount, terminal, q1, q2):
    return reward + discount * (1 - terminal) * np.minimum(q1, q2)


def summarize_decisions(path):
    import csv
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return dict(decisions=0)
    def values(name):
        return np.array([float(row[name]) for row in rows])
    def stats(name):
        x = values(name)
        return dict(mean=float(np.mean(x)),p50=float(np.percentile(x,50)),
                    p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)))
    exposure_s = float(values('actual_dt_ms').sum() / 1000)
    result = dict(decisions=len(rows),physical_exposure_s=exposure_s,
                decisions_per_physical_s=len(rows)/max(1e-9,exposure_s),
                planned_mtp_ms=stats('planned_mtp_ms'),actual_dt_ms=stats('actual_dt_ms'),
                duration_overrun_ms=stats('duration_overrun_ms'),inference_ms=stats('inference_ms'),
                truncated=int(values('truncated').sum()),
                time_weighted_raw_utility=float(values('raw_utility_integral').sum()*.02/max(1e-9,exposure_s)),
                cwnd_low_saturation=float(np.mean(values('cwnd_no_noise')<=-.99)),
                cwnd_high_saturation=float(np.mean(values('cwnd_no_noise')>=.99)),
                mtp_low_saturation=float(np.mean(values('mtp_no_noise')<=-.99)),
                mtp_high_saturation=float(np.mean(values('mtp_no_noise')>=.99)),
                cwnd_action_std=float(np.std(values('cwnd_no_noise'))),
                mtp_action_std=float(np.std(values('mtp_no_noise'))))
    control_path=path.with_name('control.csv')
    if control_path.exists():
        with control_path.open() as f:control=list(csv.DictReader(f))
        native=[float(row['time_us']) for row in control if row.get('apply')=='1']
        periods=np.diff(native)/1000
        result['native_interventions']=len(native)
        result['native_telemetry_acks']=sum(row.get('apply')=='0' for row in control)
        result['native_intervention_period_ms']=(dict(mean=float(np.mean(periods)),
            p50=float(np.percentile(periods,50)),p95=float(np.percentile(periods,95)),
            p99=float(np.percentile(periods,99))) if len(periods) else {})
    return result


def drive(env, reader, shm_w, act, cfg, stop, state_writer, run, transition_hook=None):
    """Base-clock telemetry with policy calls only at measured-time decision boundaries.

    Reuses strict state reader/cleanup/ownership from evaluate.py. Native no-op ACKs
    permit telemetry without touching TCP_CWND. Partial final decisions bootstrap
    their real next state (administrative truncation); no cross-connection transition.
    """
    import csv
    import json
    import pathlib
    import time
    validate(cfg)
    def extract(info):
        return info['state'] if isinstance(info, dict) else info
    from observation_noise_v14 import Observation
    observation=Observation(cfg)
    history = np.zeros(70, np.float32)
    history[-7:] = observation.apply(extract(env.reset()))
    # Optional diagnostic/staged startup: observe native CUBIC without window
    # writes or policy calls. Do not grant training reward/credit for this phase.
    probe_elapsed=0.;probe_observations=0
    if cfg.get('startup_probe_seconds',0):
        probe_start=time.monotonic();initial_peak=env.max_bw
        shm_w.write(command(reader.last_id))
        with (run/'startup-probe.csv').open('w',buffering=262144) as probe_file:
            probe_writer=csv.writer(probe_file);probe_writer.writerow(['state_id','delta_t','raw_utility','max_bw','cwnd'])
            while probe_elapsed<float(cfg['startup_probe_seconds']):
                if stop.is_set() or time.time()>=cfg.get('absolute_deadline_epoch',float('inf')):
                    raise RuntimeError('Stopped during bounded native CUBIC discovery')
                info,delay,utility,ok=env.get_state(evaluation=True);sample=observation.apply(extract(info));raw=reader.last_raw
                if not ok or len(sample)!=7 or not np.all(np.isfinite(sample)) or not math.isfinite(raw[3]) or raw[3]<=0:
                    raise ValueError('Invalid discovery feedback')
                history=np.concatenate((history[7:],sample)).astype(np.float32);probe_elapsed+=float(raw[3]);probe_observations+=1
                probe_writer.writerow([reader.last_id,raw[3],utility,env.max_bw,raw[5]])
                # Last feedback remains unacknowledged until the first policy
                # command, avoiding a same-ID no-op/action overwrite race.
                if probe_elapsed<float(cfg['startup_probe_seconds']):shm_w.write(command(reader.last_id))
        (run/'startup-probe.json').write_text(json.dumps(dict(requested_seconds=cfg['startup_probe_seconds'],physical_seconds=probe_elapsed,wall_seconds=time.monotonic()-probe_start,observations=probe_observations,policy_inferences=0,window_writes=0,initial_max_bw=initial_peak,final_max_bw=env.max_bw,in_replay=False)))
    state = encode_state(history, 0., bounds(cfg)[0], cfg)
    decisions = observations = inferences = 0
    physical_elapsed = 0.
    start = time.monotonic()
    last_flush = start
    deadline = cfg.get('absolute_deadline_epoch', float('inf'))
    inference_ms = 0.
    diagnostic_probe = None
    if cfg.get('diagnostic_clock_v11') or cfg.get('diagnostic_local_mtp'):
        if transition_hook is not None:
            raise ValueError('Oracle local timing interventions are frozen diagnostics only')
        import local_mtp_probe_v11
        diagnostic_probe = local_mtp_probe_v11.Probe(cfg,run)
    def infer(s):
        nonlocal inferences, inference_ms
        before = time.perf_counter()
        value = np.asarray(act(s), dtype=np.float32).reshape(-1)
        inference_ms = (time.perf_counter() - before) * 1000
        inferences += 1
        if value.shape != (2,) or not np.all(np.isfinite(value)) or np.any(np.abs(value) > 1.00001):
            raise ValueError('invalid joint policy output')
        value = np.clip(value, -1, 1)
        if cfg.get('fixed_mtp_ms') is not None:
            value[1] = mtp_action(cfg['fixed_mtp_ms'], cfg)
        if diagnostic_probe is not None:
            value[1] = local_mtp_probe_v11.encoded_choice(value[1],diagnostic_probe.choose(mtp_ms(value[1],cfg)),cfg)
        return value
    action = infer(state)
    no_noise = np.asarray(getattr(act, 'no_noise', action)).copy()
    planned = mtp_ms(action[1], cfg)
    shm_w.write(command(reader.last_id, action))
    current = Interval(cfg.get('gamma', .995), .02, cfg.get('decision_cost', 0.))
    decision_inference_ms = inference_ms
    final_partial = None
    with (run / 'decisions.csv').open('w', buffering=262144) as f:
        writer = csv.writer(f)
        writer.writerow(['decision', 'elapsed_s', 'state_id', 'cwnd_action', 'mtp_action',
                         'cwnd_no_noise', 'mtp_no_noise', 'planned_mtp_ms', 'actual_dt_ms',
                         'duration_overrun_ms', 'observation_ticks', 'reward', 'raw_utility_integral',
                         'discount', 'inference_ms', 'truncated', 'terminal'])
        while not stop.is_set():
            try:
                info, delay, utility, ok = env.get_state(evaluation=True)
            except Exception as error:
                # Only clean native end-of-run is a time truncation. Invalid feedback
                # and watchdog failures propagate to the evaluator's failure record.
                from evaluate import EndOfRun
                if isinstance(error, EndOfRun):
                    requested = (stop.is_set() or time.time() >= deadline or
                        (cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists()))
                    timed_end = (cfg.get('seconds', 0) and reader.server.poll() == 0 and
                                 time.time() - reader.server.created >= cfg['seconds'] - .5)
                    if not (requested or timed_end):
                        raise RuntimeError('Unexpected native exit before the requested joint run completed')
                    final_partial = current.ticks > 0
                    break
                raise
            sample = observation.apply(extract(info))
            raw = reader.last_raw
            if not ok or len(sample) != 7 or not np.all(np.isfinite(sample)):
                raise ValueError('invalid joint feedback')
            history = np.concatenate((history[7:], sample)).astype(np.float32)
            current.add(float(utility), float(raw[3]))
            physical_elapsed += float(raw[3])
            observations += 1
            is_stop = bool(time.time() >= deadline or
                       (cfg.get('seconds', 0) and time.monotonic() - start >= cfg['seconds']) or
                       (cfg.get('episode_control_seconds', 0) and time.monotonic() - start >= cfg['episode_control_seconds']) or
                       (cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists()) or
                       (cfg.get('physical_seconds_per_actor') and physical_elapsed >= cfg['physical_seconds_per_actor']))
            boundary = current.elapsed * 1000 + 1e-5 >= planned
            # A step budget counts decisions, never telemetry/no-op ticks.
            if boundary and cfg.get('training_steps') and decisions + 1 >= cfg['training_steps']:
                is_stop = True
            next_state = encode_state(history, current.elapsed, planned, cfg)
            if boundary or is_stop:
                decisions += 1
                completed = (state.copy(), action.copy(), current.reward, next_state.copy(),
                             current.elapsed, current.discount)
                old_action, old_no_noise, old_planned = action, no_noise, planned
                old_inference_ms = decision_inference_ms
                if not is_stop:
                    action = infer(next_state)
                    no_noise = np.asarray(getattr(act, 'no_noise', action)).copy()
                    planned = mtp_ms(action[1], cfg)
                    # Send before replay/logging to avoid learner-induced timing stalls.
                    shm_w.write(command(reader.last_id, action))
                    decision_inference_ms = inference_ms
                else:
                    shm_w.write(command(reader.last_id))
                writer.writerow([decisions, time.monotonic() - start, reader.last_id,
                                 old_action[0], old_action[1], old_no_noise[0], old_no_noise[1],
                                 old_planned, current.elapsed * 1000,
                                 current.elapsed * 1000 - old_planned, current.ticks, current.reward,
                                 current.undiscounted, current.discount, old_inference_ms,
                                 int(is_stop and not boundary), 0])
                if transition_hook:
                    transition_hook(*completed)
                state = next_state
                current = Interval(cfg.get('gamma', .995), .02, cfg.get('decision_cost', 0.))
            else:
                old_action, old_no_noise = action, no_noise
                shm_w.write(command(reader.last_id))
            state_writer.writerow([time.time() * 1e6, time.monotonic() - start, observations,
                                    reader.last_id, old_action[0], old_no_noise[0],
                                    int(100 * 4 ** float(old_action[0])), utility,
                                    raw[1] / env.max_bw, 5 * raw[7] / env.max_bw, sample[-1],
                                    env.max_bw, raw[5], raw[3], raw[8], raw[14], json.dumps(sample.tolist())])
            if is_stop:
                break
            if time.monotonic() - last_flush > 5:
                f.flush()
                last_flush = time.monotonic()
    # Native exit with an unobserved tail cannot fabricate a terminal next state.
    # Already observed subintervals are retained using their last real state.
    if current.ticks:
        next_state = encode_state(history, current.elapsed, planned, cfg)
        if transition_hook:
            transition_hook(state.copy(), action.copy(), current.reward, next_state,
                            current.elapsed, current.discount)
        decisions += 1
        final_partial = True
        with (run / 'decisions.csv').open('a') as f:
            csv.writer(f).writerow([decisions, time.monotonic() - start, reader.last_id,
                action[0], action[1], no_noise[0], no_noise[1], planned, current.elapsed * 1000,
                current.elapsed * 1000 - planned, current.ticks, current.reward,
                current.undiscounted, current.discount, decision_inference_ms, 1, 0])
    (run / 'joint-summary.json').write_text(json.dumps(dict(
        schema=SCHEMA, decisions=decisions, observations=observations,
        policy_inferences=inferences, unobserved_tail_dropped=True,
        physical_exposure_s=physical_elapsed,
        discovery_physical_exposure_s=probe_elapsed,
        discovery_observations=probe_observations,
        final_observed_partial_transition=bool(final_partial),
        scheduling='first base-clock feedback at or after planned duration; actual kernel dt used',
        terminal_rule='time/step truncations bootstrap real next state; no fabricated connection-terminal sample'), indent=2))
    if diagnostic_probe is not None:diagnostic_probe.close()
    return observations
