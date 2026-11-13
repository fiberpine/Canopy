"""One explicit candidate schema; G0/G1/G2 alter one measurement layer at a time."""
import copy
import json
from pathlib import Path
from .core import SCHEMA, BASE_COMMIT, validate

ROOT = Path(__file__).resolve().parents[2]


def configuration(profile='joint'):
    source = json.loads((ROOT / 'templates/baseline-train.json').read_text())
    cfg = dict(schema=SCHEMA, base_commit=BASE_COMMIT, state_dim=82, action_dim=2,
        reward_mode='raw', reward_semantics='observed_interval_orca_density', utility_normalization='observed_peak',
        property_reward=False, runtime_qc=False, fallback=False, capacity_oracle=False,
        base_period_s=.02, gamma=.995, decision_cost=0., updates_per_base20ms=.1,
        target_bn_consistent=True, independent_critic_adam_v34=True, unique_critic_targets=True,
        unique_online_bn_updates=True, lr_a=.0001, lr_c=.001, tau=.001,
        critic_loss='MSE', target_window_noise_std=.1, target_window_noise_clip=.2, target_time_noise_std=0.,
        seed=3, num_actors=24, memsize=150000, batch_size=512, warmup=12288,
        minimum_enqueued_per_actor=512, warmup_physical_seconds=245.76,
        calibration_updates=5000, forward_calibration_updates=1000, calibration_check_updates=500,
        calibration_rows_per_actor_period=128, minimum_holdout_per_period=16,
        holdout_rows_per_period=256, maximum_normalized_td_rmse=.25, relative_td_limit=2.,
        short_return_blocks_per_period=64, minimum_short_return_blocks=4, maximum_normalized_short_return_rmse=.35,
        time_init_ms=40., period_probe_probability=.1, clock_mode='scheduled', rate_mode='acked_bytes',
        fixed_mtp_ms=None, physical_seconds_per_actor=600., connection_reset_seconds=0.,
        exploration_std=.2, checkpoint_updates=10000, stop_reserve_s=60,
        run_id='new-joint-healthy-run', learner_endpoint='@LEARNER_IP@:25000', actor_port_base=23000,
        conditions=copy.deepcopy(source['conditions']), profile=profile,
        baseline_checkpoint=str(ROOT / 'models/healthy-baseline-seed3/training/step-6013196'),
        source_policy_sha256='631683b939596cabc85262e2294123c462913bf6b836dff10b84501644ad60e0',
        candidate_status='OFFLINE_ONLY_NOT_TCP_VALIDATED', tf_threads=4)
    if profile in ('g0', 'g1', 'g2'):
        cfg['fixed_mtp_ms'] = 20.
        if profile == 'g0': cfg.update(clock_mode='legacy_feedback', rate_mode='legacy')
        if profile == 'g1': cfg['rate_mode'] = 'legacy'
    elif profile.startswith('fixed') and profile[5:] in ('20', '40', '80', '120', '160'):
        cfg['fixed_mtp_ms'] = float(profile[5:])
    elif profile != 'joint': raise ValueError('Unknown candidate profile')
    return validate(cfg)


def checked_authorization(path):
    """A future network run needs a separately registered, bounded TCP window."""
    import time
    import math
    auth = json.loads(Path(path).read_text())
    now = time.time()
    if auth.get('allow_tcp') is not True or not auth.get('purpose'):
        raise ValueError('Explicit TCP authorization and purpose required')
    start, deadline = float(auth['start_epoch']), float(auth['deadline_epoch'])
    if not all(math.isfinite(x) for x in (start,deadline)) or not start < deadline:
        raise ValueError('A finite ordered authorization window is required')
    if not start <= now < deadline - 120:
        raise ValueError('Not inside the registered window with cleanup reserve')
    return auth


def runtime_preflight():
    """Read-only environment check; never silently tune a benchmark host."""
    import os
    import shutil
    if os.name != 'posix' or os.uname().release != '4.13.1-0521-learner':
        raise ValueError('The validated custom4.13.1-0521-learner kernel is required')
    required = {'/proc/sys/net/ipv4/tcp_congestion_control': 'cubic',
                '/proc/sys/net/ipv4/tcp_no_metrics_save': '1',
                '/sys/module/tcp_cubic/parameters/hystart': '1',
                '/sys/module/tcp_cubic/parameters/initial_ssthresh': '0'}
    actual = {name: Path(name).read_text().strip() for name in required}
    if actual != required: raise ValueError('Sender/startup settings differ from the healthy baseline: ' + json.dumps(actual))
    if not Path('/dev/net/tun').exists() or not all(shutil.which(x) for x in ('mm-link', 'mm-delay', 'sudo', 'ip')):
        raise ValueError('TUN/Mahimahi/system tools are incomplete')
    return dict(kernel=os.uname().release, sender_settings=actual)
