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
        holdout_layout='episode_rotated_blocks_v1',
        short_return_blocks_per_period=64, minimum_short_return_blocks=4, maximum_normalized_short_return_rmse=.35,
        time_init_ms=40., time_initializer='constant', calibration_main_mode='constant',
        period_probe_probability=.1, clock_mode='scheduled', rate_mode='acked_bytes',
        fixed_mtp_ms=None, physical_seconds_per_actor=600., connection_reset_seconds=0.,
        time_limit_handling='bootstrap_last_observation',
        exploration_std=.2, checkpoint_updates=10000, stop_reserve_s=60,
        run_id='new-joint-healthy-run', learner_endpoint='@LEARNER_IP@:25000', actor_port_base=23000,
        conditions=copy.deepcopy(source['conditions']), profile=profile,
        baseline_checkpoint=str(ROOT / 'models/healthy-baseline-seed3/training/step-6013196'),
        source_policy_sha256='631683b939596cabc85262e2294123c462913bf6b836dff10b84501644ad60e0',
        candidate_status='OFFLINE_ONLY_NOT_TCP_VALIDATED', tf_threads=4)
    if profile in ('continuous-v4', 'continuous-v5'):
        from .continuous import ARCHITECTURE, ARCHITECTURE_V5, SCHEMA as continuous_schema, SCHEMA_V5
        is_v5 = (profile == 'continuous-v5')
        cfg.update(schema=SCHEMA_V5 if is_v5 else continuous_schema,
                   actor_architecture=ARCHITECTURE_V5 if is_v5 else ARCHITECTURE,
                   actor_recipe=ARCHITECTURE_V5 if is_v5 else ARCHITECTURE,
                   time_parameterization='physical_sigmoid_v1', time_logit_exploration_std=.25,
                   rate_mode='acked_legacy_peak', utility_normalization='legacy_delivery_peak',
                   time_initializer='feedback_coverage_v1', calibration_main_mode='policy_initializer',
                   window_bn_statistics='frozen_migrated', time_feature_normalization='unit_l2_v1',
                   teacher_anchor_weight=10., teacher_anchor_delta=.1, teacher_anchor_clock_weighting='base20_only',
                   window_actor_lr=.00001, time_actor_lr=.0003, decision_cost=.01,
                   joint_period_probe_probability=.15, critic_initial_cost_reference_s=.0216,
                   critic_period_residual_lr=.0001, critic_common_calibration_updates=1000,
                   memsize=300000, connection_reset_seconds=180., candidate_status='OFFLINE_ONLY_NOT_TCP_VALIDATED')
        if is_v5:
            cfg.update(critic_period_zero_anchor=True,
                       envelope_stop_gradient_duration=True,
                       pre_adapt_conditioned_window_in_forward_phase=True,
                       conditioned_window_actor_lr=.001,
                       time_actor_lr=.0006,
                       time_barrier_weight=.0005,
                       critic_period_residual_lr=.0001,
                       critic_period_residual_l2=.0005,
                       calibration_critic_tau=.05,
                       common_calibration_n_step=16,
                       period_probe_probability=.2)
    elif profile=='g4':
        cfg.update(rate_mode='acked_legacy_peak', utility_normalization='legacy_delivery_peak',
                   time_initializer='feedback_coverage_v1',calibration_main_mode='policy_initializer')
    elif profile in ('g0', 'g1', 'g2', 'g3'):
        cfg['fixed_mtp_ms'] = 20.
        if profile == 'g0': cfg.update(clock_mode='legacy_feedback', rate_mode='legacy')
        if profile == 'g1': cfg['rate_mode'] = 'legacy'
        if profile == 'g3': cfg.update(rate_mode='acked_legacy_peak', utility_normalization='legacy_delivery_peak')
    elif profile.startswith('fixed') and profile[5:] in ('20', '40', '80', '120', '160'):
        cfg['fixed_mtp_ms'] = float(profile[5:])
    elif profile in ('canopy-dau-jsac', 'canopy-dau-jsac-v2', 'canopy-dau-jsac-v21', 'canopy-dau-jsac-v3', 'canopy-dau-jsac-v3-balanced', 'canopy-dau-jsac-v3-noibp'):
        from .dau_jsac import default_dau_jsac_config
        cfg.update(default_dau_jsac_config())
        cfg.update(
            profile=profile,
            rate_mode='acked_legacy_peak',
            utility_normalization='legacy_delivery_peak',
            clock_mode='scheduled',
            memsize=300000,
            batch_size=256,
            updates_per_base20ms=0.05,
            warmup=12288,
            minimum_enqueued_per_actor=512,
            warmup_physical_seconds=245.76,
            critic_warmup_updates=1000,
            connection_reset_seconds=180.0,
            physical_seconds_per_actor=600.0,
            checkpoint_updates=10000,
            warmstart_probes=str(ROOT / 'recovery/real-frozen-probes.npz'),
            warmstart_steps=600,
            warmstart_init_period_ms=40.0,
            candidate_status='OFFLINE_VERIFIED_READY_FOR_CLOUD',
        )
        if profile in ('canopy-dau-jsac-v2', 'canopy-dau-jsac-v21', 'canopy-dau-jsac-v3', 'canopy-dau-jsac-v3-balanced', 'canopy-dau-jsac-v3-noibp'):
            def _mc(name, trace, mbps, rtt, bdp=5.0):
                q = max(20, int(round(bdp * mbps * 1e6 * (rtt / 1000.0) / (8.0 * 1504.0))))
                return dict(condition=name, trace=trace, nominal_mbps=mbps, base_rtt_ms=rtt,
                            delay_ms=int(rtt // 2), queue_packets=q, queue_bytes=q * 1504, queue_bdp=bdp)
            if profile in ('canopy-dau-jsac-v3', 'canopy-dau-jsac-v3-balanced', 'canopy-dau-jsac-v3-noibp'):
                # High-SNR 6/6/6/6 RTT-balanced expanded suite:
                # 20/24 (83.3%) clean mainstream wired CBR (12..96 Mbps, 2-5 BDP) + 4/24 (16.7%) gentle dynamic wired (1/RTT)
                # Huber Critic (delta=2.0) + IBP-exact RSNorm + PID-Lagrangian + zero hard peak window jumps
                expanded = [
                    # R20 (6 actors: 5 wired + 1 dynamic)
                    _mc('W12-R20', 'constant12', 12, 20, 5.0),
                    _mc('W24-R20-q2', 'constant24', 24, 20, 2.0),
                    _mc('W36-R20', 'constant36', 36, 20, 5.0),
                    _mc('W48-R20', 'constant48', 48, 20, 5.0),
                    _mc('W72-R20', 'constant72', 72, 20, 5.0),
                    _mc('D-step24-R20', 'dyn-step24-s0', 24, 20, 5.0),
                    # R50 (6 actors: 5 wired + 1 dynamic)
                    _mc('W12-R50', 'constant12', 12, 50, 5.0),
                    _mc('W24-R50', 'constant24', 24, 50, 5.0),
                    _mc('W42-R50', 'constant42', 42, 50, 5.0),
                    _mc('W72-R50-q2', 'constant72', 72, 50, 2.0),
                    _mc('W96-R50', 'constant96', 96, 50, 5.0),
                    _mc('D-ramp48-R50', 'dyn-ramp48-s0', 48, 50, 5.0),
                    # R100 (6 actors: 5 wired + 1 dynamic)
                    _mc('W12-R100', 'constant12', 12, 100, 5.0),
                    _mc('W24-R100', 'constant24', 24, 100, 5.0),
                    _mc('W42-R100', 'constant42', 42, 100, 5.0),
                    _mc('W48-R100-q2', 'constant48', 48, 100, 2.0),
                    _mc('W72-R100', 'constant72', 72, 100, 5.0),
                    _mc('D-burst48-R100', 'dyn-burst48-s0', 48, 100, 5.0),
                    # R200 (6 actors: 5 wired + 1 dynamic)
                    _mc('W12-R200', 'constant12', 12, 200, 5.0),
                    _mc('W24-R200', 'constant24', 24, 200, 5.0),
                    _mc('W36-R200', 'constant36', 36, 200, 5.0),
                    _mc('W48-R200', 'constant48', 48, 200, 5.0),
                    _mc('W72-R200', 'constant72', 72, 200, 5.0),
                    _mc('D-ramp48-R200', 'dyn-ramp48-s0', 48, 200, 5.0),
                ]
                cfg.update(
                    eval_conditions=copy.deepcopy(source['conditions']),
                    conditions=expanded,
                    obs_norm_enabled=True,
                    obs_norm_momentum=0.005,
                    obs_norm_std_floor=0.20,
                    pid_lagrangian=True,
                    dual_kp=1.0,
                    dual_ki=0.05,
                    dual_kd=0.25,
                    critic_huber_delta=2.0,
                    peak_window_s=0.0,
                    decision_cost=0.025,
                    joint_effort_delta=0.25,
                    connection_reset_seconds=180.0,
                    physical_seconds_per_actor=900.0,
                    checkpoint_updates=15000,
                    memsize=400000,
                )
                if profile == 'canopy-dau-jsac-v3-balanced':
                    cfg.update(decision_cost=0.018, joint_effort_delta=0.20)
                elif profile == 'canopy-dau-jsac-v3-noibp':
                    cfg.update(canopy_ibp_enabled=False)
            elif profile == 'canopy-dau-jsac-v21':
                # Strict 6/6/6/6 RTT-balanced, cleaned, mainstream-weighted 24-actor suite:
                # 14/24 (58.3%) mainstream wired CBR + 8/24 (33.3%) clean dynamic wired = 91.7% mainstream
                #  2/24 ( 8.3%) clean real cellular LTE/mobile (zero blackouts, zero held-out leakage)
                expanded = [
                    # R20 (6 actors)
                    _mc('W12-R20', 'constant12', 12, 20, 5.0),
                    _mc('W24-R20-q2', 'constant24', 24, 20, 2.0),
                    _mc('W48-R20', 'constant48', 48, 20, 5.0),
                    _mc('W72-R20', 'constant72', 72, 20, 5.0),
                    _mc('D-step24-R20', 'dyn-step24-s0', 24, 20, 5.0),
                    _mc('D-mix24-R20', 'v14-mixed_timescale24-train-141240', 24, 20, 3.0),
                    # R50 (6 actors)
                    _mc('W18-R50', 'constant18', 18, 50, 5.0),
                    _mc('W36-R50', 'constant36', 36, 50, 5.0),
                    _mc('W96-R50', 'constant96', 96, 50, 5.0),
                    _mc('D-ramp48-R50', 'dyn-ramp48-s0', 48, 50, 5.0),
                    _mc('D-cdrop24-R50', 'v14-compound_drop24-train-140240', 24, 50, 5.0),
                    _mc('C-TMob-R50', 'v12-real-TMobile-LTE-short.down', 24, 50, 5.0),
                    # R100 (6 actors)
                    _mc('W12-R100', 'constant12', 12, 100, 5.0),
                    _mc('W42-R100', 'constant42', 42, 100, 5.0),
                    _mc('W72-R100-q2', 'constant72', 72, 100, 2.0),
                    _mc('D-burst48-R100', 'dyn-burst48-s0', 48, 100, 5.0),
                    _mc('D-cdrop48-R100', 'v14-compound_drop48-train-140480', 48, 100, 5.0),
                    _mc('C-Taxi1-R100', 'trace-2767958-taxi1', 24, 100, 5.0),
                    # R200 (6 actors)
                    _mc('W12-R200', 'constant12', 12, 200, 5.0),
                    _mc('W24-R200', 'constant24', 24, 200, 5.0),
                    _mc('W48-R200', 'constant48', 48, 200, 5.0),
                    _mc('W72-R200', 'constant72', 72, 200, 5.0),
                    _mc('D-ramp48-R200', 'dyn-ramp48-s0', 48, 200, 5.0),
                    _mc('D-mix48-R200', 'v14-mixed_timescale48-train-141480', 48, 200, 5.0),
                ]
                cfg.update(
                    eval_conditions=copy.deepcopy(source['conditions']),
                    conditions=expanded,
                    obs_norm_enabled=True,
                    obs_norm_momentum=0.005,
                    obs_norm_std_floor=0.20,
                    pid_lagrangian=True,
                    dual_kp=1.0,
                    dual_ki=0.05,
                    dual_kd=0.25,
                    peak_window_s=10.0,
                    decision_cost=0.025,
                    joint_effort_delta=0.25,
                    connection_reset_seconds=150.0,
                    physical_seconds_per_actor=1200.0,
                    checkpoint_updates=15000,
                    memsize=400000,
                )
            else:
                # Cleaned, mainstream-weighted 24-actor training suite (v2):
                expanded = [
                    _mc('W12-R20', 'constant12', 12, 20, 5.0),
                    _mc('W12-R100', 'constant12', 12, 100, 5.0),
                    _mc('W18-R50', 'constant18', 18, 50, 5.0),
                    _mc('W24-R20-q2', 'constant24', 24, 20, 2.0),
                    _mc('W24-R200', 'constant24', 24, 200, 5.0),
                    _mc('W36-R50', 'constant36', 36, 50, 5.0),
                    _mc('W36-R100', 'constant36', 36, 100, 5.0),
                    _mc('W42-R20', 'constant42', 42, 20, 5.0),
                    _mc('W48-R20', 'constant48', 48, 20, 5.0),
                    _mc('W48-R100-q2', 'constant48', 48, 100, 2.0),
                    _mc('W48-R200', 'constant48', 48, 200, 5.0),
                    _mc('W72-R20', 'constant72', 72, 20, 5.0),
                    _mc('W72-R200', 'constant72', 72, 200, 5.0),
                    _mc('W96-R50', 'constant96', 96, 50, 5.0),
                    _mc('D-step24-R20', 'dyn-step24-s0', 24, 20, 5.0),
                    _mc('D-ramp48-R50', 'dyn-ramp48-s0', 48, 50, 5.0),
                    _mc('D-burst48-R100', 'dyn-burst48-s0', 48, 100, 5.0),
                    _mc('D-cdrop24-R50', 'v14-compound_drop24-train-140240', 24, 50, 5.0),
                    _mc('D-cdrop48-R100', 'v14-compound_drop48-train-140480', 48, 100, 5.0),
                    _mc('D-mix24-R20', 'v14-mixed_timescale24-train-141240', 24, 20, 3.0),
                    _mc('D-mix48-R200', 'v14-mixed_timescale48-train-141480', 48, 200, 5.0),
                    _mc('C-TMob-R50', 'v12-real-TMobile-LTE-short.down', 24, 50, 5.0),
                    _mc('C-Taxi1-R50', 'trace-2767958-taxi1', 24, 50, 5.0),
                    _mc('C-Vzn-R100', 'v12-real-Verizon-LTE-short.down', 12, 100, 5.0),
                ]
                cfg.update(
                    eval_conditions=copy.deepcopy(source['conditions']),
                    conditions=expanded,
                    obs_norm_enabled=True,
                    obs_norm_momentum=0.005,
                    obs_norm_std_floor=0.20,
                    pid_lagrangian=True,
                    dual_kp=1.0,
                    dual_ki=0.05,
                    dual_kd=0.25,
                    peak_window_s=10.0,
                    decision_cost=0.03,
                    joint_effort_delta=0.25,
                    connection_reset_seconds=120.0,
                    physical_seconds_per_actor=900.0,
                    checkpoint_updates=15000,
                    memsize=400000,
                )
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
