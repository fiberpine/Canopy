#!/usr/bin/env python3
"""Explicit finite V31 calibration commands. Default plan has no network effects."""
import argparse
import json
import os
import pathlib
import sys
import time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as portable


def prepare(arm, run_id, seconds, actors, cluster, smoke=False, release=False, checkpoint=None):
    framework, cfg = portable.template('mtpjoint', 'train')
    portable.allocate(framework, run_id)
    source = checkpoint or str(ROOT/'models/mtpjoint/model')
    if not pathlib.Path(source+'.index').is_file():
        raise ValueError('Missing healthy V31 checkpoint: '+source)
    if not 0 < seconds <= 300:
        raise ValueError('This diagnostic permits at most300 seconds/actor')
    if arm not in ('control', 'candidate') or (arm == 'control' and release):
        raise ValueError('Explicit supported pair arm; --release-actor is candidate-only')
    authority, auth_path = portable.auth(framework, run_id, cluster, int(seconds)+1000)
    if not smoke and (actors != 24 or len(authority['nodes']) != 5):
        raise ValueError('Paired diagnostic uses1 learner+24 actors on the registered5 nodes')
    hosts = [x['ip'] for x in authority['nodes'][1:]]
    cfg['conditions'] = [dict(cfg['conditions'][i % len(cfg['conditions'])]) for i in range(actors)]
    cfg.update(run_id=run_id, model_id=run_id, num_actors=actors,
        actor_hosts=[i % len(hosts) for i in range(actors)],
        learner_endpoint=authority['nodes'][0]['ip']+':31200', actor_port_base=23200,
        compatible_v31_checkpoint=source, matching_clock_prior_checkpoint=None,
        initialize_joint_from_scratch=False,
        actor_update_mode_v37='legacy_count_gate' if arm == 'control' else 'quality_gate',
        actor_release_after_calibration_v37=bool(release),
        calibration_probe_seconds_v37=150., calibration_holdout_rows=1024,
        physical_seconds_per_actor=float(seconds), sample_budget=None,
        absolute_deadline_epoch=authority['deadline_epoch']-60,
        timeout_s=int(seconds)+900, shutdown_barrier_s=180, evaluation_reserve_s=120,
        stop_file=str(framework/'logs'/(run_id+'-stop.json')),
        code_commit='v31-compatible-calibration', model_source='Healthy full149-state V31; fresh optimizer/replay/counts/TCP; not uninterrupted resume',
        checkpoint_interval=1000000, capacity_only=False)
    cfg.update(portable.read(ROOT/'configs/v31-calibration-gates.json'))
    if smoke:
        cfg.update(portable_smoke_training=True, warmup=512, minimum_enqueued_per_actor=512,
            warmup_min_physical_seconds=0, timing_critic_calibration_updates=1,
            timing_min_retained_per_period=1, period_bank_rows_per_bin=16,
            memsize=10000, pretrained_window_health_guard=False,
            timing_action_health_guard=False,
            configuration_notes='Packaging/freeze smoke only. Reduced warmup/coverage; no scientific gate validation or performance claim.')
    return framework, cfg, authority, auth_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'train', 'smoke', 'evaluate'])
    parser.add_argument('--arm', choices=['control', 'candidate'], default='candidate')
    parser.add_argument('--cluster')
    parser.add_argument('--run-id')
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--checkpoint')
    parser.add_argument('--scenario', type=int, default=0)
    parser.add_argument('--release-actor', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(dict(status='PLAN_ONLY_NO_TCP', pair=dict(control='Count+coverage gate', candidate='Actor frozen; explicit release AND necessary consistency gates'),
            source=str(ROOT/'models/mtpjoint/model'), actors=24, hosts=5,
            seconds_per_actor=300, warmup_physical_s=3600,
            theoretical_max_complete_critic_updates_240s=8640,
            theoretical_max_complete_critic_updates_300s=14400,
            count_gate_updates=5000, release_default=False,
            limits='Credits are upper bounds; coverage and learner throughput can reduce actual updates. No seed or publication claim.'), indent=2))
        return
    if os.name == 'nt':
        raise RuntimeError('TCP commands require the preserved Linux/TF1/custom-kernel package')
    if args.command == 'evaluate':
        framework, cfg = portable.template('mtpjoint', 'eval')
        conditions = cfg.pop('conditions')
        if not 0 <= args.scenario < len(conditions):
            raise ValueError('Scenario index outside the registered four conditions')
        run_id = args.run_id or portable.runid('calibration-frozen')
        portable.allocate(framework, run_id)
        cfg.update(conditions[args.scenario])
        if args.checkpoint:
            cfg['checkpoint'] = args.checkpoint
        cfg.update(run_id=run_id, group=args.arm, seconds=60, port=24500,
            no_training=True, absolute_deadline_epoch=time.time()+300,
            timeout_s=180, code_commit='v31-compatible-calibration',
            evaluation_code_commit='v31-compatible-calibration')
        path = framework/'configs'/(run_id+'.json')
        portable.write(path, cfg)
        portable.call([framework/'repair_evaluate.py', '--config', path], cwd=str(framework))
        return
    if not args.cluster:
        raise ValueError('--cluster must name the private registered cluster JSON; no implicit hosts')
    smoke = args.command == 'smoke'
    run_id = args.run_id or portable.runid('calibration-'+args.arm)
    framework, cfg, authority, auth_path = prepare(args.arm, run_id,
        min(args.seconds, 90) if smoke else args.seconds, 1 if smoke else 24,
        args.cluster, smoke, args.release_actor, args.checkpoint)
    input_path = framework/'configs'/(run_id+'-input.json')
    portable.write(input_path, cfg)
    env = dict(os.environ, ORCA_AUTHORIZATION_FILE=str(auth_path))
    if authority.get('ssh_key_path'):
        env['ORCA_SSH_KEY'] = authority['ssh_key_path']
    if authority.get('known_hosts_path'):
        env['ORCA_KNOWN_HOSTS'] = authority['known_hosts_path']
    code = 'import sys,json,pathlib;sys.path.insert(0,sys.argv[1]);from cluster_control import execute;print(json.dumps(execute(json.loads(pathlib.Path(sys.argv[2]).read_text()))))'
    portable.call(['-B', '-c', code, framework, input_path], cwd=str(framework), env=env)


if __name__ == '__main__':
    main()
