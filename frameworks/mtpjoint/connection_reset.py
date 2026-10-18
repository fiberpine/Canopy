"""Reset TCP episodes without rebuilding TF, replay, exploration, or the writer.

Install before timed_distributed wraps evaluate.main. Each episode invokes the
existing evaluator's complete owned-flow lifecycle, including fresh IPC, a new
TCP wrapper, zero history, and removal of the preceding episode's flow tree.
Artificial time limits keep terminal=0 and bootstrap from the last real next
state. No replay row links the last episode to the next connection's history.
"""
import csv
import json
import pathlib
import sys
import time


def install(evaluate, cfg, index):
    root = evaluate.R
    original_main = evaluate.main
    interval = float(cfg['connection_reset_seconds'])
    assert interval >= 30, 'Very short resets need a separate capacity experiment'
    assert cfg.get('fixed_action') is None or cfg.get('capacity_only')

    def episodes():
        args_before = list(sys.argv)
        config_path = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])
        actor_cfg = json.loads(config_path.read_text())
        actor_dir = root / 'logs' / actor_cfg['run_id']
        actor_dir.mkdir(exist_ok=False)
        (actor_dir / 'config.json').write_text(json.dumps(actor_cfg, indent=2))
        producer_finish = evaluate.producer_finish
        evaluate.producer_finish = None
        started = time.monotonic()
        results = []
        valid = 0
        observations = 0
        physical_seconds = 0.
        status = 'FAILED'
        reason = 'Episode dispatcher did not complete'
        try:
            while valid < actor_cfg['training_steps']:
                if time.time()>=actor_cfg.get('absolute_deadline_epoch',float('inf')):break
                if actor_cfg.get('physical_seconds_per_actor') and physical_seconds+1e-9>=actor_cfg['physical_seconds_per_actor']:break
                if actor_cfg.get('stop_file') and pathlib.Path(actor_cfg['stop_file']).exists():
                    break
                number = len(results)
                episode_id = actor_cfg['run_id'] + '-episode%04d' % number
                episode_cfg = dict(actor_cfg)
                environments=actor_cfg.get('episode_environments')
                if environments:
                    environment=environments[number%len(environments)]
                    assert environment['condition']==actor_cfg['condition'],'One registered role per actor'
                    import hashlib
                    assert hashlib.sha256((root/'configs/traces'/environment['trace']).read_bytes()).hexdigest()==environment['trace_sha256']
                    episode_cfg.update(environment)
                episode_cfg.update(run_id=episode_id,
                    training_steps=actor_cfg['training_steps'] - valid,
                    port=actor_cfg['port'] + number * cfg['num_actors'],
                    episode_control_seconds=interval + (cfg.get('connection_reset_stagger_seconds', 5) * index if number == 0 else 0),
                    episode_number=number, actor_run_id=actor_cfg['run_id'])
                if actor_cfg.get('joint_control') and actor_cfg.get('physical_seconds_per_actor'):
                    episode_cfg['physical_seconds_per_actor']=actor_cfg['physical_seconds_per_actor']-physical_seconds
                assert episode_cfg['port'] < 65536, 'Exhausted reserved episode ports'
                ephemeral = pathlib.Path('/proc/sys/net/ipv4/ip_local_port_range')
                if ephemeral.exists():
                    low,high=map(int,ephemeral.read_text().split())
                    assert not low<=episode_cfg['port']<=high, 'Listening port overlaps OS ephemeral range'
                path = root / 'configs' / (episode_id + '.json')
                assert not path.exists(), 'Never overwrite an episode'
                path.write_text(json.dumps(episode_cfg, indent=2))
                # Status uses an explicit pointer rather than file symlinks.
                # Summaries exclude inter-episode gaps from control intervals.
                (actor_dir / 'active-episode.json').write_text(json.dumps(dict(
                    episode=number, run_id=episode_id, epoch=time.time())))
                sys.argv = ['evaluate.py', '--config', str(path)]
                try:
                    original_main()
                finally:
                    result_path = root / 'logs' / episode_id / 'result.json'
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        results.append(result)
                        observations += result['counters']['valid']
                        if actor_cfg.get('joint_control'):
                            from joint_mtp import summarize_decisions
                            summary=summarize_decisions(root/'logs'/episode_id/'decisions.csv')
                            valid+=summary['decisions'];physical_seconds+=summary.get('physical_exposure_s',0.)
                        else:valid+=result['counters']['valid']
                        (actor_dir / 'episodes.json').write_text(json.dumps(results, indent=2))
                assert results and results[-1]['cfg']['run_id'] == episode_id
                assert results[-1]['status'] == 'COMPLETED', results[-1]['reason']
                requested_stop=actor_cfg.get('stop_file') and pathlib.Path(actor_cfg['stop_file']).exists()
                assert results[-1]['counters']['valid'] >= 5 or requested_stop
            status = 'COMPLETED'
            reason = ''
        except BaseException as error:
            reason = '%s: %s' % (type(error).__name__, error)
            raise
        finally:
            sys.argv = args_before
            evaluate.producer_finish = producer_finish
            result = dict(cfg=actor_cfg, status=status, reason=reason,
                counters=dict(valid=observations,decisions=valid), physical_seconds=physical_seconds,wall_s=time.monotonic() - started,
                episodes=len(results), terminal_semantics='Time truncation: terminal=0; bootstrap last real next state; never link episodes')
            (actor_dir / 'result.json').write_text(json.dumps(result, indent=2))
            # Only the complete actor stops and drains the writer. Resetting one
            # TCP connection must not stop it, reinitialize TF, or reset noise.
            producer_finish()

    evaluate.main = episodes


def actor_rows(directory, filename):
    """Read original rows without fabricating samples across connection gaps."""
    episodes_path = directory / 'episodes.json'
    if not episodes_path.exists():
        paths = [directory / filename]
    else:
        episodes = json.loads(episodes_path.read_text())
        paths = [directory.parent / x['cfg']['run_id'] / filename for x in episodes]
    groups = []
    for path in paths:
        with path.open() as source:
            groups.append(list(csv.DictReader(source)))
    return groups
