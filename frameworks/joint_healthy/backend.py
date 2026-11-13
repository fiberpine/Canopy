"""Explicit future TF1 learner/actor workers; importing this module opens no port."""
import json
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback
import numpy as np
from .configuration import ROOT
from .core import JOINT, calibration_period, period_action
from .flow import dump, run_flow
from .replay import Replay, CalibrationGate

R = Path(__file__).resolve().parent


def setup(cfg, learner):
    import tensorflow as tf
    from .model import Model
    tf.set_random_seed(cfg['seed'])
    spec = tf.train.ClusterSpec({'learner': [cfg['learner_endpoint']]})
    options = tf.ConfigProto(intra_op_parallelism_threads=cfg.get('tf_threads', 4), inter_op_parallelism_threads=cfg.get('tf_threads', 4), operation_timeout_in_ms=10000)
    server = tf.train.Server(spec, job_name='learner', task_index=0, config=options) if learner else None
    with tf.device('/job:learner/task:0/cpu:0'):
        model = Model(cfg, training=True)
        dtypes = [tf.float32] * 7 + [tf.int32, tf.int64, tf.int64]
        dtypes[5] = tf.float64
        shapes = [[82], [2], [1], [82], [1], [1], [1], [], [], []]
        fifo = tf.FIFOQueue(10000, dtypes, shapes=shapes, shared_name='healthy_joint_replay')
        placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
        enqueue = fifo.enqueue(placeholders)
        number = tf.placeholder(tf.int32, [])
        dequeue = fifo.dequeue_many(number)
        size = fifo.size()
    session = tf.Session(server.target if server else 'grpc://' + cfg['learner_endpoint'], config=options)
    return tf, model, session, placeholders, enqueue, number, dequeue, size


def run_learner(cfg):
    folder = R / 'logs' / cfg['run_id']; folder.mkdir(parents=True, exist_ok=False)
    cp = R / 'checkpoints' / cfg['run_id']; cp.mkdir(parents=True, exist_ok=False)
    dump(folder / 'config.json', cfg)
    tf, model, sess, ph, enq, number, deq, size = setup(cfg, True)
    sess.run(tf.global_variables_initializer())
    provenance = model.initialize_baseline(sess, cfg['baseline_checkpoint'])
    dump(folder / 'initialization.json', provenance)
    replay = Replay(cfg); gate = CalibrationGate(cfg); lock = threading.RLock()
    done = {}; errors = []; stop_receiver = threading.Event()
    frozen_hash = model.digest(sess, True)
    counters = dict(critic_iterations=0, critic_optimizer_ops=0, actor_updates=0, target_updates=0)
    saver = tf.train.Saver(max_to_keep=None)
    def checkpoint(kind):
        with lock:
            prefix = str(cp / ('step-%d' % replay.credit.inserted))
            if Path(prefix + '.index').exists(): prefix += '-' + kind
            saver.save(sess, prefix, write_meta_graph=kind == 'initial')
            model.save_manifest(prefix, cfg, dict(counters, phase=gate.phase, real_experience=replay.credit.record(), kind=kind))
            dump(cp / 'selected-endpoint.json', dict(prefix=prefix, kind=kind, phase=gate.phase))
            return prefix
    initial = checkpoint('initial')
    def receive():
        try:
            while not stop_receiver.is_set():
                count = int(sess.run(size))
                if not count: stop_receiver.wait(.01); continue
                columns = sess.run(deq, {number: min(count, 256)})
                with lock:
                    for row in zip(*columns):
                        actor, episode, sequence = int(row[7]), int(row[8]), int(row[9])
                        if not 0 <= actor < cfg['num_actors']: raise ValueError('Invalid producer identity')
                        if episode == -1:
                            if actor in done: raise ValueError('Duplicate actor drain marker')
                            if sequence != replay.credit.counts[actor]: raise ValueError('Actor enqueued/inserted mismatch')
                            done[actor] = sequence
                        else:
                            if actor in done: raise ValueError('Experience after producer finished')
                            replay.append(row[:7], actor, episode, sequence)
        except Exception:
            errors.append(traceback.format_exc()); stop_receiver.set()
    receiver = threading.Thread(target=receive, daemon=True); receiver.start()
    dump(folder / 'ready.json', dict(epoch=time.time(), initial_checkpoint=initial))
    training = (folder / 'training.jsonl').open('x', buffering=65536)
    last_report = 0.; last_gate = -1; next_checkpoint = cfg.get('checkpoint_updates', 10000)
    reason = ''; status = 'RUNNING'; latest_gradients = [0., 0.]; stop_seen = None; endpoint = initial
    try:
        while True:
            if errors: raise RuntimeError(errors[0])
            if time.time() >= cfg['deadline_epoch'] - 15: raise RuntimeError('Checkpoint/drain deadline reached')
            with lock: limit = replay.credit.limit; all_done = len(done) == cfg['num_actors']
            if (folder / 'stop.json').exists():
                stop_seen = stop_seen or time.monotonic()
                if not all_done and time.monotonic() - stop_seen > 45: raise RuntimeError('Bounded stop/drain did not receive every producer marker')
            iteration = counters['critic_iterations']
            if all_done and iteration >= limit: break
            if time.monotonic() - last_report >= 5:
                with lock: progress = dict(replay.credit.record(), counters=dict(counters), phase=gate.phase,
                                          producers_done=len(done), sampled_by_actor=replay.sampled.tolist(),
                                          period_coverage=replay.coverage.tolist(), fifo_backlog=int(sess.run(size)),
                                          last_actor_gradients=list(map(float, latest_gradients)))
                training.write(json.dumps(progress) + '\n'); training.flush(); dump(folder / 'progress.json', progress)
                last_report = time.monotonic()
            if iteration >= limit: time.sleep(.01); continue
            with lock: batch = replay.sample()
            feed = model.feed(batch)
            losses = sess.run([model.critic_step, model.losses], feed)[1]
            if not np.isfinite(losses).all(): raise RuntimeError('Nonfinite critic objective')
            if gate.phase == JOINT:
                latest_gradients = sess.run([model.window_grad, model.time_grad], feed)
                if not np.isfinite(latest_gradients).all(): raise RuntimeError('Nonfinite actor gradient')
                sess.run([model.actor_step, model.window_update_ops], feed)
                counters['actor_updates'] += 1
            sess.run(model.target_update)
            counters['critic_iterations'] += 1; counters['critic_optimizer_ops'] += 2; counters['target_updates'] += 1
            iteration = counters['critic_iterations']
            if gate.phase < JOINT and iteration % cfg.get('calibration_check_updates', 500) == 0 and iteration != last_gate:
                assert model.digest(sess, True) == frozen_hash, 'Frozen window/BN changed during calibration'
                reports = []; short_reports = []
                with lock:
                    batches = [replay.holdout_batch(i) for i in range(5)]; coverage = replay.coverage.copy()
                    short_batches = [replay.short_return_batch(i) for i in range(5)]
                for period, heldout in enumerate(batches):
                    if heldout is None: continue
                    q1, q2, target = sess.run([model.q1, model.q2, model.label], model.feed(heldout))
                    scale = max(1., float(np.sqrt(np.mean(target ** 2))))
                    errors_q = [float(np.sqrt(np.mean((q - target) ** 2)) / scale) for q in (q1,q2)]
                    reports.append(dict(period=period, rows=len(q1), normalized_td_rmse=max(errors_q), twin_errors=errors_q))
                for period, block in enumerate(short_batches):
                    if block is None: continue
                    first, last, rewards, continuation = block
                    q1, q2 = sess.run([model.q1, model.q2], model.feed(first))
                    tail = sess.run(model.target_q, model.feed(last))
                    target = rewards + continuation*tail
                    scale = max(1., float(np.sqrt(np.mean(target**2))))
                    errors_q = [float(np.sqrt(np.mean((q-target)**2))/scale) for q in (q1,q2)]
                    short_reports.append(dict(period=period, blocks=len(q1), normalized_return_rmse=max(errors_q), twin_errors=errors_q,
                                              bootstrap_tail_not_full_MC=True, target_mean=float(np.mean(target))))
                decision = gate.assess(iteration, coverage, reports, short_reports)
                if gate.phase == JOINT and not cfg.get('release_joint', False):
                    gate.phase = decision['phase']; decision['admitted'] = False; decision['reasons'].append('joint_release_not_registered')
                sess.run(tf.assign(model.phase, gate.phase))
                dump(folder / 'calibration-history.json', gate.history); last_gate = iteration
            if iteration >= next_checkpoint:
                checkpoint('live'); next_checkpoint += cfg.get('checkpoint_updates', 10000)
        stop_receiver.set(); receiver.join(10)
        if receiver.is_alive(): raise RuntimeError('Receiver did not finish')
        if errors: raise RuntimeError(errors[0])
        assert sum(done.values()) == replay.credit.inserted
        endpoint = checkpoint('final')
        with lock: np.savez(str(cp / 'replay.npz'), **replay.snapshot())
        complete_exposure = bool(np.all(replay.credit.actor_seconds >= cfg['physical_seconds_per_actor'] - .001))
        status = ('COMPLETED' if complete_exposure else 'PARTIAL_EXPOSURE_NOT_COMPLETED') if counters['actor_updates'] > 0 else 'PARTIAL_CALIBRATION_ONLY'
    except Exception as error:
        status = 'FAILED'; reason = str(error); dump(folder / 'failure.json', dict(reason=reason, traceback=traceback.format_exc()))
        try:
            endpoint = checkpoint('stopped')
            with lock: np.savez(str(cp / 'replay.npz'), **replay.snapshot())
        except Exception:
            dump(folder / 'checkpoint-failure.json', dict(traceback=traceback.format_exc(), original_reason=reason))
    finally:
        stop_receiver.set(); receiver.join(3); training.close()
        result = dict(status=status, reason=reason, counters=counters, phase=gate.phase,
                      credit=replay.credit.record(), producer_done=done, endpoint=endpoint,
                      update_debt=max(0, replay.credit.limit - counters['critic_iterations']),
                      no_complete_resume=True, performance_verified=False)
        dump(folder / 'result.json', result); dump(folder / 'finished.json', dict(status=status))
    return result


def run_actor(cfg, index):
    folder = R / 'logs' / ('%s-producer%d' % (cfg['run_id'], index)); folder.mkdir(parents=True, exist_ok=False)
    tf, model, sess, ph, enq, number, deq, size = setup(cfg, False)
    local = queue.Queue(maxsize=4096); stop = threading.Event(); finished = threading.Event()
    counts = dict(generated=0, enqueued=0, policy_calls=0, exploration_calls=0)
    errors = []; rng = np.random.RandomState(cfg['seed'] + index * 1009)
    signal.signal(signal.SIGTERM, lambda *_: stop.set()); signal.signal(signal.SIGINT, lambda *_: stop.set())
    def writer():
        try:
            while not finished.is_set() or not local.empty():
                try: values, episode, sequence = local.get(timeout=.05)
                except queue.Empty: continue
                values = values + [np.int32(index), np.int64(episode), np.int64(sequence)]
                sess.run(enq, dict(zip(ph, values)), options=tf.RunOptions(timeout_in_ms=10000)); counts['enqueued'] += 1
        except Exception:
            errors.append(traceback.format_exc()); stop.set()
    writing = threading.Thread(target=writer, daemon=True); writing.start()
    def action(state):
        value, phase = sess.run([model.action, model.phase], {model.s: state[None, :]}, options=tf.RunOptions(timeout_in_ms=10000))
        value = value[0]; pure = value.copy(); explored = False
        if phase < JOINT:
            # The window and its statistics remain frozen and unperturbed.
            ms, explored = calibration_period(cfg, int(phase), rng)
            value[1] = period_action(ms)
        else:
            noise = rng.normal(0., cfg['exploration_std'], 2)
            if cfg.get('fixed_mtp_ms') is not None: noise[1] = 0.
            value = np.clip(value + noise, -1., 1.); explored = bool(np.any(noise))
        counts['policy_calls'] += 1; counts['exploration_calls'] += int(explored)
        return value, dict(phase=int(phase), pure_action=pure.tolist(), explored=explored)
    def transition(values, episode, sequence):
        if errors: raise RuntimeError(errors[0])
        local.put_nowait((values, episode, sequence)); counts['generated'] += 1
    exposure = 0.; episode = 0; runs = []
    try:
        while not stop.is_set() and exposure < cfg['physical_seconds_per_actor']:
            if (R / 'logs' / cfg['run_id'] / 'stop.json').exists() or time.time() >= cfg['deadline_epoch'] - cfg['stop_reserve_s']: break
            seconds = cfg['physical_seconds_per_actor'] - exposure
            if cfg.get('connection_reset_seconds'): seconds = min(seconds, cfg['connection_reset_seconds'])
            result = run_flow(cfg, cfg['conditions'][index], action, stop, index, episode, transition, physical_limit=seconds)
            exposure += result['physical_seconds']; runs.append(result); episode += 1
        finished.set(); writing.join(30)
        if writing.is_alive(): raise RuntimeError('Actor FIFO writer drain timeout')
        if errors: raise RuntimeError(errors[0])
        assert counts['generated'] == counts['enqueued']
        # A FIFO control record follows every real row. It is never put in replay
        # and therefore earns neither exposure credit nor an optimizer sample.
        sentinel = [np.zeros(n, np.float64 if i == 5 else np.float32) for i, n in enumerate((82, 2, 1, 82, 1, 1, 1))]
        sess.run(enq, dict(zip(ph, sentinel + [np.int32(index), np.int64(-1), np.int64(counts['enqueued'])])),
                 options=tf.RunOptions(timeout_in_ms=10000))
    except Exception:
        errors.append(traceback.format_exc()); stop.set(); finished.set(); writing.join(3)
    result = dict(index=index, status='COMPLETED' if not errors else 'FAILED', errors=errors,
                  counts=counts, physical_seconds=exposure, connections=len(runs), runs=runs)
    dump(folder / 'producer.json', result)
    return result


def worker_main():
    import argparse
    import sys
    ap = argparse.ArgumentParser(); ap.add_argument('role', choices=['learner', 'actor', 'actor-host', 'stop'])
    ap.add_argument('--config'); ap.add_argument('--index', type=int, default=0); ap.add_argument('--indices'); ap.add_argument('--run-id')
    args = ap.parse_args()
    if args.role == 'stop':
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', args.run_id or ''): raise ValueError('Invalid run ID')
        dump(R / 'logs' / args.run_id / 'stop.json', dict(reason='controller_requested_stop', epoch=time.time())); return
    cfg = json.load(sys.stdin) if args.config == '-' else json.loads(Path(args.config).read_text())
    from .core import validate
    validate(cfg)
    if time.time() >= cfg['deadline_epoch'] - 120: raise ValueError('Worker deadline expired')
    from .configuration import runtime_preflight
    cfg['local_runtime_preflight'] = runtime_preflight()
    if args.role == 'actor-host':
        import subprocess
        indices = [int(x) for x in args.indices.split(',')]
        folder = R / 'logs' / (cfg['run_id'] + '-host-' + '-'.join(map(str, indices)))
        folder.mkdir(parents=True, exist_ok=False); path = folder / 'config.json'; dump(path, cfg)
        children = []
        for index in indices:
            stream = (folder / ('actor%d.stdout' % index)).open('x')
            child = subprocess.Popen([sys.executable, str(ROOT / 'joint_healthy.py'), '_worker', 'actor', '--config', str(path), '--index', str(index)],
                                     stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, stream, index))
        for child, stream, index in children:
            child.wait(); stream.close()
        rows = [json.loads((R / 'logs' / ('%s-producer%d' % (cfg['run_id'], index)) / 'producer.json').read_text()) for _, _, index in children]
        print(json.dumps(dict(status='COMPLETED' if all(x['status'] == 'COMPLETED' for x in rows) else 'FAILED', actors=rows)), flush=True)
        return
    result = run_learner(cfg) if args.role == 'learner' else run_actor(cfg, args.index)
    print(json.dumps(result), flush=True)
    # TF1 shared-server session shutdown can otherwise hang after a peer exits.
    # All files, flow processes, IPC and producer queues were already closed.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0 if result['status'] == 'COMPLETED' else 2)
