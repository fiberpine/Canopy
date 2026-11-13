"""One genuine TCP connection; Linux I/O is imported only when explicitly run."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from .configuration import ROOT
from .core import Observation, interval_return, period_ms, wire
from .ownership import acquire_namespace_lock,release_namespace_lock,collect_orphans

R = Path(__file__).resolve().parent


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def run_flow(cfg, condition, action_fn, stop, actor=0, episode=0, transition=None,
             physical_limit=None, frozen_seconds=None):
    """action_fn returns the command and diagnostics. It never blocks on updates.

    Transition duration comes from the next REAL native boundary, not Python
    wall time or the commanded sleep. Reconnection never joins two histories.
    """
    if os.name != 'posix': raise RuntimeError('Real TCP requires the registered Linux/kernel environment')
    import sys
    sys.path.insert(0, str(ROOT / 'frameworks/baseline/code/runtime/_deps'))
    import sysv_ipc
    import psutil
    from frameworks.baseline.evaluate import StateReader, EndOfRun, cleanup
    from frameworks.baseline.packet_window_completion import start as packet_watch
    from frameworks.baseline.baseline_admission import cleanup_report
    name = '%s-actor%d-ep%d' % (cfg['run_id'], actor, episode)
    folder = R / 'logs' / name; folder.mkdir(parents=True, exist_ok=False)
    dump(folder / 'config.json', dict(cfg=cfg, condition=condition, actor=actor, episode=episode))
    binary = R / 'native/orca-server-joint-healthy'
    if not binary.is_file() or not os.access(str(binary), os.X_OK):
        raise RuntimeError('Build and chmod the independent joint native before TCP admission')
    from .identity import validate_native
    validate_native()
    b = ROOT / 'frameworks/baseline/code/runtime'
    trace = ROOT / 'traces' / condition['trace']
    source_hashes = dict(native=hashlib.sha256(binary.read_bytes()).hexdigest(), trace=hashlib.sha256(trace.read_bytes()).hexdigest())
    port = cfg['actor_port_base'] + actor
    shared_r = shared_w = server = watcher = None
    owned = {}; resource_rows = []; failure = ''; status = 'RUNNING'; model_calls = 0
    elapsed = 0.; sequence = 0; exploration = 0; last_decision_id = None
    monitor_stop = threading.Event(); flow_stop = threading.Event()
    raw_file = (folder / 'feedback.jsonl').open('x', buffering=262144)
    actions_file = (folder / 'decisions.jsonl').open('x', buffering=262144)
    log_file = (folder / 'server.log').open('x')
    start = time.monotonic()
    def monitor():
        last = 0.
        while not monitor_stop.wait(.05):
            if stop.is_set() or (R / 'logs' / cfg['run_id'] / 'stop.json').exists() or time.time() >= cfg['deadline_epoch'] - cfg.get('stop_reserve_s', 60):
                stop.set(); flow_stop.set()
            try:
                descendants = psutil.Process(server.pid).children(recursive=True)
                for p in descendants: owned[p.pid] = p.create_time()
                if time.monotonic() - last >= 2:
                    records = []
                    for p in [psutil.Process(os.getpid()), psutil.Process(server.pid)] + descendants:
                        try: records.append(dict(pid=p.pid, created=p.create_time(), name=p.name(), cpu=p.cpu_times()._asdict(), rss=p.memory_info().rss))
                        except psutil.Error: pass
                    resource_rows.append(dict(elapsed_s=time.monotonic() - start, processes=records,
                                              machine_cpu=psutil.cpu_times()._asdict()))
                    last = time.monotonic()
            except psutil.Error: pass
    thread = None;namespace_lock=None
    try:
        # Mahimahi chooses unused outer addresses by scanning interfaces without
        # reserving them. Concurrent construction can choose the same pair.
        # Hold a host-local lock through first feedback; flows run concurrently.
        namespace_lock=acquire_namespace_lock(R/'control/namespace-start.lock',stop,cfg['deadline_epoch']-cfg.get('stop_reserve_s',60))
        shared_r = sysv_ipc.SharedMemory(None, sysv_ipc.IPC_CREX, mode=0o600, size=2048)
        shared_w = sysv_ipc.SharedMemory(None, sysv_ipc.IPC_CREX, mode=0o600, size=2048)
        shared_r.write(b'\0' * 2048); shared_w.write(b'99999 99999\0')
        env = dict(os.environ, C3_TRACE_DIR=str(ROOT / 'traces'), ORCA_SHM_R=str(shared_r.key), ORCA_SHM_W=str(shared_w.key),
                   ORCA_CONTROL_LOG=str(folder / 'control.csv'), CANOPY_JOINT_CLOCK=cfg['clock_mode'], CANOPY_JOINT_RATE=cfg['rate_mode'],
                   CANOPY_JOINT_BOUNDARIES=str(folder / 'boundaries.csv'), CANOPY_JOINT_TIMING=str(folder / 'timing.csv'))
        command = [str(binary), str(port), str(b), '20', '4', 'cubic', '0', condition['trace'], 'constant192',
                   str(condition['delay_ms']), str(folder / 'down.log'), '0', str(condition['queue_packets']),
                   '0', 'pretrained', '0', '0', '200000', '0', '0', '0', '1', '0', '0', '0', '0', 'none', 'raw', '0', '-1']
        server = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        server.created = psutil.Process(server.pid).create_time()
        dump(folder / 'ownership.json', dict(pid=server.pid, started=server.created, shm_keys=[shared_r.key, shared_w.key], port=port))
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        if frozen_seconds:
            watcher = packet_watch(folder / 'down.log', frozen_seconds, flow_stop, folder / 'packet-window-completion.json')
        reader = StateReader(shared_r, server, flow_stop)
        observe = Observation(cfg['rate_mode']); planned = 20.
        reader.read(); state, utility, metadata = observe.observe(reader.last_raw, planned)
        raw_file.write(json.dumps(dict(state_id=reader.last_id, raw=reader.last_raw.tolist(), **metadata)) + '\n')
        action, info = action_fn(state); action = np.asarray(action, np.float32)
        shared_w.write(wire(reader.last_id, action)); planned = period_ms(action[1]); model_calls += 1
        last_decision_id = reader.last_id
        exploration += int(info.get('explored', False))
        actions_file.write(json.dumps(dict(time_us=time.time()*1e6, state_id=reader.last_id, action=action.tolist(), planned_ms=planned, **info)) + '\n')
        dump(folder / 'flow-ready.json', dict(epoch=time.time(), first_state=reader.last_id))
        release_namespace_lock(namespace_lock);namespace_lock=None
        while not flow_stop.is_set():
            # Sleep through the commanded interval instead of repeatedly reading
            # unchanged shared memory. Native remains the observation clock;
            # StateReader only waits for its next genuinely new publication.
            if flow_stop.wait(planned / 1000.): break
            try: reader.read()
            except EndOfRun: break
            next_state, utility, metadata = observe.observe(reader.last_raw, planned)
            dt = metadata['actual_dt_s']; elapsed += dt; sequence += 1
            reward, discount = interval_return(utility, dt, cfg['decision_cost'])
            terminal = bool(physical_limit is not None and elapsed >= physical_limit)
            raw_file.write(json.dumps(dict(state_id=reader.last_id, raw=reader.last_raw.tolist(), **metadata)) + '\n')
            if transition:
                transition([state.copy(), action.copy(), np.array([reward], np.float32), next_state.copy(),
                            np.array([float(terminal)], np.float32), np.array([dt], np.float64), np.array([discount], np.float32)],
                           episode, sequence)
            if terminal:
                shared_w.write(wire(reader.last_id, np.array([0., -1.]), apply=False)); break
            action, info = action_fn(next_state); action = np.asarray(action, np.float32)
            shared_w.write(wire(reader.last_id, action)); planned = period_ms(action[1]); model_calls += 1
            last_decision_id = reader.last_id
            exploration += int(info.get('explored', False))
            actions_file.write(json.dumps(dict(time_us=time.time()*1e6, state_id=reader.last_id, action=action.tolist(), planned_ms=planned, **info)) + '\n')
            state = next_state
        if watcher and watcher.completion_state['error']: raise RuntimeError(watcher.completion_state['error'])
        if frozen_seconds and not (folder / 'packet-window-completion.json').exists(): raise RuntimeError('Incomplete frozen packet-clock window')
        if sequence < 5 and not stop.is_set(): raise RuntimeError('Insufficient real control progress')
        status = 'COMPLETED'
    except Exception as error:
        import traceback
        status = 'FAILED'; failure = str(error); (folder / 'failure.txt').write_text(traceback.format_exc())
    finally:
        flow_stop.set(); monitor_stop.set()
        if watcher: watcher.join(2)
        if thread: thread.join(2)
        # Quiesce the last published command before killing its native. This
        # reads the execution log only once at shutdown, never TCP telemetry.
        if server and last_decision_id is not None:
            until = time.monotonic() + .5
            while time.monotonic() < until and server.poll() is None:
                try:
                    with (folder / 'control.csv').open('rb') as tail:
                        tail.seek(0, 2); tail.seek(max(0, tail.tell() - 8192)); lines = tail.read().decode().splitlines()
                    if any(len(x.split(',')) == 8 and x.split(',')[1:3] == [str(last_decision_id)]*2 for x in lines[-3:]): break
                except FileNotFoundError: pass
                time.sleep(.005)
        if server:
            collect_orphans(psutil,owned,folder,b/'client',port,server.created)
            dump(folder/'owned-processes.json',{str(k):v for k,v in owned.items()})
        cleanup(server, owned)
        release_namespace_lock(namespace_lock);namespace_lock=None
        for shm in (shared_r, shared_w):
            if shm:
                try: shm.remove(); shm.detach()
                except sysv_ipc.Error: pass
        raw_file.close(); actions_file.close(); log_file.close()
        audit = cleanup_report(owned, port, [x.key for x in (shared_r, shared_w) if x], sysv_ipc)
        dump(folder / 'cleanup-audit.json', audit); dump(folder / 'resources.json', resource_rows)
        if audit['status'] != 'PASS': status = 'FAILED'; failure = 'Owned resource cleanup failed'
        result = dict(status=status, reason=failure, run_id=name, actor=actor, episode=episode,
                      python_pid=os.getpid(), native_pid=server.pid if server else None, logical_cpus=psutil.cpu_count(),
                      physical_seconds=elapsed, transitions=sequence, policy_calls=model_calls, exploration_calls=exploration,
                      source_sha256=source_hashes, folder=str(folder), wall_s=time.monotonic() - start,
                      reward_is_boundary_approximation=True, property_reward=False)
        dump(folder / 'result.json', result)
    if status != 'COMPLETED': raise RuntimeError('Joint flow failed: ' + failure)
    return result
