"""Resource and budget contracts for opt-in scalar admission; no policy math."""
import math


def validate_ports(base, actors, learner, ephemeral, seconds, episode=None):
    connections = 1 if episode is None else int(math.ceil(seconds / episode)) + 2
    end = base + connections * actors - 1
    low, high = ephemeral
    if actors < 1 or base < 1 or end > 65535 or not 1 <= learner <= 65535:
        raise ValueError('Port allocation is out of range')
    if base <= learner <= end:
        raise ValueError('Actor allocation intersects learner port')
    if not (end < low or base > high) or low <= learner <= high:
        raise ValueError('Explicit listener allocation intersects ephemeral port range')
    return dict(first=base, last=end, learner=learner, maximum_connections=connections)


def _cleanup_snapshot(owned, port, keys, ipc_module):
    import os
    import psutil
    import subprocess
    remaining = []
    for pid, created in owned.items():
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - created) < .01 and process.status() != psutil.STATUS_ZOMBIE:
                remaining.append(dict(pid=pid, created=created))
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            remaining.append(dict(pid=pid, verification_error=type(error).__name__))
    command = subprocess.run(['sudo', '-n', 'ss', '-ltnp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    listeners = [line for line in command.stdout.decode().splitlines()
                 if any(token.rsplit(':', 1)[-1] == str(port) for token in line.split())]
    ipc = []
    for key in keys:
        segment = None
        try:
            segment = ipc_module.SharedMemory(key)
            ipc.append(key)
        except ipc_module.ExistentialError:
            pass
        finally:
            if segment is not None:
                segment.detach()
    return dict(status='PASS' if not remaining and not listeners and not ipc and command.returncode == 0 else 'FAIL',
                remaining=remaining, listeners=listeners, accessible_shm_keys=ipc,
                listener_check_returncode=command.returncode, actor_fd_count=len(os.listdir('/proc/self/fd')))


def cleanup_report(owned, port, keys, ipc_module, timeout_seconds=3.):
    """Confirm eventual teardown; record a bounded wait without killing strangers."""
    import time
    start=time.monotonic();deadline=start+timeout_seconds;checks=[]
    while True:
        result=_cleanup_snapshot(owned,port,keys,ipc_module)
        checks.append(dict(elapsed_s=time.monotonic()-start,**result))
        if result['status']=='PASS' or time.monotonic()>=deadline:
            return dict(result,checks=checks,teardown_wait_s=time.monotonic()-start)
        time.sleep(min(.05,max(0,deadline-time.monotonic())))


def drain_after_failure(actors, collect, forward, finished, timeout_seconds=45.):
    """Stop is already requested; preserve producer records and the learner drain.

    This never changes the failed experiment's status into a successful run.
    """
    import time
    deadline=time.monotonic()+timeout_seconds;forwarded={}
    while True:
        for entry in collect():
            producer=entry.get('producer');index=entry['index']
            if producer is not None and index not in forwarded:
                forward(index,producer);forwarded[index]=producer
        done=bool(finished())
        if done and len(forwarded)==actors:
            return dict(status='DRAIN_RECORDED_RUN_STILL_FAILED',forwarded=len(forwarded),learner_finished=True)
        if time.monotonic()>=deadline:
            return dict(status='DRAIN_INCOMPLETE_RUN_FAILED',forwarded=len(forwarded),learner_finished=done)
        time.sleep(min(.2,max(0,deadline-time.monotonic())))
