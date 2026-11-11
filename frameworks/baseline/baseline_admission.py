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


def cleanup_report(owned, port, keys, ipc_module):
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
    command = subprocess.run(['sudo', '-n', 'ss', '-ltn'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
