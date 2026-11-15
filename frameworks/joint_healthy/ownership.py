"""Scope orphaned emulator children and serialize only namespace construction."""
import os
from pathlib import Path
import re
import time


def sysv_segments():
    """Read kernel object identities without attaching to another flow's memory."""
    rows = Path('/proc/sysvipc/shm').read_text().splitlines()
    if not rows: raise RuntimeError('Missing Linux shared-memory identity table')
    names = rows[0].split()
    return [dict(zip(names, line.split())) for line in rows[1:] if line.strip()]


def cleanup_flow_report(owned, port, shared_objects, timeout_seconds=3.):
    """A recycled key is not a leaked object; only the recorded shmid is owned."""
    from frameworks.baseline.baseline_admission import cleanup_report
    start = time.monotonic(); checks = []
    while True:
        report = cleanup_report(owned, port, [], None, timeout_seconds=0.)
        segments = sysv_segments()
        owned_ids = {int(row['id']) for row in shared_objects}
        keys = {int(row['key']) for row in shared_objects}
        remaining = [dict(key=int(row['key']), id=int(row['shmid'])) for row in segments
                     if int(row['shmid']) in owned_ids]
        reused = [dict(key=int(row['key']), id=int(row['shmid'])) for row in segments
                  if int(row['key']) in keys and int(row['shmid']) not in owned_ids]
        report.pop('checks', None)
        report.update(remaining_shm_objects=remaining, recycled_keys_not_owned=reused,
                      shared_object_identities=shared_objects)
        if remaining: report['status'] = 'FAIL'
        checks.append(dict(elapsed_s=time.monotonic()-start, **report))
        if report['status'] == 'PASS' or time.monotonic()-start >= timeout_seconds:
            return dict(report, checks=checks, teardown_wait_s=time.monotonic()-start)
        time.sleep(min(.05, max(0., timeout_seconds-(time.monotonic()-start))))


def supervise_children(children, drain_seconds=45.):
    """Observe every direct child, including failures behind a slow first child."""
    failed = False; stopping = None
    while any(child.poll() is None for child in children):
        if not failed and any(child.poll() not in (None, 0) for child in children):
            failed = True; stopping = time.monotonic()
            for child in children:
                if child.poll() is None: child.terminate()
        if stopping is not None and time.monotonic()-stopping >= drain_seconds:
            for child in children:
                if child.poll() is None: child.kill()
        time.sleep(.05)
    for child in children: child.wait()
    return not failed and all(child.returncode == 0 for child in children)


def matches_flow(command, folder, client, port):
    command=[str(arg).replace('\\','/') for arg in command]
    folder=str(folder).replace('\\','/');client=str(client).replace('\\','/')
    if any(folder+'/' in str(arg) for arg in command): return True
    pattern=re.escape(client)+r'\s+\S+\s+1\s+'+str(int(port))+r'(?=\s|$)'
    return re.search(pattern,' '.join(command)) is not None


def collect_orphans(psutil, owned, folder, client, port, since):
    for process in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            row=process.info
            if row['name'] not in ('sudo','mm-delay','mm-link','sh','client') or row['create_time']<since-.01: continue
            if not matches_flow(row['cmdline'] or [],folder,client,port): continue
            owned[process.pid]=row['create_time']
            for child in process.children(recursive=True): owned[child.pid]=child.create_time()
        except psutil.Error: pass


def acquire_namespace_lock(path, stop, cutoff):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=os.fdopen(os.open(str(path),os.O_CREAT|os.O_RDWR,0o600),'r+')
    try:
        while not stop.is_set() and time.time()<cutoff:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);return stream
            except BlockingIOError:stop.wait(.05)
        raise RuntimeError('Stopped before exclusive Mahimahi namespace construction')
    except BaseException:
        stream.close();raise


def release_namespace_lock(stream):
    if stream is not None:
        import fcntl
        fcntl.flock(stream,fcntl.LOCK_UN);stream.close()
