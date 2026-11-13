"""Scope orphaned emulator children and serialize only namespace construction."""
import os
from pathlib import Path
import re
import time


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
