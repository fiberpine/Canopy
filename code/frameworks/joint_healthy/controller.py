"""Single learner, bounded actor hosts. Every network invocation is explicit."""
import json
import concurrent.futures
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from .configuration import ROOT
from .flow import R, dump


def execute(cfg, authorization):
    if os.name != 'posix': raise RuntimeError('Cloud controller requires the prepared Linux environment')
    hosts = authorization['nodes']
    if len(hosts) < 2 or cfg['num_actors'] > 24: raise ValueError('This recipe supports one learner and <=24 actors')
    for host in hosts:
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host['ip']): raise ValueError('Invalid node address')
    key = Path(authorization['ssh_key_path']).resolve()
    known = Path(authorization['known_hosts_path']).resolve()
    if not key.is_file() or not known.is_file(): raise ValueError('Provide verified SSH key and known_hosts files')
    endpoint_ip=authorization.get('learner_endpoint_ip',hosts[0]['ip'])
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+',endpoint_ip): raise ValueError('Invalid learner endpoint')
    cfg = dict(cfg, learner_endpoint=endpoint_ip + ':' + str(authorization.get('learner_port', 25000)),
               deadline_epoch=authorization['deadline_epoch'])
    folder = R / 'control' / cfg['run_id']; folder.mkdir(parents=True, exist_ok=False)
    config = folder / 'config.json'; dump(config, cfg)
    worker = ROOT / 'joint_healthy.py'
    ssh = ['ssh', '-i', str(key), '-o', 'UserKnownHostsFile=' + str(known), '-o', 'StrictHostKeyChecking=yes',
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
    processes = []; streams = []; succeeded = False
    def remote(host, args):
        return ssh + ['ubuntu@' + host['ip'], ' '.join(shlex.quote(x) for x in [sys.executable, str(worker), '_worker'] + args)]
    try:
        stream = (folder / 'learner.stdout').open('x'); streams.append(stream)
        learner = subprocess.Popen([sys.executable, str(worker), '_worker', 'learner', '--config', str(config)],
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(learner)
        until = min(time.time() + 120, cfg['deadline_epoch'] - 120)
        while not (R / 'logs' / cfg['run_id'] / 'ready.json').exists():
            if learner.poll() is not None or time.time() >= until: raise RuntimeError('Learner initialization failed/timed out')
            time.sleep(.2)
        for node, host in enumerate(hosts[1:]):
            indices = [str(i) for i in range(cfg['num_actors']) if i % (len(hosts) - 1) == node]
            if not indices: continue
            stream = (folder / ('host%d.stdout' % node)).open('x'); streams.append(stream)
            proc = subprocess.Popen(remote(host, ['actor-host', '--config', '-', '--indices', ','.join(indices)]),
                                    stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(proc); proc.stdin.write(json.dumps(cfg).encode()); proc.stdin.close()
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes): raise RuntimeError('A learner/actor host failed; see retained worker log')
            if time.time() >= cfg['deadline_epoch'] - 10: raise RuntimeError('Final controller deadline reached')
            time.sleep(.5)
        result = json.loads((R / 'logs' / cfg['run_id'] / 'result.json').read_text())
        if any(p.returncode != 0 for p in processes): raise RuntimeError('Worker stopped without completed joint training: ' + result['status'])
        succeeded = True; dump(folder / 'result.json', result); return result
    finally:
        if not succeeded:
            dump(R / 'logs' / cfg['run_id'] / 'stop.json', dict(reason='controller_stopped', epoch=time.time()))
            # Stop files are scoped to this candidate/run, never to baseline trees.
            def stop_host(host):
                remaining = cfg['deadline_epoch'] - time.time() - 2
                if remaining <= 0: return
                try: subprocess.run(remote(host, ['stop', '--run-id', cfg['run_id']]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=min(8., remaining))
                except (OSError, subprocess.TimeoutExpired): pass
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)-1) as pool:
                list(pool.map(stop_host, hosts[1:]))
            until = min(time.time() + 45, cfg['deadline_epoch'] - 3)
            for proc in processes:
                try: proc.wait(timeout=max(.1, until - time.time()))
                except subprocess.TimeoutExpired: proc.terminate()
            dump(folder / 'interrupted.json', dict(status='STOP_REQUESTED_REVIEW_WORKER_RESULTS', epoch=time.time(), no_automatic_resume=True))
        for stream in streams: stream.close()
