"""One authorized45-minute batch. Strict hosts; chunked small transfers only."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import time
import uuid

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = '/home/experiment/orca-baseline-target-audit-20261111-2222'
SOURCE = '/home/experiment/orca-healthy-base-20261110-1794277977/framework'
LOCAL_CODE = WORKSPACE / 'innovations/baseline-target-audit-20261111'
FLEET = json.loads((WORKSPACE / 'experiments/common-period-cluster-20261102/fleet-identity.json').read_bytes())
spec = importlib.util.spec_from_file_location('current_batch_ssh_library', WORKSPACE / 'experiments/common-period-cluster-20261102/remote.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def registration():
    return json.loads((HERE / 'registration.json').read_bytes())


def connect(node):
    cfg = remote.paramiko.SSHConfig()
    with (Path.home() / '.ssh/config').open() as stream:
        cfg.parse(stream)
    entry = cfg.lookup(FLEET[node]['host'])
    client = remote.paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(remote.paramiko.RejectPolicy())
    client.connect(entry['hostname'], username=entry.get('user', 'ubuntu'), key_filename=entry['identityfile'][0],
                   allow_agent=False, look_for_keys=False, timeout=8, banner_timeout=8, auth_timeout=8)
    return client


def mkdirs(sftp, path):
    current = ''
    for part in path.split('/'):
        if not part:
            continue
        current += '/' + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def put_chunks(sftp, path, data):
    mkdirs(sftp, path.rsplit('/', 1)[0])
    with sftp.open(path, 'wx') as stream:
        for offset in range(0, len(data), 512):
            stream.write(data[offset:offset + 512])
        stream.flush()
    assert sftp.stat(path).st_size == len(data)


def rpc(node, source, timeout=60, **values):
    reg = registration()
    if time.time() >= reg['deadline_epoch']:
        raise RuntimeError('Current batch deadline reached')
    body = ''.join('%s=%r\n' % item for item in dict(NODE=node, ROOT=ROOT, SOURCE=SOURCE,
        START=reg['start_epoch'], DEADLINE=reg['deadline_epoch'], CUTOFF=reg['task_cutoff_epoch'], **values).items()) + source
    client = connect(node)
    try:
        path = ROOT + '/controller-tasks/' + uuid.uuid4().hex + '.py'
        with client.open_sftp() as sftp:
            put_chunks(sftp, path, body.encode())
        stdin, stdout, stderr = client.exec_command('/home/experiment/venv/bin/python -B ' + path, timeout=timeout)
        stdin.channel.shutdown_write()
        output = stdout.read().decode(errors='replace')
        error = stderr.read().decode(errors='replace')
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if code:
        save('error-node%d-%d.json' % (node, int(time.time())), dict(code=code, output=output[-8000:], error=error[-8000:]))
        raise RuntimeError('node%d failed: %s' % (node, error[-2000:] or output[-2000:]))
    return json.loads(output)


def parallel(fn, nodes=range(5)):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(fn, nodes))


PREFLIGHT = r'''
import hashlib,json,pathlib,os,time
root=pathlib.Path(ROOT);source=pathlib.Path(SOURCE)
assert os.uname().release=='4.13.1-0521-learner'
assert source.is_dir() and pathlib.Path('/dev/net/tun').exists()
protected=pathlib.Path('/home/experiment/orca-fixed-scratch-20261024')
manifest=json.loads((protected/'audit/protected-models-before.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(manifest)==51 and all(sha(protected/name)==expected for name,expected in manifest.items())
sysctl={name:pathlib.Path('/proc/sys',name).read_text().strip() for name in ('net/ipv4/ip_local_port_range','net/ipv4/tcp_rmem','net/ipv4/tcp_wmem','net/core/rmem_max','net/core/wmem_max')}
core={name:sha(source/name) for name in ('code/canopy/rl-module/agent_v2.py','code/canopy/rl-module/envwrapper_v2.py','code/canopy/rl-module/utils_v2.py','code/runtime/orca-server-control-log-safe','code/runtime/client')}
cp=source/'checkpoints/healthy-base-seed3'
files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in cp.glob('step-2410158.*') if p.suffix in ('.index','.meta') or '.data-' in p.name} if NODE==0 else {}
result=dict(node=NODE,hostname=os.uname()[1],kernel=os.uname().release,protected=51,core=core,sysctl=sysctl,old_checkpoint=files,replay_bytes=(cp/'replay.npz').stat().st_size if NODE==0 else None,epoch=time.time())
(root/'preflight.json').write_text(json.dumps(result))
print(json.dumps(result))
'''


def preflight():
    path = HERE / 'registration.json'
    if not path.exists():
        start = time.time()
        save('registration.json', dict(start_epoch=start, deadline_epoch=start + 2700,
             task_cutoff_epoch=start + 2100, authorized_wall_minutes=45,
             conservative_machine_hours=3.75, cleanup_reserve_minutes=10,
             user_authorization='同意，先完成检查和短准入',
             scope=['read-only historical BN/config/optimizer audit', 'at most one225k admission'],
             formal_training_included=False, cloud_root=ROOT))
    result = parallel(lambda n: rpc(n, PREFLIGHT, 60))
    expected = ('learner-vm', 'actor1-vm', 'actor2-vm', 'actor3-vm', 'actor4-vm')
    assert all(row['hostname'] == expected[i] for i, row in enumerate(result))
    assert all(row['core'] == result[0]['core'] for row in result)
    save('preflight.json', result)
    print(json.dumps(dict(status='PASS_FIVE_HOST_PREFLIGHT',remaining_s=registration()['task_cutoff_epoch']-time.time(),ephemeral=[r['sysctl']['net/ipv4/ip_local_port_range'] for r in result])))


def deploy():
    import hashlib
    names=('distributed.py','evaluate.py','node_control.py','timed_distributed.py','cluster_control.py',
           'moving_stats_backend.py','consistent_target_bn.py','episode_training.py','baseline_admission.py','mahimahi_metrics.py')
    payload={'frameworks/baseline/'+name:(LOCAL_CODE/'frameworks/baseline'/name).read_bytes().replace(b'\r\n',b'\n') for name in names}
    for name in ('run.py','templates/baseline-train.json','tools/audit_baseline_target_bn.py','tools/audit_episode_replay.py'):
        payload[name]=(LOCAL_CODE/name).read_bytes().replace(b'\r\n',b'\n')
    for name,data in payload.items():
        if name.endswith('.py'):compile(data,name,'exec')
    create=r'''
import pathlib,json
r=pathlib.Path(ROOT);s=pathlib.Path(SOURCE)
for name in ('frameworks/baseline/configs','frameworks/baseline/logs','frameworks/baseline/checkpoints','frameworks/baseline/audit','frameworks/baseline/metrics','frameworks/baseline/report','frameworks/baseline/plots','templates','configs','runtime','tools'):(r/name).mkdir(parents=True,exist_ok=True)
f=r/'frameworks/baseline'
for name in ('code','id_cluster','known_hosts'):(f/name).symlink_to((s/name).resolve())
(f/'configs/traces').symlink_to((s/'configs/traces').resolve());(r/'runtime/venv').symlink_to('/home/experiment/venv')
if NODE==0:(r/'configs/read-only-config.json').write_bytes((s/'logs/healthy-base-seed3/config.json').read_bytes())
print(json.dumps(dict(node=NODE,created=True)))
'''
    parallel(lambda n:rpc(n,create,45))
    def put(node):
        client=connect(node)
        try:
            with client.open_sftp() as sftp:
                for name,data in payload.items():put_chunks(sftp,ROOT+'/'+name,data)
                put_chunks(sftp,ROOT+'/configs/cluster-current.json',json.dumps(dict(nodes=[dict(ip=host['ips'][0]) for host in FLEET])).encode())
        finally:client.close()
        expected={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}
        return rpc(node,"import hashlib,json,pathlib\nr=pathlib.Path(ROOT)\nassert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in EXPECTED.items())\n(r/'source-manifest.json').write_text(json.dumps(EXPECTED))\nprint(json.dumps(dict(node=NODE,verified=len(EXPECTED))))",45,EXPECTED=expected)
    save('deployment.json',parallel(put))
    save('source-manifest.json',{name:hashlib.sha256(data).hexdigest() for name,data in payload.items()})
    print(json.dumps(dict(status='DEPLOYED',remaining_s=registration()['task_cutoff_epoch']-time.time())))


def query(mode):
    assert mode in ('stock','repaired')
    body=r'''
import json,pathlib,subprocess,sys,time,os
r=pathlib.Path(ROOT);assert time.time()+180<CUTOFF
out=r/(MODE+'-target-audit.json')
cmd=[sys.executable,'-B',str(r/'tools/audit_baseline_target_bn.py'),'--framework',str(r/'frameworks/baseline'),'--config',str(r/'configs/read-only-config.json'),'--checkpoint',SOURCE+'/checkpoints/healthy-base-seed3/step-2410158','--replay',SOURCE+'/checkpoints/healthy-base-seed3/replay.npz','--output',str(out),'--mode',MODE]
with (r/(MODE+'-query.stdout')).open('wb') as stream:p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
if p.returncode:raise RuntimeError((r/(MODE+'-query.stdout')).read_text()[-7000:])
result=json.loads(out.read_text());print(json.dumps(result))
'''
    result=rpc(0,body,220,MODE=mode)
    save(mode+'-target-audit.json',result)
    print(json.dumps({key:result[key] for key in ('mode','shape','action','logit','td_target')}))


def review_old():
    anchors=rpc(0,"import pathlib,json\np=pathlib.Path(SOURCE)/'checkpoints/healthy-base-seed3'\nprint(json.dumps([dict(step=n,epoch=(p/('step-%d.index'%n)).stat().st_mtime) for n in (1800000,2410158)]))")
    body=(HERE/'old_log_review.py').read_text()
    rows=parallel(lambda node:rpc(node,body,120,CP_ANCHORS=anchors),range(1,5))
    save('old-training-review.json',rows)
    print(json.dumps([dict(condition=r['condition'],windows=[{k:w[k] for k in ('step','utilization','mean_queue_ms','max_bw_capacity','reward')} for w in r['windows']]) for host in rows for r in host]))


def refresh(revision='r3'):
    import hashlib
    names=['run.py']+['frameworks/baseline/'+n for n in ('distributed.py','evaluate.py','cluster_control.py','baseline_admission.py')]
    payload={name:(LOCAL_CODE/name).read_bytes().replace(b'\r\n',b'\n') for name in names}
    def put(node):
        client=connect(node)
        try:
            with client.open_sftp() as sftp:
                for name,data in payload.items():
                    temporary=ROOT+'/'+name+'.next-'+uuid.uuid4().hex
                    put_chunks(sftp,temporary,data)
                    sftp.posix_rename(temporary,ROOT+'/'+name)
        finally:client.close()
        expected=json.loads((HERE/'source-manifest.json').read_bytes())
        expected.update({name:hashlib.sha256(data).hexdigest() for name,data in payload.items()})
        return rpc(node,"import pathlib,hashlib,json\nr=pathlib.Path(ROOT)\nassert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in EXPECTED.items())\n(r/('source-manifest-'+REVISION+'.json')).write_text(json.dumps(EXPECTED))\nprint(json.dumps(dict(node=NODE,verified=len(EXPECTED))))",60,EXPECTED=expected,REVISION=revision)
    rows=parallel(put)
    expected=json.loads((HERE/'source-manifest.json').read_bytes());expected.update({name:hashlib.sha256(data).hexdigest() for name,data in payload.items()})
    save('source-manifest-'+revision+'.json',expected);save('deployment-'+revision+'.json',rows)
    print(json.dumps(dict(status='UPDATED_PRE_ADMISSION_SOURCE',remaining_s=registration()['task_cutoff_epoch']-time.time())))


def launch():
    import subprocess
    code_id=subprocess.check_output(['git','-C',str(LOCAL_CODE),'rev-parse','HEAD']).decode().strip()
    body=r'''
import json,pathlib,subprocess,sys,time,os
r=pathlib.Path(ROOT);assert time.time()+700<CUTOFF
sys.path.insert(0,str(r/'frameworks/baseline/code/runtime/_deps'));import psutil
assert not (r/'admission-launch.json').exists(),'Only one admission is authorized'
cmd=[sys.executable,'-B',str(r/'run.py'),'train','--variant','baseline','--run-id','targetbn-admission','--actors','24','--samples','225000','--replay-probe','--target-bn-consistent','--checkpoint-interval','100000','--performance-guard-policy','warn','--seed','3','--port','10000','--learner-port','20000','--cluster',str(r/'configs/cluster-current.json'),'--deadline-epoch',str(CUTOFF)]
with (r/'admission-launch.json').open('x') as stream:json.dump(dict(status='REQUESTED',epoch=time.time()),stream)
with (r/'admission.stdout').open('wb') as stream:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=str(r),start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
result=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),cmd=cmd,epoch=time.time(),local_source_commit=CODE_ID,source_manifest='source-manifest-r3.json')
(r/'admission-launch.json').write_text(json.dumps(result));print(json.dumps(result))
'''
    result=rpc(0,body,40,CODE_ID=code_id);save('admission-launch.json',result)
    print(json.dumps(dict(started=True,pid=result['pid'],remaining_s=registration()['task_cutoff_epoch']-time.time())))


def status():
    body=r'''
import pathlib,sys,json
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
run='targetbn-admission';s=node.status(run)
for row in s['entries']:
 if row['role']=='learner':row['training']=row['training'][-1:]
events=f/'logs'/run/'checkpoint-events.jsonl'
print(json.dumps(dict(node=NODE,entries=s['entries'],result=node.load(f/'metrics'/(run+'-cluster-result.json')),probe=node.load(f/'logs'/run/'replay-probe.json'),checkpoints=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else [])))
'''
    rows=parallel(lambda node:rpc(node,body,40));save('admission-latest.json',rows)
    summary=[]
    for row in rows:
        summary.append(dict(node=row['node'],result=row['result'],probe=row['probe'],checkpoints=row['checkpoints'],
            actors=[dict(index=e['index'],alive=e['alive'],generated=(e.get('producer') or e.get('ingress_progress') or {}).get('generated'),reason=(e.get('result') or {}).get('reason')) for e in row['entries'] if e['role']=='actor'],
            learner=[e.get('training') for e in row['entries'] if e['role']=='learner']))
    print(json.dumps(summary))


def initial_parity():
    body=r'''
import pathlib,json,sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf,numpy as np
old=tf.train.NewCheckpointReader(SOURCE+'/checkpoints/healthy-base-seed3/step-0')
new=tf.train.NewCheckpointReader(ROOT+'/frameworks/baseline/checkpoints/targetbn-admission/step-0')
a=old.get_variable_to_shape_map();b=new.get_variable_to_shape_map()
assert set(a)==set(b)
different=[name for name in a if not np.array_equal(old.get_tensor(name),new.get_tensor(name))]
print(json.dumps(dict(variables=len(a),different=different,all_identical=not different,source='Saved step-0 tensors only; no live model queries or optimizer')))
'''
    result=rpc(0,body,60);save('initial-tensor-parity.json',result);print(json.dumps(result))


def failure_review():
    body=r'''
import pathlib,json,sys,subprocess,time
r=pathlib.Path(ROOT);f=r/'frameworks/baseline'
sys.path.insert(0,str(f/'code/runtime/_deps'));import psutil
reports=[]
for path in (f/'logs').glob('targetbn-admission-actor*/cleanup-audit.json'):
 p=path.parent/'producer.json';reports.append(dict(actor=path.parent.name,audit=json.loads(path.read_text()),producer=json.loads(p.read_text()) if p.exists() else None))
listeners=subprocess.run(['sudo','-n','ss','-ltnp'],stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.decode()
remaining=[]
for p in psutil.process_iter(['pid','cmdline','create_time','status']):
 if p.pid!=__import__('os').getpid() and p.info['status']!=psutil.STATUS_ZOMBIE and any(ROOT+'/' in a for a in (p.info['cmdline'] or [])):remaining.append(p.info)
result=dict(node=NODE,reports=reports,listeners=listeners,remaining=remaining,learner_failure=(f/'logs/targetbn-admission/failure.txt').read_text() if (f/'logs/targetbn-admission/failure.txt').exists() else None,epoch=time.time())
print(json.dumps(result))
'''
    result=parallel(lambda node:rpc(node,body,45));save('admission-failure-review.json',result)
    print(json.dumps([dict(node=r['node'],reports=r['reports'],remaining=r['remaining'],learner_failure=r['learner_failure']) for r in result]))


def collect():
    body=r'''
import pathlib,json,sys,hashlib,subprocess,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np,tensorflow as tf
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';run='targetbn-admission';d=f/'logs'/run;cp=f/'checkpoints'/run
cfg=json.loads((f/'configs'/(run+'.json')).read_text())
probe=json.loads((d/'replay-probe.json').read_text());path=cp/probe['path']
with np.load(str(path)) as data:
 actors=data['actor_ids'];sequences=data['actor_sequences'];inserted=data['actor_inserted'];birth=data['birth'];rows=len(data['s0'])
 assert rows==150000 and np.all(sequences>0)
 exported=[]
 for node in range(1,5):
  mask=np.array([cfg['actor_hosts'][int(i)]==node-1 for i in actors]);out=r/('probe-origins-node%d.npz'%node)
  np.savez_compressed(str(out),s0=np.empty((int(mask.sum()),0),np.float32),actor_ids=actors[mask],actor_sequences=sequences[mask],actor_inserted=inserted)
  destination=str(r/out.name)
  command=['scp','-q','-i',str(f/'id_cluster'),'-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(f/'known_hosts'),str(out),'ubuntu@'+IPS[node-1]+':'+destination]
  completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
  assert completed.returncode==0,completed.stderr.decode()
  exported.append(dict(node=node,rows=int(mask.sum()),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
first=tf.train.NewCheckpointReader(str(cp/'step-0'));later=tf.train.NewCheckpointReader(str(cp/'step-200009'))
stats=[]
for name in first.get_variable_to_shape_map():
 if name.startswith('target_actor/') and ('moving_mean' in name or 'moving_variance' in name):
  a=first.get_tensor(name);b=later.get_tensor(name);stats.append(dict(name=name,changed=not np.array_equal(a,b),finite=bool(np.isfinite(b).all()),min=float(b.min()),max=float(b.max())))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in cp.glob('step-*') if p.suffix in ('.index','.json') or '.data-' in p.name}
training=(d/'training.csv').read_text()
result=dict(node=NODE,probe=probe,probe_inserted=int(inserted.sum()),probe_rows=rows,probe_birth_min=int(birth.min()),probe_birth_max=int(birth.max()),exports=exported,checkpoint_files=files,statistics_after_live_updates=stats,training_csv=training,config=cfg,cluster_result=json.loads((f/'metrics'/(run+'-cluster-result.json')).read_text()),final_replay_exists=(cp/'replay.npz').exists(),final_result_exists=(d/'result.json').exists())
(r/'collected-evidence.json').write_text(json.dumps(result));print(json.dumps(result))
'''
    result=rpc(0,body,120,IPS=[r['ips'][0] for r in FLEET[1:]])
    save('collected-evidence.json',result)
    join=r'''
import pathlib,sys,json,hashlib
r=pathlib.Path(ROOT);sys.path.insert(0,str(r/'tools'));from audit_episode_replay import audit
f=r/'frameworks/baseline';origins=list((f/'logs').glob('targetbn-admission-actor*/transition-origins.jsonl'))
p=r/('probe-origins-node%d.npz'%NODE)
report=audit(p,origins);report['node']=NODE;report['input_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(r/'probe-origin-audit.json').write_text(json.dumps(report));print(json.dumps(report))
'''
    audits=parallel(lambda node:rpc(node,join,60),range(1,5))
    save('probe-origin-audit.json',audits)
    assert sum(a['retained_rows'] for a in audits)==150000 and sum(len(a['actors']) for a in audits)==24
    print(json.dumps(dict(status='PASS_PROBE_ORIGIN_JOIN_NOT_FINAL_LEDGER',rows=150000,actors=24,final_replay_exists=result['final_replay_exists'],statistics_changed=[s['changed'] for s in result['statistics_after_live_updates']],VPC_metadata_bytes=sum(x['bytes'] for x in result['exports']))))


def close():
    initial=json.loads((HERE/'preflight.json').read_bytes())
    body=r'''
import pathlib,json,sys,os,time,signal,hashlib,subprocess
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
import psutil,sysv_ipc
node.cleanup('targetbn-admission')
targets={}
for p in psutil.process_iter(['pid','cmdline','create_time','status']):
 if p.pid!=os.getpid() and p.info['create_time']>=START and p.info['status']!=psutil.STATUS_ZOMBIE and any(ROOT+'/' in a for a in (p.info['cmdline'] or [])):targets[p.pid]=p.info['create_time']
for sig in (signal.SIGTERM,signal.SIGKILL):
 for pid,created in targets.items():
  try:
   p=psutil.Process(pid)
   if abs(p.create_time()-created)>=.01:continue
   try:p.send_signal(sig)
   except psutil.AccessDenied:subprocess.run(['sudo','-n','kill','-'+str(int(sig)),'--',str(pid)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  except psutil.Error:pass
 time.sleep(.4)
remaining=[]
for p in psutil.process_iter(['pid','cmdline','create_time','status']):
 if p.pid!=os.getpid() and p.info['create_time']>=START and p.info['status']!=psutil.STATUS_ZOMBIE and any(ROOT+'/' in a for a in (p.info['cmdline'] or [])):remaining.append(dict(pid=p.pid,created=p.info['create_time']))
ipc=[]
for path in (f/'logs').glob('targetbn-admission*/ownership.json'):
 for key in json.loads(path.read_text()).get('shm_keys',[]):
  segment=None
  try:segment=sysv_ipc.SharedMemory(key);ipc.append(key)
  except sysv_ipc.ExistentialError:pass
  finally:
   if segment:segment.detach()
ss=subprocess.run(['sudo','-n','ss','-ltn'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
ports=set(map(str,range(10000,10024)))|{'20000'}
listeners=[line for line in ss.stdout.decode().splitlines() if any(token.rsplit(':',1)[-1] in ports for token in line.split())]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=pathlib.Path('/home/experiment/orca-fixed-scratch-20261024');manifest=json.loads((protected/'audit/protected-models-before.json').read_text())
models=len(manifest)==51 and all(sha(protected/n)==h for n,h in manifest.items())
source_manifest=json.loads((r/'source-manifest-r3.json').read_text());source_ok=all(sha(r/n)==h for n,h in source_manifest.items())
core_ok=all(sha(pathlib.Path(SOURCE)/n)==h for n,h in ORIGINAL['core'].items())
sysctl={n:pathlib.Path('/proc/sys',n).read_text().strip() for n in ORIGINAL['sysctl']}
old_files_ok=all(sha(pathlib.Path(p))==h for p,h in OLD_FILES.items()) if NODE==0 else True
result=dict(node=NODE,remaining=remaining,recorded_ipc=ipc,listeners=listeners,protected_models=models,source_manifest_unchanged=source_ok,original_core_unchanged=core_ok,sysctl_unchanged=sysctl==ORIGINAL['sysctl'],old_checkpoint_and_replay_unchanged=old_files_ok,epoch=time.time())
result['status']='PASS' if not remaining and not ipc and not listeners and models and source_ok and core_ok and old_files_ok and result['sysctl_unchanged'] and ss.returncode==0 else 'REVIEW'
(r/'closure.json').write_text(json.dumps(result));print(json.dumps(result))
'''
    old=json.loads((HERE/'stock-target-audit.json').read_bytes())['source_files_unchanged']
    rows=parallel(lambda n:rpc(n,body,90,ORIGINAL=initial[n],OLD_FILES=old))
    save('closure.json',rows)
    assert all(row['status']=='PASS' for row in rows)
    print(json.dumps(dict(status='PASS_ALL_FIVE_CLOSED',elapsed_minutes=(time.time()-registration()['start_epoch'])/60,conservative_machine_hours=(time.time()-registration()['start_epoch'])*5/3600)))


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
