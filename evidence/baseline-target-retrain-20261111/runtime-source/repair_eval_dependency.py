"""Retry workers that exited at import, before any network run was created."""
import hashlib
import json
import control as c

payload=(c.transport.LOCAL_CODE/'frameworks/mtpjoint/common_period_metrics.py').read_bytes().replace(b'\r\n',b'\n')
def repair(n):
    client=c.transport.connect(n)
    try:
        with client.open_sftp() as sftp:
            c.transport.put_chunks(sftp,c.ROOT+'/frameworks/baseline/common_period_metrics.py',payload)
    finally:client.close()
    body=r'''
import pathlib,json,sys,subprocess,os,time,hashlib
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));sys.path.insert(0,str(f/'code/runtime/_deps'));import psutil
old=json.loads((r/'confirm-launch.json').read_text())
plan=json.loads((f/'confirm-plan.json').read_text())
assert not any((f/'configs'/(j['run']+'.json')).exists() for j in plan)
for w in old['workers']:
 assert not psutil.pid_exists(w['pid'])
 text=(r/('confirm-slot%d.stdout'%w['slot'])).read_text()
 assert "ModuleNotFoundError: No module named 'common_period_metrics'" in text
 (r/('confirm-slot%d-import-failure.stdout'%w['slot'])).write_text(text)
import frozen_worker
assert hashlib.sha256((f/'common_period_metrics.py').read_bytes()).hexdigest()==EXPECTED
assert time.time()+650<CUTOFF
records=[]
for slot in range(2):
 phase='confirm-slot%d'%slot
 with (r/(phase+'-retry.stdout')).open('wb') as stream:p=subprocess.Popen([sys.executable,'-B',str(f/'frozen_worker.py'),phase],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=str(f),start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 records.append(dict(slot=slot,pid=p.pid))
out=dict(node=NODE,workers=records,original_import_failure=old,epoch=time.time(),dependency_sha256=EXPECTED,network_runs_before_retry=0)
(r/'confirm-retry-launch.json').write_text(json.dumps(out));print(json.dumps(out))
'''
    return c.rpc(n,body,60,EXPECTED=hashlib.sha256(payload).hexdigest())
c.save('confirm-dependency-retry.json',c.parallel(repair))
print(json.dumps(dict(status='RETRIED_ALL_IMPORT_FAILURES',network_runs_before_retry=0)))
