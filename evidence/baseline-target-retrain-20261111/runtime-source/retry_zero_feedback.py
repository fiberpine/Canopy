"""Bounded repair for at most two runs with zero policy decisions."""
import json
import time
import control as c

c.save('zero-feedback-retry-registration.json',dict(epoch=time.time(),maximum_retries=2,
    eligibility='Original run has zero valid feedback and no control row: model has never acted. Never retry a performance failure or any run after first policy action.',
    model='Same frozen final12tensors',gates_unchanged=True,unique_names_ports=True,
    original_rows_retained=True,retry_selection_before_retry_data=True))
while time.time()+200<c.registration()['task_cutoff_epoch']:
    rows=c.evaluation_status()
    if all(h['finished'] for h in rows):break
    time.sleep(20)
else:raise RuntimeError('No bounded startup-retry time remaining')
bad=[(h['node'],r) for h in rows for r in h['rows'] if r['status']!='COMPLETED']
assert len(bad)<=2,bad
plan=[]
for n,r in bad:
    body=r'''
import pathlib,json,csv
f=pathlib.Path(ROOT)/'frameworks/baseline';d=f/'logs'/RUN
z=json.loads((d/'result.json').read_text())
assert z['reason']=='no valid state within15s' and z['counters']['valid']==0
control=d/'control.csv'
if control.exists():
 with control.open() as s:assert not list(csv.DictReader(s))
print(json.dumps(dict(node=NODE,zero_policy_decisions_verified=True,original=RUN)))
'''
    proof=c.rpc(n,body,45,RUN=r['job']['run'])
    job=dict(r['job']);job.update(original_run=job['run'],run=job['run']+'-startup-retry',port=14000+len(plan),slot=0)
    plan.append(dict(node=n,job=job,proof=proof))
c.save('zero-feedback-retry-plan.json',plan)
def launch(entry):
    n=entry['node'];job=entry['job'];client=c.transport.connect(n)
    try:
        with client.open_sftp() as sftp:
            c.transport.put_chunks(sftp,c.ROOT+'/frameworks/baseline/startup-retry-plan.json',json.dumps([job]).encode())
    finally:client.close()
    body=r'''
import pathlib,json,subprocess,sys,os,time
r=pathlib.Path(ROOT);f=r/'frameworks/baseline'
assert time.time()+150<CUTOFF and not (r/'startup-retry-launch.json').exists()
with (r/'startup-retry.stdout').open('wb') as stream:p=subprocess.Popen([sys.executable,'-B',str(f/'frozen_worker.py'),'startup-retry'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=str(f),start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
out=dict(node=NODE,pid=p.pid,epoch=time.time());(r/'startup-retry-launch.json').write_text(json.dumps(out));print(json.dumps(out))
'''
    return c.rpc(n,body,45)
c.save('zero-feedback-retry-launch.json',[launch(e) for e in plan])
print(json.dumps(dict(status='STARTUP_RETRY_LAUNCHED',runs=len(plan))))
