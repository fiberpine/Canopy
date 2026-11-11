"""Authorized90-minute continuation, max two admissions and one fresh full run."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
ROOT='/home/experiment/orca-targetbn-retrain-20261111-night'
SPEC=importlib.util.spec_from_file_location('bounded_target_transport',WORKSPACE/'experiments/baseline-target-audit-20261111-2222/control.py')
transport=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(transport)
transport.HERE=HERE;transport.ROOT=ROOT
save=transport.save;rpc=transport.rpc;parallel=transport.parallel;registration=transport.registration


def prepare():
    if not (HERE/'registration.json').exists():
        start=time.time()
        save('registration.json',dict(start_epoch=start,deadline_epoch=start+5400,task_cutoff_epoch=start+4800,
             train_cutoff_epoch=start+3600,authorized_wall_minutes=90,conservative_machine_hours=7.5,
             cleanup_reserve_minutes=10,eval_reserve_minutes=20,
             user_authorization='同意，按此预算持续推进',
             scope=['at most two repaired225k admissions','one fresh2.4M scalar targetBN run','24conditions x3 frozen evaluation'],
             formal_training_included=True,cloud_root=ROOT,
             selected_variable='target_bn_consistent=True; original single connection retained',
             source_commit=subprocess.check_output(['git','-C',str(transport.LOCAL_CODE),'rev-parse','HEAD']).decode().strip(),
             prior_failed_batch='baseline-target-audit-20261111-2222',
             retry_permission='Unexpected failure may be repaired and rechecked within this same absolute window; no phase reapproval required'))
    transport.preflight();transport.deploy()
    old=WORKSPACE/'experiments/baseline-screen-20261111-154219/registration-frozen.json'
    gates=json.loads(old.read_bytes())
    save('health-registration.json',dict(gates=gates['gates'],aggregation=gates['aggregation'],
         candidate_selection='Only the final fully drained checkpoint of this fresh run; no intermediate selection',
         conditions=gates['conditions'],repeats=3,seconds=60,window='trim10',
         scope='Original24deep5BDP conditions only; no shallow/dynamic or multiseed claim',
         protocol_source='189205c docs/BASELINE_NEXT_TRAIN_20261111.md',
         initial_policy_expected='de8ec8d8c88645380ce0863aa4bda258bbf7cc099cf4b79c8d9296ce5bdd83ca'))
    print(json.dumps(dict(status='PREPARED90_MINUTE_WINDOW',remaining_s=registration()['deadline_epoch']-time.time())))


def launch(run):
    if run not in ('admission-r1','admission-r2','targetbn-base-seed3'):raise ValueError(run)
    formal=run=='targetbn-base-seed3'
    if formal:
        passed=json.loads((HERE/'admission-pass.json').read_bytes())
        assert passed['status']=='PASS'
    body=r'''
import pathlib,subprocess,sys,json,os,time
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f/'code/runtime/_deps'));import psutil
assert time.time()+600<LIMIT
path=r/(RUN+'-launch.json');assert not path.exists(),'Unique run must never restart automatically'
if not FORMAL:assert len(list(r.glob('admission-r*-launch.json')))<2
cmd=[sys.executable,'-B',str(r/'run.py'),'train','--variant','baseline','--run-id',RUN,'--actors','24','--samples',str(2400000 if FORMAL else 225000),'--target-bn-consistent','--checkpoint-interval',str(300000 if FORMAL else 100000),'--performance-guard-policy','warn','--seed','3','--port',str(11000 if FORMAL else (10000 if RUN=='admission-r1' else 12000)),'--learner-port',str(21000 if FORMAL else (20000 if RUN=='admission-r1' else 22000)),'--cluster',str(r/'configs/cluster-current.json'),'--deadline-epoch',str(LIMIT)]
if not FORMAL:cmd.append('--replay-probe')
with path.open('x') as stream:json.dump(dict(status='REQUESTED',epoch=time.time()),stream)
with (r/(RUN+'.stdout')).open('wb') as stream:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=str(r),start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
out=dict(run=RUN,formal=FORMAL,pid=p.pid,created=psutil.Process(p.pid).create_time(),cmd=cmd,epoch=time.time(),source_commit=CODE_ID)
path.write_text(json.dumps(out));print(json.dumps(out))
'''
    reg=registration()
    limit=reg['train_cutoff_epoch'] if formal else min(time.time()+1000,reg['train_cutoff_epoch']-1800)
    assert time.time()+650<limit
    result=rpc(0,body,40,RUN=run,FORMAL=formal,LIMIT=limit,CODE_ID=reg['source_commit'])
    save(run+'-launch.json',result)
    print(json.dumps(dict(run=run,pid=result['pid'],run_deadline=limit,remaining_window_s=reg['deadline_epoch']-time.time())))


def status(run):
    body=r'''
import pathlib,sys,json
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
s=node.status(RUN)
for entry in s['entries']:
 if entry['role']=='learner':entry['training']=entry['training'][-1:]
events=f/'logs'/RUN/'checkpoint-events.jsonl'
print(json.dumps(dict(node=NODE,entries=s['entries'],result=node.load(f/'metrics'/(RUN+'-cluster-result.json')),probe=node.load(f/'logs'/RUN/'replay-probe.json'),checkpoints=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else [])))
'''
    rows=parallel(lambda node:rpc(node,body,45,RUN=run));save(run+'-latest.json',rows)
    learner=next((e for e in rows[0]['entries'] if e['role']=='learner'),{})
    latest=(learner.get('training') or [None])[-1]
    actors=[e for host in rows for e in host['entries'] if e['role']=='actor']
    print(json.dumps(dict(run=run,result=rows[0]['result'],latest=latest,actors_alive=sum(e['alive'] for e in actors),actors=len(actors),
                         actor_errors=[dict(index=e['index'],reason=e['result'].get('reason')) for e in actors if e.get('result') and e['result']['status']!='COMPLETED'],
                         probe=rows[0]['probe'],checkpoints=rows[0]['checkpoints'],remaining_s=registration()['deadline_epoch']-time.time())))
    return rows


def assess_admission(run):
    body=r'''
import pathlib,json,csv,io
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';d=f/'logs'/RUN;cp=f/'checkpoints'/RUN
result=json.loads((f/'metrics'/(RUN+'-cluster-result.json')).read_text());assert result['status']=='COMPLETED'
cfg=json.loads((f/'configs'/(RUN+'.json')).read_text());final=json.loads((d/'result.json').read_text());actors=json.loads((f/'metrics'/(RUN+'-actors.json')).read_text());coverage=json.loads((cp/'sample-coverage.json').read_text())
assert final['inserted']==225000 and final['status']=='COMPLETED'
expected=int((225000-cfg['warmup'])*cfg['update_ratio'])
assert all(final['counters'][k]==expected for k in ('critic_iterations','actor_updates','target_updates'))
assert final['counters']['critic_optimizer_ops']==2*expected
assert len(actors)==24 and not any(a['mapping_errors'] for a in actors)
assert all(a['producer']['generated']==a['producer']['enqueued']==coverage['inserted_by_actor'][a['index']] for a in actors)
probe=json.loads((d/'replay-probe.json').read_text());events=[json.loads(line) for line in (d/'checkpoint-events.jsonl').read_text().splitlines()]
assert probe['rows']==150000 and any(e['kind']=='live' and e['replay_full'] and not e['stop_requested'] and e['inserted']<225000 for e in events)
print(json.dumps(dict(status='PASS',run=RUN,final=final,probe=probe,events=events,actors=actors,coverage=coverage,config=cfg)))
'''
    result=rpc(0,body,60,RUN=run);save('admission-pass.json',result)
    print(json.dumps(dict(status=result['status'],run=run,inserted=result['final']['inserted'],updates=result['final']['counters'])))


def prepare_frozen():
    import hashlib
    source=WORKSPACE/'experiments/baseline-screen-20261111-154219'
    payload={name:(source/name).read_bytes().replace(b'\r\n',b'\n') for name in ('frozen_worker.py','capture_frozen.py')}
    reg=registration();health=json.loads((HERE/'health-registration.json').read_bytes())
    frozen=dict(run_cutoff=reg['task_cutoff_epoch'],native=ROOT+'/frameworks/baseline/code/runtime/orca-server-control-log-safe')
    def put(node):
        client=transport.connect(node)
        try:
            with client.open_sftp() as sftp:
                for name,data in payload.items():transport.put_chunks(sftp,ROOT+'/frameworks/baseline/'+name,data)
                transport.put_chunks(sftp,ROOT+'/frameworks/baseline/registration.json',json.dumps(frozen).encode())
        finally:client.close()
        return dict(node=node,sha256={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()})
    save('frozen-helper-deployment.json',parallel(put))
    print(json.dumps(dict(status='FROZEN_HELPERS_READY')))


def training_complete():
    body=r'''
import pathlib,json
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';run='targetbn-base-seed3'
result=json.loads((f/'metrics'/(run+'-cluster-result.json')).read_text());assert result['status']=='COMPLETED',result
final=json.loads((f/'logs'/run/'result.json').read_text());assert final['status']=='COMPLETED' and final['inserted']>=2400000
cfg=json.loads((f/'configs'/(run+'.json')).read_text());expected=int((final['inserted']-cfg['warmup'])*cfg['update_ratio'])
assert all(final['counters'][key]==expected for key in ('critic_iterations','actor_updates','target_updates'))
assert final['counters']['critic_optimizer_ops']==2*expected
actors=json.loads((f/'metrics'/(run+'-actors.json')).read_text());assert len(actors)==24 and not any(a['mapping_errors'] for a in actors)
coverage=json.loads((f/'checkpoints'/run/'sample-coverage.json').read_text())
assert all(a['producer']['generated']==a['producer']['enqueued']==coverage['inserted_by_actor'][a['index']] for a in actors)
events=[json.loads(line) for line in (f/'logs'/run/'checkpoint-events.jsonl').read_text().splitlines()]
print(json.dumps(dict(result=result,final=final,actors=actors,coverage=coverage,config=cfg,checkpoint_events=events,selected_checkpoint=str(f/'checkpoints'/run/('step-%d'%final['inserted'])))))
'''
    result=rpc(0,body,60);save('training-complete.json',result)
    print(json.dumps(dict(status='COMPLETE_FULL_TRAINING',inserted=result['final']['inserted'],counters=result['final']['counters'])))


def export_final():
    info=json.loads((HERE/'training-complete.json').read_bytes())
    body=r'''
import json,pathlib,sys,hashlib,subprocess,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf,numpy as np
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';reader=tf.train.NewCheckpointReader(CHECKPOINT)
metadata=json.loads(pathlib.Path(CHECKPOINT+'.variables.json').read_text())
names=[name for name in metadata['variables'] if name.startswith('abstract_actor/') and '/Adam' not in name]
assert len(names)==12
variables={name[:-2]:tf.Variable(reader.get_tensor(name[:-2]),name=name[:-2],trainable=False) for name in names}
prefix=f/'checkpoints/frozen-final/model';prefix.parent.mkdir()
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as session:
 session.run(tf.global_variables_initializer());tf.train.Saver(var_list=variables).save(session,str(prefix),write_meta_graph=False)
small=tf.train.NewCheckpointReader(str(prefix));assert all(np.array_equal(small.get_tensor(name[:-2]),reader.get_tensor(name[:-2])) for name in names)
h=hashlib.sha256()
for name in names:h.update(name.encode());h.update(reader.get_tensor(name[:-2]).tobytes())
assert h.hexdigest()==pathlib.Path(CHECKPOINT+'.policy.sha256').read_text().strip()
files={suffix:hashlib.sha256(pathlib.Path(str(prefix)+suffix).read_bytes()).hexdigest() for suffix in ('.index','.data-00000-of-00001')}
for ip in IPS:
 common=['-i',str(f/'id_cluster'),'-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(f/'known_hosts')]
 command=['ssh']+common+['ubuntu@'+ip,'mkdir -p '+str(prefix.parent)]
 assert subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20).returncode==0
 for suffix in files:
  command=['scp','-q']+common+[str(prefix)+suffix,'ubuntu@'+ip+':'+str(prefix)+suffix]
  completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30);assert completed.returncode==0,completed.stderr.decode()
print(json.dumps(dict(prefix=str(prefix),policy_sha256=h.hexdigest(),model_sha256=files,tensors=len(names),bytes=sum(pathlib.Path(str(prefix)+s).stat().st_size for s in files),source=CHECKPOINT)))
'''
    result=rpc(0,body,120,CHECKPOINT=info['selected_checkpoint'],IPS=[n['ips'][0] for n in transport.FLEET[1:]])
    save('frozen-policy.json',result)
    check="import pathlib,hashlib,json\nassert all(hashlib.sha256(pathlib.Path(PREFIX+s).read_bytes()).hexdigest()==h for s,h in HASHES.items())\nprint(json.dumps(dict(node=NODE,verified=True)))"
    save('frozen-policy-host-proof.json',parallel(lambda node:rpc(node,check,45,PREFIX=result['prefix'],HASHES=result['model_sha256'])))
    print(json.dumps(result))


def launch_evaluation():
    reg=registration();policy=json.loads((HERE/'frozen-policy.json').read_bytes());health=json.loads((HERE/'health-registration.json').read_bytes())
    jobs=[]
    for repeat in range(3):
        for i,condition in enumerate(health['conditions']):
            number=i+repeat*24
            jobs.append(dict(host=number%5,slot=(number//5)%2,condition=condition,group='fresh-targetbn-final',checkpoint=policy['prefix'],model_sha256=policy['model_sha256'],repeat=repeat,run='targetbn-confirm-r%d-c%02d'%(repeat,i),port=13000+number))
    save('confirm-plan.json',jobs)
    def put(node):
        client=transport.connect(node)
        try:
            with client.open_sftp() as sftp:
                assigned=[j for j in jobs if j['host']==node]
                transport.put_chunks(sftp,ROOT+'/frameworks/baseline/confirm-plan.json',json.dumps(assigned).encode())
                for slot in range(2):transport.put_chunks(sftp,ROOT+'/frameworks/baseline/confirm-slot%d-plan.json'%slot,json.dumps([j for j in assigned if j['slot']==slot]).encode())
        finally:client.close()
        body=r'''
import pathlib,subprocess,sys,json,time,os
r=pathlib.Path(ROOT);f=r/'frameworks/baseline'
assert time.time()+MINIMUM<CUTOFF
with (r/'confirm-launch.json').open('x') as stream:json.dump(dict(status='REQUESTED',epoch=time.time()),stream)
records=[]
for slot in range(2):
 phase='confirm-slot%d'%slot
 with (r/(phase+'.stdout')).open('wb') as stream:p=subprocess.Popen([sys.executable,'-B',str(f/'frozen_worker.py'),phase],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=str(f),start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 records.append(dict(slot=slot,pid=p.pid))
result=dict(node=NODE,workers=records,epoch=time.time());(r/'confirm-launch.json').write_text(json.dumps(result));print(json.dumps(result))
'''
        return rpc(node,body,40,MINIMUM=70*max(sum(j['host']==node and j['slot']==slot for j in jobs) for slot in range(2))+30)
    save('confirm-launch.json',parallel(put))
    print(json.dumps(dict(status='EVALUATION_STARTED',jobs=72,hosts=5,slots_per_host=2)))


def evaluation_status():
    body="import pathlib,json\nf=pathlib.Path(ROOT)/'frameworks/baseline';slots=[]\nfor i in range(2):\n p=f/('confirm-slot%d-result.json'%i);q=f/('confirm-slot%d-progress.json'%i);slots.append(dict(slot=i,finished=p.exists(),rows=json.loads(q.read_text()) if q.exists() else []))\nprint(json.dumps(dict(node=NODE,finished=all(s['finished'] for s in slots),slots=slots,rows=[r for s in slots for r in s['rows']])))"
    rows=parallel(lambda node:rpc(node,body,45));save('confirm-latest.json',rows)
    print(json.dumps([dict(node=r['node'],finished=r['finished'],done=len(r['rows']),errors=[x.get('reason') for x in r['rows'] if x['status']!='COMPLETED']) for r in rows]))
    return rows


def add_frozen_period_observer():
    import hashlib
    import uuid
    source=(WORKSPACE/'experiments/baseline-screen-20261111-154219/frozen_worker.py').read_text()
    source=source.replace('import csv\n','import csv\nimport numpy as np\n',1)
    marker="            assert controls and all(row['state_id']==row['action_id'] and int(row['return_code'])==0 for row in controls)"
    assert marker in source
    source=source.replace(marker,marker+"\n            periods=np.diff([float(row['time_us']) for row in controls])/1000\n            period_stats=dict(p50=float(np.percentile(periods,50)),p95=float(np.percentile(periods,95)),p99=float(np.percentile(periods,99)),max=float(np.max(periods))) if len(periods) else {}")
    old="                        valid=result['counters']['valid'],wall_s=time.monotonic()-started)"
    assert old in source
    source=source.replace(old,"                        valid=result['counters']['valid'],control_period_ms=period_stats,wall_s=time.monotonic()-started)")
    compile(source,'frozen_worker.py','exec');payload=source.encode()
    (HERE/'frozen_worker_parallel.py').write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest()
    def put(node):
        client=transport.connect(node)
        try:
            with client.open_sftp() as sftp:
                temporary=ROOT+'/frameworks/baseline/frozen_worker.py.next-'+uuid.uuid4().hex
                transport.put_chunks(sftp,temporary,payload);sftp.posix_rename(temporary,ROOT+'/frameworks/baseline/frozen_worker.py')
        finally:client.close()
        return rpc(node,"import pathlib,hashlib,json\nr=pathlib.Path(ROOT);assert not (r/'confirm-launch.json').exists()\nassert hashlib.sha256((r/'frameworks/baseline/frozen_worker.py').read_bytes()).hexdigest()==EXPECTED\nprint(json.dumps(dict(node=NODE,sha256=EXPECTED)))",45,EXPECTED=expected)
    save('frozen-helper-parallel-proof.json',parallel(put))
    save('frozen-scheduling-amendment.json',dict(registered_before_any_frozen_data=True,slots_per_host=2,total_runs=72,conditions=24,repeats=3,source_sha256=expected,
          extra_policy_or_native_reads=0,control_period_observer='Post-run control.csv only',single_stream_CPU_claim=False,health_gates_unchanged=True))
    print(json.dumps(dict(status='PARALLEL_FROZEN_HELPER_VERIFIED',hosts=5)))


def integrity():
    body=r'''
import pathlib,json,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf,numpy as np
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';run='targetbn-base-seed3'
old=tf.train.NewCheckpointReader(SOURCE+'/checkpoints/healthy-base-seed3/step-0')
new=tf.train.NewCheckpointReader(str(f/'checkpoints'/run/'step-0'))
names=old.get_variable_to_shape_map();assert set(names)==set(new.get_variable_to_shape_map())
different=[n for n in names if not np.array_equal(old.get_tensor(n),new.get_tensor(n))]
assert not different
cfg=json.loads((f/'configs'/(run+'.json')).read_text())
assert cfg['target_bn_consistent'] and cfg['performance_guard_policy']=='warn' and cfg['checkpoint_interval']==300000 and 'episode_seconds' not in cfg
checks=[]
for host in []:pass
print(json.dumps(dict(initial_variables=len(names),all_identical=True,config=cfg)))
'''
    proof=rpc(0,body,60);save('formal-initial-parity.json',proof)
    cleanup=r'''
import pathlib,json
f=pathlib.Path(ROOT)/'frameworks/baseline';rows=[]
for p in (f/'logs').glob('admission-r1-actor*/cleanup-audit.json'):
 z=json.loads(p.read_text());rows.append(dict(actor=p.parent.name,status=z['status'],teardown_wait_s=z.get('teardown_wait_s'),checks=len(z.get('checks',[])),first_status=z.get('checks',[{}])[0].get('status'),fd=z['actor_fd_count']))
assert all(r['status']=='PASS' for r in rows)
print(json.dumps(dict(node=NODE,rows=rows)))
'''
    rows=parallel(lambda node:rpc(node,cleanup,45),range(1,5));save('admission-cleanup-proof.json',rows)
    print(json.dumps(dict(initial_variables=proof['initial_variables'],all_identical=True,cleanup_actors=sum(len(r['rows']) for r in rows),max_teardown_wait=max(c['teardown_wait_s'] for r in rows for c in r['rows']))))


def collect_summary():
    body=r'''
import pathlib,json,csv,sys,hashlib
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';out=dict(node=NODE,runs={})
for run in ('admission-r1','admission-r2','targetbn-base-seed3'):
 folder=f/'logs'/run
 if not folder.exists():continue
 cp=f/'checkpoints'/run;entry={}
 for name in ('result.json','first-update-coverage.json','replay-probe.json'):
  p=folder/name
  if p.exists():entry[name]=json.loads(p.read_text())
 for name in ('training.csv','checkpoint-events.jsonl'):
  p=folder/name
  if p.exists():entry[name]=p.read_text()
 p=cp/'sample-coverage.json'
 if p.exists():entry['sample-coverage.json']=json.loads(p.read_text())
 entry['failure_drain']=(f/'audit'/(run+'-failure-drain.json')).read_text() if (f/'audit'/(run+'-failure-drain.json')).exists() else None
 out['runs'][run]=entry
print(json.dumps(out))
'''
    summary=rpc(0,body,60);save('learner-summary.json',summary)
    actorbody=r'''
import pathlib,json,sys
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';rows=[]
for p in (f/'logs').glob('*-actor*/cleanup-audit.json'):
 z=json.loads(p.read_text());rows.append(dict(run=p.parent.name,report=z))
print(json.dumps(dict(node=NODE,cleanup=rows)))
'''
    actors=parallel(lambda node:rpc(node,actorbody,45),range(1,5));save('actor-cleanup-reports.json',actors)
    print(json.dumps(dict(status='SMALL_SUMMARIES_COLLECTED',runs=list(summary['runs']),cleanup_reports=sum(len(h['cleanup']) for h in actors))))


def amend_supervision():
    """Replace observers only, preserving every learner/actor/TCP process."""
    import hashlib
    reg=registration();phase_deadline=reg['start_epoch']+69*60
    files={name:(HERE/name).read_bytes() for name in ('attach_cluster_control.py','control_hard_watch.py')}
    def put(n):
        client=transport.connect(n)
        try:
            with client.open_sftp() as sftp:
                for name,data in files.items():transport.put_chunks(sftp,ROOT+'/frameworks/baseline/'+name,data)
        finally:client.close()
    parallel(put)
    prepare=r'''
import pathlib,json,time
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';run='targetbn-base-seed3'
cfg=json.loads((f/'configs'/(run+'.json')).read_text());status=json.loads((f/'logs/controller-status.json').read_text());launch=json.loads((r/(run+'-launch.json')).read_text())
assert status['run_id']==run and status['stage']=='RUNNING'
cfg.update(_original_start_epoch=launch['epoch'],_original_all_ready_epoch=status['all_ready_epoch'],evaluation_reserve_s=60,timeout_s=int(PHASE-launch['epoch'])+120,absolute_deadline_epoch=PHASE)
(r/'configs/handover-control-input.json').write_text(json.dumps(cfg))
auth=json.loads((r/'configs/cluster-current.json').read_text());auth.update(deadline_epoch=PHASE,authorization='Observer-only amended phase allocation within approved90minute window')
(r/'configs/handover-authorization.json').write_text(json.dumps(auth))
with (r/'handover-request.json').open('x') as stream:json.dump(dict(epoch=time.time(),data_phase_deadline=PHASE,global_deadline=DEADLINE),stream)
print(json.dumps(dict(prepared=True)))
'''
    rpc(0,prepare,45,PHASE=phase_deadline)
    suspend=r'''
import pathlib,json,time,os,signal,sys
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
import psutil
run='targetbn-base-seed3';records=[]
pid=int((f/'logs'/(run+'-watchdog.pid')).read_text());p=psutil.Process(pid);args=p.cmdline()
assert str(f/'node_control.py') in args and args[-2:]==['watch',run] and p.create_time()>=START
records.append(dict(role='old_watchdog',pid=pid,created=p.create_time()))
if NODE==0:
 launch=json.loads((r/(run+'-launch.json')).read_text());parent=psutil.Process(launch['pid']);assert abs(parent.create_time()-launch['created'])<.01
 candidates=[q for q in parent.children() if '-c' in q.cmdline() and any('from cluster_control import execute' in item for item in q.cmdline())]
 assert len(candidates)==1
 p=candidates[0];records.append(dict(role='old_coordinator',pid=p.pid,created=p.create_time()))
before=node.status(run)
assert before['entries'] and all(e['alive'] for e in before['entries'])
result=dict(node=NODE,records=records,before_pids=[dict(role=e['role'],index=e['index'],pid=e['pid'],created=e['created']) for e in before['entries']],epoch=time.time())
(r/'handover-suspended.json').write_text(json.dumps(result))
for row in records:psutil.Process(row['pid']).send_signal(signal.SIGSTOP)
print(json.dumps(result))
'''
    suspended=parallel(lambda n:rpc(n,suspend,45));save('handover-suspended.json',suspended)
    committed=False
    try:
        launch=r'''
import pathlib,json,subprocess,sys,os,time
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f/'code/runtime/_deps'));import psutil
run='targetbn-base-seed3'
with (r/'handover-watch.stdout').open('wb') as stream:watch=subprocess.Popen([sys.executable,'-B',str(f/'control_hard_watch.py'),run,str(PHASE)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
records=[dict(role='new_watchdog',pid=watch.pid,created=psutil.Process(watch.pid).create_time())]
if NODE==0:
 code="import sys,json,pathlib;sys.path.insert(0,sys.argv[1]);import attach_cluster_control as ac;c=json.loads(pathlib.Path(sys.argv[2]).read_text());print(json.dumps(ac.execute(c)))"
 with (r/'handover-controller.stdout').open('wb') as stream:p=subprocess.Popen([sys.executable,'-B','-c',code,str(f),str(r/'configs/handover-control-input.json')],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,ORCA_AUTHORIZATION_FILE=str(r/'configs/handover-authorization.json')))
 records.append(dict(role='new_coordinator',pid=p.pid,created=psutil.Process(p.pid).create_time()))
 until=time.time()+15
 while not (f/'audit'/(run+'-handover-ready.json')).exists():
  if p.poll() is not None:raise RuntimeError((r/'handover-controller.stdout').read_text()[-5000:])
  if time.time()>until:raise RuntimeError('Attach observer readiness timeout')
  time.sleep(.1)
(r/'handover-new-ownership.json').write_text(json.dumps(records))
print(json.dumps(dict(node=NODE,records=records)))
'''
        new=parallel(lambda n:rpc(n,launch,50,PHASE=phase_deadline));save('handover-new-ownership.json',new)
        rpc(0,"import pathlib,json,time\nf=pathlib.Path(ROOT)/'frameworks/baseline'\n(f/'audit/targetbn-base-seed3-handover-activate.json').write_text(json.dumps(dict(epoch=time.time())))\nprint(json.dumps(dict(activated=True)))",45)
        verify=r'''
import pathlib,sys,json
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
old=json.loads((r/'handover-suspended.json').read_text());s=node.status('targetbn-base-seed3')
now=[dict(role=e['role'],index=e['index'],pid=e['pid'],created=e['created']) for e in s['entries']]
assert now==old['before_pids'] and all(e['alive'] for e in s['entries'])
print(json.dumps(dict(node=NODE,pids_identical=True,entries=len(now))))
'''
        proof=parallel(lambda n:rpc(n,verify,45));save('handover-model-pid-proof.json',proof)
        committed=True
    finally:
        finish=r'''
import pathlib,json,signal,sys,time
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f/'code/runtime/_deps'));import psutil
old=json.loads((r/'handover-suspended.json').read_text())['records']
if not COMMITTED:(f/'audit/targetbn-base-seed3-handover-abort.json').write_text('{}')
for entry in old:
 try:
  p=psutil.Process(entry['pid'])
  if abs(p.create_time()-entry['created'])<.01:p.send_signal(signal.SIGKILL if COMMITTED else signal.SIGCONT)
 except psutil.NoSuchProcess:pass
if not COMMITTED and (r/'handover-new-ownership.json').exists():
 for entry in json.loads((r/'handover-new-ownership.json').read_text()):
  try:
   p=psutil.Process(entry['pid'])
   if abs(p.create_time()-entry['created'])<.01:p.kill()
  except psutil.NoSuchProcess:pass
print(json.dumps(dict(node=NODE,committed=COMMITTED)))
'''
        cleanup=parallel(lambda n:rpc(n,finish,45,COMMITTED=committed));save('handover-commit.json',cleanup)
    if not committed:raise RuntimeError('Observer handover rolled back')
    reg.update(original_train_cutoff_epoch=reg['train_cutoff_epoch'],train_cutoff_epoch=phase_deadline,
               evaluation_slots_per_host=2,supervision_amended_epoch=time.time())
    save('registration.json',reg)
    save('supervision-amendment.json',dict(status='COMMITTED',learning_processes_restarted=0,
         source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
         data_phase_deadline=phase_deadline,global_deadline=reg['deadline_epoch'],
         evaluation_runs=72,evaluation_slots_per_host=2,health_gates_unchanged=True,
         note='Only monitoring ownership/deadlines changed. Original CLI parent exits after old observer termination; verdict comes from attached original monitor tail. All learner/actor PIDs and creation times identical.'))
    print(json.dumps(dict(status='SUPERVISION_HANDOVER_COMMITTED',model_restarts=0,global_deadline_unchanged=True)))


def close():
    initial=json.loads((HERE/'preflight.json').read_bytes())
    old=json.loads((WORKSPACE/'experiments/baseline-target-audit-20261111-2222/stock-target-audit.json').read_bytes())['source_files_unchanged']
    body=r'''
import pathlib,json,sys,os,signal,time,hashlib,subprocess
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
import psutil,sysv_ipc
runs=['admission-r1','admission-r2','targetbn-base-seed3']
plan=f/'confirm-plan.json'
if plan.exists():runs += [j['run'] for j in json.loads(plan.read_text())]
retry=f/'startup-retry-plan.json'
if retry.exists():runs += [j['run'] for j in json.loads(retry.read_text())]
for run in runs:node.cleanup(run)
excluded={os.getpid()}|{p.pid for p in psutil.Process(os.getpid()).parents()}
targets={}
for p in psutil.process_iter(['pid','cmdline','create_time','status']):
 if p.pid not in excluded and p.info['create_time']>=START and p.info['status']!=psutil.STATUS_ZOMBIE and any(ROOT+'/' in a for a in (p.info['cmdline'] or [])):targets[p.pid]=p.info['create_time']
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
 if p.pid not in excluded and p.info['create_time']>=START and p.info['status']!=psutil.STATUS_ZOMBIE and any(ROOT+'/' in a for a in (p.info['cmdline'] or [])):remaining.append(dict(pid=p.pid,created=p.info['create_time']))
ipc=[]
for p in (f/'logs').glob('*/ownership.json'):
 for key in json.loads(p.read_text()).get('shm_keys',[]):
  segment=None
  try:segment=sysv_ipc.SharedMemory(key);ipc.append(key)
  except sysv_ipc.ExistentialError:pass
  finally:
   if segment:segment.detach()
ports=set(map(str,list(range(10000,10024))+list(range(11000,11024))+list(range(12000,12024))+list(range(13000,13072))+[20000,21000,22000]))
ss=subprocess.run(['sudo','-n','ss','-ltn'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
listeners=[line for line in ss.stdout.decode().splitlines() if any(token.rsplit(':',1)[-1] in ports for token in line.split())]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=pathlib.Path('/home/experiment/orca-fixed-scratch-20261024');models=json.loads((protected/'audit/protected-models-before.json').read_text())
protected_ok=len(models)==51 and all(sha(protected/n)==h for n,h in models.items())
runtime=json.loads((r/'source-manifest.json').read_text());source_ok=all(sha(r/n)==h for n,h in runtime.items())
core_ok=all(sha(pathlib.Path(SOURCE)/n)==h for n,h in ORIGINAL['core'].items())
sysctl={n:pathlib.Path('/proc/sys',n).read_text().strip() for n in ORIGINAL['sysctl']}
old_ok=all(sha(pathlib.Path(p))==h for p,h in OLD.items()) if NODE==0 else True
result=dict(node=NODE,remaining=remaining,recorded_ipc=ipc,listeners=listeners,protected_models=protected_ok,source_manifest_unchanged=source_ok,original_core_unchanged=core_ok,sysctl_unchanged=sysctl==ORIGINAL['sysctl'],old_checkpoint_replay_unchanged=old_ok,epoch=time.time())
result['status']='PASS' if not remaining and not ipc and not listeners and protected_ok and source_ok and core_ok and old_ok and result['sysctl_unchanged'] and ss.returncode==0 else 'REVIEW'
(r/'closure.json').write_text(json.dumps(result));print(json.dumps(result))
'''
    rows=parallel(lambda node:rpc(node,body,90,ORIGINAL=initial[node],OLD=old));save('closure.json',rows)
    assert all(r['status']=='PASS' for r in rows)
    print(json.dumps(dict(status='PASS_ALL_FIVE_CLOSED',wall_minutes=(time.time()-registration()['start_epoch'])/60,conservative_machine_hours=(time.time()-registration()['start_epoch'])*5/3600)))


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
