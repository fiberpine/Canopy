import concurrent.futures,json,pathlib,sys,time
E=pathlib.Path(__file__).parent;A=E.parent/'common-period-cluster-20261102';sys.path.insert(0,str(A));from remote import run
reg=json.loads((E/'registration.json').read_text());hosts=['learner']+['canopy-actor%d'%i for i in range(1,5)]
def one(pair):
 n,h=pair;start=time.time();code='REG=%r\nNODE=%r\n'%(reg,n)+r'''
import pathlib,json,hashlib,sys,time
R=pathlib.Path(REG['cloud_root']);OLD=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');sys.path.insert(0,str(R/'framework/code/runtime/_deps'));import psutil,sysv_ipc
pre=json.loads((R/'preflight.json').read_text());owned=[]
for p in psutil.process_iter(['pid','cmdline','status']):
 try:
  if p.info['status']!=psutil.STATUS_ZOMBIE and any(str(R)+'/' in a for a in (p.info['cmdline'] or [])):owned.append(dict(pid=p.pid,command=p.info['cmdline']))
 except psutil.Error:pass
ipc=[]
for p in (R/'framework/logs').glob('*/ownership.json'):
 own=json.loads(p.read_text())
 for key in own['shm_keys']:
  try:
   m=sysv_ipc.SharedMemory(key);ipc.append(dict(key=key,creator=m.creator_pid,expected_creator=own.get('shm_creator_pid')));m.detach()
  except sysv_ipc.ExistentialError:pass
P=pathlib.Path('/home/experiment/orca-fixed-scratch-20261024');protected=json.loads((P/'audit/protected-models-before.json').read_text());protected_ok=len(protected)==51 and all(hashlib.sha256((P/n).read_bytes()).hexdigest()==v for n,v in protected.items());sel=json.loads((OLD/'evaluation/selected-model.json').read_text());model_ok=all(hashlib.sha256(pathlib.Path(sel['checkpoint']+k).read_bytes()).hexdigest()==v['sha256'] for k,v in REG['model_files'].items());native=pre['native'];native_ok=hashlib.sha256(pathlib.Path(native['joint_path']).read_bytes()).hexdigest()==native['joint_sha256'];sysctl={k:pathlib.Path('/proc/sys',k).read_text().strip() for k in pre['sysctl']}
out=dict(node=NODE,epoch=time.time(),owned_processes=owned,recorded_IPC_remaining=ipc,protected51_unchanged=protected_ok,model_unchanged=model_ok,native_unchanged=native_ok,sysctl_unchanged=sysctl==pre['sysctl'],source_unchanged=all(hashlib.sha256((R/'framework'/name).read_bytes()).hexdigest()==h for name,h in pre['framework_sha256'].items()),capture_files=[dict(run=p.parent.name,bytes=p.stat().st_size) for p in (R/'framework/logs').glob('*/captured-boundaries.npz')])
out['status']='PASS_CLOSED_FROZEN' if not owned and not ipc and protected_ok and model_ok and native_ok and out['sysctl_unchanged'] and out['source_unchanged'] else 'FAIL';(R/'cleanup.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
 r=run(h,code,80);assert r['code']==0,r;x=json.loads(r['output']);(E/('cleanup-node%d.json'%n)).write_text(json.dumps(x,indent=2));return dict(node=n,host=h,start_epoch=start,end_epoch=time.time(),data=x)
results=list(concurrent.futures.ThreadPoolExecutor(5).map(one,enumerate(hosts)));end=time.time();machine_seconds=sum(end-json.loads((E/('preflight-'+h+'.json')).read_text())['start_epoch'] for h in hosts);out=dict(status='PASS_CLOSED_FROZEN' if all(r['data']['status']=='PASS_CLOSED_FROZEN' for r in results) else 'FAIL',nodes=results,ended_epoch=end,wall_elapsed_seconds=end-reg['start_epoch'],reserved_machine_hours=machine_seconds/3600,accounting='Conservative continuous five-node reservation through final check, including idle intervals. Not provider bill.',instances_shutdown=False);assert out['wall_elapsed_seconds']<=2700 and out['reserved_machine_hours']<=4;(E/'cleanup.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],wall_seconds=out['wall_elapsed_seconds'],machine_hours=out['reserved_machine_hours'],capture_files=sum(len(r['data']['capture_files']) for r in results))));assert out['status']=='PASS_CLOSED_FROZEN'
