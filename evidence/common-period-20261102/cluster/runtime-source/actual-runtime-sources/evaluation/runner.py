
import pathlib,json,os,subprocess,time,sys,hashlib,traceback
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'joint'));sys.path.insert(0,str(R/'joint/code/runtime/_deps'))
from common_period_metrics import network
from resource_cost import summarize as cost_summary
import psutil,sysv_ipc
reg=json.loads((R/'jobs.json').read_text());templates=json.loads((R/'templates.json').read_text());native=json.loads((R/'backend-identity.json').read_text())['native'];results=[]
def cleanup(d):
 owned=[]
 for p in psutil.process_iter(['pid','cmdline','status']):
  try:
   if p.info['status']!=psutil.STATUS_ZOMBIE and any(str(d)+'/' in a for a in (p.info['cmdline'] or [])):owned.append((p.pid,p.create_time()))
  except psutil.Error:pass
 for sig in ['TERM','KILL']:
  for pid,created in owned:
   try:
    p=psutil.Process(pid)
    if abs(p.create_time()-created)<.001:subprocess.run(['sudo','-n','/bin/kill','-'+sig,str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
   except psutil.NoSuchProcess:pass
  time.sleep(.1)
 return owned
for j in reg['jobs']:
 if (R/'STOP').exists() or time.time()+100>reg['deadline']-90:
  results.append(dict(job=j,status='NOT_RUN_BUDGET_OR_STOP'));continue
 group=j['group'];common=group.startswith('A');variant='mtpjoint' if common else ('baseline' if group=='H0' else 'official-orca' if group=='official-orca' else 'cubic' if group=='CUBIC' else group)
 c=dict(templates[variant]);c.pop('conditions',None);c.update(j['condition']);F=R/('scalar' if variant in ['baseline','official-orca'] else 'joint')
 name='formalA-%03d-%s-r%d'%(j['order'],group,j['repeat']);c.update(run_id=name,seconds=60,port=28500+j['order'],absolute_deadline_epoch=reg['deadline']-60,no_training=True,training_mode=False,deployment_no_step_logs=True,stop_file=str(R/'STOP'),code_commit=reg['code_commit'],legacy_symbolic_components=None)
 if c.get('checkpoint'):c['checkpoint']=c['checkpoint'].replace('@PACKAGE@','/home/experiment/orca-release-current')
 if common:
  selection=json.loads((R/'selected-model.json').read_text());c.update(checkpoint=selection['checkpoint'],source_checkpoint='/home/experiment/orca-release-current/models/mtpjoint/model',decision_cost=0,common_reward_config=reg['reward'],common_family=True,common_fixed_period_ms=int(group[1:]),server_binary=native['joint_path'],diagnostic_fixed_clock_ms=None)
 elif variant in ['baseline','official-orca']:c.update(server_binary=native['normal_path'],baseline_native_startup='normal',legacy_protocol=False)
 elif variant!='cubic':c['server_binary']=native['joint_path']
 c['interval_audit_v38']=False
 path=F/'configs'/(name+'.json');assert not path.exists();path.write_text(json.dumps(c,indent=2));outfile=R/(name+'.stdout');start=time.time()
 with outfile.open('wb') as stream:
  proc=subprocess.Popen([sys.executable,'-B',str(F/'evaluation_entry.py'),'--config',str(path)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  try:proc.wait(timeout=100)
  except subprocess.TimeoutExpired:
   proc.terminate()
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 d=F/'logs'/name;result=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else dict(status='FAILED_NO_RESULT',reason=outfile.read_text()[-2000:]);orphan=cleanup(d)
 row=dict(job=j,run=name,status=result['status'],reason=result.get('reason'),wall_s=time.time()-start,returncode=proc.returncode,orphan_cleanup=orphan)
 try:
  if result['status']!='COMPLETED':raise RuntimeError(result.get('reason'))
  row['network']=network(d/'down.log',60);row['resource_cost']=cost_summary(d,result)
  summary=d/'joint-summary.json';row['policy']=json.loads(summary.read_text()) if summary.exists() else None
  if result['cfg']['method']!='cubic' and result['policy_sha256_before']!=result['policy_sha256_after']:raise RuntimeError('Frozen policy changed')
  if any(not x['complete_window'] for x in row['network']):raise RuntimeError('Truncated packet window')
  if common and row['policy'].get('python_step_logs') is not False:raise RuntimeError('Wrong deployment logging mode')
  row['source_sha256']={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['config.json','result.json','down.log','resources.json'] if (d/n).exists()}
 except BaseException as exc:row['analysis_failure']=repr(exc);row['status']='FAILED_AUDIT'
 results.append(row);(R/'progress.json').write_text(json.dumps(results,indent=2))
 if row['status']!='COMPLETED':(R/'STOP').write_text('Frozen run failure; preserve and diagnose')
(R/'result.json').write_text(json.dumps(results,indent=2))
