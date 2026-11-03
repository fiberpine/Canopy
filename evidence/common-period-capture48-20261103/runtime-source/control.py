import ast,base64,concurrent.futures,datetime,hashlib,json,pathlib,sys,time
E=pathlib.Path(__file__).parent;A=E.parent/'common-period-cluster-20261102';D0=E.parent/'common-period-D0-20261103';sys.path.insert(0,str(A));from remote import run
HOSTS=['learner']+['canopy-actor%d'%i for i in range(1,5)]
def save(name,value):(E/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def prepare():
 assert not (E/'registration.json').exists(),'Never reset this round budget'
 cells={(x['group'],x['condition']) for x in json.loads((D0/'next-proposal.json').read_text())['cells']};extra=[('A80','V17-anchor-Tconstant24-R20'),('A80','V17-mixed_timescale48-R20-q5p0'),('A80','V34-development-constant24-R100-q5p0-i0'),('A120','V34-development-constant24-R100-q5p0-i0'),('A160','V34-development-constant24-R100-q5p0-i0')];assert not (cells&set(extra));cells.update(extra);assert len(cells)==24
 old=json.loads((A/'evaluation-metrics.json').read_text());jobs=[]
 for row in old:
  j=row['job']
  if (j['group'],j['condition']) in cells:jobs.append(dict(group=j['group'],period_ms=int(j['group'][1:]),condition=j['condition'],repeat=j['repeat'],node=j['host'],old_run=row['run'],trace_sha256=j['trace_sha256'],old_config_sha256=row['source_sha256']['config.json'],old_network=row['network']))
 assert len(jobs)==48;import random;random.Random(20261103).shuffle(jobs)
 for i,j in enumerate(jobs):j.update(order=i,run='capture48-%03d-%s-r%d'%(i,j['group'],j['repeat']))
 source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (E/'code').glob('*.py')};[ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,6)) for p in (E/'code').glob('*.py')]
 now=time.time();selected=json.loads((A/'selected-model.json').read_text());reg=dict(status='FROZEN_AUTHORIZED_48',authorization='User explicitly approved38+10=48 frozen captures,45min wall/10min cleanup/4machinehours,zero optimization; D-Q separately approved only',start_epoch=now,deadline_epoch=now+2700,wall_cap_seconds=2700,cleanup_reserve_seconds=600,machine_hour_cap=4,node_count_cap=5,cloud_root='/home/experiment/orca-common-period-capture48-20261103-'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%H%M%S'),source_sha256=source,source_id=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:16],jobs=jobs,model_files=selected['files'],K=selected['K'],expected_A_sources=json.loads((A/'A-registration.json').read_text())['source_manifest'],port_base=31000,seconds=60,maximum_TCP_attempts=48,no_automatic_retries=True,optimizer_operations=0,lambda_scan=False,instrumentation_limits=dict(utilization_absolute=.03,mean_queue_absolute_ms=2,mean_queue_relative=.20,p95_absolute_ms=3,p95_relative=.20),instrumentation_rule='Each new independent run vs original samecell/samerepeat/samehost; full,startup0_10,trim10 all checked. Any margin failure downgrades cell interpretation concerning original A; no causal attribution to instrumentation vs connection variation; no rescue by favorable mean.',tolerances=dict(utility_abs=1e-10,reward_abs=1e-7,actor_abs=1e-6),S3=dict(uniform_state_points=32,boundary_event_rule='First4 crossings of abs(action)>.99; include preceding/at/following real boundary',gradient_shared_states_per_run=16),S4=dict(horizon_seconds=20,start_grid_seconds=[0,5,10,15,20,25,30,35],same_connection_only=True,tail='Frozen sameperiod mean(Q1,Q2),N; predicted amplitude/disagreement not error bound',independent_unit='TCP run;2 percell,overlapping suffixes not independent'),correction_to_user_summary=dict(shared_fc1_min_cosine=.9268996191893497,shared_fc2_min_cosine=.8999096501918127,approx_90ms_CPU_threshold='Conditional linear cost projection, not a measured full-run gate'),A_status='FAIL_A_UNCHANGED',D_Q='NOT_APPROVED_NOT_RUN')
 save('registration.json',reg);return reg
def preflight():
 reg=prepare();files={p.name:base64.b64encode(p.read_bytes()).decode() for p in (E/'code').glob('*.py')};identities=json.loads((A/'fleet-identity.json').read_text())
 def one(pair):
  n,h=pair;start=time.time();expected=next(x['hostname'] for x in identities if x['host']==h);code='REG=%r\nFILES=%r\nEXPECTED=%r\nNODE=%r\n'%(reg,files,expected,n)+r'''
import pathlib,json,hashlib,sys,shutil,base64,time,os
OLD=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');R=pathlib.Path(REG['cloud_root']);assert os.uname().nodename==EXPECTED and not R.exists();assert '4.13.1' in os.uname().release,'Expected the already-installed custom kernel';sys.path.insert(0,str(OLD/'A/framework/code/runtime/_deps'));import psutil
assert not [p.pid for p in psutil.process_iter(['cmdline','status']) if p.info['status']!=psutil.STATUS_ZOMBIE and any(str(OLD)+'/' in x for x in (p.info['cmdline'] or []))],'Old task active'
P=pathlib.Path('/home/experiment/orca-fixed-scratch-20261024');protected=json.loads((P/'audit/protected-models-before.json').read_text());assert len(protected)==51 and all(hashlib.sha256((P/n).read_bytes()).hexdigest()==v for n,v in protected.items())
selection=json.loads((OLD/'evaluation/selected-model.json').read_text());assert all(hashlib.sha256(pathlib.Path(selection['checkpoint']+k).read_bytes()).hexdigest()==v['sha256'] for k,v in REG['model_files'].items());assert all(hashlib.sha256((OLD/'A/framework'/n).read_bytes()).hexdigest()==v for n,v in REG['expected_A_sources'].items())
S=OLD/'evaluation/joint';native=json.loads((OLD/'evaluation/backend-identity.json').read_text())['native'];assert hashlib.sha256(pathlib.Path(native['joint_path']).read_bytes()).hexdigest()==native['joint_sha256'];ports={getattr(x.laddr,'port',None) for x in psutil.net_connections(kind='tcp') if x.laddr}
for job in REG['jobs']:
 if job['node']!=NODE:continue
 path=S/'logs'/job['old_run']/'config.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==job['old_config_sha256'];cfg=json.loads(path.read_text());assert cfg['condition']==job['condition'] and cfg['trace_sha256']==job['trace_sha256'];assert hashlib.sha256((S/'configs/traces'/cfg['trace']).read_bytes()).hexdigest()==job['trace_sha256'];assert REG['port_base']+job['order'] not in ports
R.mkdir();(R/'code').mkdir();F=R/'framework';F.mkdir();(F/'configs').mkdir();(F/'logs').mkdir()
for p in S.glob('*.py'):shutil.copy2(str(p),str(F/p.name))
(F/'code').symlink_to((S/'code').resolve());(F/'configs/traces').symlink_to((S/'configs/traces').resolve())
for name,value in FILES.items():
 b=base64.b64decode(value);assert hashlib.sha256(b).hexdigest()==REG['source_sha256'][name];(R/'code'/name).write_bytes(b)
 if name in ['common_period_deployment.py','evaluation_entry.py']:(F/name).write_bytes(b)
(R/'registration.json').write_text(json.dumps(REG,indent=2));out=dict(node=NODE,hostname=EXPECTED,epoch=time.time(),model_and_protected51=True,native=native,framework_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in F.glob('*.py')},sysctl={k:pathlib.Path('/proc/sys',k).read_text().strip() for k in ['net/ipv4/tcp_rmem','net/ipv4/tcp_wmem','net/core/rmem_max','net/core/wmem_max']});(R/'preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
  r=run(h,code,100);save('preflight-'+h+'.json',dict(start_epoch=start,end_epoch=time.time(),response=r));assert r['code']==0,r;return dict(node=n,status='PREPARED',jobs=sum(j['node']==n for j in reg['jobs']))
 print(json.dumps(list(concurrent.futures.ThreadPoolExecutor(5).map(one,enumerate(HOSTS)))))
def launch():
 reg=json.loads((E/'registration.json').read_text());assert time.time()<reg['deadline_epoch']-600-900
 def one(pair):
  n,h=pair;code='ROOT=%r\nNODE=%r\n'%(reg['cloud_root'],n)+r'''
import pathlib,subprocess,json,sys,time
R=pathlib.Path(ROOT);p=R/'launch.json';assert not p.exists() and not (R/'result.json').exists()
with (R/'runner.stdout').open('wb') as out:child=subprocess.Popen([sys.executable,'-B',str(R/'code/run_jobs.py'),str(NODE)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
data=dict(pid=child.pid,epoch=time.time());p.write_text(json.dumps(data));print(json.dumps(data))
'''
  r=run(h,code,30);save('launch-'+h+'.json',r);assert r['code']==0,r;return dict(node=n,data=json.loads(r['output']))
 print(json.dumps(list(concurrent.futures.ThreadPoolExecutor(5).map(one,enumerate(HOSTS)))))
def collect():
 reg=json.loads((E/'registration.json').read_text())
 def one(pair):
  n,h=pair;code='ROOT=%r\n'%reg['cloud_root']+r'''
import pathlib,json,collections,time
R=pathlib.Path(ROOT);p=R/('result.json' if (R/'result.json').exists() else 'progress.json');rows=json.loads(p.read_text()) if p.exists() else [];failure=(R/'runner-failure.txt').read_text() if (R/'runner-failure.txt').exists() else None
print(json.dumps(dict(finished=(R/'result.json').exists(),rows=rows,failure=failure,epoch=time.time())))
'''
  r=run(h,code,40);assert r['code']==0,r;x=json.loads(r['output']);save('node%d-progress.json'%n,x);return dict(node=n,data=x)
 rows=list(concurrent.futures.ThreadPoolExecutor(5).map(one,enumerate(HOSTS)));flat=[x for r in rows for x in r['data']['rows']];save('captured-run-metrics.json',flat);print(json.dumps(dict(completed=sum(x['status']=='COMPLETED' for x in flat),attempted=len([x for x in flat if not x['status'].startswith('NOT_RUN')]),finished=all(r['data']['finished'] for r in rows),problems=[dict(node=r['node'],failure=r['data']['failure'],runs=[dict(run=x.get('run'),status=x['status'],reason=x.get('reason')) for x in r['data']['rows'] if x['status']!='COMPLETED']) for r in rows if r['data']['failure'] or any(x['status']!='COMPLETED' for x in r['data']['rows'])],per_host=[dict(node=r['node'],completed=sum(x['status']=='COMPLETED' for x in r['data']['rows']),finished=r['data']['finished']) for r in rows])))
def analyze():
 reg=json.loads((E/'registration.json').read_text());assert time.time()<reg['deadline_epoch']-900
 def one(pair):
  n,h=pair;code='ROOT=%r\nNODE=%r\n'%(reg['cloud_root'],n)+r'''
import pathlib,json,subprocess,sys,time
R=pathlib.Path(ROOT);assert (R/'result.json').exists() and not (R/'analysis-results.json').exists();reg=json.loads((R/'registration.json').read_text());assert time.time()<reg['deadline_epoch']-600
with (R/'analysis.stdout').open('wb') as stream:
 p=subprocess.Popen([sys.executable,'-B',str(R/'code/analyze_jobs.py'),str(NODE)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);(R/'analysis-launch.json').write_text(json.dumps(dict(pid=p.pid,epoch=time.time())))
 try:p.wait(timeout=min(600,int(reg['deadline_epoch']-time.time()-600)))
 except subprocess.TimeoutExpired:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
assert p.returncode==0,(R/'analysis.stdout').read_text()[-4000:]
print((R/'analysis-results.json').read_text())
'''
  r=run(h,code,700);save('analysis-response-'+h+'.json',r);assert r['code']==0,r;x=json.loads(r['output']);save('node%d-analysis.json'%n,x);return dict(node=n,status=x['status'],runs=len(x['runs']),unchanged=x['model_variables_unchanged'])
 selected=[(n,h) for n,h in enumerate(HOSTS) if not (E/('node%d-analysis.json'%n)).exists() and (len(sys.argv)<3 or n==int(sys.argv[2]))]
 print(json.dumps(list(concurrent.futures.ThreadPoolExecutor(5).map(one,selected))))
if __name__=='__main__':globals()[sys.argv[1]]()
