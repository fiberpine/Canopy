#!/usr/bin/env python3
"""Portable entrypoint. Fresh runs only; original model/optimizer math is retained."""
import argparse,copy,json,os,pathlib,subprocess,sys,time,uuid,shutil
P=pathlib.Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def localize(x,f):
 if isinstance(x,str):return x.replace('@PACKAGE@',str(P)).replace('@FRAMEWORK@',str(f))
 if isinstance(x,list):return [localize(y,f) for y in x]
 if isinstance(x,dict):return {k:localize(v,f) for k,v in x.items()}
 return x
def template(variant,mode):
 d=read(P/'templates'/(variant+'-'+mode+'.json'));f=P/'frameworks'/d.pop('framework');return f,localize(d,f)
def python():return os.environ.get('ORCA_PYTHON',str(P/'runtime/venv/bin/python'))
def call(args,**kw):return subprocess.check_call([python()]+[str(x) for x in args],**kw)
def runid(label):return label+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
def allocate(f,r):
 if (f/'logs'/r).exists() or (f/'configs'/(r+'.json')).exists():raise ValueError('Run ID already exists: '+r)
 for n in ['logs','audit','metrics','report','plots','checkpoints','configs']:(f/n).mkdir(exist_ok=True)
def auth(f,r,cluster,cap):
 a=read(pathlib.Path(cluster)) if cluster else dict(nodes=[dict(ip='127.0.0.1'),dict(ip='local')])
 assert len(a['nodes'])>=2,'One learner and >=1 actor location required (local is supported)'
 now=time.time();a.update(start_epoch=now,deadline_epoch=now+max(1200,cap),authorization='Explicit portable CLI invocation; bounded fresh run; no automatic continuation')
 q=f/'configs'/(r+'-authorization.json');write(q,a);return a,q
def doctor():
 z=dict(package=str(P),python=python(),kernel=os.uname().release,tools={x:shutil.which(x) for x in ['g++','mm-link','mm-delay','sudo','ip']},tun=pathlib.Path('/dev/net/tun').exists())
 q=subprocess.run([python(),'-c','import sys,tensorflow,sonnet,sysv_ipc,interval_bound_propagation;print(sys.version);print(tensorflow.__version__)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);z['runtime_imports']=q.returncode==0;z['runtime_detail']=q.stdout.decode();z['runtime_error']=q.stderr.decode()[-500:];z['kernel_required']='4.13.1-0521-learner custom TCP ABI';z['compatible_kernel']=os.uname().release=='4.13.1-0521-learner'
 print(json.dumps(z,indent=2));return 0 if z['runtime_imports'] and z['compatible_kernel'] and z['tun'] and all(z['tools'].values()) else 2
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='cmd');sp.add_parser('doctor')
 for cmd in ['evaluate','train','smoke-training','verify-training']:
  p=sp.add_parser(cmd);p.add_argument('--variant',required=True);p.add_argument('--run-id');p.add_argument('--scenario',type=int,default=0);p.add_argument('--seconds',type=float,default=90 if cmd=='smoke-training' else 60);p.add_argument('--actors',type=int,default=1);p.add_argument('--samples',type=int,default=2400000);p.add_argument('--seconds-per-actor',type=float);p.add_argument('--cluster');p.add_argument('--port',type=int,default=23000);p.add_argument('--learner-port',type=int,default=25000)
 a=ap.parse_args()
 if a.cmd=='doctor':return doctor()
 if not a.cmd:ap.print_help();return 2
 if a.cmd in ['evaluate','verify-training']:
  f,c=template(a.variant,'eval');r=a.run_id or runid(a.cmd+'-'+a.variant);allocate(f,r);cs=c.pop('conditions');condition=cs[a.scenario];c.update(condition);c.update(run_id=r,group=a.variant,seconds=a.seconds,port=a.port,seed=3,absolute_deadline_epoch=time.time()+max(600,a.seconds+120),timeout_s=a.seconds+120,no_training=True,code_commit='portable-release',evaluation_code_commit='portable-release')
  p=f/'configs'/(r+'.json');write(p,c)
  if a.cmd=='evaluate':call([f/'repair_evaluate.py','--config',p],cwd=str(f));result=read(f/'logs'/r/'result.json');assert result['status']=='COMPLETED',result;print(json.dumps(dict(run_id=r,result=str(f/'logs'/r/'result.json'),status=result['status'])))
  else:
   _,t=template(a.variant,'train');t['warmup']=max(t['warmup'],10240);t['learner_endpoint']='127.0.0.1:45000';t['num_actors']=1;out=f/'audit'/(r+'.json');tp=f/'configs'/(r+'-training-template.json');write(tp,t);call([P/'tools/verify_training.py','--framework',f,'--config',tp,'--checkpoint',c['checkpoint'],'--output',out]);print(out.read_text())
  return 0
 if a.variant not in ['baseline','mtp20','mtpjoint']:raise ValueError('Production training supports three preserved implementations')
 f,c=template(a.variant,'train');r=a.run_id or runid(a.cmd+'-'+a.variant);allocate(f,r);smoke=a.cmd=='smoke-training';n=a.actors;assert n>=1
 seconds=a.seconds if smoke else (a.seconds_per_actor or c.get('physical_seconds_per_actor',600));cap=1200 if smoke else (max(1200,int(seconds)+900) if c.get('joint_control') else max(1200,int(a.samples/(40*n))+900))
 A,af=auth(f,r,a.cluster,cap);hosts=[x['ip'] for x in A['nodes'][1:]];cs=c['conditions'];c['conditions']=[copy.deepcopy(cs[i%len(cs)]) for i in range(n)]
 if c.get('actor_environment_sequences'):c['actor_environment_sequences']=[copy.deepcopy(c['actor_environment_sequences'][i%len(c['actor_environment_sequences'])]) for i in range(n)]
 for key in ['condition_sequences','actor_sequences']:
  if isinstance(c.get(key),list):c[key]=[copy.deepcopy(c[key][i%len(c[key])]) for i in range(n)]
 c.update(run_id=r,model_id=r,num_actors=n,actor_hosts=[i%len(hosts) for i in range(n)],learner_endpoint=A['nodes'][0]['ip']+':'+str(a.learner_port),actor_port_base=a.port,absolute_deadline_epoch=A['deadline_epoch']-60,stop_file=str(f/'logs'/(r+'-stop.json')),timeout_s=cap-120,shutdown_barrier_s=120,evaluation_reserve_s=90,code_commit='portable-release',capacity_only=False,steps_per_actor=1000000)
 if c.get('joint_control'):c.update(physical_seconds_per_actor=seconds,sample_budget=None)
 else:c.update(sample_budget=2048*n if smoke else a.samples)
 if smoke:
  c.update(portable_smoke_training=True,warmup=512,minimum_enqueued_per_actor=512,warmup_min_physical_seconds=0,timing_critic_calibration_updates=1,timing_min_retained_per_period=1,require_coverage_gate=True,pretrained_window_health_guard=False,timing_action_health_guard=False,memsize=10000,checkpoint_interval=100000000,configuration_notes='Short real-TCP packaging smoke; reduced warmup/calibration only; not production training or performance evidence.')
  if c.get('joint_control'):c['timing_init_ms']=40
  if c.get('period_bank_enabled'):c.update(period_bank_rows_per_bin=16)
 p=f/'configs'/(r+'-input.json');write(p,c);env=dict(os.environ,ORCA_AUTHORIZATION_FILE=str(af))
 if A.get('ssh_key_path'):env['ORCA_SSH_KEY']=A['ssh_key_path']
 if A.get('known_hosts_path'):env['ORCA_KNOWN_HOSTS']=A['known_hosts_path']
 code="import sys,json,pathlib;sys.path.insert(0,sys.argv[1]);from cluster_control import execute;c=json.loads(pathlib.Path(sys.argv[2]).read_text());print(json.dumps(execute(c)))"
 call(['-B','-c',code,f,p],env=env,cwd=str(f));q=read(f/'metrics'/(r+'-cluster-result.json'));assert q['status']=='COMPLETED',q
 counters=q['final']['counters'];assert counters['critic_iterations']>0 and counters['actor_updates']>0,'No real optimizer update; smoke is not PASS'
 print(json.dumps(dict(status='PASS_REAL_TCP_TRAINING' if smoke else 'COMPLETED_TRAINING',run_id=r,counters=counters,result=str(f/'metrics'/(r+'-cluster-result.json')))))
 return 0
if __name__=='__main__':sys.exit(main())
