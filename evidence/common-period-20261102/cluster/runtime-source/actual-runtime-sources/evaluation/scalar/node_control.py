"""Small JSON control plane. No RL traffic passes through this process."""
import pathlib,json,sys,os,subprocess,time,signal,csv,traceback,hashlib
R=pathlib.Path(__file__).resolve().parent;PY=sys.executable
sys.path.insert(0,str(R/'code/runtime/_deps'));import psutil
def load(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,ValueError):return default
def dump(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(value));t.replace(p)
def own_path(run):return R/'logs'/(run+'-node-ownership.json')
def start(cfg,role,index):
 run=cfg['run_id'];path=R/'configs'/(run+'.json')
 if path.exists():assert load(path)==cfg,'configuration differs'
 else:dump(path,cfg)
 stem=run+'-'+role+str(index);work=R/'logs'/(stem+'-work');work.mkdir(exist_ok=False)
 p=subprocess.Popen([PY,str(R/'timed_distributed.py'),'--config',str(path),'--role',role,'--index',str(index)],cwd=str(work),stdout=(R/'logs'/(stem+'.stdout')).open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 data=load(own_path(run),[]);data.append(dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),role=role,index=index));dump(own_path(run),data)
 if not (R/'logs'/(run+'-watchdog.pid')).exists():
  w=subprocess.Popen([PY,str(R/'node_control.py'),'watch',run],stdout=(R/'logs'/(run+'-watchdog.stdout')).open('w'),stderr=subprocess.STDOUT,start_new_session=True)
  (R/'logs'/(run+'-watchdog.pid')).write_text(str(w.pid))
 return dict(pid=p.pid,role=role,index=index)
def tail(path,size=65536):
 if not path.exists():return []
 with path.open('rb') as f:
  header=f.readline().decode().strip().split(',');f.seek(0,2);end=f.tell();f.seek(max(0,end-size));lines=f.read().decode(errors='replace').splitlines()
 return [dict(zip(header,row)) for row in csv.reader(lines[1:-1]) if len(row)==len(header)]
def status(run):
 entries=[]
 for z in load(own_path(run),[]):
  try:
   p=psutil.Process(z['pid']);alive=abs(p.create_time()-z['created'])<.01 and p.status()!=psutil.STATUS_ZOMBIE
   cpu=p.cpu_times()._asdict();rss=p.memory_info().rss;threads=p.num_threads()
  except psutil.Error:alive=False;cpu={};rss=0;threads=0
  stem=run if z['role'] in ['learner','evaluation'] else run+'-actor'+str(z['index']);d=R/'logs'/stem
  q=dict(z,alive=alive,cpu=cpu,rss=rss,threads=threads,ready=(d/'ready').exists(),flow_ready=(d/'flow-ready').exists(),result=load(d/'result.json'),producer=load(d/'producer.json'),finished=(d/'finished').exists())
  if z['role']=='learner':q['training']=tail(d/'training.csv',8192)[-1:]
  elif z['role']=='actor':
   q['ingress_progress']=load(d/'ingress-progress.json',{})
   recent=tail(d/'states.csv',65536)[-200:]
   if recent:q['recent']=dict(step=int(recent[-1]['transition']),min_saturation=sum(float(x['action_no_noise'])<=-.99 for x in recent)/len(recent),cwnd_mean=sum(float(x['cwnd']) for x in recent)/len(recent),action_mean=sum(float(x['action_no_noise']) for x in recent)/len(recent),reward_mean=sum(float(x['reward']) for x in recent)/len(recent))
  entries.append(q)
 return dict(host=os.uname()[1],epoch=time.time(),load=list(os.getloadavg()),mem_available=psutil.virtual_memory().available,cpu_times=psutil.cpu_times()._asdict(),network=psutil.net_io_counters()._asdict(),entries=entries)
def signal_stop(run,reason):
 dump(R/'logs'/(run+'-stop.json'),dict(reason=reason,epoch=time.time()))
def recent_network(run,index):
 path=R/'logs'/(run+'-actor%d'%index)/'down.log'
 if not path.exists():return None
 with path.open('rb') as f:
  f.seek(0,2);end=f.tell();f.seek(max(0,end-1000000));lines=f.read().decode(errors='replace').splitlines()[1:]
 events=[]
 for line in lines:
  fields=line.split()
  if len(fields)>=3:
   try:events.append((float(fields[0]),fields[1],int(fields[2])))
   except ValueError:pass
 if not events:return None
 cutoff=events[-1][0]-5000;capacity=sum(size for t,kind,size in events if t>=cutoff and kind=='#');departure=sum(size for t,kind,size in events if t>=cutoff and kind=='-')
 return dict(utilization=departure/capacity if capacity else None,window_ms=events[-1][0]-max(cutoff,events[0][0]))
def start_evaluation(cfg):
 run=cfg['run_id'];path=R/'configs'/(run+'.json');assert not path.exists();dump(path,cfg)
 work=R/'logs'/(run+'-work');work.mkdir(exist_ok=False)
 entry='remote_frozen_evaluate.py' if cfg.get('inference_endpoint') else 'frozen_evaluate.py'
 if cfg.get('evaluation_entry'):
  assert cfg['evaluation_entry']=='repair_evaluate.py';entry=cfg['evaluation_entry']
 p=subprocess.Popen([PY,str(R/entry),'--config',str(path)],cwd=str(work),stdout=(R/'logs'/(run+'.stdout')).open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 dump(own_path(run),[dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),role='evaluation',index=0)])
 subprocess.Popen([PY,str(R/'node_control.py'),'watch',run],stdout=(R/'logs'/(run+'-watchdog.stdout')).open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 return dict(pid=p.pid)
def evaluation_summary(run):
 import numpy as np
 from mahimahi_metrics import mm,csvread
 d=R/'logs'/run;result=load(d/'result.json');cfg=result['cfg'];out=[]
 if result['status']=='COMPLETED':
  records,base,bins=mm(d/'down.log',cfg['seconds']);states=csvread(d/'states.csv');control=csvread(d/'control.csv')
  for record in records:
   rows=[x for x in states if record['start_s']<=(float(x['time_us'])/1000-base)/1000<cfg['seconds']]
   c=[float(x['time_us']) for x in control if record['start_s']<=(float(x['time_us'])/1000-base)/1000<cfg['seconds']];period=np.diff(c)/1000
   extra={}
   for field in ['reward','throughput_term','loss_term','delay_term','cwnd','action_no_noise','max_bw']:extra[field+'_mean']=float(np.mean([float(x[field]) for x in rows])) if rows else None
   extra.update(min_saturation=float(np.mean([float(x['action_no_noise'])<=-.99 for x in rows])) if rows else None,max_saturation=float(np.mean([float(x['action_no_noise'])>=.99 for x in rows])) if rows else None,neutral_percent=float(np.mean([int(x['percent'])==100 for x in rows])) if rows else None)
   for percentile in [50,95,99]:extra['control_p%d_ms'%percentile]=float(np.percentile(period,percentile)) if len(period) else None
   out.append(dict(run_id=run,group=cfg['group'],condition=cfg['condition'],model_source=cfg.get('checkpoint'),policy_unchanged=result['policy_sha256_before']==result['policy_sha256_after'],policy_hash=result['policy_sha256_after'],**record,**extra))
 response=dict(result=result,windows=out);dump(R/'metrics'/(run+'-summary.json'),response);return response
def cleanup(run):
 # Every target is tied to this root, run and recorded creation time.
 records=load(own_path(run),[]);targets={}
 for z in records:
  try:
   p=psutil.Process(z['pid'])
   if abs(p.create_time()-z['created'])<.01:
    for q in p.children(recursive=True)+[p]:targets[q.pid]=q.create_time()
  except psutil.Error:pass
 for p in psutil.process_iter(['pid','cmdline','create_time','status']):
  args=p.info['cmdline'] or []
  if any(str(R/'logs'/run) in a for a in args) and p.pid!=os.getpid():
   targets[p.pid]=p.info['create_time']
   try:
    for q in p.children(recursive=True):targets[q.pid]=q.create_time()
   except psutil.Error:pass
 for sig in [signal.SIGTERM,signal.SIGKILL]:
  for pid,created in sorted(targets.items(),reverse=True):
   try:
    p=psutil.Process(pid)
    if abs(p.create_time()-created)<.01:
     try:p.send_signal(sig)
     except psutil.AccessDenied:subprocess.run(['sudo','-n','kill','-'+str(int(sig)),'--',str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   except psutil.Error:pass
  time.sleep(.3)
 # Shared memory is removed only for verified ownership from this run.
 import sysv_ipc
 ipc=[]
 for path in (R/'logs').glob(run+'*/ownership.json'):
  z=load(path,{})
  for key in z.get('shm_keys',[]):
   shm=None
   try:
    shm=sysv_ipc.SharedMemory(key)
    if shm.creator_pid==z['pid'] and shm.cuid==os.getuid() and shm.number_attached==1:shm.remove();ipc.append(dict(key=key,removed=True))
   except sysv_ipc.ExistentialError:pass
   finally:
    if shm:shm.detach()
 dump(R/'audit'/(run+'-cleanup.json'),dict(targets=targets,ipc=ipc,epoch=time.time()))
 return dict(targets=len(targets),ipc=ipc)
def summarize(run):
 import numpy as np
 from mahimahi_metrics import csvread,mm
 result=[]
 for z in load(own_path(run),[]):
  if z['role']!='actor':continue
  i=z['index'];d=R/'logs'/(run+'-actor'+str(i));states=csvread(d/'states.csv');control=csvread(d/'control.csv');timing=csvread(R/'logs'/(run+'-actor%d-timing.csv'%i))
  periods=np.diff([float(x['time_us']) for x in control])/1000
  rpc=[float(x['duration_ms']) for x in timing if x['kind']=='inference_rpc'][1:]
  def stats(v):return dict(mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99))) if len(v) else {}
  errors=sum(x['state_id']!=x['action_id'] or int(x['return_code'])!=0 or int(x['cwnd_requested'])!=min(2147483647,max(4,int(x['percent'])*int(x['cwnd_before'])//100)) for x in control)
  q=dict(index=i,producer=load(d/'producer.json'),result=load(d/'result.json'),valid_rows=len(states),control=stats(periods),rpc=stats(rpc),mapping_errors=errors,reward_mean=float(np.mean([float(x['reward']) for x in states])) if states else None)
  if states:
   q.update(cwnd_mean=float(np.mean([float(x['cwnd']) for x in states])),min_saturation=float(np.mean([float(x['action_no_noise'])<=-.99 for x in states])),max_saturation=float(np.mean([float(x['action_no_noise'])>=.99 for x in states])))
  result.append(q)
 dump(R/'metrics'/(run+'-actor-summary.json'),result);return result
def watch(run):
 cfg=load(R/'configs'/(run+'.json'));deadline=min(cfg['absolute_deadline_epoch'],time.time()+cfg.get('timeout_s',3600))
 while time.time()<deadline:
  s=status(run)
  if s['entries'] and all(not x['alive'] for x in s['entries']):return
  time.sleep(5)
 signal_stop(run,'NODE_ABSOLUTE_DEADLINE');time.sleep(10);cleanup(run)
def main():
 cmd=sys.argv[1]
 if cmd=='watch':watch(sys.argv[2]);return
 if cmd=='rpc':
  request=json.load(sys.stdin);cmd=request['op'];run=request.get('run')
  if cmd=='start':out=start(request['cfg'],request['role'],request['index'])
  elif cmd=='start_evaluation':out=start_evaluation(request['cfg'])
  elif cmd=='evaluation_summary':out=evaluation_summary(run)
  elif cmd=='recent_network':out=recent_network(run,request['index'])
  elif cmd=='status':out=status(run)
  elif cmd=='stop':signal_stop(run,request['reason']);out=True
  elif cmd=='cleanup':out=cleanup(run)
  elif cmd=='summarize':out=summarize(run)
  elif cmd=='write':
   path=(R/request['path']).resolve();assert str(path).startswith(str(R)+'/');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(request['text']);out=True
  elif cmd=='read':out=load(R/request['path'])
  else:raise ValueError(cmd)
  print(json.dumps(out),flush=True)
if __name__=='__main__':main()
