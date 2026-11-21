"""Deadline-bounded five-node control; data/inference use TF directly over VPC."""
import pathlib,json,time,subprocess,sys,os,concurrent.futures,traceback,csv,hashlib,math
R=pathlib.Path(__file__).resolve().parent;PY=sys.executable
import node_control as node
A=json.loads(pathlib.Path(os.environ.get('ORCA_AUTHORIZATION_FILE',str(R/'configs/authorization.json'))).read_text());HOSTS=[x['ip'] for x in A['nodes'][1:]];DEADLINE=A['deadline_epoch']
SSH=['ssh','-i',os.environ.get('ORCA_SSH_KEY',str(R/'id_cluster')),'-o','UserKnownHostsFile='+os.environ.get('ORCA_KNOWN_HOSTS',str(R/'known_hosts')),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2']
def rpc(host,request,timeout=35):
 p=subprocess.run(([PY,str(R/'node_control.py'),'rpc'] if host=='local' else SSH+['ubuntu@'+host,PY+' '+str(R/'node_control.py')+' rpc']),input=json.dumps(request).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
 if p.returncode:raise RuntimeError(host+': '+p.stderr.decode(errors='replace')[-1500:]+p.stdout.decode(errors='replace')[-500:])
 return json.loads(p.stdout)
def parallel(fn):
 with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(HOSTS))) as pool:return list(pool.map(fn,enumerate(HOSTS)))
def progress(stage,**data):node.dump(R/'logs/controller-status.json',dict(stage=stage,epoch=time.time(),remaining_s=DEADLINE-time.time(),**data))
def write(host,path,text):return rpc(host,dict(op='write',path=str(path),text=text))
def make_config(*args,**kwargs):
 raise ValueError('Historical deployment factory is unsupported. Use tools/scalar_research.py with your explicit cluster configuration.')

def execute(cfg,duration=None):
 run=cfg['run_id'];assert time.time()<DEADLINE-600
 node.dump(R/'configs'/(run+'.json'),cfg);start=time.time();samples=[];results={};failure=None;stop_reason=None;all_ready=None;debt_since=None;bad_since={}
 sample_stream=(R/'logs'/(run+'-cluster-samples.jsonl')).open('w',buffering=262144)
 def get_status():
  rows=parallel(lambda hi:rpc(hi[1],dict(op='status',run=run)))
  for row in rows:row['entries']=[e for e in row['entries'] if e['role']=='actor']
  return rows
 def stop(reason):
  node.signal_stop(run,reason);parallel(lambda hi:rpc(hi[1],dict(op='stop',run=run,reason=reason)))
 try:
  node.start(cfg,'learner',0)
  while not (R/'logs'/run/'ready').exists():
   st=node.status(run)
   if not st['entries'][0]['alive']:raise RuntimeError('learner failed before initialization')
   if time.time()-start>100:raise RuntimeError('learner initialization timeout')
   time.sleep(1)
  def launch(hi):
   host_index,host=hi;write(host,'logs/'+run+'/ready','verified learner initialized')
   for index,owner in enumerate(cfg['actor_hosts']):
    if owner!=host_index:continue
    rpc(host,dict(op='start',cfg=cfg,role='actor',index=index));write(host,'logs/'+run+'-launch-actor'+str(index),str(time.time()))
    until=time.time()+75
    while True:
     q=rpc(host,dict(op='status',run=run));entry=next(x for x in q['entries'] if x['index']==index and x['role']=='actor')
     if entry['flow_ready']:break
     if not entry['alive']:raise RuntimeError('actor%d failed before first state'%index)
     if time.time()>until:raise RuntimeError('actor%d first state timeout'%index)
     time.sleep(.5)
   return host_index
  progress('STARTING_ACTORS',run_id=run,actors=cfg['num_actors']);parallel(launch);all_ready=time.time()
  progress('RUNNING',run_id=run,actors=cfg['num_actors'],all_ready_epoch=all_ready)
  while True:
   statuses=get_status();local=node.status(run);now=time.time();entries=[x for s in statuses for x in s['entries']]
   for entry in entries:
    i=entry['index'];p=entry['producer']
    if p is not None:
     results[i]=p;node.dump(R/'logs'/(run+'-actor%d'%i)/'producer.json',p)
    if entry['result'] and entry['result']['status']!='COMPLETED':raise RuntimeError('actor%d: %s'%(i,entry['result']['reason']))
    if not entry['alive'] and not entry['producer']:raise RuntimeError('actor%d died without producer record'%i)
   training=local['entries'][0]['training'];latest=training[-1] if training else {}
   if cfg.get('critic_only') and (R/'logs'/run/'critic-stop.json').exists() and not stop_reason:
    stop_reason=node.load(R/'logs'/run/'critic-stop.json')['reason'];stop(stop_reason)
   if cfg.get('limited_learning_guard'):
    reports=[s.get('learning_monitor') for s in statuses]
    safety=globals().get('limited_learning_guard')
    assert safety is not None,'Explicit limited-learning safety hook required'
    reason=safety(cfg,reports,latest)
    if reason and not stop_reason:stop_reason=reason;stop(reason)
   if cfg.get('require_coverage_gate') and not (R/'logs'/run/'coverage-ready.json').exists():
    minimum=cfg.get('minimum_enqueued_per_actor',512)
    coverage={e['index']:e.get('ingress_progress',{}).get('enqueued',0) for e in entries}
    if len(coverage)==cfg['num_actors'] and min(coverage.values())>=minimum:
     node.dump(R/'logs'/run/'coverage-ready.json',dict(epoch=now,all_ready_epoch=all_ready,minimum_enqueued_per_actor=minimum,per_actor_lower_bound=coverage,definition='Coordinator confirms all actors active and minimum enqueued. Learner independently gates on its exact per-actor INSERTED counters. Delayed training-log totals must never be compared against newer producer snapshots.'))
   if latest:
    debt=max(0,(int(latest['inserted'])-cfg['warmup'])*cfg['update_ratio']-int(latest['critic_iterations']))
    debt_since=(debt_since or now) if debt>1000 else None
    if not cfg['capacity_only'] and debt_since and now-debt_since>120 and not stop_reason:stop_reason='SUSTAINED_UPDATE_DEBT';stop(stop_reason)
    if not cfg['capacity_only'] and int(latest['inserted'])>=cfg.get('collapse_guard_after_samples',500000):
     for host_index,s in enumerate(statuses):
      for entry in s['entries']:
       i=entry['index'];recent=entry.get('recent',{});bad=recent.get('min_saturation',0)>=.99 and recent.get('cwnd_mean',1e6)<=10
       if bad:
        network=rpc(HOSTS[host_index],dict(op='recent_network',run=run,index=i));bad=network is not None and network['utilization'] is not None and network['utilization']<.25;entry['recent_network']=network
       bad_since[i]=(bad_since.get(i) or now) if bad else None
       if bad_since[i] and now-bad_since[i]>=cfg.get('collapse_guard_seconds',300) and not stop_reason:stop_reason='PERSISTENT_LOW_WINDOW_LOW_UTIL_ACTOR%d'%i;stop(stop_reason)
   else:debt=None
   if not stop_reason and cfg.get('sample_budget') and latest and int(latest['inserted'])>=cfg['sample_budget']:stop_reason='SAMPLE_BUDGET_COMPLETE';stop(stop_reason)
   if not stop_reason and ((duration and now-all_ready>=duration) or now>=DEADLINE-cfg.get('evaluation_reserve_s',900) or now-start>cfg['timeout_s']-90):
    stop_reason='CAPACITY_WINDOW_COMPLETE' if duration and now-all_ready>=duration else 'RESERVE_EVALUATION_OR_TIMEOUT';stop(stop_reason)
   samples.append(dict(epoch=now,learner=local,actors=statuses,debt=debt))
   sample_stream.write(json.dumps(samples[-1])+'\n');sample_stream.flush()
   progress('RUNNING',run_id=run,actors=cfg['num_actors'],all_ready_epoch=all_ready,latest=latest,debt=debt,producer_complete=len(results),stop_reason=stop_reason)
   if (R/'logs'/run/'finished').exists():
    parallel(lambda hi:write(hi[1],'logs/'+run+'/finished',(R/'logs'/run/'finished').read_text()))
    break
   if not local['entries'][0]['alive']:raise RuntimeError('learner exited without result')
   if time.time()-start>cfg['timeout_s']:raise RuntimeError('cluster bounded timeout')
   time.sleep(5)
  until=time.time()+30
  while time.time()<until:
   if all(not x['alive'] for s in get_status() for x in s['entries']):break
   time.sleep(1)
  summaries=parallel(lambda hi:rpc(hi[1],dict(op='summarize',run=run),timeout=120));actors=[x for part in summaries for x in part]
  node.dump(R/'metrics'/(run+'-actors.json'),actors)
  final=node.load(R/'logs'/run/'result.json',{})
  if final.get('status')!='COMPLETED':raise RuntimeError('learner failed: '+str(final.get('reason')))
  produced=sum(x.get('generated',0) for x in results.values());enqueued=sum(x.get('enqueued',0) for x in results.values())
  if len(results)!=cfg['num_actors'] or produced!=enqueued or enqueued!=final.get('inserted'):raise RuntimeError('generated/enqueued/inserted mismatch')
  if any(x['mapping_errors'] for x in actors):raise RuntimeError('action mapping/alignment error')
 except Exception as error:
  failure=str(error);(R/'logs'/(run+'-failure.txt')).write_text(traceback.format_exc())
  try:stop('FAILURE: '+failure)
  except Exception:pass
  time.sleep(3)
 finally:
  sample_stream.close();node.dump(R/'logs'/(run+'-cluster-samples.json'),samples)
  try:parallel(lambda hi:rpc(hi[1],dict(op='cleanup',run=run)))
  except Exception as error:(R/'logs'/(run+'-cleanup-error.txt')).write_text(str(error))
  node.cleanup(run)
  result=dict(run_id=run,status='FAILED' if failure else 'COMPLETED',reason=failure,stop_reason=stop_reason,wall_s=time.time()-start,started_epoch=start,all_ready_epoch=all_ready,final=node.load(R/'logs'/run/'result.json'),producer_records=results)
  node.dump(R/'metrics'/(run+'-cluster-result.json'),result);progress('UNIT_COMPLETE',**result)
 if failure:raise RuntimeError(failure)
 return result
def main():
 raise SystemExit('Use tools/scalar_research.py; historical capacity/formal factory is retired.')

if __name__=='__main__':main()
