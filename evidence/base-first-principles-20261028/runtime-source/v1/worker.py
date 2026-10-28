"""Finite frozen-only worker. One TCP flow per host; no training/optimizer graph."""
import pathlib,sys,os,json,time,subprocess,traceback,csv
R=pathlib.Path(os.environ['CANOPY_BASE_AUDIT_ROOT']);F=R/'framework';sys.path[:0]=[str(F),str(F/'code/runtime/_deps')]
import psutil
import node_control as node
master=json.loads((R/'registration.json').read_text());END=master['deadline_epoch'];index=int(sys.argv[1]);phase=sys.argv[2]
out=R/'results';out.mkdir(exist_ok=True);marker=out/(phase+'-started.json');assert not marker.exists();node.dump(marker,dict(epoch=time.time(),index=index))
jobs=[j for j in master[phase] if j['host_index']==index];done=[]
def state(stage,**kw):node.dump(out/(phase+'-status.json'),dict(stage=stage,index=index,epoch=time.time(),completed=len(done),total=len(jobs),**kw))
def protocol(run):
    d=F/'logs'/run
    def read(name):
        with (d/name).open() as stream:return list(csv.DictReader(stream))
    py,cpp=read('ipc-python.csv'),read('ipc-native.csv');counts=[]
    for sent,received in [([r for r in py if r['direction']=='sent'],[r for r in cpp if r['direction']=='received']),([r for r in cpp if r['direction']=='sent'],[r for r in py if r['direction']=='received'])]:
        n=min(len(sent),len(received));assert abs(len(sent)-len(received))<=1,(len(sent),len(received))
        assert all(tuple(a[k] for k in ['kind','sequence','crc','payload'])==tuple(b[k] for k in ['kind','sequence','crc','payload']) for a,b in zip(sent,received));counts.append(dict(compared=n,sent=len(sent),received=len(received)))
    stats=json.loads((d/'ipc-python-stats.json').read_text());assert stats['incoming']['rejected']==0
    return dict(status='PASS_FRAME_PREFIX',directions=counts,python=stats,pending_tail_at_most_one_each=True)
try:
    for job in jobs:
        if (R/'STOP_ALL').exists() or time.time()>END-95:
            state('STOPPED_BEFORE_ALL_JOBS',NOT_RUN=len(jobs)-len(done));break
        cfg=job['config'];run=cfg['run_id'];state('RUNNING',current=run);node.dump(out/'active-config.json',cfg)
        path=F/'configs'/(run+'.json');assert not path.exists();node.dump(path,cfg)
        work=F/'logs'/(run+'-work');work.mkdir(exist_ok=False)
        with (F/'logs'/(run+'.stdout')).open('wb') as stream:
            p=subprocess.Popen([sys.executable,'-B',str(F/'repair_evaluate.py'),'--config',str(path)],cwd=str(work),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            node.dump(node.own_path(run),[dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),role='evaluation',index=0)])
            try:code=p.wait(timeout=min(110,END-time.time()-10))
            except subprocess.TimeoutExpired:node.cleanup(run);raise RuntimeError('FROZEN_RUN_TIMEOUT')
        result=node.load(F/'logs'/run/'result.json');assert code==0 and result and result['status']=='COMPLETED',(code,result)
        assert result['policy_sha256_before']==result['policy_sha256_after'] and all(result['counters'][k]==0 for k in ['critic_iterations','critic_optimizer_ops','actor_updates','target_updates','exploration_calls'])
        summary=node.evaluation_summary(run);summary.update(host_index=index,pair_id=job['pair_id'],arm=job['arm'],phase=phase)
        if cfg.get('verified_ipc_v2'):summary['IPC_verified']=protocol(run)
        done.append(summary);node.dump(out/(phase+'-results.json'),done);node.cleanup(run)
    else:state('COMPLETE')
except Exception:
    node.dump(out/(phase+'-failure.json'),dict(epoch=time.time(),traceback=traceback.format_exc()));state('FAILED')
    raise
finally:
    for job in jobs:
        if node.own_path(job['config']['run_id']).exists():node.cleanup(job['config']['run_id'])
