"""Single-flow frozen worker; phase list is fixed before execution."""
import pathlib,sys,os,json,time,subprocess,csv,traceback
R=pathlib.Path(os.environ['CANOPY_REPRO_ROOT']);phase=sys.argv[1];index=int(sys.argv[2]);F=R/('ipc' if phase=='ipc' else 'framework');sys.path[:0]=[str(F),str(F/'code/runtime/_deps')]
import node_control as node
import numpy as np
master=node.load(R/'registration.json');END=master['deadline_epoch'];jobs=node.load(R/'plans'/(phase+'.json'));jobs=[j for j in jobs if j['host_index']==index];done=[]
out=R/'results';out.mkdir(exist_ok=True);marker=out/(phase+'-started.json');assert not marker.exists();node.dump(marker,dict(epoch=time.time(),index=index,jobs=len(jobs)))
def state(stage,**kw):node.dump(out/(phase+'-status.json'),dict(stage=stage,host_index=index,epoch=time.time(),completed=len(done),total=len(jobs),**kw))
def stages(folder):
    p=folder/'control.csv.stages.csv'
    if not p.exists():return None
    metrics={};last={};missing=0;events=0
    with p.open() as stream:
        for row in csv.DictReader(stream):
            t=float(row['time_us']);ev=row['event'];events+=1
            if ev=='sample_missing':missing+=1
            pairs={'sample_poll_start':('cadence_start','cadence_sleep_ms'),'sample_ready':('sample_poll_start','sample_wait_ms'),'state_published':('sample_ready','publish_ms'),'action_applied':('state_published','policy_response_ms')}
            if ev in pairs:
                a,name=pairs[ev]
                if a in last:metrics.setdefault(name,[]).append((t-last[a])/1000)
            last[ev]=t
    return dict(events=events,missing_rtt_polls=missing,metrics={k:dict(n=len(v),mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),max=max(v)) for k,v in metrics.items()})
try:
    for job in jobs:
        if (R/'STOP_ASSESSMENT').exists() or time.time()>END-110:
            state('NOT_RUN_BUDGET',not_run=len(jobs)-len(done));break
        cfg=job['config'];run=cfg['run_id'];state('RUNNING',current=run)
        path=F/'configs'/(run+'.json');assert not path.exists();node.dump(path,cfg)
        work=F/'logs'/(run+'-work');work.mkdir(exist_ok=False)
        with (F/'logs'/(run+'.stdout')).open('wb') as stream:
            p=subprocess.Popen([sys.executable,'-B',str(F/'repair_evaluate.py'),'--config',str(path)],cwd=str(work),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            import psutil
            node.dump(node.own_path(run),[dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),role='evaluation',index=0)])
            try:code=p.wait(timeout=min(110,END-time.time()-10))
            except subprocess.TimeoutExpired:node.cleanup(run);raise RuntimeError('EVALUATION_TIMEOUT')
        folder=F/'logs'/run;result=node.load(folder/'result.json');assert code==0 and result and result['status']=='COMPLETED',(code,result)
        assert result['wall_s']>=cfg['seconds'] and result['policy_sha256_before']==result['policy_sha256_after']
        assert all(result['counters'][k]==0 for k in ['critic_iterations','critic_optimizer_ops','actor_updates','target_updates','exploration_calls'])
        summary=node.evaluation_summary(run);summary.update(host_index=index,phase=phase,tag=job['tag'],repeat=job['repeat'],native_stage_summary=stages(folder))
        # mm parser supplies full/trim10 and P95; P99 uses the same packet/window membership.
        delays={0:[],10:[]};base=None
        with (folder/'down.log').open() as stream:
            for line in stream:
                if line.startswith('# base timestamp:'):base=float(line.split(':')[1])
                elif not line.startswith('#') and base is not None:
                    x=line.split()
                    if len(x)>=4 and x[1]=='-':
                        t=(float(x[0])-base)/1000
                        for skip in (0,10):
                            if skip<=t<cfg['seconds']:delays[skip].append(float(x[3]))
        for w in summary['windows']:w['queue_p99_ms']=float(np.percentile(delays[w['start_s']],99)) if delays[w['start_s']] else None
        if cfg.get('verified_ipc_v2'):
            stats=node.load(folder/'ipc-python-stats.json');assert stats['incoming']['rejected']==0
            summary['IPC']=stats
            def read(n):
                with (folder/n).open() as stream:return list(csv.DictReader(stream))
            py,cpp=read('ipc-python.csv'),read('ipc-native.csv');directions=[]
            for sent,received in [([r for r in py if r['direction']=='sent'],[r for r in cpp if r['direction']=='received']),([r for r in cpp if r['direction']=='sent'],[r for r in py if r['direction']=='received'])]:
                assert abs(len(sent)-len(received))<=1
                assert all(tuple(a[k] for k in ['kind','sequence','crc','payload'])==tuple(b[k] for k in ['kind','sequence','crc','payload']) for a,b in zip(sent,received))
                directions.append(dict(sent=len(sent),received=len(received)))
            summary['frame_prefix']=directions
        done.append(summary);node.dump(out/(phase+'-results.json'),done);node.cleanup(run)
    else:state('COMPLETE')
except Exception:
    node.dump(out/(phase+'-failure.json'),dict(epoch=time.time(),traceback=traceback.format_exc()));state('FAILED');raise
finally:
    for j in jobs:
        if node.own_path(j['config']['run_id']).exists():node.cleanup(j['config']['run_id'])
