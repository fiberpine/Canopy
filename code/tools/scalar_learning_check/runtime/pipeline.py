"""Finite L1: zero-update24-actor smoke, fresh bounded learning, then110 frozen checks."""
import pathlib,sys,os,json,time,threading,subprocess,concurrent.futures,hashlib,random,csv,traceback
R=pathlib.Path(os.environ['CANOPY_L1_ROOT']);F=R/'framework';PY=sys.executable
sys.path[:0]=[str(F),str(F/'code/runtime/_deps'),str(R/'tools/scalar_learning_check')]
os.environ.update(ORCA_AUTHORIZATION_FILE=str(R/'authorization.json'),ORCA_SSH_KEY=str(R/'id_cluster'),ORCA_KNOWN_HOSTS=str(R/'known_hosts'))
import node_control as node,cluster_control as cluster
master=json.loads((R/'registration.json').read_text());cfg=master['training_config'];deadline=master['deadline_epoch']
HOSTS=['local']+cluster.HOSTS;heartbeat_stop=threading.Event();monitor_records={};last_window={}
def put(name,value):node.dump(R/'audit'/name,value)
def heartbeat():
    while not heartbeat_stop.wait(8):
        (R/'audit/lease-heartbeat').touch()
        for host in cluster.HOSTS:
            try:cluster.rpc(host,dict(op='lease_heartbeat'),timeout=15)
            except Exception:pass
def guard(training,reports,latest):
    rid=training['run_id'];history=monitor_records.setdefault(rid,[])
    exploded=[i for p in reports if p for i in p.get('exploded',[])]
    if exploded:
        row=dict(window=-1,S1='OK',S2='OK',S3='STOP',exploded=exploded,epoch=time.time());history.append(row);put('monitor-'+rid+'.json',dict(stop=True,rules=history));return 'L1_S3_WINDOW_EXPLOSION'
    actors={int(i):a for p in reports if p for i,a in p.get('actors',{}).items()}
    if len(actors)==24 and all(a['windows'] for a in actors.values()):
        until=min(a['windows'][-1]['window'] for a in actors.values())
        for k in range(last_window.get(rid,-1)+1,until+1):
            rows={i:next(w for w in a['windows'] if w['window']==k) for i,a in actors.items()}
            sat=[i for i,w in rows.items() if w['sat_low']>=.5 or w['sat_high']>=.5];low=[i for i,w in rows.items() if w['low_window_fraction']>=.5]
            level=lambda n:'STOP' if n>=12 else 'FLAG' if n>=3 else 'OK'
            row=dict(window=k,start_s=k*30,S1=level(len(sat)),S2=level(len(low)),S3='OK',saturated=sat,low_window=low,
                actors_detail=rows,inserted=int(latest.get('inserted',0)),epoch=time.time());history.append(row);last_window[rid]=k
            put('monitor-'+rid+'.json',dict(stop=any(r[x]=='STOP' for r in history for x in ['S1','S2','S3']),rules=history))
            if row['S1']=='STOP' or row['S2']=='STOP':return 'L1_'+('S1_SATURATION' if row['S1']=='STOP' else 'S2_LOW_WINDOW')
    return None
cluster.limited_learning_guard=guard
def active(training):
    # node.write is intentionally confined to framework. Explicit lease RPC below writes parent audit.
    node.dump(R/'audit/active-training-config.json',training)
    cluster.parallel(lambda hi:cluster.rpc(hi[1],dict(op='set_active_training',cfg=training)))
def export_and_transfer(result):
    cpdir=F/'checkpoints'/cfg['model_id'];steps=sorted(int(p.name.split('-')[1].split('.')[0]) for p in cpdir.glob('step-*.index'))
    stages={0:0,200000:min(n for n in steps if n>=200000),400000:min(n for n in steps if n>=400000),800000:min(n for n in steps if n>=800000),1200000:max(steps)}
    records={}
    for stage,actual in stages.items():
        source=cpdir/('step-%d'%actual);folder=F/'deploy_models'/str(stage);folder.mkdir(parents=True,exist_ok=False);out=folder/'actor'
        p=subprocess.run([PY,'-B',str(R/'export_actor.py'),'--source',str(source),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        assert p.returncode==0,p.stderr.decode()[-2000:];record=json.loads(p.stdout);record['actual_step']=actual;records[str(stage)]=record
        files=[out.with_suffix('.index')]+list(folder.glob('*.data-*'))
        for host in cluster.HOSTS:
            cluster.rpc(host,dict(op='write',path='deploy_models/'+str(stage)+'/deployment.json',text=json.dumps(record)))
            for file in files:
                subprocess.run(['scp','-q','-i',str(R/'id_cluster'),'-o','UserKnownHostsFile='+str(R/'known_hosts'),'-o','StrictHostKeyChecking=yes',str(file),'ubuntu@'+host+':'+str(file)],check=True,timeout=60)
    put('checkpoint-stages.json',records);return records
def dispatch_evaluation(records):
    cells=master['evaluation']['cells'];conditions={c['condition']:c for c in cfg['conditions']}
    shuffled=list(cells);random.Random(27102027).shuffle(shuffled);owners={c:i%5 for i,c in enumerate(shuffled)}
    assigned={i:[] for i in range(5)}
    for number,spec in enumerate(master['evaluation']['jobs']):
        stage=spec['checkpoint'];arm=spec['arm'];cond=conditions[spec['condition']];rid='L1-eval-%03d'%number
        model='cubic' if arm=='CUBIC' else 'orca'
        c=dict(cond,method=model,run_id=rid,group='L1-'+str(stage)+'-'+arm,seconds=60,port=42000+number,
            seed=3+spec.get('repeat',0),no_training=True,backend='moving_stats',control_period_ms=20,
            absolute_deadline_epoch=deadline,timeout_s=120,evaluation_entry='limited_evaluate.py',
            code_commit=cfg['code_commit'],diagnostic_goodput_floor=arm=='A4g')
        actual=None
        if model!='cubic':
            c.update(verified_ipc_v2=True,server_binary=cfg['server_binary'],require_normal_cubic_start=True)
            if stage=='old':c['checkpoint']=master['old_checkpoint'];actual=3557457
            else:c['checkpoint']=str(F/'deploy_models'/str(stage)/'actor');actual=records[str(stage)]['actual_step']
        assigned[owners[spec['condition']]].append(dict(config=c,stage_specification=spec,actual_checkpoint_step=actual))
    put('evaluation-assignment.json',dict(condition_host=owners,counts={k:len(v) for k,v in assigned.items()}))
    for i,host in enumerate(HOSTS):
        tasks=assigned[i];random.Random(27102028+i).shuffle(tasks);request=dict(tasks=tasks)
        if host=='local':(R/'eval-registration.json').write_text(json.dumps(request))
        else:cluster.rpc(host,dict(op='set_eval_registration',registration=request))
        if host=='local':
            f=(R/'eval-worker.stdout').open('w');p=subprocess.Popen([PY,'-B',str(R/'eval_worker.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CANOPY_L1_ROOT=str(R)));f.close();put('eval-worker-launch.json',dict(pid=p.pid))
        else:cluster.rpc(host,dict(op='start_eval_worker'))
    while True:
        progress=[]
        for host in HOSTS:
            x=node.load(R/'audit/evaluation-status.json') if host=='local' else cluster.rpc(host,dict(op='read_root_audit',name='evaluation-status.json'))
            progress.append(x)
        put('pipeline-status.json',dict(stage='FROZEN_EVALUATION',completed=sum(x.get('completed',0) for x in progress if x),total=110,hosts=progress,epoch=time.time()))
        assert not any(x and x.get('stage')=='FAILED' for x in progress),progress
        assert time.time()<deadline-45
        if all(x and x.get('stage')=='COMPLETE' for x in progress):break
        time.sleep(5)
    summaries=[]
    for host in HOSTS:summaries+=node.load(R/'audit/evaluation-results.json') if host=='local' else cluster.rpc(host,dict(op='read_root_audit',name='evaluation-results.json'),timeout=60)
    assert len(summaries)==110;put('evaluation-all.json',summaries)
    rows=[]
    for x in summaries:
        for window in x['windows']:
            rows.append(dict(window,checkpoint=x['stage_specification']['checkpoint'],arm=x['stage_specification']['arm'],actual_checkpoint_step=x['actual_checkpoint_step']))
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (R/'audit/evaluation-runs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    return rows
def main():
    assert not (R/'audit/pipeline-started.json').exists();put('pipeline-started.json',dict(epoch=time.time(),pid=os.getpid()))
    threading.Thread(target=heartbeat,daemon=True).start();result=None
    try:
        smoke=dict(cfg,run_id=cfg['run_id']+'-smoke',model_id=cfg['model_id']+'-smoke',diagnostic_no_optimizer=True,capacity_only=True,
            warmup=512,minimum_enqueued_per_actor=64,portable_smoke_training=True,sample_budget=None,timeout_s=240,checkpoint_interval=10000000)
        smoke['stop_file']=str(F/'logs'/(smoke['run_id']+'-stop.json'));active(smoke)
        put('pipeline-status.json',dict(stage='ZERO_UPDATE_24ACTOR_SMOKE',epoch=time.time()))
        first=cluster.execute(smoke,duration=30);put('smoke-result.json',first);assert first['stop_reason']=='CAPACITY_WINDOW_COMPLETE' and first['final']['counters']['critic_iterations']==0
        active(cfg);put('pipeline-status.json',dict(stage='FINITE_LEARNING',epoch=time.time(),sample_budget=1200000))
        result=cluster.execute(cfg);put('training-result.json',result)
        if result['stop_reason']!='SAMPLE_BUDGET_COMPLETE':
            put('outcome.json',dict(status='TRAINING_STOPPED_BEFORE_FULL_EVALUATION',stop_reason=result['stop_reason'],evaluation_NOT_RUN=110,actual_inserted=result['final']['inserted']));return
        records=export_and_transfer(result);dispatch_evaluation(records)
        from adjudicate_learning import load_rows,adjudicate
        monitor=node.load(R/'audit'/('monitor-'+cfg['run_id']+'.json'),dict(stop=False,rules=[]))
        verdict=adjudicate(load_rows(R/'audit/evaluation-runs.csv'),master,monitor)
        verdict.update(actual_final_step=result['final']['inserted'],checkpoint_labels='Predeclared budget stages; actual_checkpoint_step in every row',automatic_continuation=False)
        put('adjudication.json',verdict);put('outcome.json',dict(status='COMPLETE',training=result['final'],evaluation_runs=110,decision=verdict['decision'],no_further_training=True))
    except Exception:
        put('failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        (R/'audit/STOP_EVALUATION').touch()
        try:cluster.parallel(lambda hi:cluster.rpc(hi[1],dict(op='stop_evaluation')))
        except Exception:pass
        # Each frozen worker cleans its current run; no subsequent TCP is allowed.
        time.sleep(5)
        raise
    finally:
        heartbeat_stop.set()
        try:
            cluster.parallel(lambda hi:cluster.rpc(hi[1],dict(op='end_lease')))
        finally:(R/'audit/LEASE_COMPLETE').touch()
if __name__=='__main__':main()
