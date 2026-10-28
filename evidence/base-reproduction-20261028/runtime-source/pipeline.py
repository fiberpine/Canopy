"""One registered reproduction, bounded frozen stages, no subsequent experiments."""
import pathlib,sys,os,json,time,subprocess,concurrent.futures,traceback,copy,random,base64,io,tarfile,hashlib
R=pathlib.Path(os.environ['CANOPY_REPRO_ROOT']);F=R/'framework';sys.path.insert(0,str(F));import node_control as node
M=node.load(R/'registration.json');A=node.load(R/'authorization.json');END=M['deadline_epoch'];BASE=pathlib.Path('@ARCHIVED_BASELINE@');OLD=pathlib.Path('@READONLY_RUNTIME@')
def execute(i,script,timeout=60):
    command=[sys.executable,'-B','-'] if i==0 else ['ssh','-i',str(OLD/'id_cluster'),'-o','UserKnownHostsFile='+str(OLD/'known_hosts'),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=5','ubuntu@'+A['nodes'][i]['ip'],sys.executable+' -B -']
    p=subprocess.run(command,input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    assert p.returncode==0,(i,p.stderr.decode()[-1800:],p.stdout.decode()[-1500:]);return json.loads(p.stdout)
def parallel(fn,indices=range(5)):
    with concurrent.futures.ThreadPoolExecutor(5) as pool:return list(pool.map(fn,indices))
def status(stage,**kw):node.dump(R/'results/pipeline-status.json',dict(stage=stage,epoch=time.time(),remaining_s=END-time.time(),**kw))
def stage(phase,jobs):
    status(phase.upper(),jobs=len(jobs));node.dump(R/'plans'/(phase+'.json'),jobs)
    text=json.dumps(jobs)
    def start(i):
        code='''import pathlib,sys,os,json,subprocess,time
R=pathlib.Path(ROOT);P=R/'plans'/(PHASE+'.json');P.write_text(TEXT)
assert not (R/'results'/(PHASE+'-launch.json')).exists()
with (R/(PHASE+'.stdout')).open('wb') as out:p=subprocess.Popen([sys.executable,'-B',str(R/'worker.py'),PHASE,str(INDEX)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CANOPY_REPRO_ROOT=str(R)))
r=dict(pid=p.pid,epoch=time.time());(R/'results'/(PHASE+'-launch.json')).write_text(json.dumps(r));print(json.dumps(r))
'''
        return execute(i,'ROOT=%r\nPHASE=%r\nTEXT=%r\nINDEX=%d\n'%(str(R),phase,text,i)+code)
    parallel(start)
    while time.time()<END-30:
        def read(i):return execute(i,"import pathlib,json;R=pathlib.Path(%r);p=R/'results'/(%r+'-status.json');print(p.read_text() if p.exists() else 'null')"%(str(R),phase),30)
        rows=parallel(read);status(phase.upper(),hosts=rows)
        if any(r and r['stage'] in ('FAILED','NOT_RUN_BUDGET') for r in rows):raise RuntimeError('frozen stage incomplete:'+str(rows))
        if all(r and r['stage']=='COMPLETE' for r in rows):break
        time.sleep(5)
    else:raise RuntimeError('ABSOLUTE_WINDOW_COMPLETE')
    def collect(i):return execute(i,"import pathlib,json;R=pathlib.Path(%r);p=R/'results'/(%r+'-results.json');print(p.read_text() if p.exists() else '[]')"%(str(R),phase),60)
    rows=sum(parallel(collect),[]);assert len(rows)==len(jobs);node.dump(R/'results'/(phase+'-all.json'),rows);return rows
def config(cond,run,checkpoint,tag,port):
    c=dict(cond,method='canopy',run_id=run,group=tag,seconds=60,port=port,seed=3,checkpoint=checkpoint,backend='moving_stats',code_commit='source-manifest.json',evaluation_code_commit='source-manifest.json',control_period_ms=20,history_frames=10,use_normalizer=False,no_training=True,absolute_deadline_epoch=END-20,timeout_s=110,stop_file=str(F/'logs'/(run+'-stop.json')),server_binary=str(BASE/'code/runtime/orca-server-control-log-safe'))
    return c
def export(cp,tag):
    # Export exact policy tensors only; no optimizer graph or inference calls.
    code='''import pathlib,json,sys,hashlib
sys.path[:0]=[str(F/'code/runtime/_deps')];import tensorflow as tf,numpy as np
reader=tf.train.NewCheckpointReader(CP);names=sorted(n for n in reader.get_variable_to_shape_map() if n.startswith('abstract_actor/') and '/Adam' not in n);assert len(names)==12,names
dest=F/'checkpoints'/('export-'+TAG);dest.mkdir(exist_ok=False)
variables=[]
for n in names:variables.append(tf.Variable(reader.get_tensor(n),name=n))
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)) as sess:
 sess.run(tf.global_variables_initializer());tf.train.Saver(var_list=variables).save(sess,str(dest/'model'),write_meta_graph=False)
other=tf.train.NewCheckpointReader(str(dest/'model'));assert all(np.array_equal(reader.get_tensor(n),other.get_tensor(n)) for n in names)
print(json.dumps(dict(path=str(dest/'model'),variables=names,files={str(p.relative_to(F)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('model.*')})))
'''
    proof=execute(0,'import pathlib\nF=pathlib.Path(%r)\nCP=%r\nTAG=%r\n'%(str(F),cp,tag)+code,90);node.dump(R/'results'/('export-'+tag+'.json'),proof)
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as tar:
        for n in proof['files']:
            b=(F/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
    blob=base64.b64encode(payload.getvalue()).decode()
    def send(i):
        script='''import pathlib,base64,tarfile,io,json,hashlib
F=pathlib.Path(ROOT)/'framework';data=base64.b64decode(BLOB)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 for m in tar.getmembers():
  p=F/m.name;assert m.isfile() and str(p.resolve()).startswith(str(F)+'/');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(tar.extractfile(m).read());assert hashlib.sha256(p.read_bytes()).hexdigest()==HASHES[m.name]
print(json.dumps(dict(status='POLICY_EXACT_VPC_COPY',bytes=len(data))))'''
        return execute(i,'ROOT=%r\nBLOB=%r\nHASHES=%r\n'%(str(R),blob,proof['files'])+script,90)
    node.dump(R/'results'/('export-'+tag+'-vpc.json'),parallel(send,range(1,5)))
    return proof['path']
def evaljobs(tags,conditions,repeats,prefix):
    jobs=[]
    for condition_index,cond in enumerate(conditions):
        host=condition_index%5
        methods=[(tag,cp,r) for tag,cp in tags for r in range(repeats)]
        random.Random(28102155+condition_index).shuffle(methods)
        for tag,cp,r in methods:
            run='%s-%02d-%s-r%d'%(prefix,condition_index,tag,r);jobs.append(dict(host_index=host,tag=tag,repeat=r,config=config(cond,run,cp,tag,39300+len(jobs))))
    return jobs
assert not (R/'results/pipeline-started.json').exists();node.dump(R/'results/pipeline-started.json',dict(epoch=time.time()))
try:
    # Four same-host pairs per condition across the four actor machines, one flow each.
    jobs=[];conditions=M['training_config']['conditions'];lookup={c['condition']:c for c in conditions}
    for host in range(1,5):
        order=['G12-R200','G72-R50'];random.Random(28102155+host).shuffle(order)
        for name in order:
            arms=['legacy','verified'];random.Random(host*59+len(jobs)).shuffle(arms)
            for arm in arms:
                run='ipc-short-h%d-%s-%s'%(host,name,arm);cfg=config(lookup[name],run,str(BASE/'checkpoints/formal24-grid24-seed3-balanced-u010/step-2412694'),arm,38900+len(jobs));cfg.update(server_binary=str(R/'ipc/native/orca-server-gated-verified-ipc'),verified_ipc_v2=arm=='verified',stop_file=str(R/'ipc/logs'/(run+'-stop.json')))
                jobs.append(dict(host_index=host,tag=arm,repeat=host-1,config=cfg))
    stage('ipc',jobs)
    status('TRAINING');sys.path.insert(0,str(F));os.environ['ORCA_AUTHORIZATION_FILE']=str(R/'authorization.json')
    import cluster_control
    result=cluster_control.execute(M['training_config']);node.dump(R/'results/training-result.json',result)
    final=result['final'];assert result['status']=='COMPLETED' and final['inserted']>=2400000,'TRAINING_BUDGET_NOT_COMPLETED'
    # No more training of any kind after this point.
    D=F/'checkpoints'/M['training_config']['model_id'];steps=sorted(int(p.stem.split('-')[1]) for p in D.glob('step-*.index'));last=steps[-1];chosen=[]
    for target in M['checkpoint_selection']['candidate_targets']:
        n=last if target=='final' else next((n for n in steps if target<=n<target+200000),None)
        if n is not None and n not in chosen:chosen.append(n)
    assert last in chosen;paths={n:export(str(D/('step-%d'%n)),'s%d'%n) for n in chosen}
    screenconds=[lookup[n] for n in M['checkpoint_selection']['conditions']]
    rows=stage('screen',evaljobs([('s%d'%n,paths[n]) for n in chosen],screenconds,1,'screen'))
    means=[]
    for n in chosen:
        windows=[w for r in rows if r['tag']=='s%d'%n for w in r['windows'] if w['window']=='trim10'];assert len(windows)==6
        u=sum(w['utilization'] for w in windows)/6;q=sum(w['queue_mean_ms'] for w in windows)/6;means.append(dict(inserted=n,utilization=u,queue_mean_ms=q,eligible=u>=.95 and q<=M['checkpoint_selection']['queue_max_ms']))
    eligible=[x for x in means if x['eligible']];selected=max(eligible,key=lambda x:(x['utilization'],-x['queue_mean_ms'],x['inserted']))['inserted'] if eligible else last
    selection=dict(status='SELECTED_FOR_CONFIRMATION' if eligible else 'NO_ELIGIBLE_SCREEN_CANDIDATE_FINAL_DIAGNOSTIC',screen=means,selected_inserted=selected,final_inserted=last);node.dump(R/'results/selection.json',selection)
    # A selected final model is tested twice total; otherwise final once and selected twice.
    tags=[('final',paths[last])]
    if selected!=last:tags.append(('selected',paths[selected]))
    archived=str(BASE/'checkpoints/formal24-grid24-seed3-balanced-u010/step-2412694')
    jobs=evaljobs(tags+[('archived',archived)],conditions,1,'confirm')
    extra=evaljobs([('final' if selected==last else 'selected',paths[selected])],conditions,1,'repeat')
    for j in extra:j['repeat']=1
    stage('confirm',jobs+extra)
    status('COMPLETE');node.dump(R/'results/pipeline-outcome.json',dict(status='COMPLETE',epoch=time.time(),selection=selection))
except Exception:
    status('FAILED');node.dump(R/'results/pipeline-failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
    # No blind restarts. Cleanup only this new root's recorded runs.
    def abort(i):
        code='''import pathlib,json,sys
R=pathlib.Path(ROOT);(R/'STOP_ALL').write_text('COORDINATOR_ABORT')
for folder in ['ipc','framework']:
 F=R/folder;sys.path.insert(0,str(F));import importlib.util
 spec=importlib.util.spec_from_file_location('own_node_'+folder,str(F/'node_control.py'));node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)
 for p in (F/'logs').glob('*-node-ownership.json'):
  run=p.name[:-len('-node-ownership.json')];node.signal_stop(run,'COORDINATOR_ABORT');node.cleanup(run)
print('{}')'''
        return execute(i,'ROOT=%r\n'%str(R)+code,90)
    try:parallel(abort)
    except Exception:node.dump(R/'results/abort-failure.json',dict(traceback=traceback.format_exc()))
    raise
