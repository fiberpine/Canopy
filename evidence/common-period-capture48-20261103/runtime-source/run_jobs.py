import json,pathlib,os,sys,time,subprocess,traceback
R=pathlib.Path(__file__).resolve().parent.parent;F=R/'framework';sys.path.insert(0,str(F));sys.path.insert(0,str(F/'code/runtime/_deps'))
import psutil
from light_audit import audit_run
reg=json.loads((R/'registration.json').read_text());node=int(sys.argv[1]);results=[]
def cleanup(d,port):
    owned=[];client=str(F/'code/runtime/client')
    for p in psutil.process_iter(['pid','cmdline','status']):
        try:
            a=p.info['cmdline'] or []
            if p.info['status']!=psutil.STATUS_ZOMBIE and (any(str(d)+'/' in s for s in a) or (client in a and a[-1]==str(port))):owned.append((p.pid,p.create_time()))
        except psutil.Error:pass
    for sig in ['TERM','KILL']:
        for pid,created in owned:
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-created)<.001:subprocess.run(['sudo','-n','/bin/kill','-'+sig,str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
            except psutil.NoSuchProcess:pass
        time.sleep(.15)
    return owned
try:
    for job in reg['jobs']:
        if job['node']!=node:continue
        if (R/'STOP').exists() or time.time()+110>reg['deadline_epoch']-600:
            results.append(dict(job=job,status='NOT_RUN_STOP_OR_BUDGET'));continue
        source=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102/evaluation/joint/logs')/job['old_run']/'config.json';cfg=json.loads(source.read_text());assert cfg['condition']==job['condition'];assert cfg['trace_sha256']==job['trace_sha256']
        cfg.update(run_id=job['run'],port=reg['port_base']+job['order'],absolute_deadline_epoch=reg['deadline_epoch']-600,stop_file=str(R/'STOP'),capture_raw_state=True,deployment_no_step_logs=True,no_training=True,training_mode=False,code_commit='capture48-source-'+reg['source_id'],capture_parent_code_commit=cfg['code_commit'])
        assert cfg['common_fixed_period_ms']==job['period_ms'] and cfg['decision_cost']==0 and cfg.get('selector_lambda') is None
        path=F/'configs'/(job['run']+'.json');assert not path.exists();path.write_text(json.dumps(cfg,indent=2));start=time.time()
        with (R/(job['run']+'.stdout')).open('wb') as stream:
            p=subprocess.Popen([sys.executable,'-B',str(F/'evaluation_entry.py'),'--config',str(path)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
            try:p.wait(timeout=105)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        d=F/'logs'/job['run'];orphan=cleanup(d,cfg['port'])
        try:
            row=audit_run(d,job,reg);row.update(wall_seconds=time.time()-start,returncode=p.returncode,orphan_cleanup=orphan)
        except BaseException as e:row=dict(job=job,status='FAILED_CAPTURE_AUDIT',reason=repr(e),traceback=traceback.format_exc(),returncode=p.returncode)
        results.append(row);(R/'progress.json').write_text(json.dumps(results,indent=2))
        if row['status']!='COMPLETED':(R/'STOP').write_text('Failure preserved; no automatic retry')
except BaseException:
    (R/'runner-failure.txt').write_text(traceback.format_exc());raise
finally:(R/'result.json').write_text(json.dumps(results,indent=2))
