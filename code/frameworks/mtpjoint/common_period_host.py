"""Finite six-actor host supervisor with serialized native startup and ownership."""
import argparse,json,os,pathlib,signal,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/runtime/_deps'))
import psutil

def main(cfg,path):
    root=pathlib.Path(cfg['host_root']);root.mkdir(exist_ok=True);children=[];metrics=[];status='FAILED';reason='';cache=None
    psutil.cpu_times_percent(interval=None)
    def metrics_once():
        cpu=psutil.cpu_times_percent(interval=None)._asdict();busy=sum(cpu.get(k,0) for k in ['user','nice','system','irq','softirq'])-cpu.get('guest',0)-cpu.get('guest_nice',0)
        metrics.append(dict(epoch=time.time(),busy_all_cores=busy/100.,mem_available=psutil.virtual_memory().available))
    try:
        with (root/'cache.stdout').open('wb') as stream:cache=subprocess.Popen([sys.executable,'-B',str(ROOT/'common_period_client.py'),'--config',str(path)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        for index in cfg['actor_ids']:
            with (root/('actor%02d.stdout'%index)).open('wb') as stream:
                p=subprocess.Popen([sys.executable,'-B',str(ROOT/'common_period_actor.py'),'--config',str(path),'--index',str(index)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((index,p));started=time.time()
            while not (root/('actor%02d.ready'%index)).exists():
                if p.poll() is not None:raise RuntimeError('Actor%d failed before readiness'%index)
                if cache.poll() is not None:raise RuntimeError('Snapshot cache failed before readiness')
                if time.time()-started>90:raise RuntimeError('Actor%d startup budget exceeded'%index)
                if time.time()>=cfg['deadline_epoch']-60:raise RuntimeError('Host absolute launch cutoff')
                metrics_once();time.sleep(1)
        while any(p.poll() is None for _,p in children):
            if (root/'STOP').exists() or time.time()>=cfg['deadline_epoch']-45:raise RuntimeError('Owned stop or absolute deadline')
            for index,p in children:
                if p.poll() is not None:
                    result=root/('actor%02d.result.json'%index)
                    if p.returncode or not result.exists() or json.loads(result.read_text())['status']!='COMPLETED':raise RuntimeError('Actor%d failed'%index)
            metrics_once();(root/'resources.json').write_text(json.dumps(metrics));time.sleep(1)
        results=[json.loads((root/('actor%02d.result.json'%i)).read_text()) for i,_ in children]
        if not all(r['status']=='COMPLETED' and r['generated']==r['acknowledged'] for r in results):raise RuntimeError('Actor/ingress result mismatch')
        status='COMPLETED'
    except BaseException as exc:reason=repr(exc);(root/'host-failure.txt').write_text(traceback.format_exc());(root/'STOP').write_text(reason)
    finally:
        for _,p in children:
            if p.poll() is None:p.terminate()
        for _,p in children:
            if p.poll() is None:
                try:p.wait(timeout=25)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        if cache and cache.poll() is None:cache.terminate();cache.wait(timeout=10)
        owned=[]
        for p in psutil.process_iter(['pid','cmdline','status']):
            try:
                if p.pid!=os.getpid() and p.info['status']!=psutil.STATUS_ZOMBIE and any(str(ROOT)+'/' in a for a in (p.info['cmdline'] or [])):owned.append((p.pid,p.create_time()))
            except psutil.Error:pass
        for sig in ['TERM','KILL']:
            for pid,created in owned:
                try:
                    p=psutil.Process(pid)
                    if abs(p.create_time()-created)<.001:subprocess.run(['sudo','-n','/bin/kill','-'+sig,str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
                except psutil.NoSuchProcess:pass
            time.sleep(.2)
        (root/'resources.json').write_text(json.dumps(metrics))
        (root/'host-result.json').write_text(json.dumps(dict(status=status,reason=reason,actor_ids=cfg['actor_ids'],pid=os.getpid(),epoch=time.time(),resource_samples=len(metrics),max_busy_all_cores=max([r['busy_all_cores'] for r in metrics]+[0]),mean_busy_all_cores=sum(r['busy_all_cores'] for r in metrics)/max(1,len(metrics)),min_memory_available=min([r['mem_available'] for r in metrics]+[1<<60]),orphan_processes_cleaned=owned),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();main(json.loads(pathlib.Path(a.config).read_text()),pathlib.Path(a.config))
