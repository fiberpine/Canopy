"""Finite zero-update high-RTT recheck under enlarged buffers; restore original settings."""
import json,os,pathlib,subprocess,sys,time,traceback,signal
R=pathlib.Path(os.environ['CANOPY_IPC_DIAGNOSTIC_ROOT'])
sys.path.insert(0,str(R/'tools/scalar_ipc_diagnostic'))
sys.path.insert(0,str(R/'frameworks/canopy_single/code/runtime/_deps'))
from probe_driver import run_one,put
KEYS=['net.ipv4.tcp_wmem','net.ipv4.tcp_rmem','net.core.wmem_max','net.core.rmem_max']
def read(k):return pathlib.Path('/proc/sys',k.replace('.','/')).read_text().split()
def write(k,v):subprocess.run(['sudo','-n','sysctl','-q','-w',k+'='+' '.join(v)],check=True,timeout=10,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def main():
    plan=json.loads((R/'buffer-zero-registration.json').read_text())
    assert json.loads((R/'formal-status.json').read_text())['stage']=='COMPLETE'
    if (R/'environment').exists():assert json.loads((R/'environment/restored.json').read_text())['status']=='PASS'
    folder=R/'buffer-zero';folder.mkdir(exist_ok=False)
    original={k:read(k) for k in KEYS};put(folder/'original.json',original)
    def stop(*_):raise RuntimeError('User requested safe stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    watcher_code="""import pathlib,json,os,time,subprocess
f=pathlib.Path(%r);pid=%d;end=%r
while time.time()<end and pathlib.Path('/proc/'+str(pid)).exists() and not (f/'restored.json').exists():time.sleep(1)
if not (f/'restored.json').exists():
 old=json.loads((f/'original.json').read_text());out={}
 for k,v in old.items():out[k]=subprocess.call(['sudo','-n','sysctl','-q','-w',k+'='+' '.join(v)])
 (f/'watchdog-restored.json').write_text(json.dumps(out))
"""%(str(folder),os.getpid(),min(plan['deadline_epoch'],time.time()+250))
    stream=(folder/'watchdog.stdout').open('w')
    watcher=subprocess.Popen([sys.executable,'-B','-c',watcher_code],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);stream.close()
    results=[]
    try:
        for key,value in original.items():write(key,value[:-1]+['67108864'])
        put(folder/'active-sysctls.json',{k:read(k) for k in KEYS})
        for cfg in plan['jobs']:
            assert time.time()+cfg['probe_seconds']+130<plan['deadline_epoch']
            subprocess.run(['sudo','-n','/sbin/ip','tcp_metrics','flush','all'],check=True,timeout=10)
            results.append(run_one(cfg,plan['deadline_epoch']));put(folder/'results.json',results)
        put(folder/'status.json',dict(status='COMPLETE',runs=len(results),epoch=time.time()))
    except Exception:
        put(folder/'status.json',dict(status='FAILED',traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        for k,v in original.items():write(k,v)
        actual={k:read(k) for k in KEYS}
        assert actual==original,(actual,original)
        put(folder/'restored.json',dict(status='PASS',values=actual,epoch=time.time()))
        try:watcher.wait(timeout=3)
        except subprocess.TimeoutExpired:pass

if __name__=='__main__':main()
