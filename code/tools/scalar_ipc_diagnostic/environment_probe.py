"""Independent finite CUBIC socket-buffer diagnostic. Explicit opt-in, restores sysctls."""
import argparse,csv,json,os,pathlib,signal,subprocess,sys,time,traceback
R=pathlib.Path(os.environ['CANOPY_IPC_DIAGNOSTIC_ROOT'])
F=R/'frameworks/canopy_single'
sys.path.insert(0,str(F/'code/runtime/_deps'))
sys.path.insert(0,str(F))
import psutil
from mahimahi_metrics import mm
KEYS=['net.ipv4.tcp_wmem','net.ipv4.tcp_rmem','net.core.wmem_max','net.core.rmem_max']
MODES=['current','send64','recv64','both64']
def read(key):return pathlib.Path('/proc/sys',key.replace('.','/')).read_text().split()
def write(key,values):
    subprocess.run(['sudo','-n','sysctl','-q','-w',key+'='+' '.join(values)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
def put(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False))
def snapshot(port):
    cmd=['ss','-tmin','state','established','( sport = :%d )'%port]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
    return dict(command=cmd,code=p.returncode,output=p.stdout.decode(errors='replace'))
def cleanup(proc,identity,remembered=None):
    if proc is None:return
    owned=list((remembered or {}).items())
    try:
        p=psutil.Process(proc.pid)
        if abs(p.create_time()-identity)<.01:
            owned += [(p.pid,p.create_time()) for p in p.children(recursive=True)]+[(p.pid,p.create_time())]
    except psutil.Error:pass
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for pid,created in reversed(owned):
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-created)<.01:p.send_signal(sig)
            except psutil.Error:pass
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',required=True);args=ap.parse_args()
    condition=json.loads((R/'environment-registration.json').read_text())
    assert condition['condition']['condition']==args.condition
    assert json.loads((R/'formal-status.json').read_text())['stage']=='COMPLETE'
    until=condition['deadline_epoch'];C=condition['condition'];assert time.time()+300<until
    out=R/condition.get('output_directory','environment');out.mkdir(exist_ok=False)
    original={key:read(key) for key in KEYS}
    put(out/'original-sysctls.json',original)
    # Independent watcher restores if this owner disappears or overruns its finite budget.
    watchdog_code="""import pathlib,json,time,subprocess,os
folder=pathlib.Path(%r);owner=%d;end=%r
while time.time()<end and pathlib.Path('/proc/'+str(owner)).exists() and not (folder/'restored.json').exists():time.sleep(1)
if not (folder/'restored.json').exists():
 old=json.loads((folder/'original-sysctls.json').read_text());receipt={}
 for k,v in old.items():
  p=subprocess.run(['sudo','-n','sysctl','-q','-w',k+'='+' '.join(v)],stdout=subprocess.PIPE,stderr=subprocess.PIPE);receipt[k]=p.returncode
 (folder/'watchdog-restoration.json').write_text(json.dumps(receipt))
"""%(str(out),os.getpid(),min(until,time.time()+390))
    watchlog=(out/'watchdog.stdout').open('w')
    watcher=subprocess.Popen([sys.executable,'-B','-c',watchdog_code],stdin=subprocess.DEVNULL,stdout=watchlog,stderr=subprocess.STDOUT,start_new_session=True);watchlog.close()
    results=[];proc=None;identity=0;remembered={}
    def stop_requested(*_):raise RuntimeError('Explicit safe stop requested')
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    try:
        for mode in condition['mode_order']:
            assert time.time()+90<until
            for key,old in original.items():
                selected=mode=='both64' or (mode=='send64' and ('wmem' in key)) or (mode=='recv64' and ('rmem' in key))
                value=old[:-1]+['67108864'] if selected else old
                write(key,value);assert read(key)==value
            run=out/mode;run.mkdir()
            subprocess.run(['sudo','-n','/sbin/ip','tcp_metrics','flush','all'],check=True,timeout=10)
            env=dict(os.environ,C3_TRACE_DIR=str(F/'configs/traces'),ORCA_CONTROL_LOG=str(run/'control.csv'))
            port=condition.get('port_base',49731)+MODES.index(mode)
            cmd=[str(R/'native/tcp-buffer-probe'),str(port),str(F/'code/runtime'),'cubic',C['trace'],'constant192',str(C['delay_ms']),str(run/'down.log'),'60',str(C['queue_packets'])]
            log=(run/'server.log').open('w')
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();identity=psutil.Process(proc.pid).create_time()
            remembered={proc.pid:identity}
            put(run/'ownership.json',dict(pid=proc.pid,created=identity,command=cmd))
            started=time.time();marks=[15,30,45];snaps=[];last_owned=0
            while proc.poll() is None:
                if time.time()-started>85:raise RuntimeError('Finite CUBIC timeout')
                if marks and time.time()-started>=marks[0]:
                    marks.pop(0);snaps.append(dict(elapsed_s=time.time()-started,**snapshot(port)))
                if time.time()-last_owned>=1:
                    last_owned=time.time()
                    try:
                        for child in psutil.Process(proc.pid).children(recursive=True):remembered[child.pid]=child.create_time()
                    except psutil.NoSuchProcess:pass
                    put(run/'owned-processes.json',remembered)
                time.sleep(.2)
            cleanup(proc,identity,remembered);proc=None;remembered={}
            rows=list(csv.DictReader((run/'control.csv').open()))
            assert len(rows)>100 and 'unacked' in rows[0]
            inflight=sorted(int(r['unacked'])*int(r['mss']) for r in rows)
            windows,_,_=mm(run/'down.log',60)
            assert all(w['complete_window'] for w in windows),windows
            receipt=dict(mode=mode,sysctls={k:read(k) for k in KEYS},snapshots=snaps,
                inflight_payload_bytes_p95=inflight[int(.95*(len(inflight)-1))],inflight_payload_bytes_max=max(inflight),
                monitor_rows=len(rows),windows=windows,algorithm='CUBIC_NO_RL_NO_WINDOW_WRITES',
                monitoring='Same existing20ms TCP_INFO call, appended unacked/mss fields; no RL/NN/IPC')
            put(run/'result.json',receipt);results.append(receipt);put(out/'results.json',results)
        put(out/'status.json',dict(status='COMPLETE',runs=len(results),epoch=time.time()))
    except Exception:
        put(out/'status.json',dict(status='FAILED',epoch=time.time(),traceback=traceback.format_exc()));raise
    finally:
        cleanup(proc,identity,remembered)
        restore={}
        for key,val in original.items():
            try:write(key,val);restore[key]=dict(expected=val,actual=read(key))
            except Exception as e:restore[key]=dict(error=repr(e))
        if all(x.get('actual')==x.get('expected') for x in restore.values()):
            put(out/'restored.json',dict(status='PASS',values=restore,epoch=time.time()))
        else:put(out/'restoration-failed.json',restore)
        try:watcher.wait(timeout=4)
        except subprocess.TimeoutExpired:pass

if __name__=='__main__':main()
