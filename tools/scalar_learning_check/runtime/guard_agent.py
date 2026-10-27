"""Per-host finite buffer lease and streaming safety monitor; no control policy or extra TCP reads."""
import argparse,csv,io,json,os,pathlib,signal,subprocess,sys,time,traceback
R=pathlib.Path(os.environ['CANOPY_L1_ROOT']);F=R/'framework'
sys.path[:0]=[str(F/'code/runtime/_deps'),str(R/'tools/scalar_learning_check')]
import psutil
from learning_monitor import window_stats
KEYS=['net.ipv4.tcp_wmem','net.ipv4.tcp_rmem','net.core.wmem_max','net.core.rmem_max']
def put(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,allow_nan=False));tmp.replace(p)
def read(k):return pathlib.Path('/proc/sys',k.replace('.','/')).read_text().split()
def write(k,v):subprocess.run(['sudo','-n','sysctl','-q','-w',k+'='+' '.join(v)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
class Lines:
    def __init__(self,path):self.path=path;self.file=None;self.header=None;self.buffer=''
    def rows(self):
        if self.file is None:
            if not self.path.exists():return []
            self.file=self.path.open();line=self.file.readline()
            if not line.endswith('\n'):self.file.close();self.file=None;return []
            self.header=next(csv.reader([line]))
        self.buffer+=self.file.read();lines=self.buffer.split('\n');self.buffer=lines.pop()
        return [dict(zip(self.header,row)) for line in lines for row in csv.reader([line]) if len(row)==len(self.header)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host-index',type=int,required=True);args=ap.parse_args()
    master=json.loads((R/'registration.json').read_text());cfg=master['training_config'];rid=cfg['run_id'];old={k:read(k) for k in KEYS}
    out=R/'audit';out.mkdir(exist_ok=True);assert not (out/'lease-before.json').exists()
    put(out/'lease-before.json',dict(epoch=time.time(),values=old,pid=os.getpid()))
    active={k:v[:-1]+['67108864'] for k,v in old.items()}
    for k,v in active.items():write(k,v)
    put(out/'lease-active.json',dict(epoch=time.time(),values={k:read(k) for k in KEYS}))
    indices=[i for i,h in enumerate(cfg['actor_hosts']) if h==args.host_index]
    states={i:Lines(F/'logs'/(rid+'-actor%d'%i)/'states.csv') for i in indices}
    controls={i:Lines(F/'logs'/(rid+'-actor%d'%i)/'control.csv') for i in indices}
    pending={i:dict(window=0,rows=[],cum=0.0,completed=[]) for i in indices};exploded=[];last=0;last_rpc=time.time()
    def request_stop(*_):raise RuntimeError('Lease guard explicit stop')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        while time.time()<master['deadline_epoch'] and not (out/'LEASE_COMPLETE').exists():
            active_cfg=out/'active-training-config.json'
            if active_cfg.exists():
                candidate=json.loads(active_cfg.read_text())
                if candidate['run_id']!=rid:
                    cfg=candidate;rid=cfg['run_id'];indices=[i for i,h in enumerate(cfg['actor_hosts']) if h==args.host_index]
                    states={i:Lines(F/'logs'/(rid+'-actor%d'%i)/'states.csv') for i in indices}
                    controls={i:Lines(F/'logs'/(rid+'-actor%d'%i)/'control.csv') for i in indices}
                    pending={i:dict(window=0,rows=[],cum=0.0,completed=[]) for i in indices};exploded=[];last=0
            heartbeat=out/'lease-heartbeat'
            if heartbeat.exists():last_rpc=max(last_rpc,heartbeat.stat().st_mtime)
            if time.time()-last_rpc>180:raise RuntimeError('Coordinator heartbeat stale180s')
            for i in indices:
                for c in controls[i].rows():
                    if max(int(c['cwnd_before']),int(c['cwnd_requested']))>=1000000 and i not in exploded:exploded.append(i)
                q=pending[i]
                for row in states[i].rows():
                    k=int(float(row['elapsed_s'])//30)
                    if k!=q['window'] and q['rows']:
                        stat,q['cum']=window_stats(q['rows'],cfg['conditions'][i],q['cum']);stat.update(window=q['window'],start_s=q['window']*30)
                        q['completed'].append(stat);q['rows']=[];q['window']=k
                    q['rows'].append(row)
            if exploded:
                put(F/'logs'/(rid+'-stop.json'),dict(reason='L1_S3_WINDOW_EXPLOSION',actors=exploded,epoch=time.time()))
            if time.time()-last>=1:
                last=time.time();put(F/'audit'/(rid+'-online-monitor.json'),dict(epoch=last,host_index=args.host_index,exploded=exploded,actors={str(i):dict(condition=cfg['conditions'][i]['condition'],windows=pending[i]['completed']) for i in indices}))
            time.sleep(.5)
    except Exception:
        put(out/'lease-failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        put(F/'logs'/(rid+'-stop.json'),dict(reason='L1_LEASE_GUARD_FAILURE',epoch=time.time()))
        time.sleep(10)
    finally:
        # All mutations are temporary, even if the controller dies.
        for k,v in old.items():write(k,v)
        actual={k:read(k) for k in KEYS};assert actual==old
        put(out/'lease-restored.json',dict(status='PASS',epoch=time.time(),values=actual))
if __name__=='__main__':main()
