"""Finite learner service: ingress and versioned actor-only snapshots.

Only newly inserted training boundaries issue credit. Isolated calibration
blocks, capacity samples and offline B0 evaluation have separate accounts.
"""
import argparse,hashlib,json,os,pathlib,queue,signal,socketserver,threading,time,traceback,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'code/runtime/_deps'))
import psutil
from common_period import PERIODS,encoded_period,checksum,REWARD,discount
from common_period_tf import Model
from common_period_runtime import Runtime,require_a0
from common_period_transport import unpack_row,snapshot_bytes
from common_period_calibration import calibrate

class Server(socketserver.ThreadingMixIn,HTTPServer):daemon_threads=True

def main(cfg):
    if cfg['stage']=='A':
        require_a0(cfg['preregistration'],cfg['a0_results'])
        if cfg['capacity_receipt']['status']!='PASS':raise ValueError('24-actor capacity has not passed')
        if cfg['planned_evaluation_runs']!=264:raise ValueError('A evaluation reservation incomplete')
    root=pathlib.Path(cfg['root']);root.mkdir(exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    reward=cfg['common_reward_config'];model=Model(cfg['source_checkpoint'],0.,{str(t):1/(1-.995**(t/20.)) for t in PERIODS},reward_config=reward)
    runtime=Runtime(model,reward);inbox=queue.Queue(4096);done=set();cal_done=set();exposure=np.zeros(24);cal_exposure=np.zeros(24)
    state=dict(phase='CAPACITY' if cfg['stage']=='capacity' else 'CALIBRATION',version=0,inserted=0,backlog=0,closed=False)
    published=dict(blob=b'',hash='');state_lock=threading.Lock();stop=threading.Event();errors=[]
    statistics=dict(max_backlog=0,max_debt=0,ingress_rows=0,invalid_rows=0);recent={};guard_since=None
    def publish():
        snap=runtime.publish(window_only=True);blob=snapshot_bytes(snap)
        with state_lock:published.update(blob=blob,hash=hashlib.sha256(blob).hexdigest());state.update(version=snap['version'],snapshot_hash=published['hash'])
        (root/('snapshot-%06d.sha256'%snap['version'])).write_text(published['hash'])
    publish()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def reply(self,code,data,ctype='application/json'):
            if not isinstance(data,bytes):data=json.dumps(data).encode()
            self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def authorized(self):return self.headers.get('X-Run-Token')==cfg['token']
        def do_GET(self):
            if not self.authorized():return self.reply(403,{'error':'run identity mismatch'})
            if self.path=='/snapshot':
                with state_lock:blob=published['blob']
                return self.reply(200,blob,'application/octet-stream')
            if self.path=='/status':
                with state_lock:value=dict(state)
                return self.reply(200,value)
            return self.reply(404,{'error':'unknown endpoint'})
        def do_POST(self):
            if not self.authorized():return self.reply(403,{'error':'run identity mismatch'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4*1024*1024:raise ValueError('Invalid bounded ingress')
                body=json.loads(self.rfile.read(size));items=body['items']
                if not 1<=len(items)<=64:raise ValueError('Invalid ingress batch')
                for item in items:
                    if item['type']=='row':item['row']=unpack_row(item['row'])
                    inbox.put_nowait(item)
                statistics['max_backlog']=max(statistics['max_backlog'],inbox.qsize())
                self.reply(200,{'accepted':len(items)})
            except Exception as exc:
                errors.append('INGRESS '+repr(exc));stop.set();self.reply(409,{'error':str(exc)})
    server=Server((cfg['bind_ip'],cfg['port']),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.set())
    started=time.monotonic();last_status=0;next_checkpoint=300.;next_publish=200;status='RUNNING';reason='';resources=[]
    psutil.cpu_times_percent(interval=None)
    def consume(item):
        actor=int(item['actor'])
        if not 0<=actor<24:raise ValueError('Invalid actor')
        if item['type']=='monitor':recent[actor]=item['values'];return
        if item['type']=='cal_done':
            if actor in cal_done:raise ValueError('Duplicate calibration completion')
            cal_done.add(actor);cal_exposure[actor]=float(item['seconds']);return
        if item['type']=='done':
            if actor in done:raise ValueError('Duplicate completion')
            done.add(actor)
            if item.get('status')!='COMPLETED':raise RuntimeError('Actor%d failed: %s'%(actor,item.get('reason')))
            return
        if item['type']!='row':raise ValueError('Unknown ingress message')
        row=item['row'];is_cal=row.pop('calibration');block=row.pop('block',None)
        if row['reward_version']!=REWARD or row['reward_checksum']!=runtime.replay.reward_checksum:raise ValueError('Ingress reward schema mismatch')
        if abs(row['discount']-discount(row['duration']))>1e-7:raise ValueError('Ingress actual-time discount mismatch')
        if row['actor']!=actor:raise ValueError('Actor identity mismatch')
        if actor in done:raise ValueError('Experience after actor completion')
        if is_cal and state['phase']!='CALIBRATION':raise ValueError('Late calibration experience')
        if not is_cal and state['phase']=='CALIBRATION':raise ValueError('Training before calibration release')
        args=dict(actor=actor,connection=row['connection'],next_connection=row['next_connection'],state=row['s0'],action=row['action'],reward=row['reward'],next_state=row['s1'],dt=row['duration'],period=row['period_ms'],version=row['snapshot'],kind=row['kind'],behavior=row['behavior'],calibration=is_cal,apply=row['apply'])
        if runtime.boundary(**args):
            statistics['ingress_rows']+=1;exposure[actor]+=row['duration']
            if is_cal:runtime.calibration[-1]['block']=block
    try:
        (root/'service-ready.json').write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time(),source='freshV31,empty replay/optimizers',stage=cfg['stage'])))
        while True:
            now=time.time()
            stopping=stop.is_set() or (root/'STOP').exists() or now>=cfg['deadline_epoch']
            if stopping:
                state['phase']='STOPPING'
                if not stop.is_set():stop.set()
            consumed=0
            while consumed<256:
                try:item=inbox.get_nowait()
                except queue.Empty:break
                consume(item);consumed+=1
            if cfg['stage']=='A' and state['phase']=='CALIBRATION' and len(cal_done)==24:
                audit=calibrate(model,runtime.calibration)
                (root/'calibration-audit.json').write_text(json.dumps(audit,indent=2))
                state['phase']='TRAINING';publish();runtime.save(str(root/'checkpoints/step-0'))
            if cfg['stage']=='A' and state['phase']=='TRAINING':
                limit=runtime.credit.limit(runtime.replay);debt=limit-runtime.credit.critic
                statistics['max_debt']=max(statistics['max_debt'],debt)
                if debt>0:
                    runtime.update_once()
                    if runtime.credit.critic>=next_publish:
                        publish();next_publish+=200
            if cfg['stage']=='A' and state['phase']=='TRAINING':
                mean_exposure=float(exposure.mean())
                if mean_exposure>=next_checkpoint:
                    runtime.save(str(root/'checkpoints'/('step-%d'%runtime.replay.inserted)));next_checkpoint+=300
                unhealthy=[i for i,r in recent.items() if now-r['epoch']<=3 and r['saturation']>=.99 and (r['utilization']<.5 or r['queue_mean_ms']>2*r['reference_queue_ms'])]
                guard_since=(guard_since if guard_since is not None else time.monotonic()) if len(unhealthy)>12 else None
                if guard_since is not None and time.monotonic()-guard_since>=120:
                    (root/'early-stop-evidence.json').write_text(json.dumps(dict(actors=unhealthy,recent=recent,duration=time.monotonic()-guard_since)))
                    raise RuntimeError('PREREGISTERED_MAJORITY_SATURATION_AND_BAD_NETWORK_120S')
            if time.monotonic()-last_status>=1:
                cpu=psutil.cpu_times_percent(interval=None)._asdict();busy=sum(cpu.get(k,0) for k in ['user','nice','system','irq','softirq'])-cpu.get('guest',0)-cpu.get('guest_nice',0)
                resources.append(dict(epoch=time.time(),busy_all_cores=busy/100.,mem_available=psutil.virtual_memory().available))
                (root/'resources.json').write_text(json.dumps(resources))
                state.update(inserted=runtime.replay.inserted,backlog=inbox.qsize(),ledger=runtime.ledger(),statistics=statistics,done=sorted(done),calibration_done=sorted(cal_done),exposure_by_actor=exposure.tolist(),epoch=time.time(),elapsed_s=time.monotonic()-started)
                (root/'status.json.tmp').write_text(json.dumps(state));(root/'status.json.tmp').replace(root/'status.json')
                last_status=time.monotonic()
            if len(done)==24 and inbox.empty():
                if cfg['stage']=='A' and runtime.credit.critic<runtime.credit.limit(runtime.replay):continue
                status='COMPLETED';break
            if stopping:
                if inbox.empty():status='STOPPED';reason='deadline or owned stop request';break
            if not consumed and (cfg['stage']!='A' or runtime.credit.critic>=runtime.credit.limit(runtime.replay)):time.sleep(.002)
            if errors:raise RuntimeError(errors[0])
        if cfg['stage']=='A' and runtime.replay.inserted:
            prefix=root/'checkpoints'/('step-%d'%runtime.replay.inserted)
            if not pathlib.Path(str(prefix)+'.index').exists():runtime.save(str(prefix))
    except BaseException as exc:
        status='FAILED';reason=repr(exc);(root/'failure.txt').write_text(traceback.format_exc());state['phase']='STOPPING'
        if cfg['stage']=='A' and runtime.replay.inserted:
            try:runtime.save(str(root/'checkpoints'/('failure-step-%d'%runtime.replay.inserted)))
            except Exception:(root/'save-failure.txt').write_text(traceback.format_exc())
    finally:
        stop.set();state.update(phase='CLOSED',closed=True,status=status,reason=reason)
        (root/'status.json').write_text(json.dumps(dict(state,epoch=time.time()),indent=2))
        (root/'result.json').write_text(json.dumps(dict(stage=cfg['stage'],status=status,reason=reason,statistics=statistics,ledger=runtime.ledger(),done=sorted(done),calibration_done=sorted(cal_done),exposure_by_actor=exposure.tolist(),calibration_exposure_by_actor=cal_exposure.tolist(),gradient_history=runtime.gradient_history,elapsed_s=time.monotonic()-started,mean_busy_all_cores=sum(r['busy_all_cores'] for r in resources)/max(1,len(resources)),min_memory_available=min([r['mem_available'] for r in resources]+[1<<60])),indent=2))
        time.sleep(2);server.shutdown();server.server_close();model.session.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();main(json.loads(pathlib.Path(args.config).read_text()))
