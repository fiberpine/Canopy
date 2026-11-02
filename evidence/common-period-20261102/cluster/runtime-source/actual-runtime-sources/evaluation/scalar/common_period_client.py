"""Persistent authenticated ingress, shared host cache, no extra TCP telemetry."""
import argparse,hashlib,http.client,json,os,pathlib,queue,threading,time,traceback
from common_period_transport import pack_row

class Client:
    def __init__(self,cfg):self.cfg=cfg;self.connection=http.client.HTTPConnection(cfg['learner_ip'],cfg['port'],timeout=10)
    def request(self,method,path,body=None):
        payload=json.dumps(body).encode() if body is not None else None
        self.connection.request(method,path,payload,headers={'X-Run-Token':self.cfg['token'],'Content-Type':'application/json'})
        response=self.connection.getresponse();data=response.read()
        if response.status!=200:raise RuntimeError('Learner ingress response%d: %s'%(response.status,data[:200]))
        return data
    def status(self):return json.loads(self.request('GET','/status'))
    def close(self):self.connection.close()

class Sender:
    def __init__(self,cfg,actor):
        self.client=Client(cfg);self.actor=actor;self.queue=queue.Queue(512);self.stop=threading.Event();self.error=None;self.generated=0;self.acknowledged=0;self.max_pending=0
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def put(self,item):
        if self.error:raise RuntimeError(self.error)
        self.queue.put_nowait(item);self.generated+=int(item['type']=='row');self.max_pending=max(self.max_pending,self.queue.qsize())
    def boundary(self,row):self.put(dict(type='row',actor=self.actor,row=pack_row(row)))
    def loop(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:first=self.queue.get(timeout=.02)
                except queue.Empty:continue
                items=[first]
                while len(items)<64:
                    try:items.append(self.queue.get_nowait())
                    except queue.Empty:break
                result=json.loads(self.client.request('POST','/ingest',dict(items=items)))
                if result['accepted']!=len(items):raise RuntimeError('Partial ingress acknowledgment')
                self.acknowledged+=sum(x['type']=='row' for x in items)
                for _ in items:self.queue.task_done()
        except BaseException:self.error=traceback.format_exc();self.stop.set()
        finally:self.client.close()
    def flush(self,timeout=30):
        started=time.monotonic()
        while self.queue.unfinished_tasks and time.monotonic()-started<timeout:
            if self.error:raise RuntimeError(self.error)
            time.sleep(.01)
        if self.queue.unfinished_tasks:raise RuntimeError('Ingress did not drain within30s')
    def finish(self,status,reason=''):
        self.flush();self.put(dict(type='done',actor=self.actor,status=status,reason=reason));self.flush();self.stop.set();self.thread.join(3)

def cache_main(cfg):
    root=pathlib.Path(cfg['host_root']);root.mkdir(exist_ok=True);client=Client(cfg);current=None;errors=0
    try:
        while time.time()<cfg['deadline_epoch']:
            try:
                status=client.status();errors=0
                if status['snapshot_hash']!=current:
                    blob=client.request('GET','/snapshot');digest=hashlib.sha256(blob).hexdigest()
                    if digest!=status['snapshot_hash']:
                        # Publication may advance between two GETs. Re-read;
                        # never attach the old version to the new tensor bytes.
                        status=client.status()
                        if digest!=status['snapshot_hash']:continue
                    (root/'policy.npz.tmp').write_bytes(blob);(root/'policy.npz.tmp').replace(root/'policy.npz');current=digest
                    (root/'policy.json.tmp').write_text(json.dumps(dict(version=status['version'],hash=digest)));(root/'policy.json.tmp').replace(root/'policy.json')
                (root/'service-status.json.tmp').write_text(json.dumps(dict(status,cache_epoch=time.time())));(root/'service-status.json.tmp').replace(root/'service-status.json')
                if status['closed']:break
            except Exception:
                errors+=1
                if errors>=3:raise
            time.sleep(.2)
    except Exception:(root/'cache-failure.txt').write_text(traceback.format_exc());raise
    finally:client.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();cache_main(json.loads(pathlib.Path(a.config).read_text()))
