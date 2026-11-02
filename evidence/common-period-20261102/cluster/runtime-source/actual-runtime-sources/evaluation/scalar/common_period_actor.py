"""Local policy inference with asynchronous boundary ingress and isolated TCP episodes."""
import argparse,collections,fcntl,hashlib,json,os,pathlib,sys,time,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/canopy/rl-module'))
import evaluate
from envwrapper_v2 import TCP_Env_Wrapper
from common_period import PERIODS,REWARD,SCHEMA,encoded_period,discount,utility,checksum,Exploration,validate_execution
from common_period_transport import Policy
from common_period_client import Sender

class NetMonitor:
    def __init__(self,path):self.path=path;self.file=None;self.base=None;self.bins={};self.last=0.
    def read(self):
        if self.file is None:
            if not self.path.exists():return None
            self.file=self.path.open()
        while True:
            pos=self.file.tell();line=self.file.readline()
            if not line:break
            if not line.endswith('\n'):self.file.seek(pos);break
            if line.startswith('# base timestamp:'):self.base=float(line.split(':')[1]);continue
            if line.startswith('#') or self.base is None:continue
            fields=line.split()
            if len(fields)<3:continue
            t=(float(fields[0])-self.base)/1000.;self.last=max(self.last,t);second=int(t)
            row=self.bins.setdefault(second,[0.,0.,0.,0.])
            if fields[1]=='#':row[0]+=int(fields[2])
            if fields[1]=='-':row[1]+=int(fields[2]);row[2]+=float(fields[3]);row[3]+=1
        current=int(self.last)
        for k in list(self.bins):
            if k<current-7:del self.bins[k]
        selected=[v for k,v in self.bins.items() if current-5<=k<current]
        if len(selected)<3:return None
        total=np.sum(selected,axis=0)
        return dict(utilization=float(total[1]/total[0]) if total[0] else 0.,queue_mean_ms=float(total[2]/max(1,total[3])),network_seconds=len(selected))

def main(cfg,index):
    host=pathlib.Path(cfg['host_root']);host.mkdir(exist_ok=True);reward=cfg['common_reward_config'];policy=Policy(cfg['source_checkpoint'],checksum(reward));sender=Sender(cfg,index)
    exploration=Exploration(cfg['seed']+index*1009);exposure=0.;training=0.;episode=0;calibration_exposure=0.;results=[];status='FAILED';reason='';recent=collections.deque()
    original_read=evaluate.StateReader.read
    def guarded_read(self):
        if self.last_id is not None:return original_read(self)
        start=time.monotonic()
        while True:
            try:return original_read(self)
            except RuntimeError as e:
                if str(e)!='no valid state within15s' or self.stop.is_set() or time.monotonic()-start+15>60.1:raise
    evaluate.StateReader.read=guarded_read
    class Wrapper(TCP_Env_Wrapper):
        def get_state(self,*args,**kwargs):
            info,delay,old,ok=super().get_state(*args,**kwargs)
            if not ok:return info,delay,old,ok
            r=self.shrmem_r.last_raw
            return info,delay,utility(float(r[1]),float(r[7]),reward['scale_bps'],float(r[0]),float(r[8]),float(r[14]),float(r[2])),ok
    def service_status():
        path=host/'service-status.json'
        if not path.exists():raise RuntimeError('Host service cache unavailable')
        state=json.loads(path.read_text())
        if time.time()-state['cache_epoch']>10:raise RuntimeError('Stale learner status lease')
        if state['phase'] in ('STOPPING','CLOSED'):raise RuntimeError('Learner requested stop')
        return state
    def refresh():
        for _ in range(10):
            meta=json.loads((host/'policy.json').read_text())
            if meta['version']==policy.version:return
            blob=(host/'policy.npz').read_bytes()
            if hashlib.sha256(blob).hexdigest()==meta['hash']:policy.load(blob);return
            time.sleep(.001)
        raise RuntimeError('Host snapshot publication inconsistent')
    def cleanup(run_id,port):
        import subprocess,psutil,sysv_ipc
        path=ROOT/'logs'/run_id;owned=[]
        for p in psutil.process_iter(['pid','cmdline','status']):
            try:
                args=p.info['cmdline'] or []
                if p.info['status']!=psutil.STATUS_ZOMBIE and (any(str(path)+'/' in a for a in args) or (str(ROOT/'code/runtime/client') in args and str(port) in args)):owned.append((p.pid,p.create_time()))
            except psutil.Error:pass
        for sig in ['TERM','KILL']:
            for pid,created in owned:
                try:
                    p=psutil.Process(pid)
                    if abs(p.create_time()-created)<.001:subprocess.run(['sudo','-n','/bin/kill','-'+sig,str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
                except psutil.NoSuchProcess:pass
            time.sleep(.1)
        return owned
    def run_episode(kind,seconds,cal_part=None):
        nonlocal exposure,training,episode,calibration_exposure
        run_id='%s-actor%02d-ep%03d'%(cfg['run_id'],index,episode);episode+=1
        ecfg=dict(cfg['eval_base']);ecfg.update(cfg['conditions'][index]);ecfg.pop('conditions',None)
        ecfg.update(run_id=run_id,method='canopy',seconds=0,training_steps=1000000000,physical_seconds_per_actor=seconds,episode_control_seconds=0,port=26000+index*64+episode,absolute_deadline_epoch=cfg['deadline_epoch']-60,checkpoint=cfg['source_checkpoint'],server_binary=cfg['server_binary'],decision_cost=0,connection_reset_seconds=0,stop_file=str(host/'STOP'),common_reward_config=reward,code_commit=cfg['code_commit'],training_mode=cfg['stage']=='A',no_training=cfg['stage']!='A')
        ecfg['interval_audit_v38']=True
        validate_execution(ecfg);path=ROOT/'configs'/(run_id+'.json');path.write_text(json.dumps(ecfg,indent=2))
        pending=collections.deque();clock=0.;block=0;block_start=0.;local_exposure=0.;last_monitor=0.;monitor=NetMonitor(ROOT/'logs'/run_id/'down.log')
        lockfile=(host/'startup.lock').open('a');fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX);held=[True]
        def model_proxy(c):
            def act(state):
                nonlocal clock,block,block_start
                if sender.error:raise RuntimeError(sender.error)
                service_status();refresh();clock+=.02*np.expm1(float(state[70]))
                if kind=='calibration':
                    if cal_part==0:
                        if block<4 and clock-block_start>=40.001:
                            block+=1;block_start=clock
                        b=block if block<4 else None;t=PERIODS[min(block,3)]
                    else:b=4;t=160
                    window=policy.window(state,t);action=np.asarray([window,encoded_period(t)],np.float32);source='policy';nn=action.copy()
                elif kind=='capacity':
                    b=None;t=PERIODS[int(clock//5)%5];window=policy.window(state,t);action=np.asarray([window,encoded_period(t)],np.float32);source='policy';nn=action.copy()
                else:
                    b=None;query={}
                    def window(t):
                        query['window']=policy.window(state,t);return query['window']
                    action,source=exploration.choose(training+clock,window);t=exploration.period;nn=np.asarray([query['window'],encoded_period(t)],np.float32)
                pending.append(dict(period=t,source=source,version=policy.version,block=b,nn=nn))
                act.no_noise=nn;act.behavior_source=source
                if held[0]:fcntl.flock(lockfile.fileno(),fcntl.LOCK_UN);held[0]=False
                ready=host/('actor%02d.ready'%index)
                if not ready.exists():ready.write_text(str(time.time()))
                return action
            def transition(s,a,r,sp,dt,g):
                nonlocal local_exposure,exposure,last_monitor
                meta=pending.popleft();local_exposure+=dt;exposure+=dt
                row=dict(actor=index,connection=run_id,next_connection=run_id,s0=s,s1=sp,action=a,reward=r,duration=dt,discount=g,period_ms=meta['period'],snapshot=meta['version'],kind='continuing',behavior=meta['source'],calibration=kind=='calibration',block=meta['block'],apply=1,reward_version=REWARD,reward_checksum=checksum(reward))
                # Keep the latest complete observed transition until the next
                # boundary, so the administrative final row can be typed correctly.
                if transition.last is not None:sender.boundary(transition.last)
                transition.last=row
                if kind=='training':
                    recent.append((training+local_exposure,dt,abs(float(meta['nn'][0]))>.99,meta['period']))
                    while recent and recent[0][0]<training+local_exposure-5:recent.popleft()
                    if time.monotonic()-last_monitor>=1:
                        net=monitor.read()
                        if net:
                            mass=sum(x[1] for x in recent);sat=sum(x[1]*x[2] for x in recent)/max(1e-9,mass)
                            ref=sum(x[1]*cfg['queue_reference'][str(index)][str(x[3])] for x in recent)/max(1e-9,mass)
                            sender.put(dict(type='monitor',actor=index,values=dict(net,saturation=sat,reference_queue_ms=ref,physical_seconds=training+local_exposure,epoch=time.time())))
                        last_monitor=time.monotonic()
            transition.last=None;evaluate.transition_hook=transition
            return policy,policy.session,Wrapper,json.loads((ROOT/'code/canopy/params_base_eval.json').read_text()),act,lambda:'VERSIONED_LIVE_POLICY',[]
        evaluate.model=model_proxy;evaluate.distributed_session=True
        before_argv=list(sys.argv);sys.argv=['evaluate.py','--config',str(path)]
        try:
            evaluate.main()
            result=json.loads((ROOT/'logs'/run_id/'result.json').read_text())
            if result['status']!='COMPLETED':raise RuntimeError(result['reason'])
            hook=evaluate.transition_hook
            if hook.last is not None:hook.last['kind']='truncated';sender.boundary(hook.last);hook.last=None
            sender.flush()
            if local_exposure+1e-6<seconds:raise RuntimeError('Episode physical exposure truncated')
            results.append(dict(run=run_id,kind=kind,physical_seconds=local_exposure,status=result['status']))
            return local_exposure
        finally:
            sys.argv=before_argv
            if held[0]:fcntl.flock(lockfile.fileno(),fcntl.LOCK_UN)
            lockfile.close();cleanup(run_id,ecfg['port'])
    try:
        started=time.monotonic()
        while not (host/'policy.json').exists():
            if time.monotonic()-started>60:raise RuntimeError('No initial host snapshot')
            time.sleep(.1)
        refresh()
        if cfg['stage']=='capacity':run_episode('capacity',30.)
        else:
            calibration_exposure+=run_episode('calibration',180.+index*.5,0)
            calibration_exposure+=run_episode('calibration',40.,1)
            sender.put(dict(type='cal_done',actor=index,seconds=calibration_exposure));sender.flush()
            while service_status()['phase']!='TRAINING':time.sleep(.2)
            while exposure<1800.:
                service_status()
                amount=run_episode('training',min(180.,1800.-exposure));training+=amount
        status='COMPLETED'
    except BaseException as exc:reason=repr(exc);(host/('actor%02d.failure.txt'%index)).write_text(traceback.format_exc())
    finally:
        try:sender.finish(status,reason)
        except Exception:reason+='; finish:'+traceback.format_exc();status='FAILED'
        policy.session.close()
        (host/('actor%02d.result.json'%index)).write_text(json.dumps(dict(status=status,reason=reason,actor=index,physical_seconds=exposure,training_seconds=training,calibration_seconds=calibration_exposure,generated=sender.generated,acknowledged=sender.acknowledged,max_pending=sender.max_pending,episodes=results),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args();main(json.loads(pathlib.Path(a.config).read_text()),a.index)
