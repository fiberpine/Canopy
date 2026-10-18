"""Pure official Orca model/reward, asynchronous real-time actor, counted learner.
Run with /home/ubuntu/venv/bin/python. All paths come from the experiment root.
"""
import os, sys, time, json, csv, pathlib, random, threading, math, subprocess, signal, traceback, queue
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'_deps'))
import numpy as np
import tensorflow as tf
import sysv_ipc
import psutil
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/baseline'))
from agent import Agent
from envwrapper import TCP_Env_Wrapper
from utils import ReplayBuffer

class EndOfRun(Exception): pass

class LockedReplay(ReplayBuffer):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.cv=threading.Condition(); self.inserted=0; self.generated=0
        self.birth=np.zeros(self.size,dtype=np.int64); self.sample_age=[]
    def append(self,s,a,r,s1,terminal):
        with self.cv:
            self.birth[self.ptr]=self.inserted
            self.store(s,a,r,s1,terminal)
            self.inserted+=1; self.generated+=1; self.cv.notify_all()
    def sample(self):
        # Same contiguous block sampling as official Orca, with copies under lock.
        with self.cv:
            n=self.length_buf
            if n<1: raise RuntimeError('empty replay')
            start=int(n*random.random()) if self.batch_size<n else 0
            index=np.array(random.sample(range(n),min(n,self.batch_size))) if getattr(self,'sampler','contiguous')=='iid' else (np.arange(min(n,self.batch_size))+start)%n
            self.sample_age=(self.inserted-self.birth[index]).tolist()
            return [x[index].copy() for x in self.peek_buffer()]

class StateReader:
    def __init__(self,shm,server,stop):
        self.shm=shm; self.server=server; self.stop=stop; self.last_id=None; self.last_raw=None
    def read(self):
        deadline=time.monotonic()+15
        while not self.stop.is_set():
            if self.server.poll() is not None: raise EndOfRun('server exited')
            value=self.shm.read().split(b'\0')[0]
            fields=value.split()
            if len(fields)==16:
                try:
                    rid=int(fields[0]); vals=np.array([float(x) for x in fields[1:]])
                    valid=np.all(np.isfinite(vals)) and vals[5]>0 and vals[8]>0 and vals[14]>0
                except (ValueError,OverflowError): valid=False
                if valid and rid!=self.last_id:
                    # Only the matching actor writes an action after this snapshot.
                    # Server waits for that ID before it can overwrite the next state.
                    if self.last_id is not None and rid!=(self.last_id+1)%1000:
                        raise RuntimeError('state ID gap %s -> %s'%(self.last_id,rid))
                    self.last_id=rid; self.last_raw=vals; return value+b'\0'
            if time.monotonic()>deadline: raise RuntimeError('no valid state within 15s')
            time.sleep(0.0005)
        raise EndOfRun('actor stopped')

def safe_cleanup(server,owned):
    # Discover descendants while parent is still alive; retain PID/create_time identity.
    try:
        for p in psutil.Process(server.pid).children(recursive=True): owned[p.pid]=p.create_time()
    except psutil.Error: pass
    owned[server.pid]=getattr(server,'created',0)
    # Only these specific processes belong to this run. No name-based or global kill.
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,created in list(owned.items())[::-1]:
            try:
                p=psutil.Process(pid)
                if created and abs(p.create_time()-created)>0.01: continue
                p.send_signal(sig)
            except psutil.Error: pass
        try: server.wait(timeout=3)
        except subprocess.TimeoutExpired: pass
        time.sleep(0.2)

def build_model(cfg,seed):
    random.seed(seed); np.random.seed(seed); tf.set_random_seed(seed)
    a=Agent(70,1,cfg['h1_shape'],cfg['h2_shape'],gamma=cfg['gamma'],batch_size=cfg['batch_size'],
            lr_a=cfg['lr_a'],lr_c=cfg['lr_c'],tau=cfg['tau'],mem_size=cfg['memsize'],
            noise_type=cfg['noise_type'],noise_exp=cfg['noise_exp'],stddev=cfg['stddev'],
            PER=False,CDQ=True,LOSS_TYPE=cfg['LOSS_TYPE'])
    a.build_learn()
    # Expose exact original update operations; add read-only diagnostics.
    av=a.actor.train_var(); cv=a.critic.train_var()
    a.diag_grads=[tf.global_norm(tf.gradients(a.a_loss,av)),tf.global_norm(tf.gradients(a.c_loss,cv))]
    a.bn_ops=[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'actor' in v.name and 'target' not in v.name]
    return a

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True)
    args=ap.parse_args(); cfg=json.loads(pathlib.Path(args.config).read_text())
    run=ROOT/'logs'/cfg['run_id']; run.mkdir(exist_ok=False)
    (run/'config.json').write_text(json.dumps(cfg,indent=2))
    b=ROOT/'code/baseline'; server=None; owned={}; shm_r=shm_w=None
    stop=threading.Event(); done=threading.Event(); errors=[]
    counters={'critic_iterations':0,'critic_optimizer_ops':0,'actor_updates':0,'target_updates':0,'valid':0,'invalid':0,'exploration_calls':0}
    start=time.monotonic(); status='RUNNING'; reason=''; resources=[]; model=None; saver=None; sess=None
    states=open(str(run/'states.csv'),'w',buffering=262144)
    sw=csv.writer(states); sw.writerow(['time_us','elapsed_s','transition','state_id','action','action_no_noise','percent','reward','throughput_term','loss_term','delay_term','max_bw','cwnd','delta_t','srtt_ms','min_rtt_ms','state'])
    training=open(str(run/'training.csv'),'w',buffering=65536)
    tw=csv.writer(training); tw.writerow(['wall_s','valid','inserted','replay_length','critic_iterations','critic_optimizer_ops','actor_updates','target_updates','actual_u','post_warmup_u','sample_age_mean','critic_loss','critic2_loss','actor_loss','q_mean','actor_grad_norm','critic_grad_norm'])
    cpdir=ROOT/'checkpoints'/cfg.get('model_id',cfg['run_id']); cpdir.mkdir(exist_ok=True)
    def checkpoint(label):
        path=saver.save(sess,str(cpdir/('step-%d'%label)),write_meta_graph=(label==0))
        state=dict(counters,inserted=model.rp_buffer.inserted,code_commit=cfg['code_commit'],cfg=cfg,
                   numpy_random=np.random.get_state(),python_random=random.getstate())
        import pickle
        with open(str(cpdir/('step-%d.state.pkl'%label)),'wb') as f: pickle.dump(state,f,protocol=4)
        with model.rp_buffer.cv:
            np.savez(str(cpdir/('step-%d.replay.npz'%label)),s0=model.rp_buffer.s0_buf,a=model.rp_buffer.a_buf,
                reward=model.rp_buffer.reward_buf,s1=model.rp_buffer.s1_buf,terminal=model.rp_buffer.terminal_buf,
                birth=model.rp_buffer.birth,ptr=model.rp_buffer.ptr,full=model.rp_buffer.full)
        (run/'latest_checkpoint.txt').write_text(path)
        return path
    def learner():
        try:
            last_log=0; last_diag=[float('nan')]*6
            def record():
                n=model.rp_buffer.inserted; it=counters['critic_iterations']
                ages=model.rp_buffer.sample_age
                tw.writerow([time.monotonic()-start,counters['valid'],n,model.rp_buffer.length_buf,it,it*2,it,it,it/max(1,n),it/max(1,n-cfg['warmup']),float(np.mean(ages)) if ages else float('nan')]+last_diag)
                training.flush()
            while not done.is_set() or counters['critic_iterations']<int(max(0,model.rp_buffer.inserted-cfg['warmup'])*cfg['update_ratio']):
                rp=model.rp_buffer
                if time.monotonic()-last_log>5:
                    record(); last_log=time.monotonic()
                with rp.cv:
                    limit=int(max(0,rp.inserted-cfg['warmup'])*cfg['update_ratio'])
                    if counters['critic_iterations']>=limit:
                        rp.cv.wait(0.02); continue
                batch=rp.sample()
                fd={model.s0:batch[0],model.action:batch[1],model.reward:batch[2],model.s1:batch[3],model.terminal:batch[4],model.is_training:True}
                sess.run(model.critic_train_op,fd)
                sess.run([model.actor_train_op,model.bn_ops],fd)
                model.target_update()
                counters['critic_iterations']+=1; counters['critic_optimizer_ops']+=2
                counters['actor_updates']+=1; counters['target_updates']+=1
                it=counters['critic_iterations']
                if it%100==0:
                    vals=sess.run([model.c_loss,model.c_loss2,model.a_loss,tf_reduce_q]+model.diag_grads,fd)
                    last_diag=list(map(float,vals))
                    if not np.all(np.isfinite(last_diag)): raise RuntimeError('nonfinite learner diagnostics')
                time.sleep(cfg['update_delay']/1000.0)
        except Exception:
            errors.append(traceback.format_exc()); stop.set()
    try:
        if cfg['method'] in ('train','model'):
            model=build_model(cfg,cfg['seed']); tf_reduce_q=tf.reduce_mean(model.critic_out)
            sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2))
            model.assign_sess(sess); sess.run(tf.global_variables_initializer())
            if cfg.get('initial_output_scale',1)!=1:
                output_vars=[v for v in tf.trainable_variables() if v.op.name=='actor/dense/kernel']
                assert len(output_vars)==1
                sess.run([tf.assign(v,v*cfg['initial_output_scale']) for v in output_vars])
            model.init_target()
            model.rp_buffer=LockedReplay(cfg['memsize'],70,1,cfg['batch_size']); model.rp_buffer.sampler=cfg.get('sampler','contiguous')
            saver=tf.train.Saver(max_to_keep=None)
            if cfg.get('checkpoint'):
                saver.restore(sess,cfg['checkpoint'])
                if cfg['method']=='train':
                    import pickle
                    with open(cfg['checkpoint']+'.state.pkl','rb') as f: saved=pickle.load(f)
                    counters.update({k:saved[k] for k in counters})
                    rp=model.rp_buffer; rp.inserted=saved['inserted']; rp.generated=counters['valid']
                    with np.load(cfg['checkpoint']+'.replay.npz') as z:
                        for attr,key in [('s0_buf','s0'),('a_buf','a'),('reward_buf','reward'),('s1_buf','s1'),('terminal_buf','terminal'),('birth','birth')]:
                            getattr(rp,attr)[:]=z[key]
                        rp.ptr=int(z['ptr']); rp.full=bool(z['full']); rp.length_buf=min(rp.inserted,rp.size)
                    np.random.set_state(saved['numpy_random']); random.setstate(saved['python_random'])
            elif cfg['method']=='train': checkpoint(0)
        envv=dict(os.environ,C3_TRACE_DIR=str(ROOT/'configs/traces'),ORCA_CONTROL_LOG=str(run/'control.csv'))
        if cfg['method']=='cubic':
            cmd=[str(b/'tcp-native'),'46000',str(b),'cubic',cfg['trace'],'constant192',str(cfg['delay_ms']),str(run/'down.log'),str(cfg['seconds']),str(cfg['queue_packets'])]
        else:
            shm_r=sysv_ipc.SharedMemory(None,sysv_ipc.IPC_CREX,mode=0o600,size=2048)
            shm_w=sysv_ipc.SharedMemory(None,sysv_ipc.IPC_CREX,mode=0o600,size=2048)
            shm_r.write(b'\0'*2048); shm_w.write(b'\0'*2048)
            envv.update(ORCA_SHM_R=str(shm_r.key),ORCA_SHM_W=str(shm_w.key))
            cmd=[str(b/'orca-server'),'46000',str(b),'20','4','cubic','0',cfg['trace'],'constant192',str(cfg['delay_ms']),str(run/'down.log'),str(cfg['seconds']),str(cfg.get('steps',0)-counters['valid']+1 if cfg['method']=='train' else 0),]
            # Queue precedes duration_steps in official interface.
            cmd=cmd[:-1]+[str(cfg['queue_packets']),cmd[-1],'baseline','0','0','200000','0','0','0','1','0','0','0','0','none','raw',str(cfg['seed']),'-1']
        out=open(str(run/'server.log'),'w')
        server=subprocess.Popen(cmd,env=envv,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        server.created=psutil.Process(server.pid).create_time()
        (run/'ownership.json').write_text(json.dumps({'pid':server.pid,'started':server.created,'cmd':cmd,'shm_keys':[shm_r.key,shm_w.key] if shm_r else []},indent=2))
        def monitor():
            psutil.cpu_times_percent(interval=None)
            last_resource=0
            control_reader=None
            while not stop.is_set():
                try:
                    descendants=psutil.Process(server.pid).children(recursive=True)
                    for p in descendants: owned[p.pid]=p.create_time()
                    if cfg['method']!='cubic':
                        cp=run/'control.csv'
                        if control_reader is None and cp.exists(): control_reader=cp.open()
                        if control_reader:
                            while True:
                                pos=control_reader.tell();line=control_reader.readline()
                                if not line:break
                                if not line.endswith('\n'):control_reader.seek(pos);break
                                if line.startswith('time_us'):continue
                                c=line.strip().split(',')
                                if len(c)!=8:raise RuntimeError('malformed control acknowledgement')
                                expected=min(2147483647,max(4,int(c[3])*int(c[4])//100))
                                if c[1]!=c[2] or int(c[5])!=expected or int(c[6])!=0:
                                    raise RuntimeError('control mapping/acknowledgement mismatch: '+line.strip())
                    if time.monotonic()-last_resource<25:
                        stop.wait(1); continue
                    last_resource=time.monotonic()
                    cpu=psutil.cpu_times_percent(interval=None)._asdict()
                    resources.append({'time':time.time(),'run_id':cfg['run_id'],'load1':os.getloadavg()[0],
                        'mem_available':psutil.virtual_memory().available,'cpu':cpu,
                        'processes':[{'pid':p.pid,'cpu_times':p.cpu_times()._asdict(),'rss':p.memory_info().rss,'cmd':p.cmdline()} for p in [psutil.Process(os.getpid()),psutil.Process(server.pid)]+descendants]})
                    (run/'resources.json').write_text(json.dumps(resources))
                except psutil.Error: pass
                except Exception:
                    errors.append(traceback.format_exc());stop.set()
                stop.wait(1)
        mt=threading.Thread(target=monitor,daemon=True); mt.start()
        if cfg['method']=='cubic':
            server.wait(timeout=cfg['seconds']+30)
        else:
            reader=StateReader(shm_r,server,stop)
            class Obj: pass
            params=Obj(); params.dict=cfg
            config=Obj(); config.task=0; config.load=None; config.eval=cfg['method']!='train'
            env=TCP_Env_Wrapper('TCP',params,config=config,for_init_only=False,shrmem_r=reader,shrmem_w=shm_w,use_normalizer=False)
            def handle(sig,frame): stop.set()
            signal.signal(signal.SIGTERM,handle); signal.signal(signal.SIGINT,handle)
            # readiness signal, then initial state; history/statistics start fresh each flow.
            state=env.reset(); history=np.zeros(70,np.float32); history[-7:]=state
            def act(hist):
                if model is None: return cfg['fixed_action'],cfg['fixed_action']
                a=float(sess.run(model.actor_out,{model.s0:hist.reshape(1,-1),model.is_training:False})[0,0])
                if cfg['method']=='train':
                    counters['exploration_calls']+=1
                    return float(np.clip(a+model.actor_noise(np.array([[a]])),-1,1)),a
                return a,a
            def send(a):
                pct=env.map_action(a)
                shm_w.write(('%d %d\0'%(reader.last_id,pct)).encode())
                return pct
            action,no_noise=act(history); percent=send(action)
            lt=None
            if cfg['method']=='train': lt=threading.Thread(target=learner,daemon=True); lt.start()
            control_start=time.monotonic(); last_flush=time.monotonic()
            while not stop.is_set():
                try: s1,_,reward,valid=env.get_state(evaluation=config.eval)
                except EndOfRun: break
                raw=reader.last_raw
                if not valid or len(s1)!=7 or not np.all(np.isfinite(s1)) or not np.isfinite(reward):
                    counters['invalid']+=1; raise RuntimeError('invalid state/reward; refusing misaligned transition')
                h1=np.concatenate((history[7:],s1)).astype(np.float32)
                a1,nn1=act(h1)
                # Send next action before enqueue/replay/log work. Learner never gates sender.
                p1=send(a1)
                counters['valid']+=1
                if cfg['method']=='train': model.rp_buffer.append(history,action,reward,h1,False)
                sw.writerow([time.time()*1e6,time.monotonic()-control_start,counters['valid'],reader.last_id,action,no_noise,percent,
                    reward,raw[1]/env.max_bw,5*raw[7]/env.max_bw,s1[-1],env.max_bw,raw[5],raw[3],raw[8],raw[14],json.dumps(s1.tolist())])
                history=h1; action=a1; no_noise=nn1; percent=p1
                if time.monotonic()-last_flush>5: states.flush(); last_flush=time.monotonic()
                if cfg['method']=='train' and counters['valid']>=cfg['steps']: break
            done.set()
            # Stop the whole corresponding flow before draining learner or saving models.
            safe_cleanup(server,owned); stop.set()
            if lt:
                lt.join(timeout=cfg.get('drain_timeout_s',60))
                if lt.is_alive(): raise RuntimeError('learner did not drain earned updates')
                if errors: raise RuntimeError(errors[0])
                checkpoint(counters['valid'])
                n=model.rp_buffer.inserted; it=counters['critic_iterations']
                tw.writerow([time.monotonic()-start,counters['valid'],n,model.rp_buffer.length_buf,it,it*2,it,it,it/max(1,n),it/max(1,n-cfg['warmup']),float(np.mean(model.rp_buffer.sample_age))]+[float('nan')]*6)
        if cfg['method']=='train' and counters['valid']!=cfg['steps']: raise RuntimeError('incomplete training sample budget')
        if errors:raise RuntimeError(errors[0])
        if cfg['method']!='cubic' and counters['valid']<max(5,cfg['seconds']*20): raise RuntimeError('insufficient control progress')
        status='COMPLETED'
    except Exception as e:
        status='FAILED'; reason=str(e); (run/'failure.txt').write_text(traceback.format_exc())
    finally:
        done.set(); stop.set()
        if server is not None: safe_cleanup(server,owned)
        for shm in (shm_r,shm_w):
            if shm:
                try: shm.remove()
                except sysv_ipc.ExistentialError: pass
        states.close(); training.close()
        result=dict(cfg=cfg,status=status,reason=reason,counters=counters,wall_s=time.monotonic()-start,
                    inserted=model.rp_buffer.inserted if model else 0,server_returncode=server.returncode if server else None)
        (run/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        if sess: sess.close()
    if status!='COMPLETED': sys.exit(1)

if __name__=='__main__': main()
