"""Frozen official-model evaluation only: no learner, optimizer, replay insertion or exploration."""
import pathlib,os,sys,time,json,csv,subprocess,signal,threading,hashlib,traceback
R=pathlib.Path(__file__).resolve().parent
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(R/'code/runtime/_deps'))
import psutil,sysv_ipc,numpy as np,tensorflow as tf
class EndOfRun(Exception):pass
class StateReader:
    def __init__(self,shm,server,stop):self.shm=shm;self.server=server;self.stop=stop;self.last_id=None;self.last_raw=None;self.legacy=False
    def read(self):
        deadline=time.monotonic()+15
        while not self.stop.is_set():
            if self.server.poll() is not None:raise EndOfRun('server exited')
            value=self.shm.read().split(b'\0')[0];fields=value.split()
            if len(fields)==16:
                try:
                    rid=int(fields[0]);vals=np.array([float(x) for x in fields[1:]])
                    valid=np.all(np.isfinite(vals)) and vals[5]>0 and vals[8]>0 and vals[14]>0
                except (ValueError,OverflowError):valid=False
                if valid and rid!=self.last_id:
                    if not self.legacy and self.last_id is not None and rid!=(self.last_id+1)%1000:raise RuntimeError('state ID gap')
                    self.last_id=rid;self.last_raw=vals;return value+b'\0'
            if time.monotonic()>deadline:raise RuntimeError('no valid state within15s')
            time.sleep(.01 if self.legacy else .0005)
        raise EndOfRun('stopped')
def cleanup(server,owned):
    if server is None:return
    try:
        for p in psutil.Process(server.pid).children(recursive=True):owned[p.pid]=p.create_time()
    except psutil.Error:pass
    owned[server.pid]=server.created
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,created in list(owned.items())[::-1]:
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-created)<.01:p.send_signal(sig)
            except psutil.Error:pass
        try:server.wait(timeout=3)
        except subprocess.TimeoutExpired:pass
        time.sleep(.2)
def model(cfg):
    which=cfg['method'];repo=R/'code'/which;sys.path.insert(0,str(repo/'rl-module'))
    if which=='orca':
        from agent import Agent
        from envwrapper import TCP_Env_Wrapper
        options={};scope='actor/'
    else:
        from agent_v2 import Agent
        from envwrapper_v2 import TCP_Env_Wrapper
        options={'use_original':False,'use_snt_model_wo_ibp':False};scope='abstract_actor/'
    params=json.loads((repo/'params_base_eval.json').read_text())
    a=Agent(70,1,params['h1_shape'],params['h2_shape'],gamma=params['gamma'],batch_size=params['batch_size'],
        lr_a=params['lr_a'],lr_c=params['lr_c'],tau=params['tau'],mem_size=1,noise_type=5,PER=False,CDQ=True,LOSS_TYPE='MSE',**options)
    assert not any('ApplyAdam' in op.type or 'ApplyGradient' in op.type for op in tf.get_default_graph().get_operations())
    variables=[v for v in tf.global_variables() if v.name.startswith(scope)]
    assert variables,'no policy variables'
    reader=tf.train.NewCheckpointReader(cfg['checkpoint']);shapes=reader.get_variable_to_shape_map()
    for v in variables:assert v.op.name in shapes and list(v.shape)==shapes[v.op.name],('missing/shape',v.name)
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2))
    a.assign_sess(sess);sess.run(tf.global_variables_initializer());tf.train.Saver(var_list=variables).restore(sess,cfg['checkpoint'])
    for v,value in zip(variables,sess.run(variables)):assert np.array_equal(value,reader.get_tensor(v.op.name)),v.name
    def digest():
        h=hashlib.sha256()
        for v,value in zip(variables,sess.run(variables)):h.update(v.name.encode());h.update(value.tobytes())
        return h.hexdigest()
    def action(hist):
        value=a.get_action(hist,False)[0] if which=='orca' else a.get_concrete_action(hist,False)[0]
        x=float(np.asarray(value).reshape(-1)[0]);assert np.isfinite(x) and -1.00001<=x<=1.00001
        if cfg.get('legacy_symbolic_components'):
            from d5_v2 import initialize_symbolic_spec
            from symbolic_transitions import update_symbolic_s_single_step,get_symbolic_actions
            if not hasattr(a,'legacy_spec'):a.legacy_spec=initialize_symbolic_spec('single_step_only_latency',.25,cfg['legacy_symbolic_components'],11)
            symbolic=update_symbolic_s_single_step(hist,a.legacy_spec,11)
            get_symbolic_actions(symbolic,a,11)
        return x
    return a,sess,TCP_Env_Wrapper,params,action,digest,[v.name for v in variables]
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(pathlib.Path(args.config).read_text());assert cfg['method'] in ('orca','canopy','cubic')
    cfg['minimum_valid_feedback_samples']=5
    if cfg.get('require_normal_cubic_start'):
        assert cfg.get('server_binary') and pathlib.Path(cfg['server_binary']).is_file(),'Explicit normal-start native required'
    cfg['feedback_completion_rule']='At least 5 valid feedback samples, unchanged 15s watchdog and fixed wall-clock duration; a low ACK-limited sample rate is measured performance rather than automatically a process failure'
    if cfg.get('verified_ipc_v2'):
        from verified_ipc import validate_native
        validate_native(cfg['server_binary'])
    if not cfg.get('code_commit'):cfg['code_commit']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD']).decode().strip()
    cfg['trace_sha256']=hashlib.sha256((R/'configs/traces'/cfg['trace']).read_bytes()).hexdigest()
    pathlib.Path(args.config).write_text(json.dumps(cfg,indent=2))
    run=R/'logs'/cfg['run_id'];run.mkdir(exist_ok=False)
    (run/'config.json').write_text(json.dumps(cfg,indent=2))
    b=R/'code/runtime';server=None;owned={};shm_r=shm_w=None;stop=threading.Event();errors=[];resources=[];sess=None
    start=time.monotonic();status='RUNNING';reason='';valid=0;before=after=None;vars_used=[];verified_pair=None
    states=(run/'states.csv').open('w',buffering=262144);sw=csv.writer(states)
    sw.writerow(['time_us','elapsed_s','transition','state_id','action','action_no_noise','percent','reward','throughput_term','loss_term','delay_term','max_bw','cwnd','delta_t','srtt_ms','min_rtt_ms','state','exploration_offset','exploration_applied','exploration_clipped'])
    try:
        if cfg['method']!='cubic':a,sess,Wrapper,params,act,digest,vars_used=model(cfg);before=digest()
        envv=dict(os.environ,C3_TRACE_DIR=str(R/'configs/traces'),ORCA_CONTROL_LOG=str(run/'control.csv'))
        if cfg['method']=='cubic':cmd=[str(b/'tcp-native'),str(cfg.get('port',46000)),str(b),'cubic',cfg['trace'],'constant192',str(cfg['delay_ms']),str(run/'down.log'),str(cfg['seconds']),str(cfg['queue_packets'])]
        else:
            shm_r=sysv_ipc.SharedMemory(None,sysv_ipc.IPC_CREX,mode=0o600,size=2048);shm_w=sysv_ipc.SharedMemory(None,sysv_ipc.IPC_CREX,mode=0o600,size=2048)
            shm_r.write(b'\0'*2048);shm_w.write(b'\0'*2048);envv.update(ORCA_SHM_R=str(shm_r.key),ORCA_SHM_W=str(shm_w.key))
            if cfg.get('verified_ipc_v2'):
                from verified_ipc import Pair
                verified_pair=Pair(shm_r,shm_w,run)
                shm_r,shm_w=verified_pair.reader,verified_pair.writer_shm
                envv.update(ORCA_VERIFIED_IPC='1',ORCA_IPC_LOCK_DIR=str(run))
            cmd=[str(pathlib.Path(cfg.get('server_binary',str(b/'orca-server')))),str(cfg.get('port',46000)),str(b),'20','4','cubic','0',cfg['trace'],'constant192',str(cfg['delay_ms']),str(run/'down.log'),str(cfg['seconds']),str(cfg['queue_packets']),'0','pretrained','0','0','200000','0','0','0','1','0','0','0','0','none','raw','0','-1']
        server=subprocess.Popen(cmd,env=envv,stdout=(run/'server.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
        server.created=psutil.Process(server.pid).create_time()
        (run/'ownership.json').write_text(json.dumps({'pid':server.pid,'started':server.created,'cmd':cmd,'shm_keys':[shm_r.key,shm_w.key] if shm_r else []},indent=2))
        def monitor():
            psutil.cpu_times_percent(interval=None);last=0;f=None
            while not stop.is_set():
                try:
                    pp=psutil.Process(server.pid);desc=pp.children(recursive=True)
                    for p in desc:owned[p.pid]=p.create_time()
                    if cfg['method']!='cubic':
                        if f is None and (run/'control.csv').exists():f=(run/'control.csv').open()
                        if f:
                            while True:
                                pos=f.tell();line=f.readline()
                                if not line:break
                                if not line.endswith('\n'):f.seek(pos);break
                                if line.startswith('time_us'):continue
                                c=line.strip().split(',');assert len(c)==8
                                expected=min(2147483647,max(4,int(c[3])*int(c[4])//100))
                                known_probe=cfg.get('handshake_probe') and c[1]=='657' and c[2]=='99999' and c[3]=='99999'
                                if (c[1]!=c[2] and not known_probe and not cfg.get('legacy_protocol')) or int(c[5])!=expected or int(c[6])!=0:raise RuntimeError('control acknowledgement mismatch')
                    if time.monotonic()-last>=25:
                        last=time.monotonic()
                        from resource_cost import monitored_processes
                        resources.append({'time':time.time(),'run_id':cfg['run_id'],'resource_inventory':'Run-owned processes including reparented Mahimahi','load1':os.getloadavg()[0],'mem_available':psutil.virtual_memory().available,
                            'cpu':psutil.cpu_times_percent(interval=None)._asdict(),'processes':[{'pid':p.pid,'cpu_times':p.cpu_times()._asdict(),'rss':p.memory_info().rss,'cmd':p.cmdline()} for p in monitored_processes(psutil.Process(os.getpid()),pp,run)]})
                        (run/'resources.json').write_text(json.dumps(resources))
                except psutil.Error:pass
                except Exception:errors.append(traceback.format_exc());stop.set()
                stop.wait(1)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        if cfg['method']=='cubic':server.wait(timeout=cfg['seconds']+30)
        else:
            reader=StateReader(shm_r,server,stop);reader.legacy=cfg.get('legacy_protocol',False);issued=[];wid=[23]
            class Obj:pass
            pp=Obj();pp.dict=params;cc=Obj();cc.task=0;cc.load=None;cc.eval=True
            env=Wrapper('TCP',pp,config=cc,for_init_only=False,shrmem_r=reader,shrmem_w=shm_w,use_normalizer=False)
            signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
            def state(info):return info['state'] if isinstance(info,dict) else info
            s=state(env.reset());hist=np.zeros(70,np.float32);hist[-7:]=s
            def send(action):
                pct=env.map_action(action);aid=wid[0] if cfg.get('legacy_protocol') else reader.last_id
                shm_w.write(('%d %d\0'%(aid,pct)).encode())
                if cfg.get('legacy_protocol'):
                    issued.append(dict(time_us=time.time()*1e6,state_id=reader.last_id,action_id=aid,percent=pct));wid[0]=(wid[0]+1)%1000
                return pct
            action=act(hist);no_noise=globals().get("latest_nn",action);exploration=globals().get('latest_exploration',(None,None,None));pct=send(action);control_start=time.monotonic();lastflush=control_start
            while not stop.is_set():
                try:info,delay,reward,ok=env.get_state(evaluation=True)
                except EndOfRun:break
                s1=state(info);raw=reader.last_raw
                if not ok or len(s1)!=7 or not np.all(np.isfinite(s1)) or not np.isfinite(reward):raise RuntimeError('invalid feedback; abort without fallback')
                h1=np.concatenate((hist[7:],s1)).astype(np.float32);next_action=act(h1);next_nn=globals().get("latest_nn",next_action);next_exploration=globals().get('latest_exploration',(None,None,None));nextpct=send(next_action)
                valid+=1
                if cfg.get('old_eval_handoff') and valid==1:
                    # Outer d5 loop sends a1, then evaluate_TCP computes and sends it again
                    # with a fresh sequence ID before reading another state.
                    duplicate_action=act(h1);send(duplicate_action)
                sw.writerow([time.time()*1e6,time.monotonic()-control_start,valid,reader.last_id,action,no_noise,pct,reward,raw[1]/env.max_bw,5*raw[7]/env.max_bw,s1[-1],env.max_bw,raw[5],raw[3],raw[8],raw[14],json.dumps(s1.tolist())]+list(exploration))
                training_reward=reward
                if globals().get('reward_hook'):training_reward=reward_hook(hist,action,no_noise,reward,h1,raw)
                if globals().get('transition_hook'):transition_hook(hist,action,training_reward,h1)
                hist=h1;action=next_action;no_noise=next_nn;exploration=next_exploration;pct=nextpct
                if cfg.get('training_steps') and valid>=cfg['training_steps']:break
                if cfg.get('stop_file') and pathlib.Path(cfg['stop_file']).exists():break
                if time.monotonic()-lastflush>5:states.flush();lastflush=time.monotonic()
            if cfg.get('legacy_protocol'):(run/'issued-actions.json').write_text(json.dumps(issued))
            cleanup(server,owned);stop.set();after=digest();assert before==after,'frozen policy variables changed'
            if valid<cfg['minimum_valid_feedback_samples']:raise RuntimeError('insufficient control progress')
        if cfg.get('require_normal_cubic_start') and cfg['method']!='cubic':
            import re
            matches=re.findall(r'NORMAL_CUBIC_FIRST_FEEDBACK cwnd=(\d+) ssthresh=(\d+) srtt_scaled8=(\d+) min_rtt=(\d+) prepolicy_cwnd_writes=0', (run/'server.log').read_text())
            assert len(matches)==1,'Missing or repeated first-feedback provenance'
            assert 0<int(matches[0][0])<1000000,'Abnormal window before first policy action'
            (run/'startup-audit.json').write_text(json.dumps(dict(cwnd=int(matches[0][0]),ssthresh=int(matches[0][1]),prepolicy_cwnd_writes=0,source='Native first TCP snapshot, before first policy write')))
        if cfg.get('verified_ipc_v2'):
            native_text=(run/'server.log').read_text()
            assert 'IPC_V2_ENABLED' in native_text and 'IPC_V2_REJECT' not in native_text,'Verified native rejected transport'
        if errors:raise RuntimeError(errors[0])
        status='COMPLETED'
    except Exception as e:
        status='FAILED';reason=str(e);(run/'failure.txt').write_text(traceback.format_exc())
    finally:
        stop.set();cleanup(server,owned)
        for shm in (shm_r,shm_w):
            if shm:
                try:shm.remove()
                except sysv_ipc.ExistentialError:pass
        if verified_pair:verified_pair.close()
        states.close()
        result={'cfg':cfg,'status':status,'reason':reason,'counters':{'valid':valid,'critic_iterations':0,'critic_optimizer_ops':0,'actor_updates':0,'target_updates':0,'exploration_calls':0},
            'inserted':0,'wall_s':time.monotonic()-start,'policy_sha256_before':before,'policy_sha256_after':after,'policy_variables':vars_used,
            'server_returncode':server.returncode if server else None}
        (run/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        if globals().get('producer_finish'):producer_finish()
        if sess and not globals().get('distributed_session'):sess.close()
    if status!='COMPLETED':sys.exit(1)
if __name__=='__main__':main()
