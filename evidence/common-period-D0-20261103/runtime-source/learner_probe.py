"""Frozen D0 gradient/BN and CPU queries. Never calls a training or save entrypoint."""
import os,pathlib,json,sys,time,hashlib,math,gc
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',CUDA_VISIBLE_DEVICES='')
import numpy as np
from d0_core import PERIODS,summarize,cosine,expected_actor_period,ReadOnlySession
OLD=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102')
sys.path.insert(0,str(OLD/'A/framework'));sys.path.insert(0,str(OLD/'A/framework/code/runtime/_deps'))
from common_period_tf import Model,ONLINE
from common_period import encoded_period
from common_period_transport import Policy,snapshot_bytes
def hash_variables(model,variables):
    h=hashlib.sha256()
    for v,x in zip(variables,model.session.run(variables)):h.update(v.op.name.encode());h.update(np.asarray(x).tobytes())
    return h.hexdigest()
def run(reg):
    started=time.time();selection=json.loads((OLD/'evaluation/selected-model.json').read_text());prefix=selection['checkpoint'];meta=json.loads(pathlib.Path(prefix+'.metadata.json').read_text());cfg=json.loads((OLD/'A/config.json').read_text())
    assert selection['files']==reg['model_files']
    m=Model(cfg['source_checkpoint'],meta['K'],meta['n0'],reward_config=cfg['common_reward_config']);m.restore(prefix)
    with m.graph.as_default():variables=m.tf.global_variables()
    before=hash_variables(m,variables);actual_session=m.session;guard=ReadOnlySession(actual_session);m.session=guard
    reader=m.tf.train.NewCheckpointReader(prefix);final_version=int(reader.get_tensor('policy_snapshot_version'))
    data=np.load(prefix+'.replay.npz',allow_pickle=False);counts=np.zeros((24,5),int);idx=[];rng=np.random.RandomState(reg['seed'])
    for actor in range(24):
        for j,t in enumerate(PERIODS):
            choices=np.flatnonzero((data['actor']==actor)&(data['period_ms']==t));counts[actor,j]=len(choices);assert len(choices)>=reg['replay_samples_per_actor_period'];idx.extend(rng.choice(choices,reg['replay_samples_per_actor_period'],replace=False).tolist())
    ids=np.asarray(idx);s=data['s0'][ids].astype(np.float32);sp=data['s1'][ids].astype(np.float32);a=data['action'][ids].astype(np.float32);g=data['discount'][ids].astype(np.float32).reshape(-1,1);boot=data['bootstrap'][ids].astype(np.float32).reshape(-1,1);r=data['reward'][ids].astype(np.float32).reshape(-1,1)-np.float32(meta['K'])*(1-g)
    terminal=data['kind'][ids]=='terminated';r[terminal]=data['reward'][ids][terminal].reshape(-1,1)-np.float32(meta['K'])
    with m.graph.as_default():
        tf=m.tf;av=m.parameters(ONLINE);gr=tf.gradients(m.rl_loss,av);gz=tf.gradients(m.reg_loss,av)
        flatten=lambda grads:tf.concat([tf.reshape(v,[-1]) for v in grads if v is not None],axis=0)
        rl=flatten(gr);regularizer=flatten(gz);layers={}
        for label,piece in [('shared_fc1','fc1/'),('shared_fc2','fc2/'),('output','fc1_1/'),('period_input','period_w')]:layers[label]=flatten([grad for grad,var in zip(gr,av) if grad is not None and piece in var.op.name])
        q2mu=m.critic(m.s,tf.concat([m.mu,m.t],1),'critic2/');nmu=m.number(m.s,tf.concat([m.mu,m.t],1),'number/');dq1=tf.gradients(m.actor_q,m.mu)[0];dqmean=tf.gradients((m.actor_q+q2mu)*.5,m.mu)[0]
        cg=[tf.global_norm([x for x in tf.gradients(loss,m.parameters(p)) if x is not None]) for loss,p in zip(m.losses,['critic/','critic2/'])]
        def forward(path):
            v=m.v;bn=lambda x,n:tf.nn.batch_normalization(x,v[ONLINE+n+'/moving_mean'],v[ONLINE+n+'/moving_variance'],v[ONLINE+n+'/beta'],None,.001)
            x1=tf.matmul(m.s,v[ONLINE+'fc1/w'])+v[ONLINE+'fc1/b'];mean1,var1=tf.nn.moments(x1,[0]);h1=tf.nn.leaky_relu(bn(x1,'bn1') if path=='moving' else tf.nn.batch_normalization(x1,mean1,var1,v[ONLINE+'bn1/beta'],None,.001))
            x2=tf.matmul(h1,v[ONLINE+'fc2/w'])+v[ONLINE+'fc2/b']+tf.matmul(m.t,v[ONLINE+'period_w']);mean2,var2=tf.nn.moments(x2,[0]);h2=tf.nn.leaky_relu(tf.nn.batch_normalization(x2,mean2,var2,v[ONLINE+'bn2/beta'],None,.001) if path=='batch_both' else bn(x2,'bn2'));z=(tf.matmul(h2,v[ONLINE+'fc1_1/w'])+v[ONLINE+'fc1_1/b'])[:,:1]
            return tf.tanh(z),mean2,var2
        moving=forward('moving');batch1=forward('batch1_moving2');batch2=forward('batch_both')
    # Consistency of the actual actor-only transport with the frozen learner actor.
    p=Policy(cfg['source_checkpoint'],meta['reward_checksum']);actor_values={v.op.name:x for v,x in zip(variables,actual_session.run(variables)) if v.op.name.startswith(ONLINE) and not any('/'+n in v.op.name for n in ['Q1Adam','Q2Adam','ActorAdam','NAdam'])}
    p.load(snapshot_bytes(dict(version=final_version,schema=meta['schema'],reward_checksum=meta['reward_checksum'],values=actor_values)))
    pbefore=hashlib.sha256(b''.join(x.tobytes() for x in p.session.run(list(p.v.values())))).hexdigest()
    consistency=[]
    for i in range(reg['forward_parity_states']):
        period=PERIODS[i%5];action=np.array([[0,encoded_period(period)]],np.float32);y=guard.run(m.mu,{m.s:s[i:i+1],m.action:action})[0,0];v=p.window(s[i],period);consistency.append(abs(float(y)-v))
    assert max(consistency)<=reg['tolerances']['actor_abs']
    # S5 runs alone: no node scans or other D0 heavy tasks remain active.
    cpu=[];state_ids=rng.permutation(len(s))[:reg['cpu_state_pool']];pool=s[state_ids];candidate=np.array([[0,encoded_period(t)] for t in PERIODS],np.float32)
    def actor_call(x):return guard.run(m.mu,{m.s:x[None,:],m.action:candidate[2:3]})
    def selector_call(x):
        states=np.repeat(x[None,:],5,axis=0);actions=candidate.copy();actions[:,0]=guard.run(m.mu,{m.s:states,m.action:actions})[:,0];q1,q2,n=guard.run(m.q+[m.n],{m.s:states,m.action:actions});return int(np.argmax((q1+q2)*.5-np.float32(0)*n)),n
    from decision_clock_v21 import wire
    bridge_action=np.array([.1,encoded_period(80)],np.float32)
    def bridge_call(x):return wire(123,bridge_action,cfg['eval_base'])
    functions={'actor_only':actor_call,'selector_5_two_fetch':selector_call,'bridge_wire_only':bridge_call}
    for f in functions.values():
        for i in range(reg['cpu_warmup']):f(pool[i%len(pool)])
    for block in range(reg['cpu_blocks']):
        order=list(functions);rng.shuffle(order)
        for name in order:
            if time.time()>reg['deadline_epoch']-600:raise TimeoutError('D0 reserve reached')
            lat=[];f=functions[name];gc.collect();gc.disable();c0=time.process_time();w0=time.perf_counter()
            for i in range(reg['cpu_repetitions']):
                t0=time.perf_counter();f(pool[(i+block*17)%len(pool)]);lat.append((time.perf_counter()-t0)*1000)
            cpu_ms=(time.process_time()-c0)*1000/reg['cpu_repetitions'];wall_ms=(time.perf_counter()-w0)*1000/reg['cpu_repetitions'];gc.enable();cpu.append(dict(block=block,variant=name,process_cpu_ms_per_call=cpu_ms,wall_ms_per_call=wall_ms,latency_ms=summarize(lat)))
    projections=[]
    ac=np.mean([x['process_cpu_ms_per_call'] for x in cpu if x['variant']=='actor_only']);sel=np.mean([x['process_cpu_ms_per_call'] for x in cpu if x['variant']=='selector_5_two_fetch'])
    for period,rate in reg['observed_calls_per_s'].items():projections.append(dict(period_reference_ms=int(period),observed_fixed_actor_calls_per_s=rate,actor_cpu_one_core_percent=ac*rate/10,selector_at_same_calls_cpu_one_core_percent=sel*rate/10,increment_percent=(sel-ac)*rate/10))
    # S1/S3: fixed retained training states, never relabeled deployment states.
    local=[];bn=[];gradients=[];permutation=rng.permutation(len(s));shared=s[permutation[:reg['gradient_batches']*reg['gradient_batch_size']]]
    for block in range(reg['gradient_batches']):
        state=shared[block*reg['gradient_batch_size']:(block+1)*reg['gradient_batch_size']];vectors={};layervectors={k:{} for k in layers}
        for period in PERIODS:
            if time.time()>reg['deadline_epoch']-600:raise TimeoutError('D0 reserve reached')
            action=np.zeros((len(state),2),np.float32);action[:,1]=encoded_period(period);feed={m.s:state,m.action:action};vals=guard.run([m.mu,m.z,dq1,dqmean,rl,regularizer]+list(layers.values()),feed);mu,z,qgrad,meanqgrad,rg,zg=vals[:6];derivative=1-mu*mu
            clone,b1,b2=guard.run([moving[0],batch1[0],batch2[0]],feed);assert np.max(abs(clone-mu))<=reg['tolerances']['actor_abs']
            local.append(dict(block=block,queried_period_ms=period,samples=len(state),boundary_fraction=float(np.mean(abs(mu)>.99)),float32_exact_saturation=float(np.mean(abs(mu)==1)),float32_tanh_derivative=summarize(derivative),z=summarize(z),Q1_dQ_da=summarize(qgrad),Qmean_dQ_da=summarize(meanqgrad),RL_dloss_dz=summarize(-qgrad*derivative),regularizer_dloss_dz=summarize(2*.001*z),zero_RL_local_fraction=float(np.mean(qgrad*derivative==0))))
            bn.append(dict(block=block,period_ms=period,moving_forward_max_error=float(np.max(abs(clone-mu))),batch1_moving2_action_abs_difference=summarize(abs(b1-mu)),batch_both_action_abs_difference=summarize(abs(b2-mu))))
            gradients.append(dict(block=block,period_ms=period,rl_norm=float(np.linalg.norm(rg)),regularizer_norm=float(np.linalg.norm(zg)),rl_regularizer_cosine=cosine(rg,zg)));vectors[period]=rg
            for key,value in zip(layers,vals[6:]):layervectors[key][period]=value
        for i,t1 in enumerate(PERIODS):
            for t2 in PERIODS[i+1:]:
                gradients.append(dict(block=block,pair_ms=[t1,t2],rl_cosine=cosine(vectors[t1],vectors[t2]),by_layer={key:cosine(v[t1],v[t2]) for key,v in layervectors.items()}))
    critics=[]
    for period in PERIODS:
        pos=np.flatnonzero(data['period_ms'][ids]==period);pos=pos[:reg['critic_samples_per_period']];feed={m.s:s[pos],m.sp:sp[pos],m.action:a[pos],m.r:r[pos],m.g:g[pos],m.b:boot[pos],m.noise:np.zeros((len(pos),1),np.float32)}
        losses,norms=guard.run([m.losses,cg],feed);critics.append(dict(period_ms=period,samples=len(pos),Q_MSE=[float(v) for v in losses],Q_gradient_norm=[float(v) for v in norms],target_description='Original A smoothed/min target graph queried with fixed zero smoothing-noise realization; no optimization'))
    startup=[]
    for item in reg.get('startup_samples',[]):
        x=np.asarray(item['state'],np.float32)[None,:];action=np.array([[0,encoded_period(item['period_ms'])]],np.float32);mu,z,gradient=guard.run([m.mu,m.z,dq1],{m.s:x,m.action:action});startup.append(dict(actor=item['actor'],run=item['run'],period_ms=item['period_ms'],z=float(z[0,0]),action=float(mu[0,0]),tanh_derivative=float(1-mu[0,0]**2),Q1_action_gradient=float(gradient[0,0]),source='Recorded training STARTUP components; queried final frozen model, not the historical online policy'))
    after=hash_variables(m,variables);pafter=hashlib.sha256(b''.join(x.tobytes() for x in p.session.run(list(p.v.values())))).hexdigest();assert before==after and pbefore==pafter
    versions=data['snapshot'];out=dict(status='COMPLETED_FROZEN_QUERIES',started_epoch=started,ended_epoch=time.time(),model_variables_unchanged=before==after,actor_transport_variables_unchanged=pbefore==pafter,variable_digest_before=before,variable_digest_after=after,optimizer_operations=0,new_TCP=0,read_only_query_calls=guard.calls,sample_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),sample_count=len(ids),source='Final retained real TRAINING replay; not frozen deployment-state support',S1=dict(retained_actor_period_counts=counts.tolist(),expected_final_actor_balanced_period_weights=expected_actor_period(counts),actual_sampled_by_actor=meta['ledger']['sampled_by_actor'],historical_sampled_by_period='NOT_RECORDED',contemporaneous_policy_lag='NOT_RECORDED',final_policy_version=final_version,retained_distance_to_final_version=summarize(final_version-versions),version_distance_caveat='Final-model distance includes post-collection updates and is not learner/actor lag at decision time',actor_gradients=gradients,critic_gradients=critics),S3=dict(deployment_failure_state_queries='NOT_RECORDED',training_state_local_queries=local,BN_counterfactual_paths=bn,actor_only_vs_learner_max_abs_error=max(consistency),interpretation='BN alternatives are read-only diagnostic forwards, not production modifications; batch observations are not independent TCP replications'),S5=dict(status='MEASURED_MICROBENCHMARK',threads=dict(intra=1,inter=1,OMP=1,OPENBLAS=1),records=cpu,actor_cpu_ms=float(ac),selector_cpu_ms=float(sel),ratio=float(sel/ac),cost_projection=projections,lambda_scan=False,interpretation='Same-host same-state actual two-fetch selector structure; includes read-only guard overhead and Python bridge. Projections assume fixed observed call rates; not closed-loop policy/network/native CPU benefit. Six within-process blocks measure timing variation, not independent network runs.'))
    out['S3']['training_startup_queries']=startup;p.session.close();actual_session.close();return out
if __name__=='__main__':
    root=pathlib.Path(__file__).resolve().parent.parent;reg=json.loads((root/'registration.json').read_text());result=run(reg);(root/'learner-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],unchanged=result['model_variables_unchanged'],samples=result['sample_count'],actor_cpu_ms=result['S5']['actor_cpu_ms'],selector_cpu_ms=result['S5']['selector_cpu_ms'],ended_epoch=result['ended_epoch'])))
