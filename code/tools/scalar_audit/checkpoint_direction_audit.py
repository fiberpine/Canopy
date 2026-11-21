"""Local-Session, read-only scalar checkpoint audit. No TCP, SSH, or optimizers run.

Execute only after the user starts the learner. Model/replay remain on that host.
Earlier checkpoints query final-replay states: distribution-shift sensitivity,
not historical on-policy observations or true-value labels.
"""
import argparse,hashlib,json,os,pathlib,sys,time

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--per-actor',type=int,default=32)
    p.add_argument('--seed',type=int,default=261026)
    p.add_argument('--max-seconds',type=int,default=600)
    return p.parse_args()

def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    F=pathlib.Path(args.framework).resolve();out=pathlib.Path(args.output).resolve()
    assert out.suffix=='.json' and not out.exists(),'Use a new diagnostic output path'
    assert 1<=args.per_actor<=64 and 1<=args.max_seconds<=900
    assert args.run and '/' not in args.run and '\\' not in args.run
    cpdir=F/'checkpoints'/args.run;cfgpath=F/'configs'/(args.run+'.json')
    assert cpdir.is_dir() and cfgpath.is_file()
    cfg=json.loads(cfgpath.read_text());assert cfg['policy_family']=='orca' and cfg['num_actors']==24
    out.parent.mkdir(parents=True,exist_ok=True);os.chdir(str(out.parent))
    sys.path[:0]=[str(F),str(F/'code/runtime/_deps')]
    import numpy as np,tensorflow as tf
    assert tf.__version__.startswith('1.'),'Run in the actual preserved TF1 environment'
    from orca_backend import create
    from orca_moving_actor_loss import forward
    started=time.time();result=dict(status='RUNNING_READ_ONLY',framework=str(F),run=args.run,
        tf_version=tf.__version__,cloud_network_connections=0,new_TCP_runs=0,optimizer_updates=0,actor_release=False,
        provenance=dict(config_sha256=hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
            source_sha256={n:hashlib.sha256((F/n).read_bytes()).hexdigest() for n in ['orca_backend.py','orca_moving_actor_loss.py','independent_critic_v34.py']}),
        interpretation='Actual saved networks on retained factual training-replay rows. This is model sensitivity/self-consistency; not heldout true-Q accuracy or real-network counterfactual performance.',
        earlier_checkpoint_limit='All use final retained replay states; earlier checkpoint queries are OOD sensitivity, not the actual states at that training phase.',rows=[])
    def write():
        temp=out.with_name(out.name+'.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(str(out))
    write()
    try:
        result['provenance']['vendor_sha256']={n:hashlib.sha256((F/'code/orca/rl-module'/n).read_bytes()).hexdigest() for n in ['agent.py','envwrapper.py','utils.py']}
        data=np.load(str(cpdir/'replay.npz'));ids=data['actor_ids'];rng=np.random.RandomState(args.seed)
        chosen=[]
        for i in range(24):
            pool=np.where(ids==i)[0];assert len(pool)>=args.per_actor,(i,len(pool))
            chosen.extend(rng.choice(pool,args.per_actor,replace=False).tolist())
        ix=np.asarray(chosen);s0=data['s0'][ix];actions=data['a'][ix]
        assert s0.shape==(24*args.per_actor,70) and actions.shape==(len(ix),1)
        raw_batch={k:data[k][ix] for k in ['s1','reward','terminal']}
        result['query_rows']=len(ix);result['query_array_sha256']=hashlib.sha256(s0.tobytes()+actions.tobytes()).hexdigest()
        result['sample_birth_range']=[int(data['birth'][ix].min()),int(data['birth'][ix].max())]
        tf.reset_default_graph();tf.set_random_seed(args.seed)
        build_cfg=dict(cfg,memsize=1)
        a=create(F,build_cfg,'learner');a.build_learn()
        variables=tf.global_variables();before_names=[v.op.name for v in variables]
        deployed=forward(a)
        deployed_q=a.critic.build(a.s0,deployed)
        actual_grad=[g for g in tf.gradients(a.a_loss,a.actor.train_var()) if g is not None]
        deployed_grad=[g for g in tf.gradients(-tf.reduce_mean(deployed_q),a.actor.train_var()) if g is not None]
        action_grad=[tf.gradients(q,a.action)[0] for q in [a.critic_out,a.critic_out2]]
        assert before_names==[v.op.name for v in tf.global_variables()],'Diagnostic must not create model variables'
        saver=tf.train.Saver(var_list=variables)
        def digest(sess):
            h=hashlib.sha256()
            for v,x in zip(variables,sess.run(variables)):h.update(v.op.name.encode());h.update(x.tobytes())
            return h.hexdigest()
        def flatten(gs):return np.concatenate([g.ravel() for g in gs])
        def simple(x):
            x=np.asarray(x);assert np.isfinite(x).all()
            return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()))
        cps=sorted(cpdir.glob('step-*.index'),key=lambda p:int(p.stem.split('-')[1]))
        assert cps
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1,device_count={'GPU':0})) as sess:
            a.assign_sess(sess)
            for index in cps:
                assert time.time()-started<args.max_seconds,'Read-only diagnostic budget exceeded'
                cp=str(index)[:-6];reader=tf.train.NewCheckpointReader(cp)
                assert set(reader.get_variable_to_shape_map())==set(before_names),'Unexpected checkpoint schema'
                saver.restore(sess,cp);before=digest(sess)
                fd={a.s0:s0,a.is_training:False}
                pi=sess.run(deployed,fd)
                pi_training=sess.run(a.actor_out,{a.s0:s0,a.is_training:True})
                gs=sess.run(actual_grad,{a.s0:s0,a.is_training:True});gd=sess.run(deployed_grad,fd)
                gs,gd=flatten(gs),flatten(gd)
                q1,q2,g1,g2=sess.run([a.critic_out,a.critic_out2]+action_grad,
                    {a.s0:s0,a.action:pi,a.is_training:False})
                # Explicit grid: a critic extrapolation diagnostic, never a deployment selector.
                grid=np.linspace(-1.,1.,9,dtype=np.float32)
                xx=np.repeat(s0,9,axis=0);aa=np.tile(grid,len(s0)).reshape(-1,1)
                z1,z2=sess.run([a.critic_out,a.critic_out2],{a.s0:xx,a.action:aa,a.is_training:False})
                grid_q=np.minimum(z1,z2).reshape(len(s0),9);best=grid[grid_q.argmax(axis=1)]
                neutral_gap=grid_q[:,4]-np.minimum(q1,q2).reshape(-1)
                factual={a.s0:s0,a.action:actions,a.s1:raw_batch['s1'],a.reward:raw_batch['reward'],a.terminal:raw_batch['terminal'],a.is_training:False}
                factual1,factual2,y1,y2=sess.run([a.critic_out,a.critic_out2,a.y,a.y2],factual)
                target=np.minimum(y1,y2)
                by_actor={}
                for i in range(24):
                    z=slice(i*args.per_actor,(i+1)*args.per_actor)
                    by_actor[str(i)]=dict(condition=cfg['conditions'][i]['condition'],policy=simple(pi[z]),factual_action=simple(actions[z]),
                        negative_boundary_fraction=float(np.mean(pi[z]<=-.99)),nonreducing_factual_fraction=float(np.mean(actions[z]>=0)),
                        critic1_action_gradient=simple(g1[z]),critic2_action_gradient=simple(g2[z]),
                        neutral_Q_minus_policy_Q=simple(neutral_gap[z]),
                        both_critics_predict_increase_fraction=float(np.mean((g1[z]>0)&(g2[z]>0))),
                        critic_direction_disagreement_fraction=float(np.mean(g1[z]*g2[z]<0)),
                        grid_greedy_action=simple(best[z]),note='Grid ranking/gradients are learned-network predictions; unsupported actions may be wrong.')
                assert digest(sess)==before,'A read-only query changed checkpoint variables'
                record=dict(checkpoint=cp,model_variables_unchanged=True,training_global_step=int(sess.run(a.global_step)),
                    policy=simple(pi),training_forward_gap=simple(np.abs(pi-pi_training)),
                    actual_actor_gradient_norm=float(np.linalg.norm(gs)),deployed_actor_gradient_norm=float(np.linalg.norm(gd)),
                    gradient_cosine=float(gs.dot(gd)/max(1e-30,np.linalg.norm(gs)*np.linalg.norm(gd))),
                    tanh_derivative=simple(1-pi*pi),q1_at_policy=simple(q1),q2_at_policy=simple(q2),
                    factual_TD_rmse_q1=float(np.sqrt(np.mean((factual1-target)**2))),factual_TD_rmse_q2=float(np.sqrt(np.mean((factual2-target)**2))),
                    factual_TD_limit='Own stochastic smoothed bootstrap target, retained training replay; self-consistency only.',by_actor=by_actor)
                result['rows'].append(record);write()
        result.update(status='COMPLETE_READ_ONLY_CHECKPOINT_DIRECTION_AUDIT',elapsed_s=time.time()-started);write()
        print(json.dumps(dict(status=result['status'],checkpoints=len(result['rows']),query_rows=len(ix),elapsed_s=result['elapsed_s'],optimizer_updates=0,output=str(out))))
    except Exception:
        import traceback
        result.update(status='FAIL_READ_ONLY_DIAGNOSTIC',traceback=traceback.format_exc());write();raise

if __name__=='__main__':main(parse_args())
