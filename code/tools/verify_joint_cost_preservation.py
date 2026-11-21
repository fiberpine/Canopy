"""Actual TF1 repair graph checks; only temporary synthetic updates, never TCP."""
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def verify(config,output,states_file=None):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model,WINDOW
    from frameworks.joint_healthy.core import interval_return
    cfg=json.loads(Path(config).read_text(encoding='utf-8').replace('@PACKAGE@',str(ROOT).replace('\\','/')))
    if states_file:
        with np.load(str(states_file)) as a:states=a['states'][:128]
    else:states=np.random.RandomState(53).uniform(0,1,(128,82)).astype(np.float32)
    with tf.Graph().as_default():
        model=Model(cfg,training=True)
        anchor_time=tf.gradients(model.anchor_loss,model._variables('mtp_actor',True))
        stats=[v for v in model.window_vars if v.op.name.endswith(('/moving_mean','/moving_variance'))]
        target_stats=[b for a,b in model.target_pairs if a.op.name.endswith(('/moving_mean','/moving_variance'))]
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
            teacher,action=sess.run([model.teacher_output,model.action],{model.s:states})
            parity=float(np.max(np.abs(teacher[:,0]-action[:,0])));assert parity<1e-5,parity
            old_stats=sess.run(stats+target_stats);window_before=sess.run(model.window_vars);time_before=sess.run(model.time_vars)
            features=sess.run(model.time_features,{model.s:states})
            if cfg.get('time_feature_normalization')=='unit_l2_v1':
                assert np.max(np.linalg.norm(features,axis=1))<=1.000001
            if cfg.get('critic_initial_cost_reference_s',0)>0:
                rd=tf.train.NewCheckpointReader(cfg['baseline_checkpoint'])
                offset=cfg['decision_cost']/(1-cfg['gamma']**(cfg['critic_initial_cost_reference_s']/cfg['base_period_s']))
                for scope in ('critic','critic2','target_critic','target_critic2'):
                    source=scope.replace('target_','')+'/dense/bias'
                    value=sess.run(next(v for v in model._variables(scope) if v.op.name==scope+'/dense/bias'))
                    assert np.max(np.abs(value-(rd.get_tensor(source)-offset)))<1e-6
            durations=np.resize(np.array([.02,.04,.08,.16],np.float64),len(states))[:,None]
            returns=[interval_return(.9,float(t),cfg['decision_cost']) for t in durations[:,0]]
            rewards=np.array([v[0] for v in returns],np.float32)[:,None];discounts=np.array([v[1] for v in returns],np.float32)[:,None]
            batch=[states,action,rewards,states*.99,np.zeros((len(states),1),np.float32),durations,discounts,teacher]
            feed=model.feed(batch);sess.run(tf.assign(model.phase,2))
            anchor_gradients=sess.run([g for g in anchor_time if g is not None],feed)
            assert all(np.count_nonzero(g)==0 for g in anchor_gradients),'Teacher regularization must not train the time head'
            for _ in range(30):
                sess.run(model.critic_step,feed);sess.run([model.actor_step,model.window_update_ops],feed);sess.run(model.target_update)
            window_after=sess.run(model.window_vars);time_after=sess.run(model.time_vars)
            assert all(np.array_equal(a,b) for a,b in zip(old_stats,sess.run(stats+target_stats)))
            window_change=max(float(np.max(np.abs(a-b))) for a,b in zip(window_before,window_after));assert window_change>0
            time_change=max(float(np.max(np.abs(a-b))) for a,b in zip(time_before,time_after));assert time_change>0
            assert np.array_equal(teacher,sess.run(model.teacher_output,{model.s:states}))
            assert sess.run(model.global_step)==90
            for variable in model._variables(WINDOW,True):
                assert model.window_actor_optimizer.get_slot(variable,'m') is not None
                assert model.time_actor_optimizer.get_slot(variable,'m') is None
            for variable in model._variables('mtp_actor',True):
                assert model.time_actor_optimizer.get_slot(variable,'m') is not None
                assert model.window_actor_optimizer.get_slot(variable,'m') is None
            gradients=sess.run([model.window_grad,model.time_grad],feed);assert all(np.isfinite(g) and g>0 for g in gradients),gradients
            expected=sess.run(model.action,{model.s:states})
            with tempfile.TemporaryDirectory(prefix='joint-cost-verify-') as td:
                prefix=str(Path(td)/'temporary');tf.train.Saver().save(sess,prefix,write_meta_graph=False);model.save_manifest(prefix,cfg,{})
                sess.run(tf.global_variables_initializer());model.restore(sess,prefix)
                restore_error=float(np.max(np.abs(expected-sess.run(model.action,{model.s:states}))));assert restore_error==0
    result=dict(status='PASS_ACTUAL_TF1_COST_PRESERVATION_GRAPH_NOT_TCP_HEALTH',fixed_inputs=len(states),initial_teacher_window_error=parity,
        time_feature_normalization=cfg.get('time_feature_normalization','none'),initial_time_feature_max_norm=float(np.max(np.linalg.norm(features,axis=1))),
        critic_initial_cost_reference_s=cfg.get('critic_initial_cost_reference_s',0.),
        both_branches_changed=True,window_parameter_max_change=window_change,time_parameter_max_change=time_change,
        all_online_and_target_bn_statistics_bitwise_unchanged=True,teacher_prediction_bitwise_unchanged=True,
        teacher_loss_time_parameter_gradients_exactly_zero=True,independent_actor_optimizer_slots=True,
        synthetic_complete_critic_updates=30,synthetic_complete_joint_updates=30,global_step=90,
        gradient_norms=list(map(float,gradients)),save_restore_action_error=restore_error,real_tcp_runs=0,physical_experience=0)
    Path(output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--states')
    a=p.parse_args();verify(a.config,a.output,a.states)
