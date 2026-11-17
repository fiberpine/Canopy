"""Predeclared small-advantage and state-dependent continuous-action tasks.

Known continuation values are used ONLY in these constructed offline tasks.
No prediction is presented as a real TCP counterfactual or calibration proof.
"""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(config, states_file, output, updates=10000, critic_reference=False):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    cfg=json.loads(Path(config).read_text().replace('@PACKAGE@',str(ROOT).replace('\\','/')))
    critic_cfg=dict(cfg)
    if critic_reference:
        from frameworks.joint_healthy.configuration import configuration
        critic_cfg=configuration('g4')
        critic_cfg.update(decision_cost=cfg['decision_cost'],critic_initial_cost_reference_s=cfg['critic_initial_cost_reference_s'],
                          baseline_checkpoint=cfg['baseline_checkpoint'])
    with np.load(str(states_file)) as p:source=p['states']
    # Two saved real states, deliberately different in their measured RTT.
    indices=[int(np.argmin(source[:,79])),int(np.argmax(source[:,79]))]
    states=source[indices].copy();states[:,66]=1.  # same declared fresh-feedback prior
    session_cfg=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)
    gamma,cost=.995,.01
    minimum_margin=cost*(gamma-gamma**2)/(1-gamma)
    threshold=minimum_margin/4.
    critic_results=[]
    for sign in (1.,0.,-1.):
        with tf.Graph().as_default():
            tf.set_random_seed(3);model=Model(critic_cfg,True)
            rng=np.random.RandomState(100)
            t=np.tile(np.exp(rng.uniform(np.log(20.),np.log(160.),256)),2)
            ss=np.repeat(states,256,axis=0)
            base=np.repeat(np.array([168.,169.]),256)
            a=np.column_stack([np.zeros(len(t)),2*np.log(t/20.)/np.log(8.)-1]).astype(np.float32)
            d=gamma**(t/20.)
            advantage=sign*cost*(gamma-d)/(1-gamma)
            labels=(base+advantage)[:,None].astype(np.float32)
            # Reconstruct a single current reward plus a SAME 20ms continuation.
            reward=labels[:,0].astype(np.float64)-d*base
            reconstructed=reward+d*base
            assert np.max(np.abs(reconstructed-labels[:,0]))<1e-12
            queries=np.tile(np.array([20.,40.,80.,160.]),2)
            qs=np.repeat(states,4,axis=0)
            qa=np.column_stack([np.zeros(8),2*np.log(queries/20.)/np.log(8.)-1]).astype(np.float32)
            truth=(sign*cost*(gamma-gamma**(queries/20.))/(1-gamma)).reshape(2,4)
            with tf.Session(config=session_cfg) as sess:
                sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
                trajectory=[]
                for i in range(updates+1):
                    if i in (0,500,1500,5000,updates):
                        predicted=sess.run([model.q1,model.q2],{model.s:qs,model.executed:qa})
                        errors=[]
                        for q in predicted:
                            q=q.reshape(2,4);errors.append(float(np.mean(np.abs((q-q[:,:1])-truth))))
                        trajectory.append(dict(updates=i,advantage_mae=errors))
                    if i<updates:
                        at=rng.choice(len(t),min(int(cfg['batch_size']),len(t)),replace=False)
                        sess.run(model.critic_step,{model.s:ss[at],model.executed:a[at],model.label:labels[at]})
                predictions=sess.run([model.q1,model.q2],{model.s:qs,model.executed:qa})
            reports=[]
            for prediction in predictions:
                q=prediction.reshape(2,4);adv=q-q[:,:1];difference=adv-truth
                active=np.abs(truth)>=minimum_margin*.999
                ranking=float(np.mean(np.sign(adv[active])==np.sign(truth[active]))) if sign else None
                mae=float(np.mean(np.abs(difference[:,1:])))
                reports.append(dict(advantage_mae=mae,ranking=ranking,maximum_spurious_advantage=float(np.max(np.abs(adv))) if not sign else None,
                    pass_gate=bool(mae<=threshold and (ranking is None or ranking>=.95)),predicted_advantages=adv.tolist()))
            critic_results.append(dict(sign=sign,known_advantages=truth.tolist(),critics=reports,trajectory=trajectory))
    # Exact known Q gradient tests the *actor* separately, not learned-Q validity.
    actor_results=[]
    for time_lr in (() if critic_reference else (.0001,.0003)):
        with tf.Graph().as_default():
            tf.set_random_seed(3);model=Model(cfg,False)
            goal=tf.constant([[40.],[120.]],tf.float32)
            reference_window=tf.placeholder(tf.float32,(2,1),name='fixed_initial_window_reference')
            optimal_window=tf.clip_by_value(reference_window+.12*tf.log(goal/20.)/np.log(8.),-.85,.85)
            value=-.1*tf.square((model.planned_ms-goal)/140.)-.1*tf.square(model.action[:,:1]-optimal_window)
            window_vars=model._variables('abstract_actor/abstract_actor',True)+model._variables('conditioned_window',True)
            time_vars=model._variables('mtp_actor',True)
            steps=[tf.train.AdamOptimizer(cfg['window_actor_lr'],name='capability_window').minimize(-tf.reduce_mean(value),var_list=window_vars),
                   tf.train.AdamOptimizer(time_lr,name='capability_time').minimize(-tf.reduce_mean(value),var_list=time_vars)]
            with tf.Session(config=session_cfg) as sess:
                sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
                reference=sess.run(model.base_window_action,{model.s:states})
                actor_feed={model.s:states,reference_window:reference}
                trajectory=[]
                for i in range(5001):
                    if i in (0,500,2000,5000):
                        ms,aw,ow=sess.run([model.planned_ms,model.action[:,:1],optimal_window],actor_feed)
                        trajectory.append(dict(updates=i,periods_ms=ms[:,0].tolist(),maximum_period_error_ms=float(np.max(np.abs(ms[:,0]-[40.,120.]))),
                            maximum_window_error=float(np.max(np.abs(aw-ow)))))
                    if i<5000:sess.run(steps,actor_feed)
            actor_results.append(dict(time_lr=time_lr,trajectory=trajectory,
                pass_gate=trajectory[-1]['maximum_period_error_ms']<=5. and trajectory[-1]['maximum_window_error']<=.05))
    selected_actor=[r for r in actor_results if r['time_lr']==cfg['time_actor_lr']]
    result=dict(status='PASS_OFFLINE_KNOWN_VALUE_CAPABILITY' if all(x['pass_gate'] for r in critic_results for x in r['critics']) and (critic_reference or (selected_actor and all(r['pass_gate'] for r in selected_actor))) else 'FAIL_OFFLINE_KNOWN_VALUE_CAPABILITY',
        critic_updates_per_task=updates,minimum_known_margin=minimum_margin,maximum_advantage_mae=threshold,
        minimum_ranking_accuracy=.95,critic_tasks=critic_results,actor_tasks=actor_results,
        configured_time_lr=cfg['time_actor_lr'],critic_batch_size=min(int(cfg['batch_size']),len(t)),legacy_critic_reference=critic_reference,
        source_state_indices=indices,source_states_modified_fresh_flag_only=True,real_tcp_runs=0,
        limits='Small constructed single-decision Q with known same-policy continuation; no TCP causal/value/performance proof. Actor task uses exact Q, not the fitted critics.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],critic_tasks=critic_results,actor_tasks=actor_results)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--states',required=True)
    p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--legacy-critic-reference',action='store_true')
    args=p.parse_args();verify(args.config,args.states,args.output,args.updates,args.legacy_critic_reference)
