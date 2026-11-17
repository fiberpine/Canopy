"""Fixed-budget constructed test: calibrate common Q before time residual.

Uses the actual TF1 Model and existing Adam instances. No runtime algorithm
change, TCP, target-derived counterfactual truth, or extra optimizer iterations.
"""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(config,states_file,output,warmup=1000,updates=10000):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    cfg=json.loads(Path(config).read_text().replace('@PACKAGE@',str(ROOT).replace('\\','/')))
    # This driver owns the schedule. Do not silently combine --warmup with
    # the production Model's independently configured common-value stage.
    cfg['critic_common_calibration_updates']=0
    assert 0<=warmup<updates
    with np.load(str(states_file)) as data:source=data['states']
    indices=[int(np.argmin(source[:,79])),int(np.argmax(source[:,79]))]
    states=source[indices].copy();states[:,66]=1.
    reports=[];threshold=.01*(.995-.995**2)/.005/4.
    for sign in (1.,0.,-1.):
        with tf.Graph().as_default():
            tf.set_random_seed(3);model=Model(cfg,True)
            main_only=[]
            for optimizer,loss,variables in zip(model.critic_optimizers,model.losses,model.critic_parameters):
                main_only.append(optimizer.minimize(loss,var_list=[v for v in variables if '/period_residual/' not in v.op.name]))
            with tf.control_dependencies(main_only):common_step=tf.assign_add(model.global_step,2)
            rng=np.random.RandomState(100)
            t=np.tile(np.exp(rng.uniform(np.log(20.),np.log(160.),256)),2)
            ss=np.repeat(states,256,axis=0)
            action=np.column_stack([np.zeros(512),2*np.log(t/20.)/np.log(8.)-1]).astype(np.float32)
            labels=(np.repeat([168.,169.],256)+sign*.01*(.995-.995**(t/20.))/.005)[:,None].astype(np.float32)
            qs=np.repeat(states,4,axis=0);qt=np.tile([20.,40.,80.,160.],2)
            qa=np.column_stack([np.zeros(8),2*np.log(qt/20.)/np.log(8.)-1]).astype(np.float32)
            truth=(sign*.01*(.995-.995**(qt/20.))/.005).reshape(2,4)
            with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
                sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
                period_vars=[v for variables in model.critic_parameters for v in variables if '/period_residual/' in v.op.name]
                before=sess.run(period_vars);trajectory=[]
                for i in range(updates+1):
                    if i==warmup:
                        assert all(np.array_equal(a,b) for a,b in zip(before,sess.run(period_vars)))
                    if i in (0,warmup,5000,updates) or (i>=updates//2 and i%250==0):
                        qs_value=sess.run([model.q1,model.q2],{model.s:qs,model.executed:qa})
                        errors=[]
                        for q in qs_value:
                            q=q.reshape(2,4);errors.append(float(np.mean(np.abs((q-q[:,:1]-truth)[:,1:]))))
                        trajectory.append(dict(updates=i,advantage_mae=errors))
                    if i==updates:break
                    at=rng.choice(512,min(int(cfg['batch_size']),512),replace=False)
                    sess.run(common_step if i<warmup else model.critic_step,
                             {model.s:ss[at],model.executed:action[at],model.label:labels[at]})
                assert int(sess.run(model.global_step))==2*updates
                main_powers=[float(sess.run(opt._get_beta_accumulators()[1])) for opt in model.critic_optimizers]
                period_powers=[float(sess.run(opt._get_beta_accumulators()[1])) for opt in model.critic_period_optimizers]
                assert np.allclose(main_powers,.999**(updates+1),rtol=3e-4)
                assert np.allclose(period_powers,.999**(updates-warmup+1),rtol=3e-4)
            cells=[]
            for q in qs_value:
                q=q.reshape(2,4);advantage=q-q[:,:1]
                mae=float(np.mean(np.abs((advantage-truth)[:,1:])))
                active=np.abs(truth)>=4*threshold*.999
                ranking=float(np.mean(np.sign(advantage[active])==np.sign(truth[active]))) if sign else None
                cells.append(dict(advantage_mae=mae,ranking=ranking,pass_gate=bool(mae<=threshold and (ranking is None or ranking>=.95))))
            late=np.array([r['advantage_mae'] for r in trajectory if r['updates']>=updates//2])
            reports.append(dict(sign=sign,critics=cells,trajectory=trajectory,
                late_maximum_mae=late.max(axis=0).tolist(),late_fraction_within_gate=(late<=threshold).mean(axis=0).tolist(),
                period_parameters_bitwise_unchanged_during_common_calibration=True,adam_clock_checks=True))
    result=dict(status='PASS_CONSTRUCTED_COMMON_CALIBRATION' if all(c['pass_gate'] for r in reports for c in r['critics']) else 'FAIL_CONSTRUCTED_COMMON_CALIBRATION',
        critic_tasks=reports,total_complete_critic_iterations_per_task=updates,common_only_iterations=warmup,
        period_iterations=updates-warmup,adam_applications_per_task=2*warmup+4*(updates-warmup),
        maximum_advantage_mae=threshold,minimum_ranking=.95,real_tcp_runs=0,runtime_algorithm_changed=False,
        driver_owns_schedule_and_disables_model_common_stage=True,
        limits='Three known-label tasks and a single seed, fixed 10k endpoint. Diagnostic only; no runtime/TD/actor/TCP or generalization validation.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--states',required=True);p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=1000);p.add_argument('--updates',type=int,default=10000)
    a=p.parse_args();outcome=verify(a.config,a.states,a.output,a.warmup,a.updates)
    sys.exit(0 if outcome['status'].startswith('PASS') else 1)
