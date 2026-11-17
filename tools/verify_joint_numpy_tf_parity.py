"""Check the NumPy reproduction against actual TF1 using IDENTICAL weights.

Initial values/gradients and one Adam step are gated. Later steps diagnose
floating-point trajectory drift, not TCP or long-run training equivalence.
"""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools/joint_numpy_reproduction'))


def verify(config,states_file,output):
    import numpy as np
    import tensorflow as tf
    import critic_np as C
    import tfckpt
    from frameworks.joint_healthy.model import Model
    cfg=json.loads(Path(config).read_text().replace('@PACKAGE@',str(ROOT).replace('\\','/')))
    registered_common=int(cfg.get('critic_common_calibration_updates',0))
    cfg['critic_common_calibration_updates']=0  # compare the reproduction's simultaneous-update algorithm
    with np.load(str(states_file)) as data:source=data['states']
    indices=[int(np.argmin(source[:,79])),int(np.argmax(source[:,79]))]
    states=source[indices].copy();states[:,66]=1.
    durations=np.tile(np.exp(np.random.RandomState(100).uniform(np.log(20.),np.log(160.),16)),2)
    batch_states=np.repeat(states,16,axis=0)
    actions=np.column_stack([np.zeros(32),C.encode(durations)]).astype(np.float32)
    truth=.01*(.995-.995**(durations/20.))/.005
    labels=(np.repeat([168.,169.],16)+truth)[:,None].astype(np.float32)
    names={'W1':'fc1/kernel','b1':'fc1/bias','W2':'fc2/kernel','b2':'fc2/bias',
           'W3':'dense/kernel','b3':'dense/bias','W4':'period_residual/state/kernel',
           'b4':'period_residual/state/bias','W5':'period_residual/joint/kernel',
           'b5':'period_residual/joint/bias','W6':'period_residual/output/kernel','b6':'period_residual/output/bias'}
    ckpt=tfckpt.read(cfg['baseline_checkpoint']);checks=[];trajectory=[]
    with tf.Graph().as_default():
        tf.set_random_seed(3);model=Model(cfg,True)
        variable_groups=[];gradient_groups=[]
        for index,scope in enumerate(('critic','critic2')):
            by_name={v.op.name:v for v in model._variables(scope,True)}
            variables=[by_name[scope+'/'+name] for name in names.values()]
            variable_groups.append(variables);gradient_groups.append(tf.gradients(model.losses[index],variables))
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
            numpy_models=[]
            feed={model.s:batch_states,model.executed:actions,model.label:labels}
            initial_reference=sess.run([model.q1,model.q2],{model.s:states,model.executed:np.array([[0.,-1.],[0.,-1.]],np.float32)})
            for i,scope in enumerate(('critic','critic2')):
                clone=C.Critic(ckpt,scope,np.random.RandomState(0))
                for key,value in zip(names,sess.run(variable_groups[i])):clone.p[key]=value.astype(np.float64)
                numpy_models.append(clone)
                q,_,_=clone.forward(batch_states.astype(np.float64),actions.astype(np.float64))
                grad=clone.backward(2.*(q-labels.astype(np.float64))/len(q))
                actual_q,actual_grads=sess.run([(model.q1,model.q2)[i],gradient_groups[i]],feed)
                q_error=float(np.max(np.abs(q-actual_q)))
                differences=[float(np.max(np.abs(grad[key]-g))) for key,g in zip(names,actual_grads)]
                norm=max(1.,max(float(np.max(np.abs(g))) for g in actual_grads))
                relative=max(differences)/norm
                assert q_error<2e-4 and relative<1e-4,(scope,q_error,relative)
                checks.append(dict(critic=scope,initial_q_max_error=q_error,gradient_max_scaled_error=relative))
            for step in range(101):
                if step in (0,1,10,100):
                    tf_values=sess.run([model.q1,model.q2],feed)
                    rows=[]
                    for i,clone in enumerate(numpy_models):
                        np_q,_,_=clone.forward(batch_states.astype(np.float64),actions.astype(np.float64))
                        parameter_error=max(float(np.max(np.abs(clone.p[key]-v))) for key,v in zip(names,sess.run(variable_groups[i])))
                        if step==1:assert parameter_error<2e-5,parameter_error
                        rows.append(dict(q_max_error=float(np.max(np.abs(np_q-tf_values[i]))),parameter_max_error=parameter_error))
                    trajectory.append(dict(updates=step,critics=rows))
                if step==100:break
                for clone in numpy_models:
                    clone.step(batch_states.astype(np.float64),actions.astype(np.float64),labels.astype(np.float64),cfg['lr_c'],cfg.get('critic_period_residual_lr'))
                sess.run(model.critic_step,feed)
    result=dict(status='PASS_TF1_NUMPY_INITIAL_AND_SINGLE_STEP_PARITY',source_state_indices=indices,critics=checks,
        identical_tf_initial_weights_copied_into_numpy=True,numpy_dtype='float64',tensorflow_dtype='float32',
        original_initial_q_levels=[q[:,0].tolist() for q in initial_reference],
        tf_isolated_task_labels_use_twin_minimum=True,numpy_isolated_task_labels_use_each_critic_own_value=True,
        production_common_calibration_updates=registered_common,common_calibration_disabled_for_simultaneous_update_parity=True,
        later_trajectory_diagnostic=trajectory,real_tcp_runs=0,
        limits='Initial/gradient/single-step checks only. Different precision and initialization do not imply matched 10k endpoints; the two isolated-level task definitions differ.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--states',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.config,a.states,a.output)
