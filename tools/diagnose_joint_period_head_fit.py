"""Oracle linear-head capacity check on constructed labels, never TCP training.

Freeze every feature/backbone at the actual TF initialization. Fit only the
residual's last linear layer using the SAME known synthetic labels. This can
distinguish representation from optimization in this small task; it cannot
certify TD learning, generalization or real TCP counterfactual values.
"""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(config,states_file,output):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    cfg=json.loads(Path(config).read_text().replace('@PACKAGE@',str(ROOT).replace('\\','/')))
    with np.load(str(states_file)) as data:source=data['states']
    indices=[int(np.argmin(source[:,79])),int(np.argmax(source[:,79]))]
    states=source[indices].copy();states[:,66]=1.
    durations=np.tile(np.exp(np.random.RandomState(100).uniform(np.log(20.),np.log(160.),256)),2)
    ss=np.repeat(states,256,axis=0)
    actions=np.column_stack([np.zeros(512),2*np.log(durations/20.)/np.log(8.)-1]).astype(np.float32)
    qs=np.repeat(states,4,axis=0);qt=np.tile([20.,40.,80.,160.],2)
    qa=np.column_stack([np.zeros(8),2*np.log(qt/20.)/np.log(8.)-1]).astype(np.float32)
    reports=[];limit=.01*(.995-.995**2)/.005/4.
    with tf.Graph().as_default():
        tf.set_random_seed(3);model=Model(cfg,True)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
            for scope,q in (('critic',model.q1),('critic2',model.q2)):
                variables={v.op.name:v for v in model._variables(scope,True)}
                w=variables[scope+'/period_residual/output/kernel'];b=variables[scope+'/period_residual/output/bias']
                # First forward consumer belongs to the executed-action Q;
                # the policy-loss call is constructed later using shared vars.
                products=[op for op in tf.get_default_graph().get_operations()
                          if op.type=='MatMul' and op.inputs[1].name==w.value().name]
                assert products
                features=products[0].inputs[0]
                assert tf.gradients(tf.reduce_sum(features),model.executed)[0] is not None
                x,base=sess.run([features,q],{model.s:ss,model.executed:actions})
                design=np.column_stack([x,np.ones(len(x))]).astype(np.float64)
                scale=np.maximum(np.linalg.norm(design,axis=0),1.)
                for sign in (1.,0.,-1.):
                    labels=(np.repeat([168.,169.],256)+sign*.01*(.995-.995**(durations/20.))/.005).astype(np.float32)[:,None]
                    weights,_,rank,singular=np.linalg.lstsq(design/scale,(labels-base).astype(np.float64),rcond=1e-7)
                    weights=weights/scale[:,None]
                    sess.run([tf.assign(w,weights[:-1].astype(np.float32)),tf.assign(b,weights[-1].astype(np.float32))])
                    prediction=sess.run(q,{model.s:qs,model.executed:qa}).reshape(2,4)
                    advantage=prediction-prediction[:,:1]
                    truth=(sign*.01*(.995-.995**(qt/20.))/.005).reshape(2,4)
                    mae=float(np.mean(np.abs((advantage-truth)[:,1:])))
                    active=np.abs(truth)>=4*limit*.999
                    ranking=float(np.mean(np.sign(advantage[active])==np.sign(truth[active]))) if sign else None
                    reports.append(dict(critic=scope,sign=sign,advantage_mae=mae,ranking=ranking,
                        pass_gate=bool(mae<=limit and (ranking is None or ranking>=.95)),effective_rank=int(rank),
                        maximum_absolute_fitted_weight=float(np.max(np.abs(weights)))))
                sess.run([tf.assign(w,tf.zeros_like(w)),tf.assign(b,tf.zeros_like(b))])
    result=dict(status='PASS_CONSTRUCTED_ORACLE_HEAD_CAPACITY_ONLY' if all(x['pass_gate'] for x in reports) else 'FAIL_CONSTRUCTED_ORACLE_HEAD_CAPACITY',
        cells=reports,maximum_advantage_mae=limit,model_features_and_backbone_frozen=True,
        predictions_evaluated_in_actual_float32_tensorflow=True,oracle_labels_used_for_last_linear_layer=True,
        updates_to_runtime_algorithm=0,real_tcp_runs=0,not_a_replacement_for_registered_adam_gate=True,
        limits='Two constructed states, known labels, frozen initialization features. Capacity evidence for this task only, no TD/actor/TCP learning or generalization claim.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--states',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();outcome=verify(a.config,a.states,a.output)
    sys.exit(0 if outcome['status'].startswith('PASS') else 1)
