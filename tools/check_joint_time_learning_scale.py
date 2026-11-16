"""Constructed exact-value capability task, not a TCP performance evaluation."""
import argparse,json,os,sys
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def verify(config,states_file,output):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    cfg=json.loads(Path(config).read_text(encoding='utf-8').replace('@PACKAGE@',str(ROOT).replace('\\','/')))
    with np.load(str(states_file)) as p:states=p['states'][:64]
    results=[]
    for lr in (.0001,.001):
        with tf.Graph().as_default():
            tf.set_random_seed(3);model=Model(cfg,False)
            dt=.02*tf.pow(8.,(model.action[:,1]+1.)/2.)
            discount=tf.pow(.995,dt/.02)
            exact_q=200.-.01/(1-discount)
            step=tf.train.AdamOptimizer(lr).minimize(-tf.reduce_mean(exact_q),var_list=model._variables('mtp_actor',True))
            trajectory=[]
            with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
                sess.run(tf.global_variables_initializer());model.initialize_baseline(sess,cfg['baseline_checkpoint'])
                for count in range(5001):
                    if count in (0,500,2000,5000):
                        values=sess.run(dt,{model.s:states})*1000
                        trajectory.append(dict(updates=count,mean_ms=float(np.mean(values)),minimum_ms=float(np.min(values)),maximum_ms=float(np.max(values))))
                    if count<5000:sess.run(step,{model.s:states})
            results.append(dict(time_lr=lr,trajectory=trajectory))
    result=dict(status='OFFLINE_CONSTRUCTED_EXACT_VALUE_CAPABILITY_TASK',results=results,actual_tcp_runs=0,cloud_training=0,
                limits='Constant utility1, repeated identical period,c=.01. Not TCP values, performance or proof of a suitable cost coefficient.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--states',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();print(json.dumps(verify(args.config,args.states,args.output)))
