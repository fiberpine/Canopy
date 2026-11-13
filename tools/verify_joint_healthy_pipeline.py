"""Exercise the actual learner/replay/gates with a local FIFO and synthetic rows.

No Server, ClusterSpec, network, process controller or TCP flow is created.
Small fixture thresholds are NOT training recommendations or calibration proof.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')


def verify(output):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy import backend
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.core import interval_return
    from frameworks.joint_healthy.model import Model
    cfg = configuration('fixed20')
    cfg.update(run_id='synthetic-pipeline', num_actors=2, memsize=512, batch_size=8,
        warmup=8, minimum_enqueued_per_actor=4, warmup_physical_seconds=.16,
        calibration_updates=2, forward_calibration_updates=2, calibration_check_updates=1,
        calibration_rows_per_actor_period=1, minimum_holdout_per_period=1,
        minimum_short_return_blocks=1,
        checkpoint_updates=10000, release_joint=True, physical_seconds_per_actor=1., deadline_epoch=time.time()+90)
    original_r, original_setup = backend.R, backend.setup
    state = np.tile(np.array([1., 1.2, 0., 1., .02, .98, 1.], np.float32), 10)
    state = np.r_[state, np.zeros(12, np.float32)].astype(np.float32)
    action = np.array([0., -1.], np.float32)
    def local_setup(config, learner):
        assert learner
        model = Model(config, training=True)
        dtypes = [tf.float32]*7+[tf.int32, tf.int64, tf.int64]; dtypes[5] = tf.float64
        shapes = [[82], [2], [1], [82], [1], [1], [1], [], [], []]
        fifo = tf.FIFOQueue(512, dtypes, shapes=shapes)
        ph = [tf.placeholder(t, s) for t, s in zip(dtypes, shapes)]
        enq = fifo.enqueue(ph); number = tf.placeholder(tf.int32, []); deq = fifo.dequeue_many(number); size = fifo.size()
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        session.run(tf.global_variables_initializer()); model.initialize_baseline(session, config['baseline_checkpoint'])
        q = min(float(v[0,0]) for v in session.run([model.q1, model.q2], {model.s:state[None,:], model.executed:action[None,:]}))
        # A constant-state Bellman-consistent fixture, not a measured reward.
        discount = interval_return(1., .02)[1]
        for i in range(1,201):
            for actor in range(2):
                terminal = i == 200
                row = [state, action, [q if terminal else q*(1-discount)], state, [float(terminal)], [.02], [discount], actor, 0, i]
                session.run(enq, dict(zip(ph,row)))
        for actor in range(2):
            row = [np.zeros(n) for n in (82,2,1,82,1,1,1)]+[actor,-1,200]
            session.run(enq,dict(zip(ph,row)))
        return tf, model, session, ph, enq, number, deq, size
    try:
        backend.setup = local_setup
        with tempfile.TemporaryDirectory(prefix='joint-pipeline-') as temporary, tf.Graph().as_default():
            backend.R = Path(temporary)
            result = backend.run_learner(cfg)
            gates = json.loads((backend.R/'logs'/cfg['run_id']/'calibration-history.json').read_text())
            assert result['status'] == 'COMPLETED', result
            assert result['credit']['inserted'] == 400
            assert result['producer_done'] == {0:200,1:200}
            assert result['update_debt'] == 0
            assert result['counters']['actor_updates'] > 0
            assert result['counters']['critic_optimizer_ops'] == 2*result['counters']['critic_iterations']
            assert [g['phase'] for g in gates if g['admitted']] == [0,1]
            result.pop('endpoint')
            result.update(status='PASS_SYNTHETIC_LEARNER_PIPELINE_ONLY', gates=gates,
                no_server_or_socket_created=True, new_TCP=0, measured_physical_experience=0,
                fixture_exposure_is_synthetic=True, reduced_thresholds_are_fixture_only=True)
    finally:
        backend.R, backend.setup = original_r, original_setup
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True)
    verify(ap.parse_args().output)
