"""Exercise the actual learner/replay/gates with a local FIFO and synthetic rows.

No Server, ClusterSpec, network, process controller or TCP flow is created.
Small fixture thresholds are NOT training recommendations or calibration proof.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')


def conditioned_schedule(cfg, actors, rows_per_actor):
    """Every actor visits all five continuous strata; one complete held-out
    short-return block per actor starts in a distinct stratum.

    The gate counts retained rows per ACTOR and per bucket, and the rotated
    held-out block of actor a starts at the first sequence s with
    heldout_position(cfg,a,0,s)==0. Checked here before any TF work, so a
    fixture that cannot satisfy its own gate fails with a reason instead of
    silently stalling in calibration (the 59cf69a fixture: one period per actor
    and 100 rows, so three actors never reached their block).
    """
    from frameworks.joint_healthy.core import heldout_position, period_bucket
    periods = [20. * 8. ** ((k + .5) / 5.) for k in range(5)]  # one per v4 stratum
    if [period_bucket(cfg, p) for p in periods] != list(range(5)):
        raise AssertionError('Fixture periods must occupy the five continuous strata')
    starts = []
    for actor in range(actors):
        found = [s for s in range(1, rows_per_actor - 6) if heldout_position(cfg, actor, 0, s) == 0]
        if not found: raise AssertionError('Actor %d has no complete held-out block within %d rows' % (actor, rows_per_actor))
        starts.append(found[0])
    schedule = [[periods[(i - starts[actor] + actor) % 5] for i in range(1, rows_per_actor + 1)] for actor in range(actors)]
    block_strata = sorted(period_bucket(cfg, schedule[actor][starts[actor] - 1]) for actor in range(actors))
    if block_strata != list(range(5)): raise AssertionError('Short-return blocks do not cover every stratum: %r' % block_strata)
    return periods, schedule, starts


def verify(output,config=None,keep=None):
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy import backend
    from frameworks.joint_healthy.configuration import configuration
    from frameworks.joint_healthy.core import interval_return, bootstrap_terminal
    from frameworks.joint_healthy.model import Model
    from frameworks.joint_healthy.continuous import enabled
    cfg = configuration('fixed20')
    if config:cfg.update(json.loads(Path(config).read_text(encoding='utf-8').replace('@PACKAGE@',str(ROOT).replace('\\','/'))))
    conditioned=enabled(cfg);actors=5 if conditioned else 2;rows_per_actor=300 if conditioned else 200
    fixture_common=1 if conditioned and int(cfg.get('critic_common_calibration_updates',0))>0 else 0
    cfg.update(run_id='synthetic-pipeline', num_actors=actors, memsize=2048 if conditioned else 512, batch_size=8,
        profile=(cfg.get('profile', 'continuous-v4') if conditioned else 'fixed20'), fixed_mtp_ms=None if conditioned else 20.,
        warmup=8, minimum_enqueued_per_actor=4, warmup_physical_seconds=.16,
        calibration_updates=2, forward_calibration_updates=2, calibration_check_updates=1,
        critic_common_calibration_updates=fixture_common,
        calibration_rows_per_actor_period=1, minimum_holdout_per_period=1,
        minimum_short_return_blocks=1,
        checkpoint_updates=100 if conditioned else 10000, release_joint=True, physical_seconds_per_actor=1.,
        deadline_epoch=time.time()+(300 if conditioned else 90))
    if conditioned: periods, schedule, block_starts = conditioned_schedule(cfg, actors, rows_per_actor)
    else: periods, schedule, block_starts = [20.,20.], [[20.]*rows_per_actor for _ in range(actors)], []
    fifo_capacity = actors*(rows_per_actor+1) + 16  # every row is enqueued before the learner starts
    if keep and Path(keep).exists():
        raise ValueError('Use a new --keep directory; existing evidence must be preserved')
    original_r, original_setup = backend.R, backend.setup
    sessions = []
    state = np.tile(np.array([1., 1.2, 0., 1., .02, .98, 1.], np.float32), 10)
    state = np.r_[state, np.zeros(12, np.float32)].astype(np.float32)
    action = np.array([0., -1.], np.float32)
    def local_setup(config, learner):
        assert learner
        model = Model(config, training=True)
        dtypes = [tf.float32]*7+[tf.int32, tf.int64, tf.int64]; dtypes[5] = tf.float64
        shapes = [[82], [2], [1], [82], [1], [1], [1], [], [], []]
        fifo = tf.FIFOQueue(fifo_capacity, dtypes, shapes=shapes)
        ph = [tf.placeholder(t, s) for t, s in zip(dtypes, shapes)]
        enq = fifo.enqueue(ph); number = tf.placeholder(tf.int32, []); deq = fifo.dequeue_many(number); size = fifo.size()
        session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        sessions.append((model,session))
        session.run(tf.global_variables_initializer()); model.initialize_baseline(session, config['baseline_checkpoint'])
        from frameworks.joint_healthy.core import period_action
        actions={ms:np.array([0.,period_action(ms)],np.float32) for ms in periods}
        qs={ms:min(float(v[0,0]) for v in session.run([model.q1, model.q2], {model.s:state[None,:], model.executed:a[None,:]})) for ms,a in actions.items()}
        # A constant-state Bellman-consistent fixture, not a measured reward.
        for i in range(1,rows_per_actor+1):
            for actor in range(actors):
                terminal = i == rows_per_actor; ms = schedule[actor][i-1]
                discount=interval_return(1.,ms/1000.)[1];q=qs[ms]
                absorbing = bool(bootstrap_terminal(cfg, np.array([float(terminal)]))[0])
                row = [state, actions[ms], [q if absorbing else q*(1-discount)], state, [float(terminal)], [ms/1000.], [discount], actor, 0, i]
                session.run(enq, dict(zip(ph,row)))
        for actor in range(actors):
            row = [np.zeros(n) for n in (82,2,1,82,1,1,1)]+[actor,-1,rows_per_actor]
            session.run(enq,dict(zip(ph,row)))
        return tf, model, session, ph, enq, number, deq, size
    temporary = tempfile.mkdtemp(prefix='joint-pipeline-'); passed = False; result = {}
    try:
        backend.setup = local_setup
        with tf.Graph().as_default():
            backend.R = Path(temporary)
            result = backend.run_learner(cfg)
            gates = json.loads((backend.R/'logs'/cfg['run_id']/'calibration-history.json').read_text())
            assert result['status'] == 'COMPLETED', result
            assert result['credit']['inserted'] == actors*rows_per_actor
            assert result['producer_done'] == {a:rows_per_actor for a in range(actors)}
            assert result['update_debt'] == 0
            assert result['counters']['actor_updates'] > 0
            fixture_model,fixture_session=sessions[-1]
            applies=fixture_model.critic_adam_applies_per_step
            count=result['counters']['critic_iterations'];common=min(fixture_model.common_calibration_updates,count)
            expected_ops=2*common+applies*(count-common)
            assert result['counters']['critic_optimizer_ops'] == expected_ops
            optimizers=list(fixture_model.critic_optimizers)+list(getattr(fixture_model,'critic_period_optimizers',[]))
            powers=[float(fixture_session.run(opt._get_beta_accumulators()[1])) for opt in optimizers]
            expected=[.999**(count+1)]*2+[.999**(count-common+1)]*(applies-2)
            assert len(powers)==applies and np.allclose(powers,expected,rtol=2e-5),powers
            result['critic_adam_applications_per_iteration']=applies
            result['fixture_common_only_iterations']=common
            result['all_critic_adam_clocks_match_complete_iterations']=True
            assert [g['phase'] for g in gates if g['admitted']] == [0,1]
            if cfg.get('teacher_anchor_weight',0)>0:
                with np.load(str(Path(result['endpoint']).parent/'replay.npz')) as saved:
                    assert len(saved['teacher_actions'])==actors*rows_per_actor and np.isfinite(saved['teacher_actions']).all()
                    result['verified_cached_teacher_rows']=actors*rows_per_actor
            if conditioned:
                diagnostic=json.loads((backend.R/'logs'/cfg['run_id']/'value-diagnostics-last.json').read_text())
                assert diagnostic['phase']==2 and diagnostic['rows']>0
                assert len(list(Path(result['endpoint']).parent.glob('*.joint.json')))>2
                result['joint_value_monitoring_and_live_checkpoints_verified']=True
            result.pop('endpoint')
            result.update(status='PASS_SYNTHETIC_LEARNER_PIPELINE_ONLY', gates=gates,
                no_server_or_socket_created=True, new_TCP=0, measured_physical_experience=0,
                fixture_exposure_is_synthetic=True, reduced_thresholds_are_fixture_only=True,
                fixture_periods_ms=periods, fixture_rows_per_actor=rows_per_actor, fixture_block_starts=block_starts)
            passed = True
    except Exception:
        # Keep the failure scene: gate reasons, learner failure/result files and checkpoints.
        logs = Path(temporary)/'logs'/cfg['run_id']
        def load(name):
            try: return json.loads((logs/name).read_text())
            except Exception: return None
        retained = Path(keep) if keep else Path(output).with_name(Path(output).stem + '-failure-scene')
        if retained.exists():
            suffix=1
            while retained.with_name(retained.name+'-'+str(suffix)).exists():suffix+=1
            retained=retained.with_name(retained.name+'-'+str(suffix))
        shutil.copytree(temporary, str(retained))
        history = load('calibration-history.json') or []
        result = dict(status='FAIL_SYNTHETIC_PIPELINE_FIXTURE_NOT_TCP', exception=traceback.format_exc(),
            learner_result=load('result.json'), learner_failure=load('failure.json'),
            last_gate_reasons=history[-1]['reasons'] if history else None, gates=history,
            retained_scene=str(retained), fixture_periods_ms=periods, fixture_rows_per_actor=rows_per_actor,
            fixture_block_starts=block_starts, new_TCP=0)
    finally:
        for _,session in sessions:session.close()
        backend.R, backend.setup = original_r, original_setup
        if passed and keep:
            shutil.copytree(temporary, keep)
        shutil.rmtree(temporary, ignore_errors=True)  # failures were already copied above
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True);ap.add_argument('--config')
    ap.add_argument('--keep',help='also retain the learner directory on success (always retained on failure)')
    args=ap.parse_args();outcome=verify(args.output,args.config,args.keep)
    sys.exit(0 if str(outcome.get('status','')).startswith('PASS') else 1)
