"""Offline test of S20 calibration and the existing TF1 optimizer on V31 weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frameworks/mtpjoint'))
from common_period import PERIODS, checksum, encoded_period
from common_period_calibration import calibrate
from common_period_runtime import Runtime
from common_period_tf import Model
from common_period_transport import Policy, snapshot_bytes


def audit(source, original, reward):
    spec = importlib.util.spec_from_file_location('original_A_tf', original)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    prior = {str(t): 1/(1-.995**(t/20.)) for t in PERIODS}
    a = legacy.Model(source, 0., prior, reward_config=reward)
    s20 = Model(source, 0., prior, reward_config=reward)
    policy = Policy(source, checksum(reward))
    runtime = Runtime(s20, reward, active_periods=(20,))
    rng = np.random.RandomState(20261109)
    states = rng.normal(0., .15, (64, 149)).astype(np.float32)
    action = np.asarray([[0., encoded_period(20)]], np.float32)

    def windows(model):
        return np.asarray([model.session.run(model.mu, {model.s: s[None, :], model.action: action})[0,0]
                           for s in states])

    try:
        original_windows = windows(a)
        np.testing.assert_array_equal(windows(s20), original_windows)
        snapshot = runtime.publish(window_only=True)
        policy.load(snapshot_bytes(snapshot))
        np.testing.assert_array_equal(
            np.asarray([policy.window(s, 20) for s in states], np.float32), original_windows)
        for actor in range(24):
            for block in range(5):
                connection = 'cal-%d-%d' % (actor, block)
                for step in range(40):
                    assert runtime.boundary(
                        actor, connection, connection, states[step], action[0], 1., states[step+1],
                        1., 20, snapshot['version'], calibration=True)
                    runtime.calibration[-1]['block'] = block
        assert runtime.replay.inserted == runtime.credit.limit(runtime.replay) == 0
        cal = calibrate(s20, runtime.calibration, (20,))
        assert cal['fit']['points'] == cal['independent_check']['points'] == 1200
        assert all(s20.metadata['n0'][str(t)] == prior[str(t)] for t in PERIODS[1:])
        np.testing.assert_array_equal(windows(s20), original_windows)
        # Exercise the real graph after calibration. These tiny synthetic batches
        # deliberately bypass the online credit scheduler and cannot earn credit.
        batch = dict(s0=states, s1=np.roll(states, -1, axis=0),
                     action=np.repeat(action, len(states), axis=0),
                     reward=np.ones((len(states), 1), np.float32),
                     discount=np.full((len(states), 1), .995, np.float32),
                     bootstrap=np.ones((len(states), 1), np.float32))
        noise = np.zeros((len(states), 1), np.float32)
        s20.train_batch(batch, noise, False)
        np.testing.assert_array_equal(windows(s20), original_windows)
        s20.train_batch(batch, noise, True)
        assert np.any(windows(s20) != original_windows)
        return dict(status='PASS', tf=s20.tf.__version__, initial_actor_queries_bitwise=len(states),
                    calibration_blocks=120, observed_periods_ms=[20], calibration_credit=0,
                    calibrate_then_critic_only_then_actor=True, synthetic_optimizer_steps=2,
                    synthetic_data=True, cloud_TCP_runs=0)
    finally:
        for model in (a, s20, policy):
            model.session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--reward', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.original, json.loads(args.reward.read_text())), indent=2))
