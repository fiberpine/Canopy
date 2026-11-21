"""Exercise the scalar actor entry point with fake TCP and a fake TF session."""

import ast
import hashlib
import io
import json
import pathlib
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "frameworks/baseline"
sys.path.insert(0, str(FRAMEWORK))
from episode_training import run_episodes
from audit_episode_replay import audit


class ActorExit(BaseException):
    pass


def source_definition(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node, = [node for node in ast.walk(tree) if getattr(node, "name", None) == name]
    tree.body = [node]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


class Noise:
    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return np.array([self.calls * .001])


class ScalarActorEpisodes(unittest.TestCase):
    def run_actor(self, episode_seconds=20, stop_after_first=False, short=False, exit_code=0, replay_probe=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        (root / "configs").mkdir()
        (root / "logs/run").mkdir(parents=True)
        (root / "logs/run/ready").write_text("ready")
        (root / "logs/run/finished").write_text("COMPLETED")
        cfg = dict(run_id="run", seed=3, conditions=[dict(condition="one")],
                   steps_per_actor=20, stop_file=str(root / "STOP"),
                   absolute_deadline_epoch=time.time() + 300)
        if episode_seconds is not None:
            cfg["episode_seconds"] = episode_seconds
        if replay_probe:
            cfg['capacity_replay_probe'] = True
        sent = []
        noise = Noise()
        agent = types.SimpleNamespace(
            actor_noise=noise,
            get_concrete_action=lambda history, explore: [np.array([[.1]])])

        class Session:
            def run(self, operation, feed=None):
                if operation == "enqueue":
                    sent.append([feed[str(i)] for i in range(6)])
                    return None
                return []

        session = Session()
        evaluator = types.ModuleType("evaluate")
        evaluator.model = lambda cfg: None
        invocations = []

        def evaluate():
            ecfg = json.loads(pathlib.Path(sys.argv[-1]).read_text())
            invocations.append(ecfg)
            directory = root / "logs" / ecfg["run_id"]
            directory.mkdir()
            _, live, wrapper, params, action, digest, variables = evaluator.model(ecfg)
            self.assertIs(live, session)
            history = np.full(70, len(invocations) * 100, dtype=np.float32)
            count = 5 if episode_seconds is not None else 20
            count = min(count, ecfg["training_steps"])
            current_action = action(history)
            for step in range(1, count + 1):
                successor = history + 1
                next_action = action(successor)
                if hasattr(evaluator, "transition_context_hook"):
                    evaluator.transition_context_hook(dict(duration=.02, elapsed_s=step * .02,
                                                           state_id=step, connection_step=step))
                evaluator.transition_hook(history, current_action, .5, successor)
                history = successor
                current_action = next_action
            result = dict(status="COMPLETED", reason="", wall_s=1 if short else 20,
                          server_returncode=exit_code)
            (directory / "result.json").write_text(json.dumps(result))
            if stop_after_first:
                (root / "STOP").write_text("stop")
            if evaluator.producer_finish:
                evaluator.producer_finish()

        evaluator.main = evaluate
        evaluator.main = source_definition(FRAMEWORK / "timed_distributed.py", "main",
            dict(original_main=evaluate, f=io.StringIO(), cfg=cfg))
        params = root / "code/canopy/params_base_eval.json"
        params.parent.mkdir(parents=True)
        params.write_text("{}")
        wrapper = types.ModuleType("envwrapper_v2")
        wrapper.TCP_Env_Wrapper = object
        node = types.ModuleType("node_control")

        def dump(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        node.dump = dump
        namespace = dict(
            R=root, np=np, tf=types.SimpleNamespace(global_variables=lambda: []),
            setup=lambda cfg, role, index: (
                agent, session, None, [str(i) for i in range(6)], "enqueue", None, None, None),
            hashlib=hashlib, queue=queue, threading=threading, time=time,
            json=json, pathlib=pathlib, sys=sys, random=random,
            os=types.SimpleNamespace(_exit=lambda code: (_ for _ in ()).throw(ActorExit(code))))
        actor = source_definition(FRAMEWORK / "distributed.py", "actor", namespace)
        arguments = sys.argv[:]
        with mock.patch.dict(sys.modules, evaluate=evaluator, envwrapper_v2=wrapper, node_control=node):
            try:
                if short or exit_code:
                    with self.assertRaisesRegex(RuntimeError, "ended before|Native process failed"):
                        actor(cfg, 0)
                else:
                    with self.assertRaises(ActorExit) as exited:
                        actor(cfg, 0)
                    self.assertEqual(exited.exception.args[0], 0)
            finally:
                sys.argv = arguments
        return root, sent, noise, invocations

    def test_normal_actor_reconnects_without_cross_connection_rows_or_rng_reset(self):
        root, sent, noise, invocations = self.run_actor()
        self.assertEqual(len(invocations), 4)
        self.assertEqual(len({cfg['port'] for cfg in invocations}), 4)
        self.assertEqual(len(sent), 20)
        self.assertEqual(noise.calls, 24)
        self.assertTrue(all(np.array_equal(row[3], row[0] + 1) for row in sent))
        self.assertTrue(all(float(row[4][0]) == 0 for row in sent))
        self.assertEqual([int(row[0][0]) for row in sent[::5]], [100, 200, 300, 400])
        parent = root / "logs/run-actor0"
        origins = [json.loads(line) for line in (parent / "transition-origins.jsonl").read_text().splitlines()]
        self.assertEqual([row["actor_sequence"] for row in origins], list(range(1, 21)))
        self.assertEqual([row["connection_step"] for row in origins], [1, 2, 3, 4, 5] * 4)
        self.assertEqual([row["connection"] for row in origins[::5]], [0, 1, 2, 3])
        producer = json.loads((parent / "producer.json").read_text())
        self.assertEqual(producer["generated"], producer["enqueued"])
        self.assertEqual(producer["enqueued"], 20)
        self.assertEqual(producer["errors"], [])
        replay = root / "retained.npz"
        np.savez(str(replay), s0=np.zeros((4, 70)), actor_ids=np.zeros(4, np.int32),
                 actor_sequences=np.array([17, 18, 19, 20]), actor_inserted=np.array([20]))
        coverage = audit(replay, [parent / "transition-origins.jsonl"])
        self.assertEqual(coverage["actors"]["0"]["startup_rows"], 4)
        self.assertAlmostEqual(coverage["actors"]["0"]["retained_physical_seconds"], .08)
        with self.assertRaisesRegex(ValueError, "Duplicate origin"):
            audit(replay, [parent / "transition-origins.jsonl"] * 2)

    def test_existing_single_connection_entry_remains_available(self):
        root, sent, noise, invocations = self.run_actor(episode_seconds=None)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(len(sent), 20)
        self.assertEqual(noise.calls, 21)
        self.assertFalse((root / "logs/run-actor0/episodes.json").exists())

    def test_single_connection_probe_records_true_origins_without_reset(self):
        root, sent, noise, invocations = self.run_actor(episode_seconds=None, replay_probe=True)
        self.assertEqual(len(invocations), 1)
        rows = [json.loads(line) for line in (root/'logs/run-actor0/transition-origins.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), len(sent))
        self.assertEqual([row['actor_sequence'] for row in rows], list(range(1,21)))
        self.assertTrue(all(row['connection']==0 and row['generation_epoch']>0 for row in rows))

    def test_stop_between_episodes_does_not_create_a_phantom_connection(self):
        root, sent, noise, invocations = self.run_actor(stop_after_first=True)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(len(sent), 5)
        episodes = json.loads((root / "logs/run-actor0/episodes.json").read_text())
        self.assertEqual(episodes[0]["reason"], "coordinator_stop")

    def test_unexpected_early_native_end_is_a_failure(self):
        root, sent, noise, invocations = self.run_actor(short=True)
        result = json.loads((root / "logs/run-actor0/result.json").read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(invocations), 1)

    def test_native_failure_after_full_duration_is_not_success(self):
        root, sent, noise, invocations = self.run_actor(exit_code=1)
        result = json.loads((root / "logs/run-actor0/result.json").read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(invocations), 1)

    def test_replay_ring_keeps_per_actor_sequences_through_overwrite(self):
        namespace = dict(np=np, threading=threading, random=random)
        namespace["ReplayBuffer"] = source_definition(
            FRAMEWORK / "code/canopy/rl-module/utils_v2.py", "ReplayBuffer", namespace)
        replay_type = source_definition(FRAMEWORK / "distributed.py", "Replay", namespace)
        replay = replay_type(dict(memsize=4, num_actors=2, episode_seconds=20, sampler="iid"))
        actors = [0, 1, 0, 1, 0, 1]
        data = [np.zeros((6, width), np.float32) for width in (70, 1, 1, 70, 1)]
        replay.append_many(data + [np.array(actors)])
        self.assertEqual(replay.inserted, 6)
        self.assertEqual(replay.actor_inserted.tolist(), [3, 3])
        self.assertEqual(replay.actor_sequences.tolist(), [3, 3, 2, 2])
        self.assertEqual(replay.birth.tolist(), [4, 5, 2, 3])
        snapshot = replay.snapshot()
        replay.append_many([array[:1] for array in data] + [np.array([0])])
        self.assertEqual(snapshot["actor_sequences"].tolist(), [3, 3, 2, 2])
        self.assertEqual(snapshot["actor_inserted"].tolist(), [3, 3])
        self.assertEqual(len(snapshot["s0"]), 4)

    def test_connection_summary_does_not_measure_reconnection_gaps_as_control_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            actor = root / "logs/run-actor0"
            actor.mkdir(parents=True)
            (root / "logs/run-node-ownership.json").write_text(json.dumps([
                dict(role="actor", index=0)]))
            episodes = []
            for connection, offset in enumerate((1000, 1000000)):
                name = "run-actor0-episode%04d" % connection
                folder = root / "logs" / name
                folder.mkdir()
                episodes.append(dict(active_run=name))
                (folder / "states.csv").write_text(
                    "reward,cwnd,action_no_noise\n0.5,10,0.1\n")
                control = "time_us,state_id,action_id,percent,cwnd_before,cwnd_requested,return_code,socket\n"
                control += "%d,1,1,100,10,10,0,4\n%d,2,2,100,10,10,0,4\n" % (offset, offset + 1000)
                (folder / "control.csv").write_text(control)
            (actor / "episodes.json").write_text(json.dumps(episodes))
            namespace = dict(R=root, json=json, pathlib=pathlib)
            for name in ("load", "dump", "own_path"):
                source_definition(FRAMEWORK / "node_control.py", name, namespace)
            namespace["os"] = types.SimpleNamespace(getpid=lambda: 123)
            summarize = source_definition(FRAMEWORK / "node_control.py", "summarize", namespace)
            result, = summarize("run")
            self.assertEqual(result["mapping_errors"], 0)
            self.assertEqual(result["connections_completed"], 2)
            self.assertEqual(result["control"]["mean"], 1.)

    def test_privileged_cleanup_requires_a_verified_creation_time(self):
        class ProcessError(Exception):
            pass

        class AccessDenied(ProcessError):
            pass

        class Process:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive):
                return []

            def create_time(self):
                if self.pid == 102:
                    raise AccessDenied()
                return 2. if self.pid == 101 else 1.

            def send_signal(self, sig):
                raise AccessDenied()

        with mock.patch.object(subprocess, 'run') as elevated:
            namespace=dict(psutil=types.SimpleNamespace(Process=Process, Error=ProcessError,
                          AccessDenied=AccessDenied), subprocess=subprocess,
                          signal=types.SimpleNamespace(SIGTERM=15,SIGKILL=9),
                          time=types.SimpleNamespace(sleep=lambda seconds: None))
            cleanup=source_definition(FRAMEWORK/'evaluate.py','cleanup',namespace)
            server=types.SimpleNamespace(pid=100,created=1.,wait=lambda timeout: None)
            cleanup(server,{101:1.,102:1.})
            self.assertEqual(elevated.call_count,2)
            self.assertEqual({call.args[0][-1] for call in elevated.call_args_list},{'100'})


if __name__ == "__main__":
    unittest.main()
