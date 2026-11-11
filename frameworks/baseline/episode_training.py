"""Bounded reconnection using the existing scalar evaluator and live TF session."""

import json
import math
import pathlib
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run_episodes(root, cfg, run_id, counts, evaluation_cfg, evaluator):
    """Every evaluator invocation creates fresh native/TCP/history/max_bw state.

    The caller keeps the model, exploration generator, producer queue and its
    writer alive. Only completed real transitions reach that queue. The final
    observation of a time-limited episode remains its own successor state.
    """
    seconds = float(cfg["episode_seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("episode_seconds must be finite and positive")
    maximum_steps = int(cfg["steps_per_actor"])
    if maximum_steps <= 0:
        raise ValueError("steps_per_actor must be positive")
    stop_file = pathlib.Path(cfg["stop_file"])
    parent = root / "logs" / run_id
    episodes = []
    deadline = float(cfg["absolute_deadline_epoch"])
    status = "FAILED"
    reason = ""
    arguments = sys.argv[:]
    # The producer finish hook belongs to the whole actor, not to an episode.
    evaluator.producer_finish = None
    try:
        while counts["generated"] < maximum_steps:
            if stop_file.exists() or time.time() >= deadline:
                break
            connection = len(episodes)
            name = run_id + "-episode%04d" % connection
            port = evaluation_cfg["port"] + connection * int(cfg.get("num_actors", 1))
            if not 1 <= port <= 65535:
                raise ValueError("Episode port allocation exceeds the TCP port range")
            current = dict(evaluation_cfg, run_id=name, seconds=seconds,
                           port=port, training_steps=maximum_steps - counts["generated"])
            path = root / "configs" / (name + ".json")
            with path.open("x") as stream:
                json.dump(current, stream, indent=2)
            progress = dict(connection=connection, active_run=name,
                            first_actor_sequence=counts["generated"] + 1)
            write_json(parent / "episode-progress.json", progress)
            evaluator.training_connection = connection
            before = counts["generated"]
            sys.argv = ["evaluate.py", "--config", str(path)]
            evaluator.main()
            result = json.loads((root / "logs" / name / "result.json").read_text())
            if result["status"] != "COMPLETED":
                raise RuntimeError("Episode failed: " + name)
            generated = counts["generated"] - before
            if generated <= 0:
                raise RuntimeError("Episode produced no real transitions: " + name)
            reason = "time_limit"
            if stop_file.exists():
                reason = "coordinator_stop"
            elif counts["generated"] >= maximum_steps:
                reason = "actor_step_limit"
            elif time.time() >= deadline:
                reason = "absolute_deadline"
            elif result["wall_s"] < seconds - .5:
                raise RuntimeError("Episode ended before its registered limit: " + name)
            if reason == "time_limit" and result.get("server_returncode", 0) != 0:
                raise RuntimeError("Native process failed at episode end: " + name)
            progress.update(last_actor_sequence=counts["generated"], generated=generated,
                            end_kind="truncated", reason=reason, wall_s=result["wall_s"])
            episodes.append(progress)
            write_json(parent / "episodes.json", episodes)
        status = "COMPLETED"
        reason = ""
    except BaseException as error:
        status = "FAILED"
        reason = repr(error)
        raise
    finally:
        sys.argv = arguments
        write_json(parent / "result.json", dict(
            status=status, reason=reason,
            cfg=evaluation_cfg, episodes=episodes,
            counters=dict(valid=counts["generated"]),
            meaning="Actor episode summary; each connection has a separate result and packet log."))


def episode_logs(root, actor_directory):
    """Keep per-connection clocks separate when measuring control intervals."""
    progress = actor_directory / "episodes.json"
    if not progress.exists():
        return [actor_directory]
    episodes = json.loads(progress.read_text())
    return [root / "logs" / episode["active_run"] for episode in episodes] or [actor_directory]
