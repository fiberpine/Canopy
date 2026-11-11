"""Join retained replay rows with actor-side connection origins, near the data."""

import argparse
from collections import defaultdict
import json
import math
import pathlib

import numpy as np


def audit(replay_path, origin_paths):
    with np.load(str(replay_path), allow_pickle=False) as replay:
        actors = replay["actor_ids"].copy()
        sequences = replay["actor_sequences"].copy()
        inserted = replay["actor_inserted"].copy()
        rows = len(replay["s0"])
        if actors.shape != (rows,) or sequences.shape != (rows,):
            raise ValueError("Replay origin metadata does not match its real rows")
    needed = {(int(actor), int(sequence)) for actor, sequence in zip(actors, sequences)}
    if len(needed) != rows or any(actor < 0 or sequence <= 0 for actor, sequence in needed):
        raise ValueError("Duplicate or uninitialized replay origins")
    if any(actor >= len(inserted) or sequence > inserted[actor] for actor, sequence in needed):
        raise ValueError("Replay origin is ahead of confirmed learner insertion")
    found = set()
    by_actor = defaultdict(list)
    for path in origin_paths:
        previous = {}
        with pathlib.Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                actor, sequence = int(row["actor"]), int(row["actor_sequence"])
                if actor in previous and sequence != previous[actor] + 1:
                    raise ValueError("Actor origin sequence is not continuous")
                previous[actor] = sequence
                key = actor, sequence
                if key not in needed:
                    continue
                if key in found:
                    raise ValueError("Duplicate origin file or record")
                if (not math.isfinite(row["duration"]) or row["duration"] <= 0 or
                        not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0 or
                        row["connection"] < 0 or row["connection_step"] <= 0):
                    raise ValueError("Invalid connection origin")
                found.add(key)
                by_actor[actor].append(row)
    if found != needed:
        raise ValueError("Missing origins for %d retained rows" % len(needed - found))
    summary = {}
    for actor, retained in sorted(by_actor.items()):
        startup = [row for row in retained if row["phase"] == "startup"]
        summary[str(actor)] = dict(
            retained_rows=len(retained), startup_rows=len(startup),
            retained_physical_seconds=sum(row["duration"] for row in retained),
            startup_physical_seconds=sum(row["duration"] for row in startup),
            connections=len({row["connection"] for row in retained}),
            oldest_actor_sequence=min(row["actor_sequence"] for row in retained),
            newest_actor_sequence=max(row["actor_sequence"] for row in retained))
    return dict(status="PASS_ORIGIN_JOIN", retained_rows=rows, actors=summary,
                meaning="Retained experience exposure, not uninterrupted wall time or proof of recovery coverage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=pathlib.Path, required=True)
    parser.add_argument("--origin", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay, args.origin)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
