"""Read-only scan of original local native control logs; output is small JSON."""
import csv
import hashlib
import json
import os
import pathlib
import time
from collections import deque

OLD = pathlib.Path(os.environ['CANOPY_E2_HISTORY_ROOT']).resolve()
OUT = pathlib.Path(os.environ['CANOPY_E2_WORKSPACE']).resolve()
FIRST = {'orca_control': 1792933298.2028503, 'orca_moving': 1792938285.2141287}


def scan(path, arm):
    before = path.stat()
    recent = deque(maxlen=12)
    events = {}
    n = 0
    max_before = max_request = positive = bad_id = failed_write = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            n += 1
            cwnd = int(row['cwnd_before'])
            requested = int(row['cwnd_requested'])
            pct = int(row['percent'])
            epoch = float(row['time_us']) / 1e6
            max_before = max(max_before, cwnd)
            max_request = max(max_request, requested)
            positive += pct > 100
            bad_id += row['state_id'] != row['action_id']
            failed_write += int(row['return_code']) != 0
            for threshold in (10000, 1000000, 100000000):
                if max(cwnd, requested) >= threshold and str(threshold) not in events:
                    events[str(threshold)] = dict(
                        line=n, epoch=epoch, seconds_from_first_learner_update=epoch-FIRST[arm],
                        first_update_not_actor_weight_delivery_time=True,
                        previous=list(recent), record=row)
            if len(recent) < 12 or n % 1 == 0:
                recent.append(row)
    annotation = {}
    sp = path.with_name('states.csv')
    if events and sp.exists():
        target = events.get('1000000', next(iter(events.values())))['epoch']
        nearest = []
        prev = deque(maxlen=5)
        with sp.open() as f:
            for row in csv.DictReader(f):
                epoch = float(row['time_us']) / 1e6
                if epoch < target-.08:
                    prev.append(row)
                    continue
                nearest = list(prev)
                nearest.append(row)
                for _, row in zip(range(6), csv.DictReader(f, fieldnames=list(row))):
                    nearest.append(row)
                break
        annotation = dict(near_time_states=nearest,
                          alignment='Timestamp neighborhood, not asserted one-to-one action/state join')
    assert before.st_size == path.stat().st_size and before.st_mtime == path.stat().st_mtime
    return dict(path=str(path), size=before.st_size, rows=n, max_cwnd_before=max_before,
                max_cwnd_requested=max_request, positive_percent_writes=positive,
                bad_action_ids=bad_id, failed_writes=failed_write, first_events=events, **annotation)


def main():
    started = time.time()
    results = []
    for arm in FIRST:
        for p in sorted((OLD/'frameworks'/arm/'logs').glob('twotrack-formal-*-actor*/control.csv')):
            results.append(dict(arm=arm, **scan(p, arm)))
    OUT.mkdir(exist_ok=True)
    data = dict(host=os.uname().nodename, epoch=time.time(), wall_s=time.time()-started,
                mode='READ_ONLY_NO_TCP', results=results,
                limitations=['Snapshot cnt counts RTT samples, not acknowledged packets.',
                             'Cwnd alone cannot establish cwnd-limited sending.',
                             'Snapshots near a write do not prove exact actor-weight version.'])
    (OUT/'history-scan.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(dict(host=data['host'], wall_s=data['wall_s'], logs=len(results),
                         huge=[dict(arm=x['arm'], log=pathlib.Path(x['path']).parent.name,
                                    max_cwnd=x['max_cwnd_before'],
                                    seconds_to_first_update=x['first_events']['1000000']['seconds_from_first_learner_update'])
                               for x in results if '1000000' in x['first_events']])))


if __name__ == '__main__':
    main()
