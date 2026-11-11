"""Recompute screening from the published per-run evidence, without cloud access."""
import csv
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METRICS = ('utilization', 'queue_mean_ms', 'queue_p95_ms', 'queue_p99_ms', 'drop_fraction')


def load(name):
    return json.loads((HERE / name).read_bytes())


def main():
    registration = load('registration-frozen.json')
    assessment = load('screen-assessment.json')
    screen = [row for host in load('screen-latest.json') for row in host['rows']]
    positive = [row for host in load('positive-latest.json') for row in host['rows']]
    assert len(screen) == 18 and len(positive) == 6
    assert all(row['status'] == 'COMPLETED' for row in screen + positive)
    runs = list(screen)
    for row in load('historical-endpoint-runs.json'):
        job = row['job']
        assert job['group'] == 'fresh' and job['condition']['condition'] in registration['screen_conditions']
        tail = row['second_bins'][50:60]
        runs.append(dict(row, job=dict(job, group=str(registration['final_step'])),
                         windows=row['network'], tail_utilization=sum(x['departure_bytes'] for x in tail) / sum(x['service_bytes'] for x in tail)))
    csv_rows = list(csv.DictReader((HERE / 'checkpoint-screen.csv').open()))
    assert len(csv_rows) == 24
    for model in assessment['rankings']:
        cells = {}
        for name in registration['screen_conditions']:
            part = [r for r in runs if int(r['job']['group']) == model['step'] and r['job']['condition']['condition'] == name]
            assert part
            values = {key: median(next(w for w in r['windows'] if w['window'] == 'trim10')[key] for r in part) for key in METRICS}
            tail = median(r['tail_utilization'] for r in part)
            assert values == model['cells'][name]['metrics']
            assert tail == model['cells'][name]['tail_utilization']
            cells[name] = values
            row = next(r for r in csv_rows if int(r['step']) == model['step'] and r['condition'] == name)
            for key in ('utilization', 'queue_mean_ms', 'queue_p95_ms'):
                assert float(row[key]) == values[key]
            assert float(row['tail_utilization']) == tail
        util = mean(c['utilization'] for c in cells.values())
        queue = mean(c['queue_mean_ms'] for c in cells.values())
        assert util == model['mean_utilization'] and queue == model['mean_queue_ms']
        eligible = util >= .90 and queue <= 25 and sum(c['utilization'] < .80 for c in cells.values()) <= 1
        assert eligible == model['eligible']
    assert assessment['status'] == 'NO_CANDIDATE' and assessment['selected_step'] is None
    assert all(not model['eligible'] for model in assessment['rankings'])
    print(json.dumps(dict(status='PASS_PUBLIC_RECOMPUTATION', frozen_runs=24, screen_cells=24, decision=assessment['status'])))


if __name__ == '__main__':
    main()
