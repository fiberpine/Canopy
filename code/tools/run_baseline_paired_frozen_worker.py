"""Run a bound paired plan using the already validated single-flow frozen worker.

Deploy in a separate framework containing frozen_worker.py, capture_frozen.py
and packet_window_completion.py. No optimizer, exploratory or retry path exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import frozen_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.plan.read_text())
    registration = json.loads(args.registration.read_text())
    output = args.output_prefix
    marker = Path(str(output) + '-started.json')
    with marker.open('x') as stream:
        json.dump(dict(epoch=time.time(), jobs=len(jobs)), stream)
    rows = []
    for job in jobs:
        if time.time()+110 >= registration['run_cutoff'] or (frozen_worker.ROOT / 'STOP').exists():
            row = dict(job=job, status='NOT_RUN_BUDGET_OR_STOP')
        else:
            try:
                native = Path(job['native'])
                assert hashlib.sha256(native.read_bytes()).hexdigest() == job['native_sha256']
                row = frozen_worker.run(job, dict(registration, native=str(native)))
                assert hashlib.sha256(native.read_bytes()).hexdigest() == job['native_sha256']
                row['native_sha256'] = job['native_sha256']
                receipt = frozen_worker.ROOT / 'logs' / job['run'] / 'packet-window-completion.json'
                row['packet_window_completion'] = json.loads(receipt.read_text()) if receipt.exists() else dict(status='NATIVE_EXIT_COMPLETION_CHECKED_BY_ORIGINAL_WORKER')
            except Exception as error:
                row = dict(job=job, status='EXECUTION_ERROR', reason=repr(error), traceback=traceback.format_exc())
        rows.append(row)
        frozen_worker.atomic(Path(str(output) + '-progress.json'), rows)
    frozen_worker.atomic(Path(str(output) + '-result.json'), rows)


if __name__ == '__main__':
    main()
