"""Same-trajectory reward alias audit, requiring REAL ACK-counter boundaries.

Coarsening is an offline measurement counterfactual, not a new TCP policy run.
It cannot reconstruct unobserved queue tails or substitute for fixed controls.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from frameworks.joint_healthy.core import interval_return


def audit(feedback_path, boundaries_path):
    frames = [json.loads(x) for x in Path(feedback_path).read_text().splitlines() if x.strip()]
    with Path(boundaries_path).open() as stream: boundaries = list(csv.DictReader(stream))
    if len(frames) < 9 or len(boundaries) < len(frames): raise ValueError('At least9 complete, paired boundaries required')
    rows = []
    for frame, boundary in zip(frames, boundaries):
        raw = np.asarray(frame['raw'], dtype=float)
        if raw.shape != (15,) or not np.isfinite(raw).all() or raw[3] <= 0: raise ValueError('Invalid raw feedback')
        if int(boundary['state_id']) != frame['state_id']: raise ValueError('Boundary identity mismatch')
        dt = int(boundary['dt_us']) / 1e6; acked = int(boundary['delta_bytes'])
        if not math.isclose(dt, raw[3], rel_tol=1e-6, abs_tol=1e-7): raise ValueError('Native duration mismatch')
        if not math.isclose(acked / dt, raw[1], rel_tol=2e-6, abs_tol=1e-6):
            raise ValueError('These are not ACKed-byte rate observations; legacy averages cannot reconstruct byte delivery')
        rows.append(dict(dt=dt, acked=acked, loss_bytes=raw[7]*dt, minimum=raw[14], srtt=raw[8], rate=raw[1]))
    if rows[0]['rate'] <= 0: raise ValueError('Invalid initial normalization reference')
    results = []
    # The initial feedback is bootstrap; only subsequent action intervals enter
    # returns, exactly as in the actor/replay transition ledger.
    intervals = rows[1:]
    for group in (1, 2, 4, 8):
        peak = rows[0]['rate']; total_reward = 0.; discount = 1.; seconds = 0.; durations = []; weighted_u = 0.
        for start in range(0, len(intervals), group):
            part = intervals[start:start+group]; dt = math.fsum(x['dt'] for x in part)
            rate = sum(x['acked'] for x in part) / dt
            loss = math.fsum(x['loss_bytes'] for x in part) / dt
            peak = max(peak, rate)
            endpoint = part[-1]
            u = (rate - 5*loss) / peak * min(1., 1.25*endpoint['minimum']/endpoint['srtt'])
            reward, d = interval_return(u, dt)
            total_reward += discount*reward; discount *= d; seconds += dt; weighted_u += u*dt; durations.append(dt)
        results.append(dict(group_original_decisions=group, actual_mean_interval_ms=float(np.mean(durations))*1000,
            measured_seconds=seconds, final_observed_peak=peak, discounted_return=total_reward,
            time_weighted_utility=weighted_u/seconds, intervals=len(durations)))
    base = results[0]
    for result in results:
        result['relative_return_change_same_trajectory'] = (result['discounted_return'] - base['discounted_return'])/max(1e-9, abs(base['discounted_return']))
    return dict(status='OFFLINE_MEASUREMENT_ALIAS_AUDIT_NOT_POLICY_PERFORMANCE', input_frames=len(frames),
                return_increase_without_changed_TCP_trajectory=any(x['relative_return_change_same_trajectory']>.02 for x in results[1:]),
                nonlinear_interval_utility_is_approximate=True, p95_p99_reconstructed=False,
                no_fixed_control_training_performed=True, results=results)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--feedback',required=True);ap.add_argument('--boundaries',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();result=audit(args.feedback,args.boundaries)
    path=Path(args.output)
    if path.exists():raise ValueError('Preserve an existing audit result')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
