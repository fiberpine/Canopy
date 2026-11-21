#!/usr/bin/env python3
"""Per-scenario sign agreement between the randomized exploration response (E1) and the critic (stdlib only).

Pooled RTT-group comparisons mix scenarios with different weights (E1 weights rows, the critic probe set weights
actors equally). This compares like with like: each actor (scenario) x pre-action delay class, E1 cumulative
response at J (default 10) from response-alignment.json against the critic's endpoint dQ1/da and dQ2/da on the
same scenario's probes from snapshot-probe-by-actor.json. Only cells with |z| >= threshold and probes are scored.

    python3 scenario_alignment.py --evidence evidence/scalar-critic-only-L2A-20261028 [--J 10] [--z 3]
"""
import argparse
import json
import pathlib
import sys
import math


def analyze(root, J='10', z_thr=3.0):
    if J not in ('10','40') or not math.isfinite(z_thr) or z_thr<=0:
        raise ValueError('Explicit J10/J40 and finite positive nominal z threshold required')
    root = pathlib.Path(root)
    actors = json.loads((root / 'results/response-alignment.json').read_text())['actors']
    rec = json.loads((root / 'results/snapshot-probe-by-actor.json').read_text())['result']['records'][-1]
    cells, agree, disagree,excluded = [], 0, 0,[]
    for a in actors:
        for cls in ('penalized', 'free'):
            full = a['classes'][cls]['full']
            q1 = rec['dq1_by_actor_class'][str(a['actor'])][cls]
            q2 = rec['dq2_by_actor_class'][str(a['actor'])][cls]
            if not full or not q1 or not q2 or full['se'][J] <= 0:
                excluded.append(dict(actor=a['actor'],cls=cls,reason='Missing response/probe or nonpositive SE'))
                continue
            if not all(math.isfinite(v) for v in [full['beta'][J],full['se'][J],q1['mean'],q2['mean']]):
                excluded.append(dict(actor=a['actor'],cls=cls,reason='Nonfinite response/SE/critic'))
                continue
            z = full['beta'][J] / full['se'][J]
            row = dict(actor=a['actor'], rtt=a['rtt'], cls=cls, e1_rows=full['n'], e1=full['beta'][J], z=z,
                       dq1=q1['mean'], dq2=q2['mean'], probes=q1['n'])
            if abs(z) >= z_thr:
                sign=lambda x:1 if x>0 else -1 if x<0 else 0
                row['agree'] = sign(full['beta'][J])==sign(q1['mean'])==sign(q2['mean'])
                agree += row['agree']
                disagree += not row['agree']
            cells.append(row)
    return dict(J=J, z_threshold=z_thr, update=rec['update'], scored=agree + disagree, agree=agree, disagree=disagree,
                cells=cells,excluded=excluded,scope='Post-hoc nominal threshold, no multiplicity correction or prospective gate.',
                note='Per actor/delay-class strata align scenario identities; within-class state/time weights can still differ. Innovation response vs deterministic persistent-action gradient: sign comparison only, not proof of critic correctness or online safety.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evidence', required=True)
    p.add_argument('--J', default='10')
    p.add_argument('--z', type=float, default=3.0)
    a = p.parse_args()
    out = analyze(a.evidence, a.J, a.z)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
