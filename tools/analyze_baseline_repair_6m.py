"""Recompute this run's training/support aggregates; no TF or cloud access."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


def analyze(folder):
    folder = Path(folder)
    load = lambda name: json.loads((folder/name).read_bytes())
    if (folder/'training-analysis-input.json').exists():
        data = load('training-analysis-input.json')
    else:
        with ZipFile(folder/'training-small-analysis.zip') as archive:
            data = json.loads(archive.read('analysis-input.json'))
    done = load('training-complete.json')
    rows = data['training']
    valid = [r for r in rows if math.isfinite(float(r['actor_grad']))]
    zero = [r for r in valid if float(r['actor_grad']) == 0]
    recovered = next(r for r in valid if int(r['inserted']) > 1000000 and float(r['actor_grad']) > .001)
    support = [a for host in load('action-support-audit.json') for a in host['actors']]
    names = ('rows', 'lower_boundary_rows', 'upper_boundary_rows', 'exact_boundary_rows',
             'lower_to_positive', 'upper_to_negative', 'expected_sign_reversals_under_source_Gaussian')
    behavior = {key: sum(a['behavior'][key] for a in support) for key in names}
    assert behavior['rows'] == done['final']['inserted']
    assert sum(done['coverage']['inserted_by_actor']) == done['final']['inserted']
    assert sum(done['coverage']['sampled_by_actor']) == 512*done['final']['counters']['critic_iterations']
    behavior['near_boundary_fraction'] = (behavior['lower_boundary_rows']+behavior['upper_boundary_rows'])/behavior['rows']
    behavior['exact_boundary_fraction'] = behavior['exact_boundary_rows']/behavior['rows']
    debt = [r for r in data['history'] if r['debt'] is not None]
    age = [float(r['sample_age_mean']) for r in rows if math.isfinite(float(r['sample_age_mean']))]
    native = {key: sum(a['native_control'][key] for a in support)
              for key in ('rows','requested_min4','requested_max2147483647',
                          'before_max_and_percent_ge100','violations_of_upper_clamp_alias')}
    inserted = done['coverage']['inserted_by_actor']
    sampled = done['coverage']['sampled_by_actor']
    counters = done['final']['counters']
    assert counters['critic_optimizer_ops'] == 2*counters['critic_iterations']
    assert done['global_optimizer_step'] == 3*counters['critic_iterations']
    return dict(status='TRAINING_ACCOUNTING_COMPLETE_FROZEN_PERFORMANCE_SEPARATE',
        training_commit=done['config']['code_commit'],
        run=done['result'], final=done['final'], global_optimizer_step=done['global_optimizer_step'],
        training_rows=len(rows), finite_actor_gradient_records=len(valid), exact_zero_actor_gradient_records=len(zero),
        first_recorded_recovery_after_1m={k:recovered[k] for k in ('inserted','wall_s','actor_grad')},
        final_recorded_gradient=valid[-1]['actor_grad'],
        history_records=len(data['history']),
        maximum_recorded_debt=max(r['debt'] for r in debt),
        maximum_recorded_debt_after_100k=max(r['debt'] for r in debt if r['inserted']>=100000),
        final_recorded_debt=debt[-1]['debt'],
        maximum_recorded_FIFO=max(int(r['fifo_backlog']) for r in rows),
        sample_age_mean_record_max=max(age),
        per_actor_real_inserted=dict(minimum=min(inserted),maximum=max(inserted),mean=mean(inserted)),
        per_actor_replay_rows_used=dict(minimum=min(sampled),maximum=max(sampled),spread=max(sampled)-min(sampled)),
        behavior=behavior,native=native,
        caveats=['Recorded diagnostic batches are not every optimizer step.',
                 'Boundary reversals are sign reversals of behavior actions, not measured TCP escape probabilities.',
                 'Weight/BN function swaps break their covariance and are not online causal interventions.',
                 'No frozen performance inference follows from gradient recovery or training utility.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=analyze(args.folder)
    value=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(value,encoding='utf-8')
    else: print(value)
