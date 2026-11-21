"""Read-only source/evidence audit. No TF, SSH, TCP or optimizer execution.

Numerical illustrations are explicitly not network counterfactuals. Output is
deterministic and records hashes of all inputs used in the new calculations.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import types
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = 'evidence/base-reproduction-20261028/'
METRICS = ('utilization', 'queue_mean_ms', 'queue_p95_ms',
           'queue_p99_ms', 'drop_fraction')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(rows):
    """Exactly one complete full/trim pair per arm and condition is required."""
    if len(rows) != 96:
        raise ValueError('Expected 48 completed runs and 96 analysis windows')
    keys = set()
    for row in rows:
        key = (row['group'], row['condition'], row['window'])
        if key in keys:
            raise ValueError('Duplicate condition/arm/window')
        keys.add(key)
        if row['complete_window'] != 'True' or row['policy_unchanged'] != 'True':
            raise ValueError('Incomplete or mutable-policy run')
        if any(not math.isfinite(float(row[k])) for k in METRICS + ('reward_mean',)):
            raise ValueError('Nonfinite scientific metric')
    groups = {}
    for arm in ('archived', 'stopped'):
        for window in ('full', 'trim10'):
            selected = [r for r in rows if r['group'] == arm and r['window'] == window]
            if len(selected) != 24:
                raise ValueError('Missing arm/condition/window')
            groups[(arm, window)] = {r['condition']: r for r in selected}
    condition_sets = [set(v) for v in groups.values()]
    if any(s != condition_sets[0] for s in condition_sets[1:]):
        raise ValueError('Mismatched condition sets')
    means = {}
    for (arm, window), values in groups.items():
        selected = list(values.values())
        means[arm + '_' + window] = {
            k: statistics.mean(float(r[k]) for r in selected)
            for k in METRICS + ('reward_mean',)
        }
        means[arm + '_' + window].update(
            conditions=24,
            utilization_below_80=sum(float(r['utilization']) < .8 for r in selected),
            min_saturation_ge_99=sum(float(r['min_saturation']) >= .99 for r in selected),
            max_saturation_ge_99=sum(float(r['max_saturation']) >= .99 for r in selected))
    inversions = []
    for window in ('full', 'trim10'):
        for condition, old in groups[('archived', window)].items():
            new = groups[('stopped', window)][condition]
            for lo_name, lo, hi_name, hi in [('stopped', new, 'archived', old),
                                            ('archived', old, 'stopped', new)]:
                worse = float(lo['utilization']) < float(hi['utilization']) and all(
                    float(lo[k]) >= float(hi[k]) for k in METRICS[1:])
                if worse and float(lo['reward_mean']) > float(hi['reward_mean']):
                    inversions.append(dict(condition=condition, window=window,
                                           lower_quality=lo_name, higher_quality=hi_name))
    return dict(condition_equal=means, observed_pareto_reward_inversions=inversions,
                scope='One training seed, one frozen repeat per condition/arm; no causal inference. '
                'Reward is the existing decision-weighted mean; averaged condition P95 is not pooled P95.')


def numerical(joint):
    gamma = .995
    value = .7 / (1 - gamma)
    targets = []
    for dt in (.02, .04, .08, .12, .16):
        interval = joint.Interval(gamma, .02, 0.)
        interval.add(.7, dt)
        targets.append(dict(dt_ms=dt * 1000, reward=interval.reward,
                            discount=interval.discount,
                            cold_Q0_target=interval.reward,
                            calibrated_constant_Q_target=interval.reward + interval.discount * value,
                            nominal_updates_per_decision=.08 * dt / .02))
    assert max(abs(t['calibrated_constant_Q_target'] - value) for t in targets) < 1e-10
    whole = joint.Interval(gamma, .02, 0.)
    whole.add(.7, .16)
    parts = joint.Interval(gamma, .02, 0.)
    for _ in range(8):
        parts.add(.7, .02)
    split_error = abs(whole.reward - parts.reward)
    assert split_error < 1e-12
    impulses = []
    for action in (-.05, -.000001, 0., .000001, .05):
        percent = int(100 * 4 ** action)
        for dt in (.02, .16):
            impulses.append(dict(action=action, period_ms=dt * 1000, percent=percent,
                ideal_continuous_multiplier_per_s=4 ** (action / dt),
                percent_quantized_multiplier_per_s=(percent / 100.) ** (1 / dt)))
    return dict(status='PASS_NUMERICAL_IDENTITIES_NOT_TCP',
        constant_utility=.7, calibrated_constant_value=value,
        constant_utility_split_error=split_error, time_targets=targets,
        window_impulses=impulses,
        discount_e_folding_seconds=[dict(interval_ms=dt * 1000,
            fixed_gamma=dt / -math.log(gamma), physical_gamma=.02 / -math.log(gamma))
            for dt in (.02, .05, .1, .16)],
        caveats=['Impulse products omit CUBIC/ACK dynamics, integer window floor and bounds.',
                 'Constant utility and fixed continuation Q isolate a mathematical bias; '
                 'they do not prove a particular trained critic caused collapse.',
                 'A cost is charged once per real decision; splitting a decision must not add costs.',
                 'Nominal physical credit per decision is not actual UTD with dispatch/warmup effects.'])


def reward_source_fixtures():
    """Run the actual get_state methods on synthetic, finite wire inputs only."""
    results = {}
    for variant in ('baseline', 'mtpjoint'):
        path = ROOT / ('frameworks/' + variant + '/code/canopy/rl-module/envwrapper_v2.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'TCP_Env_Wrapper')
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'get_state')
        # Reuse the parsed module container; no original module-level imports run.
        tree.body = [fn]
        namespace = {'np': np, 'sleep': lambda _: None}
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
        def evaluate(rate, cwnd, samples, srtt=20., loss=0., peak=0.):
            raw = [srtt, rate, samples, .02, 50., cwnd, 1.2*rate, loss,
                   srtt, 2147483647., cwnd, 0., cwnd, 1460., 20.]
            wire = ('1 ' + ' '.join(map(str, raw)) + '\0').encode()
            obj = types.SimpleNamespace(params=types.SimpleNamespace(dict={
                'input_dim': 15, 'delay_margin_coef': 1.25}),
                shrmem_r=types.SimpleNamespace(read=lambda: wire), prev_rid=-1,
                use_normalizer=False, local_counter=0, max_bw=peak, max_cwnd=0.,
                max_smp=0., min_del=9999999., rohit_randomness_exp=False)
            info, _, reward, ok = namespace['get_state'](obj, evaluation=True)
            assert ok and np.isfinite(info['state']).all() and math.isfinite(reward)
            return dict(state=info['state'].tolist(), reward=float(reward))
        low = evaluate(100000., 4., 1.)
        high = evaluate(2500000., 100., 25.)
        assert low == high and low['reward'] == 1.
        loss40 = evaluate(1000000., 100., 25., srtt=40., loss=300000., peak=2500000.)
        loss80 = evaluate(1000000., 100., 25., srtt=80., loss=300000., peak=2500000.)
        assert loss80['reward'] > loss40['reward']
        results[variant] = dict(equal_state_and_reward_at_synthetic_24Mbps_utilizations=[1/30, 5/6],
            equal_state=low['state'], equal_reward=low['reward'],
            higher_delay_less_negative_reward=dict(
                throughput_bytes_s=1000000, loss_bytes_s=300000, peak_bytes_s=2500000,
                min_rtt_ms=20, srtt40_reward=loss40['reward'], srtt80_reward=loss80['reward']),
            scope='Actual source formula on synthetic inputs, not a TCP trace or causal explanation.')
    return results


def run():
    paths = [BASE + 'all-network-windows.csv', BASE + 'final-analysis.json',
             BASE + 'timing-refined.json', 'templates/baseline-train.json',
             'templates/mtpjoint-train.json', 'frameworks/mtpjoint/joint_mtp.py',
             'frameworks/baseline/distributed.py',
             'frameworks/baseline/code/canopy/rl-module/agent_v2.py',
             'frameworks/baseline/moving_stats_backend.py',
             'frameworks/mtpjoint/distributed.py', 'frameworks/mtpjoint/decision_clock_v21.py',
             'frameworks/mtpjoint/independent_critic_v34.py',
             'frameworks/baseline/code/canopy/rl-module/envwrapper_v2.py',
             'frameworks/mtpjoint/code/canopy/rl-module/envwrapper_v2.py']
    read = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8'))
    with (ROOT / paths[0]).open(newline='', encoding='utf-8') as f:
        endpoint = summarize(list(csv.DictReader(f)))
    original = read(BASE + 'final-analysis.json')
    for arm in ('archived', 'stopped'):
        for window in ('full', 'trim10'):
            reference = original['formal_summary'][arm][window]['condition_equal']
            for metric in METRICS:
                assert abs(endpoint['condition_equal'][arm + '_' + window][metric] - reference[metric]) < 1e-10
    coverage = original['replay_coverage']
    insert = coverage['inserted_by_actor']
    sampled = coverage['sampled_by_actor']
    retained = coverage['retained_by_actor']
    replay = dict(inserted_min=min(insert), inserted_max=max(insert),
                  inserted_max_min_ratio=max(insert)/min(insert),
                  retained_min=min(retained), retained_max=max(retained),
                  sampled_contribution_spread=max(sampled)-min(sampled),
                  rows=[dict(actor=i, inserted=n, sampled=sampled[i], retained=retained[i],
                             sampled_rows_per_inserted=sampled[i]/n) for i, n in enumerate(insert)],
                  caveat='Aggregate reuse ratios, not each transition replay count, independent '
                  'sample counts, or proof that balanced sampling is wrong.')
    timing = read(BASE + 'timing-refined.json')
    slow = [r['windows']['trim10'] for r in timing if r['condition'] == 'G12-R200']
    timing_out = dict(runs=len(timing), G12_R200_slow_cycles=sum(x['cycle_over_50ms'] for x in slow),
                     G12_R200_sample_wait_dominant=sum(x['wait_sample_dominates_slow'] for x in slow))
    configs = {}
    for variant in ('baseline', 'mtpjoint'):
        cfg = read('templates/' + variant + '-train.json')
        configs[variant] = {k: cfg.get(k, False if k in ('target_bn_consistent',
            'independent_critic_adam_v34') else None) for k in
            ('backend', 'target_bn_consistent', 'independent_critic_adam_v34',
             'update_ratio', 'update_credit_basis', 'updates_per_base20ms',
             'decision_cost', 'connection_reset_seconds', 'time_head_nonlinearity')}
    joint = load_module(ROOT / 'frameworks/mtpjoint/joint_mtp.py', 'audit_joint_numerics')
    return dict(status='COMPLETE_LOCAL_AUDIT_NO_NEW_NETWORK_OR_TF',
                source_commit='1fa2c756762df42cc45326b072811ab61122057a',
                input_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                endpoint=endpoint, replay=replay, timing=timing_out,
                current_template_flags=configs, numerics=numerical(joint),
                reward_source_fixtures=reward_source_fixtures(),
                limitations=['No new checkpoint queries or TF graph execution.',
                             'No proof of a unique historical root cause.',
                             'No new policy training or performance certification.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite existing audit evidence')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(result['status'])
