"""Prepare bounded retesting, a single-seed recipe repeat, and conditional archival.

Local evidence only. Does not connect, train, evaluate or delete anything.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from statistics import median
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence/baseline-online-bn-repair-20261112'
RUNTIME = '9bb1500eaf791bcb0e97724c63dcb725323dd5cd'


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    load = lambda p: json.loads(p.read_bytes())
    complete = load(E / 'training-complete.json')
    cfg = complete['config']
    h0 = load(ROOT / 'evidence/baseline-scale-audit-20261112/h0-and-current-facts.json')['h0']['config']
    with ZipFile(E / 'training-small-analysis.zip') as z:
        history = json.loads(z.read('analysis-input.json'))
    train = [r for r in history['training'] if r['actor_grad'] != 'nan']
    points = []
    for n in (100000, 300000, 500000, 733192, 1000000, 1500000, 6000000):
        row = min(train, key=lambda r: abs(int(r['inserted']) - n))
        points.append({k: float(row[k]) for k in ('inserted', 'wall_s', 'actor_grad', 'q_mean')})
    probe = load(E / 'fixed-probe-analysis.json')['rows']
    early = min(probe, key=lambda r: abs(r['step'] - 300000))
    collapse = dict(status='OBSERVED_TRANSIENT_SATURATION_CAUSE_PARTLY_LOCALIZED_NOT_UNIQUELY_PROVEN',
                    timeline=points, fixed_historical_probe_300k=early,
                    probe_scope='512 fixed historical states, not the contemporaneous on-policy distribution',
                    direct_mechanism='Large pre-tanh logits and exact action saturation suppress the actor derivative. Critic action gradients remain nonzero on this probe.',
                    proven_repairs=['target actor moving-statistics contract', 'independent critic Adam clocks and exact scopes',
                                    'unique critic target variable pairs', 'one canonical online BN writer per moving statistic'],
                    not_proven=['unique root cause of initial actor saturation', 'why every possible asynchronous run will recover',
                                'multi-seed training success probability', 'startup intervention solves all weak connections'],
                    new_training_gradient_zero_records=6, new_training_finite_gradient_records=1171,
                    previous_failed_6m_gradient_zero_records=543, previous_failed_6m_finite_gradient_records=1271)
    (output / 'collapse-audit.json').write_text(json.dumps(collapse, indent=2) + '\n')
    unchanged = ('num_actors', 'memsize', 'sampler', 'warmup', 'update_ratio', 'reward_mode', 'LOSS_TYPE',
                 'control_period_ms', 'tf_threads', 'exploration_std', 'backend', 'minimum_enqueued_per_actor')
    assert all(h0[k] == cfg[k] for k in unchanged)
    differences = dict(unchanged={k: cfg[k] for k in unchanged},
                       changed=[dict(setting=k, H0=h0.get(k, False), current=cfg[k]) for k in (
                           'target_bn_consistent', 'independent_critic_adam_v34', 'unique_critic_targets',
                           'unique_online_bn_updates', 'sample_budget', 'checkpoint_interval', 'performance_guard_policy')],
                       current_runtime_commit=RUNTIME,
                       note='H0 missing optional boolean flags means disabled; the public historical facts select only relevant config fields. Not a claim of bitwise identical H0/repair optimizer initialization.')
    (output / 'H0-recipe-differences.json').write_text(json.dumps(differences, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('startup_plan', ROOT / 'tools/plan_scalar_startup_screen.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    screen = module.plan()
    for job in screen['jobs']:
        job['host_slot'] = job['pair'] % 2
    screen.update(status='OFFLINE_TWO_HOST_DRAFT_NOT_AUTHORIZED', slots_per_host=1,
                  suggested_budget=dict(initial_phase_minutes=45, active_hosts=2, maximum_host_hours=1.5, authorized=False))
    screen['screen_gates']['failure_to_reproduce_legacy'] = 'MECHANISM_INCONCLUSIVE; performance confirmation may proceed if current first-feedback arm passes all registered performance/protocol gates'
    screen['screen_gates']['legacy_G12_R200_catastrophes_at_least'] = '2/6 is mechanism-screen evidence only, not a performance confirmation prerequisite'
    assert len(screen['jobs']) == 40 and all(sum(j['host_slot'] == h for j in screen['jobs']) == 20 for h in (0, 1))
    (output / 'two-host-screen.draft.json').write_text(json.dumps(screen, indent=2) + '\n')
    replica = copy.deepcopy(cfg)
    run = 'onlinebn-recipe-seed7-6m'
    replica.update(run_id=run, model_id=run, seed=7, code_commit=RUNTIME,
                   model_source='fresh random seed7; never load seed3 or H0 trained weights',
                   learner_endpoint='@NEW_LEARNER_PRIVATE_ENDPOINT@', server_binary='@EXACT_LEGACY_632BCA97_NATIVE@',
                   stop_file='@NEW_ROOT@/frameworks/baseline/logs/' + run + '-stop.json',
                   absolute_deadline_epoch='@BIND_NEW_TRAIN_CUTOFF@', timeout_s='@BIND_NEW_TRAIN_TIMEOUT@',
                   configuration_notes='Recipe repetition: only algorithmic random seed changes from the completed seed3 run. Same legacy training native; deployment evaluation uses the newly validated startup protocol for both models. Fresh replay/optimizer/counters.')
    (output / 'seed7-config.draft.json').write_text(json.dumps(replica, indent=2) + '\n')
    cores = ['distributed.py', 'consistent_target_bn.py', 'independent_critic_v34.py', 'unique_critic_targets.py', 'unique_online_bn.py']
    frozen = {}
    for name in cores:
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', RUNTIME + ':frameworks/baseline/' + name])
        assert raw == (ROOT / 'frameworks/baseline' / name).read_bytes()
        frozen[name] = hashlib.sha256(raw).hexdigest()
    plan = dict(status='OFFLINE_PREPARATION_AWAITING_NEW_BUDGET_AND_POWER_ON',
                authorized=False, requested_overall_minutes=240, conservative_maximum_host_hours=20,
                cleanup_reserve_minutes=10, first_power_on=['canopy-learner-test', 'canopy-actor1'],
                other_hosts='Power on actor2-4 only after the initial screen permits full confirmation',
                old_budget_reused=False, seed3_policy='631683b939596cabc85262e2294123c462913bf6b836dff10b84501644ad60e0',
                training_core_sha256=frozen,
                stages=[dict(name='build_and_weak_screen', hosts=2, maximum_minutes=45,
                             jobs=40, extra_protocol_trials_max=4, training=False,
                             gates='same frozen model; build/kernel/clock/action checks, first-feedback weak cells and sentinels pass; keep legacy mechanism outcome separate'),
                        dict(name='independent_full_seed3_confirmation', hosts=5, jobs=96,
                             candidate_repeats=3, H0_repeats=1, conditions=24,
                             reuse_screen_results=False, training=False, gates='original absolute health gates'),
                        dict(name='single_recipe_replication', hosts=5, learners=1, actors=24, seed=7,
                             fresh=True, nominal_samples=6000000, maximum_runs=1,
                             minimum_remaining_minutes_to_start=150,
                             training_native='exact original legacy632bca97; only seed differs in learning recipe',
                             evaluation_startup='same validated first-feedback deployment protocol as seed3',
                             final_checkpoint='fully drained endpoint; no score-based selection'),
                        dict(name='seed7_frozen_confirmation', hosts=5, candidate_jobs=72, conditions=24,
                             H0_sentinels=6, reference_identity='same H0, same native and metric clock',
                             training=False, gates='same absolute health gates; no performance retries')],
                training_stability=dict(reward_network_noise_U_replay_unchanged=True,
                                        early_boundary_saturation_alone_is_not_failure=True,
                                        no_guard_induced_stop_change=True,
                                        prolonged_boundary_audit=dict(after_samples=1500000, actors_at_least=20,
                                            no_noise_boundary_fraction_at_least=.99, actor_gradient_median_at_most=1e-6,
                                            duration_seconds_at_least=180, stale_actor_data_cannot_pass=True),
                                        if_prolonged_boundary='Recipe stability PARTIAL even if final performance passes; do not clean old code'),
                outcome_rules=dict(seed3_pass='healthy endpoint under explicitly named deployment protocol; recipe reproducibility still pending',
                                   seed3_and_seed7_pass='initial two-training-seed reproduction; not an estimated universal success rate',
                                   any_performance_failure='retain negative result; diagnose; no automatic second long training',
                                   insufficient_time='do not launch a training that cannot reserve final evaluation and cleanup',
                                   cleanup='only after seed3/seed7 health, training-stability and byte-reproduction checks pass'),
                archive_before_cleanup=['immutable source tag plus verified Git bundle', 'both selected policies and complete-model checksums',
                                        'train/eval configs, native/kernel/dependencies, run/sample/update/seed ledgers',
                                        'all acceptance results and failure/repair summaries', 'fresh-checkout reproduction and inventory verification'],
                conditional_cleanup=dict(delete_now=False, scope='GitHub main obsolete baseline execution copies and duplicate checkpoints only after archive gates',
                                         preserve=['protected original H0 local package/tag/history', 'negative evidence and issue/repair records',
                                                   'innovation branches and implementations unless independently proven unused'],
                                         method='dependency/reference inventory; move historical runnable source links to immutable tags; retain one canonical baseline entry'))
    (output / 'stage-plan.draft.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(status=plan['status'], screen_jobs=40, initial_hosts=2,
                          full_confirmation_jobs=96, fresh_training_seed=7, old_code_deleted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
