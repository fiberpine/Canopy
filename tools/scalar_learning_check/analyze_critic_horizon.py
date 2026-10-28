"""Summarize finite historical-data TF diagnostics; never launch jobs or assert Q truth."""
import argparse
import json
import math
import pathlib


def crossing_interval(records, rtt, cls, field='dq1_by_rtt_class'):
    previous = None
    for row in records:
        entry = row[field][str(rtt)][cls]
        value = entry['mean'] if entry is not None else None
        if row['update'] >= 500 and value is not None and value <= 0:
            return dict(last_recorded_update=previous, first_recorded_nonpositive=row['update'])
        previous = row['update']
    return None


def summarize_horizon(z):
    result = z['result']
    records = result['records']
    final = records[-1]
    grouped = {}
    for rtt in (20, 50, 100, 200):
        grouped[str(rtt)] = {}
        for cls in ('penalized', 'free'):
            hist = [(r['update'], r['dq1_by_rtt_class'][str(rtt)][cls]) for r in records]
            values = [(t, d['mean']) for t, d in hist if d is not None]
            grouped[str(rtt)][cls] = dict(
                probe=result['probe_classes'][str(rtt)][cls],
                first_nonpositive=crossing_interval(records, rtt, cls),
                first_nonpositive_q2=crossing_interval(records, rtt, cls, 'dq2_by_rtt_class'),
                sign_reversals=sum(a*b < 0 for (_, a), (_, b) in zip(values, values[1:])),
                final_dq1=final['dq1_by_rtt_class'][str(rtt)][cls],
                final_dq2=final['dq2_by_rtt_class'][str(rtt)][cls],
                final_fixed_initial_action_dq1=final['fixed_initial_action_dq1_by_rtt_class'][str(rtt)][cls])
    # Audit the proposed gate only. Passing this is not authorization or evidence of online safety.
    qualifying = []
    for row in records:
        if row['update'] < 5000:
            continue
        good = True
        for rtt in (20, 50, 100, 200):
            cls = result['probe_classes'][str(rtt)]['penalized']
            d1 = row['dq1_by_rtt_class'][str(rtt)]['penalized']
            agreement = row['twin_agree_by_rtt_class'][str(rtt)]['penalized']
            good = good and cls['n'] >= 10 and d1 is not None and d1['mean'] < 0
            good = good and agreement is not None and agreement['mean'] >= .8
        qualifying.append((row['update'], good))
    gate = next((b[0] for a, b in zip(qualifying, qualifying[1:]) if a[1] and b[1]), None)
    return dict(tau=result['tau'], updates=result['updates'], groups=grouped,
                proposed_gate_first_two_consecutive=gate, replay_reward_mean=result['replay_reward_mean'],
                final_q1_behavior=final['q1_behavior_mean'], final_twin_agreement=final['twin_agree'],
                final_policy_max_shift_from_initial=final['policy_max_shift_from_initial'],
                actor_trainables_unchanged=result['actor_trainables_unchanged'],
                final_descriptive_lambda=final['fitted_lambda'],
                note='Recorded intervals are coarse and may include reversals; scalar lambda is not an identified horizon.')


def summarize_structure(z):
    release = next(r for r in z['records'] if r['iteration'] == z['critic_only_updates'])
    final = z['records'][-1]
    p = final['per_state']
    ranked = sorted(range(len(p['delta'])), key=lambda i: abs(p['delta'][i]), reverse=True)
    top = ranked[:int(math.ceil(.05 * len(ranked)))]
    correct = [i for i in top if p['rtt'][i] in (20, 50) and p['features'][i][6] < .999 and p['delta'][i] < 0]
    first_birth = {actor: min(b for a, b in zip(p['actor'], p['probe_birth']) if a == actor)
                   for actor in set(p['actor'])}
    nonfirst = [i for i in ranked if p['probe_birth'][i] != first_birth[p['actor'][i]]]
    nonfirst_top = nonfirst[:int(math.ceil(.05 * len(nonfirst)))]
    posthoc = dict(status='POST_HOC_NOT_REPLACEMENT_FOR_ORIGINAL_PREDICTION',
                   nonfirst_probe_count=len(nonfirst), top_count=len(nonfirst_top),
                   negative_R20_R50_penalized=sum(p['rtt'][i] in (20, 50) and p['features'][i][6] < .999
                                                 and p['delta'][i] < 0 for i in nonfirst_top),
                   max_abs_shift=max(abs(p['delta'][i]) for i in nonfirst))
    return dict(sampler_seed=z['sampler_seed'], release=release['per_state_structure'],
                final=final['per_state_structure'], top5pct_count=len(top),
                top5pct_negative_R20_R50_penalized=len(correct), fraction=len(correct)/float(len(top)),
                top5pct_free_count=sum(p['features'][i][6] >= .999 for i in top),
                top5pct_first_probe_per_actor=sum(p['probe_birth'][i] == first_birth[p['actor'][i]] for i in top),
                posthoc_excluding_first_probe_per_actor=posthoc,
                top5pct_rows=[dict(actor=p['actor'][i], rtt=p['rtt'][i], birth=p['probe_birth'][i],
                                  delta=p['delta'][i], policy=p['policy'][i], delay_factor=p['features'][i][6]) for i in top],
                final_saturated_fraction=final['pi_sat'], final_absolute_shift_p95=final['pi_delta_p95'],
                final_absolute_shift_max=final['pi_delta_max'],
                interpretation='State association on fixed historical probes, not proof of a beneficial network correction.')


def analyze(root):
    root = pathlib.Path(root)
    out = dict(status='OFFLINE_DIAGNOSIS', jobs={}, hypotheses={}, new_TCP_runs=0,
               candidate_checkpoints_written=0, online_L2='NOT_RUN')
    for job in ('pilot-H1', 'E1R', 'H1', 'H2', 'S-R5-s3', 'S-R5-s4', 'S-R5-s5'):
        p = root / (job+'.json')
        if not p.exists():
            out['jobs'][job] = dict(status='NOT_RUN_OR_INCOMPLETE')
            continue
        z = json.loads(p.read_text())
        assert z['new_TCP_runs'] == 0 and z['source_checkpoints_written'] == 0
        if job in ('H1', 'H2'):
            out['jobs'][job] = summarize_horizon(z)
        elif job.startswith('S-'):
            out['jobs'][job] = summarize_structure(z)
        else:
            out['jobs'][job] = dict(status=z['status'], result=z.get('result'))
    p = root / 'E1R-response.json'
    if p.exists():
        z = json.loads(p.read_text())
        out['E1R_response'] = z
    if 'H1' in out['jobs'] and 'groups' in out['jobs']['H1']:
        crossing = out['jobs']['H1']['groups']['200']['penalized']['first_nonpositive']
        first = crossing['first_recorded_nonpositive'] if crossing else None
        band = ('FAIL_EARLY' if first is not None and first < 12000 else
                'RECORDED_WITHIN_BAND' if first is not None and first <= 35000 else 'NOT_WITHIN_BAND')
        structures = [r for name, r in out['jobs'].items() if name.startswith('S-') and 'fraction' in r]
        out['hypotheses'] = dict(
            H1_R200=dict(expected_band=[12000, 35000], recorded_crossing=crossing, status=band),
            H2='Compare recorded crossing intervals; exact5x scaling is not identified by coarse intervals and one paired seed alone',
            S=dict(expected_min_fraction=.8, fractions=[r['fraction'] for r in structures],
                   pass_count=sum(r['fraction'] >= .8 for r in structures), completed=len(structures)),
            E1R='Use signed cumulative responses and HAC errors below; no persistent-action identification',
            proposed_online_gate=dict(H1_static_first_pass=out['jobs']['H1']['proposed_gate_first_two_consecutive'],
                                      interpretation='Not a Q-truth or online-safety certificate'),
            research_status='PARTIAL; onlineL2 NOT_RUN; no new accepted baseline')
    out['limitations'] = [
        'Static off-policy data reuse and neural/target/BN dynamics do not satisfy the ideal fixed-operator propagation model.',
        'H1/H2 are one paired sampler seed; S repetitions are not independent training seeds.',
        'HAC intervals describe innovation responses; neither persistent-action effects nor Q gradients follow automatically.',
        'No new throughput, tail delay or system CPU benefit was measured.']
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = pathlib.Path(args.output)
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analyze(args.results), indent=2, allow_nan=False)+'\n')
