"""Summarize controlled group-loss interventions; no experiment launch or safety gate."""
import argparse
import json
import pathlib


def first_nonpositive(records, field, rtt, cls):
    prev = None
    for row in records:
        value = row[field][str(rtt)][cls]
        if row['update'] >= 500 and value is not None and value['mean'] <= 0:
            return dict(previous_recorded_update=prev,first_recorded_nonpositive=row['update'])
        prev = row['update']
    return None


def analyze(root):
    root=pathlib.Path(root)
    out=dict(status='CONTROLLED_OFFLINE_DIAGNOSIS',seeds={},new_TCP_runs=0,new_models=0,
             online_L2='NOT_RUN',model_or_gate_accepted=False)
    pilot=json.loads((root/'pilot.json').read_text())
    assert pilot['all_pairing_checks_pass']
    hashes={v['final_parameter_sha256'] for v in pilot['arms'].values()}
    assert len(hashes)==1
    out['pilot']=dict(A_A_all_variables_bit_exact=True,weighted_allone_vs_stock_bit_exact=True,
                      maximum_gradient_error=max(x['maximum_allone_stock_gradient_error'] for x in pilot['arms'].values()))
    for seed in (3,4):
        path=root/('paired-s%d.json'%seed)
        if not path.exists():
            out['seeds'][str(seed)]=dict(status='NOT_RUN_OR_INCOMPLETE')
            continue
        data=json.loads(path.read_text());assert data['all_pairing_checks_pass']
        summary={}
        arms=dict(data['arms'])
        own_path=root/('own-s%d.json'%seed)
        if own_path.exists():
            own=json.loads(own_path.read_text())
            assert own['all_pairing_checks_pass'] and own['sampler_seed']==seed
            arms.update(own['arms'])
        full=data['arms']['full'];baseline=full['records'][-1]
        for name,arm in arms.items():
            final=arm['records'][-1]
            assert full['paired_hashes']==arm['paired_hashes']
            assert [r['BN_sha256'] for r in full['records']]==[r['BN_sha256'] for r in arm['records']]
            assert [r['actor_policy_sha256'] for r in full['records']]==[r['actor_policy_sha256'] for r in arm['records']]
            summary[name]=dict(critic_updates=arm['critic_updates'],actor_updates=arm['actor_updates'],
                               retained_rows=arm['retained_rows'],retained_rows_by_RTT=arm['retained_rows_by_RTT'],
                               final_q1=final['q1'],final_dq1=final['dq1'],final_dq2=final['dq2'],
                               first_nonpositive_R200_penalized=first_nonpositive(arm['records'],'dq1',200,'penalized'),
                               first_nonpositive_R200_penalized_Q2=first_nonpositive(arm['records'],'dq2',200,'penalized'),
                               final_R200_penalized_delta_vs_full=final['dq1']['200']['penalized']['mean']-baseline['dq1']['200']['penalized']['mean'],
                               final_fixed_action_R200_penalized=final['fixed_initial_action_dq1']['200']['penalized'],
                               actor_weights_and_optimizer_unchanged=arm['actor_weights_and_optimizer_unchanged'])
            summary[name]['last_three_R200_penalized_q1']=[r['dq1']['200']['penalized']['mean'] for r in arm['records'][-3:]]
            summary[name]['last_three_R200_penalized_q2']=[r['dq2']['200']['penalized']['mean'] for r in arm['records'][-3:]]
        out['seeds'][str(seed)]=dict(status=data['status'],wall_s=data['wall_s'],arms=summary,
                                    matched_inputs_and_actor_BN_policy=True,target_order=data['target_order'],smoothing=data['smoothing'])
    out['limits']=[
        'Two sampler seeds from one source model/data seed are not independent online training seeds.',
        'Same controlled duplicate-target composition and external smoothing distribution are used in every arm; this is not historical H1 bitwise replay.',
        'Masks retain equal actor-group counts; actual row totals are reported and may differ slightly with balanced remainder rotation.',
        'Group-loss masking changes optimization and future bootstrap labels. It identifies controlled group-loss contribution, not unique gradient provenance.',
        'Unchanged R200 samples are not removed. A retained negative sign without short-RTT losses disproves necessity of those losses in this setting.',
        'No throughput, delay, CPU advantage or online-release rule is tested.']
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    p=pathlib.Path(args.output);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(analyze(args.results),indent=2)+'\n')
