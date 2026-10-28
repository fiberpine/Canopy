"""Read-only adjudication of published L2-A JSON. Never releases an actor.

The registered J10/J40 references concern randomized compensated innovations;
their equality to a deterministic persistent-action Q gradient is unproved.
Literal gate failure is therefore a reason to withhold release, not a proof of
the unique mechanism of critic error. Fitted lambda is descriptive throughout.
"""
import argparse
import json
import math
import pathlib


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def fixed_references(e1):
    rows=[]
    for rtt in ('20','50','100','200'):
        for cls in ('free','penalized'):
            item=e1['by_rtt'][rtt][cls]
            if item is None:
                continue
            for horizon in (10,40):
                key=str(horizon);b=item['cumulative'][key];se=item['cumulative_se'][key]
                assert math.isfinite(b) and math.isfinite(se) and se>0
                z=b/se
                rows.append(dict(rtt=int(rtt),cls=cls,J=horizon,beta=b,se=se,z=z,
                                 p=math.erfc(abs(z)/math.sqrt(2)),n=item['samples']))
    assert len(rows)==16,'Missing registered reference cell'
    maximum=0.
    for rank,row in enumerate(sorted(rows,key=lambda r:r['p'])):
        maximum=max(maximum,(len(rows)-rank)*row['p'])
        row['holm_p']=min(1.,maximum);row['significant_holm_05']=row['holm_p']<.05
    if 'rows' in e1:
        assert len(rows)==len(e1['rows'])
        for computed,published in zip(rows,e1['rows']):
            assert set(computed)==set(published)
            for key,value in computed.items():
                assert (math.isclose(value,published[key],rel_tol=1e-12,abs_tol=1e-300) if isinstance(value,float) else value==published[key]),'Published fixed-reference table differs from raw class estimates'
    return rows


def gate_a_c(records,references):
    assert len(records)>=5
    counts=[r['update'] for r in records]
    assert counts[0]==0 and all(b>a for a,b in zip(counts,counts[1:])), 'Duplicate/missing initial snapshot'
    cells={(str(r['rtt']),r['cls']) for r in references if r['significant_holm_05']}
    checks=[]
    for record in records:
        conflicts=[];twin_failures=[]
        for ref in references:
            if not ref['significant_holm_05']:continue
            rtt,cls=str(ref['rtt']),ref['cls']
            for critic in (1,2):
                value=(record['dq%d_by_rtt_class'%critic].get(rtt) or {}).get(cls)
                if not value or sign(value['mean'])!=sign(ref['beta']):
                    conflicts.append(dict(rtt=int(rtt),cls=cls,J=ref['J'],critic=critic,
                                          reference=ref['beta'],gradient=value['mean'] if value else None))
        weighted,total,all_agreements=0.,0,[]
        for rtt,classes in record['twin_agree_by_rtt_class'].items():
            for cls,item in classes.items():
                if item:
                    weighted+=item['mean']*item['n'];total+=item['n'];all_agreements.append(item['mean'])
                if (rtt,cls) in cells and (not item or not math.isfinite(item['mean']) or item['mean']<.9):
                    twin_failures.append(dict(rtt=int(rtt),cls=cls,agreement=item['mean'] if item else None))
        checks.append(dict(update=record['update'],fitted_lambda=record['fitted_lambda'],
                           a_signs_pass=not conflicts,conflicts=conflicts,
                           c_significant_cells_pass=not twin_failures,twin_failures=twin_failures,
                           twin_agreement_all_probes=weighted/total if total else None,
                           twin_agreement_min_class=min(all_agreements) if all_agreements else None))
    first=None
    for index in range(4,len(checks)):
        if all(c['a_signs_pass'] and c['c_significant_cells_pass'] for c in checks[index-4:index+1]):
            first=checks[index]['update'];break
    return dict(a_last_five_pass=all(c['a_signs_pass'] for c in checks[-5:]),
                c_last_five_pass=all(c['c_significant_cells_pass'] for c in checks[-5:]),
                first_five_consecutive_a_c_pass=first,checks=checks,
                c_scope='Per significant RTT/class cell, as prior gate implementation; also report global weighted agreement. No global average silently replaces a failing cell.')


def paired_own(own):
    assert own['status']=='PASS_CONTROLLED_OFFLINE' and own['updates']==40000
    arms=own['arms'];assert set(arms)=={'full','own100','own200'}
    base=arms['full'];checks={}
    for name,arm in arms.items():
        assert arm['actor_updates']==0 and arm['critic_updates']==40000
        assert arm['actor_weights_and_optimizer_unchanged']
        assert arm['paired_hashes']==base['paired_hashes'],'Unpaired batches, noise or target actions'
        assert [r['BN_sha256'] for r in arm['records']]==[r['BN_sha256'] for r in base['records']]
        assert [r['actor_policy_sha256'] for r in arm['records']]==[r['actor_policy_sha256'] for r in base['records']]
        assert [r['update'] for r in arm['records']]==[r['update'] for r in base['records']]
        checks[name]=dict(final=arm['records'][-1],retained_rows=arm['retained_rows'],
                          retained_rows_by_RTT=arm['retained_rows_by_RTT'],actor_updates=arm['actor_updates'])
    return checks


def gate_b(own_results,online_matched):
    if len(own_results)!=2:
        return dict(status='NOT_ASSESSED_MISSING_MATCHED_REPEAT',pass_condition=False,completed_repeats=len(own_results))
    checks=[]
    online=online_matched['result']['records'][-1]
    for rtt in ('100','200'):
        for critic in (1,2):
            differences=[]
            for seed,own in own_results.items():
                rows=own['arms']['own'+rtt]['records'];last=rows[-1];full=own['arms']['full']['records'][-1]
                key='dq%d'%critic;values=last[key][rtt]
                assert values['free'] and values['penalized'],'Absent own-class probe'
                difference=values['free']['mean']-values['penalized']['mean'];differences.append(difference)
                late=[r for r in rows if r['fitted_lambda'] is not None and r['fitted_lambda']>=30]
                own_signs={c:sign(values[c]['mean']) for c in ('free','penalized')}
                same_offline=all(own_signs[c]==sign(full[key][rtt][c]['mean']) for c in own_signs)
                same_online=all(own_signs[c]==sign(online[key+'_by_rtt_class'][rtt][c]['mean']) for c in own_signs)
                checks.append(dict(rtt=int(rtt),critic=critic,sampler_seed=seed,lambda_final=last['fitted_lambda'],
                                   reached_lambda_30=last['fitted_lambda'] is not None and last['fitted_lambda']>=30,free_minus_penalized=difference,
                                   own_means={c:values[c]['mean'] for c in own_signs},
                                   same_sign_as_offline_matched_full=same_offline,
                                   same_sign_as_online_terminal=same_online,
                                   late_differences=[dict(update=r['update'],lambda_fit=r['fitted_lambda'],
                                       free_minus_penalized=r[key][rtt]['free']['mean']-r[key][rtt]['penalized']['mean']) for r in late]))
            same_structure=sign(differences[0])!=0 and sign(differences[0])==sign(differences[1])
            for row in checks:
                if row['rtt']==int(rtt) and row['critic']==critic:row['both_repeats_same_class_order']=same_structure
    passed=all(r['reached_lambda_30'] and r['both_repeats_same_class_order'] and r['same_sign_as_offline_matched_full'] and r['same_sign_as_online_terminal'] for r in checks)
    return dict(status='ASSESSED',pass_condition=passed,checks=checks,
                interpretation='Report final class structure for both critics and both sampler repeats, together with all lambda>=30 observations. E1 class-difference significance was not separately registered/tested; no post-hoc choice among J10/J40 class point contrasts is made. This reports sampler-consistent class order, without proving that order correct. Lambda is not a visibility certificate. Online and offline target/BN paths differ; matched offline full arm isolates the loss-mask comparison.')


def analyze(root):
    root=pathlib.Path(root);read=lambda n:json.loads((root/n).read_text())
    e1=read('results/e1-fixed-reference.json');refs=fixed_references(e1)
    online=read('results/snapshot-probe.json');matched=read('results/snapshot-probe-matched-own-grid.json')
    ga=gate_a_c(online['result']['records'],refs);gm=gate_a_c(matched['result']['records'],refs)
    own={}
    for seed in (3,4):
        path=root/'results'/('own-seed%d.json'%seed)
        if path.exists():
            own[seed]=json.loads(path.read_text());paired_own(own[seed])
    gb=gate_b(own,matched)
    lags=[]
    old={20:(1,2,2),50:(4,7,2),100:(7,10,2),200:(23,24,5)}
    for rtt,(first_old,peak_old,tolerance) in old.items():
        r=e1['by_rtt'][str(rtt)]['penalized'];zs=[b/s for b,s in zip(r['beta'],r['se'])]
        candidates=[k for k,z in enumerate(zs) if z<-1.96];first=candidates[0] if candidates else None;peak=min(range(len(zs)),key=lambda k:zs[k])
        lags.append(dict(rtt=rtt,first_nominal_negative=first,strongest_negative=peak,
                         old_first=first_old,old_peak=peak_old,tolerance=tolerance,
                         prediction_pass=first is not None and abs(first-first_old)<=tolerance and abs(peak-peak_old)<=tolerance,
                         significance='Nominal pointwise |z|>1.96 for registered descriptive lag prediction; not gate references or proof of first causal effect.'))
    final=online['result']['records'][-1]
    ledger={}
    for name in ('training-result','admission-result'):
        path=root/'collected/node0/audit'/(name+'.json')
        if path.exists():
            item=json.loads(path.read_text());assert item['final']['counters']['actor_updates']==0
            ledger[name]=dict(inserted=item['final']['inserted'],counters=item['final']['counters'],
                              wall_s=item['wall_s'],stop_reason=item['stop_reason'])
    closure_path=root/'results/final-learner-closure.json'
    closure=json.loads(closure_path.read_text()) if closure_path.exists() else None
    support=None
    if (root/'results/response-alignment.json').exists() and (root/'results/snapshot-probe-by-actor.json').exists():
        aligned=read('results/response-alignment.json');actor_query=read('results/snapshot-probe-by-actor.json')['result']
        assert actor_query['probe_birth']==matched['result']['probe_birth']
        actor_last=actor_query['records'][-1];dominant=[]
        for rtt in (100,200):
            members=[a for a in aligned['actors'] if a['rtt']==rtt]
            dominant_actor=max(members,key=lambda a:(a['classes']['free']['full'] or {}).get('n',0))
            actor=dominant_actor['actor'];cl=dominant_actor['classes']['free']
            n=sum((a['classes']['free']['full'] or {}).get('n',0) for a in members)
            probe_total=sum(a['classes']['free']['probe_weight'] for a in members)
            dominant.append(dict(rtt=rtt,actor=actor,E1_free_rows=cl['full']['n'],
                                 E1_free_fraction=cl['full']['n']/n,probe_free_rows=cl['probe_weight'],
                                 probe_free_fraction=cl['probe_weight']/probe_total,E1=cl,
                                 matched_endpoint_online_Q1=actor_last['dq1_by_actor_class'][str(actor)]['free'],
                                 matched_endpoint_online_Q2=actor_last['dq2_by_actor_class'][str(actor)]['free']))
        support=dict(status='POST_HOC_DIAGNOSTIC_NOT_GATE_REPLACEMENT',dominant_free_actors=dominant,
                     probe_weighted_E1_sensitivity=aligned['probe_weighted_sensitivity'],
                     interpretation='Original pooled E1 is observation/innovation-variance weighted, whereas retrospective critic probes approximately balance actors within class. The dominant positive-E1 actors have positive critic gradients despite negative pooled critic means. This identifies a composition problem in the proposed comparison, without proving all states correct or allowing post-hoc actor release. The innovation-versus-persistent-action estimand difference also remains.')
    complete=len(own)==2 and closure is not None and closure['status']=='PASS_FINAL_LEARNER_CLOSURE'
    return dict(status='L2A_COMPLETE_NO_ACTOR_RELEASE' if complete else 'L2A_COLLECTION_COMPLETE_OFFLINE_PARTIAL',
                decision='WITHHOLD_L2B_DIRECTION_REVIEW',automatic_actor_release=False,
                online_ledger=ledger,final_learner_closure_status=closure['status'] if closure else 'NOT_YET_CHECKED',
                E1_fixed_references=refs,online_gate_a_c=ga,matched_probe_robustness_gate_a_c=gm,own_group_gate_b=gb,
                literal_gate_pass=ga['a_last_five_pass'] and ga['c_last_five_pass'] and gb['pass_condition'],
                P1=lags,P2_R200_free_J10_positive=next(r for r in refs if r['rtt']==200 and r['cls']=='free' and r['J']==10),
                P3_terminal_R200=dict(lambda_fit=final['fitted_lambda'],dq1=final['dq1_by_rtt_class']['200'],dq2=final['dq2_by_rtt_class']['200']),
                completed_sampler_repeats=sorted(own),one_online_training_seed=True,post_hoc_support_alignment=support,
                limitations=[
                    'No actor learning or frozen performance/CPU evaluation in L2A; no new accepted base, M1 remains unmet.',
                    'J10 innovation response and persistent-action Q derivative are different estimands. Their disagreement rejects the registered operational gate, not uniquely critic correctness.',
                    'Exploration latent offsets are logged; all formal rows have no clipping. Actor weights/Adam frozen but BN evolves, so the full policy is not frozen.',
                    'HAC120 and Holm16 are finite-sample approximations on six connections per RTT, one source seed. Cross-connection shared BN and nonstationarity remain limitations.',
                    'Final replay retains only last300k of514075 rows; startup states are evicted. Early snapshots queried on final replay are retrospective, not contemporaneous validation.',
                    'Offline paired loss intervention composes historical target duplicates deterministically and feeds paired Gaussian smoothing; it is not bit-exact online replay.',
                    'Both offline sampler repeats use the same source model and replay; not independent training seeds.',
                    'No complete TCP/history/exploration/TF queue resume or actor-release token was implemented.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=pathlib.Path(args.output);assert not output.exists(),'Preserve prior analyses; choose a new output path'
    output.write_text(json.dumps(analyze(args.evidence),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
