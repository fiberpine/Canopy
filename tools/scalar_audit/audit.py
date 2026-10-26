"""Local evidence audit only. No SSH, TCP, training, or model modification."""
import ast, hashlib, importlib.util, json, math, pathlib, statistics, types
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path(__file__).resolve().parents[2]
VENDOR=ROOT/'frameworks/baseline/code/orca'
FRAME=ROOT/'frameworks/canopy_single'
RUN=ROOT/'evidence/scalar-20261025'
SMALL=RUN

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_method(path, class_name, method):
    """Execute only the actual pure computation method, without import side effects."""
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name)
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==method)
    namespace={'np':np,'math':math,'sleep':lambda _:None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[method]

def feature_fixture(rate,cwnd,samples,reference=None):
    """Synthetic input to real envwrapper computation; not a TCP validation."""
    raw=[20.,rate,samples,.02,50.,cwnd,1.2*rate,0.,20.,2147483647.,cwnd,0.,cwnd,1460.,20.]
    wire=('1 '+' '.join(map(str,raw))+'\0').encode()
    obj=types.SimpleNamespace(params=types.SimpleNamespace(dict={'input_dim':15,'delay_margin_coef':1.25}),
        shrmem_r=types.SimpleNamespace(read=lambda:wire),prev_rid=99999,use_normalizer=False,
        local_counter=0,max_bw=0. if reference is None else reference,max_cwnd=0.,max_smp=0.,min_del=9999999.)
    fn=source_method(VENDOR/'rl-module/envwrapper.py','TCP_Env_Wrapper','get_state')
    state,delay,reward,ok=fn(obj,evaluation=True)
    assert ok and np.isfinite(state).all() and math.isfinite(reward)
    return dict(rate_bytes_per_s=rate,cwnd=cwnd,synthetic_capacity_mbps=24.,
        synthetic_utilization=rate*8/24e6,state=state.tolist(),reward=float(reward),observed_peak=obj.max_bw,
        scope='Pure original method on synthetic input. No TCP or trained policy claim.')

def reward_alias():
    low=feature_fixture(100000.,4.,1.);high=feature_fixture(2500000.,100.,25.)
    assert np.array_equal(low['state'],high['state']) and low['reward']==high['reward']==1.
    # Diagnostic only: supplying a common observed reference removes this alias.
    # This does not implement an oracle, reward patch, or valid baseline.
    low_ref=feature_fixture(100000.,4.,1.,reference=3000000.)
    high_ref=feature_fixture(2500000.,100.,25.,reference=3000000.)
    return dict(status='PASS_SOURCE_FORMULA_COUNTEREXAMPLE',low=low,high=high,
        equal_70_history=bool(np.array_equal(np.tile(low['state'],10),np.tile(high['state'],10))),
        common_reference_diagnostic=[low_ref,high_ref],
        interpretation='Own observed peak can make low and high absolute throughput yield identical normalized state and reward. This is inherited source semantics, not a newly introduced arithmetic bug. It does not prove a unique training root cause or global impossibility.')

def evaluation_analysis():
    fields=['utilization','departure_mbps','reward_mean','queue_mean_ms','queue_p95_ms','queue_p99_ms',
            'drop_fraction','max_bw_mean','throughput_term_mean','delay_term_mean',
            'action_no_noise_mean','action_no_noise_std','min_saturation','max_saturation']
    by={};sources={}
    for arm in ['orca_control','orca_moving']:
        p=SMALL/'audit'/('evaluation-'+arm+'.json');sources[str(p)]=sha(p)
        rows=json.loads(p.read_text());assert len(rows)==72
        by[arm]={}
        for row in rows:
            assert row['result']['status']=='COMPLETED'
            assert row['result']['policy_sha256_before']==row['result']['policy_sha256_after']
            for w in row['windows']:
                assert w['complete_window'] and w['window'] in ['full','trim10']
                by[arm][(w['condition'],row['result']['cfg']['seed'],w['window'])]={k:float(w[k]) for k in fields}
    assert set(by['orca_control'])==set(by['orca_moving'])
    paired=[]
    for (condition,seed,window),control in by['orca_control'].items():
        candidate=by['orca_moving'][(condition,seed,window)]
        for lower_name,lower,higher_name,higher in [('orca_control',control,'orca_moving',candidate),('orca_moving',candidate,'orca_control',control)]:
            # Exact descriptive domination in this observed pair; no inference or noise margin.
            dominated=lower['utilization']<higher['utilization'] and all(lower[k]>=higher[k] for k in ['queue_mean_ms','queue_p95_ms','queue_p99_ms','drop_fraction'])
            if dominated and lower['reward_mean']>higher['reward_mean']:
                paired.append(dict(condition=condition,evaluation_seed=seed,window=window,
                    lower_quality_arm=lower_name,lower_quality=lower,higher_quality_arm=higher_name,higher_quality=higher))
    summaries={};aggregate_flips=[]
    for window in ['full','trim10']:
        for arm in by:
            groups={}
            for (c,seed,w),row in by[arm].items():
                if w==window:groups.setdefault(c,[]).append(row)
            summaries[arm+'_'+window]={c:{k:statistics.mean(x[k] for x in vs) for k in fields} for c,vs in groups.items()}
        for c,a in summaries['orca_control_'+window].items():
            b=summaries['orca_moving_'+window][c]
            if (a['utilization']-b['utilization'])*(a['reward_mean']-b['reward_mean'])<0 and abs(a['utilization']-b['utilization'])>.05:
                aggregate_flips.append(dict(window=window,condition=c,control=a,candidate=b))
    return dict(sources=sources,runs=144,windows=288,training_seeds_per_arm=1,freeze_repeats_per_condition=2,
        dominated_but_higher_reward_observed_pairs=paired,aggregate_throughput_reward_reversals=aggregate_flips,
        condition_means=summaries,
        limitation='Seed denotes frozen repetition, not training seed. Pairs are independent connections with paired cfg seeds, not identical TCP randomness. reward_mean is decision-weighted; it is not time-integrated utility, true Q, or a causal counterfactual. Multiplying mean reward by mean peak would not reconstruct mean unnormalized utility.')

def protocol_analysis():
    cfg=json.loads((SMALL/'configs/registration-formal.json').read_text())
    upstream=json.loads((VENDOR/'params_base_eval.json').read_text())
    arm=cfg['arms']['orca_control'];moving=cfg['arms']['orca_moving']
    identity=json.loads((SMALL/'audit/paired-step0-identity.json').read_text())
    actual=identity['models'][0]['tensor_sha256']
    scopes={name:sorted(k for k in actual if k.startswith(name+'/') and '/Adam' not in k) for name in ['actor','critic','critic2','target_actor','target_critic','target_critic2']}
    assert len(scopes['critic'])==len(scopes['critic2'])==6
    assert not any('/dense_' in k for keys in scopes.values() for k in keys)
    # Both completed arms used an insertion-count credit, not a physical-time credit.
    outcomes={name:json.loads((SMALL/'audit'/('outcome-formal-'+name+'.json')).read_text()) for name in ['orca_control','orca_moving']}
    numerical={name:x['result']['final'] for name,x in outcomes.items()}
    comparisons={
      'replay_capacity':dict(upstream=upstream['memsize'],actual=arm['memsize']),
      'replay_sampling':dict(upstream='Contiguous sequence of up to512 rows from circular buffer',actual=arm['sampler']),
      'warmup':dict(upstream='d5: replay ptr>200 or full',actual=dict(total=arm['warmup'],per_actor=arm['minimum_enqueued_per_actor'])),
      'optimizer_schedule':dict(upstream='d5 sleeps update_delay=5ms per iteration; no fixed insertion credit',actual=dict(update_per_inserted_transition=arm['update_ratio'])),
      'critic_adam':dict(upstream='One shared optimizer',actual='Two independent optimizers'),
      'target_BN':dict(upstream='Target actor receives common is_training=True in training feed',actual='Explicit moving statistics with copied/Polyak moving moments'),
      'actor_loss_forward':dict(upstream='Batch BN under training feed',control=arm['orca_actor_loss_moving_stats'],candidate=moving['orca_actor_loss_moving_stats']),
      'startup':dict(upstream='Wait for slow-start helper exit; auxiliary cwnd writes',actual='First valid TCP feedback; no pre-policy assistance or natural slow-start completion gate'),
      'deployment':dict(upstream='Shared distributed variables; public launcher protocol',actual='Training remote learner Session; frozen evaluation local Session with strict state IDs'),
      'network_and_reward':dict(upstream='70history state,1action,gamma=.995,window multiplier4**a,self-peak normalized reward',actual='Same nominal network and original envwrapper reward, not V28 interval-rate measurement')}
    matched={k:dict(upstream=upstream[k],actual=v) for k,v in dict(batch_size=512,h1_shape=256,h2_shape=256,gamma=.995,lr_a=.0001,lr_c=.001,tau=.001,stddev=arm['exploration_std']).items()}
    return dict(comparisons=comparisons,matched_hyperparameters=matched,complete_pair_initial_identity=identity['all_variables_equal'],
        tensor_scope_evidence=scopes,no_separate_random_actor_Q_head=True,actual_training=numerical,
        attribution='A single-flag fixed-exposure treatment comparison; changed data/update counts are downstream effects, not additional configured treatments. It does not isolate a direct gradient effect at identical data/update count. One seed and source-adapted protocols cannot establish a failure of exact upstream Orca.')

def bias_analysis():
    return json.loads((RUN/'sparse-bias-snapshots.json').read_text())


def main(output):
    files=[VENDOR/'rl-module/agent.py',VENDOR/'rl-module/envwrapper.py',VENDOR/'rl-module/utils.py',VENDOR/'rl-module/d5.py',
           VENDOR/'params_base_eval.json',FRAME/'distributed.py',FRAME/'orca_backend.py',FRAME/'orca_moving_actor_loss.py',FRAME/'native/src/orca-server-normal-start.cc']
    out=dict(status='COMPLETE_LOCAL_ONLY',cloud_connections=0,new_TCP_runs=0,new_training_runs=0,
        sources={str(p):sha(p) for p in files},protocol=protocol_analysis(),reward_formula=reward_alias(),
        empirical=evaluation_analysis(),training_bias=bias_analysis(),
        local_tensorflow_module_available=importlib.util.find_spec('tensorflow') is not None,
        runtime_checkpoint_gradient_audit='BLOCKED_BY_ENVIRONMENT: current scalar weights/replay remain cloud. Compatible TF1 must be checked in the execution environment. No gradient or factual-Q result fabricated.',
        priority='Inspect same-state critic action-direction and data support on existing scalar checkpoints before selecting a reward/exploration/startup patch. Do not repeat long training.',
        candidate_hypothesis='Self-referential reward scale plus near-absent positive exploration after early negative bias can sustain underutilization. Immediate RL handoff before natural CUBIC slow start is a separate untested contributor, not an identified sole cause.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=out['status'],runs=out['empirical']['runs'],windows=out['empirical']['windows'],
        observed_dominated_higher_reward_pairs=len(out['empirical']['dominated_but_higher_reward_observed_pairs']),
        reward_alias=out['reward_formula']['status'],output=str(output))))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,default=ROOT/'diagnostics/scalar-offline-audit.json');main(p.parse_args().output)
