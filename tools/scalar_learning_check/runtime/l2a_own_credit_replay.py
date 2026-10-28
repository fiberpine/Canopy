"""Bounded historical-data critic-loss interventions. No TCP or checkpoint writes.

All arms retain the full balanced batch, R200 rows, actor/BN path and denominator.
External paired smoothing draws replace TF stateful RNG. Historical overlapping
target assignments are composed in a fixed order (critic2 twice), identically in
all arms. This controlled operator is not historical bit-exact replay.
"""
import os
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
import numpy as np


def loss_weights(rtts, arm):
    rtts = np.asarray(rtts)
    if arm in ('full', 'full_repeat', 'stock_full'):
        return np.ones((len(rtts), 1), np.float32)
    if arm in ('own_only','own100','own200'):
        return (rtts == (100 if arm=='own100' else 200)).astype(np.float32).reshape(-1,1)
    removed = {'mask_short': (20, 50), 'mask_mid': (50, 100)}[arm]
    return (~np.isin(rtts, removed)).astype(np.float32).reshape(-1, 1)


def balanced_indices(pools, call, rng):
    base, rem = divmod(512, len(pools))
    extra = {(call * rem + j) % len(pools) for j in range(rem)}
    return np.asarray([j for i, pool in enumerate(pools)
                       for j in rng.sample(pool, base + int(i in extra))], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('pilot', 'formal', 'own', 'l2a'), required=True)
    ap.add_argument('--paired-reference')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--updates', type=int, default=8000)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = pathlib.Path(args.output)
    assert not output.exists()
    source = pathlib.Path(os.environ['CANOPY_FIRST_SOURCE'])
    framework = pathlib.Path(os.environ['CANOPY_FIRST_FRAMEWORK'])
    checkpoint = pathlib.Path(os.environ['CANOPY_FIRST_CHECKPOINT_DIR']) / 'step-0'
    sys.path[:0] = [str(framework), str(framework/'code/runtime/_deps')]
    import tensorflow as tf
    from orca_backend import create
    cfg = json.loads((source/'registration.json').read_text())['training_config']
    assert cfg['independent_critic_adam_v34'] and cfg['LOSS_TYPE'] == 'MSE'
    tf.set_random_seed(cfg['seed'])
    agent = create(framework, cfg, 'learner')
    agent.build_learn()
    original = list(tf.global_variables())
    saver = tf.train.Saver(original)
    weight_ph = tf.placeholder(tf.float32, [None, 1], name='diagnostic_loss_weight')
    losses = [tf.reduce_mean(tf.square(agent._v34_critic_target-pred) * weight_ph)
              for pred in (agent.critic_out, agent.critic_out2)]
    updates = [opt.minimize(loss, var_list=params) for opt, loss, params in zip(
        agent._v34_critic_optimizers, losses, agent._v34_critic_parameters)]
    with tf.control_dependencies(updates):
        weighted_train = tf.assign_add(agent.global_step, 2, use_locking=True)
    params = agent._v34_critic_parameters[0]
    gradient_difference = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(x-y)) for x, y in zip(
        tf.gradients(agent.c_loss, params), tf.gradients(losses[0], params))]))
    bn_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
              if 'actor' in op.name and 'target' not in op.name]
    assert bn_ops
    # Preserve logical legacy multiplicity, remove racing writes to the same variable.
    pairs = list(zip(agent.target_critic.train_var(), agent.critic.train_var()))
    pairs += list(zip(agent.target_critic2.train_var(), agent.critic2.train_var()))
    grouped = {}
    for target, current in pairs:
        if target.op.name in grouped:
            old = grouped[target.op.name]
            assert old[1] is current
            old[2] += 1
        else:
            grouped[target.op.name] = [target, current, 1]
    assignments = []
    for target, current, count in grouped.values():
        expression = target
        for unused in range(count):
            expression = .001 * current + .999 * expression
        assignments.append(tf.assign(target, expression))
    target_train = tf.group(*(assignments + agent.target_actor_update_op))
    g1, g2 = tf.gradients(agent.critic_out, agent.action)[0], tf.gradients(agent.critic_out2, agent.action)[0]
    assert original == list(tf.global_variables()), 'Diagnostic must reuse all original optimizer slots'
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
    agent.assign_sess(sess)
    reader = tf.train.NewCheckpointReader(str(checkpoint))
    assert {v.op.name:list(v.shape) for v in original} == reader.get_variable_to_shape_map()
    initial = [reader.get_tensor(v.op.name) for v in original]
    actor_vars = agent.actor.train_var()
    actor_optimizer = [agent.actor_optimizer.get_slot(v, name) for v in actor_vars
                       for name in agent.actor_optimizer.get_slot_names()]
    actor_optimizer += list(agent.actor_optimizer._get_beta_accumulators())
    bn_vars = [v for v in original if v.op.name.startswith('actor/')
               and ('moving_mean' in v.op.name or 'moving_variance' in v.op.name)]
    replay = np.load(str(source/'replay.npz'))
    if args.mode == 'l2a':
        order=np.argsort(replay['birth'],kind='stable')
        assert len(np.unique(replay['birth']))==len(order)
        data={k:replay[k][order] for k in ('s0','a','reward','s1','terminal')}
        aid=replay['actor_ids'][order]
        pools=[list(np.flatnonzero(aid==i)) for i in range(24)]
        assert min(map(len,pools))>=512
        rtt_of=np.asarray([c['base_rtt_ms'] for c in cfg['conditions']])
        padded=np.any(np.all(data['s0'][:,:63].reshape(-1,9,7)==0,axis=2),axis=1)
        probe=[];FR=[]
        for pool in pools:
            assert all(np.array_equal(data['s1'][x],data['s0'][y]) and not data['terminal'][x].any() for x,y in zip(pool,pool[1:])), 'Replay contiguity break'
            rows=np.asarray(pool);ok=np.arange(len(rows)-200);ok=ok[~padded[rows[ok]]]
            for pen in (True,False):
                cand=ok[(data['s0'][rows[ok],69]<.999)==pen]
                if len(cand):
                    picks=cand[np.linspace(0,len(cand)-1,min(8,len(cand)),dtype=int)]
                    probe+=list(rows[picks]);FR += [data['reward'][rows[j:j+200]].ravel() for j in picks]
        FR=np.asarray(FR);probe=np.asarray(probe)
    else:
        coverage = json.loads((source/'first-update-coverage.json').read_text())
        n = int(coverage['inserted'])
        assert n == 27436 and np.array_equal(replay['birth'], np.arange(n))
        data = {k:replay[k] for k in ('s0','a','reward','s1','terminal')}
        aid = replay['actor_ids']
        pools = [list(np.flatnonzero(aid == i)) for i in range(24)]
        assert [len(p) for p in pools] == coverage['per_actor_inserted']
        rtt_of = np.asarray([c['base_rtt_ms'] for c in cfg['conditions']])
        probe = []
        for pool in pools:
            assert len(pool) >= 215
            assert all(np.array_equal(data['s1'][x],data['s0'][y]) and not data['terminal'][x].any() for x,y in zip(pool,pool[1:]))
            probe += list(np.asarray(pool)[np.linspace(0,len(pool)-200,16,dtype=int)])
        probe = np.asarray(probe)
    p0 = data['s0'][probe]
    prtt = rtt_of[aid[probe]]
    penalized = p0[:,69] < .999
    progress = output.with_suffix('.progress.jsonl')
    assert not progress.exists()
    def digest(values):
        h = hashlib.sha256()
        for value in values:
            h.update(value.tobytes())
        return h.hexdigest()
    def classes(values):
        return {str(r):{name:(dict(n=int(sel.sum()),mean=float(values[sel].mean()),
                                  positive=float(np.mean(values[sel]>0))) if sel.any() else None)
                       for name,sel in (('penalized',(prtt==r)&penalized),('free',(prtt==r)&~penalized))}
                for r in (20,50,100,200)}
    results = {}
    started = time.time()
    arms = (('full','own100','own200') if args.mode == 'l2a' else ('full','full_repeat','stock_full') if args.mode == 'pilot' else
            ('own_only',) if args.mode == 'own' else ('full','mask_short','mask_mid'))
    def fit_lambda(qmean,qzero,selected):
        import math
        k=np.arange(200);r=FR[selected].mean(axis=0);lo,hi=0.,199.
        def pred(lam):
            if lam<=0:S=(k==0).astype(float)
            else:
                pmf=np.exp(-lam+k*np.log(lam)-np.array([math.lgamma(int(j)+1) for j in k]))
                S=np.clip(1-np.r_[0.,np.cumsum(pmf)[:-1]],0,1)
            return float(np.dot(r,.995**k*S))+.995*qzero
        if pred(lo)>=qmean:return 0.
        if pred(hi)<=qmean:return None
        for unused in range(45):
            mid=(lo+hi)/2
            if pred(mid)<qmean:lo=mid
            else:hi=mid
        return (lo+hi)/2
    for arm in arms:
        saver.restore(sess,str(checkpoint))
        assert all(np.array_equal(x,y) for x,y in zip(initial,sess.run(original)))
        actor_before, opt_before = sess.run(actor_vars), sess.run(actor_optimizer)
        fixed_pi = sess.run(agent.actor_out,{agent.s0:p0,agent.is_training:False})
        sampler, noise_rng = random.Random(args.seed), np.random.RandomState(args.seed+7319)
        hashes = {name:hashlib.sha256() for name in ('batch','R200_batch','noise','target_actions')}
        records = []
        retained_rows, rtt_rows = 0, {str(r):0 for r in (20,50,100,200)}
        qzero=[None]
        def record(t):
            pi = sess.run(agent.actor_out,{agent.s0:p0,agent.is_training:False})
            fd = {agent.s0:p0,agent.action:pi,agent.is_training:False}
            q, d1, d2 = sess.run([agent.critic_out,g1,g2],fd)
            fixed_d1 = sess.run(g1,{agent.s0:p0,agent.action:fixed_pi,agent.is_training:False})
            ownmask=(prtt==100) if arm=='own100' else (prtt==200) if arm=='own200' else np.ones(len(prtt),bool)
            q_behavior=sess.run(agent.critic_out,{agent.s0:p0,agent.action:data['a'][probe],agent.is_training:False}).ravel()
            if qzero[0] is None:qzero[0]=float(q_behavior[ownmask].mean())
            row = dict(lambda_fit_scope=arm,q_behavior_own_mean=float(q_behavior[ownmask].mean()),fitted_lambda=fit_lambda(float(q_behavior[ownmask].mean()),qzero[0],ownmask) if args.mode=='l2a' else None,update=t,q1=float(q.mean()),dq1=classes(d1.ravel()),dq2=classes(d2.ravel()),
                       fixed_initial_action_dq1=classes(fixed_d1.ravel()),
                       BN_sha256=digest(sess.run(bn_vars)),actor_policy_sha256=digest([pi]),
                       policy_max_shift=float(np.max(np.abs(pi-fixed_pi))))
            assert np.isfinite(q).all() and np.isfinite(d1).all() and np.isfinite(d2).all()
            records.append(row)
            with progress.open('a') as stream:
                stream.write(json.dumps(dict(arm=arm,**row),allow_nan=False)+'\n')
        record(0)
        max_grad_error = 0.
        for t in range(args.updates):
            if t%100==0 and time.time()>=float(os.environ.get('CANOPY_OFFLINE_DEADLINE','inf')):raise RuntimeError('OFFLINE_ABSOLUTE_BUDGET; retain progress; later arms NOT_RUN')
            idx = balanced_indices(pools,t,sampler)
            rtts = rtt_of[aid[idx]]
            noise = np.clip(noise_rng.normal(0,.1,(512,1)),-.2,.2).astype(np.float32)
            target_pi = sess.run(agent.target_actor_out,{agent.s1:data['s1'][idx],agent.is_training:False})
            target_actions = np.clip(target_pi+noise,-1,1)
            weights = loss_weights(rtts,arm)
            retained_rows += int(weights.sum())
            for r in (20,50,100,200):
                rtt_rows[str(r)] += int(weights[rtts==r].sum())
            if arm!='own100':assert np.all(weights[rtts==200] == 1)
            fd = {agent.s0:data['s0'][idx],agent.action:data['a'][idx],agent.reward:data['reward'][idx],
                  agent.s1:data['s1'][idx],agent.terminal:data['terminal'][idx],agent.is_training:True,
                  agent.is_concrete:True,agent.s0_delta:np.zeros_like(data['s0'][idx]),
                  agent.target_actor_policy:target_actions,weight_ph:weights}
            if args.mode == 'pilot':
                max_grad_error = max(max_grad_error,float(sess.run(gradient_difference,fd)))
            sess.run(agent.critic_train_op if arm=='stock_full' else weighted_train,fd)
            sess.run(bn_ops,fd)
            sess.run(target_train)
            for key,value in (('batch',idx),('R200_batch',idx[rtts==200]),('noise',noise),('target_actions',target_actions)):
                hashes[key].update(value.tobytes())
            if t+1 in (10,100,300,args.updates) or (t+1)%(2000 if args.mode=='l2a' else 500)==0:
                record(t+1)
        final = sess.run(original)
        assert all(np.isfinite(x).all() for x in final)
        assert all(np.array_equal(x,y) for x,y in zip(actor_before,sess.run(actor_vars)))
        assert all(np.array_equal(x,y) for x,y in zip(opt_before,sess.run(actor_optimizer)))
        assert int(sess.run(agent.global_step)) == 2*args.updates
        results[arm] = dict(records=records,paired_hashes={k:h.hexdigest() for k,h in hashes.items()},
                           final_parameter_sha256=digest(final),initial_all96_verified=True,
                           actor_weights_and_optimizer_unchanged=True,critic_updates=args.updates,
                           critic_optimizer_ops=2*args.updates,actor_updates=0,target_sets=args.updates,
                           retained_rows=retained_rows,retained_rows_by_RTT=rtt_rows,
                           maximum_allone_stock_gradient_error=max_grad_error)
    first = results[arms[0]]
    if args.mode == 'own':
        assert args.paired_reference, 'Own-only intervention requires completed matched full arm'
        reference = json.loads(pathlib.Path(args.paired_reference).read_text())
        assert reference['sampler_seed'] == args.seed and reference['updates'] == args.updates
        baseline = reference['arms']['full']
        assert first['paired_hashes'] == baseline['paired_hashes']
        assert [r['BN_sha256'] for r in first['records']] == [r['BN_sha256'] for r in baseline['records']]
        assert [r['actor_policy_sha256'] for r in first['records']] == [r['actor_policy_sha256'] for r in baseline['records']]
    for arm in arms[1:]:
        other = results[arm]
        assert first['paired_hashes'] == other['paired_hashes'], 'Unpaired inputs or target-policy path'
        assert [r['BN_sha256'] for r in first['records']] == [r['BN_sha256'] for r in other['records']]
    if args.mode == 'pilot':
        assert first['final_parameter_sha256'] == results['full_repeat']['final_parameter_sha256'], 'A/A not bit-exact'
        assert first['final_parameter_sha256'] == results['stock_full']['final_parameter_sha256'], 'All-one weighted path differs from stock'
        assert max(r['maximum_allone_stock_gradient_error'] for r in results.values()) <= 1e-6
    result = dict(status='PASS_CONTROLLED_OFFLINE',mode=args.mode,sampler_seed=args.seed,updates=args.updates,
                  wall_s=time.time()-started,arms=results,all_pairing_checks_pass=True,
                  target_logical_multiplicity={k:v[2] for k,v in grouped.items()},
                  smoothing='Same externally generated Gaussian(.1), clip.2; not historical TF RNG sequence',
                  target_order='Fixed composition of historical duplicate assignments; critic2 twice per set',
                  new_TCP_runs=0,source_checkpoints_written=0,model_variables=len(original),replay_rows=len(aid),retained_rows_by_actor=list(map(len,pools)),probe_actor=aid[probe].tolist(),probe_class_counts=classes(np.zeros(len(probe))),
                  limitations='Static one-source-seed replay. Masking changes optimization and bootstrap labels; '
                              'a difference shows controlled group-loss contribution, not unique provenance or network benefit.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=1,allow_nan=False))
    sess.close()
    print(json.dumps(dict(status=result['status'],mode=args.mode,seed=args.seed,wall_s=result['wall_s'])),flush=True)


if __name__ == '__main__':
    main()
