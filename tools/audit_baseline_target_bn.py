"""Read saved scalar policy/critic queries locally in TF1; no server or optimizer.

Queries preserve every variable. A separately labeled synthetic contract check
uses in-memory assignments only, never writing the source checkpoint/replay.
"""
import argparse
import collections
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def audit(args):
    os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = Path(args.framework)
    sys.path[:0] = [str(root), str(root/'code/runtime/_deps'), str(root/'code/canopy/rl-module')]
    d = importlib.import_module('distributed')
    tf, np = d.tf, d.np
    cfg = json.loads(Path(args.config).read_text())
    cfg['target_bn_consistent'] = args.mode == 'repaired'
    real_server, real_device = tf.train.Server, tf.device
    class LocalServer:
        target = ''
    def local_device(name):
        return real_device('/cpu:0' if isinstance(name,str) and name.startswith('/job:') else name)
    tf.train.Server = lambda *a, **k: LocalServer()
    tf.device = local_device
    try:
        a, sess, fifo, ph, enqueue, nph, dequeue, size = d.setup(cfg,'learner',0)
    finally:
        tf.train.Server, tf.device = real_server, real_device
    assert not any('/job:' in op.device for op in tf.get_default_graph().get_operations())
    variables = tf.global_variables()
    saver = tf.train.Saver(var_list=variables)
    cpfiles = [Path(args.checkpoint+s) for s in ('.index','.data-00000-of-00001')]
    files_before = {str(p):digest(p) for p in cpfiles+[Path(args.replay)]}
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,args.checkpoint)
    def values_hash():
        h = hashlib.sha256()
        for v,value in zip(variables,sess.run(variables)):
            h.update(v.name.encode());h.update(value.tobytes())
        return h.hexdigest()
    before = values_hash()
    with np.load(args.replay) as replay:
        s1=replay['s1'].copy();rewards=replay['reward'].copy();terminals=replay['terminal'].copy()
        actors=replay['actor_ids'].copy();birth=replay['birth'].copy()
    ids=sorted(set(map(int,actors)))
    assert ids==list(range(24)) and s1.shape[1]==70
    pools={i:np.flatnonzero(actors==i) for i in ids}
    focal=[]
    for i in ids:
        pool=pools[i][np.argsort(birth[pools[i]])]
        focal.extend(pool[np.linspace(0,len(pool)-1,4).astype(int)].tolist())
    focal=np.array(focal);count=len(focal);rng=np.random.RandomState(20261111)
    ops=[];seen=set();pending=[a.target_actor_out.op]
    while pending:
        op=pending.pop()
        if op in seen:continue
        seen.add(op)
        if op.type=='Tanh':ops.append(op)
        else:pending.extend(t.op for t in op.inputs)
    assert len(ops)==1
    logits=ops[0].inputs[0]
    actions=[];preacts=[];targets=[]
    for repeat in range(8):
        context=[]
        #512 rows exactly follow balanced_actor quotas;96 focal rows stay fixed.
        for i in ids:
            quota=22 if i in {(repeat*8+j)%24 for j in range(8)} else 21
            context.extend(rng.choice(pools[i],quota-4,replace=False).tolist())
        batch=np.concatenate([s1[focal],s1[np.array(context)]])
        assert len(batch)==512
        out,pre=sess.run([a.target_actor_out,logits],{a.s1:batch,a.is_training:True})
        out=out[:count];pre=pre[:count]
        # Feeding policy output bypasses smoothing RandomNormal entirely.
        fd={a.s1:s1[focal],a.target_actor_policy:out,a.reward:rewards[focal],
            a.terminal:terminals[focal],a.is_training:True}
        y1,y2=sess.run([a.y,a.y2],fd)
        actions.append(out[:,0]);preacts.append(pre[:,0]);targets.append(np.minimum(y1,y2)[:,0])
    actions=np.asarray(actions);preacts=np.asarray(preacts);targets=np.asarray(targets)
    after=values_hash();assert before==after,'Read-only query changed model variables'
    def stats(x):
        spread=np.ptp(x,axis=0)
        return dict(max_context_range=float(spread.max()),median_context_range=float(np.median(spread)),
                    p95_context_range=float(np.percentile(spread,95)),
                    median_context_std=float(np.median(np.std(x,axis=0))))
    byactor={str(i):dict(action=stats(actions[:,j*4:j*4+4]),logit=stats(preacts[:,j*4:j*4+4]),
                        td_target=stats(targets[:,j*4:j*4+4])) for j,i in enumerate(ids)}
    target_sets=[a.target_actor_update_op,a.target_critic_update_op,a.target_critic_update_op2]
    assignments=[]
    for group in target_sets:
        for entry in group:
            op=entry.op if hasattr(entry,'op') else entry
            assignments.append(dict(op=op.name,type=op.type,destination=op.inputs[0].op.name))
    counts=collections.Counter(row['destination'] for row in assignments)
    result=dict(mode=args.mode,shape=dict(replay_rows=len(s1),focal_rows=count,contexts=8,batch=512),
        query_variables_sha256_before=before,query_variables_sha256_after=after,
        action=stats(actions),logit=stats(preacts),td_target=stats(targets),by_actor=byactor,
        focal_actor_ids=actors[focal].tolist(),focal_birth=birth[focal].tolist(),
        focal_actions=actions.tolist(),focal_logits=preacts.tolist(),focal_td_targets=targets.tolist(),
        critic_variables=[v.name for v in a.critic.train_var()],critic2_variables=[v.name for v in a.critic2.train_var()],
        critic_scope_intersection=sorted(set(v.name for v in a.critic.train_var())&set(v.name for v in a.critic2.train_var())),
        target_assignments=assignments,repeated_target_destinations={k:v for k,v in counts.items() if v>1},
        variables=[dict(name=v.name,shape=v.shape.as_list()) for v in variables],
        new_optimizer_operations=0,new_TCP_flows=0,TF_Server_suppressed=True,
        scope='Retained single-connection replay only; not startup coverage or causal performance proof')
    if args.mode=='repaired':
        statsvars=[v for v in variables if v.name.startswith('target_actor/') and ('moving_mean' in v.name or 'moving_variance' in v.name)]
        assert len(statsvars)==4
        pairs=[]
        for target in statsvars:
            suffix='/bn'+target.op.name.split('/bn',1)[1]
            online,=[v for v in variables if v.name.startswith('abstract_actor/') and v.op.name.endswith(suffix)]
            pairs.append((online,target))
        # Explicit synthetic state assignment, isolated after immutable queries.
        for online,target in pairs:
            sess.run(tf.assign(online,np.full(online.shape.as_list(),1.25,np.float32)))
        sess.run(a.target_act_init_op)
        initial=max(float(np.max(np.abs(sess.run(t)-sess.run(o)))) for o,t in pairs)
        old=[sess.run(t) for o,t in pairs]
        for online,target in pairs:sess.run(tf.assign(online,np.full(online.shape.as_list(),2.5,np.float32)))
        sess.run(a.target_actor_update_op)
        error=max(float(np.max(np.abs(sess.run(t)-(.001*2.5+.999*x)))) for (o,t),x in zip(pairs,old))
        assert initial==0 and error<=1e-6
        result['synthetic_statistics_contract']=dict(initial_copy_error=initial,tau_error=error,pairs=len(pairs),
            in_memory_assignments_only=True,optimizer_operations=0,old_checkpoint_written=False)
        saver.restore(sess,args.checkpoint)
        assert values_hash()==before
    assert all(digest(Path(p))==h for p,h in files_before.items())
    result['source_files_unchanged']=files_before
    sess.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('framework','config','checkpoint','replay','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=['stock','repaired'],required=True)
    args=parser.parse_args()
    result=audit(args)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('mode','shape','action','logit','td_target')}))
