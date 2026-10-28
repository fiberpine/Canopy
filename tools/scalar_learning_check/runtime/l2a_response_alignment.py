"""Post-hoc support alignment, NumPy only. Does not replace registered full-run E1.

Compare full-connection vs retained-replay intervals, and class-specific actor
weights vs retrospective critic probe weights. No new data, TCP, TF, model writes
or gate selection. Probe-weighted standard errors assume independent actor
estimates; shared evolving BN makes this only a sensitivity diagnostic.
"""
import argparse,json,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import exploration_response as e


def compact(result):
    if result is None:return None
    return dict(n=result['samples'],beta={str(j):result['cumulative'][str(j)] for j in (10,40)},
                se={str(j):result['cumulative_se'][str(j)] for j in (10,40)})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('series-npz','replay','snapshot-probe','output'):ap.add_argument('--'+name,required=True)
    args=ap.parse_args();out=pathlib.Path(args.output);assert not out.exists()
    z=np.load(args.series_npz);rp=np.load(args.replay);probe=json.loads(pathlib.Path(args.snapshot_probe).read_text())['result']
    counts=np.bincount(rp['actor_ids'],minlength=len(rp['actor_inserted']));total=rp['actor_inserted']
    # Probe class layout is recoverable from birth IDs in the same final replay.
    where={int(b):k for k,b in enumerate(rp['birth'])};probe_counts={}
    for birth,actor in zip(probe['probe_birth'],probe['probe_actor']):
        index=where[birth];assert int(rp['actor_ids'][index])==actor
        cls='penalized' if rp['s0'][index,69]<.999 else 'free';key=(int(actor),cls);probe_counts[key]=probe_counts.get(key,0)+1
    actors=[];terminal_series={};full_series={};by_actor={}
    for actor in range(len(total)):
        m=z['series_id']==actor;o,r,c=z['o'][m],z['r'][m],z['cls'][m];assert len(o)==total[actor]
        rtt=int(np.unique(z['rtt'][m])[0]);start=int(total[actor]-counts[actor]);assert start>=0
        full=(o,r,c);term=(o[start:],r[start:],c[start:]);full_series.setdefault(rtt,[]).append(full);terminal_series.setdefault(rtt,[]).append(term)
        row=dict(actor=actor,rtt=rtt,full_rows=len(o),retained_rows=int(counts[actor]),excluded_prefix=start,classes={})
        for name,klass in [('free',e.FREE),('penalized',e.PENALIZED)]:
            f=compact(e.response([full],60,cls_value=klass));t=compact(e.response([term],60,cls_value=klass))
            row['classes'][name]=dict(full=f,terminal=t,probe_weight=probe_counts.get((actor,name),0));by_actor[(actor,name)]=row['classes'][name]
        actors.append(row)
    weighted=[]
    for rtt in sorted(terminal_series):
        for cls in ('free','penalized'):
            selected=[a for a in actors if a['rtt']==rtt and a['classes'][cls]['probe_weight']>0]
            for scope in ('full','terminal'):
                present=[a for a in selected if a['classes'][cls][scope] is not None]
                weights=np.asarray([a['classes'][cls]['probe_weight'] for a in present],float)
                if not len(weights):continue
                weights/=weights.sum();refs={}
                for J in ('10','40'):
                    b=np.asarray([a['classes'][cls][scope]['beta'][J] for a in present]);s=np.asarray([a['classes'][cls][scope]['se'][J] for a in present])
                    refs[J]=dict(beta=float(weights@b),independence_approx_se=float(np.sqrt(np.sum((weights*s)**2))))
                weighted.append(dict(rtt=rtt,cls=cls,scope=scope,actors=[a['actor'] for a in present],normalized_probe_weights=weights.tolist(),references=refs))
    result=dict(status='POST_HOC_SUPPORT_ALIGNMENT_COMPLETE',actors=actors,probe_weighted_sensitivity=weighted,
                terminal_by_rtt={str(k):e.split_response(v,60) for k,v in terminal_series.items()},
                full_rows=int(sum(total)),retained_rows=int(sum(counts)),
                registered_full_run_E1_unchanged=True,new_TCP_runs=0,TF_updates=0,
                limitations='Post-hoc support/weight sensitivity only. Does not reselect the gate or turn innovation responses into deterministic persistent-action Q effects. Actor-specific HAC and approximate weighted errors ignore cross-actor dependence through shared BN; six actors per RTT, one training seed.')
    out.write_text(json.dumps(result,indent=1,allow_nan=False));print(json.dumps(dict(status=result['status'],actors=len(actors))))


if __name__=='__main__':main()
