
import pathlib,json,time,hashlib
R=pathlib.Path(ROOT);F=R/'framework';p=F/'logs/base-reproduction-seed3-2400k-cluster-samples.jsonl';cfg=json.loads((R/'registration.json').read_text())['training_config'];rows=[];streak=None;guard=[]
with p.open() as stream:
 for line in stream:
  r=json.loads(line);learner=r['learner']['entries'][0];latest=learner.get('training',[]);latest=latest[-1] if latest else None
  actors=[]
  for host in r['actors']:
   for e in host['entries']:
    actors.append(dict(index=e['index'],alive=e['alive'],recent=e.get('recent'),recent_network=e.get('recent_network'),producer=e.get('producer')))
  row=dict(epoch=r['epoch'],learner_latest=latest,debt=r.get('debt'),actors=actors);rows.append(row)
  if latest and int(latest['inserted'])>=cfg['collapse_guard_after_samples']:
   a=next(a for a in actors if a['index']==0);recent=a.get('recent') or {};network=a.get('recent_network') or {}
   bad=recent.get('min_saturation',0)>=.99 and recent.get('cwnd_mean',1e6)<=10 and network.get('utilization') is not None and network['utilization']<.25
   streak=(streak or r['epoch']) if bad else None
   guard.append(dict(epoch=r['epoch'],inserted=int(latest['inserted']),actor0=recent,network=network,registered_bad=bad,continuous_bad_s=r['epoch']-streak if streak is not None else None))
proof=dict(status='ORIGINAL_LATE_GUARD_TRIGGER_REPRODUCED_FROM_RECORDED_SAMPLES',condition=cfg['conditions'][0]['condition'],threshold_samples=cfg['collapse_guard_after_samples'],threshold_seconds=cfg['collapse_guard_seconds'],rule='Noiseless min_saturation>=.99 AND recent mean cwnd<=10 AND recent5s utilization<.25, uninterrupted after sample threshold',rows=guard,maximum_continuous_bad_s=max(g['continuous_bad_s'] or 0 for g in guard),raw_cluster_samples_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Projection of existing controller observations; no extra policy calls, optimizer or TCP experiment.')
assert proof['maximum_continuous_bad_s']>=cfg['collapse_guard_seconds'],proof
proof['analysis_source_sha256']=ANALYSIS_SHA
(R/'results/guard-trigger-proof.json').write_text(json.dumps(proof,indent=2));(R/'results/training-trajectory.json').write_text(json.dumps(rows))
print(json.dumps(dict(status=proof['status'],points=len(rows),maximum_continuous_bad_s=proof['maximum_continuous_bad_s'],last=guard[-1])))
