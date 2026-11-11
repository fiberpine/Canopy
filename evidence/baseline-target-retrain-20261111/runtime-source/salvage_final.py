"""Evaluate a fully persisted endpoint without erasing the teardown failure."""
import json
import shutil
import time
import control as c

c.close()
shutil.copy2(c.HERE/'closure.json',c.HERE/'pre-evaluation-resource-check.json')
body=r'''
import pathlib,json,sys
f=pathlib.Path(ROOT)/'frameworks/baseline';sys.path.insert(0,str(f));import node_control as node
print(json.dumps(dict(node=NODE,actors=node.summarize('targetbn-base-seed3'))))
'''
parts=c.parallel(lambda n:c.rpc(n,body,120),range(1,5))
c.save('postfailure-actor-summaries.json',parts)
actors=[a for p in parts for a in p['actors']]
assert len(actors)==24 and not any(a['mapping_errors'] for a in actors)
body=r'''
import pathlib,json
r=pathlib.Path(ROOT);f=r/'frameworks/baseline';run='targetbn-base-seed3'
result=json.loads((f/'metrics'/(run+'-cluster-result.json')).read_text())
assert result['status']=='FAILED' and result['reason'].endswith('Owned resource cleanup verification failed')
assert result['stop_reason']=='SAMPLE_BUDGET_COMPLETE'
final=json.loads((f/'logs'/run/'result.json').read_text())
assert final['status']=='COMPLETED' and final['inserted']>=2400000
cfg=json.loads((f/'configs'/(run+'.json')).read_text())
expected=int((final['inserted']-cfg['warmup'])*cfg['update_ratio'])
assert all(final['counters'][k]==expected for k in ('critic_iterations','actor_updates','target_updates'))
assert final['counters']['critic_optimizer_ops']==2*expected
coverage=json.loads((f/'checkpoints'/run/'sample-coverage.json').read_text())
assert len(result['producer_records'])==24
for a in ACTORS:
 p=a['producer'];i=a['index']
 assert p==result['producer_records'][str(i)]
 assert not p['errors'] and p['generated']==p['enqueued']==coverage['inserted_by_actor'][i]==a['valid_rows']
 assert a['result']['status']=='COMPLETED' or a['result']['reason']=='Owned resource cleanup verification failed'
assert sum(coverage['inserted_by_actor'])==final['inserted']
events=[json.loads(line) for line in (f/'logs'/run/'checkpoint-events.jsonl').read_text().splitlines()]
assert events[-1]['kind']=='final' and events[-1]['inserted']==final['inserted']
print(json.dumps(dict(result=result,final=final,actors=ACTORS,coverage=coverage,config=cfg,checkpoint_events=events,selected_checkpoint=str(f/'checkpoints'/run/('step-%d'%final['inserted'])),eligibility='PERSISTED_ENDPOINT_DIAGNOSTIC_ONLY_RUN_REMAINS_FAILED')))
'''
info=c.rpc(0,body,90,ACTORS=actors)
c.save('training-complete.json',info)
c.save('endpoint-evaluation-amendment.json',dict(epoch=time.time(),run_status_unchanged='FAILED',
    reason='Two actors failed3s teardown. Full learner drain and final checkpoint persisted. All24 ledgers and mappings independently verified; all five resource checks now PASS.',
    evaluation='Same final endpoint, same72 jobs and health gates. Independent diagnostic of learned policy, not a repaired successful training run.',
    new_training_samples=0,models_restarted=0))
c.export_final()
c.launch_evaluation()
c.save('phase-state.json',dict(status='FROZEN_EVALUATION_RUNNING_FAILED_RUN_ENDPOINT',epoch=time.time()))
