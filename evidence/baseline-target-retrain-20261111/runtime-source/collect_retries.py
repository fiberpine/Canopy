"""Collect preserved startup repairs and form an explicitly labeled72-run view."""
import copy
import json
import control as c
import score

original=c.evaluation_status()
assert all(h['finished'] for h in original)
c.save('confirm-original-final.json',original)
score.assess('confirm-original-final.json','health-original-execution')
plan=json.loads((c.HERE/'zero-feedback-retry-plan.json').read_bytes())
canonical=copy.deepcopy(original);proof=[]
for e in plan:
    body=r'''
import pathlib,json
f=pathlib.Path(ROOT)/'frameworks/baseline';p=f/'startup-retry-result.json';assert p.exists()
z=json.loads(p.read_text());assert len(z)==1
print(json.dumps(dict(node=NODE,rows=z)))
'''
    out=c.rpc(e['node'],body,45);r=out['rows'][0]
    assert r['job']==e['job'] and r['status']=='COMPLETED'
    host=next(h for h in canonical if h['node']==e['node'])
    old=next(x for x in host['rows'] if x['job']['run']==r['job']['original_run'])
    assert old['status']=='EXECUTION_ERROR'
    host['rows'][host['rows'].index(old)]=r
    proof.append(dict(node=e['node'],original=old,retry=r,eligibility=e['proof']))
c.save('zero-feedback-retry-results.json',proof)
c.save('confirm-with-startup-retries.json',canonical)
score.assess('confirm-with-startup-retries.json')
