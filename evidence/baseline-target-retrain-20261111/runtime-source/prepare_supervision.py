"""Prepare an observer-only handover; no cloud access and no learning edits."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'innovations/baseline-target-audit-20261111/frameworks/baseline/cluster_control.py'
text=SOURCE.read_text()
begin=text.index(" try:\n  node.start(cfg,'learner',0)")
marker="  while True:\n   statuses=get_status();local=node.status(run);now=time.time();entries=[x for s in statuses for x in s['entries']]"
end=text.index(marker,begin)
prefix=''' # Attach preflight/activation errors must never stop the existing training.
 try:
  local=node.status(run);statuses=get_status()
  assert len(local['entries'])==1 and local['entries'][0]['alive']
  entries=[entry for host in statuses for entry in host['entries']]
  assert len(entries)==cfg['num_actors'] and all(entry['alive'] for entry in entries)
  all_ready=cfg['_original_all_ready_epoch'];start=cfg['_original_start_epoch']
  node.dump(R/'audit'/(run+'-handover-ready.json'),dict(learner=local['entries'][0]['pid'],actors={entry['index']:entry['pid'] for entry in entries},epoch=time.time()))
 except BaseException:
  (R/'audit'/(run+'-handover-preflight-error.txt')).write_text(traceback.format_exc());os._exit(2)
 until=time.time()+30
 while not (R/'audit'/(run+'-handover-activate.json')).exists():
  if time.time()>until or (R/'audit'/(run+'-handover-abort.json')).exists():os._exit(3)
  time.sleep(.1)
 try:
'''
new=text[:begin]+prefix+text[end:]
new=new.replace(" node.dump(R/'configs'/(run+'.json'),cfg);start=time.time();samples=[];results={};failure=None;stop_reason=None;all_ready=None;debt_since=None;bad_since={}",
                " start=time.time();samples=[];results={};failure=None;stop_reason=None;all_ready=None;debt_since=None;bad_since={}")
oldline=" sample_stream=(R/'logs'/(run+'-cluster-samples.jsonl')).open('w',buffering=262144)"
newline=" sample_path=R/'logs'/(run+'-cluster-samples.jsonl');samples=[json.loads(line) for line in sample_path.read_text().splitlines()]\n sample_stream=sample_path.open('a',buffering=262144)"
assert oldline in new;new=new.replace(oldline,newline)
tree=ast.parse(new,feature_version=(3,6))
execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='node' and n.func.attr=='start' for n in ast.walk(execute))
assert text[end:]==new[new.index(marker):], 'Monitoring, stop, drain, mapping checks and summary must remain identical'
(HERE/'attach_cluster_control.py').write_text(new,encoding='utf-8')
watch='''"""Independent data-phase watchdog; no model session or inference calls."""
import pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import node_control as node
run=sys.argv[1];deadline=float(sys.argv[2])
while time.time()<deadline:
 status=node.status(run)
 if status['entries'] and all(not entry['alive'] for entry in status['entries']):sys.exit(0)
 time.sleep(5)
node.signal_stop(run,'AMENDED_DATA_PHASE_DEADLINE')
time.sleep(30)
node.cleanup(run)
'''
ast.parse(watch,feature_version=(3,6));(HERE/'control_hard_watch.py').write_text(watch,encoding='utf-8')
proof=dict(status='PASS_ATTACH_SOURCE_CONTRACT',original_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           attach_sha256=hashlib.sha256(new.encode()).hexdigest(),new_model_or_actor_start_calls=0,
           original_monitor_drain_mapping_tail_byte_identical=True,old_sample_history_appended=True,
           no_tensorflow_import_in_new_watch=True)
(HERE/'supervision-source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
