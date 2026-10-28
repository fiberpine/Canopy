"""Frozen endpoint only. Distinct controller, worker, phase and stop markers."""
import pathlib,sys,os,json,time,ast,copy,subprocess,concurrent.futures,traceback,random,io,tarfile,base64,hashlib
R=pathlib.Path(os.environ['CANOPY_REPRO_ROOT']);F=R/'framework';sys.path.insert(0,str(F));import node_control as node
M=node.load(R/'registration.json');A=node.load(R/'authorization.json');END=M['deadline_epoch'];BASE=pathlib.Path('@ARCHIVED_BASELINE@');OLD=pathlib.Path('@READONLY_RUNTIME@')
source=(R/'pipeline.py').read_text();tree=ast.parse(source);wanted={'execute','parallel','status','stage','config','evaljobs'}
defs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],**({'type_ignores':[]} if 'type_ignores' in ast.Module._fields else {}));exec(compile(defs,'endpoint-definitions','exec'))
T=node.load(F/'metrics/base-reproduction-seed3-2400k-cluster-result.json');assert T['final']['status']=='COMPLETED' and T['stop_reason'].startswith('PERSISTENT_LOW_WINDOW_LOW_UTIL_ACTOR') and T['final']['inserted']<2400000
assert time.time()<END-1000
worker=(R/'worker.py').read_text();assert worker.count("(R/'STOP_ALL').exists()")==1;worker=worker.replace("(R/'STOP_ALL').exists()","(R/'STOP_ENDPOINT').exists()")
def provision(i):
 return execute(i,"import pathlib,json,hashlib;R=pathlib.Path(%r);p=R/'endpoint_worker.py';assert not p.exists();p.write_text(%r);print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"%(str(R),worker))
node.dump(R/'results/endpoint-worker-source.json',parallel(provision))
start=next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage')-1;end=next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='config')-1
stage_source='\n'.join(source.splitlines()[start:end]);assert "str(R/'worker.py')" in stage_source;exec(compile(stage_source.replace("str(R/'worker.py')","str(R/'endpoint_worker.py')"),'endpoint-stage','exec'))
last=T['final']['inserted'];proof=node.load(R/'results'/('export-stopped%d.json'%last));assert proof
def verify(i):
 code="""import pathlib,json,hashlib
R=pathlib.Path(ROOT);F=R/'framework';assert all(hashlib.sha256((F/n).read_bytes()).hexdigest()==h for n,h in HASHES.items());print(json.dumps(dict(status='EXISTING_POLICY_EXPORT_EXACT')))"""
 return execute(i,'ROOT=%r\nHASHES=%r\n'%(str(R),proof['files'])+code)
node.dump(R/'results/endpoint-policy-parity.json',parallel(verify))
node.dump(R/'results/endpoint-authorization.json',dict(epoch=time.time(),scope='Complete already-authorized actual endpoint frozen48. Prior diagnostic dispatch had0native spawns and0valid completed cells, all preserved. No new training, no full-budget claim.',controller_marker='endpoint-controller-launch.json',worker_marker='endpoint-launch.json',stop_marker='STOP_ENDPOINT'))
try:
 stage('endpoint',evaljobs([('stopped',proof['path']),('archived',str(BASE/'checkpoints/formal24-grid24-seed3-balanced-u010/step-2412694'))],M['training_config']['conditions'],1,'endpoint-confirm'))
 node.dump(R/'results/endpoint-outcome.json',dict(status='STOPPED_MODEL_FROZEN48_COMPLETE',epoch=time.time(),inserted=last,training_budget_complete=False,training_continuations=0))
 status('STOPPED_MODEL_ASSESSMENT_COMPLETE')
except Exception:
 node.dump(R/'results/endpoint-failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
 def abort(i):
  return execute(i,"import pathlib,sys,json;R=pathlib.Path(%r);(R/'STOP_ENDPOINT').write_text('ENDPOINT_FAILURE');sys.path.insert(0,str(R/'framework'));import node_control as node;[node.cleanup(p.name[:-len('-node-ownership.json')]) for p in (R/'framework/logs').glob('endpoint-confirm-*-node-ownership.json')];print('{}')"%str(R),90)
 parallel(abort);raise
