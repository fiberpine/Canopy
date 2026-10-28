"""Only finish the already authorized endpoint evaluation after a stock late guard.
Never trains or retries the failed full-budget pipeline.
"""
import pathlib,sys,os,json,time,ast,copy,subprocess,concurrent.futures,traceback,random,io,tarfile,base64,hashlib
R=pathlib.Path(os.environ['CANOPY_REPRO_ROOT']);F=R/'framework';sys.path.insert(0,str(F));import node_control as node
M=node.load(R/'registration.json');A=node.load(R/'authorization.json');END=M['deadline_epoch'];BASE=pathlib.Path('@ARCHIVED_BASELINE@');OLD=pathlib.Path('@READONLY_RUNTIME@')
# Definitions only: never execute pipeline.py's launch/training body.
tree=ast.parse((R/'pipeline.py').read_text());wanted={'execute','parallel','status','stage','config','export','evaljobs'}
defs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],**({'type_ignores':[]} if 'type_ignores' in ast.Module._fields else {}));exec(compile(defs,str(R/'pipeline.py')+'::definitions','exec'))
T=node.load(F/'metrics/base-reproduction-seed3-2400k-cluster-result.json');assert T and T['final']['status']=='COMPLETED';assert T['stop_reason'].startswith('PERSISTENT_LOW_WINDOW_LOW_UTIL_ACTOR'),T
assert T['final']['inserted']<2400000,'Only a stock late guard below the full budget permits this endpoint-only assessment'
assert time.time()<END-1000
# Keep the original STOP_ALL/failure immutable. This endpoint-only worker has its own stop marker.
worker=(R/'worker.py').read_text();assert worker.count("(R/'STOP_ALL').exists()")==1
worker=worker.replace("(R/'STOP_ALL').exists()","(R/'STOP_ASSESSMENT').exists()")
def provision(i):
 code="""import pathlib,json,hashlib
R=pathlib.Path(ROOT);p=R/'assessment_worker.py';assert not p.exists();p.write_text(SOURCE)
print(json.dumps(dict(status='ASSESSMENT_WORKER_PROVISIONED',sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"""
 return execute(i,'ROOT=%r\nSOURCE=%r\n'%(str(R),worker)+code)
node.dump(R/'results/diagnostic-worker-source.json',parallel(provision))
stage_source=ast.get_source_segment((R/'pipeline.py').read_text(),next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage')) if hasattr(ast,'get_source_segment') else '\n'.join((R/'pipeline.py').read_text().splitlines()[next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage')-1:next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='config')-1])
assert "str(R/'worker.py')" in stage_source
exec(compile(stage_source.replace("str(R/'worker.py')","str(R/'assessment_worker.py')"),str(R/'pipeline.py')+'::assessment-stage','exec'))
node.dump(R/'results/diagnostic-previous-pipeline-status.json',node.load(R/'results/pipeline-status.json'))
with (R/'results/diagnostic-authorization.json').open('x') as stream:json.dump(dict(epoch=time.time(),scope='Complete authorized frozen endpoint assessment, not a training continuation. Original full2.4M,screen/selected confirmation remain NOT_RUN. One actual-stopped model and original archived comparator x24conditions x1x60s. Original pipeline failure/result preserved.',reason=T['stop_reason']),stream)
D=F/'checkpoints'/M['training_config']['model_id'];steps=sorted(int(p.stem.split('-')[1]) for p in D.glob('step-*.index'));last=steps[-1]
cp=export(str(D/('step-%d'%last)),'stopped%d'%last)
try:rows=stage('diagnostic',evaljobs([('stopped',cp),('archived',str(BASE/'checkpoints/formal24-grid24-seed3-balanced-u010/step-2412694'))],M['training_config']['conditions'],1,'stopped-assess'))
except Exception:
 node.dump(R/'results/diagnostic-failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
 def abort(i):
  return execute(i,"import pathlib,sys,json;R=pathlib.Path(%r);(R/'STOP_ASSESSMENT').write_text('ASSESSMENT_FAILURE');sys.path.insert(0,str(R/'framework'));import node_control as node;[node.cleanup(p.name[:-len('-node-ownership.json')]) for p in (R/'framework/logs').glob('stopped-assess-*-node-ownership.json')];print('{}')"%str(R),90)
 parallel(abort);raise
node.dump(R/'results/diagnostic-outcome.json',dict(status='STOPPED_MODEL_FROZEN48_COMPLETE',epoch=time.time(),inserted=last,training_budget_complete=False,training_continuations=0,screen_selection='NOT_RUN_FULL_BUDGET_INCOMPLETE'))
status('STOPPED_MODEL_ASSESSMENT_COMPLETE')
