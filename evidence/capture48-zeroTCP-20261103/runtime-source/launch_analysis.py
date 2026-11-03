import pathlib,json,sys,base64,hashlib
E=pathlib.Path(__file__).parent;reg=json.loads((E/'registration.json').read_text());sys.path.insert(0,str(E.parent/'common-period-cluster-20261102'));from remote import run
blob=base64.b64encode((E/'analyze.py').read_bytes()).decode();code='ROOT=%r\nBLOB=%r\n'%(reg['cloud_root'],blob)+r'''
import pathlib,json,base64,subprocess,sys,time,hashlib
R=pathlib.Path(ROOT);assert (R/'gather-result.json').exists() and not (R/'analysis-launch.json').exists();source=base64.b64decode(BLOB);compile(source,'analyze.py','exec');(R/'analyze.py').write_bytes(source)
with (R/'analysis.stdout').open('wb') as out:p=subprocess.Popen([sys.executable,'-B',str(R/'analyze.py')],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
x=dict(pid=p.pid,epoch=time.time(),source_sha256=hashlib.sha256(source).hexdigest());(R/'analysis-launch.json').write_text(json.dumps(x));print(json.dumps(x))
'''
r=run('learner',code,30);(E/'analysis-launch-response.json').write_text(json.dumps(r,indent=2));assert r['code']==0,r;print(r['output'])
