#!/usr/bin/env python3
"""Verify a fresh source checkout, including every selected model tensor."""
import hashlib,json,pathlib,os,sys,ast
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 inventory=json.loads((ROOT/'repository-files.json').read_text());bad=[];links=0
 for x in inventory['files']:
  p=ROOT/x['path']
  if x.get('symlink'):
   actual=os.readlink(str(p)) if p.is_symlink() else (p.read_text().strip() if p.is_file() else None)
   if actual!=x['symlink']:bad.append(dict(path=x['path'],reason='Missing or changed symlink target'))
   links+=1
  elif not p.is_file() or digest(p)!=x['sha256']:bad.append(dict(path=x['path'],reason='Missing file or SHA256 mismatch'))
 manifest=json.loads((ROOT/'manifest.json').read_text());models=0
 for record in manifest['proof'].values():
  for x in record.get('selected_files',[]) if isinstance(record,dict) else record:
   p=ROOT/x['path']
   if not p.is_file() or digest(p)!=x['sha256']:bad.append(dict(path=x['path'],reason='Original selected model mismatch'))
   models+=1
 for variant in ['baseline','mtp20','mtpjoint']:
  f=ROOT/'frameworks'/variant
  for name in manifest['proof'][variant]['modules']:
   p=f/(name+'.py')
   try:ast.parse(p.read_text(encoding='utf-8'))
   except Exception as e:bad.append(dict(path=str(p.relative_to(ROOT)),reason='Python parse: '+str(e)))
  for name in ['code/runtime/src/flow.cc','code/runtime/src/client.c','code/canopy/rl-module/agent_v2.py','configs/traces']:
   if not (f/name).exists():bad.append(dict(path=variant+'/'+name,reason='Required framework component missing'))
 print(json.dumps(dict(status='PASS_FRESH_CHECKOUT' if not bad else 'FAIL',checked_files=len(inventory['files']),original_model_files=models,symlinks=links,errors=bad),indent=2));return bool(bad)
if __name__=='__main__':sys.exit(main())
