#!/usr/bin/env python3
"""Install exact runtime assets from the private, SHA-verified full release."""
import argparse,hashlib,json,pathlib,shutil,tarfile,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXPECTED='7cb05dbbf33ca91e7ad237585df858063fc6fa3b5e1bf57115f6c59f6ecff17b'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',required=True,type=pathlib.Path);a=ap.parse_args()
 assert digest(a.archive)==EXPECTED,'Full-release SHA256 mismatch; nothing was installed'
 assert not (ROOT/'runtime/venv').exists(),'Runtime already exists; refusing to overwrite'
 assert not (ROOT/'runtime/kernel').exists(),'Kernel asset directory already exists; refusing to overwrite'
 inventory=json.loads((ROOT/'environment/package-files.json').read_text())
 expected={x['path']:x for x in inventory['files'] if x['path'].startswith(('runtime/venv/','runtime/kernel/'))}
 with tempfile.TemporaryDirectory(prefix='.runtime-assets-',dir=str(ROOT)) as td:
  stage=pathlib.Path(td)
  with tarfile.open(str(a.archive)) as t:
   members=[]
   for ti in t.getmembers():
    parts=pathlib.PurePosixPath(ti.name).parts
    assert parts[0]=='orca-release-20261018' and '..' not in parts and not ti.name.startswith('/')
    rel='/'.join(parts[1:])
    if not rel.startswith(('runtime/venv/','runtime/kernel/')):continue
    if ti.issym():assert ti.linkname in ['python3','/usr/bin/python3','lib'],('Unexpected runtime link',rel,ti.linkname)
    assert ti.isfile() or ti.isdir() or ti.issym(),('Unsupported member',rel)
    ti.name=rel;members.append(ti)
   t.extractall(str(stage),members=members)
  for name,x in expected.items():
   p=stage/name;assert p.is_file() and digest(p)==x['sha256'],name
  for name in ['venv','kernel']:shutil.move(str(stage/'runtime'/name),str(ROOT/'runtime'/name))
 print(json.dumps(dict(status='INSTALLED_VERIFIED_RUNTIME_ASSETS',verified_files=len(expected),next='Run bash setup.sh after preparing the matching OS/kernel; no system settings changed by this installer')))
if __name__=='__main__':main()
