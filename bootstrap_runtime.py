#!/usr/bin/env python3
"""Relocate copied Python3.6 environment, touching only package-owned text files."""
import pathlib,os,sys,subprocess,json,platform,re
P=pathlib.Path(__file__).resolve().parent;V=P/'runtime/venv';vendor=P/'runtime/vendor/interval-bound-propagation'
assert platform.system()=='Linux' and platform.machine()=='x86_64','Linux x86_64 is required'
assert pathlib.Path('/usr/bin/python3.6').exists(),'Install Python3.6 (validated platform Ubuntu18.04 x86_64)'
assert V.is_dir() and vendor.is_dir()
changed=[]
old_venvs={'/home/ubuntu/venv'}
activate=V/'bin/activate'
if activate.is_file():old_venvs.update(re.findall(r'VIRTUAL_ENV="([^"]+)"',activate.read_text()))
for p in list((V/'bin').glob('*'))+list((V/'lib/python3.6/site-packages').glob('*.pth'))+list((V/'lib/python3.6/site-packages').glob('*.egg-link')):
 if not p.is_file() or p.is_symlink():continue
 try:s=p.read_text()
 except (UnicodeDecodeError,OSError):continue
 t=s.replace('/home/ubuntu/interval-bound-propagation',str(vendor))
 for old in sorted(old_venvs,key=len,reverse=True):
  t=t.replace(str(pathlib.Path(old).parent/'vendor/interval-bound-propagation'),str(vendor)).replace(old,str(V))
 if t!=s:p.write_text(t);changed.append(str(p.relative_to(P)))
(V/'lib/python3.6/site-packages/portable_ibp.pth').write_text(str(vendor)+'\n')
(V/'lib/python3.6/site-packages/portable_native_deps.pth').write_text(str(P/'frameworks/baseline/code/runtime/_deps')+'\n')
q=subprocess.run([str(V/'bin/python'),'-c','import sys,tensorflow,sonnet,interval_bound_propagation,sysv_ipc,numpy,psutil;assert tensorflow.__version__=="1.14.0";print(sys.version);print(tensorflow.__version__)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out=dict(status='PASS' if q.returncode==0 else 'FAIL',python=str(V/'bin/python'),changed_files=changed,stdout=q.stdout.decode(),stderr=q.stderr.decode()[-1500:]);(P/'environment/runtime-bootstrap.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));sys.exit(q.returncode)
