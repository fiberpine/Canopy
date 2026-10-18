#!/usr/bin/env python3
import pathlib,subprocess,json,hashlib
P=pathlib.Path(__file__).resolve().parent;proof=[]
for name in ['baseline','mtp20','mtpjoint']:
 f=P/'frameworks'/name;s=f/'code/runtime/src';out=f/'code/runtime'
 target=(f/'native-v28/orca-server-joint-mtp') if name=='mtpjoint' else out/('orca-server-control-log-safe' if name=='baseline' else 'orca-server-joint-mtp')
 target.parent.mkdir(exist_ok=True);source=s/('orca-server-control-log-safe.cc' if name=='baseline' else 'orca-server-joint-mtp.cc')
 cmd=['g++','-std=c++11','-pthread',str(source),str(s/'flow.cc'),'-o',str(target)];subprocess.check_call(cmd)
 proof.append(dict(variant=name,command=cmd,binary=str(target.relative_to(P)),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(P/'environment/native-rebuild.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
