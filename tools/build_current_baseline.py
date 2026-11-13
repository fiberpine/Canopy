"""Rebuild the two accepted native roles without overwriting historical binaries."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
LOGICAL_SOURCE='/home/ubuntu/orca-capacity-repair-20261112/runtime/orca-server-control-log-safe.cc'


def main():
    current=json.loads((ROOT/'CURRENT_BASELINE.json').read_text())
    receipts=[]
    for mode,audit,name in [('legacy',False,'orca-server-cloexec'),('natural-cubic',True,'orca-server-natural-cubic')]:
        relative='frameworks/baseline/code/runtime/'+name
        target=ROOT/relative
        with TemporaryDirectory(prefix='canopy-native-') as temporary:
            output=Path(temporary)/'native'
            cmd=[sys.executable,str(ROOT/'tools/build_scalar_startup_candidate.py'),'--framework',str(ROOT/'frameworks/baseline'),
                 '--output',str(output),'--startup',mode,'--logical-source-path',LOGICAL_SOURCE]
            if audit:cmd.append('--audit')
            subprocess.check_call(cmd)
            sha=hashlib.sha256(output.read_bytes()).hexdigest()
            expected=current['models_and_natives'][relative]['sha256']
            if sha!=expected:
                raise RuntimeError('Rebuilt native differs from accepted GCC7.5 artifact; preserved binary was not replaced: '+name)
            receipt=json.loads(Path(str(output)+'.startup-build.json').read_text());receipt['delivered_path']=relative
            if not target.exists() or target.read_bytes()!=output.read_bytes():shutil.copyfile(output,target)
            target.chmod(0o755);receipts.append(receipt)
    folder=ROOT/'environment';folder.mkdir(exist_ok=True)
    (folder/'current-baseline-native-rebuild.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(dict(status='PASS_ACCEPTED_NATIVE_BYTES',roles=2)))


if __name__=='__main__':main()
