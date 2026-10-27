"""Build an independent opt-in scalar native and its integrity sidecar."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--framework',type=pathlib.Path,required=True)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    args=ap.parse_args()
    framework=args.framework.resolve()
    source=framework/'native/src/orca-server-verified-ipc.cc'
    header=framework/'native/src/verified_scalar_ipc.h'
    runtime=framework/'code/runtime/src'
    output=args.output.resolve()
    sidecar=pathlib.Path(str(output)+'.ipc-schema.json')
    if output.exists() or sidecar.exists():
        ap.error('Use a new output path; existing native/sidecar must be preserved')
    for path in [source,header,runtime/'flow.cc',runtime/'define_v2.h']:
        if not path.is_file():ap.error('Required source missing: '+str(path))
    output.parent.mkdir(parents=True,exist_ok=True)
    command=['g++','-std=c++11','-O2','-pthread','-I',str(source.parent),
             '-I',str(runtime),str(source),str(runtime/'flow.cc'),'-o',str(output)]
    subprocess.run(command,check=True)
    metadata=dict(schema=2,binary_sha256=digest(output),
                  source_sha256={p.name:digest(p) for p in [source,header]},command=command)
    sidecar.write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
