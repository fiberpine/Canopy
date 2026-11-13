"""Compile a separate natural-handoff dual-action native; no TCP is run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler', default='g++')
    ap.add_argument('--compiler-arg', action='append', default=[])
    ap.add_argument('--output', type=Path, default=ROOT / 'frameworks/joint_healthy/native/orca-server-joint-healthy')
    args = ap.parse_args()
    native = ROOT / 'frameworks/joint_healthy/native'
    baseline = ROOT / 'frameworks/baseline/code/runtime/src'
    files = [native / 'orca-server-joint-healthy.cc', native / 'joint_protocol.h'] + [baseline / n for n in ('flow.cc', 'flow.h', 'define_v2.h', 'common.h', 'util.h')]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='joint-native-') as temporary:
        binary = Path(temporary) / 'native'
        command = [args.compiler] + args.compiler_arg + ['-std=c++11', '-pthread', '-DORCA_NATURAL_CUBIC_HANDOFF', '-DORCA_STARTUP_AUDIT',
            '-I', str(native), '-I', str(baseline), str(native / 'orca-server-joint-healthy.cc'), str(baseline / 'flow.cc'), '-o', str(binary)]
        subprocess.check_call(command)
        data = binary.read_bytes()
        assert data.startswith(b'\x7fELF'), 'A Linux ELF is required; no host executable may become the native'
        # Source cannot change between planning and publishing a compiled binary.
        assert hashes == {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if args.output.exists() and args.output.read_bytes() != data: raise ValueError('Preserve the existing binary; choose a new --output')
        args.output.write_bytes(data); args.output.chmod(0o755)
    receipt = dict(status='COMPILED_NOT_TCP_VALIDATED', sha256=hashlib.sha256(data).hexdigest(), source_sha256=hashes,
                   command=command[:-1] + ['@TEMP_OUTPUT@'], startup='natural-cubic', new_TCP=0)
    Path(str(args.output) + '.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
