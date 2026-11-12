"""Build a separate frozen-policy startup diagnostic; never replace a runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(framework, output, startup, audit=False, compiler='g++'):
    framework, output = framework.resolve(), output.resolve()
    source = framework / 'code/runtime/src'
    inputs = [source / name for name in (
        'orca-server-control-log-safe.cc', 'flow.cc', 'define_v2.h', 'flow.h',
        'common.h', 'util.h')]
    sidecar = Path(str(output) + '.startup-build.json')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise ValueError('Existing output must be preserved: ' + str(path))
    for path in inputs:
        if not path.is_file():
            raise ValueError('Required source missing: ' + str(path))
    if startup not in ('legacy', 'first-feedback'):
        raise ValueError('Unknown startup mode: ' + startup)
    flags = ['-std=c++11', '-pthread']
    if startup == 'first-feedback':
        flags.append('-DORCA_FIRST_FEEDBACK_HANDOFF')
    if audit:
        flags.append('-DORCA_STARTUP_AUDIT')
    # Match the existing scalar build flags: no added optimization or ABI flags.
    command = [compiler] + flags + [str(inputs[0]), str(inputs[1]), '-o', str(output)]
    return dict(startup=startup, audit=audit, command=command,
                source_sha256={p.name: digest(p) for p in inputs}), output, sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup', choices=('legacy', 'first-feedback'), required=True)
    parser.add_argument('--audit', action='store_true', help='Same one-shot handoff telemetry in both builds')
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without compiling or writing files')
    args = parser.parse_args(argv)
    try:
        metadata, output, sidecar = prepare(args.framework, args.output, args.startup, args.audit, args.compiler)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        metadata['status'] = 'PLAN_ONLY_NOT_COMPILED'
        print(json.dumps(metadata, indent=2))
        return
    if shutil.which(args.compiler) is None:
        parser.error('C++ compiler unavailable; use the supported Linux build environment')
    metadata['compiler_version'] = subprocess.check_output([args.compiler, '--version']).decode().splitlines()[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a temporary file. An unsuccessful compiler cannot corrupt a native.
    with tempfile.TemporaryDirectory(prefix='orca-startup-', dir=str(output.parent)) as temporary:
        built = Path(temporary) / 'native'
        command = metadata['command'][:-1] + [str(built)]
        subprocess.check_call(command)
        for name, expected in metadata['source_sha256'].items():
            if digest(args.framework.resolve() / 'code/runtime/src' / name) != expected:
                raise RuntimeError('Source changed during compilation: ' + name)
        metadata['binary_sha256'] = digest(built)
        metadata['status'] = 'BUILT_NOT_NETWORK_VALIDATED'
        # Exclusive creation also refuses an output created since prepare().
        with output.open('xb') as destination, built.open('rb') as incoming:
            shutil.copyfileobj(incoming, destination)
        os.chmod(str(output), 0o755)
        with sidecar.open('x') as destination:
            json.dump(metadata, destination, indent=2)
            destination.write('\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
