"""Bind future admission records to the candidate actually being executed."""
import hashlib
import json
from .configuration import ROOT
from .core import BASE_COMMIT


def validate_native():
    binary = ROOT/'frameworks/joint_healthy/native/orca-server-joint-healthy'
    record = json.loads(binary.with_name(binary.name+'.build.json').read_text())
    if record['sha256'] != hashlib.sha256(binary.read_bytes()).hexdigest(): raise ValueError('Native/build receipt differs')
    expected = {'frameworks/joint_healthy/native/'+n for n in ('orca-server-joint-healthy.cc','joint_protocol.h')}
    expected.update('frameworks/baseline/code/runtime/src/'+n for n in ('flow.cc','flow.h','define_v2.h','common.h','util.h'))
    if set(record['source_sha256']) != expected: raise ValueError('Native build source list is incomplete')
    for name, digest in record['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest: raise ValueError('Rebuild native after source changes: '+name)
    if record['startup'] != 'natural-cubic': raise ValueError('Wrong native startup contract')
    return record


def implementation_identity(require_binary=False):
    paths = [ROOT/'joint_healthy.py'] + sorted((ROOT/'frameworks/joint_healthy').glob('*.py'))
    paths += sorted((ROOT/'frameworks/joint_healthy/native').glob('*.cc'))
    paths += sorted((ROOT/'frameworks/joint_healthy/native').glob('*.h'))
    hashes = {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
    binary = ROOT/'frameworks/joint_healthy/native/orca-server-joint-healthy'
    if require_binary and not binary.is_file(): raise ValueError('Build the admitted native first')
    return dict(base_commit=BASE_COMMIT, source_hash_mode='LF_normalized', sources=hashes,
                native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None)


def check_admission(receipt):
    if receipt.get('status') != 'PASS_PREREGISTERED_G0_G1_G2':
        raise ValueError('Registered G0/G1/G2 admission is missing')
    if receipt.get('implementation') != implementation_identity(require_binary=True):
        raise ValueError('Admission record does not match the current source/native')
    if not receipt.get('registration') or not receipt.get('evidence'):
        raise ValueError('Admission requires the preregistration and result references')
