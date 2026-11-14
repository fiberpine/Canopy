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


def check_admission(receipt,cfg=None):
    g3=bool(cfg and cfg.get('rate_mode')=='acked_legacy_peak')
    expected='PASS_PREREGISTERED_G0_G3_FULL_GRID' if g3 else 'PASS_PREREGISTERED_G0_G1_G2'
    if cfg and cfg.get('time_initializer')=='feedback_coverage_v1':expected='PASS_PREREGISTERED_G0_G4_FULL_GRID'
    if receipt.get('status') != expected:
        raise ValueError('Registered admission for this measurement contract is missing')
    if receipt.get('implementation') != implementation_identity(require_binary=True):
        raise ValueError('Admission record does not match the current source/native')
    if not receipt.get('registration') or not receipt.get('evidence'):
        raise ValueError('Admission requires the preregistration and result references')
    if g3:
        required=dict(conditions=24,repeats=3,candidate_runs=72,paired_base_runs=72,health_pass=True,g0_mechanism_pass=True)
        if any(receipt.get('validation',{}).get(k)!=v for k,v in required.items()):
            raise ValueError('G3 needs its own full24x3 matched frozen admission')
        if (receipt.get('rate_mode')!='acked_legacy_peak' or receipt.get('utility_normalization')!='legacy_delivery_peak'
                or cfg.get('utility_normalization')!='legacy_delivery_peak' or cfg.get('decision_cost')!=0.):
            raise ValueError('G3 admission is only for the declared zero-cost legacy-peak contract')
        initializer=cfg.get('time_initializer','constant')
        if receipt.get('validated_time_initializer','constant')!=initializer:
            raise ValueError('Initializer differs from the frozen full-grid validation')
        if initializer=='feedback_coverage_v1':
            if cfg.get('calibration_main_mode')!='policy_initializer':
                raise ValueError('Calibration must preserve the validated state-dependent initializer')
        else:
            initial=float(receipt.get('validated_initial_period_ms',20.))
            if initial!=float(cfg.get('calibration_main_ms',20.)):
                raise ValueError('Calibration main period differs from the validated initializer')
            if initial!=20. and initial!=float(cfg['time_init_ms']):
                raise ValueError('Forward calibration must use the validated initializer')
