"""Bind future admission records to the candidate actually being executed."""
import hashlib
import json
from .configuration import ROOT
from .core import BASE_COMMIT
from . import continuous


def continuous_contract(cfg):
    """Bind learned semantics and exploration, independently of host/run paths."""
    keys = ('schema','actor_architecture','actor_recipe','time_parameterization','gamma','base_period_s',
            'decision_cost','time_initializer','calibration_main_mode','clock_mode','rate_mode','utility_normalization',
            'window_bn_statistics','teacher_anchor_weight','teacher_anchor_delta','teacher_anchor_clock_weighting',
            'time_feature_normalization','window_actor_lr','time_actor_lr','conditioned_window_actor_lr',
            'critic_initial_cost_reference_s','critic_period_residual_lr','critic_common_calibration_updates',
            'critic_period_zero_anchor','envelope_stop_gradient_duration',
            'pre_adapt_conditioned_window_in_forward_phase','time_barrier_weight',
            'critic_period_residual_l2','calibration_critic_tau','common_calibration_n_step',
            'time_logit_exploration_std','exploration_std','joint_period_probe_probability','period_probe_probability',
            'time_limit_handling','updates_per_base20ms','connection_reset_seconds')
    return {key: cfg.get(key) for key in keys}


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
    if cfg and continuous.enabled(cfg):
        from .core import validate
        validate(cfg)
        expected_status = 'PASS_PREREGISTERED_CONTINUOUS_V5_INITIALIZATION' if continuous.is_v5(cfg) else 'PASS_PREREGISTERED_CONTINUOUS_V4_INITIALIZATION'
        if receipt.get('status') != expected_status:
            raise ValueError('Continuous conditioned policy needs a fresh admission, not the old receipt')
        if receipt.get('implementation') != implementation_identity(require_binary=True):
            raise ValueError('Continuous admission source/native mismatch')
        if receipt.get('contract') != continuous_contract(cfg):
            raise ValueError('Continuous admission algorithm contract mismatch')
        required = ('offline_graph_pass','offline_small_advantage_pass','initial_real_input_parity_pass',
                    'same_decision_exploration_pass','short_tcp_mechanism_pass','initial_frozen_grid_health_pass')
        if not receipt.get('registration') or not receipt.get('evidence') or any(receipt.get(k) is not True for k in required):
            raise ValueError('Continuous initialization evidence is incomplete')
        return
    g3=bool(cfg and cfg.get('rate_mode')=='acked_legacy_peak')
    expected='PASS_PREREGISTERED_G0_G3_FULL_GRID' if g3 else 'PASS_PREREGISTERED_G0_G1_G2'
    if cfg and cfg.get('time_initializer')=='feedback_coverage_v1':expected='PASS_PREREGISTERED_G0_G4_FULL_GRID'
    cost_repair=bool(cfg and cfg.get('actor_recipe')=='cost_preserve_v2')
    if cost_repair:expected='PASS_PREREGISTERED_G0_G4_COST_PRESERVE_INITIALIZATION'
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
        valid_cost=(cfg.get('decision_cost')==0.)
        if cost_repair:
            bridge=receipt.get('training_only_initialization_bridge',{})
            valid_cost=(float(cfg.get('decision_cost',0))>0 and receipt.get('validated_decision_cost')==cfg['decision_cost']
                and all(bridge.get(k) is True for k in ('initial_policy_function_unchanged','observation_and_action_mapping_unchanged',
                    'native_unchanged','same_real_input_parity_pass','prior_full_grid_witness_retained')))
            if (cfg.get('window_bn_statistics')!='frozen_migrated' or float(cfg.get('teacher_anchor_weight',0))<=0
                    or cfg.get('teacher_anchor_clock_weighting')!='base_over_planned_squared_stopgrad'):
                raise ValueError('Cost repair preservation contract differs from its registered initialization')
        if (receipt.get('rate_mode')!='acked_legacy_peak' or receipt.get('utility_normalization')!='legacy_delivery_peak'
                or cfg.get('utility_normalization')!='legacy_delivery_peak' or not valid_cost):
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
