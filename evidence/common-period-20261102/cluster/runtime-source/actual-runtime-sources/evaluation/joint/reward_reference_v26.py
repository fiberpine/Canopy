"""Optional training reward scale. Never changes policy state or clock.

The reference is frozen from PRIOR TRAINING telemetry, shared across policies
and durations for each environment. It is not an instantaneous trace oracle.
Default is the historical per-connection observed peak, bit-for-bit unchanged.
"""
import hashlib,json,math,pathlib

MODE='fixed_training_reference_v26'

def key(cfg):
    # Buffer depth and observation noise do not define service capacity. Pool
    # prior training peaks across them, then freeze one scale per trace/RTT.
    fields=dict(trace_sha256=cfg['trace_sha256'],base_rtt_ms=float(cfg['base_rtt_ms']))
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load(cfg):
    mode=cfg.get('utility_normalization','observed_peak')
    if mode=='observed_peak':return None
    assert mode==MODE and cfg.get('decision_observation_clock')
    assert not cfg.get('use_normalizer',False),'Reference and raw rates must use same units'
    p=pathlib.Path(cfg['reward_reference_table']);data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==cfg['reward_reference_sha256']
    table=json.loads(data);assert table['schema']==MODE and table['source_split']=='prior_training_only'
    k=key(cfg);entry=table['entries'][k];v=float(entry['reference_native_rate'])
    assert math.isfinite(v) and v>0
    return dict(mode=mode,key=k,reference_native_rate=v,table_sha256=cfg['reward_reference_sha256'],
                source_split=table['source_split'],state_normalization='unchanged observed peak',
                reference_is_not_policy_input=True)

def density(original_density,observed_peak,reference):
    if reference is None:return original_density
    peak=float(observed_peak);u=float(original_density)
    assert math.isfinite(peak) and peak>0 and math.isfinite(u)
    # Same raw throughput/loss/delay utility numerator; only denominator changes.
    # Do not clip at one: the empirical reference is a scale, not a capacity bound.
    return u*peak/reference['reference_native_rate']
