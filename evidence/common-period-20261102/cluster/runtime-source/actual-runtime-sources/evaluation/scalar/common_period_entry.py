"""Frozen A0 bridge from real scheduled boundaries to the new replay contract.

This entrypoint certifies the new data path; it never starts a learner or runs
an optimizer. Formal A must use a separately registered capacity-tested launcher.
"""
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/canopy/rl-module'))
import evaluate
from envwrapper_v2 import TCP_Env_Wrapper
from common_period import encoded_period,utility,validate_execution
from common_period_tf import Model
from common_period_runtime import Runtime

cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
if not cfg.get('common_period_a0_frozen') or not cfg.get('no_training'):
    raise ValueError('This is a frozen A0 entrypoint only')
validate_execution(cfg)
reward=cfg['common_reward_config'];model=None;runtime=None
class Wrapper(TCP_Env_Wrapper):
    def get_state(self,*args,**kwargs):
        info,delay,old,ok=super().get_state(*args,**kwargs)
        if not ok:return info,delay,old,ok
        r=self.shrmem_r.last_raw
        value=utility(float(r[1]),float(r[7]),reward['scale_bps'],float(r[0]),float(r[8]),float(r[14]),float(r[2]))
        return info,delay,value,ok

def build(c):
    global model,runtime
    n0={str(t):1/(1-.995**(t/20.)) for t in [20,40,80,120,160]}
    model=Model(c['checkpoint'],0.,n0,reward_config=reward)
    runtime=Runtime(model,reward);snapshot=runtime.publish();period=c['diagnostic_fixed_clock_ms']
    def action(state):
        pair=np.asarray([[0,encoded_period(period)]],np.float32)
        window=model.session.run(model.mu,{model.s:np.asarray(state,np.float32)[None,:],model.action:pair})[0,0]
        if c.get('a0_neutral_mapping_probe'):window=0.
        result=np.asarray([window,encoded_period(period)],np.float32)
        action.no_noise=result.copy();action.behavior_source='a0-frozen-neutral' if c.get('a0_neutral_mapping_probe') else 'a0-frozen-conditional-window'
        return result
    def digest():
        values=model.session.run(list(model.v.values()));h=hashlib.sha256()
        for value in values:h.update(value.tobytes())
        return h.hexdigest()
    params=json.loads((ROOT/'code/canopy/params_base_eval.json').read_text())
    def transition(s,a,r,sp,dt,gamma):
        if abs(gamma-.995**(dt/.02))>1e-7:raise ValueError('Wrong actual-time discount')
        runtime.boundary(0,c['run_id'],c['run_id'],s,a,r,sp,dt,period,snapshot['version'],behavior='policy')
    evaluate.transition_hook=transition
    return model,model.session,Wrapper,params,action,digest,list(model.v)

original_read=evaluate.StateReader.read
def read(self):
    if self.last_id is not None:return original_read(self)
    start=time.monotonic()
    while True:
        try:return original_read(self)
        except RuntimeError as exc:
            if str(exc)!='no valid state within15s' or self.stop.is_set() or time.monotonic()-start+15>60.1:raise
evaluate.StateReader.read=read;evaluate.model=build
try:evaluate.main()
finally:
    if runtime is not None:
        rows=runtime.replay.rows
        if rows:rows[-1]['kind']='truncated'
        out=ROOT/'logs'/cfg['run_id']/'common-period-a0.json'
        out.write_text(json.dumps(dict(ledger=runtime.ledger(),rows=len(rows),reward_version=runtime.model.metadata['reward_version'],reward_checksum=runtime.replay.reward_checksum,all_snapshots_known=all(r['snapshot'] in runtime.known_snapshots for r in rows),all_apply_one=all(r['apply']==1 for r in rows),same_connection=all(r['connection']==r['next_connection'] for r in rows),last_kind=rows[-1]['kind'] if rows else None,calibration_credit=0,formal_training=False),indent=2))
