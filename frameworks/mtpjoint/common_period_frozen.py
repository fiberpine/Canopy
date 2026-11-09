"""Frozen evaluation of a saved conditional family or its deterministic selector."""
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/canopy/rl-module'))
import evaluate
from common_period_deployment import install
install(evaluate)
from envwrapper_v2 import TCP_Env_Wrapper
from common_period import encoded_period,utility,Hysteresis,validate_execution
from common_period_tf import Model
cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text());validate_execution(cfg)
if not cfg.get('no_training') or cfg.get('training_mode'):raise ValueError('Frozen evaluation only')
class Wrapper(TCP_Env_Wrapper):
    def get_state(self,*args,**kwargs):
        info,delay,r,ok=super().get_state(*args,**kwargs)
        if not ok:return info,delay,r,ok
        x=self.shrmem_r.last_raw
        return info,delay,utility(float(x[1]),float(x[7]),cfg['common_reward_config']['scale_bps'],float(x[0]),float(x[8]),float(x[14]),float(x[2])),ok
def build(c):
    meta=json.loads(pathlib.Path(c['checkpoint']+'.metadata.json').read_text())
    if meta.get('active_training_periods_ms') == [20] and (
            c.get('common_fixed_period_ms') != 20 or c.get('common_rule') or
            c.get('selector_lambda') is not None):
        raise ValueError('S20 checkpoint requires fixed 20 ms frozen evaluation')
    model=Model(c['source_checkpoint'],meta['K'],meta['n0'],reward_config=c['common_reward_config']);model.restore(c['checkpoint']);rule=Hysteresis()
    with model.graph.as_default():variables=model.tf.global_variables()
    def digest():
        h=hashlib.sha256()
        for v,x in zip(variables,model.session.run(variables)):h.update(v.op.name.encode());h.update(np.asarray(x).tobytes())
        return h.hexdigest()
    def action(state):
        if c.get('selector_lambda') is not None:result,_=model.choose(state,c['selector_lambda'])
        else:
            t=rule.choose(state) if c.get('common_rule') else c['common_fixed_period_ms']
            result=np.asarray([0,encoded_period(t)],np.float32)
            result[0]=model.session.run(model.mu,{model.s:np.asarray(state,np.float32)[None,:],model.action:result[None,:]})[0,0]
        action.no_noise=result.copy();action.behavior_source='frozen-value-selector' if c.get('selector_lambda') is not None else 'frozen-conditional-family'
        return result
    return model,model.session,Wrapper,json.loads((ROOT/'code/canopy/params_base_eval.json').read_text()),action,digest,[v.op.name for v in variables]
original=evaluate.StateReader.read
def read(self):
    if self.last_id is not None:return original(self)
    start=time.monotonic()
    while True:
        try:return original(self)
        except RuntimeError as e:
            if str(e)!='no valid state within15s' or self.stop.is_set() or time.monotonic()-start+15>60.1:raise
evaluate.StateReader.read=read;evaluate.model=build
evaluate.main()
