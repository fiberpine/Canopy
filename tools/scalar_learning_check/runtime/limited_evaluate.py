"""Frozen scalar A0/A4g/CUBIC evaluation, no learner or exploration."""
import pathlib,json,sys,math
F=pathlib.Path(__file__).resolve().parent
import evaluate
cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
sys.path.insert(0,str(F/'code/canopy/rl-module'))
if cfg.get('backend')=='moving_stats':
    import agent_v2,moving_stats_backend
    moving_stats_backend.install(agent_v2)
original=evaluate.model
def model(c):
    a,sess,wrapper,params,predict,digest,variables=original(c)
    if c.get('diagnostic_goodput_floor'):
        floor=.963*c['capacity_mbps']*1e6/8
        class FlooredWrapper(wrapper):
            def get_state(self,*args,**kwargs):
                self.max_bw=max(getattr(self,'max_bw',0.0),floor)
                return super().get_state(*args,**kwargs)
        wrapper=FlooredWrapper
        c['diagnostic_floor_Bps']=floor
    def action(history):
        x=float(predict(history))
        if not math.isfinite(x) or not -1.00001<=x<=1.00001:raise ValueError('Invalid frozen actor output')
        if x<-1 or x>1:
            # Enforce the declared tanh domain only for float-roundoff overshoot.
            # Apply identically to current/old checkpoints and record every event.
            c['float_domain_corrections']=c.get('float_domain_corrections',0)+1
            x=max(-1.0,min(1.0,x))
        return x
    return a,sess,wrapper,params,action,digest,variables
evaluate.model=model
evaluate.main()
