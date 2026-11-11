"""Explicit frozen backend and startup intervention; no optimization or exploration."""
import pathlib,json,sys,time
R=pathlib.Path(__file__).resolve().parent
import evaluate
cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
sys.path.insert(0,str(R/'code/canopy/rl-module'))
if cfg.get('backend')=='moving_stats':
 import agent_v2,moving_stats_backend
 moving_stats_backend.install(agent_v2)
original=evaluate.model
def model(c):
 a,sess,wrapper,params,act,digest,variables=original(c);started=[None]
 def action(history):
  if started[0] is None:started[0]=time.monotonic()
  if c.get('policy_mode')=='neutral' or time.monotonic()-started[0]<c.get('startup_neutral_seconds',0):return 0.
  return act(history)
 return a,sess,wrapper,params,action,digest,variables
evaluate.model=model

raw_stream=(R/'logs'/(cfg['run_id']+'-raw15.jsonl')).open('x',buffering=262144)
Reader=evaluate.StateReader
class RecordedReader(Reader):
 def read(self):
  result=super().read()
  raw_stream.write(json.dumps(dict(time_us=time.time()*1e6,state_id=self.last_id,raw=self.last_raw.tolist()))+'\n')
  return result
evaluate.StateReader=RecordedReader
try:evaluate.main()
finally:raw_stream.close()

