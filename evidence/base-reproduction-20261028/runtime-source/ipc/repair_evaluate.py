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

original_read=evaluate.StateReader.read
stage_path=None
stage_writer=None
def mark(event,state_id=''):
 global stage_path,stage_writer
 if stage_writer is None:
  import csv
  stage_path=(R/'logs'/cfg['run_id']/'python-stages.csv').open('w',buffering=65536);stage_writer=csv.writer(stage_path);stage_writer.writerow(['time_us','event','state_id'])
 stage_writer.writerow([time.time()*1e6,event,state_id])
def read(self):
 mark('state_read_start',self.last_id)
 value=original_read(self);mark('state_read_end',self.last_id);return value
evaluate.StateReader.read=read
model0=model
def stage_model(c):
 a,sess,wrapper,params,act,digest,variables=model0(c)
 def action(h):
  mark('inference_start');v=act(h);mark('inference_end');return v
 return a,sess,wrapper,params,action,digest,variables
evaluate.model=stage_model
try:evaluate.main()
finally:
 if stage_path:stage_path.flush();stage_path.close()
