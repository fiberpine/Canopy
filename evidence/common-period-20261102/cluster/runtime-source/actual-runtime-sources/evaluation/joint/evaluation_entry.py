
import pathlib,json,sys,time,runpy
import evaluate
cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
original=evaluate.StateReader.read
def read(self):
 if self.last_id is not None:return original(self)
 start=time.monotonic()
 while True:
  try:return original(self)
  except RuntimeError as e:
   if str(e)!='no valid state within15s' or self.stop.is_set() or time.monotonic()-start+15>60.1:raise
evaluate.StateReader.read=read
if cfg.get('common_family'):
 runpy.run_path(str(pathlib.Path(__file__).with_name('common_period_frozen.py')),run_name='__main__')
else:
 if cfg.get('joint_control'):
  from common_period_deployment import install
  install(evaluate)
 runpy.run_path(str(pathlib.Path(__file__).with_name('repair_evaluate.py')),run_name='__main__')
