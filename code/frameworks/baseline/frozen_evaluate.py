"""Identical frozen evaluator; explicit neutral-action comparison only."""
import pathlib,sys,json
import evaluate
cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
if cfg.get('policy_mode')=='neutral':
 original=evaluate.model
 def model(c):
  a,sess,wrapper,params,action,digest,variables=original(c)
  return a,sess,wrapper,params,lambda hist:0.,digest,variables
 evaluate.model=model
evaluate.main()
