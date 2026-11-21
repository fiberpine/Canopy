#!/usr/bin/env python3
"""Real TF1 graph verification. No TCP; absent TF is BLOCKED, never PASS."""
import argparse,importlib.util,json,os,pathlib,sys,tempfile,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]

def verify(result):
 if importlib.util.find_spec('tensorflow') is None:
  result.update(status='BLOCKED_BY_ENVIRONMENT',reason='TensorFlow1/Sonnet unavailable on this machine');return
 import numpy as np
 sys.path[:0]=[str(ROOT),str(ROOT/'frameworks/mtpjoint')]
 os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 import run as entry,distributed as d,compatible_calibration_v37 as c
 tf=d.tf
 if not tf.__version__.startswith('1.'):raise RuntimeError('Requires the preserved TensorFlow1 runtime')
 _,cfg=entry.template('mtpjoint','train')
 cfg.update(compatible_v31_checkpoint=str(ROOT/'models/mtpjoint/model'),matching_clock_prior_checkpoint=None,actor_update_mode_v37='quality_gate')
 real_server,real_device=tf.train.Server,tf.device
 class LocalServer:
  target=''
  def __init__(self,*a,**kw):pass
 tf.train.Server=LocalServer
 tf.device=lambda name:real_device('/cpu:0' if isinstance(name,str) and name.startswith('/job:learner/') else name)
 try:a,sess,*unused=d.setup(cfg,'learner',0)
 finally:tf.train.Server,tf.device=real_server,real_device
 try:
  sess.run(tf.global_variables_initializer());a.init_target()
  variables={v.op.name:v for v in tf.global_variables()}
  names=[n for n in variables if not c.optimizer_name(n) and n not in ['epoch','global_step']]
  reader=tf.train.NewCheckpointReader(cfg['compatible_v31_checkpoint'])
  canonical=tf.train.Saver(var_list={n:variables[n] for n in names})
  canonical.restore(sess,cfg['compatible_v31_checkpoint'])
  rng=np.random.RandomState(831);s=rng.uniform(.05,.8,(512,149)).astype(np.float32)
  act=rng.uniform(-.8,.8,(512,2)).astype(np.float32)
  fd={a.s0:s,a.s1:s*.99,a.action:act,a.target_actor_policy:act,
      a.reward:np.ones((512,1),np.float32),a.terminal:np.zeros((512,1),np.float32),
      a.transition_discount:np.full((512,1),.98,np.float32),a.is_training:False,
      a.is_concrete:True,a.s0_delta:np.zeros_like(s)}
  probes=[a._moving_action,a.critic_out,a.critic_out2,a.target_actor_out,
          a.target_critic_actor_out,a.target_critic_actor_out2]
  expected=sess.run(probes,fd)
  proof=c.initialize(a,cfg)
  actual=sess.run(probes,fd)
  for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)
  snapshot=c.policy_snapshot(a)
  actor_slots=[v for n,v in variables.items() if n.startswith('abstract_actor/') and c.optimizer_name(n)]
  old_slots=sess.run(actor_slots)
  qvars=[v for n,v in variables.items() if n.startswith(('critic/','critic2/')) and not c.optimizer_name(n)]
  qb=sess.run(qvars)
  for _ in range(3):sess.run(a.critic_train_op,fd);a.target_update()
  c.assert_policy_frozen(a,snapshot)
  for x,y in zip(old_slots,sess.run(actor_slots)):np.testing.assert_array_equal(x,y)
  np.testing.assert_array_equal(sess.run(a._moving_action,fd),expected[0])
  assert any(not np.array_equal(x,y) for x,y in zip(qb,sess.run(qvars)))
  assert sess.run(a.global_step)==6
  assert bool(sess.run(a.timing_learning_enabled)) and np.std(expected[0][:,1])>1e-5
  holdout=c.Holdout(cfg);holdout.tensors(a)
  for i in range(512):
   holdout.store([s[i],act[i],np.array([1.],np.float32),s[i]*.99,np.array([0.],np.float32),np.array([.08],np.float32),np.array([.98],np.float32)],i%24,i*17)
  first=holdout.diagnose(a,0);graph_ops=len(tf.get_default_graph().get_operations())
  second=holdout.diagnose(a,1000)
  assert second['count']==512 and np.isfinite(second['rmse_q1'])
  assert np.isfinite(second['period_rank_flip_fraction']) and not c.actor_gate(cfg,6000,np.full((24,5),128),holdout.history)[0]
  assert graph_ops==len(tf.get_default_graph().get_operations()),'Diagnostic graph growth'
  # This is an operation-wiring check, not a factual-data quality-gate approval.
  gradient=tf.gradients(a.a_loss,a._moving_logits)[0]
  g=sess.run(gradient,fd);assert np.all(np.mean(np.abs(g),axis=0)>0)
  sess.run(a.actor_train_op,fd);assert sess.run(a.global_step)==7
  assert any(not np.array_equal(x,y) for x,y in zip(snapshot[1],sess.run(snapshot[0])))
  with tempfile.TemporaryDirectory(prefix='v31-compatible-test-') as tmp:
   saver=tf.train.Saver();path=saver.save(sess,str(pathlib.Path(tmp)/'model'),write_meta_graph=False)
   before=sess.run(tf.global_variables());saver.restore(sess,path)
   for x,y in zip(before,sess.run(tf.global_variables())):np.testing.assert_array_equal(x,y)
  result.update(status='PASS',copied_tensors=len(proof['copied_names']),
    actor_critic_target_probe_max_error=0.,actor_bn_slots_exact_while_frozen=True,
    factual_diagnostic_graph_executed=True,diagnostic_graph_operations_constant=True,
    critics_updated=True,time_forward_active=True,time_action_std=float(expected[0][:,1].std()),
    both_head_gradient_mean_abs=np.mean(np.abs(g),axis=0).tolist(),
    optimizer_operations=dict(critic_iterations=3,critic_optimizer_ops=6,actor_updates=1),
    save_reload_exact=True,scope='Synthetic real TF1 graph; no TCP, no empirical quality-gate approval')
 finally:sess.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
 result={'status':'NOT_RUN'}
 try:verify(result)
 except Exception:result.update(status='FAIL',traceback=traceback.format_exc())
 out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
 return 0 if result['status']=='PASS' else 2
if __name__=='__main__':sys.exit(main())
