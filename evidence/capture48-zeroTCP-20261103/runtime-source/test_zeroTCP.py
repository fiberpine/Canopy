import pathlib,json,unittest,ast,collections,numpy as np
HERE=pathlib.Path(__file__).parent;E=HERE.parent if HERE.name=='runtime-source' else HERE;C=E.parent/'common-period-capture48-20261103'
def read(n):return json.loads((E/n).read_text(encoding='utf-8'))
class ZeroTCPContracts(unittest.TestCase):
 def test_capacity_clock_and_independent_log_integral(self):
  metrics={r['run']:next(n['capacity_mbps'] for n in r['network'] if n['window']=='full') for r in json.loads((C/'captured-run-metrics.json').read_text(encoding='utf-8'))};count=0
  for host in read('exports-metadata.json'):
   for m in host['metadata']:
    self.assertAlmostEqual(m['capacity_mean_mbps'],metrics[m['job']['run']],places=10);self.assertGreaterEqual(m['relative_state_time_range'][0],0);self.assertLessEqual(m['relative_state_time_range'][1],61);count+=m['rows']
  self.assertEqual(count,63156)
 def test_whole_run_folds_without_temporal_leakage(self):
  jobs={m['job']['run']:m['job'] for h in read('exports-metadata.json') for m in h['metadata']};records=read('probe-results.json')['records'];self.assertEqual(len(records),32)
  for r in records:
   ids={a['run'] for a in r['per_test_run']};self.assertEqual(r['train_rows']+r['test_rows'],63156);self.assertEqual(r['train_runs']+r['test_runs'],48);self.assertEqual(len(ids),r['test_runs'])
   if r['fold'].startswith('repeat'):self.assertTrue(all(jobs[i]['repeat']==(1 if r['fold']=='repeat0_to1' else 0) for i in ids))
   else:self.assertTrue(all(jobs[i]['condition']==r['fold'][len('leave_condition_'):] for i in ids))
 def test_constant_target_score_and_weighted_loss(self):
  tree=ast.parse((HERE/'analyze.py').read_text(encoding='utf-8'));selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['weighted','scores']];ns={'np':np};exec(compile(ast.Module(body=selected,type_ignores=[]),'metric-functions','exec'),ns)
  x=np.array([[1.,2.],[1.,4.]]);p=np.array([[1.,3.],[1.,3.]]);r=ns['scores'](x,p,np.ones(2),np.array([1.,3.]));self.assertEqual(r['R2'],[None,0.]);self.assertEqual(r['MAE_mbps'],[0.,1.]);self.assertEqual(r['baseline_MAE_mbps'],[0.,1.])
 def test_full_coverage_and_readonly_proof(self):
  s=read('training-support.json');r=read('results.json');self.assertEqual(sum(a['rows'] for a in s['full_training']),711285);self.assertEqual(sum(a['high_SRTT_rows'] for a in s['full_training'] if a['period_ms']==20),r['A20_full_high_SRTT_rows']);self.assertEqual(s['replay_sha256_before'],s['replay_sha256_after']);p=read('model-proof.json');self.assertEqual(p['variables_before'],p['variables_after']);self.assertEqual(p['Canopy_optimizer_operations'],0)
 def test_same_states_under_three_period_queries(self):
  grouped=collections.defaultdict(list)
  for p in read('gradient-points.json'):grouped[p['run'],p['state_index']].append(p)
  self.assertEqual(len(grouped),1536)
  for ps in grouped.values():self.assertEqual({p['queried_period_ms'] for p in ps},{20,40,80});self.assertEqual(len({p['SRTT_over_min'] for p in ps}),1);self.assertEqual(len({p['own_A20_window'] for p in ps}),1)
 def test_review_claims_keep_observed_ranking(self):
  checks=read('review-checks.json');self.assertAlmostEqual(checks['run48']['Q_on_label_slope'],.12270975936013179,places=12);self.assertAlmostEqual(checks['run48']['correlation'],.3140295205886854,places=12);self.assertEqual(sorted(r['descriptive_best_period_ms'] for r in checks['four_scans']),[40,40,80,80])
if __name__=='__main__':unittest.main()
