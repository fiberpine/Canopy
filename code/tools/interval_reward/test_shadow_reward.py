"""Numerical/fixture tests only; fixtures are never empirical TCP evidence."""
import importlib.util,json,pathlib,sys,unittest
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from shadow_reward_audit import pack,mass,RAW,SHADOW

class Tests(unittest.TestCase):
 def test_original_149_encoder_exact(self):
  root=HERE.parents[1]/'frameworks/mtpjoint'
  self.assertTrue(root.exists(),'Packaged149-state source must be present')
  sys.path.insert(0,str(root));import joint_mtp
  cfg=dict(mtp_min_ms=20,mtp_max_ms=160,timing_context_features=True,decision_observation_clock=True,time_statistics_v27=True,time_statistics_mode='rates',feedback_context_v31=True,joint_state_dim=149)
  class Features:
   def __init__(self,x):self.x=x
   def features(self):return self.x
  rng=np.random.RandomState(48)
  for dt in [0.,.020003,.080001,.160238,1.2]:
   for planned in [20.,40.,80.,120.,160.]:
    history=rng.uniform(0,2,70).astype(np.float32);gaps=rng.uniform(0,.2,10).astype(np.float32);gaps[:2]=0
    context=rng.normal(0,.1,4).astype(np.float32);feedback=rng.normal(0,.1,53).astype(np.float32)
    expected=joint_mtp.encode_state(history,dt,planned,cfg,Features(context),gaps,Features(feedback))
    np.testing.assert_array_equal(pack(history,gaps,context,feedback,dt,planned),expected)
 def test_constant_density_integral_split_merge(self):
  gamma=.995;first=.0329;second=.1272;u=.711
  self.assertAlmostEqual(float(u*mass(first,gamma)+gamma**(first/.02)*u*mass(second,gamma)),float(u*mass(first+second,gamma)),places=12)
 def test_reward_identity_with_once_cost(self):
  gamma=.995;u=.51;dt=.080003;c=.02
  reward=float(u*mass(dt,gamma)-c)
  self.assertAlmostEqual((reward+c)/float(mass(dt,gamma)),u,places=14)
 def test_schema_is_explicit_reward_change(self):
  self.assertNotEqual(RAW,SHADOW);self.assertIn('same-149',SHADOW)
 def test_existing_empirical_summaries_not_fabricated_dense(self):
  path=HERE.parents[1]/'evidence/interval-reward-20261025/existing-measurement-checks.json'
  self.assertTrue(path.exists(),'Packaged evidence must be present')
  z=json.loads(path.read_text());self.assertEqual(z['new_tcp_runs'],0);self.assertFalse(z['actor_release'])
  self.assertEqual(len(z['check2']['runs']),24);self.assertTrue(z['check1']['passed'])

if __name__=='__main__':unittest.main()
