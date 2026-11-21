import pathlib,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
from factual_mtp_probe_v37 import Probe

class Tests(unittest.TestCase):
 def config(self):return dict(no_training=True,mtp_min_ms=20,mtp_max_ms=160,
  factual_mtp_probe_v37=dict(period_ms=160,trigger_s=1,horizons_s=[1,2]))
 def test_exactly_one_change_and_discount(self):
  with tempfile.TemporaryDirectory() as d:
   p=Probe(self.config(),pathlib.Path(d));s=np.zeros(149,np.float32);policy=np.array([.2,0],np.float32);changes=[]
   for i in range(6):
    action,changed=p.choose(s,policy,float(i),i+1);changes.append(changed)
    self.assertEqual(action[0],policy[0]);np.testing.assert_array_equal(policy,[np.float32(.2),0])
    p.transition(s,action,1.,s,1.,.9)
   p.finish();self.assertEqual(sum(changes),1)
   self.assertAlmostEqual(p.endpoints['2']['observed_return'],1.9)
   self.assertAlmostEqual(p.endpoints['2']['tail_discount'],.81)
 def test_opt_in_window_delta_is_one_action_and_clipped(self):
  c=self.config();c['factual_mtp_probe_v37']['window_action_delta_v38']=.1
  p=Probe(c,pathlib.Path('.'));s=np.zeros(149,np.float32);a=np.array([.95,0],np.float32)
  x,changed=p.choose(s,a,1.,1);self.assertTrue(changed);self.assertEqual(x[0],1.)
  y,changed=p.choose(s,a,2.,2);self.assertFalse(changed);np.testing.assert_array_equal(y,a)
  self.assertAlmostEqual(p.event['executed_window_delta'],.05,places=6)
 def test_unsafe_window_probe_rejected(self):
  for value in [.2,float('nan')]:
   c=self.config();c['factual_mtp_probe_v37']['window_action_delta_v38']=value
   with self.assertRaises(ValueError):Probe(c,pathlib.Path('.'))
 def test_training_rejected(self):
  c=self.config();c['training_mode']=True
  with self.assertRaises(ValueError):Probe(c,pathlib.Path('.'))
 def test_incomplete_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Probe(self.config(),pathlib.Path(d))
   with self.assertRaises(RuntimeError):p.finish()
if __name__=='__main__':unittest.main()
