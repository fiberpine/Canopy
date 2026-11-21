import pathlib,sys,unittest,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
from reward_centering_audit_v38 import mass,center,estimate
class Tests(unittest.TestCase):
 def test_td_invariance(self):
  d=.995**(np.array([.02,.08,.16])/.02);r=np.array([1.,2.,3.]);q=np.array([10.,11.,12.]);next_q=q+2;b=.6;c=b/.005
  rc=center(r,d,np.zeros(3,int),{'0':b});np.testing.assert_allclose(rc+d*(next_q-c)-(q-c),r+d*next_q-q,atol=3e-14)
 def test_finite_partition_tail(self):
  d=.995**(np.array([.02,.16,.08,.02])/.02);w=np.r_[1.,np.cumprod(d)];b=.6
  self.assertAlmostEqual(float(np.dot(w[:-1],b*mass(d))+w[-1]*b/.005),b/.005,places=11)
 def test_time_weighted_density_not_decision_mean(self):
  t=np.array([.02,.16]);d=.995**(t/.02);u=np.array([.2,.8]);r=u*mass(d)-.02
  b=estimate(r,d,t,np.zeros(2,int));self.assertAlmostEqual(b['0'],np.average(u,weights=t))
 def test_unknown_environment_rejected(self):
  with self.assertRaises(ValueError):center([1],[.99],[3],{'0':.5})
 def test_invalid_values_rejected(self):
  for d in [[1.],[0.],[float('nan')]]:
   with self.assertRaises(ValueError):mass(d)
if __name__=='__main__':unittest.main()
