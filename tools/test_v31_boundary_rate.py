import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
from boundary_rate_audit_v38 import BoundaryRate
class BoundaryRateTest(unittest.TestCase):
 def test_constant_under_sparse_and_dense_boundaries(self):
  for tick in [.02,.08,.16]:
   x=BoundaryRate();out=[]
   for i in range(30):
    z=x.add(i*tick,round(3000000*i*tick),.1)
    if z:out.append(z)
   self.assertTrue(out)
   for z in out:self.assertAlmostEqual(z['rate'],3000000,places=7);self.assertGreaterEqual(z['span_s']+1e-12,.16)
 def test_recovery_burst_is_not_a_capacity_peak(self):
  x=BoundaryRate();raw=[];out=[]
  for i in range(25):
   # Alternate delayed cumulative ACK release every160ms; exact delivered trend3MB/s.
   b=(i//8)*480000;z=x.add(i*.02,b,.02)
   if z:out.append(z)
  self.assertAlmostEqual(max(z['rate'] for z in out),3000000,places=7)
 def test_rtt_increase_uses_older_real_boundaries(self):
  x=BoundaryRate()
  for i in range(20):x.add(i*.02,i*60000,.02)
  result=x.add(.4,1200000,.3)
  self.assertIsNotNone(result);self.assertGreaterEqual(result['span_s']+1e-12,.3)
 def test_reset_rejected_and_startup_explicit(self):
  x=BoundaryRate();self.assertIsNone(x.add(0,100,.02));self.assertIsNone(x.add(.08,200,.02))
  with self.assertRaises(ValueError):x.add(.1,0,.02)
if __name__=='__main__':unittest.main()
