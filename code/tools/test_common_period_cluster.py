import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
from common_period import SCHEMA,PERIODS,discount
from common_period_transport import pack_row,unpack_row,snapshot_bytes,decode_snapshot
from common_period_calibration import calibrate

class FakeSession:
    def run(self,fetch,feed):
        n=len(feed['s'])
        if fetch=='mu':return np.zeros((n,1),np.float32)
        return [np.full((n,1),2.,np.float32),np.full((n,1),2.,np.float32)]
class CalModel:
    s='s';action='a';mu='mu';q=['q1','q2'];session=FakeSession()
    def initialize_centering(self,K,n0):self.K=K;self.n0=n0
    def calibrate_bias_once(self,b):self.offset=b

class ClusterContracts(unittest.TestCase):
    def test_binary_boundaries_bitwise(self):
        rng=np.random.RandomState(3);row=dict(s0=rng.randn(149).astype(np.float32),s1=rng.randn(149).astype(np.float32),action=np.array([-.1,.333],np.float32),reward=3.4)
        decoded=unpack_row(pack_row(row))
        for k in ['s0','s1','action']:np.testing.assert_array_equal(row[k],decoded[k])
        self.assertEqual(decoded['reward'],3.4)
    def test_bad_state_rejected(self):
        row=pack_row(dict(s0=np.zeros(148),s1=None,action=np.zeros(2)))
        with self.assertRaises(ValueError):unpack_row(row)
    def test_snapshot_no_pickle(self):
        snap=dict(schema=SCHEMA,version=3,reward_checksum='r',values={'actor/w':np.arange(6,dtype=np.float32).reshape(2,3)})
        decoded=decode_snapshot(snapshot_bytes(snap));self.assertEqual(decoded['version'],3)
        np.testing.assert_array_equal(decoded['values']['actor/w'],snap['values']['actor/w'])
    def test_complete_calibration_and_tail_bias(self):
        rows=[];g=discount(1.);r=.7*(1-g)/.005;s=np.zeros(149,np.float32)
        for actor in range(24):
            for block,t in enumerate(PERIODS):
                connection='%d-c%d'%(actor,int(block==4))
                rows += [dict(actor=actor,block=block,period_ms=t,connection=connection,next_connection=connection,duration=1.,reward=r,discount=g,kind='continuing',s0=s,s1=s) for _ in range(40)]
        m=CalModel();audit=calibrate(m,rows)
        self.assertAlmostEqual(m.K,140.,places=8);self.assertAlmostEqual(audit['global_bias_offset'],-2.,places=7)
        self.assertLess(abs(audit['independent_check']['residual_mean']),1e-6)
        self.assertGreater(audit['independent_check']['tail_abs_mean'],0)
        with self.assertRaises(ValueError):calibrate(CalModel(),rows[:-40])
if __name__=='__main__':unittest.main()
