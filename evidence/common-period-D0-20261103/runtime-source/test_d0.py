import unittest,numpy as np
from d0_core import reward_parts,cosine,expected_actor_period,suffix_label,ReadOnlySession,initial_state
class D0Contracts(unittest.TestCase):
    def test_original_RTT_not_packet_proxy(self):
        r=dict(acked_bytes_per_s=6000000,lost_bytes=0,dt_us=20000,rtt_count=2,avg_rtt_us=50000,srtt_us_scaled8=160000,min_rtt_us=20000)
        a=reward_parts(r,100);self.assertAlmostEqual(a['utility'],.5);self.assertAlmostEqual(a['centered'],0)
        r['rtt_count']=0;self.assertAlmostEqual(reward_parts(r,100)['utility'],1.)
        r['srtt_us_scaled8']=160007;r['min_rtt_us']=10000;self.assertEqual(reward_parts(r,100)['utility'],.625)
    def test_missing_fields_fail_closed(self):
        with self.assertRaises(ValueError):reward_parts(dict(dt_us=20000,avg_queue=5),100)
    def test_actor_period_weight_not_physical_time(self):
        w=expected_actor_period([[8,4,2,1,1],[4,2,1,1,0]])
        self.assertAlmostEqual(sum(w),1);self.assertGreater(w[0],w[-1])
    def test_zero_gradient_angle_is_unknown(self):
        self.assertIsNone(cosine([0,0],[1,1]));self.assertAlmostEqual(cosine([1,0],[-1,0]),-1.)
    def test_suffix_keeps_tail(self):
        a=suffix_label(np.full(1100,.5),np.full(1100,.02),0,10,100)
        self.assertGreaterEqual(a['H'],20);self.assertAlmostEqual(a['label'],a['tail'],places=8)
    def test_mutating_ancestor_rejected(self):
        class Op:
            def __init__(self,name,kind,parents=[]):self.name=name;self.type=kind;self.inputs=[];self.control_inputs=parents
        class Session:
            def run(self,*args,**kwargs):return 'read'
        guard=ReadOnlySession(Session());safe=Op('read','Identity');bad=Op('train','ApplyAdam');dependent=Op('hidden','Identity',[bad])
        self.assertEqual(guard.run([[safe],safe]),'read')
        with self.assertRaises(RuntimeError):guard.run(dependent)
        self.assertEqual(guard.calls,1)
    def test_complete_startup_required(self):
        with self.assertRaises(KeyError):initial_state(dict(history=[0]*70))
if __name__=='__main__':unittest.main()
