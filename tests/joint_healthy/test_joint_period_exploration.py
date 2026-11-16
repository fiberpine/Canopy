import unittest
import numpy as np
from frameworks.joint_healthy.core import joint_exploration,period_ms,period_action,PERIODS

class JointPeriodExploration(unittest.TestCase):
    def test_zero_probe_preserves_exact_legacy_rng_and_actions(self):
        a=np.random.RandomState(17);b=np.random.RandomState(17);cfg=dict(exploration_std=.2)
        pure=np.array([.4,-.98],np.float32)
        for _ in range(100):
            expected=np.clip(pure+b.normal(0,.2,2),-1,1)
            actual,explored,probe=joint_exploration(cfg,pure,a)
            np.testing.assert_array_equal(actual,expected);self.assertTrue(explored);self.assertFalse(probe)
        self.assertEqual(a.random_sample(),b.random_sample())

    def test_all_periods_remain_explored_near_shortest_policy(self):
        rng=np.random.RandomState(19);cfg=dict(exploration_std=.2,joint_period_probe_probability=.1)
        counts={p:0 for p in PERIODS};pure=np.array([.4,-.99999],np.float32)
        for _ in range(20000):
            action,_,probe=joint_exploration(cfg,pure,rng)
            self.assertTrue(np.all(np.abs(action)<=1))
            if probe:counts[min(PERIODS,key=lambda p:abs(period_ms(action[1])-p))]+=1
        self.assertTrue(all(v>300 for v in counts.values()),counts)
        self.assertLess(sum(counts.values()),2300)
        np.testing.assert_array_equal(pure,np.array([.4,-.99999],np.float32))

    def test_independent_fixed_diagnostic_never_probes_time(self):
        rng=np.random.RandomState(21);cfg=dict(exploration_std=.2,joint_period_probe_probability=.2,fixed_mtp_ms=120.)
        for _ in range(100):
            action,_,probe=joint_exploration(cfg,np.array([0.,period_action(120.)]),rng)
            self.assertFalse(probe);self.assertAlmostEqual(period_ms(action[1]),120.)
