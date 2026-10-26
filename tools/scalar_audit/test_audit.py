import math, unittest
import numpy as np
import audit

class OriginalRewardTests(unittest.TestCase):
    def test_actual_source_reward_and_state_alias(self):
        x=audit.reward_alias()
        self.assertEqual(x['low']['reward'],1.)
        self.assertEqual(x['high']['reward'],1.)
        self.assertLess(x['low']['synthetic_utilization'],.04)
        self.assertGreater(x['high']['synthetic_utilization'],.8)
        self.assertTrue(x['equal_70_history'])

    def test_shared_reference_is_diagnostic_not_runtime_patch(self):
        low,high=audit.reward_alias()['common_reference_diagnostic']
        self.assertAlmostEqual(low['reward'],1/30)
        self.assertAlmostEqual(high['reward'],5/6)
        self.assertLess(low['state'][0],high['state'][0])

    def test_upstream_integer_action_mapping(self):
        fn=audit.source_method(audit.VENDOR/'rl-module/envwrapper.py','TCP_Env_Wrapper','map_action')
        self.assertEqual(fn(None,0.),100)
        self.assertEqual(fn(None,-1.),25)
        self.assertEqual(fn(None,1.),400)
        self.assertEqual(fn(None,-1e-6),99)

    def test_small_window_floor_is_not_zero_change(self):
        fn=audit.source_method(audit.VENDOR/'rl-module/envwrapper.py','TCP_Env_Wrapper','map_action')
        self.assertEqual(max(4,fn(None,-1e-6)*8//100),7)

    def test_gaussian_nonreducing_probability(self):
        p=.5*math.erfc(1/(.2*math.sqrt(2)))
        self.assertAlmostEqual(p,2.866515718791946e-7)
        self.assertLess(p*46*3600,.05)

    def test_recorded_pair_has_one_training_seed(self):
        x=audit.evaluation_analysis()
        self.assertEqual(x['runs'],144)
        self.assertEqual(x['windows'],288)
        self.assertEqual(x['training_seeds_per_arm'],1)
        self.assertTrue(x['dominated_but_higher_reward_observed_pairs'])

    def test_no_separate_random_critic_head_in_actual_checkpoint(self):
        x=audit.protocol_analysis()
        self.assertEqual(len(x['tensor_scope_evidence']['critic']),6)
        self.assertTrue(x['no_separate_random_actor_Q_head'])

if __name__=='__main__':unittest.main()
