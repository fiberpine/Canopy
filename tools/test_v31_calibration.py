"""Local numerical/guard tests; never claims TF/TCP validation."""
import copy
import json
import pathlib
import sys
import unittest
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'frameworks/mtpjoint'))
from compatible_calibration_v37 import validate_metadata, validate_tensor, actor_gate, optimizer_name
import joint_mtp, update_credit
from decision_clock_v21 import validate, wire


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'templates/mtpjoint-train.json').read_text())
        self.cfg.update(compatible_v31_checkpoint='explicit149', matching_clock_prior_checkpoint=None,
            actor_update_mode_v37='quality_gate', actor_release_after_calibration_v37=True)
        self.meta = dict(self.cfg)
        self.markers = {k: 1 for k in ['time_statistics_schema', 'interval_rate_schema',
            'feedback_context_schema', 'feedback_context_mode_schema', 'timer_boundary_schema', 'duration_head_schema']}
        self.markers['timing_learning_enabled'] = True
        self.coverage = np.full((24, 5), 128)
        row = dict(count=1024, rmse_q1=2., rmse_q2=2.1, twin_q_gap_mean=1.,
            actor_grad_norm=3., q1_mean=50., target_mean=50., period_rank_flip_fraction=.05,
            window_single_boundary_fraction=.01,mtp_single_boundary_fraction=.01,
            by_period=[dict(count=200) for _ in range(5)])
        self.history = [copy.deepcopy(row) for _ in range(3)]

    def test_metadata_and_provenance(self):
        validate_metadata(self.meta, self.cfg, self.markers);validate(self.cfg)

    def test_wrong_dimension(self):
        with self.assertRaises(ValueError):validate_metadata(dict(self.meta,joint_state_dim=96),self.cfg,self.markers)

    def test_wrong_interval_marker(self):
        with self.assertRaises(ValueError):validate_metadata(self.meta,self.cfg,dict(self.markers,interval_rate_schema=0))

    def test_wrong_tensor_shape_and_nonfinite(self):
        with self.assertRaises(ValueError):validate_tensor('actor/fc1/w',np.zeros((96,256)),(149,256))
        with self.assertRaises(ValueError):validate_tensor('critic/dense/bias',np.array([np.nan]),(1,))

    def test_missing_bn_schema(self):
        m=dict(self.markers);del m['feedback_context_schema']
        with self.assertRaises(ValueError):validate_metadata(self.meta,self.cfg,m)

    def test_wrong_reward_and_action_range(self):
        for key,value in [('decision_cost',0),('gamma',.9),('mtp_max_ms',320),('time_head_nonlinearity','tanh')]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_metadata(dict(self.meta,**{key:value}),self.cfg,self.markers)

    def test_closed_gate_does_not_require_masking_time(self):
        c=dict(self.cfg,actor_release_after_calibration_v37=False)
        self.assertFalse(actor_gate(c,6000,self.coverage,self.history)[0])
        self.assertTrue(self.markers['timing_learning_enabled'])

    def test_missing_updates_or_coverage(self):
        self.assertFalse(actor_gate(self.cfg,4999,self.coverage,self.history)[0])
        c=self.coverage.copy();c[0,0]=127
        self.assertFalse(actor_gate(self.cfg,6000,c,self.history)[0])
        c=c.astype(float);c[0,0]=np.nan
        self.assertFalse(actor_gate(self.cfg,6000,c,self.history)[0])

    def test_nonfinite_diagnostic_blocks_gate(self):
        for key in ['rmse_q1','target_mean','actor_grad_norm']:
            h=copy.deepcopy(self.history);h[-1][key]=float('nan')
            self.assertFalse(actor_gate(self.cfg,6000,self.coverage,h)[0])

    def test_large_target_drift(self):
        h=copy.deepcopy(self.history);h[-1]['q1_mean']=h[-1]['target_mean']=80
        self.assertEqual(actor_gate(self.cfg,6000,self.coverage,h)[1],'target_drift')

    def test_instability_and_gradient_limits(self):
        for key,value in [('period_rank_flip_fraction',.8),('actor_grad_norm',1000),('rmse_q1',100),('mtp_single_boundary_fraction',.99),('window_single_boundary_fraction',.99)]:
            h=copy.deepcopy(self.history);h[-1][key]=value
            self.assertFalse(actor_gate(self.cfg,6000,self.coverage,h)[0])

    def test_explicit_release_with_checks(self):
        self.assertTrue(actor_gate(self.cfg,6000,self.coverage,self.history)[0])

    def test_count_control_uses_same_coverage(self):
        c=dict(self.cfg,actor_update_mode_v37='legacy_count_gate')
        self.assertTrue(actor_gate(c,6000,self.coverage,[])[0])
        self.assertFalse(actor_gate(c,4000,self.coverage,[])[0])

    def test_original_initialization_still_valid(self):
        c=json.loads((ROOT/'templates/mtpjoint-train.json').read_text());validate(c)
        c['compatible_v31_checkpoint']='ambiguous'
        with self.assertRaises(AssertionError):validate(c)

    def test_optimizer_exclusion(self):
        self.assertTrue(optimizer_name('critic/fc1/kernel/Adam_1'))
        self.assertFalse(optimizer_name('target_critic/fc1/kernel'))

    def test_scratch_fixed_configuration_has_no_prior(self):
        c=dict(self.cfg,compatible_v31_checkpoint=None,initialize_joint_from_scratch=True,
               timing_policy_mode='fixed',fixed_mtp_ms=20,fixed_period_train_window=True,
               fixed_window_exploration_from_start=True)
        validate(c)
        import staged_timing
        staged_timing.validate(c)
        with self.assertRaises(AssertionError):
            validate(dict(c,matching_clock_prior_checkpoint='unexpected-prior'))

    def test_initial_window_exploration_rejects_legacy_and_learned(self):
        import staged_timing
        with self.assertRaises(ValueError):
            staged_timing.validate(dict(self.cfg,fixed_window_exploration_from_start=True))

    def test_duration_discount_cost_and_partition(self):
        a=joint_mtp.Interval(decision_cost=.02);b=joint_mtp.Interval(decision_cost=.02)
        a.add(.7,.16)
        for _ in range(8):b.add(.7,.02)
        self.assertAlmostEqual(a.discount,.995**8,14)
        self.assertAlmostEqual(a.reward,b.reward,12)
        self.assertAlmostEqual(joint_mtp.bellman(1,.5,1,20,30),1)
        self.assertAlmostEqual(joint_mtp.bellman(1,.5,0,20,30),11)

    def test_no_wall_clock_or_duplicate_credit(self):
        args=(20000,12288,.1,5760,3600,'physical_exposure',.08)
        self.assertEqual(update_credit.limit(*args),8640)
        self.assertEqual(update_credit.limit(*args),8640)
        self.assertEqual(update_credit.limit(20000,12288,.1,3600,3600,'physical_exposure',.08),0)

    def test_native_action_definition_unchanged(self):
        for ms in [20,40,80,120,160]:
            a=np.array([.25,joint_mtp.mtp_action(ms,self.cfg)])
            command=wire(7,a,self.cfg).decode().strip('\0').split()
            self.assertEqual(int(command[1]),int(100*4**.25))
            self.assertEqual(int(command[3]),ms*1000)


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CalibrationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
