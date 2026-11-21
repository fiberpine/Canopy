import unittest
from unittest import mock
import numpy as np
from frameworks.joint_healthy.configuration import configuration
from frameworks.joint_healthy.continuous import exploration_inputs, schema
from frameworks.joint_healthy.core import validate, period_action, period_ms, interval_return, wire
from frameworks.joint_healthy import identity


class ContinuousContract(unittest.TestCase):
    def test_new_schema_does_not_accept_old_checkpoints_by_shape(self):
        old=configuration('g4');new=configuration('continuous-v4')
        self.assertNotEqual(schema(old),schema(new))
        with self.assertRaises(ValueError):validate(dict(new,schema=old['schema']))

    def test_preservation_and_parameterization_cannot_be_silently_disabled(self):
        cfg=configuration('continuous-v4')
        for change in ({'window_bn_statistics':'adaptive_moving'},{'teacher_anchor_clock_weighting':'constant'},
                       {'time_parameterization':'softsign_log_v1'},{'fixed_mtp_ms':80.},
                       {'joint_period_probe_probability':0.},{'time_logit_exploration_std':float('nan')}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate(dict(cfg,**change))

    def test_seeded_proposals_are_continuous_and_cover_the_range(self):
        cfg=configuration('continuous-v4');a=np.random.RandomState(51);b=np.random.RandomState(51)
        draws=[exploration_inputs(cfg,a) for _ in range(10000)]
        self.assertEqual(draws,[exploration_inputs(cfg,b) for _ in range(10000)])
        fractions=np.array([r['global_fraction'] for r in draws])
        self.assertEqual(len(np.unique(fractions)),10000)
        counts=np.histogram(fractions,np.linspace(0,1,6))[0]
        self.assertTrue(np.all(counts>1750),counts)
        duration=20.*8.**fractions
        self.assertLess(duration.min(),20.05);self.assertGreater(duration.max(),159.8)
        self.assertFalse(np.any(np.isin(duration,[20,40,80,120,160])))

    def test_wire_keeps_arbitrary_continuous_periods_and_once_cost(self):
        for ms in (20.035,31.417,73.829,159.273):
            action=period_action(ms)
            self.assertAlmostEqual(period_ms(action),ms)
            self.assertEqual(int(wire(0,np.array([0.,action])).rstrip(b'\0').split()[-1]),round(ms*1000))
            r,d=interval_return(.8,ms/1000,.01)
            self.assertAlmostEqual(r,.8*(1-d)/.005-.01)

    def test_old_admission_cannot_release_continuous_policy(self):
        cfg=configuration('continuous-v4')
        with self.assertRaises(ValueError):identity.check_admission({'status':'PASS_PREREGISTERED_G0_G4_COST_PRESERVE_INITIALIZATION'},cfg)

    def test_source_and_algorithm_receipt_need_actual_mechanism_evidence(self):
        cfg=configuration('continuous-v4');impl={'fixture':True}
        receipt=dict(status='PASS_PREREGISTERED_CONTINUOUS_V4_INITIALIZATION',implementation=impl,
            contract=identity.continuous_contract(cfg),registration='fixture',evidence='fixture',
            offline_graph_pass=True,offline_small_advantage_pass=True,initial_real_input_parity_pass=True,
            same_decision_exploration_pass=True,short_tcp_mechanism_pass=True,initial_frozen_grid_health_pass=True)
        with mock.patch.object(identity,'implementation_identity',return_value=impl):
            identity.check_admission(receipt,cfg)
            with self.assertRaises(ValueError):identity.check_admission(dict(receipt,short_tcp_mechanism_pass=False),cfg)
            with self.assertRaises(ValueError):identity.check_admission(receipt,dict(cfg,time_actor_lr=.001))


if __name__=='__main__':unittest.main()
