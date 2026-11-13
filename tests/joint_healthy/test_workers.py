import unittest
from frameworks.joint_healthy.backend import completed_worker


class WorkerOutcome(unittest.TestCase):
    def test_planned_calibration_is_not_a_trained_endpoint(self):
        r=dict(status='PARTIAL_CALIBRATION_ONLY',exposure_complete=True,producer_done={0:100,1:100})
        cfg=dict(release_joint=False,num_actors=2)
        self.assertTrue(completed_worker(r,cfg))
        self.assertFalse(completed_worker(r,dict(cfg,release_joint=True)))

    def test_partial_or_failed_actor_does_not_pass_diagnostic(self):
        cfg=dict(release_joint=False,num_actors=2)
        self.assertFalse(completed_worker(dict(status='PARTIAL_CALIBRATION_ONLY',exposure_complete=False,producer_done={0:100,1:100}),cfg))
        self.assertFalse(completed_worker(dict(status='PARTIAL_CALIBRATION_ONLY',exposure_complete=True,producer_done={0:100}),cfg))
        self.assertFalse(completed_worker(dict(status='FAILED'),cfg))
