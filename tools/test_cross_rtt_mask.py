"""Pure input-intervention checks; actual TF path is tested by the bounded A/A pilot."""
import importlib.util
import pathlib
import random
import unittest
import numpy as np

path = pathlib.Path(__file__).resolve().parent/'scalar_learning_check/runtime/cross_rtt_mask_replay.py'
spec = importlib.util.spec_from_file_location('cross_rtt_mask',str(path))
mask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask)


class MaskInputsTest(unittest.TestCase):
    def test_r200_is_retained_and_equal_number_actor_groups_masked(self):
        rtts = np.tile([20,50,100,200],6)
        for arm in ('full','mask_short','mask_mid'):
            w=mask.loss_weights(rtts,arm).ravel()
            np.testing.assert_array_equal(w[rtts==200],np.ones(6))
            self.assertEqual(int(w.sum()),24 if arm=='full' else 12)

    def test_batch_contributions_rotate_without_treatment_affecting_rng(self):
        pools=[list(range(i*1000,i*1000+100)) for i in range(24)]
        streams=[]
        for unused in range(3):
            rng=random.Random(3)
            streams.append([mask.balanced_indices(pools,t,rng) for t in range(3)])
        for a,b in zip(streams[0],streams[1]):np.testing.assert_array_equal(a,b)
        contributions=np.zeros(24,dtype=int)
        for batch in streams[0]:contributions+=np.bincount(batch//1000,minlength=24)
        np.testing.assert_array_equal(contributions,np.full(24,64))

    def test_unknown_arm_cannot_silently_become_full(self):
        with self.assertRaises(KeyError):mask.loss_weights([200],'unknown')

    def test_own_only_has_no_other_rtt_critic_loss(self):
        np.testing.assert_array_equal(mask.loss_weights([20,50,100,200],'own_only').ravel(),[0,0,0,1])


if __name__=='__main__':unittest.main()
