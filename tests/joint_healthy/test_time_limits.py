import unittest
from types import SimpleNamespace
import numpy as np
from frameworks.joint_healthy.core import bootstrap_terminal,validate
from frameworks.joint_healthy.configuration import configuration
from frameworks.joint_healthy.model import Model


class ArtificialTimeLimits(unittest.TestCase):
    def test_legacy_checkpoint_contract_keeps_historical_terminal_mask(self):
        end=np.array([[0.],[1.]],np.float32)
        self.assertIs(bootstrap_terminal({},end),end)
        self.assertTrue(np.array_equal(bootstrap_terminal(configuration(),end),np.zeros_like(end)))
        self.assertEqual(end[1,0],1.)

    def test_model_feed_masks_bootstrap_only_and_keeps_real_next_state(self):
        batch=[np.full((2,1),float(i),np.float32) for i in range(7)]
        batch[4][:]=1.
        model=SimpleNamespace(cfg=configuration(),s='s',executed='a',reward='r',s1='s1',terminal='terminated',discount='discount')
        feed=Model.feed(model,batch)
        self.assertIs(feed['s1'],batch[3]);self.assertIs(feed['discount'],batch[6])
        self.assertTrue(np.all(feed['terminated']==0.));self.assertTrue(np.all(batch[4]==1.))

    def test_unknown_bootstrap_mode_is_rejected(self):
        with self.assertRaises(ValueError):validate(dict(configuration(),time_limit_handling='silently_join_episodes'))
