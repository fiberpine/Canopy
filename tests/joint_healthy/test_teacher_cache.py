import unittest
import numpy as np
from frameworks.joint_healthy.configuration import configuration
from frameworks.joint_healthy.core import interval_return,validate
from frameworks.joint_healthy.replay import Replay


class SameStateTeacherCache(unittest.TestCase):
    def config(self):
        cfg=configuration('joint')
        cfg.update(num_actors=2,memsize=64,batch_size=8,teacher_anchor_weight=10.,
                   warmup=8,minimum_enqueued_per_actor=4,warmup_physical_seconds=.16)
        return cfg

    def row(self,value,end=False):
        r,d=interval_return(.9,.02,.01)
        return [np.full(82,value,np.float32),np.zeros(2,np.float32),np.array([r]),np.ones(82),
                np.array([float(end)]),np.array([.02]),np.array([d])]

    def test_cache_follows_sample_after_ring_evictions_and_resets(self):
        replay=Replay(self.config())
        for episode in range(4):
            for sequence in range(1,21):
                for actor in range(2):
                    value=(actor*20+sequence)/100.
                    replay.append(self.row(value,sequence==20),actor,episode,sequence,teacher_action=value)
        for _ in range(20):
            batch=replay.sample();self.assertEqual(len(batch),8)
            np.testing.assert_array_equal(batch[0][:,0],batch[7][:,0])
        self.assertEqual(replay.credit.inserted,160)
        self.assertEqual(replay.snapshot()['teacher_actions'].shape,(64,1))

    def test_required_label_missing_fails_before_earning_credit(self):
        replay=Replay(self.config())
        for invalid in (None,float('nan'),2.):
            with self.assertRaises(ValueError):replay.append(self.row(.1),0,0,1,teacher_action=invalid)
            self.assertEqual(replay.credit.inserted,0)

    def test_legacy_replay_still_returns_seven_tensors(self):
        cfg=self.config();cfg['teacher_anchor_weight']=0;replay=Replay(cfg)
        for sequence in range(1,21):
            for actor in range(2):replay.append(self.row(.1),actor,0,sequence)
        self.assertEqual(len(replay.sample()),7)

    def test_unsupported_statistics_and_clock_contract_rejected(self):
        for key,value in (('window_bn_statistics','restore_final_later'),('teacher_anchor_clock_weighting','learn_to_hide_loss'),('window_actor_lr',0.)):
            cfg=self.config();cfg[key]=value
            with self.assertRaises(ValueError):validate(cfg)
