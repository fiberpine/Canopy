import unittest
import numpy as np
from frameworks.joint_healthy.configuration import configuration
from frameworks.joint_healthy.core import interval_return
from frameworks.joint_healthy.replay import Replay


def transition(sequence,episode,end=False):
    state=np.zeros(82,np.float32);state[0]=episode*1000+sequence
    next_state=state.copy();next_state[0]+=1
    reward,discount=interval_return(.9,.02)
    return [state,np.array([0.,-1.],np.float32),np.array([reward],np.float32),next_state,
            np.array([float(end)],np.float32),np.array([.02]),np.array([discount],np.float32)]


class StartupCoverage(unittest.TestCase):
    def replay(self,layout):
        return Replay(dict(configuration(),num_actors=1,memsize=20000,batch_size=8,holdout_layout=layout))

    def test_fixed_legacy_split_excludes_every_startup_but_rotated_split_does_not(self):
        old=self.replay('connection_start_fixed_legacy');new=self.replay('episode_rotated_blocks_v1')
        for episode in range(40):
            for seq in range(1,25):
                for replay in (old,new):replay.append(transition(seq,episode,seq==24),0,episode,seq)
        cold=np.array([i for i in range(960) if i%24<8])
        self.assertTrue(np.all(old.actor_ids[cold]==-1))
        self.assertGreater(int(np.sum(new.actor_ids[cold]>=0)),240)
        self.assertGreater(len(new.holdout[0]),0)
        self.assertTrue(all(new.birth[i]%17!=0 for i in new.pools[0]))

    def test_rotated_short_blocks_stay_contiguous_and_inside_their_connection(self):
        replay=self.replay('episode_rotated_blocks_v1')
        for episode in range(8):
            for seq in range(1,301):replay.append(transition(seq,episode,seq==300),0,episode,seq)
        blocks=[block for group in replay.short_returns for block in group]
        self.assertGreater(len(blocks),5)
        for block in blocks:
            ids=[int(row[0][0]) for row in block]
            self.assertEqual(ids,list(range(ids[0],ids[0]+len(ids))))
            self.assertEqual(ids[0]//1000,ids[-1]//1000)
            self.assertTrue(all(int(replay.actor_ids[i])<0 for i in range(2400)
                                if int(replay.arrays[0][i,0]) in ids))
