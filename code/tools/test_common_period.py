import copy
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'frameworks/mtpjoint'))
import common_period as c


class Contracts(unittest.TestCase):
    def test_potential_and_terminal(self):
        rng = np.random.RandomState(3)
        K = 60.
        for _ in range(100):
            g = c.discount(rng.uniform(.01, .4))
            r, q = rng.randn(2)
            self.assertAlmostEqual(c.center(r,g,'continuing',K)+g*(q-K), r+g*q-K)
            self.assertEqual(c.center(r,g,'terminated',K),r-K)

    def test_decision_cost_counterexample(self):
        gs = [c.discount(t / 1000) for t in c.PERIODS]
        q = [(.5 * (1-g)/.005 - .5)/(1-g) for g in gs]
        self.assertGreater(q[-1], q[0])
        self.assertTrue(np.allclose([(.5*(1-g)/.005 - 100*(1-g))/(1-g) for g in gs],0))

    def test_reward_negative_penalty(self):
        self.assertEqual(c.utility(1,.3,2.5,40,40,20,1),-.35)
        self.assertEqual(c.utility(1,.3,2.5,80,80,20,1),-.475)
        self.assertEqual(c.utility(1,.3,2.5,0,40,20,0),-.35)
        self.assertGreater(c.utility(2,0,3,20,20,20,1),c.utility(1,0,3,20,20,20,1))

    def test_endings(self):
        self.assertEqual(c.ending('terminated','a',None,False),(True,False))
        self.assertEqual(c.ending('truncated','a','a',True),(True,True))
        self.assertEqual(c.ending('truncated','a',None,False),(False,False))
        self.assertEqual(c.ending('censored','a',None,False),(False,False))
        with self.assertRaises(ValueError):c.ending('truncated','a','b',True)

    def test_jensen_and_action_dependent_number(self):
        g = np.asarray([c.discount(.02),c.discount(.16)])
        prior = 1/(1-g.mean())
        wrong = 1/(1-c.discount(.09))
        self.assertNotAlmostEqual(prior,wrong,places=4)
        self.assertNotEqual(1/(1-c.discount(.02)),1/(1-c.discount(.03)))
        n=0.
        for _ in range(10000):n=1+g.mean()*n
        self.assertAlmostEqual(n,prior,places=8)

    def test_fixed_duration_number_convergence(self):
        for period in c.PERIODS:
            g=c.discount(period/1000.)
            value=0.
            for _ in range(10000):value=1+g*value
            self.assertAlmostEqual(value,1/(1-g),places=8)

    def test_calibration_bias(self):
        d=np.array([.005,.007,.006]);error=13.
        self.assertAlmostEqual(c.bias_correction(-error*(1-d),np.zeros(3),d),-error)
        with self.assertRaises(ValueError):c.bias_correction([1],[0],[1])

    def row(self):
        return dict(s0=np.zeros(149),s1=np.ones(149),reward=3.,reward_version=c.REWARD,reward_checksum='x',duration=.08,
                    discount=c.discount(.08),period_ms=80,action=[0,c.encoded_period(80)],kind='continuing',connection='a',next_connection='a',snapshot=0,behavior='policy',apply=1,actor=0)

    def test_replay_separation(self):
        rp=c.Replay('x');row=self.row();rp.append(row)
        self.assertEqual(rp.rows[0]['reward'],3.)
        for key,value in [('calibration',True),('apply',0),('reward_checksum','bad'),('next_connection','b'),('reward',float('nan'))]:
            bad=dict(row);bad[key]=value
            with self.assertRaises(ValueError):rp.append(bad)
        self.assertEqual(rp.inserted,1)

    def test_legal_suffix(self):
        row=self.row();row.update(duration=10.,discount=c.discount(10.))
        label=c.return_label([row,row],0,7.,5.)
        self.assertAlmostEqual(label['suffix_discount'],c.discount(20.))
        with self.assertRaises(ValueError):c.return_label([row,dict(row,connection='b')],0,0,5.)

    def test_credit_never_wallclock(self):
        rp=c.Replay('x');book=c.Credit()
        self.assertEqual(book.limit(rp),0)
        rp.inserted=12288;rp.by_actor[:]=512;rp.seconds=1800.
        self.assertEqual(book.limit(rp),0)
        rp.seconds+=.25
        self.assertEqual(book.limit(rp),1)
        actor=book.next(rp);self.assertFalse(actor);book.record(actor)
        with self.assertRaises(ValueError):book.next(rp)

    def test_period_mapping_and_neutral_apply(self):
        from joint_mtp import command
        for p in c.PERIODS:
            self.assertAlmostEqual(20*8**((c.encoded_period(p)+1)/2),p)
        self.assertEqual(command(3,[0,0]),b'3 100 1\0')
        exploration=c.Exploration(3)
        for time in np.arange(0,30,.02):
            a,source=exploration.choose(time,lambda t: 0.)
            self.assertTrue(np.isfinite(a).all())
            if source=='neutral_segment':self.assertEqual(a[0],0.)

    def test_ring_ownership_survives_overwrite(self):
        rp=c.Replay('x',capacity=30)
        for i in range(100):
            rp.append(dict(self.row(),actor=i%24))
        self.assertEqual(len(rp.rows),30)
        self.assertEqual(sum(map(len,rp.slots)),30)
        for actor,slots in enumerate(rp.slots):
            for pos,slot in enumerate(slots):
                self.assertEqual(rp.rows[slot]['actor'],actor)
                self.assertEqual(rp.slot_position[slot],pos)
        batch=rp.sample(np.random.RandomState(3),60.)
        self.assertEqual(batch['s0'].shape,(512,149))
        for i in range(23):rp.sample(np.random.RandomState(i),60.)
        self.assertEqual(int(rp.sampled_by_actor.max()-rp.sampled_by_actor.min()),0)


if __name__=='__main__':unittest.main()
