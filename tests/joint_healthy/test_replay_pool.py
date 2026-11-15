import random
import unittest
import numpy as np
from frameworks.joint_healthy.replay import IndexPool, Replay


class DenseReplayPool(unittest.TestCase):
    def test_swapped_removal_preserves_membership_after_repeated_eviction(self):
        pool = IndexPool(); reference = set()
        for step in range(2000):
            index = step % 71; pool.add(index); pool.add(index); reference.add(index)
            old = (step-23) % 71; pool.discard(old); pool.discard(old); reference.discard(old)
            self.assertEqual(set(pool), reference)
            self.assertEqual(len(pool), len(reference))
            self.assertEqual({value:i for i,value in enumerate(pool.values)}, pool.positions)
        chosen = pool.sample(random.Random(3), len(pool))
        self.assertEqual(len(set(chosen)), len(reference)); self.assertEqual(set(chosen), reference)

    def test_same_mutations_and_rng_reproduce_same_uniform_without_replacement_draws(self):
        a, b = IndexPool(), IndexPool()
        for pool in (a,b):
            for i in range(100): pool.add(i)
            for i in range(0,100,3): pool.discard(i)
        x, y = random.Random(7), random.Random(7)
        for _ in range(100):
            chosen = a.sample(x,21)
            self.assertEqual(chosen,b.sample(y,21)); self.assertEqual(len(set(chosen)),21)
            self.assertTrue(set(chosen).issubset(set(a)))

    def test_starved_actor_cannot_partially_consume_rng_or_sampling_ledger(self):
        replay = Replay(dict(memsize=100,num_actors=2,batch_size=8,seed=3,warmup=4,
                             minimum_enqueued_per_actor=2,updates_per_base20ms=.1))
        for i in range(8):replay.pools[0].add(i)
        before = replay.rng.getstate()
        with self.assertRaisesRegex(ValueError,'actor=1 retained=0 required=4'):replay.sample()
        self.assertEqual(before,replay.rng.getstate())
        self.assertEqual(replay.sample_calls,0); self.assertTrue(np.all(replay.sampled==0))
