"""Fail-closed authorization/ledger checks and latent exploration recovery."""
import csv
import pathlib
import sys
import tempfile
import unittest
import importlib.util
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frameworks/canopy_single'))
sys.path.insert(0,str(ROOT/'tools/scalar_learning_check'))
from critic_only_control import enabled,FrozenActor
from exploration_response import read
from analyze_l2a import gate_a_c,fixed_references


class CriticOnlyTests(unittest.TestCase):
    def test_sign_conflict_or_missing_twin_never_releases(self):
        refs=[dict(rtt=200,cls='free',J=10,beta=.15,significant_holm_05=True)]
        def row(t,gradient,twin):
            return dict(update=t,fitted_lambda=40.,dq1_by_rtt_class={'200':{'free':dict(mean=gradient)}},
                        dq2_by_rtt_class={'200':{'free':dict(mean=gradient)}},
                        twin_agree_by_rtt_class={'200':{'free':dict(mean=twin,n=32) if twin is not None else None}})
        rows=[row(t,-6.,1.) for t in range(0,10000,2000)]
        result=gate_a_c(rows,refs)
        self.assertFalse(result['a_last_five_pass']);self.assertIsNone(result['first_five_consecutive_a_c_pass'])
        rows=[row(t,1.,None) for t in range(0,10000,2000)]
        result=gate_a_c(rows,refs)
        self.assertFalse(result['c_last_five_pass']);self.assertIsNone(result['first_five_consecutive_a_c_pass'])
        rows[-1]['update']=rows[-2]['update']
        with self.assertRaises(AssertionError):gate_a_c(rows,refs)

    def test_fixed_references_use_all_sixteen_registered_comparisons(self):
        cell=dict(cumulative={'10':.2,'40':.01},cumulative_se={'10':.02,'40':.1},samples=1000)
        e1={'by_rtt':{str(r):{c:cell for c in ('free','penalized')} for r in (20,50,100,200)}}
        rows=fixed_references(e1);self.assertEqual(len(rows),16)
        self.assertTrue(all(row['holm_p']>=row['p'] for row in rows))
        e1['rows']=rows;fixed_references(e1)
        e1['rows'][0]=dict(rows[0],beta=-10.)
        with self.assertRaises(AssertionError):fixed_references(e1)

    def test_own_loss_masks_keep_common_full_batch_denominator(self):
        path=ROOT/'tools/scalar_learning_check/runtime/l2a_own_credit_replay.py'
        spec=importlib.util.spec_from_file_location('l2a_own',str(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rtt=np.repeat([20,50,100,200],128)
        np.testing.assert_array_equal(module.loss_weights(rtt,'own100').ravel(),rtt==100)
        np.testing.assert_array_equal(module.loss_weights(rtt,'own200').ravel(),rtt==200)
        self.assertEqual(module.loss_weights(rtt,'own100').shape,(512,1))
        pools=[list(range(i*100,(i+1)*100)) for i in range(24)]
        import random
        rows=module.balanced_indices(pools,0,random.Random(3))
        self.assertEqual(len(rows),512);self.assertEqual(len(np.unique(rows)),512)
        self.assertTrue(np.all(np.bincount(rows//100,minlength=24)>=21))

    def test_explicit_bounded_orca_only(self):
        self.assertFalse(enabled({}))
        self.assertTrue(enabled(dict(critic_only=True,policy_family='orca',critic_update_budget=50000)))
        for cfg in (dict(critic_only=1),dict(critic_only=True,policy_family='canopy',critic_update_budget=1),dict(critic_only=True,policy_family='orca',critic_update_budget=0)):
            with self.assertRaises(ValueError):enabled(cfg)

    def test_frozen_actor_rejects_weight_slot_or_ledger_change(self):
        class Actor:
            def train_var(self):return ['weight']
        class Opt:
            def get_slot_names(self):return ['m']
            def get_slot(self,v,n):return 'slot'
            def _get_beta_accumulators(self):return ['beta']
        class Agent:actor=Actor();actor_optimizer=Opt();global_step='step'
        class Session:
            values=dict(weight=np.array([1.]),slot=np.array([0.]),beta=np.array([.9]),step=0)
            def run(self,v):return [self.values[k].copy() for k in v] if isinstance(v,list) else self.values[v]
        for name in ('weight','slot','beta','step'):
            sess=Session();sess.values={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in Session.values.items()}
            frozen=FrozenActor(Agent(),sess);frozen.verify(0)
            sess.values[name]=sess.values[name]+1
            with self.assertRaises(AssertionError):frozen.verify(0)

    def test_logged_latent_survives_clipping_and_preaction_class(self):
        with tempfile.TemporaryDirectory() as directory:
            p=pathlib.Path(directory)/'states.csv'
            with p.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,['action','action_no_noise','reward','delay_term','exploration_offset']);writer.writeheader()
                writer.writerow(dict(action=1,action_no_noise=.9,reward=.5,delay_term=.5,exploration_offset=.4))
                writer.writerow(dict(action=.7,action_no_noise=.9,reward=.7,delay_term=1,exploration_offset=-.1))
            o,r,classes=read(p)
            np.testing.assert_array_equal(o,[.4,-.1]);np.testing.assert_array_equal(classes,[-1,1])


if __name__=='__main__':unittest.main()
