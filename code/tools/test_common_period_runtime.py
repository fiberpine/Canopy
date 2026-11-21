import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
from common_period import checksum,encoded_period,validate_execution
from common_period_runtime import Runtime,require_a0

class Stub:
    metadata={'K':60.}
    def snapshot(self):return dict(version=1,reward_checksum=checksum({'scale':6000000}))
    def train_batch(self,batch,noise,actor):self.last=(batch,noise,actor)

class RuntimeContract(unittest.TestCase):
    def setUp(self):
        self.model=Stub();self.r=Runtime(self.model,{'scale':6000000});self.r.publish()
    def put(self,**changes):
        args=dict(actor=0,connection='train',next_connection='train',state=np.zeros(149),
                  next_state=np.ones(149),action=[0,encoded_period(80)],reward=3,dt=.08,
                  period=80,version=1)
        args.update(changes);return self.r.boundary(**args)
    def test_calibration_isolation(self):
        self.put(connection='cal',next_connection='cal',calibration=True)
        self.assertEqual(self.r.replay.inserted,0);self.assertEqual(self.r.replay.seconds,0)
        with self.assertRaises(ValueError):self.put(connection='cal',next_connection='cal')
    def test_snapshot_and_connection(self):
        with self.assertRaises(ValueError):self.put(version=99)
        with self.assertRaises(ValueError):self.put(next_connection='reset')
        self.assertFalse(self.put(kind='censored',next_state=None))
        self.assertTrue(self.put(kind='truncated'))
    def test_immutable_state(self):
        s=np.zeros(149);self.put(state=s);s[:]=7
        self.assertEqual(float(self.r.replay.rows[0]['s0'].sum()),0)
    def test_no_unearned_update(self):
        self.put()
        with self.assertRaises(ValueError):self.r.update_once()
        self.assertEqual(self.r.ledger()['critic_iterations'],0)
    def test_required_gates(self):
        with self.assertRaises(ValueError):require_a0({'status':'DRAFT'},{'checks':[]})
        with self.assertRaises(ValueError):require_a0({'status':'FROZEN'},{'checks':[{'name':'model','status':'NOT_RUN'}]})
        require_a0({'status':'FROZEN'},{'checks':[{'name':'revival','status':'NOT_RUN','required':False}]})
    def test_legacy_cost_rejected_before_tcp(self):
        import json
        root=pathlib.Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'templates/mtpjoint-eval.json').read_text())
        cfg['common_reward_config']={'denominator':'global','scale_bps':6000000}
        with self.assertRaises(ValueError):validate_execution(cfg)
        cfg['decision_cost']=0;validate_execution(cfg)
    def test_B0_has_independent_registered_cap(self):
        with self.assertRaises(ValueError):self.r.evaluate_b0(1,2)
        self.r.credit.b0=1
        with self.assertRaises(ValueError):self.r.evaluate_b0(2,2)
if __name__=='__main__':unittest.main()
