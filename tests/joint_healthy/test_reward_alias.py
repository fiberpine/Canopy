import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.audit_joint_healthy_reward import audit


class RewardAlias(unittest.TestCase):
    def inputs(self,p,rates):
        frames=[];boundaries=[]
        for i,rate in enumerate(rates):
            raw=[25.,rate,1.,.02,0.,10.,1000.,0.,25.,8.,10.,0.,10.,1500.,20.]
            frames.append(dict(state_id=i,raw=raw));boundaries.append(dict(state_id=i,dt_us=20000,delta_bytes=int(rate*.02)))
        (p/'feedback.jsonl').write_text('\n'.join(json.dumps(x) for x in frames))
        with (p/'boundaries.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(boundaries[0]));w.writeheader();w.writerows(boundaries)

    def test_detects_reward_gain_without_a_different_policy(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.inputs(p,[500.]+[2000.,0.]*16)
            result=audit(p/'feedback.jsonl',p/'boundaries.csv')
            self.assertTrue(result['return_increase_without_changed_TCP_trajectory'])
            self.assertFalse(result['p95_p99_reconstructed'])
            self.assertEqual(len(result['results']),4)

    def test_constant_density_has_no_artificial_period_gain(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.inputs(p,[1000.]*33)
            result=audit(p/'feedback.jsonl',p/'boundaries.csv')
            self.assertFalse(result['return_increase_without_changed_TCP_trajectory'])
            self.assertLess(max(abs(x['relative_return_change_same_trajectory']) for x in result['results']),1e-12)

    def test_legacy_delivery_mean_is_not_accepted_as_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.inputs(p,[1000.]*33)
            text=(p/'boundaries.csv').read_text().replace(',20000,20',',20000,0');(p/'boundaries.csv').write_text(text)
            with self.assertRaises(ValueError):audit(p/'feedback.jsonl',p/'boundaries.csv')
