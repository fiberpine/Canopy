import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from frameworks.joint_healthy.audit import audit_flow


class ExecutionAudit(unittest.TestCase):
    def fixture(self, folder):
        (folder/'result.json').write_text(json.dumps(dict(status='COMPLETED')))
        (folder/'config.json').write_text(json.dumps(dict(cfg=dict(clock_mode='scheduled'))))
        decisions=[dict(state_id=i,action=[0.,-1.]) for i in range(5)]
        feedback=[dict(state_id=i,raw=[0.,1.,1.]+[0.]*12,fresh_rtt=True,actual_dt_s=.02) for i in range(6)]
        (folder/'decisions.jsonl').write_text('\n'.join(json.dumps(x) for x in decisions))
        (folder/'feedback.jsonl').write_text('\n'.join(json.dumps(x) for x in feedback))
        rows=[dict(time_us=i,state_id=i,action_id=i,percent=100,cwnd_before=10,cwnd_requested=10,return_code=0,socket=4) for i in range(5)]
        timing=[dict(state_id=i,command_mono_us=i,planned_wait_us=20000,apply=1) for i in range(5)]
        boundaries=[dict(state_id=i,dt_us=20000,kernel_queries=i+10) for i in range(6)]
        for name,data in [('control.csv',rows),('timing.csv',timing),('boundaries.csv',boundaries)]:
            with (folder/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        (folder/'server.log').write_text('ORCA_STARTUP_HANDOFF mode=natural-cubic scheme=cubic assist_writes=0 ssthresh=9 cwnd=10\n')

    def test_valid_execution_is_not_a_performance_claim(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.fixture(p);result=audit_flow(p)
            self.assertTrue(result['no_hidden_periodic_rl_observations']);self.assertFalse(result['performance_health_claim'])

    def test_hidden_socket_reads_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.fixture(p)
            text=(p/'boundaries.csv').read_text().replace('3,20000,13','3,20000,14');(p/'boundaries.csv').write_text(text)
            with self.assertRaises(ValueError):audit_flow(p)

    def test_wrong_integer_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.fixture(p)
            text=(p/'control.csv').read_text().replace('100,10,10','100,10,11');(p/'control.csv').write_text(text)
            with self.assertRaises(ValueError):audit_flow(p)

    def test_fabricated_freshness_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.fixture(p)
            text=(p/'feedback.jsonl').read_text().replace('"fresh_rtt": true','"fresh_rtt": false');(p/'feedback.jsonl').write_text(text)
            with self.assertRaises(ValueError):audit_flow(p)
