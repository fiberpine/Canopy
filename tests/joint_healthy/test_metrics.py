import json
from pathlib import Path
import tempfile
import unittest
from frameworks.joint_healthy.metrics import cpu_summary, queue_tails


class Measurements(unittest.TestCase):
    def test_cpu_does_not_double_count_children_or_pid_reuse(self):
        def row(t):
            return dict(elapsed_s=t, machine_cpu=dict(user=t,system=t,idle=6*t),processes=[
                dict(pid=1,created=1,name='python',cpu=dict(user=t,system=0,children_user=100*t)),
                dict(pid=2,created=1,name='native',cpu=dict(user=0,system=t)),
                dict(pid=3,created=t,name='reused',cpu=dict(user=10*t,system=0))])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'resources.json').write_text(json.dumps([row(1),row(3)]))
            r=cpu_summary(p,dict(python_pid=1,native_pid=2,logical_cpus=8))
            self.assertEqual(r['python_cpu_seconds'],2);self.assertEqual(r['observed_run_tree_cpu_seconds'],4)
            self.assertEqual(r['observed_run_tree_all_core_fraction'],.25)
            self.assertEqual(r['machine_busy_fraction'],.25)

    def test_packet_p99_obeys_packet_window(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'down.log';p.write_text('# base timestamp: 1000\n1000 - 1500 1\n11000 - 1500 10\n20999 - 1500 20\n21000 - 1500 999\n')
            r=queue_tails(p,20)
            self.assertEqual(r['full']['departure_samples'],3)
            self.assertAlmostEqual(r['trim10']['queue_p99_ms'],19.9)
