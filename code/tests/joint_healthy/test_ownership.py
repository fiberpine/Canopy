from pathlib import Path
import unittest
from unittest import mock
import subprocess
import sys
import time
from frameworks.joint_healthy.ownership import matches_flow, cleanup_flow_report, supervise_children


class ScopedEmulator(unittest.TestCase):
    def test_exact_flow_log_or_client_and_port_are_required(self):
        folder=Path('/work/logs/a');client=Path('/work/code/client')
        self.assertTrue(matches_flow(['mm-link','--downlink-log=/work/logs/a/down.log'],folder,client,23008))
        self.assertTrue(matches_flow(['sudo','-u','ubuntu',str(client),'10.0.0.1','1','23008'],folder,client,23008))
        self.assertFalse(matches_flow([str(client),'10.0.0.1','1','230080'],folder,client,23008))
        self.assertFalse(matches_flow(['mm-link','--downlink-log=/work/logs/another/down.log'],folder,client,23008))
        self.assertFalse(matches_flow(['/other/client','10.0.0.1','1','23008'],folder,client,23008))

    def test_recycled_key_is_not_owned_and_original_object_is_still_rejected(self):
        good = dict(status='PASS', remaining=[], listeners=[], checks=[])
        with mock.patch('frameworks.baseline.baseline_admission.cleanup_report', return_value=good.copy()), \
                mock.patch('frameworks.joint_healthy.ownership.sysv_segments', return_value=[dict(key='45', shmid='200')]):
            result = cleanup_flow_report({}, 23000, [dict(key=45, id=100)], timeout_seconds=0)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['recycled_keys_not_owned'], [dict(key=45, id=200)])
        with mock.patch('frameworks.baseline.baseline_admission.cleanup_report', return_value=good.copy()), \
                mock.patch('frameworks.joint_healthy.ownership.sysv_segments', return_value=[dict(key='0', shmid='100')]):
            result = cleanup_flow_report({}, 23000, [dict(key=45, id=100)], timeout_seconds=0)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['remaining_shm_objects'], [dict(key=0, id=100)])

    def test_recycled_key_does_not_hide_a_live_listener(self):
        with mock.patch('frameworks.baseline.baseline_admission.cleanup_report', return_value=dict(status='FAIL', listeners=['23000'])), \
                mock.patch('frameworks.joint_healthy.ownership.sysv_segments', return_value=[]):
            self.assertEqual(cleanup_flow_report({}, 23000, [], timeout_seconds=0)['status'], 'FAIL')

    def test_later_failed_child_is_detected_before_slow_first_child_finishes(self):
        children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']),
                    subprocess.Popen([sys.executable, '-c', 'raise SystemExit(2)'])]
        start = time.monotonic()
        try:
            self.assertFalse(supervise_children(children, drain_seconds=.5))
            self.assertLess(time.monotonic()-start, 5.)
            self.assertTrue(all(child.poll() is not None for child in children))
        finally:
            for child in children:
                if child.poll() is None: child.kill()
                child.wait()

    def test_successful_children_remain_successful(self):
        children = [subprocess.Popen([sys.executable, '-c', 'pass']) for _ in range(2)]
        self.assertTrue(supervise_children(children))
