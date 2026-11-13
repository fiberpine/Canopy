from pathlib import Path
import unittest
from frameworks.joint_healthy.ownership import matches_flow


class ScopedEmulator(unittest.TestCase):
    def test_exact_flow_log_or_client_and_port_are_required(self):
        folder=Path('/work/logs/a');client=Path('/work/code/client')
        self.assertTrue(matches_flow(['mm-link','--downlink-log=/work/logs/a/down.log'],folder,client,23008))
        self.assertTrue(matches_flow(['sudo','-u','ubuntu',str(client),'10.0.0.1','1','23008'],folder,client,23008))
        self.assertFalse(matches_flow([str(client),'10.0.0.1','1','230080'],folder,client,23008))
        self.assertFalse(matches_flow(['mm-link','--downlink-log=/work/logs/another/down.log'],folder,client,23008))
        self.assertFalse(matches_flow(['/other/client','10.0.0.1','1','23008'],folder,client,23008))
