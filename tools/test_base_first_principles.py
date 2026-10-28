import importlib.util
import math
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]


def load(relative,name):
    spec=importlib.util.spec_from_file_location(name,str(ROOT/relative));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class BaselineIPCCompatibilityTests(unittest.TestCase):
    def test_legacy_float_mapping_is_preserved_without_policy_clamp(self):
        audit=load('frameworks/baseline_ipc_audit/verified_ipc.py','baseline_audit_ipc')
        scalar=load('frameworks/canopy_single/verified_ipc.py','unchanged_scalar_ipc')
        value=-1.0000001192092896;percent=int(100*math.pow(4,value))
        self.assertEqual(percent,24)
        self.assertEqual(audit.action_payload(657,percent,1,42),b'657 24 1 42')
        with self.assertRaises(audit.ProtocolError):audit.action_payload(657,23,1,42)
        with self.assertRaises(scalar.ProtocolError):scalar.action_payload(657,24,1,42)

    def test_archived_gated_assistance_is_identical(self):
        baseline=(ROOT/'frameworks/baseline/code/runtime/src/orca-server-control-log-safe.cc').read_text(encoding='utf-8')
        candidate=(ROOT/'frameworks/baseline_ipc_audit/native/src/orca-server-gated-verified-ipc.cc').read_text(encoding='utf-8')
        def gate(source):
            begin=source.index('                    if (!slow_start_passed)')
            return source[begin:source.index('                    // Send message to RL-Module',begin)]
        self.assertEqual(gate(baseline),gate(candidate))
        self.assertNotIn('NORMAL_CUBIC_FIRST_FEEDBACK',candidate)


if __name__=='__main__':unittest.main()
