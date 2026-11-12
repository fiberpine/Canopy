import importlib.util
import math
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]


def load(relative,name):
    spec=importlib.util.spec_from_file_location(name,str(ROOT/relative));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def default_startup_source(source):
    # Inspect the default branch, rather than treating inactive audit code as
    # an execution change. This is a static check, not a C++ compilation test.
    lines=[];parents=[];active=True
    for line in source.splitlines(keepends=True):
        directive=line.strip()
        if directive.startswith('#ifdef ORCA_'):
            if directive not in ('#ifdef ORCA_STARTUP_AUDIT','#ifdef ORCA_FIRST_FEEDBACK_HANDOFF'):
                raise AssertionError('Unreviewed startup conditional: '+directive)
            parents.append(active);active=False
        elif directive=='#else' and parents:
            active=parents[-1] and not active
        elif directive=='#endif' and parents:
            active=parents.pop()
        elif directive.startswith('#line '):
            if not directive.split()[1].isdigit():raise AssertionError('Unexpected line directive')
        elif active:
            lines.append(line)
    if parents:raise AssertionError('Unclosed startup conditional')
    return ''.join(lines)


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
        self.assertEqual(gate(default_startup_source(baseline)),gate(candidate))
        self.assertNotIn('NORMAL_CUBIC_FIRST_FEEDBACK',candidate)


if __name__=='__main__':unittest.main()
