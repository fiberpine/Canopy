import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from frameworks.joint_healthy import identity
from frameworks.joint_healthy.configuration import configuration


class NativeProvenance(unittest.TestCase):
    def test_failed_g1_g2_cannot_be_relabelled_as_g3_admission(self):
        cfg=configuration('g3');impl={'fixture':True}
        receipt=dict(status='PASS_PREREGISTERED_G0_G3_FULL_GRID',implementation=impl,registration='registered.json',evidence='actual-results.json',
            rate_mode='acked_legacy_peak',utility_normalization='legacy_delivery_peak',
            validation=dict(conditions=24,repeats=3,candidate_runs=72,paired_base_runs=72,health_pass=True,g0_mechanism_pass=True))
        with mock.patch.object(identity,'implementation_identity',return_value=impl):
            identity.check_admission(receipt,cfg)
            with self.assertRaises(ValueError):identity.check_admission(dict(receipt,status='SCREEN_PERFORMANCE_FAIL'),cfg)
            with self.assertRaises(ValueError):identity.check_admission(dict(receipt,validation=dict(receipt['validation'],health_pass=False)),cfg)
            with self.assertRaises(ValueError):identity.check_admission(receipt,dict(cfg,decision_cost=.02))
            forty=dict(cfg,calibration_main_ms=40.)
            with self.assertRaises(ValueError):identity.check_admission(receipt,forty)
            identity.check_admission(dict(receipt,validated_initial_period_ms=40.),forty)
            with self.assertRaises(ValueError):identity.check_admission(dict(receipt,validated_initial_period_ms=40.),dict(forty,time_init_ms=80.))
            coverage=configuration('g4')
            with self.assertRaises(ValueError):identity.check_admission(receipt,coverage)
            identity.check_admission(dict(receipt,status='PASS_PREREGISTERED_G0_G4_FULL_GRID',validated_time_initializer='feedback_coverage_v1'),coverage)

    def test_binary_and_all_build_sources_must_still_match(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); binary=root/'frameworks/joint_healthy/native/orca-server-joint-healthy'
            names=['frameworks/joint_healthy/native/'+n for n in ('orca-server-joint-healthy.cc','joint_protocol.h')]
            names+=['frameworks/baseline/code/runtime/src/'+n for n in ('flow.cc','flow.h','define_v2.h','common.h','util.h')]
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'source')
            binary.write_bytes(b'fixture')
            receipt=dict(sha256=hashlib.sha256(b'fixture').hexdigest(),startup='natural-cubic',
                         source_sha256={n:hashlib.sha256(b'source').hexdigest() for n in names})
            Path(str(binary)+'.build.json').write_text(json.dumps(receipt))
            with mock.patch.object(identity,'ROOT',root):
                self.assertEqual(identity.validate_native(),receipt)
                (root/names[0]).write_bytes(b'changed')
                with self.assertRaises(ValueError):identity.validate_native()
