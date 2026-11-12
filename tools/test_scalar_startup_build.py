"""Protect preserved runtimes and make the treatment's build flags explicit."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('startup_build', ROOT / 'tools/build_scalar_startup_candidate.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
FRAMEWORK = ROOT / 'frameworks/baseline'


class StartupBuildProtection(unittest.TestCase):
    def test_existing_binary_or_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'native'
            for path in (output, Path(str(output) + '.startup-build.json')):
                path.write_bytes(b'preserved')
                with self.assertRaisesRegex(ValueError, 'must be preserved'):
                    build.prepare(FRAMEWORK, output, 'first-feedback')
                self.assertEqual(path.read_bytes(), b'preserved')
                path.unlink()

    def test_only_startup_define_differs_between_matched_audited_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'native'
            legacy = build.prepare(FRAMEWORK, output, 'legacy', True)[0]
            normal = build.prepare(FRAMEWORK, output, 'first-feedback', True)[0]
            self.assertEqual(legacy['source_sha256'], normal['source_sha256'])
            self.assertEqual(legacy['command'], [v for v in normal['command'] if v != '-DORCA_FIRST_FEEDBACK_HANDOFF'])
            self.assertIn('-DORCA_STARTUP_AUDIT', legacy['command'])
            self.assertNotIn('-O2', normal['command'])

    def test_dry_run_needs_no_compiler_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'new-directory/native'
            with contextlib.redirect_stdout(io.StringIO()) as result:
                build.main(['--framework', str(FRAMEWORK), '--output', str(output),
                            '--startup', 'first-feedback', '--compiler', 'not-an-installed-compiler', '--dry-run'])
            self.assertEqual(json.loads(result.getvalue())['status'], 'PLAN_ONLY_NOT_COMPILED')
            self.assertFalse(output.parent.exists())


if __name__ == '__main__':
    unittest.main()
