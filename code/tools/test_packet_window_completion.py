"""A short/partial packet log must never satisfy the complete-window stop rule."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('packet_completion', Path(__file__).with_name('packet_window_completion.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PacketCompletion(unittest.TestCase):
    def test_base_offset_and_incomplete_trailing_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'down.log'
            p.write_text('# base timestamp: 32\n60031 # 1504\n60032')
            self.assertEqual(module.progress(p), 59.999)
            p.write_text('# base timestamp: 32\n60032 # 1504\n')
            self.assertEqual(module.progress(p), 60)

    def test_missing_base_is_not_a_complete_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'down.log'
            p.write_text('999999 # 1504\n')
            self.assertIsNone(module.progress(p))

    def test_low_throughput_can_end_capture_without_overriding_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'down.log'; receipt = Path(temporary) / 'completion.json'
            p.write_text('# base timestamp: 32\n32 - 1504 0\n60032 # 1504\n')
            stop = threading.Event()
            worker = module.start(p, 60, stop, receipt); worker.join(timeout=2)
            self.assertFalse(worker.is_alive()); self.assertTrue(stop.is_set())
            self.assertFalse(json.loads(receipt.read_text())['metric_or_health_override'])

    def test_window_end_cannot_stop_at_59_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'down.log'; receipt = Path(temporary) / 'completion.json'
            p.write_text('# base timestamp: 32\n59032 # 1504\n')
            stop = threading.Event(); worker = module.start(p, 60, stop, receipt)
            self.assertFalse(stop.wait(.1)); self.assertFalse(receipt.exists())
            stop.set(); worker.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
