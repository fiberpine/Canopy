"""Frozen capture with raw feedback and predeclared packet-clock completion.

Deploy as capture_frozen.py inside a separate evaluation framework directory.
"""
import json
from pathlib import Path
import sys
import evaluate
import packet_window_completion

ROOT = Path(__file__).resolve().parent
cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
assert cfg.get('no_training') is True and cfg.get('backend') == 'moving_stats'
sys.path.insert(0, str(ROOT / 'code/canopy/rl-module'))
import agent_v2
import moving_stats_backend
moving_stats_backend.install(agent_v2)

raw_stream = (ROOT / 'logs' / (cfg['run_id'] + '-raw15.jsonl')).open('x', buffering=262144)
Reader = evaluate.StateReader
readers = []


class RecordedReader(Reader):
    def __init__(self, shm, server, stop):
        super().__init__(shm, server, stop)
        readers.append(self)
        folder = ROOT / 'logs' / cfg['run_id']
        self.window_thread = packet_window_completion.start(
            folder / 'down.log', cfg['seconds'], stop, folder / 'packet-window-completion.json')

    def read(self):
        result = super().read()
        import time
        raw_stream.write(json.dumps(dict(time_us=time.time()*1e6, state_id=self.last_id,
                                         raw=self.last_raw.tolist())) + '\n')
        return result


evaluate.StateReader = RecordedReader
try:
    evaluate.main()
finally:
    for reader in readers:
        reader.stop.set()
        reader.window_thread.join(timeout=1)
    raw_stream.close()
    errors = [r.window_thread.completion_state['error'] for r in readers if r.window_thread.completion_state['error']]
    if errors:
        raise RuntimeError('Packet-clock completion observer failed: ' + repr(errors))
