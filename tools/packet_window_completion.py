"""End a frozen capture after its complete packet-clock window, not missing ACKs.

This only requests clean shutdown. Valid feedback, hashes, full metrics and cleanup
are still checked by evaluate.py and the frozen worker; low throughput is retained.
"""
import json
import math
from pathlib import Path
import threading
import time


def progress(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(16384).decode(errors='replace')
        base = None
        for line in header.splitlines():
            if line.startswith('# base timestamp:'):
                base = float(line.split(':', 1)[1])
                break
        if base is None or not math.isfinite(base):
            return None
        stream.seek(0, 2)
        size = stream.tell()
        offset = max(0, size - 8192)
        stream.seek(offset)
        lines = stream.read().decode(errors='replace').splitlines()
        if offset:
            lines = lines[1:]
        timestamps = []
        for line in lines:
            fields = line.split()
            if len(fields) < 3 or fields[1] not in ('#', '+', '-', 'd'):
                continue
            try:
                stamp = float(fields[0])
            except ValueError:
                continue
            if math.isfinite(stamp):
                timestamps.append(stamp)
        return (max(timestamps) - base) / 1000 if timestamps else None


def start(path, seconds, stop_event, receipt):
    """Arm once per capture; the original 15s no-feedback watchdog stays active."""
    if not math.isfinite(float(seconds)) or seconds <= 0:
        raise ValueError('Packet window must be finite and positive')
    receipt = Path(receipt)
    if receipt.exists():
        raise ValueError('Existing completion receipt must be preserved')
    def watch():
        while not stop_event.is_set():
            try:
                elapsed = progress(path)
            except FileNotFoundError:
                elapsed = None
            if elapsed is not None and elapsed >= seconds:
                value = dict(status='COMPLETE_PACKET_WINDOW_REQUESTED_CLEAN_STOP',
                             observed_packet_seconds=elapsed, required_seconds=seconds,
                             requested_epoch=time.time(), metric_or_health_override=False)
                with receipt.open('x') as stream:
                    json.dump(value, stream, indent=2)
                stop_event.set()
                return
            stop_event.wait(.05)
    thread = threading.Thread(target=watch, name='complete-packet-window', daemon=True)
    thread.start()
    return thread
