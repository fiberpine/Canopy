#!/usr/bin/env python3
"""Zero-TCP per-run diagnostics: does the window bind, and where does log-window growth come from?

Reads one actor/evaluation run folder written with verified_ipc_v2.  It only reads logs.
  ipc-python.csv  every accepted native STATE frame; payload = state id + FIELDS below
  control.csv     every native command (percent, cwnd_before, cwnd_requested)
  states.csv      Python rows: executed action, deterministic action (action_no_noise), percent
Python 3.6+, stdlib only.

Window binding (kernel 4.13 rules, approximated from the per-decision snapshots)
  binding     packets_out >= 0.9*cwnd.  Congestion avoidance grows cwnd only when the
              flow is cwnd-limited (in_flight >= cwnd at send time).
  unanchored  cwnd > 2*max(max_packets_out, packets_out).  Slow-start growth is off
              (tcp_is_cwnd_limited needs cwnd < 2*max_packets_out) and cwnd is not what
              limits sending (receive window, send buffer or application).  No loss can
              be inferred solely from this ratio. The label is a descriptive proxy,
              not proof of socket cause or absence of kernel cwnd validation.
  in-flight   packets_out*mss.  The same ceiling at different capacities indicates an
              end-host buffer cap rather than a policy choice.
  binding_rtt packets_out >= 0.9*min(cwnd over the trailing max(2, ceil(2*srtt/dt)) decisions).
              Per-step multiplicative jitter (zero_sum exploration) moves cwnd every 20 ms
              while in-flight follows only within about an RTT, so the snapshot proxy above
              settles near 0.45 at RTT 200 ms even for an uncapped flow.  This variant asks
              whether the window bounded sending over the last RTTs; it stays near 0 when
              cwnd is persistently far above in-flight (end-host cap).
  ss_mode     cwnd < ssthresh: the kernel adds one packet per ACK between RL writes
              (about x2 per RTT), so holding the window needs a negative action.

Command decomposition (ln units, sums over the run)
  command_log_growth          sum ln(cwnd_requested/cwnd_before) over bound native commands
                              (includes the MIN_CWND floor and INT_MAX cap)
  ideal_command_log_impulse   sum a_exec*ln4 over states.csv rows
  deterministic_log_impulse   sum pi*ln4
  exploration_log_impulse     ideal - deterministic (includes clipping)
  percent_truncation_log      sum ln(int(100*4**a)/100) - a_exec*ln4 (always <= 0)
  integer_rounding_residual   command - ideal (percent truncation + integer cwnd floor + clamps)
  between_command_native_log_growth  ln(cwnd_last/cwnd_first) - command (kernel growth/loss response)
  max_abs_cumulative_exploration     max_t |sum_{s<=t}(a_exec - pi)| in action units:
                              a random walk under iid noise, bounded under zero_sum_offset.
"""
import argparse
import csv
import json
import math
import os
import pathlib
import subprocess
import sys

FIELDS = ('delay', 'thr', 'cnt', 'time_delta', 'target', 'cwnd', 'pacing', 'lost_rate', 'srtt_ms',
          'ssthresh', 'packets_out', 'retrans_out', 'max_packets_out', 'mss', 'min_rtt')
STATE_KIND = '1'                 # verified_ipc.STATE
INFINITE_SSTHRESH = 2147483647   # TCP_INFINITE_SSTHRESH: no loss and no HyStart exit yet
LN4 = math.log(4.0)
SYSCTLS = ('net/ipv4/tcp_rmem', 'net/ipv4/tcp_wmem', 'net/core/rmem_max', 'net/core/wmem_max',
           'net/ipv4/tcp_adv_win_scale', 'net/ipv4/tcp_moderate_rcvbuf', 'net/ipv4/tcp_slow_start_after_idle',
           'net/ipv4/tcp_congestion_control', 'net/ipv4/tcp_no_metrics_save', 'net/ipv4/tcp_notsent_lowat')


def read_states(folder):
    out = []
    with open(str(pathlib.Path(folder) / 'ipc-python.csv'), newline='') as f:
        for r in csv.DictReader(f):
            if r['direction'] != 'received' or r['kind'] != STATE_KIND:
                continue
            tok = r['payload'].split()
            if len(tok) != 1 + len(FIELDS):
                raise ValueError('unexpected state payload: %r' % r['payload'][:80])
            rec = dict(zip(FIELDS, (float(x) for x in tok[1:])))
            rec['state_id'] = int(tok[0])
            out.append(rec)
    return out


def _quantile(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def _summary(rows, binding_ratio):
    total = sum(dt for _, dt, _ in rows)
    if not rows or total <= 0:
        return dict(decisions=len(rows), seconds=total)

    def frac(pred):
        return sum(dt for _, dt, s in rows if pred(s)) / total

    inflight = [s['packets_out'] * s['mss'] for _, _, s in rows]
    ratio = [s['cwnd'] / max(s['packets_out'], 1.0) for _, _, s in rows]
    to_ssthresh = [s['cwnd'] / max(s['ssthresh'], 1.0) for _, _, s in rows if s['ssthresh'] < INFINITE_SSTHRESH]
    return dict(
        decisions=len(rows), seconds=total,
        binding_fraction=frac(lambda s: s['packets_out'] >= binding_ratio * s['cwnd']),
        binding_rtt_fraction=frac(lambda s: s['packets_out'] >= binding_ratio * s['_cwnd_rtt_min']),
        unanchored_fraction=frac(lambda s: s['cwnd'] > 2.0 * max(s['max_packets_out'], s['packets_out'], 1.0)),
        slow_start_growth_eligible_fraction=frac(lambda s: s['cwnd'] < s['ssthresh'] and s['cwnd'] < 2.0 * s['max_packets_out']),
        ss_mode_fraction=frac(lambda s: s['cwnd'] < s['ssthresh']),
        infinite_ssthresh_fraction=frac(lambda s: s['ssthresh'] >= INFINITE_SSTHRESH),
        cwnd_over_ssthresh_p50=_quantile(to_ssthresh, .5),
        inflight_bytes_p50=_quantile(inflight, .5), inflight_bytes_p95=_quantile(inflight, .95),
        inflight_bytes_max=max(inflight), cwnd_max=max(s['cwnd'] for _, _, s in rows),
        cwnd_over_packets_out_p50=_quantile(ratio, .5), cwnd_over_packets_out_p95=_quantile(ratio, .95))


def _annotate_rtt_min(states, rtt_multiple=2.0):
    """Attach the trailing min cwnd over about rtt_multiple sRTTs (at least 2 decisions) to each state."""
    from collections import deque
    window, elapsed = deque(), 0.0
    for s in states:
        dt = max(s['time_delta'], 1e-6)
        elapsed += dt
        window.append((elapsed, s['cwnd']))
        horizon = max(rtt_multiple * s['srtt_ms'] / 1000.0, 0.0)
        while len(window) > 2 and elapsed - window[1][0] >= horizon:
            window.popleft()
        s['_cwnd_rtt_min'] = min(c for _, c in window)


def binding_metrics(states, trim_s=10.0, binding_ratio=0.9):
    _annotate_rtt_min(states)
    t, rows = 0.0, []
    for s in states:
        dt = max(s['time_delta'], 0.0)
        t += dt
        rows.append((t, dt, s))
    return dict(full=_summary(rows, binding_ratio), trim10=_summary([r for r in rows if r[0] > trim_s], binding_ratio),
                binding_ratio=binding_ratio, trim_s=trim_s, rtt_window='trailing max(2 decisions, 2*srtt)')


def command_decomposition(folder, states=None):
    folder = pathlib.Path(folder)
    with open(str(folder / 'control.csv'), newline='') as f:
        ctl = list(csv.DictReader(f))
    bound = [r for r in ctl if r['state_id'] == r['action_id']]
    with open(str(folder / 'states.csv'), newline='') as f:
        st = list(csv.DictReader(f))
    # states row B logs action A. A next action is sent before row B is logged;
    # exclude a possible final pending command from both impulse decompositions.
    if len(bound) not in (len(st),len(st)+1):
        raise ValueError('Native commands cannot align with completed transitions')
    completed=bound[:len(st)]
    for native,python in zip(completed,st):
        if int(native['percent'])!=int(python['percent']):
            raise ValueError('Previous-action percent alignment failed')
    command = percent_log_ctl = 0.0
    for r in completed:
        before = max(int(float(r['cwnd_before'])), 1)
        requested = max(int(float(r['cwnd_requested'])), 1)
        command += math.log(requested / before)
        percent_log_ctl += math.log(int(float(r['percent'])) / 100.0)
    ideal = det = trunc = cum = peak = 0.0
    for r in st:
        a, pi = float(r['action']), float(r['action_no_noise'])
        ideal += a * LN4
        det += pi * LN4
        trunc += math.log(int(float(r['percent'])) / 100.0) - a * LN4
        cum += a - pi
        peak = max(peak, abs(cum))
    n = max(len(st), 1)
    out = dict(bound_commands=len(bound), completed_commands=len(completed),pending_commands=len(bound)-len(completed),
               unbound_control_rows=len(ctl) - len(bound), states_rows=len(st),
               command_log_growth=command, ideal_command_log_impulse=ideal, deterministic_log_impulse=det,
               exploration_log_impulse=ideal - det, percent_truncation_log=trunc,
               cwnd_floor_and_clamp_log=command - percent_log_ctl, integer_rounding_residual=command - ideal,
               per_step=dict(deterministic=det / n, exploration=(ideal - det) / n, percent_truncation=trunc / n,
                             integer_rounding=(command - ideal) / n),
               max_abs_cumulative_exploration=peak, final_cumulative_exploration=cum)
    if states:
        total = math.log(max(states[-1]['cwnd'], 1.0) / max(states[0]['cwnd'], 1.0))
        out.update(total_log_growth=total, between_command_native_log_growth=total - command)
    return out


def analyze(folder, binding_ratio=0.9):
    folder = pathlib.Path(folder)
    out = dict(run=str(folder))
    states = None
    if (folder / 'ipc-python.csv').exists():
        states = read_states(folder)
        out['binding'] = binding_metrics(states, binding_ratio=binding_ratio)
    else:
        out['binding'] = dict(status='NO_VERIFIED_STATE_LOG')
    out['decomposition'] = command_decomposition(folder, states)
    return out


def host_receipt():
    """Read-only snapshot of the TCP buffer settings that decide the end-host in-flight cap."""
    values = {}
    for key in SYSCTLS:
        try:
            values[key.replace('/', '.')] = pathlib.Path('/proc/sys', key).read_text().split()
        except OSError as e:
            values[key.replace('/', '.')] = 'UNREADABLE: ' + repr(e)
    return dict(kernel=os.uname().release, sysctl=values)


def ss_snapshot(port):
    """Server-side socket memory/window snapshot (sport filter), e.g. at 15/30/45 s of a run."""
    cmd = ['ss', '-tmin', 'state', 'established', '( sport = :%d )' % int(port)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=10)
    return dict(cmd=cmd, returncode=p.returncode, output=p.stdout)


def _write_run(d, cwnd, packets_out, max_po, ssthresh, actions, no_noise, mss=1448.0):
    d.mkdir(parents=True, exist_ok=True)
    with open(str(d / 'ipc-python.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'kind', 'sequence', 'crc', 'payload'])
        for i, (c, po, mp, ss) in enumerate(zip(cwnd, packets_out, max_po, ssthresh)):
            vals = dict(delay=0, thr=0, cnt=1, time_delta=.02, target=0, cwnd=c, pacing=0, lost_rate=0, srtt_ms=200,
                        ssthresh=ss, packets_out=po, retrans_out=0, max_packets_out=mp, mss=mss, min_rtt=200)
            w.writerow(['received', 1, i + 1, 0, ' '.join(['%d' % (i % 1000)] + ['%.7f' % vals[k] for k in FIELDS])])
            w.writerow(['sent', 2, i + 1, 0, 'ignored'])
    with open(str(d / 'states.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['action', 'action_no_noise', 'percent'])
        for a, pi in zip(actions, no_noise):
            w.writerow([a, pi, int(100 * 4 ** a)])
    with open(str(d / 'control.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_us', 'state_id', 'action_id', 'percent', 'cwnd_before', 'cwnd_requested', 'return_code', 'socket'])
        for i, (c, a) in enumerate(zip(cwnd, actions)):
            pct = int(100 * 4 ** a)
            w.writerow([i, i % 1000, i % 1000, pct, int(c), max(4, pct * int(c) // 100), 0, 3])
        w.writerow([len(cwnd), 7, 8, 100, 10, 10, 0, 3])   # an unbound timeout record is excluded


def self_test(tmp):
    import random
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'frameworks/canopy_single'))
    import numpy as np
    import scalar_exploration as se
    root = pathlib.Path(tmp)
    rng = random.Random(1)
    n = 3000
    # Receive-window-capped flow: 2074 packets in flight, the window wanders far above it.
    cwnd = [4000.0 * math.exp(0.002 * i) for i in range(n)]
    pi = [-0.003] * n
    iid = [max(-1.0, min(1.0, p + rng.gauss(0, .2))) for p in pi]
    _write_run(root / 'capped', cwnd, [2074.0] * n, [2074.0] * n, [INFINITE_SSTHRESH] * n, iid, pi)
    r = analyze(root / 'capped')
    b, d = r['binding']['trim10'], r['decomposition']
    assert b['unanchored_fraction'] == 1.0 and b['binding_fraction'] == 0.0 and b['infinite_ssthresh_fraction'] == 1.0, b
    assert b['inflight_bytes_max'] == 2074 * 1448.0 and b['inflight_bytes_p50'] == 2074 * 1448.0, b
    assert d['bound_commands'] == n and d['unbound_control_rows'] == 1, d
    assert abs(d['exploration_log_impulse'] - (d['ideal_command_log_impulse'] - d['deterministic_log_impulse'])) < 1e-9
    assert d['percent_truncation_log'] < 0 and d['integer_rounding_residual'] <= d['percent_truncation_log'] + 1e-9, d
    assert -0.0065 < d['per_step']['percent_truncation'] < -0.0035, d['per_step']   # dithered quantizer, about -0.5%
    # Window-limited flow driven by the zero-sum process: cumulative exploration stays bounded.
    z = se.ZeroSumOffset(sigma=.2, rng=np.random.RandomState(3))
    zs = [z(p) for p in pi]
    small = [150.0 + 10 * math.sin(i / 50.0) for i in range(n)]
    _write_run(root / 'binding', small, small, small, [120.0] * n, zs, pi)
    r2 = analyze(root / 'binding')
    b2, d2 = r2['binding']['trim10'], r2['decomposition']
    assert b2['binding_fraction'] == 1.0 and b2['unanchored_fraction'] == 0.0, b2
    assert d2['max_abs_cumulative_exploration'] < 0.75, d2      # about 5 sigma_o for sigma=.2, rho=0
    assert d['max_abs_cumulative_exploration'] > 2.0, d          # the iid walk on the same length
    assert b['binding_rtt_fraction'] == 0.0 and b['ss_mode_fraction'] == 1.0, b
    assert b2['binding_rtt_fraction'] == 1.0 and b2['ss_mode_fraction'] == 0.0, b2
    # Uncapped long-RTT flow with per-step window jitter: in-flight follows the recent median with lag.
    jr = np.random.RandomState(5)
    jitter = [1500.0 * 4 ** (0.2 / math.sqrt(2) * jr.randn()) for _ in range(n)]
    lagged = [0.87 * sorted(jitter[max(0, i - 10):i] or [jitter[0]])[len(jitter[max(0, i - 10):i] or [0]) // 2] for i in range(n)]
    _write_run(root / 'jitter', jitter, lagged, lagged, [900.0] * n, zs, pi)
    b3 = analyze(root / 'jitter')['binding']['trim10']
    assert b3['binding_fraction'] < 0.6 and b3['binding_rtt_fraction'] > 0.9 and b3['unanchored_fraction'] < 0.01, b3
    return dict(capped=dict(trim10=b, per_step=d['per_step'], max_abs_cumulative_exploration=d['max_abs_cumulative_exploration']),
                binding=dict(trim10=b2, max_abs_cumulative_exploration=d2['max_abs_cumulative_exploration']),
                jitter=dict(trim10=b3))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*')
    p.add_argument('--binding-ratio', type=float, default=0.9)
    p.add_argument('--self-test', metavar='TMPDIR')
    p.add_argument('--host-receipt', action='store_true')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(dict(self_test='PASSED', result=self_test(a.self_test)), indent=1))
        return 0
    if a.host_receipt:
        print(json.dumps(host_receipt(), indent=1))
    for run in a.runs:
        try:
            res = analyze(run, a.binding_ratio)
        except Exception as e:
            res = dict(run=run, status='FAILED', reason=repr(e))
        print(json.dumps(res))
    return 0


if __name__ == '__main__':
    sys.exit(main())
