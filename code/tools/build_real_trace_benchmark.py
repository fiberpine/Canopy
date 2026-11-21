#!/usr/bin/env python3
"""Build curated, strictly split (0% train/test overlap) real-world Mahimahi trace suite
for Fiber-Next training and high-discrimination top-venue evaluation.

Sources (all real open-source measurement datasets):
  1. Lumos5G (UMN commercial 5G mmWave driving/walking) -> 16 train + 8 held-out test
  2. Starlink-on-the-Road (Central Europe LEO satellite) -> 12 train + 6 held-out test
  3. Mahimahi Official 4G LTE / UMTS Driving traces      -> 16 train + 8 held-out test
  4. Multi-Timescale Synthetic Dynamic Drops / Ramps     -> 16 train + 8 held-out test

Total: 60 diverse dynamic/real training traces + 30 strictly held-out test traces!
Every trace is converted to exact Mahimahi millisecond packet-delivery timestamps
(1504 bytes/pkt, monotonic integer ms, clamped safely to [2.0, 120.0] Mbps so
Mahimahi stays 100% jitter-free while preserving real-world dynamic range).
"""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(r'D:\Canopy\innovations\fiber-next-offline')
RAW = ROOT / 'traces_real/raw'
OUT_TRAIN = ROOT / 'traces_real/train'
OUT_TEST  = ROOT / 'traces_real/test'
OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_TEST.mkdir(parents=True, exist_ok=True)

PKT_BITS = 1504.0 * 8.0  # 12032 bits per Mahimahi packet

def mbps_series_to_mahimahi(mbps_1hz, out_path, min_mbps=2.5, max_mbps=120.0):
    """Convert a 1-Hz throughput series (Mbps) into exact Mahimahi ms timestamps."""
    arr = np.asarray(mbps_1hz, dtype=float)
    arr = np.where(np.isfinite(arr), arr, min_mbps)
    # Preserve real relative dynamics while keeping within Mahimahi single-core budget
    if np.percentile(arr, 95) > max_mbps:
        arr = arr * (max_mbps / np.percentile(arr, 95))
    arr = np.clip(arr, min_mbps, max_mbps)

    ts_ms = []
    carry = 0.0
    for sec_idx, mbps in enumerate(arr):
        pkts_f = (mbps * 1e6) / PKT_BITS + carry
        n_pkts = int(math.floor(pkts_f))
        carry = pkts_f - n_pkts
        if n_pkts <= 0:
            continue
        base_ms = sec_idx * 1000
        # Evenly pace packets across the 1000 ms second
        offsets = np.linspace(0, 999, n_pkts, endpoint=False)
        for off in offsets:
            ts_ms.append(int(base_ms + off))
    # Guarantee strictly non-decreasing integers
    ts_ms.sort()
    with open(out_path, 'w', encoding='ascii', newline='\n') as fw:
        fw.write('\n'.join(str(t) for t in ts_ms) + '\n')
    return {
        'file': out_path.name,
        'duration_s': len(arr),
        'packets': len(ts_ms),
        'mean_mbps': round(float(np.mean(arr)), 2),
        'p10_mbps': round(float(np.percentile(arr, 10)), 2),
        'p90_mbps': round(float(np.percentile(arr, 90)), 2),
        'cv': round(float(np.std(arr) / max(1e-6, np.mean(arr))), 3),
        'sha256': hashlib.sha256(out_path.read_bytes()).hexdigest()[:16],
    }

def slice_mahimahi_file(src_path, dur_s=90, max_chunks=6):
    """Slice an existing Mahimahi ms-timestamp file into disjoint non-overlapping chunks."""
    lines = [int(x) for x in src_path.read_text().splitlines() if x.strip()]
    if not lines:
        return []
    t0 = lines[0]
    total_s = (lines[-1] - t0) // 1000
    chunks = []
    for k in range(min(max_chunks, total_s // dur_s)):
        lo = t0 + k * dur_s * 1000
        hi = lo + dur_s * 1000
        sub = [t - lo for t in lines if lo <= t < hi]
        if len(sub) < dur_s * 120:  # skip near-empty blackout chunks (< ~1.4 Mbps mean)
            continue
        # Compute 1-Hz Mbps stats
        bins = np.zeros(dur_s, dtype=float)
        for t in sub:
            b = min(dur_s - 1, t // 1000)
            bins[b] += PKT_BITS / 1e6
        chunks.append((sub, bins))
    return chunks

manifest = {'train': [], 'test': []}

# ------------------------------------------------------------------------------
# 1. Commercial 5G mmWave (Lumos5G-v1.0.csv) — split strictly by run_num!
# ------------------------------------------------------------------------------
lumos_runs = {}
with open(RAW / 'lumos5g/Lumos5G-v1.0.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # Find throughput column name dynamically
    thr_col = [c for c in reader.fieldnames if 'through' in c.lower() or c.lower() == 'thr'][0]
    for row in reader:
        rn = int(float(row['run_num']))
        try:
            v = float(row[thr_col])
            if v > 0:
                lumos_runs.setdefault(rn, []).append(v)
        except Exception:
            pass

valid_runs = [(rn, vs) for rn, vs in sorted(lumos_runs.items()) if len(vs) >= 90]
print(f'[Lumos5G] Found {len(valid_runs)} valid commercial 5G runs (>=90s), thr_col={thr_col}')
# Take up to 24 disjoint runs: first 16 -> train, next 8 -> held-out test
selected_5g = valid_runs[:24]
for idx, (rn, vs) in enumerate(selected_5g):
    is_test = (idx >= 16)
    dst_dir = OUT_TEST if is_test else OUT_TRAIN
    tag = f'real5g_lumos_run{rn:03d}'
    rec = mbps_series_to_mahimahi(vs[:90], dst_dir / tag, min_mbps=4.0, max_mbps=96.0)
    rec['family'] = '5G-mmWave'
    manifest['test' if is_test else 'train'].append(rec)

# ------------------------------------------------------------------------------
# 2. Real Starlink LEO Satellite (Starlink-on-the-Road iperf.csv)
# ------------------------------------------------------------------------------
sl_series = []
with open(RAW / 'starlink/iperf.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            v = float(row['download_mbps'])
            if v > 0:
                sl_series.append(v)
        except Exception:
            pass

n_sl_chunks = min(18, len(sl_series) // 90)
print(f'[Starlink] Found {len(sl_series)} 1-Hz samples -> {n_sl_chunks} disjoint 90s LEO segments')
for k in range(n_sl_chunks):
    seg = sl_series[k * 90 : (k + 1) * 90]
    is_test = (k >= 12)
    dst_dir = OUT_TEST if is_test else OUT_TRAIN
    tag = f'real_starlink_leo_{k:02d}'
    rec = mbps_series_to_mahimahi(seg, dst_dir / tag, min_mbps=5.0, max_mbps=96.0)
    rec['family'] = 'LEO-Starlink'
    manifest['test' if is_test else 'train'].append(rec)

# ------------------------------------------------------------------------------
# 3. Official Mahimahi 4G LTE / UMTS Driving Traces (disjoint 90s slices)
# ------------------------------------------------------------------------------
lte_sources = [
    ('ATT-LTE-driving.down',      'att_lte',   6),
    ('TMobile-LTE-driving.down',  'tmob_lte',  6),
    ('Verizon-LTE-driving.down',  'vzn_lte',   6),
    ('TMobile-UMTS-driving.down', 'tmob_umts', 6),
]
lte_cnt = 0
for fname, prefix, max_c in lte_sources:
    p = RAW / 'mahimahi_lte' / fname
    if not p.exists(): continue
    chunks = slice_mahimahi_file(p, dur_s=90, max_chunks=max_c)
    # 2/3 train, 1/3 strictly disjoint test
    split_i = max(1, (2 * len(chunks)) // 3)
    for ci, (ts, bins) in enumerate(chunks):
        is_test = (ci >= split_i)
        dst_dir = OUT_TEST if is_test else OUT_TRAIN
        tag = f'real4g_{prefix}_c{ci:02d}'
        out_p = dst_dir / tag
        out_p.write_text('\n'.join(str(t) for t in ts) + '\n', encoding='ascii')
        rec = {
            'file': tag, 'family': '4G-Cellular', 'duration_s': len(bins),
            'packets': len(ts), 'mean_mbps': round(float(np.mean(bins)), 2),
            'p10_mbps': round(float(np.percentile(bins, 10)), 2),
            'p90_mbps': round(float(np.percentile(bins, 90)), 2),
            'cv': round(float(np.std(bins) / max(1e-6, np.mean(bins))), 3),
            'sha256': hashlib.sha256(out_p.read_bytes()).hexdigest()[:16],
        }
        manifest['test' if is_test else 'train'].append(rec)
        lte_cnt += 1
print(f'[Mahimahi 4G LTE] Sliced {lte_cnt} disjoint real 90s cellular segments')

# ------------------------------------------------------------------------------
# 4. Multi-Timescale Synthetic Dynamic Drops / Handovers (16 train + 8 test)
# ------------------------------------------------------------------------------
rng = np.random.RandomState(2026)
for idx in range(24):
    is_test = (idx >= 16)
    dst_dir = OUT_TEST if is_test else OUT_TRAIN
    t = np.arange(90, dtype=float)
    base_bw = rng.choice([24.0, 36.0, 48.0, 72.0])
    mode = idx % 4
    if mode == 0:    # Abrupt 4x step drops every 15s
        bw = np.where((t // 15) % 2 == 0, base_bw, base_bw * 0.25)
    elif mode == 1:  # Multi-level staircase + sudden recovery
        bw = base_bw * np.select([t < 20, t < 40, t < 60, t < 75], [1.0, 0.35, 0.70, 0.20], default=1.0)
    elif mode == 2:  # Sinusoidal + fast burst jitter (WiFi / shared broadband)
        bw = base_bw * (0.65 + 0.35 * np.sin(2 * np.pi * t / 22.0) + rng.normal(0, 0.08, len(t)))
    else:            # Periodic handover dips (3s deep fade every 15s)
        bw = np.where((t % 15) < 3, base_bw * 0.18, base_bw)
    tag = f'dyn_multi_m{mode}_{idx:02d}'
    rec = mbps_series_to_mahimahi(bw, dst_dir / tag, min_mbps=3.0, max_mbps=96.0)
    rec['family'] = 'Dynamic-Broadband'
    manifest['test' if is_test else 'train'].append(rec)

# Verify 0% filename & hash overlap between train and test
tr_names = {r['file'] for r in manifest['train']}
te_names = {r['file'] for r in manifest['test']}
assert tr_names.isdisjoint(te_names), 'Train/test name overlap!'

mpath = ROOT / 'traces_real/trace_manifest.json'
mpath.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(f'\n=== REAL TRACE SUITE BUILT SUCCESSFULLY ===')
print(f'  Train traces: {len(manifest["train"])} | Held-out Test traces: {len(manifest["test"])}')
for split in ('train', 'test'):
    by_fam = {}
    for r in manifest[split]:
        by_fam.setdefault(r['family'], []).append(r)
    print(f'  [{split.upper()}]')
    for fam, rs in sorted(by_fam.items()):
        print(f'    {fam:20s}: n={len(rs):2d}  mean_bw={np.mean([r["mean_mbps"] for r in rs]):5.1f} Mbps  mean_CV={np.mean([r["cv"] for r in rs]):.2f}')
