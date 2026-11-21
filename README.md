# Fiber: Joint Congestion-Window and Decision-Period Control for Low-Latency, Low-Overhead DRL Congestion Control

This repository contains the official mainline implementation, 100%-fresh full-suite evaluation datasets, and all 10 publication figures for **Fiber** — a hybrid kernel/DRL congestion control framework that jointly optimizes **congestion window scaling (`cwnd`)** and **adaptive decision period (`MTP`)** with **2D Joint-Action Interval Bound Propagation (2D IBP)** formal safety verification and a **Copa-principled competitive coexistence wrapper**.

---

## 1. Mainline & Paper Model Family

| Model Name | Checkpoint Identifier | Role in Paper |
| :--- | :--- | :--- |
| **Fiber (Mainline)** | `dau-jsac-v3-eco-seed7-900/step-517519` + `CopaCompetitiveWrapper` | Primary proposed protocol (best utilization–delay–overhead Pareto knee + competitive coexistence) |
| **Fiber-Bal** | `dau-jsac-v3-balanced-seed3-900/step-428741` | Ultra-low-delay operating point (`16.47 ms` global queue) |
| **Fiber-s3 / Fiber-s11** | `dau-jsac-v3-seed3-900`, `dau-jsac-v3-eco-seed11-900` | Multi-seed training stability family (Fig. 10) |
| **Fiber-NoIBP** | `dau-jsac-v3-noibp-seed3-900/step-305875` | Formal safety ablation without 2D Joint-Action IBP (Fig. 9 & Fig. 10) |

Key implementation files in [`code/frameworks/joint_healthy/`](code/frameworks/joint_healthy/):
- `copa_coexistence.py` — Runtime Copa NSDI'18-principled competitive coexistence wrapper (`beta = 0.0` on solo flows; smooth blending toward exact hybrid Kernel-CUBIC fixed point `a_w = 0` under persistent standing queues).
- `dau_runtime.py` — Unified observation, action mapping, and SMDP runtime.
- `configuration.py` — Canonical `canopy-dau-jsac-v3` training/evaluation configuration.

---

## 2. Headline Results (100% Fresh Full-Suite Rerun, 10 Algorithms × 40 Scenarios)

Evaluated across **40 scenarios** (**24 Wired Grid** `12–72 Mbps × 20–200 ms` + **16 Real Dynamic Traces**: 4 Commercial 5G mmWave Lumos5G, 4 Real LEO Starlink, 4 Real 4G LTE Driving, 4 Multi-Timescale Dynamic Drops):

| Algorithm | Mean Utilization (%) | Mean Queue Delay (ms) | P95 Queue (ms) |
| :--- | :---: | :---: | :---: |
| **Fiber (Ours, Mainline)** | **95.86 ± 1.39%** | **18.95 ± 3.34 ms** | **42.61 ms** |
| **Fiber-Bal** | 92.76 ± 2.52% | 16.47 ± 2.72 ms | 40.73 ms |
| **Official Orca** | 95.35 ± 1.90% | 35.19 ± 8.43 ms | 71.51 ms |
| **Orca-Base** | 92.01 ± 3.22% | 22.79 ± 6.08 ms | 51.85 ms |
| **Official Canopy** | 89.78 ± 2.56% | 22.79 ± 9.51 ms | 77.99 ms |
| **PCC-Vivace** | 94.88 ± 0.51% | 163.26 ± 61.78 ms | 468.44 ms |
| **BBR** | 96.52 ± 0.18% | 86.38 ± 11.62 ms | 181.43 ms |
| **CUBIC** | 99.81 ± 0.07% | 300.03 ± 48.95 ms | 551.35 ms |
| **Aurora** | 65.00 ± 4.13% | 33.27 ± 12.68 ms | 182.74 ms |
| **Indigo** | 54.92 ± 4.20% | 7.58 ± 1.81 ms | 18.52 ms |

- **Dynamic Step Tracking (Fig. 4, `48→12→48→24→72→24 Mbps`)**: Fiber achieves `96.72%` util and `15.24 ms` queue (`48 ms` P95) while cutting DRL policy calls by **−65.2%** vs. Official Orca (`16.7 vs 48.0 calls/s`, `20.44 ms` queue, `111 ms` P95).
- **Intra-Protocol Fairness (Fig. 7, 120s 4-Flow Join-and-Leave)**: Fiber achieves `Jain = 0.723, Queue = 22.7 ms`, beating both Official Orca (`0.689, 116.8 ms`) and BBR (`0.672, 155.8 ms`) with 5.1×–16.2× lower queue.
- **Inter-Protocol Coexistence & RTT Fairness (Fig. 8)**: On `5 BDP vs. CUBIC`, Fiber secures `25.7%` bandwidth share (`Jain = 0.809`, >50× Orca-Base/Canopy `0.5%`); on `20 vs. 100 ms` heterogeneous RTT, Fiber achieves `Jain = 0.9940` at `27.3 ms` queue (vs. CUBIC `0.9420` at `84.2 ms`).

---

## 3. Repository Layout

```text
├── code/                    # Mainline Fiber framework (Python + C++ native actors + Copa wrapper)
├── results/
│   ├── data/
│   │   ├── fresh_20261121/  # 100%-fresh JSON datasets (Stage 1-4, Fig. 4 step, Fig. 7 join-leave)
│   │   └── supplementary/   # Exp 4 (MTP Pareto), Exp 5 (2D IBP), Exp 7 (CPU scaling) JSONs
│   └── figures/             # All 10 publication figures (fig1..fig10 in PNG, SVG, PDF)
└── scripts/                 # Self-contained plotting scripts to regenerate all figures
```

## 4. Regenerating Publication Figures

```bash
python scripts/plot_100pct_fresh_all10.py
python scripts/plot_fig3_upgraded.py
python scripts/plot_fig4_fig7_upgraded.py
python scripts/plot_fig7_orca_style.py
```
