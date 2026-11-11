# Canopy 双动作拥塞控制

在 Orca/Canopy（CUBIC + 强化学习调窗口）之上，让同一个策略同时输出**窗口倍率**和**下一个监测周期 MTP**，目标是在保持性能的同时降低推理调用开销。

## 当前状态（2026-11-11）

- **单动作基线尚未合格。** 固定 20 ms、70 维状态、单窗口动作的基线从随机初始化训练过三次（同一初始化），结果分别是：归档模型（高吞吐、高排队）、中途塌缩、2026-11-10 终点（低排队、部分条件低吞吐）。最新一次见 [evidence/healthy-base-20261110/result.md](evidence/healthy-base-20261110/result.md)。
- **双动作（mtpjoint，V31）暂停。** 在基线合格之前不继续扩展。
- 逐次实验的结论、失败记录和旧入口见 [docs/HISTORY.md](docs/HISTORY.md) 与 [EXPLORATION_HISTORY.md](EXPLORATION_HISTORY.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `run.py` | 唯一支持的训练/评测入口 |
| `frameworks/baseline/` | 单动作基线：`distributed.py`（learner/actor）、`evaluate.py`（真实 TCP 闭环）、`cluster_control.py`（五机调度） |
| `frameworks/mtpjoint/` | 双动作 V31：观测与执行同周期（论文主线） |
| `frameworks/mtp20/` | 双动作 V14：固定 20 ms 观测（早期方案，保留作对照） |
| `frameworks/canopy_single/`、`frameworks/baseline_ipc_audit/` | 基线诊断用的研究框架，不是训练入口 |
| `frameworks/*/code/canopy`、`code/orca` | 上游 Canopy / Orca 源码（模型、奖励、内核补丁） |
| `frameworks/*/code/runtime` | 本项目的 native 服务端源码与二进制 |
| `models/` | 选定模型：`baseline`、`mtp20`、`mtpjoint`、`fixed20…160`、`official-orca` 及两个先验 |
| `templates/` | 各模型的训练/评测配置模板 |
| `traces/` | Mahimahi 链路 trace（训练格、动态与留出条件） |
| `configs/` | 集群配置样例 `cluster.example.json` |
| `tools/` | 离线审计与单元测试（不连云、不跑 TCP） |
| `evidence/` | 每轮实验的登记、结果和失败记录，按日期命名 |
| `docs/` | 各实现说明与研究文档 |
| `runtime/` | Mahimahi 源码与 IBP 依赖；Python 环境和内核包在 Release 里 |
| `environment/` | 依赖锁定、原始训练配置、可移植性补丁 |

## 环境

Ubuntu 18.04 x86_64、Python 3.6、TensorFlow 1.14、定制内核 `4.13.1-0521-learner`、Mahimahi、TUN、sudo。Windows 只能看源码和跑离线测试，不能跑 TCP 实验。

```sh
git clone https://github.com/fiberpine/Canopy.git && cd Canopy
python3 tools/verify_repository.py                 # 校验源码与模型文件
mkdir -p assets
gh release download consolidated-20261018 --repo fiberpine/Canopy \
  --pattern orca-portable-release-20261018.tar.gz --dir assets
python3 tools/install_runtime_assets.py --archive assets/orca-portable-release-20261018.tar.gz
bash setup.sh --system    # 新机：装依赖、编译并安装 Mahimahi、装内核；不会自动重启
# 重启进入 4.13.1-0521-learner 之后：
bash setup.sh
runtime/venv/bin/python build_native.py
runtime/venv/bin/python run.py doctor
```

Python 环境和内核 deb 不在 Git 里，在 [Release `consolidated-20261018`](../../releases/tag/consolidated-20261018)。细节见 [docs/PORTABLE_RELEASE.md](docs/PORTABLE_RELEASE.md)。

## 常用命令

```sh
# 冻结评测（关闭探索和优化），结果在 frameworks/<实现>/logs/<run_id>/result.json
runtime/venv/bin/python run.py evaluate --variant baseline --scenario 0 --seconds 60

# 本机 1 actor 短测，只验证真实更新链路
runtime/venv/bin/python run.py smoke-training --variant baseline --actors 1

# 五机正式训练：先复制 configs/cluster.example.json 填好自己的主机和 SSH key
runtime/venv/bin/python run.py train --variant baseline --actors 24 --samples 2400000 \
  --cluster configs/my-cluster.json
```

`--variant` 可取 `baseline`、`mtp20`、`mtpjoint`；评测还可取 `fixed20/40/80/120/160`、`official-orca`、`cubic`、`reno`。训练规模是 1 learner + 24 actor（四台 actor 机各 6 个环境）。每次运行使用新的 run ID，不覆盖旧结果。

离线测试（不需要内核和 TensorFlow，只需 NumPy）：

```sh
python3 -m unittest discover -s tools -p 'test_*.py'
```

## 文档

- 实现说明：[baseline](docs/BASELINE.md)、[mtp20 / V14](docs/MTP20_V14.md)、[mtpjoint / V31](docs/MTPJOINT_V31.md)、[模型清单](docs/MODELS.md)
- 研究审阅入口：[当前问题与已确认事实](docs/RESEARCH_REVIEW_BRIEF.md)、[基线第一性原理审计](docs/BASE_FIRST_PRINCIPLES_REVIEW.md)
- 验证范围与限制：[docs/VALIDATION.md](docs/VALIDATION.md)
- 第三方代码与许可证：[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)

## 维护约定

- 仓库用 `repository-files.json` 记录每个文件的 SHA256。增删文件后运行 `python3 tools/update_repository_inventory.py` 重新生成，再用 `tools/verify_repository.py` 校验。
- Mahimahi 的编译产物不入库，由 `setup.sh --system` 重新生成。
- 上游 Orca 的示例 trace（`wired6/48/96` 等）不入库，需要时从 [Soheil-ab/Orca](https://github.com/Soheil-ab/Orca) 获取；本项目训练和评测只用 `traces/`。
- 密钥、`configs/my-cluster.json`、运行日志和 checkpoint 不入库（见 `.gitignore`）。
