# Canopy：baseline与两种MTP双动作实现

本仓库保存三个原始交付实现、选定模型、训练/评测框架，以及后续基线与双动作研究的代码和证据。**截至2026-10-27，研究仍为PARTIAL：新的Orca基座未通过多场景验收，Canopy正式训练中途按用户要求暂停，双动作模型尚未证明稳定的性能—CPU优势。**

最新审阅入口是[研究交接](docs/RESEARCH_REVIEW_BRIEF.md)。 最新完成[真实TF首批更新取证](docs/SCALAR_FIRST_UPDATES_DIAGNOSIS.md)：16个离线case复现快速失稳，冻结BN/仅校准都不足；校准＋低学习率更稳定，但局部状态仍有较大偏移。[完整结果](evidence/scalar-first-updates-20261027/final-analysis.json)保留严格逐位等值失败及未通过预测；没有新TCP实验或合格基座。以下L1为此前历史。最新[Orca单动作有限学习L1](docs/SCALAR_LIMITED_LEARNING_PLAN.md)在24 actor零优化检查通过后，28,513条真实经验时因百万窗口安全停止；14 actor触发，原计划110次冻结评测全部NOT_RUN。没有新增合格基座或双动作性能—CPU收益。[48条actor连接结果](evidence/scalar-limited-learning-20261027/actor-outcomes.csv)、[机器可读结论](evidence/scalar-limited-learning-20261027/final-results.json)及[TF/BN同输入诊断](evidence/scalar-limited-learning-20261027/tf-postmortem.json)完整保留失败。最终checkpoint在排空更新后偏负，不能代替在线正饱和时的策略快照。

此前[累计探索/缓冲诊断](docs/SCALAR_EXPLORATION_AND_BUFFER_CAP.md)、[双向IPC/初始化诊断](tools/scalar_ipc_diagnostic/README.md)、[E2接管诊断](tools/e2_handover/README.md)保留全部结果。原始快照tag `consolidated-20261018` 和51个选定模型文件未变；归档源码不意味着论文复现或新模型合格。

| 入口名称 | 选定实现 | 状态/动作 | 观测周期 | RL窗口动作执行周期 | 详细说明 |
|---|---|---|---|---|---|
| `baseline` | 冻结seed3，step2412694 | 70/1 | 原20ms | 原20ms | [baseline](docs/BASELINE.md) |
| `mtp20` | V14，step277317 | 72/2 | 固定20ms | 学习20—160ms | [V14](docs/MTP20_V14.md) |
| `mtpjoint` | 健康V31，step157704 | 149/2 | 学习20—160ms | 与观测同周期 | [V31](docs/MTPJOINT_V31.md) |

两个创新均保留原窗口动作：归一化输出映射为窗口倍率，执行端按整数百分比写入；CUBIC继续逐ACK工作。MTP不是重复施加窗口倍率的次数。V31长MTP期间没有隐藏20ms RL观测，没有提前唤醒、QC或fallback。

## 完整性与交付方式

- Git直接包含原三套源码、native源码和二进制、必要本地依赖、**原始11组模型与先验**、训练/评测模板、trace、IBP和Mahimahi源码、性能与失败证据、原始来源清单及许可证文件。后续研究新增checkpoint/replay仍在原实验存储，不在Git中，不能把证据中的来源路径当成已下载模型。
- 完整Python3.6/TF1.14环境和定制内核安装包放在同一仓库的[Release](../../releases/tag/consolidated-20261018)。完整迁移包也包含上述源码与模型，可独立解压使用。它不依赖原五台服务器的路径。
- Git与Release共同构成完整交付；直接clone尚未包含Python环境和内核deb。历史replay、所有旧checkpoint、逐包历史档案不是当前运行依赖，没有上传到GitHub。选定模型和必要先验均已包含，绝不是只有checkpoint路径的空索引。
- [模型清单](docs/MODELS.md)、[原便携包说明](docs/PORTABLE_RELEASE.md)、[验证范围](docs/VALIDATION.md)、[试错记录](EXPLORATION_HISTORY.md)、[来源清单](manifest.json)。历史绝对路径仅说明来源，原实验目录已退役，不用于新运行。

## 快速使用完整迁移包

要求：**Ubuntu18.04 x86_64、Python3.6、定制4.13.1-0521-learner内核、Mahimahi、TUN、sudo。** Windows只用于查看源码，不能直接执行TCP实验。未验证任意新OS的一键安装，也未通过重装系统验收。

```sh
gh release download consolidated-20261018 --repo fiberpine/Canopy \
  --pattern orca-portable-release-20261018.tar.gz \
  --pattern orca-portable-release-20261018.tar.gz.sha256
sha256sum -c orca-portable-release-20261018.tar.gz.sha256
tar -xzf orca-portable-release-20261018.tar.gz
cd orca-release-20261018
bash setup.sh --system # 新机安装依赖、构建Mahimahi、安装定制内核；不会自动重启
# 重启到4.13.1-0521-learner后：
bash setup.sh
runtime/venv/bin/python build_native.py
runtime/venv/bin/python run.py doctor
```

仓库已于2026-10-20从Private改为Public，源码与Release资产均可公开访问。上述gh命令需要官方GitHub CLI，也可直接在Release网页下载同名资产再传到目标服务器。没有在仓库中嵌入token或SSH私钥，不需要访问原五台服务器。若服务器已有匹配内核与工具，可跳过系统安装与重启。

## 使用Git源码

```sh
git clone https://github.com/fiberpine/Canopy.git
cd Canopy
python3 tools/verify_repository.py
mkdir -p assets
gh release download consolidated-20261018 --repo fiberpine/Canopy \
  --pattern orca-portable-release-20261018.tar.gz --dir assets
python3 tools/install_runtime_assets.py --archive assets/orca-portable-release-20261018.tar.gz
bash setup.sh # 新机仍须先按上节安装系统依赖及定制内核
runtime/venv/bin/python build_native.py
runtime/venv/bin/python run.py doctor
```

`run.py`是本交付支持的训练/评测入口。旧helper中保留的历史pipeline/factory CLI不是迁移入口；不要直接重启旧pipeline。仓库包含`setup.sh`依赖与完整原生代码。

## 三套模型的评测和训练

```sh
runtime/venv/bin/python run.py evaluate --variant baseline --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant mtp20 --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant mtpjoint --scenario 0 --seconds 60
# 90秒打包短测降低warmup，仅验证真实更新链路：
runtime/venv/bin/python run.py smoke-training --variant mtpjoint --actors 1 --seconds 90
```

正式训练先复制并填写`configs/cluster.example.json`；使用自己的SSH key和known_hosts，保持主机密钥验证。所有机器采用相同绝对入口路径，或使用同名软链接；路径不能含空格。

```sh
runtime/venv/bin/python run.py train --variant baseline --actors 24 --samples 2400000 --cluster configs/my-cluster.json
runtime/venv/bin/python run.py train --variant mtp20 --actors 24 --seconds-per-actor 1800 --cluster configs/my-cluster.json
runtime/venv/bin/python run.py train --variant mtpjoint --actors 24 --seconds-per-actor 600 --cluster configs/my-cluster.json
```

实际规模为1learner＋24actor，四台actor机器各6个独立环境；通过重连轮换环境，不是同时运行56actor。默认不传cluster时为本机1actor。每次使用独立run ID，禁止覆盖旧结果。训练会使用明确的有限时间/样本预算；调用命令即启动对应工作，不会自动继续下一轮实验。

评测关闭探索和优化，结果位于`frameworks/<实现>/logs/<run_id>/result.json`。默认场景列表因实现不同而不同；正式比较须统一trace、RTT、队列、启动、评测窗口和重复数。固定周期入口`fixed20/fixed40/fixed80/fixed120/fixed160`均为独立训练模型，另可评测`official-orca/cubic/reno`。同权重强制周期诊断不能冒充独立训练对照。

## 结果和限制

本仓库保留逐场景吞吐/利用率、平均/P95/P99排队、丢包、调用及CPU指标；不能只凭总平均或调用数宣布优势。[V31说明](docs/MTPJOINT_V31.md)列出了弱场景与正常启动参照，完整多算法结果在`evidence/multialgorithm`。

V14四个浅队列动态条件MTP变化不足；V31仍有浅队列/高RTT吞吐及尾延迟弱项。一个训练seed来源的评测重复不等于多训练seed。少调用不等于同比例省CPU。完整resume加载器不存在；V14/V31正式模板按原谱系从包内先验新建优化器/replay/计数/TCP，并非从最终交付模型无缝续训。

三套通过重编译、离线模型更新与重载、迁移后1actor真实训练、四机4actor训练、冻结评测及完整包解压验证。初次同机并行解包冻结中两MTP首次反馈超时，原因未定位，失败日志保留；前后顺序冻结均通过。本交付认证顺序冻结流程，不宣称同机多冻结流已通过。

上游代码及第三方文件保留原许可证；仓库现公开可读；本次没有为新增研究代码另行选择开源许可证，第三方代码继续遵循各自许可证。来源见[第三方说明](docs/THIRD_PARTY.md)。

## 下一步研究审阅

请先读[当前研究状态与优先问题](docs/RESEARCH_REVIEW_BRIEF.md)。它区分已确认事实、待验证机制与最小实验。2026-10-26更新同步后续研究源码和小证据，没有为发布新增TCP实验或训练，也没有替换原模型。

可直接检查：

- [单动作研究框架及有限运行入口](frameworks/canopy_single/README.md)，包含明确的artifact/initializer QC模式及正常启动native源码；与原`baseline`身份不同。
- [V31兼容校准实现与实验](docs/V31_COMPATIBLE_CRITIC_CALIBRATION.md)，新方案均需显式启用，价值门槛通过不等于网络性能收益。
- [两组Orca全部144次冻结结果](evidence/scalar-20261025/evaluation_runs.csv)、[逐条件均值](evidence/scalar-20261025/evaluation_summary.csv)，保留full/trim10和失败方向，不只报告总体平均。
- [16个checkpoint只读方向审计](evidence/scalar-direction-20261026/learner-outcome.json)和[可复现原奖励数值核查](tools/scalar_audit/audit.py)。Q网格和梯度是模型预测，不能冒充真实网络反事实。

```sh
# 本机数值/证据检查，无网络控制实验；需要NumPy
python3 -m unittest discover -s tools -p 'test_v31*.py'
python3 -m unittest discover -s tools/scalar_audit -p 'test_audit.py'
python3 -m unittest discover -s tools/interval_reward -p 'test_shadow_reward.py'
python3 tools/scalar_audit/audit.py --output diagnostics/scalar-offline-audit.json
```
