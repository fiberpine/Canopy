# Canopy：baseline与两种MTP双动作实现

[11月13日启动诊断续接结果](docs/BASELINE_STARTUP_VALIDATION_RESULT_20261113.md)：40次配对全部完成、0执行错误，首反馈接管仍FAIL；另12次自然Cubic接管仅4次完成，吞吐99.51%—100%，2次中断、6次未开始，整组INCOMPLETE。窗口已过期，续接只读证据和终检；未运行seed7、未删旧基座，下一步先补齐有限诊断再独立确认。

[在线BN最小修复：23/24格通过，完整健康门仍FAIL](docs/BASELINE_ONLINE_BN_REPAIR_20261112.md)：1learner+24actor从零完成601万经验/600090完整更新；新模型72次全部完成，96.81%利用率/16.79ms排队/trim10零丢包。G12-R200三次为89.58%/32.00%/3.43%，触发灾难门。已有日志证实该格第一条RL动作前native窗口已达357万—384万；首条策略实际最大减窗。完整结果、真实最终策略、原生二进制和校验已保留，不覆盖原H0，未追加训练。

[Claude复核与启动单变量候选](docs/BASELINE_STARTUP_NEXT_20261112.md)：该页记录候选最初的离线准备状态；后续已编译和完成40次配对，首反馈方案未通过，最新结果见页首。原FAIL不变，窗口不自动续期。

[当前候选与最初H0、三次波动详解](docs/BASELINE_COMPARISON_AND_VARIANCE_20261112.md)：同23格吞吐基本相同、平均排队少49.92%；12格排队改善、11格变差，总体收益较多来自两处旧高队列。G12-R200三次约16秒/45秒恢复或60秒内未恢复，逐秒计数重建与评分一致；启动冲量和策略恢复需隔离。仅本机重算与制图，无新云端实验。

[600万样本修复重训：最终端点不通过健康门](docs/BASELINE_6M_REPAIR_RESULT_20261112.md)：训练完成601万经验/约60万完整更新；冻结95次完成、1次中断。新模型已完成重复的条件等权98.86%利用率/281.06ms排队/15.53%丢包，23个完整三重复格排队均失败，缺失run不能逆转结论。五机最终终检通过，未补跑到期任务。

[本轮RL源码调研与BN新实测](docs/BASELINE_RL_RESEARCH_20261112.md)：记录上一轮诊断时4个BN统计各有2个未加锁写入和训练梯度恢复。之后已完成去重BN的独立长训及最终冻结，最新状态见页首；该文的“尚待评测/长训”为历史阶段。附TD3/CrossQ/DDPG原始资料及与当前配方的边界。

[11月12日更新契约实测与容量准入](evidence/baseline-capacity-repair-20261112/summary.md)：已修复可复现的早期共享Adam竞争，24/32容量通过、48因更新积压未准入；24actor随机seed3的名义6M训练已完成，终点因排队/丢包过大不接受为新健康基座。上轮实际云端native与H0相同，已更正此前将本机缓存ELF差异当作实际运行差异的表述。

[五机低利用率与actor扩容分析](docs/BASELINE_CAPACITY_AUDIT_20261112.md)：现有五台值得短测32/48，但低整机CPU不代表learner吞吐有同比余量。历史48actor在CPU约47%时已有更新积压；当前U=.1的更高并发仍待实测，本轮0云连接。

[规模、开源差异和全程异常的离线分析](docs/BASELINE_SCALE_AND_FAILURE_AUDIT_20261112.md)：官方一轮名义12.8M与本轮2.418M须区分；已核对固定上游源码、588行训练日志及同权重重复的不稳定。不能仅归因于机器少，也不能排除训练不足。本轮0云连接，五机可以关闭。

[目标BN单变量完整重训与最终评测](evidence/baseline-target-retrain-20261111/summary.md)：短准入通过，fresh seed3完成2,418,460条/240,617完整更新；最终健康 **FAIL**（利用率91.77%、mean queue 274.49ms）。原72项70完成、2项首次策略动作前启动失败，2项同权重补测单列。正式训练因清理超时保留FAILED，完整终点已持久化且独立验账；五机最终收尾通过。没有新健康基座、第二训练seed或双动作收益结论。[下一步复核方向](docs/BASELINE_TARGET_RETRAIN_20261111.md)。

[与 Claude 复核后的下一轮方案](docs/BASELINE_NEXT_TRAIN_20261111.md)：先只读核对目标BN，再决定BN修复或重连这个单一训练变量。正常入口配置、逐连接清理和真实保存压力准入仍需补齐；目前正式重训NO-GO，本次没有新云端实验。

[11 月 11 日筛查与准入结果](evidence/baseline-screen-20261111/summary.md)：24 次冻结执行成功，四个候选未过六条件粗筛；两次重连准入均在更新前失败，72 次确认与正式重训未启动。[独立重连修复](docs/BASELINE_RECOVERY_20261111.md)已通过 173 项离线测试，最新端口及清理修复仍待真实准入，没有新健康基座。

[2026-11-10 健康基座重训](evidence/healthy-base-20261110/result.md)。固定 20 ms 从头训练完成，终点未达到完整健康门槛。包含五机负载诊断、同期归档对照和清理记录。

[固定 20 ms 训练 S20（2026-11-09）](evidence/s20-20261109/s20-result.md)。学习器在第 600 秒里程碑保存期间入库队列溢出，训练提前结束，116 次正式冻结评测均未执行。离线保存修复候选已验证数据等价，尚待并发短测。原 FAIL_A 保留，B0/C 未启动。

本仓库保存三个原始交付实现、选定模型、训练/评测框架，以及后续基线与双动作研究的代码和证据。**截至2026-11-12，研究仍为PARTIAL：新的单动作基座仍未通过完整性能验收，双动作模型尚未证明稳定的性能—CPU优势。**

此前：[D0补采后的零TCP复核](docs/COMMON_PERIOD_D0_20261103.md)：容量有部分可读信息、跨条件泛化差；A20高队列训练状态约4万条。四个周期扫描的原效用最高均值是40/40/80/80ms，不能宣称80ms统一最优或critic因果已证。0新TCP/0 Canopy优化；奖励A′与D-Q均未启动，原FAIL_A不变。

此前：[共周期条件窗口执行与264次冻结评测](docs/COMMON_PERIOD_20261102.md)：A0和24actor短准入已通过；A单seed完成711,285条/143,919完整更新，264次评测0运行失败。A门槛状态 **FAIL_A**，研究仍PARTIAL，不宣称最终周期选择器或双动作CPU收益已通过。完整正负结果、原方案和运行谱系保留。

此前入口：[2026-11-02全面离线审计：环境、数据、目标与双动作控制](docs/BASE_FIRST_PRINCIPLES_REVIEW.md)。复算96个历史冻结窗口、核对有效训练链，并执行原奖励/SMDP数值检查；没有新云连接、TCP或TF训练。原奖励存在吞吐尺度别名及负净效用时延迟惩罚反向，但不能单独解释最新失败；已知修复也没有在所有默认路径中启用。新方案先建立可复算、同测量/真实时钟的固定周期参考，再验证有支持的双动作学习。[本轮计算证据与边界](evidence/comprehensive-audit-20261102/results-v2.json)。

此前五机fresh seed3原配方实际1,675,235条/166,294完整更新，触发原1.5M后180秒持续低窗口低利用守卫，**2.4M未完成、没有新健康H0**。48停点/归档冻结trim10分别44.66%/190.57ms和97.90%/32.25ms。106初始变量逐项相同不保证异步轨迹相同；历史曾恢复不保证本次恢复。[原失败、NOT_RUN、停止快照与完整证据](evidence/base-reproduction-20261028/final-analysis.json)保留；没有追加训练。

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
