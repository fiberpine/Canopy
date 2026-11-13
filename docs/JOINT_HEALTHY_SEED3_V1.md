# 从健康 seed3 迁移的双动作候选：离线实现

2026-11-13。用户已批准按两轮 Claude 讨论后的方案先离线实现。独立分支 `innovation/joint-cwnd-mtp-healthy-seed3-v1`，起点为 `baseline-healthy-seed3-natural-20261113`（`c67717d`）。**本版只有离线证据，没有新 TCP 测试、云端部署、训练模型或性能收益。** 当前可用健康基座仍由 `CURRENT_BASELINE.json` 指定。

## 实现与入口

|组件|入口及行为|
|---|---|
|模型|`frameworks/joint_healthy/model.py`：原窗口 actor＋独立 82→64→64→1 softsign 时间分支；联合 twin Q。窗口新输入零扩展，旧 critic 仅近似初始化|
|状态/奖励|`core.py`：原 70 维前缀＋log实际 dt、上一计划 MTP、10 个实际帧间隔，共 82 维。保留 Orca 观察峰值效用，显式记录空 RTT 帧和年龄界|
|原生调度|`native/`：自然 CUBIC 接管；四字段严格协议；每决策一次整数窗口写入；20—160 ms 等待后再读反馈，无间隔内固定 20 ms 遥测|
|计量|显式区分旧 delivery-sample mean 与 `TCP_INFO.bytes_acked` 差/单调实际 dt；计数器长度/单调性失败即停止，不静默回退|
|学习|`replay.py`、`backend.py`：真实入库物理经验账本、按 actor 均衡采样、独立保留集、critic 校准→新时间策略前向校准→联合更新|
|验证|`audit.py`、`metrics.py`：动作/ID/整数窗/时间/接管/额外读取审计，独立包日志 full/trim10 的 mean/P95/P99/利用率/丢包，采样 CPU|
|控制器|`joint_healthy.py`：独立入口。默认只生成计划/离线验证。未来真实网络需要另行明确预算、截止时间和显式授权文件|

原 `run.py`、baseline 源码、native 和模型不修改。旧 `mtpjoint`/V31 不覆盖。新模型不能按相同张量形状冒充旧模型；保存 schema、计量/时钟/成本/固定周期语义，加载时核对。迁移先校验指定 seed3 tensor 的 SHA。

窗口分支与当前 Sonnet 参数名称、moving-BN 计算一致；新实现不构造 IBP/性质图。四项修复重新实现并验证：target 使用 moving BN、双 critic 各用独立 Adam、target 变量唯一配对、online BN 四个统计各有一个 writer。时间分支无 BN，有限初值对应 40 ms；固定20诊断显式屏蔽它。

继承基座的双 Q MSE、学习率及窗口 target smoothing（高斯标准差.1，截断±.2，再裁动作界）。新增时间坐标首版不加 target smoothing；固定周期不被目标噪声改变。探索仍显式区分校准探测与联合阶段两坐标高斯噪声。

## 奖励、周期与训练账本

`D=.995**(actual_dt/.02)`，`R=u*(1-D)/(1-.995)-decision_cost`。这是区间效用恒定的近似，不能恢复未观测尖峰。默认 `decision_cost=0`，Canopy 性质奖励、QC、fallback、容量 oracle 全部关闭。显式非零调用成本留作后续配对消融，不能当 CPU 实测换算。

Python 在写出动作后等待该计划周期，再等 native 的新反馈，避免持续轮询不变的共享内存；native 才是观测时钟，实际 dt 含真实执行/调度开销。CUBIC 在等待期间正常处理 ACK。native 边界记录累计 DeepCC 读取次数；ACK 模式在同一边界另读一次 TCP_INFO。未来审计核对相邻发布间 DeepCC 次数恰增1；首次自然启动的等待读取单列。

仅真实入库的 dt 发放额度：warmup 同时达到 12288 条、每 actor 512 条和累计 245.76 s 后，按新增物理秒每20ms `.1` 次完整 critic 更新；不追补 warmup、不靠墙钟、无新经验无额度。完整 critic 迭代/两次优化器操作/actor/target 分开记录。默认 1 learner＋24 actor，尚未测试本候选的实际容量。

校准阶段冻结整个窗口参数、BN 和 actor targets，不探索窗口。联合组主要保持20ms，10%决策在20/40/80/120/160周期库均匀探测；前向阶段主要使用初始化40ms，仍少量探测。独立 fixed20/80/160 组从第一条经验开始始终保持自己的周期，包括校准，时间梯度为0。各组共享迁移起点及账本，但不能要求转移数相同。

自动阶段门同时检查：最低 critic 次数（5000，前向校准1000）、每 actor×周期保留128条；每周期至少16条未参与优化的一步样本；双 Q 的归一化 TD RMSE≤.25且相对参考不过度恶化。另保留每257条中的连续8条作为独立短回报块（不进优化器），每周期至少4块，双 Q 对折扣短回报＋末端 target bootstrap 的误差≤.35及参考相对门。**短块仍带 bootstrap，不能称完整 Monte Carlo 回报或真实价值已证准。** 默认阈值是首版待真实短准入检验的工程选择，不是文献保证；只多跑更新不能绕过覆盖/误差门。

没有登记的 G0/G1/G2 源码/native 准入记录时，不能通过公开入口开放联合学习。记录包含 `joint_healthy.py identity` 的当前实现指纹及登记/结果引用；改实现后需重新核对。它记录研究验收，不代替真实结果审核。

重连默认关闭；可显式设如180秒，各新组须相同。重连前 terminal，清空 history/max_bw，不跨连接拼接转移。checkpoint保存模型/target/BN/优化器/计数，另存 replay 与汇总；本版不提供继续训练加载入口，也不保存完整 TCP/RNG/队列现场，**不支持无缝 resume**。没有完成物理预算或没开放 actor 时必须标 PARTIAL。

## 已完成的离线证据

见 `evidence/joint-healthy-seed3-offline-20261113/`，每项写明适用范围。

- 用真实 TF1.14/Sonnet1.34、选定 seed3 checkpoint 与原 `AbstractActor`/moving 后端比较512条固定输入：窗口输出最大误差 **0**；一次 canonical Sonnet BN EMA 与新 writer 最大误差 **0**。旧 Q 输入迁移最大绝对差约 `1.22e-4`，满足相对数值容差；不称 Q 逐位一致或已校准。
- 目标策略改变同批次的其他行，固定查询输出误差0；冻结阶段窗口和 targets 逐位不变，强取 actor 更新被拒绝；开放后两个分支都有有限非零梯度且参数变化。四 BN writer、30个唯一 target 对、独立 Adam、保存加载均核对。
- 独立 fixed20/80/160 的时间输出固定、梯度0、窗口更新不改时间权重。
- 实际 learner/Replay/gate/FIFO 代码在本机合成400条数据上通过阶段切换、39次完整 critic、35次 actor、排空标记/额度检查。使用缩小的测试阈值，没有 TF Server/端口或 TCP；这些400条不是物理经验。
- C++协议小程序在 Windows 编译执行通过；整个 native 已交叉编译为面向Ubuntu18.04的 Linux ELF。**没有在目标 GCC7.5、定制内核或真实 TCP 上执行过新程序。** 交叉编译产物不进入仓库/部署路径。

## 离线复核方式

在仓库根目录，普通 Python＋NumPy可运行纯离线测试：

```sh
python -B joint_healthy.py verify-offline
python -B joint_healthy.py plan --profile joint
python -B joint_healthy.py identity
```

在现有 Python3.6/TF1.14/Sonnet1.34 环境中运行真实模型及合成学习链检查（会创建随后删除的临时 checkpoint，不连接云端）：

```sh
python -B tools/verify_joint_healthy_model.py --output /tmp/joint-model-check.json
python -B tools/verify_joint_healthy_pipeline.py --output /tmp/joint-pipeline-check.json
```

配置模板在 `templates/joint-healthy/`；`@PACKAGE@` 由 CLI 展开为当前仓库根目录，网络地址不随模板发布。G0旧时钟/旧计量固定20 → G1新时钟/旧计量固定20 → G2新时钟/ACK计量固定20，三个变化分别核对。`joint` 与 fixed 模板是后续微调候选，不会自动启动。

未来在目标 Linux **仅编译**：`python tools/build_joint_healthy_native.py`。它生成独立 binary 和源码 SHA 回执，不覆盖基座；若已有不同 binary，会要求另选输出，避免覆盖证据。使用前部署到各机相同绝对路径和运行环境，检查每机源码/native/trace 指纹、内核接口及严格 SSH known_hosts。

## 尚未通过的上线阶段

1. 新 native 的旧时钟兼容 TCP 门；G1 scheduled20 的空 RTT 帧/分布/配对性能；G2 ACK计量与独立包日志核对。三者均未跑，不能以离线窗口一致代替。
2. 同一固定20轨迹的奖励偏差审计：`tools/audit_joint_healthy_reward.py` 能按真实字节与 dt 离线合并20/40/80/160段；目前只有合成测试，没有符合新 ACK 边界格式的真实轨迹。它不能重建 P95/P99或推断改周期后的 TCP 行为。observed-peak 归一化的周期偏差尚未解决。
3. 有限真实校准、解除冻结的固定20回归、双方梯度/BN/饱和监测，再做有预算的联合微调；不能直接长训掩盖迁移失败。
4. 匹配起点独立训练 fixed20/80/160，完整24条件×3及动态/浅缓冲留出；使用同轮参照、既定利用率/mean/P95/P99/丢包门。只强制同权重周期是诊断，不能替代这些对照。
5. 同机同日志同核数的真实 CPU 对照。当前资源工具报 Python/native/含Mahimahi进程树 CPU 秒及整机 busy，使用两秒采样窗口；可能漏短命进程和末尾CPU，窗口也不自动等于 packet trim10。需先核对采样覆盖和 full/trim10 一致口径，不能直接把采样下界叫精确完整run开销，不能据调用减少宣布CPU收益。

已批准设计和两轮 Claude 讨论摘要作为历史记录保存在同目录证据中；没有请 Claude 审查本次新代码，不把此前设计共识当成代码审计。所有后续真实网络实验仍需新的明确预算，当前可保持服务器关闭。
