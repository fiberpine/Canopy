# E2冻结接管诊断

该目录保存2026-10-27实际使用的单动作Orca接管状态机、评测hook、逐机顺序driver、历史窗口扫描和边界审计，以及本次发布后的路径适配。**这是诊断工具，不是已通过性能验收的新基线或双动作模型。** 完整结果和失败边界见[研究交接](../../docs/RESEARCH_REVIEW_BRIEF.md)。

## 无服务器复算

在仓库根目录执行：

```sh
python -m unittest discover -s tools/e2_handover -p test_e2_support.py -v
python tools/e2_handover/analyze.py
```

第一项为5项NumPy状态机测试，不冒充真实TF/TCP验证。第二项读取全部72次已发布正式结果，重算144个窗口、48组均值、配对差及预注册门槛，默认输出到被忽略的`diagnostics/e2-handover-20261027`。不会连接云服务器、读取大checkpoint或启动训练。`audit-notes.json`中的尾部和遥测限制是原始日志后续审计记录，复算时作为明确的附加说明保留。

## 干预定义

| 臂 | 行为 |
|---|---|
| A0 | 第一条反馈立即执行冻结策略 |
| A1 | 10秒中性窗口动作（100%），CUBIC继续运行；观测、max_bw及历史照常更新，再交权 |
| A2 | 同A1；交权边界先清零max_bw，随后由原环境处理当前反馈；历史为零63维加当前7维 |
| A3 | 只重置历史、保留max_bw；代码支持，但本轮正式评测没有执行 |
| A4 | 立即交权，max_bw设配置容量下限；仅特权诊断，不能用于部署性能主张 |

A1/A2预启动期间仍查询并记录原策略，但执行中性动作；这是单动作固定20ms诊断，不是V31部署调度。A2保留TCP/CUBIC、RTT等完整连接状态，不能称为“只保留cwnd”。当前满速反馈能在清零后立即重建参考，因此A1与A2相近并不排除带宽参考的作用。

正式72次为6条件×4臂×3冻结重复。18个condition/repeat块分配到5机，每块4臂同机顺序、块内随机顺序，每机同时只有一条TCP流。CPU只做同机配对；full/trim10共享同一次25..50s CPU测量。单训练seed，3次冻结重复不是3个训练seed。

## 在兼容Linux环境准备新诊断

需要文档规定的TF1.14/Python3.6、自定义4.13.1内核、Mahimahi、原生正常启动程序和完整scalar框架。源moving step3557457 checkpoint仍保留原实验存储，**此次没有把它上传至Git**；原有51模型不能代替该模型冒充复现。

先自行审阅并创建有限逐机计划JSON：顶层为`deadline_epoch`、`pilot`和`formal`，后两项是完整评测config列表。每个config必须包含`e2_arm`、`e2_warmup_s`、`e2_repeat`、独立`run_id`、真实checkpoint/native路径和完整场景配置。字段形式见[注册记录](../../evidence/e2-handover-20261027/registration.json)。公开记录中的路径占位符和旧绝对截止时间只是provenance，不能直接执行，也不能默认为新授权。

```sh
export CANOPY_E2_FRAMEWORK=/absolute/path/to/scalar-moving-framework
export CANOPY_E2_WORKSPACE=/absolute/path/to/new-e2-workspace
python tools/e2_handover/prepare.py \
  --framework "$CANOPY_E2_FRAMEWORK" --workspace "$CANOPY_E2_WORKSPACE" \
  --plan /absolute/path/to/reviewed-node-plan.json
# prepare只复制小工具、建立只读来源链接和输出目录，不启动TCP。
# 以下命令会执行各自明确选择的冻结计划：
python "$CANOPY_E2_WORKSPACE/node_driver.py" --stage pilot
# 先检查pilot原始结果和e2-contract，再明确执行：
python "$CANOPY_E2_WORKSPACE/node_driver.py" --stage formal
```

driver要求发送端`tcp_no_metrics_save=1`，每次使用`sudo -n /sbin/ip tcp_metrics flush all`清缓存。需拥有相应权限。它不自动修改全机sysctl、重试、延长预算、训练或启动其他节点。新目录和run ID禁止复用；STOP标记在下一次运行前检查，正在运行的单次评测仍有时间上限。机器重启后的TF/TCP迁移、路径适配及新增prepare包装**未在本次发布时重新验收**。

`history_scan.py`需显式设置`CANOPY_E2_HISTORY_ROOT`及`CANOPY_E2_WORKSPACE`；其FIRST优化时间是本次历史实验常数，换实验必须重审，不能直接套用。`boundary_audit.py`只在本节点正式计划COMPLETE后读原始raw.csv；其packets_out/cwnd并非完整的cwnd-limited判据。已知异常字段必须保留，不能拿异常均值作结论。

## 身份与限制

目录常数改为显式环境变量，改动可逆，原执行字节SHA和逐项替换记录在[发布来源](../../evidence/e2-handover-20261027/publication-provenance.json)。这不等于修改原网络策略，也不能把发布commit称作10-25训练入口。个人SSH调度和传输包装未作为通用CLI发布；核心实际执行逻辑、完整小结果、模型/IPC收尾校验均保留。

本轮支持接管条件影响性能，但正常条件守门失败，固定10秒预启动被拒绝。原V31、baseline、V14及其模型未改，无新训练、无新性能合格模型。
