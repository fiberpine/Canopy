# 最新：累计探索与端主机缓冲对照（2026-10-27）

新增的是零优化训练采样诊断，未训练成功模型。zero_sum16/16过窗口门槛，iid2/16百万窗口、8/16持续时间失败；32次配对模型初值96变量相同。4次大缓冲复核过窗口门槛，但binding proxy>.5的预测全未过，不能省略。原始初始化诊断和IPC工具说明保留在后文。

无需服务器的主试验复算：

```sh
python3 tools/scalar_ipc_diagnostic/analyze_exploration.py --results evidence/scalar-exploration-20261027/formal-results.json --registration evidence/scalar-exploration-20261027/registration-master.json
python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/verify_repository.py
```

`analyze_exploration.py`使用`same_day_all`支持全部16组同日同机配对；默认历史reference仍是a969f67的初始化结果，仅供来源对照，P2主判断实际使用本轮同日iid。判定只供后续审阅，不自动开长训。`make_exploration_registration.py --same-day-all`生成32次计划；默认20次是Claude原提案，实际执行32次。必须另用新目录、新预算与实际路径，不能重启证据中的已结束计划。

新代码：`frameworks/canopy_single/scalar_exploration.py`的rho0平稳offset差分，`exploration_process=zero_sum_offset`显式启用；缺省iid原路径保留。Replay仍存真实执行动作。累计探索平稳不是硬界限，不保证CUBIC真实窗口或学习后策略稳定。`window_binding.py`用已完成动作配对，排除终止后pending命令；`unacked*mss`/binding均是代理，丢包时不是严格pipe。

独立环境工具`environment_probe.py`与`tcp-buffer-probe.cc`对应三条件×current/send64/recv64/both64的12次纯CUBIC运行，读取同一原有20ms TCP_INFO调用、仅增加unacked/mss日志字段，没有RL写窗。`buffer_zero_probe.py`对应4次大缓冲零优化复核。两个Python工具显式读取`CANOPY_IPC_DIAGNOSTIC_ROOT`和注册JSON，包含有限截止、原参数回执、异常恢复及独立恢复watcher；会实际启动TCP/临时改内核缓冲，**本次发布不执行它们**。格式整理后的CUBIC源与执行SHA区别见`validation.json`。

构建纯CUBIC诊断程序（Linux，FRAME须为含完整code/runtime的独立scalar框架）：

```sh
g++ -std=c++11 -O2 -pthread -I "$FRAME/code/runtime/src" tools/scalar_ipc_diagnostic/tcp-buffer-probe.cc "$FRAME/code/runtime/src/flow.cc" -o "$CANOPY_IPC_DIAGNOSTIC_ROOT/native/tcp-buffer-probe"
```

实际运行须准备`environment-registration.json`（condition/mode_order/deadline_epoch，可选output_directory/port_base）或`buffer-zero-registration.json`（jobs/deadline_epoch）；58次有效TCP的计划、完整结果、3次端口失败、7项缺失补测与恢复记录都在[evidence/scalar-exploration-20261027](../../evidence/scalar-exploration-20261027)。原始sysctl均已恢复，没有采用64MiB为新全局默认值。每个CUBIC组合只有一次；初始化种子不是训练seed。原源码/大模型/replay/逐包仍在实验存储，未新增Release资产。

[实测及限制](../../docs/SCALAR_EXPLORATION_AND_BUFFER_CAP.md) · [发布来源](../../evidence/scalar-exploration-20261027/publication-provenance.json)

---

# 单动作IPC完整性与零优化初始化检查

这组工具对应2026-10-27实际完成的诊断，不是新增训练方案。结论为：**新IPC完整性检查通过，小幅初始化仍失败，长训NO-GO**。健康V31及原baseline模型未改。

## 先离线审阅，不需要云服务器

```sh
python3 tools/scalar_ipc_diagnostic/analyze.py
python3 tools/test_scalar_verified_ipc.py
python3 tools/verify_repository.py
```

`analyze.py`重新计算32次的两个门槛和159166个匹配帧，并核对汇总与64行full/trim10 CSV。它使用已有证据，不运行模型或TCP。原始注册条件为G12-R20/R200、G24-R20/R100、G48-R20/R200、G72-R20/R200、全部2BDP；初始化种子3/4，每次60秒。不是两个独立训练seed。

协议代码在`frameworks/canopy_single/verified_ipc.py`与`native/src/verified_scalar_ipc.h`。新native为独立文件`orca-server-verified-ipc.cc`，使用`tools/build_scalar_verified_ipc.py --framework /absolute/framework --output /absolute/new/native`构建；原native保留。`verified_ipc_v2=true`要求显式二进制及schema/SHA sidecar，两端必须同时启用，不允许自动回退。新锁和逐帧日志未证明CPU收益。

## 发布了哪些实际工具

- `probe_driver.py`：真实`distributed.py --role learner/actor`、RPC、FIFO、Gaussian std=.2、真实TCP，`diagnostic_no_optimizer=true`。保存前后完整96变量对照与两端帧echo，smoke通过后才可进入注册的formal。没有伪装冻结部署路径，未优化任何模型。
- `ipc_stress.py`与`ipc_stress_peer.cc`：真实Linux Python/C++ mmap双向5000次交换；不涉及TF或TCP，不称为网络smoke。
- `ipc_echo_audit.py`：旧日志只读匹配。`states.csv`日志中的percent属于**前一动作**；同state ID匹配cwnd，previous native action匹配percent。6714230配对中36个cwnd不同、0个percent不同，9个percent=24两端一致，7个末尾状态未配对单列。不能只凭这些记录唯一定位旧半帧来源。

旧日志审计的显式入口（只读取指定日志、写小JSON）：

```sh
python3 tools/scalar_ipc_diagnostic/ipc_echo_audit.py --scalar-root /absolute/old/frameworks --e2-logs /absolute/old/e2/logs --output-directory /absolute/new/audit-directory
```

`--scalar-root`下须有`orca_control/logs`与`orca_moving/logs`。历史完整原始日志未上传；缺少时不能伪造此次扫描或从汇总恢复所有字段。

## 真正运行时的目录与边界

需要原定制TCP内核、TF1.14/Python3.6、Mahimahi、sudo等完整环境。复用本仓库的独立scalar framework，不能直接把原三套模型当作本次随机诊断模型。

设`CANOPY_IPC_DIAGNOSTIC_ROOT`为全新实验根，其下应有：

```
frameworks/canopy_single/   # 完整scalar副本，含code、configs/traces
configs/                   # 每个run的配置输出
audit/                     # stop/worker stdout
native/orca-server-verified-ipc
native/orca-server-verified-ipc.ipc-schema.json
native/ipc-stress-peer
registration.json          # 新预算、新run ID、绝对路径均已重新注册
```

压力测试peer的Linux编译命令：

```sh
g++ -std=c++11 -O2 -pthread -I "$CANOPY_IPC_DIAGNOSTIC_ROOT/frameworks/canopy_single/native/src" tools/scalar_ipc_diagnostic/ipc_stress_peer.cc -o "$CANOPY_IPC_DIAGNOSTIC_ROOT/native/ipc-stress-peer"
python3 tools/scalar_ipc_diagnostic/ipc_stress.py
```

实际TCP的入口是`probe_driver.py --stage smoke`，之后才可能`--stage formal`。这两条会运行真实TCP，**本次发布不执行它们，不授权新实验**。证据中的`registration.json`是已结束历史计划，截止时间已过，路径已用`${EXPERIMENT_ROOT}`占位；不能直接复制重跑。新计划必须另定预算、run ID、完整路径及实验门槛；已有状态文件阻止重复stage。probe依赖外部有限监督，verified动作等待未做对端死亡故障注入，不当作独立部署watchdog认证。

## 判断与限制

原初始化16次中6次百万窗口、8次超时长门槛；小初始化16次中2次和7次。高吞吐不是初始化检查成功，完整吞吐/利用率、mean/P95/P99、丢包、低窗口均保留。时长由边界actual_dt近似加权，没有新增隐藏TCP采样。

中性输出仅在fresh单动作Orca设置actor输出层kernel/bias为U[-.003,.003]，随后原init_target复制；拒绝对载入checkpoint这样修改。没有改架构、奖励、窗口映射或MTP。没有加入窗口cap、E2b或后续训练。两个残余失败在无学习更新时仍由连续探索乘窗与native区间增长产生；下一步累计探索约束仅为待检验方向，尚未实施。

参见[最新研究交接](../../docs/RESEARCH_REVIEW_BRIEF.md)、[全部结果](../../evidence/scalar-ipc-init-20261027/formal-results.json)、[验证状态](../../evidence/scalar-ipc-init-20261027/validation.json)与[发布来源](../../evidence/scalar-ipc-init-20261027/publication-provenance.json)。工具只参数化路径和CLI，逆替换可还原运行源码（LF归一后），已注明哪些包装仅本机检查、没有新增TF/TCP验证。
