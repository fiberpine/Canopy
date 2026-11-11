# 固定周期基座重连候选

本分支从 `e1730d2` 独立准备。新增的训练改动只有定期重建连接，保留原 24 条件、150k 环形回放、70 维状态、单窗口动作、raw 奖励、MSE、BN 路径、探索和更新配方。算法是否改善需要真实训练与独立冻结确认。

## 入口

正常 `run.py train --variant baseline` 增加了以下参数。

| 参数 | 用途 |
| --- | --- |
| `--episode-seconds 120` | 每条连接限定 120 秒，随后新建 native、TCP、共享内存和环境包装实例 |
| `--seed 3` | 设置该次随机初始化、采样和探索的 seed |
| `--deadline-epoch ...` | 可选绝对截止时间，缩短已有运行上限，至少须余十分钟 |
| `--replay-probe` | 容量准入专用，持续入库时保存一份满回放快照，结束时要求各 actor 至少完成三次重连 |

连接之间保留 live TF session、模型、探索 RNG、生产者队列、回放和更新账本。没有重新播种。每次调用已有 evaluator 重建其 TCP/native/环境状态，最后一个已收到真实后继的完整转移仍按原 gamma 自举。尚未收到后继的动作没有转移。跨连接的空档和保存时间不产生新经验。

原来的单连接调用不传 `--episode-seconds`，继续走原入口。评测和训练日志仍分开。

## 部署后的容量准入

在登记的新预算内执行，先填写集群配置并核对实际内核与运行时。以下命令会启动真实 TCP 和优化，部署前需完成当前轮次登记。

```sh
runtime/venv/bin/python run.py train --variant baseline \
  --run-id reconnect-admission-SEPARATE_ID \
  --actors 24 --samples 225000 --episode-seconds 20 --replay-probe \
  --seed 3 --cluster configs/my-cluster.json --deadline-epoch REGISTERED_EPOCH
```

它使用生产回放容量和真实更新路径。learner 的 `replay-probe.json` 记录满回放复制、写盘时间、FIFO 及入库变化。该 NPZ 是保存压力测试文件。每个 actor 要有至少四条已完成连接，各组生成、确认、入库对齐，更新额度排空，动作映射无错。容量准入结束还须做归属进程和 IPC 检查。

正式候选另起随机初始化、全新 replay 和计数。首 seed 的样本预算可沿用 2.4M，连接周期暂定 120 秒。是否采用该周期，要先核对准入时各 actor 的真实回放保留跨度和启动覆盖。短期策略波动作为记录，数值、协议、进程故障和预算截止作为停止边界。

## 回放覆盖

每个 actor 的父目录保存 `transition-origins.jsonl`，包含连续的 actor 序号、连接号、连接内步数、真实时长和启动标签。learner 回放增加对应 `actor_sequences`。FIFO 的原训练数据结构、数学目标及采样权重保持原配方。

在数据所在地，用所需 actor 的来源文件连接回放索引。

```sh
python tools/audit_episode_replay.py --replay PATH/replay.npz \
  --origin PATH/actor0/transition-origins.jsonl \
  --origin PATH/actor1/transition-origins.jsonl --output coverage.json
```

实际审计要包含全部 24 个 actor 的必要来源。工具拒绝缺失、重复和未初始化索引，输出各 actor 留存连接数、启动样本数和物理暴露。`startup` 指该连接前十秒，`steady` 标签不证明没有恢复事件。来源文件和大 replay 优先留云端，汇总可以单独回传。

逐连接保存 `states.csv`、`control.csv`、`down.log` 和 ownership。监控追踪当前连接，控制周期统计在每条连接内分别计算，再汇总。重连间隙不会算成控制周期。父目录的 `episodes.json` 和 `result.json` 表示整个 actor 的结果。

## 当前验证边界

本机测试直接执行现有 scalar actor 函数，替换 TF session 与 TCP evaluator，检查多次建连、探索流连续、真实后继归属、排空、停止及早退失败。另检查环形回放覆盖后的索引和快照、逐连接统计及来源审计。

11 月 11 日已部署真实 TF1 与 24 actor 短准入。第一次在第二条连接写计时日志时失败，计时包装器提前关闭了文件。修复后的第二次在重用 native 端口时发生 bind 冲突，两次均未打开覆盖热身门，记录到的 critic 更新为零，满回放保存尚未执行。两次失败和实际运行源码均保留。

当前新增了连接间保持计时文件打开、按连接序号分配不同端口，以及核对 PID 创建时间后清理权限受限子进程。计时修复已用于第二次短准入，其余两项仅完成离线检查。24 路持续优化、反复重连和满回放保存仍需重新准入。冷启动、优化器和回放的恢复能力沿用原限制，载入权重重建 TCP 不能称无缝恢复。

## 双动作的下一接口

先取得按新登记确认的 PASS 基座。后续连续窗口头和时间头仍用原 `4^a` 窗口倍率，周期对数映射 20–160 ms，采用真实共周期、SMDP 积分和实际 dt 折扣。复用已有 `joint_mtp.Interval`、周期映射、协议和独立双 critic 修复，在新目标下校准 critic，再用有周期覆盖的数据联合更新两个头。

时间头使用有限内部点初始化，30 ms 是待冻结的默认值。扩展状态输入的初始权重置零，先核查原窗口参数和保存输入上的前向迁移误差。真实 TCP 采用配对和预登记容差验证。初始化输出一致并不证明闭环性能相同。

调用成本系数及强固定周期对照在看到新收益前登记。对照匹配真实暴露和完整更新数，分别报告吞吐、mean/P95/P99、丢包、恢复、调用及完整 CPU。当前分支没有新健康 checkpoint，因此在线双动作校准与训练仍有基座验收这个前置条件。
