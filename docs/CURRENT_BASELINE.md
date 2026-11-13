# 当前暂定健康基座

当前采用seed3、step-6013196及自然Cubic接管协议。独立24个已训练静态5BDP条件×3次60秒，72/72完成且原健康门全部通过：条件等权trim10利用率99.7453%、mean queue12.7384ms、条件P95均值18.0694ms、零丢包。最差单次利用率97.4635%。另24次同启动H0参照也全部执行完成。身份、模型、实际接管≤10秒、动作映射、优化/探索0和五机清理审计通过。

用户决定先采用当前模型，停止进一步重训并清理旧版本。**这是一个训练seed的工程基座；可重复训练成功率、小窗口恢复、动态/浅缓冲与留出条件仍未验证。**每格三次评测不是三个训练seed。seed7在新指令到达前已运行约3分钟，随后安全停止；165396条经验、15310完整更新、无终点评测，不作第二次复训成功或算法失败。

|同自然接管、24条件trim10|当前模型（每格3次）|H0（每格1次）|
|---|---:|---:|
|条件等权利用率|99.7453%|96.0562%|
|mean queue|12.7384ms|12.2976ms|
|条件P95均值|18.0694ms|23.2500ms|
|丢包|0|0|

H0只有一次，不能估计其连接方差；两模型也不是各项全面支配。旧启动的历史H0均值不混作本表参照。[完整结果及过程](BASELINE_HEALTH_CONTINUATION_20261113.md)保留原失败与外部中断。

## 执行契约

70维状态、1个cwnd动作、固定20ms；保留原Orca奖励和Cubic逐ACK控制，没有QC、fallback或隐藏控制器。学习核心为`9bb1500eaf791bcb0e97724c63dcb725323dd5cd`：目标BN统计一致、两个critic独立Adam、target变量唯一配对、在线BN四个统计各一次更新。网络/损失/回放/噪声/U不再额外更改。

训练保留原legacy启动native，以便忠实保存实际成功配方；部署采用自然Cubic：保留`ssthresh<cwnd`退出条件和原遥测轮询，移除首次策略动作之前的1.1倍写窗，接管后执行不变。部署ELF SHA为`3def82b33ba9848313eaab8b0e3779b643d30a333b314171b5a451358b4b50f8`；训练ELF为`632bca97e981ddece04379508662ce76c482c2a78213ab2c084c26ed0e12fe1d`。用途不同，不能交换后二者仍称同一训练复现。

评测两种旧启动的40项失败记录保留。自然接管可能避开策略不善爬升的状态，不能据静态通过断言恢复弱点已被学习修好。

## 模型与代码

- [机器可读当前契约](../CURRENT_BASELINE.json)包含模型/native逐文件SHA及验收范围。
- `models/healthy-baseline-seed3/policy/model`是实际使用的12个部署张量；`training`含108变量的选定完整tensor、变量清单和step0图定义。
- 默认baseline评测模板已指向当前权重、自然native和本轮实际capture；原51个模型文件仍保留作历史证据，不再是默认模型。
- baseline训练模板取自实际seed3配方：24actor、6M名义样本、150k均衡回放、warmup12288且每actor512、U=.1、Gaussian sigma=.2、MSE、checkpoint300k、性能告警而非早停。

完整tensor不等于无缝恢复。交付未包含TCP/history/max_bw/actor探索现场的完整加载器；replay和历史大checkpoint留云端。不要把载权重重建连接称作完整resume。

## 使用

使用原Ubuntu18.04、Python3.6/TF1.14/Sonnet及定制4.13.1-0521-learner内核环境。全环境见仓库原`consolidated-20261018` Release；本次不更换内核或环境。

```sh
# 已有环境下，只读检查
export ORCA_PYTHON=/home/ubuntu/venv/bin/python
$ORCA_PYTHON run.py doctor

# 一次60秒冻结评测，不训练、不探索；scenario对应模板conditions顺序
$ORCA_PYTHON run.py evaluate --variant baseline --scenario 0
```

评测沿用完整60秒包时钟停止和15秒无反馈watchdog，另在流结束后核对自然接管、完整窗口和优化/探索0；后置核对不改变策略或网络行为。新封装仅做离线契约测试，不冒充新增在线性能试验。

```sh
# 仅供将来明确授权的从零复训；本轮不会再执行
$ORCA_PYTHON run.py train --variant baseline --cluster /path/to/private-cluster.json \
  --actors 24 --samples 6000000 --seed 3 --checkpoint-interval 300000 \
  --performance-guard-policy warn --scalar-update-contract --unique-online-bn-updates
```

集群配置含1learner＋4actor节点及已验证的SSH凭据路径。部署包须位于五机同一路径；不把地址/密钥写进公开仓库。复训方法仍待独立seed完整验证。

原源码、问题和修复记录可由历史tag恢复。清理清单只移除已证明无当前依赖的旧执行管线、重复资源或有完整可恢复归档的结束工作副本；原H0、51模型、负面证据及双动作创新保留。
