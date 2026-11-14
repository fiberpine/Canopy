# Canopy

当前单动作基座：**seed3 step-6013196＋自然Cubic接管**。24个已训练静态5BDP条件×3次独立60秒测试全部通过，利用率99.75%、平均排队12.74ms、零丢包。

用户决定先采用当前模型并停止重训。只有一个完整训练seed；重新训练的稳定性、动态/浅缓冲及留出条件仍未验证。

- [当前基座：结果、执行契约、使用和限制](docs/CURRENT_BASELINE.md)
- [完整96次测试及用户停止seed7的记录](docs/BASELINE_HEALTH_CONTINUATION_20261113.md)
- [三地清理、独立交付与恢复记录](docs/BASELINE_RETIREMENT_20261113.md)
- [模型与native逐文件校验](CURRENT_BASELINE.json)
- [历史问题、失败和修复索引](docs/HISTORY_STATUS_20261113.md)
- [从健康基座迁移的双动作候选：初次离线实现](docs/JOINT_HEALTHY_SEED3_V1.md)（历史契约）
- [双动作首次上线：计量修复、容量与完整性能失败](docs/JOINT_HEALTHY_SEED3_ONLINE_20261113.md)
- [当前双动作续接：反馈时延诊断与G4初始化准入](docs/JOINT_HEALTHY_RECOVERY_20261114.md)（尚无完成的新联合训练，不训练独立固定MTP组）

## 当前入口

使用Ubuntu18.04、Python3.6/TF1.14和定制4.13.1-0521-learner内核。完整环境保留在原[Release](../../releases/tag/consolidated-20261018)，本次不改变环境或内核。

```sh
export ORCA_PYTHON=/home/ubuntu/venv/bin/python
$ORCA_PYTHON run.py doctor
$ORCA_PYTHON run.py evaluate --variant baseline --scenario 0
```

baseline默认评测使用当前权重、自然接管native及完整60秒包时钟；不优化、不探索。网络结束后核对接管≤10秒、动作映射、完整窗口和模型不变。

当前训练配方已显式固定24actor、名义6M、均衡回放和四项更新/BN修复；训练保留实际legacy native，部署使用自然接管native。复训需明确授权和五机配置，方法尚未通过第二个完整seed，不会自动启动。

## 目录

|位置|内容|
|---|---|
|`frameworks/baseline`|当前基座训练、推理、协议与原生源码|
|`frameworks/joint_healthy`、`joint_healthy.py`|当前健康窗口迁移的新双动作离线候选，82维/两动作，无性质约束|
|`templates/baseline-*.json`|实际执行配方的便携配置|
|`models/healthy-baseline-seed3`|当前策略及选定完整tensor|
|`docs`、`evidence`|结果、问题、修复及校验记录|
|`frameworks/mtp20`、`frameworks/mtpjoint`|保留的双动作研究实现，未因本次基座确认获得新性能结论|

旧H0和原51个模型文件保留，原始快照tag `consolidated-20261018`不变。旧执行树由归档tag恢复，当前入口不依赖旧独立Orca管线或v0文件。
