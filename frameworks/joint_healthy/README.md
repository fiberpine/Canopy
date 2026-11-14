# Healthy-seed3 joint candidate

独立的82维状态/cwnd＋MTP候选；本轮上线准入进行中，不替换健康基座。

从仓库根目录使用 `joint_healthy.py`。完整设计、验证、阶段门和限制见 [实施记录](../../docs/JOINT_HEALTHY_SEED3_V1.md)。

旧计量修复、负面结果和容量检查见 [在线记录](../../docs/JOINT_HEALTHY_SEED3_ONLINE_20261113.md)。当前G4与RTT状态schema见 [续接记录](../../docs/JOINT_HEALTHY_RECOVERY_20261114.md)，需要自己的完整网格回执；不能拿G0或失败G3代替性能准入。

- 不导入旧V31管线，不改原baseline文件。
- 默认成本0，无Canopy性质奖励、QC、fallback或容量oracle。
- 不自动连接云主机，不支持无缝resume。
