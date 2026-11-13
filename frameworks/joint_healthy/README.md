# Healthy-seed3 joint candidate

独立的82维状态/cwnd＋MTP候选；只完成离线验证，不替换健康基座。

从仓库根目录使用 `joint_healthy.py`。完整设计、验证、阶段门和限制见 [实施记录](../../docs/JOINT_HEALTHY_SEED3_V1.md)。

- 不导入旧V31管线，不改原baseline文件。
- 默认成本0，无Canopy性质奖励、QC、fallback或容量oracle。
- 不自动连接云主机，不支持无缝resume。
