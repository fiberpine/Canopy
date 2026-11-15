# Healthy-seed3 joint candidate

独立的82维状态/cwnd＋MTP候选；一轮3600秒/actor完整训练和24×3冻结已结束，健康门FAIL，不替换健康基座。

从仓库根目录使用 `joint_healthy.py`。完整设计、验证、阶段门和限制见 [实施记录](../../docs/JOINT_HEALTHY_SEED3_V1.md)。

旧计量修复、负面结果和容量检查见 [在线记录](../../docs/JOINT_HEALTHY_SEED3_ONLINE_20261113.md)。当前G4与RTT状态schema见 [续接记录](../../docs/JOINT_HEALTHY_RECOVERY_20261114.md)，需要自己的完整网格回执；不能拿G0或失败G3代替性能准入。

最终结果、全部弱项和模型见[两小时完整训练](../../docs/JOINT_HEALTHY_TWOHOUR_20261115.md)；此前修复见[11月15日续接](../../docs/JOINT_HEALTHY_RESTART_20261115.md)。新配置显式将人工重连与Bellman终止mask分开：回放仍记录物理边界，禁止跨连接；在最后真实观测自举。旧checkpoint缺省保留历史语义，不能作为新语义checkpoint载入。共享内存按shmid审计，均衡采样用可增删数组池；新旧seed不承诺逐位相同的采样轨迹。

- 不导入旧V31管线，不改原baseline文件。
- 默认成本0，无Canopy性质奖励、QC、fallback或容量oracle。
- 不自动连接云主机，不支持无缝resume。
