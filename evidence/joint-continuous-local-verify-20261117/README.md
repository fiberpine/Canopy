# 本机实际TF复核证据

先读[key-results.json](key-results.json)和[复核报告](../../docs/JOINT_CONTINUOUS_LOCAL_REVIEW_20261117.md)。原Claude TF实测失败、新公共价值校准对照、实际Model的原门通过均保留。离线关键门通过，不是TCP模型或论文收益。

- claude-*: 未修改Claude 2d2de61上的实际TF结果，能力门仍FAIL。
- matched-high-period-lr-*: 同初始化/数据/10000预算的1e-3残差学习率对照，FAIL。
- common-first-1000: 独立驱动顺序诊断，1000仅主干+9000共同更新，PASS；不是从最好点挑结果。
- staged-production-capability: 实际Model.critic_step生产配方W1000的完整原门，PASS；6个critic任务加已知Q actor任务。
- final-staged-graph: 原与新配方网络初值逐位相同、阶段冻结与开放、迁移/梯度/保存恢复。
- staged-pipeline: 合成小fixture W1，真正运行learner/回放/阶段门/账本和四Adam时钟；阈值不是生产建议。
- oracle-head-capacity: 使用已知标签的直接线性拟合，仅为此小任务的表达能力诊断，不替代原门。
- final-numpy-tf-parity: 相同TF初值的前向/梯度/一步Adam；显式关闭公共校准以匹配NumPy同时更新实现。
- expected-fail-exit-control: 零更新旧critic负控，故意FAIL并验证退出1，不是新训练失败。
- figures: JSON轨迹对应PNG/SVG/PDF/CSV。全部stdout/stderr及合成checkpoint只在本机H保留；没有新云端大文件传输。
