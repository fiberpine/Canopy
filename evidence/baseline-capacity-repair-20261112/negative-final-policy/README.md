# 未通过健康门的最终策略，用于复核

本目录是本轮step6010470的12张量冻结部署策略，policy SHA dd5fcd0f0d1f3f3326ccfc67b2d1f0770e738ff5999a71b35a19930eba1886c5。
不是新的健康基座，也不覆盖原H0。两个checkpoint文件共343558字节，SHA见上级policy-local-delivery.json。
使用与本轮冻结入口相同的70维/单动作moving_stats schema、20ms、无探索/无训练路径；评测配置的checkpoint字段指向本目录model前缀。
复核应采用本轮相同native源码/二进制及trace/5BDP条件，参照frozen-final-plan.json和formal-source-manifest.json。
只有部署策略，没有critic/target/优化器/replay/完整随机现场；不能用于声称无缝恢复。
