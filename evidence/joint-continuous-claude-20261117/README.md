# Claude 离线接手证据（2026-11-17）

先读[结果文档](../../docs/CLAUDE_OFFLINE_RESULT_20261117.md)。沙盒没有 TensorFlow，这里没有任何 TF 图执行结果；没有 TCP、SSH 或云端训练。

- `coverage-nearest-prefix.json`：真实 `replay.py`、正式配置、修复前的最近锚点分桶，`COVERAGE_UNREACHABLE`（160 ms 桶每 actor 57–69 行，门 128）。
- `coverage-strata-fixed.json`：同设定、修复后的等宽对数分层，`COVERAGE_REACHABLE`（每桶 200–205 行）。
- `capability-matrix-numpy.json`：NumPy 重建 critic 的能力矩阵，24 格，1 万步终点判定，门未改。原登记任务两种残差学习率均 FAIL；去掉水平偏移的诊断任务在残差 lr=1e-4 下 PASS。
- `critic-development-diagnostics-numpy.json`：定位抖动与水平偏移时的开发运行，含 2 万步和被否决的变体，不是预登记结果。

复现命令见各 JSON 的 `reproduce` 字段与工具文档字符串。
