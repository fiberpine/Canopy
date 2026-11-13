# 当前基座归档与三地清理（2026-11-13）

按用户决定，先将 seed3 step-6013196＋自然 CUBIC 接管固定为暂定健康工程基座，停止重训并退役无当前依赖的旧代码。独立测试96/96完成：当前24条件各3次、H0同启动各1次。当前24格全通过，条件等权trim10利用率99.7453%、mean queue12.7384ms、条件P95均值18.0694ms、零丢包，最差单次利用率97.4635%。[结果、契约和适用范围](CURRENT_BASELINE.md)。

只有一个完整训练seed。seed7在用户停止指令到达前已启动，随后安全排空停止于165396条经验、15310完整更新；未作终点评测，不计第二次成功，也不判训练失败。清理未增加训练或TCP实验。

## 当前交付与恢复入口

- 当前发布tag：`baseline-healthy-seed3-natural-20261113`；删除前快照：`baseline-before-retirement-20261113`。原`consolidated-20261018`及51个原始模型文件不变。
- 本地独立包：`D:/Canopy/baselines/canopy-healthy-seed3-natural-20261113`及同名`.tar.gz`，99个受检文件，压缩包4837437字节，SHA256 `2827bf5beb0d791546e414e1912ccfdd5876e5adf67ccd8ac7b8ff2f989675db`。
- 五台云主机均有独立目录`/home/ubuntu/canopy-healthy-seed3-natural-20261113`及同名压缩包。源码、策略、完整选定tensor、两种native、实际trace和校验清单已逐项一致；learner上精确native重建及现有环境检查通过。
- 包内源码版本为`de0ffb05718a865ce42be427880838316b196430`，学习核心仍等于实际运行的`9bb1500eaf791bcb0e97724c63dcb725323dd5cd`。最终清理提交只退役旧资源和增加交付记录，不冒充新的训练版本。
- 完整TF1环境/定制内核仍引用原Release和已有安装，不重复下载或覆盖。包内完整tensor不代表具备TCP/history/max_bw/actor随机现场的无缝resume。

## 已执行清理

|位置|处理|校验与保留|
|---|---|---|
|当前Git工作树及GitHub主线|删除120个旧文件，合计182974334逻辑字节|87个旧独立Orca/baseline/v0文件、33个Mahimahi生成物；先逐文件ZIP校验并保留删除前tag|
|本地历史工作副本|退役13个干净工作树，合计7856499924逻辑字节|先建立126041724字节全引用Git bundle，独立裸克隆及fsck通过；每个HEAD/tree均可恢复|
|云端五机|移除2482项旧源码副本，共7520907逻辑字节|每目录先压缩、成员SHA校验，归档共2007178字节留原目录；正式实验数据保留|
|云端结束容量测试日志|42文件无损gzip，减少1224715861载荷字节|原文件SHA与解压SHA相同后才移除未压缩副本；正式训练/评测日志未压缩清理|

逻辑字节不等于实际磁盘回收；Git历史未重写，删除文件仍在历史中，因此不宣称Git仓库历史体积减少。

本地恢复文件位于`D:/Canopy/archive/retired-baselines-20261113`：`shared-baseline-history.bundle`、`main-obsolete-baseline-sources.zip`、`mahimahi-generated-artifacts.zip`及对应校验记录。云端旧目录的`RETIRED_CODE.md`指向独立当前包，`retired-source-20261113.tar.gz`保留旧源码。归档可能含私有调度参数，只留云端，不公开原包。

GitHub旧`cleanup-20261111`和`research/baseline-capacity-repair-20261112`分支归档为同名`archive/` tag后退役。PR1由本次已验证清理取代，不合并旧冲突补丁。当前main与研究分支指向同一版本。分支操作最终状态以交付收据为准。

## 保留的材料及边界

原H0完整本地交付、51模型、实际成功训练目录、完整checkpoint/replay/正式原始日志、负面实验和问题修复记录均保留。四份上游Orca审计源码、原始评测配置及版权放在`evidence/upstream-orca-source`，只读审计不再依赖旧执行管线。`frameworks/canopy_single`仍供历史分析测试使用；双动作实现及探索记录未被判为无用，保持原状。

本地`github-delivery-20261018/Canopy`和`innovations/scalar-safe-learning-20261028`有未提交修改，保留用户工作。没有为追求目录为空或Git clean删除未知产物、密钥、环境或内核。

Mahimahi只删除可由`setup.sh --system`重新生成的构建产物，源码和云端已安装程序保留。新系统全量重装路径本轮未实测；现有环境的native精确重建与doctor检查已通过。

五机最终核查：当前99文件、51原模型、受保护core/sysctl均一致，本轮进程/登记IPC/端口残留为0。最终小报告交付只做文件传输，不追加网络实验。

## 复核证据

[清理汇总](../evidence/baseline-retirement-20261113/summary.json)、[本地工作树及bundle](../evidence/baseline-retirement-20261113/local-checkout-retirement.json)、[云端退役清单](../evidence/baseline-retirement-20261113/cloud-code-retirement-applied.json)、[五机终检](../evidence/baseline-retirement-20261113/closure-after-retirement.json)、[独立包校验](../evidence/baseline-retirement-20261113/current-capsule-delivery.json)。所有小证据的逐文件SHA在同目录`file-integrity.json`。
