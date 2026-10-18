# 交付验证范围

原完整便携包SHA256：`7cb05dbbf33ca91e7ad237585df858063fc6fa3b5e1bf57115f6c59f6ecff17b`，389839092 bytes，12812个文件逐字节归档校验。此SHA对应Release原始完整包；Git根README及新增文档有自己的`repository-files.json`，不要拿原包清单错误地校验新增Git文件。

三套native从包内源码重编译通过。离线验证使用真实TF1框架、选定模型、两个critic优化操作、actor和target更新，权重/BN/优化状态有限，临时完整checkpoint重载一致。变量数baseline106、V14 107、V31 113；不是假环境推演当在线通过。

| 实现 | 迁移路径单actor短训练：critic/actor | learner＋四机各1actor：critic/actor |
|---|---:|---:|
| baseline | 195 / 195 | 924 / 924 |
| V14 | 177 / 176 | 1254 / 1253 |
| V31 | 177 / 176 | 1250 / 1249 |

完整critic迭代各含两个critic optimizer操作。critic比actor多一次来自短测critic校准，不是丢失actor更新。真实生成/入库计数一致，最终训练checkpoint数值有限。

三主模型冻结评测及fixed20/40/80/120/160、官方Orca、CUBIC、Reno八入口短评测通过。完整包在独立目录解压，bootstrap、doctor、pip路径重定位与三主模型顺序评测通过；旧现场退役后再次运行也通过，冻结模型SHA不变。原冻结baseline受保护，五机最终无实验进程/用户IPC残留。

保留失败：初始core传输遗漏framework本地依赖；smoke校准0被拒绝；本机learner/actor索引0混淆；43000端口与临时端口区间重叠；45秒MTP仅504/505条低于512 warmup，正确判为零更新失败。这些仅在新打包副本修复，不隐瞒失败、不改原算法数学。详见`evidence/validation/packaging-corrections.json`。

初次独立解包同机并行冻结，两MTP出现首次反馈超时，尚未定位原因；同机顺序评测前后均通过。当前不认证同机并行冻结，采用顺序流程。失败日志与最终后置验证回执作为Release配套验证附件保留。

验证仅覆盖Ubuntu18.04 x86_64/定制内核、1actor与4actor短更新链路。没有重装操作系统、重启验收其他OS、重新证明24actor容量、最大集群、长训收敛、多seed或性能领先。原24actor科学证据和打包短测必须分开。

`python3 tools/verify_repository.py`针对未修改的新checkout；重编译、bootstrap或新实验生成的元数据可能改变工作树，届时请用Git diff核对，不冒称仍是原始字节快照。
