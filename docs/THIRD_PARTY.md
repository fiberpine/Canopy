# 来源与第三方文件

本项目基于原Canopy/Orca实现，保留`frameworks/*/code`内原始源码与许可证/版权文件。IBP与Mahimahi源码及自带许可证保留在`runtime/vendor/interval-bound-propagation`、`runtime/mahimahi`。Python运行时发行依赖和定制Linux内核安装包在私有Release完整包内；各第三方原有许可证仍适用。

上游项目包括Canopy、Orca、Interval Bound Propagation、Mahimahi以及NYC Cellular Traces；具体trace来源URL、Git tree/blob及SHA记录在模板/证据配置中。不要删去原文件头版权或将第三方代码宣称为本项目原创。新增研究代码未在本次交付中另行选择开源许可证，仓库保持私有。

原训练commit、冻结commit、本仓库收敛提交是不同对象。原始来源绝对路径为审计信息，不是跨机器运行依赖。当前主实现通过`run.py`和包内模板工作；历史helper CLI与现场恢复脚本不能直接当迁移入口。
