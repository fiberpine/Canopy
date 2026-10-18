# 冻结baseline：原始单窗口动作

## 身份与实现

本实现是seed3、24个缩减环境的Orca/Canopy工程基线，70维状态、单窗口动作、原20ms执行链。原冻结commit为`ff5395a085d8940cc8c48c9c2b26262370176b9c`，选定checkpoint为step2412694。`manifest.json`另记录实际训练现场commit；训练入口、事后归档和本仓库新提交不是同一个Git快照，不能混称原训练commit。

源码在`frameworks/baseline`，模型在`models/baseline/model.*`。`distributed.py`负责TF learner/actor/FIFO与经验更新，`moving_stats_backend.py`负责部署moving BN推理，`code/canopy/rl-module/agent_v2.py`定义模型和critic/actor优化；`evaluate.py`与`repair_evaluate.py`负责native TCP闭环及冻结核验。原native与源码均包含，定制内核提供TCP控制接口。

训练集合为12/24/36/42/48/72Mbps ×基础RTT20/50/100/200ms，约5BDP队列。1learner＋24actor，四机各6个环境。正式模板从随机初始化开始；不会默认加载交付模型继续训练。

## 运行

```sh
runtime/venv/bin/python run.py evaluate --variant baseline --scenario 0 --seconds 60
runtime/venv/bin/python run.py verify-training --variant baseline
runtime/venv/bin/python run.py smoke-training --variant baseline --actors 1
runtime/venv/bin/python run.py train --variant baseline --actors 24 --samples 2400000 --cluster configs/my-cluster.json
```

先完成根README的系统/运行时准备。场景见`templates/baseline-eval.json`；正式训练见`templates/baseline-train.json`。模板保留原冻结执行链及启动语义，不能在未标注时与正常CUBIC启动的另一种参照混用。

## 已有性能证据

原24条件、每条件三次冻结评测、trim10条件等权：自训baseline利用率97.75%、平均排队32.32ms；官方Orca参照96.72%、30.60ms。官方部分参照复用此前轮次，因此这不是严格同轮因果比较，也不是完整Orca论文复现。

后续V36使用正常CUBIC启动的四条件参照，结果与原启动链分别报告，见[V31文档](MTPJOINT_V31.md)。不能因历史总均值高，就宣称baseline在所有动态/浅队列场景都良好。完整原始指标与审计位于`evidence/baseline`，其他匹配比较见`evidence/causal`和`evidence/multialgorithm`。

## 验证与恢复边界

打包验证包括原模型加载、106变量的实际框架更新/重载、迁移路径真实TCP训练及跨四机训练。短测不是新收敛结果。模型含保存的网络/target/BN及相关优化状态，但本交付没有完整TCP/history/max_bw/随机现场resume加载器，也未将历史replay装入默认训练。不要把加载模型建立新连接描述为无缝恢复。
