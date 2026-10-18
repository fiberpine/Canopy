# Orca/Canopy 三套可迁移实现

主入口 `run.py`。三个主实现相互独立，原数学定义、窗口动作和网络结构保留。选型依据、所有负面路线及科学局限统一见 `EXPLORATION_HISTORY.md`；实际来源文件及SHA见 `manifest.json`。

| 名称 | 实现和模型 | 观测/执行语义 |
|---|---|---|
| baseline | 冻结seed3，step2412694，70维/单窗口动作 | 原始20ms工程基线 |
| mtp20 | V14 step277317，72维/双动作 | 固定20ms观测，动态20—160ms执行 |
| mtpjoint | 健康V31 step157704，149维/双动作 | MTP同时决定观测与执行；长周期没有隐藏20ms RL观测，CUBIC仍逐ACK工作 |

V14是修复时钟/IPC后有完整证据的保留版本，并非跨轮绝对性能冠军；四个浅队列条件周期变化不足。V31双动作健康但仍有浅队列、高RTT和尾延迟弱项。两者研究结论均为PARTIAL，整理交付通过不代表论文性能通过。

## 新服务器准备

验证平台：Ubuntu18.04 x86_64、Python3.6.9、TF1.14、Sonnet1.34及定制 `4.13.1-0521-learner` 内核。需要sudo、TUN和Mahimahi。包内含复制的Python环境、IBP源码、Mahimahi源码、定制内核image/headers安装包、全部native源码及已编译程序。其他操作系统/内核没有获得兼容性验证。

解压到任意**无空格的绝对路径**，所有参与机器使用相同绝对入口路径；允许不同实际目录通过同名软链接暴露。不得将历史源码目录作为隐式依赖。

```sh
cd /absolute/path/orca-release
bash setup.sh --system   # 新机安装系统依赖、构建Mahimahi、安装定制内核
# 按机器维护方式重启，选择4.13.1-0521-learner；脚本不会自动重启
bash setup.sh            # 重定位包内Python环境并检查运行条件
runtime/venv/bin/python build_native.py
runtime/venv/bin/python run.py doctor
```

已具备匹配内核及系统工具时，从第二次 `setup.sh` 开始。运行环境重定位仅修改包内文件，不修改系统Python/原venv。不包含SSH私钥；按自己的机器填写 `configs/cluster.example.json` 并保存为另一个文件，先配置learner到actor的SSH访问。安装步骤需要可访问Ubuntu软件源；本次未实际重装操作系统或重启服务器。

## 模型评测

```sh
runtime/venv/bin/python run.py evaluate --variant baseline --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant mtp20 --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant mtpjoint --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant fixed80 --scenario 0 --seconds 60
```

`templates/*-eval.json` 内 `conditions` 定义场景，`--scenario` 是从0开始的索引。还支持 `fixed20/fixed40/fixed120/fixed160/official-orca/cubic/reno`。五个固定周期模型均独立训练，非同权重强制周期；具体谱系及训练覆盖见manifest和原配置。不同实现的默认场景列表不同，正式比较须显式统一trace、RTT、队列、窗口、重复和启动语义。baseline模板保留原冻结执行链，不把它与另一个正常CUBIC启动参考混用。短20秒交付评测只验证模型可用，不作性能排名。

每次生成唯一run ID，可用 `--run-id` 显式指定；同名存在会拒绝。端口默认23000，learner25000，避开验证平台的系统临时端口区间；并行任务请分别指定 `--port/--learner-port`。评测关闭探索和优化，结果在对应 `frameworks/实现/logs/run_id/result.json`，包含控制记录和网络日志。

## 训练

先运行有限短测；短测显式降低warmup/校准门槛，必须出现真实critic和actor更新才通过。

```sh
runtime/venv/bin/python run.py smoke-training --variant mtpjoint --actors 1 --seconds 90
runtime/venv/bin/python run.py verify-training --variant mtpjoint
```

正式训练保留原warmup、均衡采样、真实经验更新额度和周期覆盖门槛。常规规模为1learner＋24actor，四台actor各6个；每个actor独立环境并按配置轮换，不是同时56个actor。需要自行明确时间/样本预算并保留评测余量。

```sh
runtime/venv/bin/python run.py train --variant baseline --actors 24 --samples 2400000 --cluster configs/my-cluster.json
runtime/venv/bin/python run.py train --variant mtp20 --actors 24 --seconds-per-actor 1800 --cluster configs/my-cluster.json
runtime/venv/bin/python run.py train --variant mtpjoint --actors 24 --seconds-per-actor 600 --cluster configs/my-cluster.json
```

不传cluster默认本机1actor；正式24actor使用四机配置。模板保留原训练环境集合，新增场景须在训练配置中明确注册trace及环境序列。新模型保存在对应framework的checkpoints下；初始模板、最终计数、来源和本轮日志分别保存。不要覆盖 `models/` 下交付模型。

baseline正式训练从随机初始化开始；V14按原路线加载包内V13 fixed80先验，V31按原路线加载V28区间吞吐先验并扩展反馈状态，重新建立优化器、replay、计数和TCP连接。这些不是从交付最终模型无缝续训，也不是整个网络随机初始化。必要先验在 `models/prior-v13`、`models/prior-v28`。checkpoint无完整TCP/history/max_bw/actor随机现场恢复器，禁止称为完整resume。

## 计量与证据

V31用观测边界累计bytes_acked差除实际dt计量吞吐，checkpoint含对应schema。旧ACK速率样本均值模型不可直接接新接口。V14保留自己的固定20ms计量接口，不能拿旧CP静默升级后继续引用旧结果。

`evidence/` 保留基线、V14、V31、多算法240次正式比较、ABC、V36因果诊断、Aurora参考等有用汇总与审计。完整/trim10、吞吐利用率、平均/P95/P99排队、丢包、调用、CPU应分场景读取。Aurora参考的协议/预训练覆盖限制和单训练seed限制见探索记录。一次完整critic迭代含两个optimizer操作，不能当环境步；少调用不等于同比例CPU收益。

本次交付检查及故障修正另存 `evidence/validation/`。新路径上的真实短训练验证更新链路，冻结评测验证保留模型；不用于新增算法收益结论。历史源代码/配置/负面诊断在五机各自的压缩历史档案中，退役模型、replay和日志的确切去留以cleanup清单为准；历史路径只作来源说明，不是交付运行依赖。
