# V31：MTP同时控制观测与执行

## 身份与选择

选定健康V31 step157704，149维状态、窗口/MTP两个动作。原现场来源和训练commit见`manifest.json`与`environment/mtpjoint-original-training-config.json`；原路径已退役，仅作来源说明。后续V35失败候选不替换此模型。V34/V36复评支持其动作健康，但性能仍未全面优于固定周期，结论PARTIAL。

## 真共周期机制

策略输出窗口动作和20—160ms MTP。native在真实调度边界读取一次观测并执行动作，下一边界才再次观测、推理和执行；**长MTP期间没有隐藏20ms RL观测**。CUBIC仍按ACK更新TCP窗口。没有early wake、QC、fallback或运行时teacher。窗口动作仍为原倍率定义，不因延长MTP而反复施加同一倍率。

MTP映射仍为`20 * 8**((a_mtp+1)/2)`ms，时间头使用该版本配置的softsign。20ms在SMDP折扣和物理经验账本中是计量单位，不是暗中运行的观测时钟。

## 统计变量适配与状态

旧吞吐是ACK触发的delivery-rate样本均值；它不等于区间确认字节数/真实时间，且偏差受MTP影响。V28修复已纳入V31：在相邻真实观测边界取累计`TCP_INFO.bytes_acked`差，再除以实际dt。没有为计量增加中间20ms观测。checkpoint含`interval_rate_schema=1`；禁止旧CP直接接新接口或新CP接旧接口。

149维状态由保留的96维前缀及53维反馈组成。96维中含时间统计适配；53维为10帧已执行窗口动作、动作有效性、RTT是否有新样本、RTT年龄上下界各一列（50维），加3个实际TCP快照比例。执行动作采用native整数映射后的值；RTT年龄只称边界统计推导的区间界限，不冒充精确采样时间。没有未来trace/实时容量oracle输入。

尚存限制：旧前缀仍含MTP敏感字段；观察峰值归一化会受策略影响；边界奖励仍是非线性效用的区间近似，不能精确恢复未观察的拥塞尖峰。这些问题不能说已全部解决，更不能把吞吐修复称所有历史坍塌的唯一原因。

## 学习和代码位置

从包内V28区间计量96维先验`models/prior-v28`迁移为149维，新增输入权重零扩展，保留窗口/critic/BN/target先验，重新建立Adam、replay、计数和TCP；不是随机从零或最终V31无缝续训。正式模板保留critic校准、周期覆盖、真实物理经验更新额度和双动作学习逻辑。

源码在`frameworks/mtpjoint`：`feedback_context_v31.py`定义反馈/迁移；`joint_mtp.py`定义动作和SMDP；`distributed.py`处理learner/actor及物理账本；`connection_reset.py`轮换独立环境；`evaluate.py`驱动冻结闭环；`code/runtime/src`和`native-v28`保留匹配的调度边界/累计字节计量native源码与程序。native目录沿用谱系名称，不能据名字误认接口仍是旧版本。

## 匹配性能证据与弱项

下表是V36四个预选诊断场景的trim10重复结果：baseline和官方Orca采用正常CUBIC启动参照；不是原baseline历史启动均值，也不是所有场景总体排名。利用率% / 平均排队ms：

| 场景 | 自训baseline参照 | 官方Orca参照 | 健康V31 |
|---|---:|---:|---:|
| 浅队列公交 | 90.40 / 13.05 | 87.81 / 11.52 | 76.86 / 7.28 |
| 公交RTT100ms | 91.91 / 67.97 | 95.93 / 101.18 | 87.04 / 33.14 |
| 带宽突降 | 86.93 / 29.19 | 87.21 / 44.14 | 98.83 / 31.49 |
| 混合时间尺度 | 98.49 / 8.40 | 92.36 / 3.58 | 98.85 / 9.68 |

V31浅队列和高RTT吞吐弱；突降吞吐改善，但尾延迟仍高；混合条件没有全面延迟优势。V36八次学习MTP评测动作健康8/8，原生MTP/区间字节审计通过，说明控制链路在工作，不能据此证明MTP选择最优。全部完整/trim10、P95/P99/丢包/calls/CPU见`evidence/causal`。动作变化只是必要条件，不能代替实际收益。

同权重强制20ms在两个高RTT条件曾降至39.12%和5.87%利用率，后者窗口约8但新鲜RTT比例约61%，支持窗口和动作频率耦合，并非“全局MTP失效”或“简单缩短周期就好”。突降强制80ms可约98%利用率/18.69ms均值，P99约310.5ms对学习周期约404.33ms，但CPU约多8.8%；混合条件学习周期平均排队较强制80低约43%，CPU约多8%。这是同权重诊断，不能冒充独立训练固定基线。

完整多算法正式评测位于`evidence/multialgorithm`：12模型/算法×10条件×2重复，共240次。包含五个独立训练20/40/80/120/160ms对照，不能只比fixed80；所有场景属于现有开发/训练覆盖体系，不宣称独立盲测泛化。Aurora参考的协议与预训练覆盖局限另见探索记录，不应包装为匹配重训纯RL强对照。

当前仍缺多独立训练seed、稳定盲测和全场景第一梯队证据；CPU收益必须用匹配实际测量说明。尚不适合宣称论文级成功。

## 运行

```sh
runtime/venv/bin/python run.py evaluate --variant mtpjoint --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant fixed20 --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant fixed80 --scenario 0 --seconds 60
runtime/venv/bin/python run.py evaluate --variant fixed160 --scenario 0 --seconds 60
runtime/venv/bin/python run.py verify-training --variant mtpjoint
runtime/venv/bin/python run.py smoke-training --variant mtpjoint --actors 1 --seconds 90
runtime/venv/bin/python run.py train --variant mtpjoint --actors 24 --seconds-per-actor 600 --cluster configs/my-cluster.json
```

选定模型在`models/mtpjoint/model.*`。场景见`templates/mtpjoint-eval.json`；各对照有独立模板。场景索引必须按各模板核对，不能把相同索引误当相同trace。
