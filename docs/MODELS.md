# 模型与训练先验清单

所有下列模型tensor/index都直接在Git中，不依赖旧服务器路径。完整便携Release内也包含同一字节副本。SHA和原来源详见manifest.json；每个模型的counters/variables等元数据按原样保存，不把历史路径误作当前依赖。

| 目录 | 状态/动作 | 身份 | tensor SHA256 |
|---|---|---|---|
| `models/baseline` | 70/1 | 冻结主模型step2412694 | `bc85696e43a066544865a2be274bc2f8cee4529a9d7ddbb4db5b199a9ed65e6f` |
| `models/mtp20` | 72/2 | V14主模型step277317 | `875bbbad8788706c88e42bfa28e9420fa73f7c5536e0056f53d985688496e17e` |
| `models/mtpjoint` | 149/2 | 健康V31主模型step157704 | `c268025287679988c80168758c0dbc0c2a404486dff04fe41ed868f378eebc36` |
| `models/prior-v13` | 72/2 | V14原训练必需的V13独训fixed80先验 | `11da296e85d7e757a9cec22c9b29ef30ecfad2575ab95de2fc2bca636dec059e` |
| `models/prior-v28` | 96/2 | V31原训练必需的V28区间计量先验 | `231a788a12c2402c61467a62aaa658083e01be44e951eb3563d6f1ad0d8b743e` |
| `models/fixed20` | 149/2 | 独立训练的固定20ms强对照，非同权重override | `f4d5ada93511366e59e038eb8474bac06daf95c8470e6cca83db3bc3d13ecedd` |
| `models/fixed40` | 149/2 | 独立训练的固定40ms强对照，非同权重override | `c42c9e3f14019662d9b33dae2dea74ca2f26959905cbfcc6702e147a620972c0` |
| `models/fixed80` | 149/2 | 独立训练的固定80ms强对照，非同权重override | `762f31d8e861c6a3d83e3b958b19b2c9db3346f653d348be5d61cff1c85899db` |
| `models/fixed120` | 149/2 | 独立训练的固定120ms强对照，非同权重override | `e624ddf7d0b1f0798659b0a13f0fbbc746027bd8f7ecf7dea9ed03743b312b2a` |
| `models/fixed160` | 149/2 | 独立训练的固定160ms强对照，非同权重override | `ecd12b23afccceb5920587a88e9cbde3147f754a1d0c961ea2bf9af0fa19e233` |
| `models/official-orca` | 70/1 | 官方发布冻结参照 | `0492c22b07509acaef85ceafd919a5315e79fb993a39745fe32ae993a656aaa7` |

旧单动作模型不得直接当双动作模型加载；V14固定20ms计量、V28/V31新区间计量不可静默混用。五固定周期对照均为149维同新接口版本，原训练覆盖与谱系不完全相同，不能隐瞒为多seed同预算实验。

正式V14/V31训练模板从各自先验建立新优化器/replay/计数/TCP。选定最终模型默认用于冻结评测。没有完整连续resume；也未提供历史replay作为默认训练依赖。
