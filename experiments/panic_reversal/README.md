# 恐慌机制与反转因子的最小实验

本目录是**未经真实数据校准的、可离线运行的机制实验**。目标是检验：同样出现下跌，信息下修、现金需求、信念传播和杠杆约束是否会通过不同路径，改变反转因子的未来条件收益。四类是可干预的机制假设，不是已经识别出的真实恐慌类别；现实事件可能混合多种机制。

## 1. 运行与交付

在仓库根目录运行。最低依赖为 Python 3.10+ 和 NumPy；不需要配置 `config/api.yaml`、嵌入模型、原数据库或付费模型 API。精确复现实测环境使用 Python 3.12、`requirements_repro.txt`；不同 NumPy/平台的数值结果可能略有差异。

```bash
python -m pip install -r experiments/panic_reversal/requirements_repro.txt
python -m unittest experiments.panic_reversal.test_framework -v
python -m experiments.panic_reversal.run --output experiments/panic_reversal/outputs_default
```

Python 3.10 可安装 `requirements.txt` 中的兼容依赖范围；其输出不承诺与锁定环境逐位一致。

默认 32 个代理、8 个资产、90 个交易步、5 个种子（42—46），4 种机制 × 6 种设定，共 120 次模拟。冲击发生在第 30 步。参数来自人为设定，不是 A 股估计值。一个步长代表一个简化交易日，不能直接解释为一个 Tick、5 分钟或某个真实交易日。

```bash
# 低开销试运行：24 次模拟
python -m experiments.panic_reversal.run --output experiments/panic_reversal/outputs_smoke_new --seeds 42 --logs none

# 检查反转是否依赖价值型需求的估值锚
python -m experiments.panic_reversal.run --output experiments/panic_reversal/outputs_no_anchor --value-anchor 0 --logs none

# 将流动性/保证金压力恢复放慢，避免“先规定恢复再发现反转”
python -m experiments.panic_reversal.run --output experiments/panic_reversal/outputs_slow_recovery --recovery-steps 1000 --logs none

# 修改严重程度、代理数量、持有期等
python -m experiments.panic_reversal.run --config experiments/panic_reversal/config_example.json --output experiments/panic_reversal/outputs_custom
```

输出目录必须是新目录，禁止覆盖旧实验。`--logs all` 记录全部代理路径；默认 `example` 只为各机制的首个种子基准组记录完整代理路径，其余组保留全部市场、资产、目标和审计记录。需研究某个对照的具体传播链时，重跑该组并指定 `--logs all`。

| 文件 | 用途 |
|---|---|
| `manifest.json` | 全部参数、种子、Python/NumPy 版本、源文件 SHA256、快照哈希、完成状态与证据等级 |
| `initial_states.json` | 完整日志组的初始持仓、现金、债务、行为参数、网络、冲击对象 |
| `agents_*.jsonl` | 每人每日的信念、私人信息增量、滞后社交增量、恐慌阈值穿越、现金、债务、持仓、委托、实际成交、被动卖出与未成交需求 |
| `market.csv` / `assets.csv` | 全矩阵逐日市场和资产观测，含带 `_latent` 标识的不可直接观测变量 |
| `conditional_targets.csv` | 信号时点、进入/退出时点、组合成员、阶段、分歧与未来条件收益 |
| `fixed_event_targets.csv` | 统一在冲击后第 2 步形成信号，含 0/10/30 bps 单边成本情形 |
| `conditional_summary.csv` | 机制 × 对照 × 阶段 × 分歧 × 持有期 × 成本的条件比较 |
| `paired_effects.csv` | 同种子、固定信号时点的基准减对照差异 |
| `accounting_audit.csv` | 每次运行的现金、股票和债务守恒与非负约束 |
| `RUN_REPORT.md` | 机器生成的简要运行记录；所有收益明确属于合成实验 |

`validation/` 中保存本次运行的参数、审计和紧凑结果，便于评审；原始大体量日志可用相同命令重建，不提交代理数据到版本库。

## 2. 文献依据与可证伪假设

| 待验证机制 | 文献给出的依据 | 实验干预 | 可证伪的预期；不是结果承诺 |
|---|---|---|---|
| 信息冲击 | Savor (2012, JFE)：伴随分析师信息的大幅价格变化更易漂移，无信息变化更易反转；信息代理并不等于因果识别 | 受冲击资产的潜在基本价值**永久下修**；不同信息源以不同概率收到带噪声的价值信号 | 若下跌主要来自合理重估，短期买跌可能继续受损；不能预先规定恢复至旧价格 |
| 流动性挤兑 | Campbell, Grossman & Wang (1993, QJE)：非信息交易与做市风险承受能力可影响量价和后续收益；Nagel (2012, RFS)：反转收益可反映时变的流动性供给补偿 | 部分账户需要外部提现，现金不足时被动卖出；同时降低经销商的报价深度，资金与库存仍有限 | 只有卖压耗尽且风险承接恢复时才可能反转；压力持续时反转可以失效 |
| 羊群传播 | Bikhchandani, Hirshleifer & Welch (1992, JPE)：观察他人行为可能产生信息瀑布 | 小部分代理收到悲观传言；不改变基本价值，信息/恐慌沿有向加权网络逐步传播 | 去除社交边后，传播速度或价格路径应改变；否则当前模型中的社交渠道没有增量。加权均值是简化传播规则，不是完整贝叶斯瀑布模型 |
| 强制去杠杆 | Brunnermeier & Pedersen (2009, RFS)：融资与市场流动性可形成螺旋；Bian et al. (2018, NBER 工作论文，后并入 2026 年 JF 论文)：中国 2015 年账户数据支持杠杆边界附近抛售及随后价格反转 | 收紧最高资产/权益比；先用现金还债，再卖资产还债；价格下跌可触发下一步追加卖出 | 反转取决于约束账户是否完成清算、承接资金是否足够；下跌初段不一定能买 |

这里的“挤兑”具体指投资组合现金流出/赎回压力，不是银行存款挤兑模型。强制卖出可以发生于并不恐慌的理性代理；主观恐慌率低不等于资金约束弱。不同机制允许通过内生损失、情绪和保证金反馈相互作用，但外生触发被分别控制。

Da, Engelberg & Gao (2015, RFS) 的 FEARS 指数提供了“用可观测搜索行为测量情绪”的参考；它不能证明 Tick 数据本身能测出恐慌，更不能为本框架的潜变量提供真实标签。

**文献原文/作者或机构来源**：

1. Savor, P. (2012). *Stock returns after major price shocks: The impact of information*. JFE, 106(3), 635–659. [作者稿](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/10/Stock-Returns-After-Major-Price-Shocks---May-2012---Final.pdf), [DOI](https://doi.org/10.1016/j.jfineco.2012.06.011).
2. Campbell, J. Y., Grossman, S. J., & Wang, J. (1993). *Trading Volume and Serial Correlation in Stock Returns*. QJE, 108(4), 905–939. [论文](https://people.stern.nyu.edu/lpederse/courses/LAP/papers/InventoryRisk/CampbellGrossmanWang.pdf).
3. Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). *A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades*. JPE, 100(5), 992–1026. [论文](https://snap.stanford.edu/class/cs224w-readings/bikhchandani92fads.pdf), [DOI](https://doi.org/10.1086/261849).
4. Brunnermeier, M. K., & Pedersen, L. H. (2009). *Market Liquidity and Funding Liquidity*. RFS, 22(6), 2201–2238. [NBER 工作论文与发表信息](https://www.nber.org/papers/w12939), [DOI](https://doi.org/10.1093/rfs/hhn098).
5. Bian, J., He, Z., Shue, K., & Zhou, H. (2018). *Leverage-Induced Fire Sales and Stock Market Crashes*. [NBER Working Paper 25040](https://www.nber.org/papers/w25040). 该工作与另一篇工作论文已并入 Bian, J., Da, Z., He, Z., Lou, D., Shue, K., & Zhou, H. (2026), *The Drivers and Implications of Retail Margin Trading*, JF, 81(4), 2217–2270，[期刊原文](https://doi.org/10.1111/jofi.70049)。本框架使用账户约束机制作为假设，不迁移历史收益估计。
6. Nagel, S. (2012). *Evaporating Liquidity*. RFS, 25(7), 2005–2039. [NBER 工作论文与发表信息](https://www.nber.org/papers/w17653), [DOI](https://doi.org/10.1093/rfs/hhs066).
7. Da, Z., Engelberg, J., & Gao, P. (2015). *The Sum of All FEARS: Investor Sentiment and Asset Prices*. RFS, 28(1), 1–32. [DOI](https://doi.org/10.1093/rfs/hhu072).

## 3. 代理与市场如何产生过程

每个代理持有资产数量矩阵、现金、债务、主观对数估值和恐慌强度，并具有不同的风险厌恶、损失厌恶、信息到达概率、信息更新权重、社交权重、趋势反应和杠杆上限。初始财富为对数正态分布；资产配置由 Dirichlet 分布生成；部分账户有初始债务。它们只是最小异质性设定。

1. 冲击作用于基本价值、提现义务、传言种子或杠杆上限。资金压力与情绪不是同一个变量。
2. 代理收到有噪声的信息。主观估值更新为“旧估值 + 私人信号增量 + **上一日**邻居估值增量”。代码不让当天更新后的代理先影响当天稍晚的代理；孤立节点无社交增量。更新式中的估值是主观量，精确基本价值仅作为模拟真值保存。
3. 恐慌由自身持续性、上一日亏损、权益回撤、已收到的负面消息、上一日邻居恐慌、提现/还债压力及传言种子共同驱动。日志保存各人的阈值穿越；`peer_associated_crossings_latent` 只表示穿越时有恐慌邻居，不等于已经识别的因果传播次数。
4. 代理按主观估值、趋势和恐慌调整目标持仓。价值锚参数只改变交易需求中的估值项；设为 0 后仍可有消息和情绪，但没有该估值交易项。`simple_rule` 只做上一日收益的逆向调整，并保留资产/现金/债务约束。
5. 提现义务和保证金约束优先触发卖出；受约束账户不能同时提交买单。成交后提现进入外部现金池，还款同时减少债务并进入债权人现金池。
6. 聚合买卖需求改变经销商报价；代理先互相成交，剩余订单受经销商现金和库存限制，按比例分配。未成交被动需求不会伪装为成交。该模型允许价格对未成交需求移动，不能解释为真实盘口逐笔撮合。

没有一条价格规则写着“恐慌后价格必须回到旧水平”。价值型需求、信息到达、报价冲击、资金约束松弛仍是研究者选择的模型结构，因此要运行 `value_anchor=0`、慢恢复、不同冲击幅度/影响系数/网络密度等敏感性分析。默认系数位于代码与配置中，不能通过挑选最赚钱参数来声称发现了反转规律。

现金审计包括：代理现金 + 经销商现金 + 提现池 + 债权人已收现金 + 费用池。股票审计包括：代理持仓 + 经销商库存。债务审计包括：剩余债务 + 已还本金。交易价格浮动会改变账面财富，不会被错误要求为财富总量恒定。

当前约束：无卖空、无新增借贷、单日单次批量成交、可有小数股、无完整订单簿、没有真实交易时段和各板块涨跌幅规则。每步买入下一步才能再卖，只有最简的跨日持仓时序；这不构成完整 A 股 T+1 制度。债务不足清偿时保留未偿债务和清算约束，不提供隐含救助。经销商深度与资金是两个不同限制。

## 4. 对照与识别

| 设定 | 改动 | 保留/解释 |
|---|---|---|
| `baseline` | 异质代理 + 加权社交 | 完整简化机制 |
| `no_social` | 信念与恐慌的社交权重均为 0 | 相同初始账户和随机冲击带；不删除资金约束 |
| `homogeneous` | 初始现金/债务/各资产持仓、信念和行为参数替换为截面均值 | 保持初始现金/债务/每只资产总持仓；网络与个体信息随机实现仍可不同，指属性同质而非路径完全相同 |
| `random_emotion` | 恐慌替换为预设分布的独立随机情绪 | 不使用未来情绪分布；基本信息与被动约束保留，随机情绪幅度未事后配平 |
| `simple_rule` | 禁用情绪交易及社交，使用固定上一日逆向规则 | 仍能提现、还债和被动清算，用于检验复杂信念机制的增量 |
| `no_shock` | 取消被指定的外生冲击 | 保留背景噪声与内生市场反馈；不是价格必须不动的市场 |

同种子对应同一组初始化和相互独立的固定随机带。延长实验不改变已运行前缀；实验不因分支多消耗几个随机数而失去配对。不同机制未通过事后修改严重程度强行配成相同跌幅，`shock_size` 也不是不同机制经济量纲完全相同的证据。

阶段为因果、向后看的规则标签：`onset`、`propagation`、`stress`、`exhaustion`、`recovery`、`calm`，使用当前及过去恐慌率/被动成交。`latent_stage` 是模拟内部解释标签，不能当成现实可交易信号。另有只用行情收益的粗略 `proxy_stage`（selloff/stabilizing/rebound/ordinary）。二者的区别必须保留。

`belief_disagreement_latent` 为各资产主观估值的代理间标准差再取平均；low/medium/high 阈值在配置中预先给定（0.02/0.06），不利用未来样本重分组。行情收益离散度不等于信念分歧。

按阶段/分歧分组时，干预会改变组内成员，存在处理后选择与构成差异；这些表只做条件描述。更干净的模拟干预比较是 `paired_effects.csv`：对所有设定统一在冲击后第 2 日形成信号，同种子配对计算差异。这仍是模型内的干预证据，不是现实市场因果效应。

## 5. 从模拟变量到日频/Tick 的观测

| 量 | 模拟定义 | A 股日频可对应什么 | Tick/额外资料可对应什么；识别缺口 |
|---|---|---|---|
| 卖压集中度 | `sell_concentration_assets`：各资产卖出成交金额份额的平方和 | 仅靠 OHLCV 无法确定发起买卖方向；可研究下跌成交额集中度，但定义不同 | 有发起方向的逐笔成交可算股票间卖压 HHI；时间桶卖压 HHI 衡量时段集中。`seller_hhi_latent` 是账户 HHI，普通 Tick 无法还原 |
| 成交量异常 | `volume_z20`：当前 log(1+成交额) 相对**此前**20步均值/标准差 | 成交额、换手率异常；按当时可得的股票池与流通股本 | 同分钟历史季节性标准化、尾盘成交占比、卖单突发；不能把全天成交额用于午盘信号 |
| 价格冲击 | `amihud`：绝对市场收益/成交额；另记净委托压力 | Amihud 类代理，不是结构性的因果冲击估计 | 同步中价变化/签名成交流；有盘口才能看深度、价差、撤单和补单。需要处理买卖价跳动与异步报价 |
| 市场广度 | 当步收益为正资产占比 | 当时成分中的上涨/下跌家数、跌停家数、跌破均线比例 | 同步时间桶内上涨占比、跌停封单/开板；复权、停牌与股票池变化要一致处理 |
| 波动率 | 20步等权市场收益标准差，不年化 | 日收益滚动波动；由高频计算的实现波动率可另加 | 过滤微观结构噪声后的 RV、下行波动、跳跃代理；单个模拟步不能直接对齐秒级 RV |
| 个股收益离散度 | 当步资产收益截面标准差 | 个股截面离散度，必要时控制行业和市场因子 | 同步收益离散度；不直接等同于投资者意见分歧 |
| 恐慌扩散速度 | `fear_new_share_latent`、`fear_recovered_share_latent`、两者差值和累计曾恐慌比例 | 日频价格不能直接给出恐慌传播率 | 需要带时间戳的论坛/新闻/搜索行为及传播边；行情只能提供待验证代理，不能从量价反推出“恐慌人数” |
| 被动卖出/分歧 | 账户现金义务、保证金缺口、强制委托和主观估值标准差 | 两融余额/融资偿还、基金赎回等粗粒度代理 | 账户杠杆、经纪商规则、基金现金流、投资者标识才更接近机制识别；普通 Tick 不足 |

为真实数据提供了无 pandas 的最小计算入口：

```bash
python -m experiments.panic_reversal.observables daily --input daily.csv --output experiments/panic_reversal/outputs_daily/features.csv
python -m experiments.panic_reversal.observables tick --input trades.csv --bucket-minutes 5 --output experiments/panic_reversal/outputs_tick/features.csv
```

- 日频 CSV 必需 `date,asset,close,adj_close,volume,amount`。价格为元、成交量为股、成交额为元，收益使用一致口径复权价，原价保留给交易约束。日期格式 `YYYY-MM-DD`。同一证券同日重复、非法价格或非有限值会拒绝。数据由调用方提供当时可得的有效交易样本和股票池；不能把停牌不交易伪装为正常零收益。
- Tick CSV 是**逐笔成交的增量**，必需 `timestamp,asset,price,volume`，可选 `aggressor` 为 -1（主动卖）、+1（主动买）、0/空（未知）。不能把累计成交量快照直接当逐笔成交，也不能从 Level-1 快照声称得到了撤单。时间戳必须带时区，例如 `2023-01-03T09:31:00+08:00`；桶以北京时间生成，桶结束后才可使用。
- 缺失方向保持未知，输出签名成交覆盖率；不按涨跌随意补成主动买卖。`sell_time_hhi_day_to_date` 只使用到当前桶为止的数据，起始时 HHI=1，需匹配分钟位置比较。Tick 收益使用各资产最后观察价，跨缺口/隔夜收益会混入，因此生产检验必须同步行情、约束时段、标记陈旧报价；当前入口不是完整高频微观结构清洗管线。
- 日频收益只比较相邻市场日期都出现的证券，缺失不填零，输出覆盖率。该入口不会读取未知的 ClickHouse 表结构，也不会猜测账户行为或生成四类恐慌的真标签。

## 6. 反转目标与成本口径

主要研究对象为**横截面短期反转**，不是宽基指数底部择时。信号 `f(j,t) = -[P(j,t)/P(j,t-L)-1]`，默认 L=5；每端取 floor(K/3) 个资产等权。行情与标签的时序为：

- 收盘 t 形成信号与状态；未来价格不得参与排序、阶段或分歧标签。
- t+1 收盘进入；t+1+h 收盘退出，h 为 1、5、20。最短持有期也跨一个交易步。
- 理论多空收益：输家未来收益均值减赢家未来收益均值，长端 +1、短端 -1，毛敞口为 2。
- A 股替代观察量：输家多头相对等权全市场的超额收益；同时输出多头、赢家和市场各自毛收益，避免把市场回升全部归为反转收益。
- 单边固定成本 c 下，多头超额代理减 2c，理论多空减 4c。默认 c 为 0、10、30 bps；代理撮合过程另有 2 bps 费用，仅影响合成市场账户行为，不再重复计入事件组合收益。

这些是**重叠事件组合的条件目标**，没有连续资金账户、跨期净额交易、冲击容量、借券可得性/费率、卖出税费细分、涨跌停队列、无法退出或权重漂移处理。固定 bps 不是现实成本校准。多空结果不是默认可在 A 股执行的策略；`long_only_excess_net_proxy` 也是成本代理，不能改名为已实现 Alpha。

条件均值先在每个种子内聚合，再跨种子平均。95% 区间对独立种子均值进行 bootstrap；少于3个种子不提供区间。不把重叠日期当独立样本，不报告伪精确的 t 值或年化夏普。五个种子只用于小规模可复现验证，正式模拟研究需更多种子、参数不确定性分析和独立验证场景。

## 7. 接入 TwinMarket 现有接口

核查基线：`de5f2446fcba0d0aba533a6adaede034160e29b4`。

`simulation.py::process_user_input()` 已提供 `user_profile`、`current_user_graph`、`df_stock`、`import_news` 和上一日 `belief`。`trader/trading_agent.py` 的 `belief` 为定性状态，并依赖 `BaseAgent`、模型配置和加载的新闻库；本目录不导入该模块，不改变现有 LLM 模拟默认行为。

`twinmarket_adapter.py::from_twinmarket()` 接受这些边界对象的只读快照：

```python
from experiments.panic_reversal.twinmarket_adapter import from_twinmarket

# profiles 来自同一 as_of 的 get_user_profile 结果；prices 使用对应时点的可得价格。
# beliefs: user_id -> 原始 belief；traits: user_id -> 显式行为参数与债务。
# numeric_belief_encoder 必须明确给出每个资产的 log(主观估值/当前价格)。
snapshot = from_twinmarket(
    profiles=profiles,
    user_graph=current_user_graph,
    beliefs=beliefs,
    belief_encoder=numeric_belief_encoder,
    traits=traits,
    prices=prices,
    as_of="2023-01-03T15:00:00+08:00",
)
```

支持实际 `current_cash` 和 `cur_positions[stock_code]['shares']` 字段，持仓也可为数据库 JSON 字符串。有向边的 source 影响 target；无向 NetworkX 图展开为双向；孤立点保持无社交影响。TwinMarket 的关系图可能来自交易相似性，不能直接解释成观察到的真实关注、信息接触或因果影响。

所有资产的数值信念、债务（即使假设为0也需明填）、风险厌恶、损失厌恶、社交/信息权重、信息到达概率、趋势权重、杠杆上限和信息源标签均要求显式提供；不从“我看空”推断恐慌率，不从原有持仓猜测杠杆。可先用双人盲标的粗等级编码定性信念，再检查一致性与敏感性；编码器应记录版本及人工标注来源。

快照保存为 JSON 后：

```bash
python -m experiments.panic_reversal.run --snapshot snapshot.json --output experiments/panic_reversal/outputs_snapshot
```

`orders_to_twinmarket()` 将显式数值策略的委托转换成原 `matching_engine.py` 已用的 `user_id,stock_code,direction,amount,target_price` 字段，向下取整到 100 股。它只转换委托，不能把请求量当成交；需在原系统中再次检查现金、可卖持仓、整手、时点、涨跌停等约束。原撮合器有需求侧复制/放大逻辑，因此本实验使用独立的守恒清算器；两种清算结果不能不加说明混用。

这是可测试的快照/委托边界，不是已经替换 `PersonalizedStockTrader.input_info()` 的完整兼容实现，也没有运行原 LLM 全链路。后续若替换原代理，需要同时映射论坛状态和五元组返回值，不能只给原构造器加一个参数就宣称离线集成完成。

## 8. 校准与真实市场检验需要什么

| 数据 | 可校准对象 | 仍无法仅凭这些数据识别的部分 |
|---|---|---|
| 点时股票池、复权日线、成交额/流通股本、停复牌、各板块价格限制及历史成分 | 收益/波动/广度/离散度/换手的分布与自相关、可交易覆盖 | 账户恐慌、传播边与强平身份 |
| 有发起方向的逐笔成交、带时间戳的多档盘口，最好逐笔委托和撤单 | 卖压时段集中、流动性消耗与补充、冲击/价差/执行容量 | 没有账户标识就难区分同一人的拆单与多人羊群 |
| 公告/新闻/分析师修正的**首次公开时间**、可量化预期差 | 信息冲击的方向、强度、信息到达延迟 | “没找到新闻”不等于没有信息；价值真值不可直接观测 |
| 历史账户持仓、资金流、融资余额/保证金与经纪商风控规则；或基金份额/赎回与持仓 | 现金/杠杆分布、现金缺口、强平阈值、持仓重叠 | 聚合两融余额不足以恢复账户强平机制，参数应做区间而非点识别 |
| 论坛帖/回复/转发、投资者关注边、搜索指数，含时间戳与采集可得性 | 情绪编码、传播延迟/权重、分歧与扩散代理 | 同步反应可能来自共同新闻，不能全部算为社交传染 |
| 历史费率、印花税、买卖价差、滑点、涨跌停可成交性与借券 | 策略执行、容量、净收益 | 日频收盘价上的漂亮收益不代表真实可成交 |

建议对现有 ClickHouse `bars_daily/calendar/corporate_actions/features_long/sec_master` 先做显式字段映射；这些表名本身不能保证有点时成分、资金账户、新闻或 Level-2 数据。宽基日频可以检验市场级回归关系，但无法单独计算横截面反转组合，至少还需要对应历史股票池的个股收益。Tick 尾盘成交量适合作为候选同步/领先变量，校准时必须按时间可得性处理。

建议的预注册流程：

1. **校准区间**：只用较早训练时段，以模拟矩方法/距离最小化匹配量价矩、尾部频率、卖压持续性、波动/收益自相关、传播滞后等；不要把测试期反转收益放入校准目标。当前交付提供观测计算与参数接口，尚未执行估计，也没有声称参数可被唯一识别。
2. **冻结机制与测量**：固定冲击假设、信念编码、阶段规则、分组阈值与成本网格。新闻、赎回和强平证据不足的事件保留“混合/未知”，不强制四分类；输出机制分数也应承认标签噪声。
3. **真实相关性**：在独立时段/事件中检验反转暴露与未来收益的交互，例如 `future_return = alpha + beta*f + gamma*state + delta*f*state + controls`。用点时行业/规模/流动性等控制；以事件和日期聚类或时间块 bootstrap 处理共振和重叠标签。效应先称条件相关性。
4. **样本外预测**：按时间滚动/扩展窗口训练；对未来 h 日标签 purge + embargo，所有变换/阈值仅在训练折估计；将简单反转、仅行情状态、加入新闻/资金/传播代理逐层比较，检验真正增量。不可用未来低点标“恐慌结束”。
5. **交易层验证**：收盘信号延迟执行、A 股 T+1/板块规则、停牌/涨跌停/冲击容量与历史成本完整纳入；多头策略与可实现基准比较，多空需单列借券约束。只有经过独立样本外交易检验，才讨论可交易 Alpha。

证据必须依次标明：**模拟内机制 → 真实市场相关性/识别证据 → 样本外净交易表现**。本次交付仅完成第一层的实验基础和第二层的数据接口，未完成真实校准、真实机制分类或可交易 Alpha 验证。
