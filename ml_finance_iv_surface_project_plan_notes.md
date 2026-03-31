# ML × Finance 项目执行笔记

## 项目名称（当前工作名）
**Neural / Latent Inference for Sparse Implied-Volatility Surfaces**

---

## 0. 这份笔记是干什么的
这不是给面试、给教授、或者给新 chat 的短摘要。

这是一份**我自己执行项目时用的项目计划笔记**，目的是把整个项目的：
- 最终目标（end goal）
- 问题定义（problem framing）
- 输入输出（inputs / outputs）
- phase 划分（phase roadmap）
- Phase 1 的具体执行计划（detailed Phase 1 plan）
- baseline、数据、评测、产出物（deliverables）

全部写清楚，方便我自己推进项目、记录思路、后续补实验，以及随时恢复上下文。

---

## 1. 项目一句话定义
这个项目要解决的问题是：

> **如何从稀疏、带噪、非规则的真实期权市场报价（option quotes）中，恢复一张对 pricing / hedging / risk management 真正有用的、稳定的、尽量 no-arbitrage 的 implied-volatility surface（隐含波动率曲面）。**

这不是普通插值（interpolation），也不是为了重构而重构（reconstruction for its own sake）。

更准确地说，这个项目是：
- 一个 **latent inverse problem（潜变量逆问题）**
- 一个 **structured inference problem（结构化推断问题）**
- 一个 **market representation recovery problem（市场表示恢复问题）**

长期目标是把它推进成一个：
- **latent-variable**
- **uncertainty-aware**
- **structure-constrained / arbitrage-aware**
- **energy-based**

的 ML × Finance 项目。

---

## 2. 这个项目为什么重要
现实市场不会直接给我一张干净、完整、平滑、无套利的 implied-volatility surface。

现实里拿到的是：
- 不同到期（maturity）上 coverage 不一样
- 不同行权价 / moneyness 上点分布不均匀
- 深虚值 / 深实值、临近到期区域的 quote 往往更脏
- bid/ask、staleness、liquidity 会污染很多点
- observation pattern 本身是稀疏且不规则的

但是交易、做市、定价、风险管理、relative value 分析真正需要的是：
- 一张完整的 surface
- 平滑、稳定、结构合理
- 跨 strike / maturity 一致
- 尽量 no-arbitrage
- 可作为 downstream pricing / hedging / risk 的输入

所以这个问题本身是一个真实的 industrial pain point（行业痛点），不是我为了做 deep learning 硬捏出来的 toy problem。

---

## 3. 这个项目不是什么
这个项目**不是**：
- 普通 regression
- 普通 interpolation / smoothing
- 只画一张更好看的 3D curve
- 只做 forecasting
- 只做数据清洗 + benchmark 然后 claim ML
- 只靠现成模型拼装

我想要的是一个真正体现 ML / deep learning / latent inference / structured modeling 的项目。

---

## 4. 项目的最终目标（End Goal）
最终要做出来的，不是一个“分数高一点的拟合器”，而是一个**可用于决策的市场表示层（decision-grade market representation layer）**。

### 最终系统应该能做什么
给定不完整的真实市场期权报价，系统应当能够：
1. 识别和吸收稀疏、带噪、非规则的 observations
2. 推断一个潜在的 surface state（latent surface state）
3. 输出一张：
   - 平滑（smooth）
   - 稳定（stable）
   - 尽量无套利（arbitrage-aware / preferably arbitrage-free）
   - 带不确定性估计（uncertainty-aware）
   的 implied-volatility surface
4. 这张 surface 对 downstream 有用：
   - pricing
   - Greeks / hedging
   - risk monitoring
   - relative value analysis
   - scenario analysis

### 最终要强调的不是一个网络，而是一整套 inference system
所以最终交付不是“某个模型结构本身”，而是：
- input-to-surface inference pipeline
- latent state representation
- uncertainty layer
- arbitrage diagnostics / constraints
- downstream usability

---

## 5. 完整任务定义（Task Definition）

### 5.1 输入（Inputs）
更准确地说，输入分三层。

#### A. 市场观测层（Market Observation Layer）
给定某个交易日、某个标的（underlying），我会观测到：
- option bid / ask / mid quote
- strike
- expiry / time to maturity
- option type（call / put）
- spot price of the underlying asset（标的资产现货价）
- risk-free rate / dividend assumptions
- （可选）volume / open interest / liquidity proxy
- mask：哪些点观测到了，哪些点没观测到

#### B. 曲面坐标层（Surface Coordinate Layer）
我会把原始合约映射到某个更适合建模的 surface coordinate system，例如：
- moneyness / log-moneyness
- time-to-maturity
- standardized grid coordinate

#### C. 任务扰动层（Task Corruption Layer）
为了把问题正式定义成 inference task，我需要从相对完整的数据里构造：
- sparse observations
- noisy observations
- irregular observation patterns
- missing regions

也就是说，模型最终看到的不是完整 surface，而是一个**部分观测 + 带 mask 的输入**。

---

### 5.2 输出（Outputs）

#### Phase 1 输出
- dense implied-volatility surface estimate
- 模型恢复出的 dense surface 图
- 和 observed quotes / baseline / vendor reference 的对比图
- reconstruction / robustness metrics

#### 最终项目输出
- inferred dense IV surface
- latent surface representation
- uncertainty map / confidence map
- arbitrage diagnostics
- downstream-relevant stable surface input

---

### 5.3 潜变量（Latent Variable）
潜变量不是为了“看起来高级”，而是为了表达：

> 零散 observed quotes 背后，存在一个更低维、更稳定、更结构化的 latent surface state。

这个 latent variable 可以理解成：
- 某一天的 hidden surface code
- 控制 smile / skew / term structure 的 latent factors
- 一种 market state representation

形式化一点：
- `x = observed sparse/noisy quotes`
- `z = latent surface state`
- `y = dense usable surface`

最终关心的是：
**给定不完整观测 `x`，如何推断合理的 `z`，再由 `z` 支持或生成 `y`。**

---

## 6. 整个项目的 Phase Map
我把整个项目分成 4 个 phase。这个划分主要用于项目管理和叙事，不代表代码一定分 4 次写。

---

## Phase 1：真实数据 + benchmarkable neural prototype
### Phase 1 的目标
把这个项目从“想法”变成一个：
- 有真实数据
- 有明确任务定义
- 有 baseline
- 有最小 neural model
- 有图和表
- 有一些初步结果
- 我能讲 20–30 分钟

的严肃 prototype。

### Phase 1 不追求的东西
- 完整 EBM
- 完整 uncertainty-aware modeling
- 完整 no-arbitrage guarantee
- 全市场 / 高频 / production system

### Phase 1 必须得到的东西
- 一个真数据 pipeline
- 一个可 benchmark 的 task definition
- 一个 PyTorch neural baseline
- 一个简单非神经 baseline
- 一个 vendor-style / traditional reference
- 一些结果图与表

---

## Phase 2：Latent representation / conditional inference
### 核心问题
- simple deterministic reconstruction 到底哪里不够？
- 是否需要显式 latent variable？
- 一天的 surface 能否用低维 latent state 表示？

### 可能的方法
- conditional autoencoder
- latent bottleneck model
- variational latent surface model
- neural process / conditional neural process flavor
- amortized inference for latent code

### 这个阶段的意义
把问题从：
- “recover dense surface”

升级成：
- “infer latent market state and recover a usable surface from it”

---

## Phase 3：Uncertainty-aware + arbitrage-aware structured model
### 核心问题
- 哪些区域模型最不确定？
- 哪些 surface 预测虽然 fit 数据，但结构上不可信？
- no-arbitrage / arbitrage-awareness 应该放在 loss、architecture、还是 inference 里？

### 可能的方法
- heteroscedastic uncertainty
- interval / quantile prediction
- predictive distribution over surfaces
- penalty-based arbitrage diagnostics
- constrained output parameterization
- structure-aware objective

### 这个阶段的意义
让模型从“能补 surface”变成：
- 能告诉我哪里可信、哪里不可信
- 能对结构风险有意识
- 更适合金融语境中的使用

---

## Phase 4：Full latent-variable EBM / structured energy formulation
### 核心问题
如何把整个任务写成一个 energy-based formulation，例如：

`E_theta(x, z, y)`

其中 energy 同时编码：
- observed quote consistency
- latent prior / structure prior
- smoothness / geometry prior
- arbitrage penalties
- uncertainty-aware scoring
- （可选）temporal consistency across days

然后 inference 不再只是一次前向传播，而是：
- `argmin_{z,y} E_theta(x,z,y)`

### 这个阶段的意义
这一步才真正把项目推进到我想要的 intellectual line：
- latent-variable
- structured prediction
- inference-centric ML
- EBM flavor

---

## 7. Baseline 地图
这部分非常重要，因为这个项目不能只和一个自己写的弱基线比。

### 7.1 传统 / Classical Baselines
#### Parametric surface family
- SVI
- raw SVI
- arbitrage-free SVI / eSSVI 类

#### Classical fitting / smoothing family
- spline interpolation
- smoothing
- no-arbitrage smoothing
- calibration-based fitting

#### Vendor-style baselines
这类 baseline 很重要，因为它们更接近真实行业 workflow：
- ORATS-style smoothed / theoretical references
- （未来升级）OptionMetrics / Hanweck / institutional-grade references

---

### 7.2 ML / Neural Baselines
项目最终会和下列思路发生关系：
- autoencoder / VAE-style latent models
- neural process / conditional neural process
- operator smoothing
- physics-informed / constraint-aware variants

但在 **Phase 1** 我不需要把所有这些都做出来。

### Phase 1 的最小 neural baseline
我只需要一个：

> **PyTorch masked neural surface reconstruction baseline**

输入：
- sparse / partial surface observations
- mask
- coordinate features

输出：
- dense implied-volatility surface

可选实现：
- small MLP
- small CNN over surface-like tensor
- conditional autoencoder / bottleneck autoencoder

其中最适合作为后续 latent stage 前身的，是：
- **conditional autoencoder with mask input**

---

## 8. 这个项目最合理的 Novelty 放在哪里
我不应该把 novelty 写成：
- “我发明了 volatility surface”
- “我发明了新的插值法”

更合理的 novelty 是：

> **把 sparse / noisy / irregular IV surface recovery 重新表述成一个 latent-variable, uncertainty-aware, structure-constrained inference problem，并最终走向 energy-based formulation。**

可以拆成几层：
1. 不是 simple fit，而是 **structured inference**
2. 不是 purely deterministic mapping，而是 **latent surface state inference**
3. 不是只给 point estimate，而是 **uncertainty-aware surface recovery**
4. 不是只追 smoothness，而是 **arbitrage-aware / structure-constrained objective**
5. 评测不是只看 reconstruction，而是看 **downstream pricing / risk usability**

---

## 9. 数据与现实约束
### 当前现实约束
- 第一阶段不能依赖 WRDS / OptionMetrics / Wharton 高门槛资源
- 先要做出一个 serious prototype
- 预算是有限但可投入的
- 需要先拿真实数据做出能展示的东西

### 当前现实路线
#### 第一阶段
- 1–3 个高流动性标的
- daily / EOD
- full option chains
- 然后人工构造 sparse / noisy / irregular observations

#### 后续升级
- 有成果后再争取更高门槛数据资源
- 再升级 benchmark 和 literature comparability

---

## 10. Phase 1 的详细计划（Detailed Phase 1 Plan）

## 10.1 Phase 1 的总目标
把这个题变成一个：
- 真实数据上的
- 有 neural baseline 的
- 可严肃评测的
- 能产生图表与结论的

prototype。

一句话：

> **Phase 1 的核心不是做出最终 EBM，而是先把问题落地成一个 benchmarkable neural inference task。**

---

## 10.2 Phase 1 的任务定义
### 数据视角
给定某一天、某个 liquid underlying 的相对完整 option chain，我先构造一个 proxy full surface / reference，然后人为施加：
- missingness
- sparsity
- irregular observation patterns
- quote noise

### 最终模型看到的输入
- partial observed IV / quote surface
- observation mask
- coordinate features（moneyness, maturity）
- （可选）underlying metadata

### 模型要输出的东西
- dense IV surface estimate

### 比较对象
- simple interpolation / smoothing baseline
- vendor-smoothed / traditional reference
- observed quote consistency

---

## 10.3 Phase 1 先做哪些标的
最稳的优先顺序：
1. **SPY**
2. **QQQ**
3. **AAPL**（可选）

### 具体建议
如果想把风险降到最低，就先只做 **SPY**。

原因：
- 流动性高
- quotes 丰富
- surface 形状更有代表性
- 更适合做第一版 prototype

---

## 10.4 Phase 1 用什么模型
### 核心判断
Phase 1 不能只有 benchmark；必须有一个最小的、能跑起来的 neural model。

### 推荐模型
> **PyTorch masked neural surface reconstruction baseline**

### 推荐实现路径
优先级从高到低：
1. **conditional autoencoder with mask input**
2. small CNN / surface tensor model
3. small MLP baseline

### 为什么推荐 conditional autoencoder
因为它同时满足：
- Deep Learning 味足够
- 容易实现
- 能自然讲 latent bottleneck
- 非常适合作为后续 latent-variable phase 的前身

---

## 10.5 Phase 1 的 Baselines
### Baseline A：最简单非神经 baseline
必须要有一个。

可能包括：
- nearest / linear interpolation
- spline interpolation
- simple smoothing

### Baseline B：traditional workflow baseline
如果后面有精力，可以逐步补：
- SVI family
- simple calibrated surface fit

### Baseline C：vendor-style / industry-style reference
这很关键，因为这能让比较更像真实 workflow，而不是只打 strawman。

例如：
- ORATS-style smoothed / theoretical reference

### Phase 1 的最佳组合
- 一个简单 baseline
- 一个 vendor-style reference
- 一个 PyTorch neural baseline

---

## 10.6 Phase 1 的评测指标
不要只用一个 MSE。

### A. Surface reconstruction accuracy
- MSE / MAE on dense IV surface
- weighted error（后续可按区域 / liquidity 加权）

### B. Observed quote consistency
- 在 observed points 上的 fit
- 在 unobserved points 上的 generalization

### C. Robustness under sparsity / noise
- error vs sparsity level
- error vs noise level
- error by maturity region
- error by moneyness region

### D. Structural sanity checks（后续可加）
- smoothness diagnostics
- simple arbitrage diagnostics

---

## 10.7 Phase 1 必须产出的 Deliverables
没有 deliverables，这个 phase 就不算完成。

### 必须有的图（minimum）
1. 一张完整 reference surface 图
2. 一张 sparse / noisy observed surface 图
3. 一张 model-reconstructed surface 图
4. 一张 baseline vs model error comparison 图
5. 一张 error vs sparsity 的曲线图
6. 最好再加一张按 maturity / moneyness 分区的 heatmap

### 必须有的表
至少一个表格总结：
- model
- simple baseline
- vendor reference alignment / gap
- metrics under different sparsity regimes

### 必须有的 narrative
我必须能讲清楚：
- 为什么这个 task 是真实的
- neural baseline 做了什么
- 跟谁比
- 哪些区域做得好
- 哪些区域很差
- 为什么这反过来证明 Phase 2 / Phase 3 是有必要的

---

## 10.8 Phase 1 成功的定义
不是“我指标碾压”。

### 满足下面这些，就算成功：
1. 真实数据 pipeline 跑通
2. 至少一个标的（最好 SPY）完整跑通
3. 一个 PyTorch neural baseline 跑通
4. 一个简单 baseline 跑通
5. 有 vendor-style / traditional reference
6. 有图、有表、有初步结论
7. 我可以围绕它讲 20–30 分钟，不虚

---

## 10.9 如果 Phase 1 结果不好怎么办
结果一般、甚至很差，也不是坏事。

因为这正好可以告诉我：
- deterministic reconstruction 不够
- 某些区域 inherently uncertain
- missingness 结构比我想象中更关键
- interpolation baseline 在局部失效
- vendor reference 和 raw quotes 之间本身存在 tension

这些发现可以直接支持后续 phase：
- 为什么要 latent variable
- 为什么要 uncertainty-aware inference
- 为什么要 arbitrage-aware / structure-aware objective
- 为什么最终要 EBM

所以：
> **Phase 1 不要求结果漂亮，但一定要有东西（there should be something）。**

---

## 11. Phase 1 之后的升级路线
### Phase 2 升级
- explicit latent code
- conditional latent inference
- better latent surface representation

### Phase 3 升级
- uncertainty estimation / calibration
- structure constraints
- arbitrage-aware penalties / diagnostics

### Phase 4 升级
- full latent-variable EBM
- energy-based inference
- maybe temporal consistency across trading days

---

## 12. 我当前最小可执行版本（Minimum Viable Research Prototype）
如果我现在要以最小代价启动项目，那我应该瞄准这个版本：

### 最小版本定义
- 标的：**SPY**
- 数据：真实 **daily / EOD option chains**
- 模型：一个 **PyTorch neural baseline**（最好带 mask 输入，最好有 latent bottleneck）
- baseline：一个简单 interpolation/smoothing + 一个 vendor-style reference
- 输出：
  - 3–6 张图
  - 1 张 summary table
  - 一套最初步的结论

### 为什么这是最优起点
因为它已经足够：
- 写到简历里
- 在面试里讲清楚
- 作为后续 phase 的踏板
- 证明这个项目不是空想

---

## 13. 项目管理上的提醒
### 当前最重要的不是
- 一上来实现 full EBM
- 一上来证明完整 no-arbitrage
- 一上来做很 fancy 的文献级新结构

### 当前最重要的是
1. 任务定义站稳
2. 真数据 pipeline 跑通
3. baseline 站稳
4. neural baseline 跑通
5. 有结果图和表

只要这 5 件事站稳，后面的 latent / uncertainty / EBM 才不虚。

---

## 14. 我现在的优先级（Execution Priority）
### 当前优先级顺序
1. **确定并拉通真实 EOD 数据 pipeline**
2. **只选 SPY 做第一个 end-to-end 跑通版本**
3. **定义 sparse / noisy / irregular masking scheme**
4. **实现最小 PyTorch neural baseline**
5. **实现一个简单 non-neural baseline**
6. **接入 vendor-style / traditional reference**
7. **做第一轮可视化和 error 分析**
8. **写出第一版结论：哪里有效，哪里失败，下一阶段为什么需要 latent / uncertainty / structure-aware methods**

---

## 15. 最后一句话总结
这个项目的本质不是“预测一个 surface”，而是：

> **从不完整、带噪、非规则的期权市场观测中，恢复一个可用于 pricing / risk 的 latent volatility surface representation。**

Phase 1 的任务不是直接冲向最终 EBM，而是先把这个问题变成一个：
- 真实数据上的
- benchmarkable 的
- 有 neural model 的
- 有图表和结果的

严肃 research prototype。

