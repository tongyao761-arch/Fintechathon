# 跨模型固定实验预登记 v1

登记日期：2026-10-11。**环境安装与合成验收已完成，正式实验未执行。** 本计划只比较冻结29列、原始标签和相同训练总体上的两个固定CPU配置，判断是否值得进入下一阶段受限复核；不研究各模型最优参数或最优特征。

机器可读权威配置：[model_comparison_v1.json](../../configs/model_comparison_v1.json)。本轮证据目录：[20261011T073612105802Z](../../artifacts/model_comparison_v1/20261011T073612105802Z/)。配置、本文和环境的最终SHA256由[HANDOFF](HANDOFF.json)登记。本轮账本独立于旧账本，正式fit/predict/评分/上传全部为0。

## 1. 数据合同与原方案保护

当前正式主方案PJ_a080、备选PJ_a090共用P+J LightGBM模型，平滑为0.8/0.2和0.9/0.1。以[最终v2冻结](../../artifacts/final_selection_v2/20261008T180831618991Z/FINAL_SCHEMES_V2.json)、[训练交接](../final_training_v1/HANDOFF.json)、[交付交接](../final_delivery_v1/HANDOFF.json)、[风险复核](../final_risk_review_v1/RESULTS.md)为依据。历史阶段的“未训练/未导出”仅表示当时。

新实验固定以下有序列，不能增删或换序：

```text
ret_1d, ret_5d, ret_20d, gap_1d, intraday_ret, high_low_range,
volatility_5d, volatility_20d, volume_ratio_5d, amount_ratio_5d,
close_location, upper_shadow, body_ratio, bias_20d, bias_60d,
price_position_60d, amount_ratio_20d, log_mean_amount_20d, volatility_60d,
flag_limit_up, flag_limit_down, limit_up_count_5d, limit_down_count_5d,
ret_1d_rank_pct, ret_20d_rank_pct, amount_ratio_5d_rank_pct,
log_mean_amount_20d_rank_pct, downside_deviation_20d, illiquidity_proxy_20d
```

原训练CSV SHA256为`e9d23e87f7e1f9439f28f7dd951d129ea49eaacda5ad650da8576a830a6deac9`，本轮实际按字节核验，未把样本输入新模型。29列完整历史缓存为`artifacts/final_training_v1/20261010T082317559902Z/history_features.parquet`，7,900,350行，SHA256为`6511acccdf628af932f1f450fb0918d8e523b5a191e766f512435e377fe72b6e`；本轮核验文件哈希和Parquet页脚列序/float32类型，未重新计算真实X。其因果性和原始总体一致性来自已验收风险复核，不能称本轮已重新做逐值重建。

未来正式入口复用原函数，不修改旧业务代码：

1. `src/data/baseline_panel.py::load_raw_baseline_panel`读取完整按股票/日期排序的原面板。价格资格先用原始float64 OHLC计算，再将原数值X按旧合同转float32；价格须有限、为正且满足高低价包络。
2. `src/features/feature_round2.py::project_x`只提供X及当日价格质量标记，调用`build_feature_round2`，原P委托未改动的`build_features34`。在完整原面板计算全部X，之后才筛训练；不删缺失行凑窗口、不补日期，按原实现将非有限特征转NaN。与封存29列缓存按原键对齐，有限值精确相等且NaN位置相等；NaN符号位不当作数值差异。
3. `scripts/run_lightgbm_baseline.py::split_masks`筛时间区间、`is_trainable==1`、原Y存在，检查原float64 Y及转float32训练Y均有限；原始`y_ret_1d`不填补、不去均值、不排序归一化、不截尾。验证保留完整区间原键，包括价格无效和Y缺失行；评分使用原CSV标签精度。
4. 包装放在未来新入口：XGBoost接受原float32数值矩阵，缺失哨兵`np.nan`；hist的内部量化只拟合训练X。CatBoost用数值Pool，`cat_features=[]`，无股票或日期类别。内部量化与缺失分支是库算法，不改变输入值/NaN/行序，不允许外部填充。标签、标签缺失状态、`is_trainable`不进入29列；股票/日期只用于键、分组、切分和日历。原X-only测试adapter仅是后续测试阶段参考，本次与开发阶段均不运行正式测试桥接或测试预测。

### 固定年度与身份

| 年度/名称 | 训练起止 | 排除的purge日Y | 完整验证区间 | 合格训练行 | 验证行 |
| --- | --- | --- | --- | ---: | ---: |
| 2021 / dev_2021 | 20180102—20201230 | 20201231 | 20210104—20211231 | 2,594,048 | 1,129,950 |
| 2022 / dev_2022 | 20180102—20211230 | 20211231 | 20220104—20221230 | 3,568,878 | 1,125,300 |
| 2023 / primary_2023 | 20180102—20221229 | 20221230 | 20230103—20231229 | 4,597,785 | 1,125,300 |

三年合同从对应P+J实例的`actual_config.json`提取，与`configs/feature_round2.json`和原`split_checks.json`逐项一致。每年purge均为4,650行，保留该日X作历史，不纳入该日Y；检查每个训练标签依赖的下一市场日早于验证起日。不得随机切分或缩成统一三年窗。

精确训练键、验证键及float32标签SHA256全部写入配置`data.splits.*.expected`；列序哈希、完整X缓存哈希及源码哈希写入同配置和[reference_manifest](../../artifacts/model_comparison_v1/20261011T073612105802Z/reference_manifest.json)。这些是已验收记录及本轮文件核验，不虚构不存在的逐年度训练矩阵字节哈希。未来fit前必须按同一原函数重建并匹配键/标签哈希，另登记训练/验证X有限值、NaN位图、source_row_id、列序和矩阵字节哈希，两模型使用同一封存矩阵。这一步只读准备，不消耗fit；失败即停止，不能调合同补救后混入结果。

## 2. 矩阵、参数与固定后处理

矩阵仅XGBoost 3.4.1、CatBoost 1.2.10 × 2021/2022/2023，六个实例串行，先XGBoost三年，再CatBoost三年。每实例固定100轮、learning_rate=0.05、种子20260927、32个模型线程，不进行种子比较、早停、搜索或年度参数变更。保存重载容差不证明重新fit确定性。

所有显式构造参数在配置`models.*.params`，合成验证了同一配置仅把轮数改为5的合法性；完整有效参数另封存在`synthetic_*_effective.json`。正式100轮尚未实跑。参数理由如下：

| 项目 | XGBoost | CatBoost |
| --- | --- | --- |
| 损失/设备 | reg:squarederror、gbtree、hist、cpu | RMSE、CPU、Plain |
| 树结构 | depthwise、max_depth=6、max_leaves=64 | SymmetricTree、depth=6，最多64叶；不套用仅其他grow_policy适用的叶数/最少样本参数 |
| 约束/正则 | min_child_weight=100、gamma=0、L1=0、L2=1 | l2_leaf_reg=3、score_function=L2、random_strength=0 |
| 抽样 | 行及三层列采样比例全1、uniform | bootstrap_type=No、rsm=1 |
| 量化/缺失 | max_bin=256、原NaN保留 | border_count=254、GreedyLogSum、nan_mode=Min |
| 初始化/叶估计 | base_score=0、max_delta_step=0、单目标单树 | boost_from_average=false、Newton一次、backtracking=No |
| 其余固定项 | num_parallel_tree=1、one_output_per_tree、无类别、无early_stopping、参数校验启用 | has_time=true、fold_permutation_block=1、shrink_rate=0、Constant、无langevin/posterior_sampling、use_best_model=false、禁止额外写文件 |

100轮和0.05是事前固定小规模起点，不代表两库与LightGBM复杂度相等；depth=6是限制单树规模的选择，也不是把LightGBM的31叶机械搬过来。CatBoost对称树不具有LightGBM的同一最少叶样本合同，必须披露这种差异。XGBoost显式base_score避免自动估计初始截距；CatBoost显式学习率、boosting、bootstrap和叶估计避免样本大小/目标导致关键默认值切换。has_time只保留现有输入顺序，不把股票优先排序解释为严格全局时间序，也不代替时间切分或purge；数值特征没有类别目标编码。其余未显式设置项由精确库版本冻结，正式每实例保存完整有效配置，已知CPU/目标/轮数/约束必须核对，未知参数警告视为技术失败。

[XGBoost v3.4.1参数源码](https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/parameter.rst)与[CatBoost v1.2.10 Python源码](https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/python-package/catboost/core.py)用于版本对应；[CatBoost参数说明](https://catboost.ai/docs/en/references/training-parameters/common)为滚动文档，不能替代已安装版本的实际验收。CatBoost学习率有效存储约0.050000000745属于float32表示，不另作调参。

后处理恰好两种：原预测`raw=[1,0]`与`a080=[0.8,0.2]`。复用每实例一次原预测，调用原`select_final_candidates_v2.py::smooth`：同股票、同模型、精确前一市场日**原**预测，非递归。首个验证日、跨年/模型边界或缺失精确前值回退当日原预测；不为了平滑额外推断purge日。没有0.9/0.1、新权重、融合或缺失值压低规则。

对照逐年复用最终v2`run_index.json`登记的PJ_a100/PJ_a080封存预测、对应历史模型、truth/X及原评分文件；配置列出全部绝对路径和SHA256。原预测分项对PJ_a100，平滑分项对PJ_a080，最终跨模型开发排序全部对PJ_a080。禁止当前全量最终模型预测历史年份作对照。

## 3. 调用预算和资源停止规则

| 项目 | 后续开发硬上限/登记方式 |
| --- | --- |
| 真实fit | 6，失败入账，无自动重试；一格失败即矩阵未完成，停止，不用五格均值选胜者 |
| 原完整验证predict | 6，每实例1次；按库内部操作不能偷偷增加第二次高层predict |
| 保存重载predict | 最多6，同总体每实例至多1次；合计真实predict≤12 |
| 新方案年度逻辑评分 | 2模型×3年×2后处理=12 |
| 对照年度逻辑评分 | 2对照×3年=6；合计18，不因两模型重复计算同一对照 |
| 月度逻辑评分 | 新144+对照72=216；每个方案每年12月，不能取月度均值替代年度等权选择 |
| 实际年度计算 | 官方18+严格wrapper18+独立18=54；三者失败调用仍计数，不额外追加 |
| 月度调用/聚合 | 官方整月独立运行0次；从wrapper和独立的年度逐日序列各聚合216项，共432次聚合，保留跨月换手 |

官方源函数仅返回年度标量；严格wrapper及独立实现输出逐日和月度。不能把月度聚合称216次官方evaluate调用。账本分别记logical、函数进入、成功/失败、逐日/月度聚合、保存重载预测，事件在函数进入前落盘；年度评分失败即停止，不擅自重跑。本轮这些正式计数全部为0，合成fit=2、合成初次predict=2、重载predict=2单独列明。

本轮机器32逻辑CPU，内存总约31.72 GiB，起点可用约16.53 GiB、磁盘空闲约170.00 GiB；精确值见环境快照。已有P+J三年运行收据分别16.00/17.78/18.69秒、峰值3505.66/3535.25/3614.15 MiB，涵盖fit、推断、序列化及评分等。最终全量LightGBM阶段已有收据736.49秒、峰值7,925,067,776字节，涵盖prepare/fit/predict/audit。以上只用于认识本机既有流程规模，**不是新模型实测或耗时预测**。

新库真实资源未知，事前明确以下是预算而非测量：

- 模型串行，每实例fit+原/重载predict+保存共≤1,800秒；六实例累计≤10,800秒，失败计入。特征准备≤1,800秒，评分和核算≤7,200秒，整轮开发从准备到验收≤21,600秒（6小时），其余余量包含最终写档。
- 启动时可用内存≥14 GiB、磁盘空闲≥40 GiB、逻辑CPU≥32，否则停止并报告，不能自动改线程/另建环境。32线程同时满足逻辑CPU数和32硬上限；模型入口局部限制BLAS为1，不修改全局PATH或系统环境。
- 含父进程及全部子进程的RSS之和≤12 GiB；系统可用内存不得低于4 GiB。每0.5秒采样RSS/可用内存，记录实际峰值及采样间隔，不能声称捕捉瞬时完整峰值。
- 本轮新产物≤20 GiB，运行期间磁盘空闲≥20 GiB，每2秒检查。已有文件大小不计新产物预算但不得删除或覆盖；共享原输入只读，所有输出独占新目录。
- 未来runner必须先实现外部父监控器和持久账本，第一次超限终止当前登记进程树、封存失败和部分文件、停止整矩阵。CatBoost自身软内存参数不能代替RSS硬监控。不得临时缩样本、减少轮数、切GPU或把失败预算改成参数搜索。

资源上限、采样精度和所有命令墙钟起止必须记录；安装/纯合成耗时不推断百万行训练资源。未来无完整资源监控即不得启动正式fit。

## 4. 官方口径、独立核算和风险披露

按[官方evaluate.py](../../赛题五/evaluate.py)计算：IC排除Y NaN且少于30行的日跳过，Spearman平手平均秩；收益排除涨停和Y NaN、少于100行日跳过；换手仅排除涨停、少于100行日重置前集合。Top取`max(n//10,1)`，收益为当日Top均值减池均值，年化乘252。换手是相邻合格日集合Jaccard距离。年度总分=`0.4*IC+0.3*annual_excess+0.3*(1-turnover)`，不替换价格有效换手。

键唯一完整、原验证行序和默认CSV解析输入须相同；按旧源码`sort_values(pred, ascending=False)`及当前精确pandas版本的默认平手行为，不新增代码/日期第二排序键。少样本日、不连续记录、常量预测导致NaN指标都如实保留；年度技术验收要求用于门槛的指标有限，不能填零。月度从年度逐日序列聚合，换手包含跨月首日与上月末的真实比较，不另作12次月初重置评分。

严格wrapper复用`src/metrics/official.py::score_official`；独立使用`audit_final_candidates_v2.py::oracle_scores`的平均秩Pearson和独立Top集合，并用`oracle_smooth`独立核查平滑。只调用函数，不运行旧CLI或其旧矩阵。官方/独立/wrapper年度、逐日、月度标量最大绝对差≤1e-12，rtol=0；Top成员及键/资格完全一致。保存重载预测atol≤1e-12、rtol=0；CSV以`%.17g`输出，round_trip与封存数组逐值一致。对照来源/哈希以及已有评分与本轮核算一致也必须通过。上述容差均不构成fit确定性、统计显著性或隐藏泛化证明。

每版本披露年度分差及`0.4*ΔIC`、`0.3*Δannual_excess`、`-0.3*Δturnover`贡献；全部36个月分差、负月数、最差年/月、收益/IC冲突、Top收益与官方换手两套总体的构成。价格无效、Y缺失、输入NaN按股票—日期出现次数计分母，另报告当日候选池比例、价格有效样本Top进出与Jaccard稳定性。价格有效诊断不替代官方分，不把价格无效与历史Y缺失关联推广成隐藏Y事实。

此前去均值标签方向均分下降，收益/IC增长可能被换手代价抵消；缩短训练窗/加树等既有负结果不能通过新轮悄悄复活。新库的缺失分支或量化不同可能改变Top构成，换模型不意味着缺失机制、漂移、负月、有效样本表现或选择偏差已解决。2021—2023已反复开发，2024已查看，追加四个版本又增加选择偏差；未胜出仅说明本次固定配置未胜出，不说明库本身无效。

## 5. 提名门槛与固定排序

四个新版本（两模型×raw/a080）均相对PJ_a080计算三年年度综合分等权均值增量。必须同时满足：均分增量严格>0.003，至少2/3开发年分差>0，最差年分差≥−0.005，全部技术/预算检查通过。0.003和−0.005是本轮事前决策阈值，非赛题规定、显著性标准或未来保证。

月度诊断完整披露，不事后增加或放宽隐藏阈值。另固定“无效样本占位主导”否决：至少2年官方换手Top的价格无效出现比例高于PJ_a080，同时三年均值换手得分贡献>0、IC加收益得分贡献≤0，三条件全部成立则不提名。这是保守的本轮解释规则，不证明相关性具有因果关系；不能只因低官方换手便宣称可交易稳定性改善。分母和年度等权算法在配置中固定。

最多提名一个模型的一个后处理版本。合格候选按开发均分降序，用1e-12固定最高锚值分组处理平手，组内依次最差年度相对PJ_a080分差降序（同样固定最高锚分组）、raw优先、XGBoost优先于CatBoost。禁止链式相邻容差产生排序漂移。没有候选通过即停止，保留PJ_a080及当前submission；不尝试新参数补救。

## 6. 2024受限复核及交付边界

本次不执行2024。未来须开发完整结束、候选模型/参数/后处理及身份哈希预冻结、过门且得到该阶段明确授权后，才复核此一个候选。固定训练20180102—20231228，排除20231229 Y并保留X，验证20240102—20241231；合格训练5,659,954行、验证1,125,300行，精确键/标签合同在配置中。只准1次fit，1次完整验证predict，至多1次同总体重载predict，失败计入且停止；不开第二名。

复核对照固定PJ_a080历史2024封存预测。年度逻辑评分2、月度24；官方/wrapper/独立各2次、年度实际6次，月度聚合48次，额外官方整月运行0。沿用单实例30分钟、RSS/内存/磁盘上限和监控规则，另登记整阶段≤7,200秒。通过须技术验收全部通过，2024综合分差≥0，12月中至少6个月分差>0，最差月分差≥−0.05，且不符合单年占位否决（无效Top比例升高、换手贡献>0、IC/收益贡献≤0同时成立）。这些补充阈值均为本轮事前保守规则，不冒充赛题规定；不得看2024后调参、改平滑或换候选。

2024已查看，不能称独立盲测。复核通过只表示值得进入单独交付决策，不自动替换主备，不重训全量、不产生测试新预测或新submission、不上传。本轮正式submission仍是`artifacts/final_delivery_v1/20261010T090539553866Z/submission.csv`，SHA256 `96f0f30b5c52c526705c6b2dff9e89ebad730dbe15474219025e187a3c2faaa5`。

## 7. 下一阶段起点与停止点

本次完成环境安装、两次纯合成验收、预登记、保护验收和状态导航后停止。无真实fit、predict、官方/独立评分，无Git提交、上传或远端操作。保留全部既有成果，旧依赖锁/账本/业务代码原样。

后续若获得正式开发授权，先读取本计划、配置及HANDOFF，核验源/配置/环境/对照哈希，使用新命名空间准备带预算保护的runner和独立审计入口；新入口尚未实现，本次交付不声称已经完成完整runner。无需重跑安装或合成fit。确认机器资源门槛、矩阵身份和计数事件可审计之后，才开始登记的六次fit。不能把现有安装辅助脚本当真实实验入口或自动启动命令，也不执行任何旧提示词。
