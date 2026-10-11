# 提交前风险复核：唯一可用执行提示词

请实际完成本项目的“提交前风险复核与固定历史诊断对照”，不要只给计划。先核查赛题原文、官方评分源码和项目实际文件，再登记、实施、运行、独立验收、保存证据与交接；完成本轮后停止。

你不需要此前聊天。以下是完整任务、当前状态、授权和边界。参考数值只用于定位问题，必须重新计算、核对分母并落盘，不得当作已验收证据直接抄写。

## 一、项目、目标和明确授权

项目根目录：`C:\fintechathon`；唯一工作分支：`ivor-work`；使用现有 `.venv`，不安装或升级依赖。读取并遵守 `AGENTS.md` 和 `AGENTS.override.md`。本任务明确不提交Git，优先于一般完成任务后提交的约定；不调用子代理。

本轮独立阶段命名 `final_risk_review_v1`。目标是回答：

1. 高预测组为何富集价格无效样本：是实现错误、训练与预测总体差异、缺失特征输入行为，还是目前只能确认相关性？
2. 历史主方案的官方成绩和改善，有多少依赖官方换手分项及其Top样本构成？不能把分项贡献直接当成无效样本的因果贡献。
3. 一个固定、仅用当日X的历史诊断对照，能否降低无效Top比例，其官方IC、收益、换手和综合分代价是什么？
4. 现有漂移、负年/负月和重复筛选局限是否阻断交付，还是应保留原方案并披露风险？

本提示词明确授权：

- 读取允许的数据与已有模型、特征、预测、评分、账本和验收记录；加载保存模型检查结构和参数，不调用推断。
- 重算并保存本任务需要的历史及测试X描述性统计；历史Y只能用于历史评分和诊断。
- 新增必要的独立代码、配置、mock/合成X或预测测试、登记、诊断产物、验收和交接文档。
- 从已保存主方案年度预测生成下文唯一一个固定的**历史诊断**对照，计算最多8项年度逻辑评分并核对官方口径；不新增训练或模型推断。
- 对本轮新代码的普通实现错误做同配置修复，留存原代码、diff、失败日志及哈希链接，继续使用同一登记/运行目录。
- 验收后更新 `docs\PROJECT_PROGRESS.md`，先存档原文并建立版本链；如本轮结论需更新README，也先存档并接续已存在的README版本链。

已授权步骤不用逐项再次询问。若没有技术错误、诊断对照没有支持替换，结论可以是“保留当前主方案并披露局限”；这同样是完成任务，不能为了必须改进而继续搜索。

## 二、本轮禁止和预算

禁止：

- 任何fit，包括合成fit、失败fit重试、继续训练、拟合变换或辅助缺失标签模型。
- 任何新的模型predict，包括小样本、分块、重载后推断、`pred_leaf`，以及手工遍历树计算新预测。
- 特征、标签、模型、参数、平滑权重搜索；新增第二个处理规则或反复调整本轮固定规则。
- 访问隐藏测试Y；从测试价格重建未来收益；估计隐藏IC、收益、综合分、排名或提升概率。
- 利用测试X统计选择处理规则、参数或候选；本轮测试X只做固定描述性复核。
- 将诊断对照应用到2025—2026正式测试预测；修改、重新序列化、重算或覆盖原/主备/reload预测及正式submission。
- 删除任何测试行；用标签是否缺失、`is_trainable`或未来信息决定预测处理。
- 改写旧冻结、旧账本、旧验收和旧交接；重新执行历史训练/审计入口或全仓测试来“刷新结果”。
- Git提交、远程操作、上传、制作比赛报告、修改push URL/pre-push钩子、删除历史文件或清理既有临时目录。

固定预算：

| 项目 | 上限 |
| --- | ---: |
| 新fit / 合成fit / fit重试 | 0 / 0 / 0 |
| 新模型推断 | 0 |
| 新后处理规则 | 1，见第六节 |
| 新历史诊断预测表 | 4，2021、2022、2023、2024各一份 |
| 年度逻辑评分 | 8：原主方案4项重核 + 唯一诊断对照4项 |

诊断对照是对保存值的后处理，不是新模型推断；分别记账，不能将它们混称为零新增预测文件。官方/独立算法的同一评分交叉核验不构成新候选，但要记录实际计算次数、耗时和失败重算。禁止反复重新评分增加搜索。

## 三、先读哪些实际文件

按顺序读取；大型JSON先检查结构，再选择性读取，不输出整份大JSON。

赛题和状态：

```
C:\fintechathon\AGENTS.md
C:\fintechathon\AGENTS.override.md
C:\fintechathon\README.md
C:\fintechathon\docs\PROJECT_PROGRESS.md
C:\fintechathon\赛题五\完整赛题说明_本地.md
C:\fintechathon\赛题五\evaluate.py
C:\fintechathon\src\metrics\official.py
```

最终方案和历史评分权威入口：

```
C:\fintechathon\docs\final_selection_v2\HANDOFF.json
C:\fintechathon\configs\final_selection_v2.json
C:\fintechathon\artifacts\final_selection_v2\20261008T180831618991Z\FINAL_SCHEMES_V2.json
C:\fintechathon\artifacts\final_selection_v2\20261008T180831618991Z\run_index.json
C:\fintechathon\artifacts\final_selection_v2\20261008T180831618991Z\annual_metrics.csv
C:\fintechathon\artifacts\final_selection_v2\20261008T180831618991Z\monthly_metrics.csv
C:\fintechathon\artifacts\final_selection_v2\20261008T180831618991Z\pairwise_annual.csv
C:\fintechathon\artifacts\final_selection_v2\20261008T180831618991Z\pairwise_monthly.csv
```

从v2 HANDOFF定位独立acceptance、artifact index；从38条 `run_index.json` 记录中选取 `id=PJ_a080` 的四年记录，核对其预测、评分CSV、历史truth/X、基础模型、split和SHA256。不要猜测历史预测路径。P_a080、PJ_a100等其他方案只读已有指标/证据，不产生额外候选或评分。

最终训练：

```
C:\fintechathon\docs\final_training_v1\HANDOFF.json
C:\fintechathon\docs\final_training_v1\RESULTS.md
C:\fintechathon\docs\final_training_v1\AUDIT_REPAIR.md
C:\fintechathon\configs\final_training_v1.json
C:\fintechathon\artifacts\final_training_v1\task_ledger.json
C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\
```

该运行目录内核对registration、actual_config、training_input_summary、train_samples、history/train/test_features、test_keys、market_calendar、final_model、raw/主备/reload预测、smoothing_trace、events、run_index及 `source_repair_01`。

当前正式交付及漂移：

```
C:\fintechathon\docs\final_delivery_v1\HANDOFF.json
C:\fintechathon\docs\final_delivery_v1\RESULTS.md
C:\fintechathon\artifacts\final_delivery_v1\20261010T090539553866Z\
```

从当前HANDOFF定位 `final_acceptance.json`、`final_run_index.json`、`final_delivery_verification.json`、drift_summary、raw/feature_statistics、feature_population、feature_drift_comparisons、coverage、版本链及 `source_repair_01`—`04`。初次run_index、初次资源和旧交接属于历史版本，不能当成最新完整结论。

文档更新版本链：

```
C:\fintechathon\docs\project_docs_update_v1\HANDOFF.json
C:\fintechathon\artifacts\project_docs_update_v1\20261010T120147223008Z\prior_README.md
C:\fintechathon\artifacts\project_docs_update_v1\20261010T120147223008Z\readme_version_link.json
```

README已获授权更新，旧交付保护快照记录的是旧版哈希。必须核验精确版本链接和存档，不将其误判为篡改，也不能据此忽略其他哈希差异。旧验收入口的保护检查不会自动解析后来README版本；不要直接跑其完整入口，独立核验或复用已读源码的纯函数。

实现源码按需读取，先确认接口和副作用：

```
C:\fintechathon\src\data\data_contract.py
C:\fintechathon\src\data\baseline_panel.py
C:\fintechathon\src\data\frozen_test_adapter.py
C:\fintechathon\src\features\baseline_v1.py
C:\fintechathon\src\features\features34.py
C:\fintechathon\src\features\feature_round2.py
C:\fintechathon\scripts\run_final_training_v1.py
C:\fintechathon\scripts\audit_final_training_v1.py
C:\fintechathon\scripts\run_final_delivery_v1.py
C:\fintechathon\scripts\audit_final_candidates_v2.py
```

## 四、当前状态和哈希锚点

以下均须实际核验：

- 主方案 `PJ_a080 [0.8,0.2]`，备选 `PJ_a090 [0.9,0.1]`；同一P+J基础模型、精确29列、100棵树、实际32线程。完整参数和列序读取冻结/config，不自行重建名单。
- 最终训练6,754,006行，原始 `y_ret_1d` float32、标签全部有限。日期20180102—20241230；20241231全部4,650个Y排除，全部4,650行X留作历史。
- 最终训练累计fit=1、完整测试推断=2，预算已用完；当前主备和正式文件不变。
- 正式submission为1,599,600行、4,650股票、344交易日、20250102—20260608，完整键和原行序、有限值、模型加载、保存重载证据及独立平滑均已验收。
- submission已生成，未上传，比赛报告尚未完成。历史开发/复核成绩来自年度验证模型，不是当前全量模型的训练内成绩，也不是测试成绩。

哈希锚点：

| 文件 | SHA256 |
| --- | --- |
| `赛题五\赛题五数据\训练集.csv` | `e9d23e87f7e1f9439f28f7dd951d129ea49eaacda5ad650da8576a830a6deac9` |
| `赛题五\赛题五数据\测试集_X.csv` | `dceffbf68bbab07a254bc6caa4d8a1e33de8de9c15489d936c5d15020dbfa90f` |
| v2 `FINAL_SCHEMES_V2.json` | `1d6bdab3d45c5faa4c9cb40191c540308eefeab55fcefc3ebfe020c0c53b7ede` |
| 最终训练 `final_model.txt` | `76d9fc38e08e797cab5b939561085bf392ce3dbcb191b17878a6b50646f38ede` |
| 最终训练 `raw_predictions.parquet` | `10e1f481cf79e4bb811848efccc8a4877120ab63e9d23b26c715d8dca1076d0f` |
| 最终训练 `PJ_a080_predictions.parquet` | `77de51875fe8c6232233ecd870023736a4f69306fb807304a3a9d4830fbe8652` |
| 最终训练 `PJ_a080_predictions.csv`及正式submission | `96f0f30b5c52c526705c6b2dff9e89ebad730dbe15474219025e187a3c2faaa5` |

唯一正式submission路径：

`C:\fintechathon\artifacts\final_delivery_v1\20261010T090539553866Z\submission.csv`

其余路径与哈希按相应HANDOFF、registration、index和修复链接核验。未获授权的来源/数据/模型/预测/合同不一致时停止报告，不能改数据、补训、推断恢复或重新导出掩盖问题。

## 五、先登记再重算的固定风险诊断

先登记口径、来源哈希、候选公式、预算、接受标准、环境、Git状态和保护快照。检查是否已有本阶段登记；存在则审计并复用同一任务，不换目录掩盖失败。修改或新增文件前说明路径和原因。

价格有效规则：用原始CSV的float64 OHLC计算，所有价格有限且>0，high>=该行OHLC最大值，low<=该行OHLC最小值；与冻结加载器完全一致。不要因float32转换改变原始价格门槛。

必须重算保存：

1. 历史2018—2024逐年及总体的价格有效/无效×Y有限/NaN/Inf交叉表；记录实际分母、日期范围和数值编码。
2. 已保存主方案2021—2024的收益Top、官方换手Top和价格有效Top构成，分别统计标签缺失、价格无效、特征缺失；后两者均是辅助诊断，不能改官方评分池。
3. 原方案测试X：只排除涨停后按保存pred降序取每日 `max(n//10,1)`，统计Top与候选池的价格无效比例、逐日/年/月分布及高预测分位；禁止读取/估计测试Y或计算测试换手/其他隐藏评分。明确这些是样本构成，不是隐藏成绩。
4. 主方案训练总体与预测总体的差异：利用封存矩阵和keys/source_row_id严格对齐，比较价格状态、至少一列/全部29列NaN、缺失模式和保存预测分布；辅助行号不作为模型输入。可读模型结构说明可能机制，不能另算模型输出或把相关性写成确定因果。
5. 官方总分和分项的年度/月度对照：从已有指标核验PJ_a080对PJ_a100和P_a080的改善来源、负年/负月；全年分数按全年逐日数据计算，不取月度分数平均。
6. 重核已留档漂移：完整历史X、2024历史X、实际训练样本、全部测试X及2025/2026、测试各月分别保留分母；检查29列NaN/Inf和有限值P5/P25/中位数/P75/P95。重点downside_deviation_20d、illiquidity_proxy_20d、60日窗口；2026和202606为不完整期间。不重算特征定义，不填补或删行，不拟合测试统计。

聊天中尚需独立留档的定位参考：

- 历史价格无效1,139,980行，均为Y NaN；历史Y NaN共1,141,819行，价格有效但Y NaN为1,839行，价格无效但Y有值为0行。不要从这种历史关联推出隐藏Y一定缺失。
- PJ_a080官方换手Top的标签缺失占比（按Top股票—日期出现次数加权）：2021=100%，2022≈85.411376%，2023≈56.118969%，2024≈20.945426%。不是按每日比例简单均值，也不包含收益Top。
- 正式保存预测的测试X非涨停Top：共157,180个股票—日期出现次数，价格无效占27.812699%；非涨停候选池为2.849865%。2025 Top≈26.967880%，2026≈29.848577%。价格无效并不等于已知测试Y缺失。
- 2024 PJ_a080对P_a080：综合分+0.010010720671；IC贡献+0.000005167216，收益贡献−0.001444303222，换手贡献+0.011449856677。2024的1、2、9、10月相对该对照退化。这里是分项会计分解，不是因果归因。
- 测试至少一列特征NaN为5.482558%；2024历史X为4.907491%；实际训练为6.507723%。测试illiquidity有限值中位数≈0.01116660563275218，2024≈0.02534625306725502。封存float32两中心值的float64分位数与float32中位数舍入可能差半个ULP，不能用过严的小数近似比较误报技术错误。

官方持有隐藏Y文件不保证每条Y有限；官方源码明确排除IC/收益池的Y NaN，却保留换手池的Y NaN。不能声称这些行必然损害IC/收益，也不能声称低换手证明有效样本预测稳定。

## 六、唯一固定历史诊断对照

候选ID：`INVALID_BELOW_VALID_PJ080_D1`；角色：`diagnostic_only`，不得正式采纳或导出测试文件。

使用每年**已保存、已平滑的PJ_a080预测** `p`，不使用原预测重新平滑。按原键对齐该年原始X，计算第五节价格有效标记 `V`。

对每个市场日期t：

```
m_t = min(float64(p_i) for all rows i on date t with V_i=True)
q_i = float64(p_i)                         if V_i=True
q_i = nextafter(float64(m_t), -infinity)   if V_i=False
```

- 计算m_t时包含所有价格有效行，不按涨停或Y缺失过滤；规则只依赖当日X和当日保存预测。
- 价格有效行逐值不变；所有价格无效行置于当天全部价格有效预测之下。
- 规则无可调权重、分位点、epsilon、标准化或其他候选；不可看结果后换公式。
- 若某日无价格有效行或nextafter结果非有限，记录该边界，停止该对照并报告，不另造回退公式。mock测试覆盖这些失败路径。
- 保留该年全部键与原行序；记录变更行数、各日标记、变更前后值与哈希。CSV仅三列、%.17g、round_trip逐值一致，默认解析误差≤1e-15。
- 输出只用 `diagnostic_predictions.parquet/csv` 等明确名称，不使用 `submission.csv`，包括合成测试文件。
- `nextafter`的一ULP差异可能被官方默认CSV解析压成平局。存储/round_trip公式验收与官方默认解析排序分别记录；如出现相关平局，沿用官方处理并披露，不追加epsilon、向下多步或其他新规则。

这是测试排名/评分池机制的诊断变换，并非新训练模型或校准后的经济收益预测；在价格有效候选足够时，它应减少无效Top，但不保证官方分提高。若收益池排除的历史行恰好包括所有改动行，IC/收益理论上应不变；必须实际逐值、逐集合和官方评分复核，不预先写结果。问题来自已经查看的测试X描述性统计；该对照也是额外探索，不能把它包装成独立的盲测确认。

在任何评分前冻结上述规则和四年输入。2021—2023是反复使用的开发集；2024也已经查看，本轮只作事先固定的受限诊断，不用它选择规则。即便开发对照较差，也完成事先登记的2024诊断，不增加候选或重选；它的目的在于了解机制，而非筛选新权重。

对原PJ_a080四年重核4项，再对诊断对照计算4项，交叉核验未修改的官方evaluate.py与独立算法。历史truth只能使用被run_index登记的原始历史标签；评分保留原精度，不用float32训练标签替代；不要覆盖或重建旧truth目录。

保存年度、逐日及月度指标，分别报告IC、年化超额、官方换手、综合分、价格有效换手辅助诊断；负年/负月、Top成员变化、缺失标签/价格状态构成及分项代价全部披露。不得把价格有效评分替代官方评分，也不得把仅换手Top的变化解释成收益Top效果。

## 七、什么算通过，以及如何给出决策

技术验收必须基于实际文件和独立计算：

- 数据、冻结、模型、预算、旧预测/交接哈希与版本链一致；分支和HEAD不变，保留已有未提交改动。
- 键唯一、完整、无丢行，X-only门槛和对照公式正确；有效行精确不变，改动行不依赖Y；有限值和CSV一致性通过。
- 原4年基线复现与已有指标、官方算法一致，容差固定1e-12；对照官方/独立计算一致。
- 同日期同pred平局按官方源实现处理，不自行换排序/打破平局；独立Top核验需兼容官方实际排序结果。
- 重算统计完整保存，不把聊天近似数值当作证据；特征语义比较有限值精确、NaN位置一致，文件完整性另核SHA256。
- 专项mock测试、预算和调用核查通过；真实/合成fit与模型predict均0。

本轮不进行正式方案选择，没有“无效Top下降就通过”的性能门槛，也不临时编造漂移否决阈值。技术通过与诊断对照改善分别给出结论；平均分高而有负年、负月、IC或收益代价时不能只按均分推荐。

按证据区分：

1. **实现/合同错误**：明确文件、函数、触发条件、影响及能复现的证据。若影响旧模型/正式预测，需要额外训练/推断或改变冻结，则本轮停止于具体修复说明和必要授权范围，不擅自修补旧文件或恢复预算。
2. **评分口径与样本构成风险**：规则符合官方，但历史换手收益可能受缺失样本影响；对照只能验证特定改变的结果，不能证明隐藏效果。
3. **不可消除的证据限制**：没有隐藏Y、已反复查看开发/2024、测试X漂移；不能通过更多搜索恢复盲测，也不能保证成绩。

给出明确决策：是否存在上传前必须解决的技术阻断；是否有充分证据申请下一步修改；是否应保留原PJ_a080及当前submission；哪些报告结论必须收敛。即使对照更好，也只提出有数据支持的下一步请求，当前冻结、主备和正式文件不变。

## 八、产物、验收与停止

建议新增：

```
C:\fintechathon\configs\final_risk_review_v1.json
C:\fintechathon\scripts\run_final_risk_review_v1.py
C:\fintechathon\scripts\audit_final_risk_review_v1.py
C:\fintechathon\tests\test_final_risk_review_v1.py
C:\fintechathon\docs\final_risk_review_v1\RESULTS.md
C:\fintechathon\docs\final_risk_review_v1\HANDOFF.json
C:\fintechathon\artifacts\final_risk_review_v1\<UTC时间戳>\
```

新运行目录独占创建，不覆盖；保留首次失败与每次修复。测试使用本轮新建且此前不存在的任务临时目录，不复用/删除已有pytest临时目录，不更改系统权限。不要跑包含fit/predict的旧测试。

至少保存：登记/固定合同/源哈希/环境/Git状态；调用防护与预算账本；各交叉表、Top构成、训练与预测总体对齐、漂移表；四份历史诊断预测及逐值改动追踪；8项年度逻辑评分、分项与负年/月表；官方/独立一致性；专项tests_result；独立acceptance；run_index；资源耗时/峰值RSS/磁盘；失败修复；受保护文件前后核验；更新前进度/README版本链。

本任务已存在的EXECUTION_PROMPT.md保留为授权原文，纳入来源哈希，不覆盖它。新HANDOFF明确记录：

```
risk_review_completed=true
technical_accepted=<实际验收结果>
new_fits=0
new_model_predictions=0
historical_diagnostic_prediction_sets=<实际数，不超过4>
annual_logical_scores=<实际数，不超过8>
primary_changed=false
official_test_predictions_changed=false
submission_changed=false
submission_uploaded=false
competition_report_completed=false
```

正式submission的原SHA256、本轮前后最终训练账本fit=1/predictions=2、主备冻结、旧来源和Git保护不变。旧accepted与本轮结论保留各自阶段含义。更新进度前后都核验版本链接，不改旧index。

最终简洁报告：实际验证结果、是否发现技术错误、无效样本富集的可确认机制/不确定性、固定对照各年度与分项得失、最差年/月、漂移和筛选局限、是否阻断上传、是否保留当前方案、资源与失败、RESULTS/HANDOFF路径。完成后停止；不自动进入训练、替换方案、正式导出、上传或比赛报告制作。
