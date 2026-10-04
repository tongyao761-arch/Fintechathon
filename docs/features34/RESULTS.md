# 34特征阶段最终结果与交付

验收日期：2026-10-05。当前第1—6步完成；本阶段停止。

在已核验的2021—2024同年官方评分范围内，两个冻结候选均优于十特征。沿用2023开发冻结的顺序，27列为研究主候选，31列为备选；没有按2024重新排序或选择。支持用冻结候选开展后续研究比较，但本阶段不覆写十特征baseline，也不把这组证据包装成已证明未来表现、实盘能力或正式比赛最终模型。

## 1. 范围与已执行实验

固定十特征骨架、24列新增合同、LightGBM参数、种子、y_ret_1d、训练资格、数据和官方评分；完整历史X面板先算特征，再投影列及筛样本，全部验证键保留。未调参、换模型、平滑、加新列或生成最终比赛submission。2023主筛，2021/2022开发验证，2024仅复核已冻结候选；2024此前已查看，不能称未接触的盲测。

|步骤|实际实验|新增训练|证据|
|---|---|---:|---|
|1|10列2023/2024再现、34列2023框架验证|3|artifacts/features34_step1/|
|2|2023固定14组；复用10/34两项|12|features34_step2/comparison.csv、summary.json|
|3|10、34、lean31、lean27、lean23的三个开发年；复用三项2023|12|features34_step3/comparison.csv、run_index.json|
|4|full34/lean31背景45种单删×三个开发年|135|features34_step4/deletion_results.csv、run_index.json|
|4修订|观察旧单删后的得分导向修订；两版联合删除×三年；无回补|6|features34_step4_revision/joint_results.csv、frozen_candidates.json|
|5|事前冻结27/31列，各复核2024一次|2|features34_step5/FROZEN_CANDIDATES.json、run_index.json|
|6|仅重复两冻结版本的2023和2024；10列只读模型/评分回归|4|features34_step6/run_index.json、acceptance.json|

本阶段合计174次新增全量模型运行；既有实验和失败记录保留。第6步仅新增4次，未以重复结果再次筛选。第4步修订不是原135项实验的事前登记；探索性修订、重复使用开发年份和多重选择风险均保留。

## 2. 跨年份对照

|版本|year|feature_count|ic_mean|annual_excess|mean_turnover|final_score|final_score_minus_baseline10|
|---|---|---|---|---|---|---|---|
|冻结31列|2021|31|0.077344426|0.535815595|0.316522414|0.396725725|0.159159983|
|冻结27列|2021|27|0.077189771|0.507022290|0.313435111|0.388952062|0.151386320|
|十特征|2021|10|0.035106919|0.056899852|0.311823269|0.237565742|0.000000000|
|完整34|2021|34|0.075466202|0.507335903|0.318525974|0.386829460|0.149263717|
|冻结31列|2022|31|0.090022936|0.466249634|0.235996463|0.405085126|0.122912849|
|冻结27列|2022|27|0.094236866|0.504472570|0.240834802|0.416786077|0.134613800|
|十特征|2022|10|0.042714578|0.102047998|0.218426514|0.282172276|0.000000000|
|完整34|2022|34|0.089826173|0.457337313|0.232165614|0.403481979|0.121309703|
|冻结31列|2023|31|0.066000245|0.250913510|0.505328145|0.250075707|0.065182177|
|冻结27列|2023|27|0.071825386|0.287592350|0.505058203|0.263490399|0.078596868|
|十特征|2023|10|0.021134020|0.070551265|0.482418189|0.184893531|0.000000000|
|完整34|2023|34|0.066816599|0.230706969|0.507452243|0.243703058|0.058809527|
|冻结31列|2024|31|0.092694632|0.452954843|0.746383584|0.249049231|0.119016132|
|冻结27列|2024|27|0.094958854|0.552933946|0.812392244|0.260146052|0.130112953|
|十特征|2024|10|0.052523479|0.074031877|0.710619520|0.130033099|0.000000000|

完整34的2024未运行，不能填分数或声称联合删除在2024相对完整34仍有优势。年度分差均对同年十特征、同训练截止日计算；2021/2022训练窗起点20180102，逐年扩展。各年切分、资格数、预测行数和目录均见annual_comparison.csv；2021验证1,129,950键，2022—2024各1,125,300键。

2023：27列0.263490398718、31列0.250075707180、完整34为0.243703057508、十特征0.184893530509。2024：27列0.260146051905、31列0.249049230743、十特征0.130033098566（完整数值以CSV为准）。27列对十特征四年最小优势+0.078596868；31列+0.065182177。候选间优势随年份变化，未证明27列普遍优于31列。

## 3. 新增24列的保留、删除和不确定

名单严格依背景成立，沿用第4步修订规则：单删后减同年原背景的三年分差均<−0.001为保留，均>+0.001为删除，否则不确定。阈值、统计区间/月度风险按原记录，不由2024重判。模型实际移出列与确定删除是两个不同结论。

- full34背景：保留5列：bias_60d, close_location, flag_limit_down, log_mean_amount_20d, ret_1d_rank_pct。
- full34背景：删除0列：无。
- full34背景：不确定19列：amount_ratio_20d, amount_ratio_5d_rank_pct, bias_20d, body_ratio, flag_limit_up, limit_down_count_5d, limit_up_count_5d, log_mean_amount_20d_rank_pct, lower_shadow, price_position_20d, price_position_60d, ret_10d, ret_20d_rank_pct, ret_2d, ret_5d_rank_pct, ret_60d, upper_shadow, volatility_20d_rank_pct, volatility_60d。
- lean31背景：保留4列：bias_60d, close_location, flag_limit_down, ret_1d_rank_pct。
- lean31背景：删除2列：lower_shadow, ret_5d_rank_pct。
- lean31背景：不确定18列：amount_ratio_20d, amount_ratio_5d_rank_pct, bias_20d, body_ratio, flag_limit_up, limit_down_count_5d, limit_up_count_5d, log_mean_amount_20d, log_mean_amount_20d_rank_pct, price_position_20d, price_position_60d, ret_20d_rank_pct, upper_shadow, volatility_20d_rank_pct, volatility_60d, ret_2d, ret_10d, ret_60d。

下表括号内为单删年度分差（2021/2022/2023）；最后两列表示实际进入27/31列模型。原lean31缺席A三列没有直接单删证据，继续不确定。

|新增列|full34结论（单删分差）|lean31结论（单删分差）|27列输入|31列输入|
|---|---|---|---|---|
|ret_2d|不确定（+0.000163/-0.005922/-0.000628）|不确定（无直接证据）|否|是|
|ret_10d|不确定（+0.002269/-0.000003/+0.002462）|不确定（无直接证据）|否|否|
|ret_60d|不确定（+0.011244/-0.004667/+0.002173）|不确定（无直接证据）|否|是|
|close_location|保留（-0.003945/-0.008240/-0.002015）|保留（-0.005607/-0.004246/-0.002755）|是|是|
|upper_shadow|不确定（+0.001736/-0.004287/-0.000186）|不确定（+0.001353/-0.002993/+0.003967）|是|是|
|lower_shadow|不确定（+0.005198/-0.001582/+0.005564）|删除（+0.008916/+0.012347/+0.002501）|否|是|
|body_ratio|不确定（+0.004511/+0.002078/+0.000067）|不确定（+0.004515/-0.005542/+0.002721）|是|是|
|bias_20d|不确定（+0.002972/-0.000457/-0.003679）|不确定（+0.003254/-0.005320/+0.002892）|是|是|
|bias_60d|保留（-0.003864/-0.008968/-0.002369）|保留（-0.001585/-0.001686/-0.003335）|是|是|
|price_position_20d|不确定（-0.002282/-0.007054/+0.009819）|不确定（+0.006578/-0.000745/+0.018773）|否|是|
|price_position_60d|不确定（+0.004700/-0.000338/+0.000568）|不确定（+0.007374/-0.000714/+0.000972）|是|是|
|amount_ratio_20d|不确定（-0.001463/-0.004999/+0.006516）|不确定（-0.005019/+0.001649/+0.012620）|是|是|
|log_mean_amount_20d|保留（-0.003220/-0.002309/-0.001848）|不确定（+0.000180/+0.000453/-0.003315）|是|是|
|volatility_60d|不确定（+0.000932/-0.003086/+0.002188）|不确定（+0.000726/+0.006411/-0.003206）|是|是|
|flag_limit_up|不确定（-0.005786/+0.006097/-0.002610）|不确定（-0.006035/+0.005789/-0.000901）|是|是|
|flag_limit_down|保留（-0.160942/-0.201298/-0.109652）|保留（-0.162151/-0.189441/-0.109885）|是|是|
|limit_up_count_5d|不确定（+0.007698/+0.000407/+0.003259）|不确定（+0.007500/-0.004189/-0.003028）|是|否|
|limit_down_count_5d|不确定（+0.006938/-0.005398/-0.000728）|不确定（+0.003087/+0.004226/-0.002527）|是|是|
|ret_1d_rank_pct|保留（-0.014828/-0.039583/-0.012186）|保留（-0.021567/-0.028361/-0.016229）|是|是|
|ret_5d_rank_pct|不确定（+0.004651/-0.003504/+0.002174）|删除（+0.002014/+0.001838/+0.002363）|否|是|
|ret_20d_rank_pct|不确定（-0.006661/-0.010677/+0.004077）|不确定（-0.002667/-0.000290/+0.001146）|是|是|
|amount_ratio_5d_rank_pct|不确定（-0.004008/-0.001347/+0.001870）|不确定（+0.002838/-0.004355/-0.000981）|是|是|
|volatility_20d_rank_pct|不确定（-0.000354/+0.002036/+0.001824）|不确定（+0.001887/+0.000074/+0.005328）|否|否|
|log_mean_amount_20d_rank_pct|不确定（+0.000129/-0.004995/-0.003967）|不确定（-0.000950/-0.002428/-0.000395）|是|是|

48条完整证据及实验ID见artifacts/features34_step6/feature_decisions.csv；本步逐条核对原单删目录摘要哈希、配置、模型列和三个开发年度分差，未重训单删。对应原run_index在features34_step4/，修订single_decisions.csv/probe_evidence.csv在features34_step4_revision/。条件性分数同向不等于统计显著；块区间跨零和旧诊断护栏失败仍保留在逐列证据中。没有可不加背景限制地宣布24列全体必需或可删的依据。

## 4. 最终冻结候选的输入、公式与命令

第5步冻结时刻：2026-10-04T23:30:44.643992+08:00。原冻结SHA-256：`12790b50793fb4a3faf5b0df0dfe9cdf937f90c6b68f63d064b1f37c26542063`。第6步registration.json引用原冻结并保存本步新增入口源码；此前实际执行的全部源码、配置、公式均逐文件保持哈希。

### S4R_lean31_minus4（27列）

按模型精确顺序：

```text
ret_1d,
ret_5d,
ret_20d,
gap_1d,
intraday_ret,
high_low_range,
volatility_5d,
volatility_20d,
volume_ratio_5d,
amount_ratio_5d,
close_location,
upper_shadow,
body_ratio,
bias_20d,
bias_60d,
price_position_60d,
amount_ratio_20d,
log_mean_amount_20d,
volatility_60d,
flag_limit_up,
flag_limit_down,
limit_up_count_5d,
limit_down_count_5d,
ret_1d_rank_pct,
ret_20d_rank_pct,
amount_ratio_5d_rank_pct,
log_mean_amount_20d_rank_pct
```

### S4R_full34_minus3（31列）

按模型精确顺序：

```text
ret_1d,
ret_5d,
ret_20d,
gap_1d,
intraday_ret,
high_low_range,
volatility_5d,
volatility_20d,
volume_ratio_5d,
amount_ratio_5d,
ret_2d,
ret_60d,
close_location,
upper_shadow,
lower_shadow,
body_ratio,
bias_20d,
bias_60d,
price_position_20d,
price_position_60d,
amount_ratio_20d,
log_mean_amount_20d,
volatility_60d,
flag_limit_up,
flag_limit_down,
limit_down_count_5d,
ret_1d_rank_pct,
ret_5d_rank_pct,
ret_20d_rank_pct,
amount_ratio_5d_rank_pct,
log_mean_amount_20d_rank_pct
```

原始X列：ts_code、trade_date、open、high、low、close、vol、amount、flag_limit_up、flag_limit_down。键只用于股票历史及当日截面计算；is_price_valid由当日原始float64 OHLC推导，仅作F排名股票池/诊断。y_ret_1d、is_trainable、标签缺失状态不进入特征。

C/O/H/L/A/V为当日收盘/开盘/最高/最低/成交额/成交量。以下公式覆盖两候选所用的全部列（排除状态见表）；未进入模型的排名源也保留中间计算。

|列|完整公式|27列|31列|
|---|---|---|---|
|ret_1d|close / close.shift(1) - 1|是|是|
|ret_5d|close / close.shift(5) - 1|是|是|
|ret_20d|close / close.shift(20) - 1|是|是|
|gap_1d|open / close.shift(1) - 1|是|是|
|intraday_ret|close / open - 1|是|是|
|high_low_range|high / low - 1|是|是|
|volatility_5d|ret_1d.rolling(5, min_periods=5).std(ddof=1)|是|是|
|volatility_20d|ret_1d.rolling(20, min_periods=20).std(ddof=1)|是|是|
|volume_ratio_5d|vol / vol.shift(1).rolling(5, min_periods=5).mean()|是|是|
|amount_ratio_5d|amount / amount.shift(1).rolling(5, min_periods=5).mean()|是|是|
|ret_2d|C / C.shift(2) - 1|否|是|
|ret_10d|C / C.shift(10) - 1|否|否|
|ret_60d|C / C.shift(60) - 1|否|是|
|close_location|(C-L)/(H-L)|是|是|
|upper_shadow|(H-max(O,C))/C|是|是|
|lower_shadow|(min(O,C)-L)/C|否|是|
|body_ratio|(C-O)/(H-L)|是|是|
|bias_20d|C/close.rolling(20, min_periods=20).mean()-1|是|是|
|bias_60d|C/close.rolling(60, min_periods=60).mean()-1|是|是|
|price_position_20d|(C-low.rolling(20).min())/(high.rolling(20).max()-low.rolling(20).min()); min_periods=20|否|是|
|price_position_60d|(C-low.rolling(60).min())/(high.rolling(60).max()-low.rolling(60).min()); min_periods=60|是|是|
|amount_ratio_20d|A/amount.shift(1).rolling(20, min_periods=20).mean()|是|是|
|log_mean_amount_20d|log1p(amount.rolling(20, min_periods=20).mean())|是|是|
|volatility_60d|baseline ret_1d.rolling(60, min_periods=60).std(ddof=1)|是|是|
|flag_limit_up|raw current flag_limit_up|是|是|
|flag_limit_down|raw current flag_limit_down|是|是|
|limit_up_count_5d|flag_limit_up.rolling(5, min_periods=5).sum()|是|否|
|limit_down_count_5d|flag_limit_down.rolling(5, min_periods=5).sum()|是|是|
|ret_1d_rank_pct|same-date ret_1d.rank(method='average', pct=True); price valid and finite source only|是|是|
|ret_5d_rank_pct|same-date ret_5d.rank(method='average', pct=True); price valid and finite source only|否|是|
|ret_20d_rank_pct|same-date ret_20d.rank(method='average', pct=True); price valid and finite source only|是|是|
|amount_ratio_5d_rank_pct|same-date amount_ratio_5d.rank(method='average', pct=True); price valid and finite source only|是|是|
|volatility_20d_rank_pct|same-date volatility_20d.rank(method='average', pct=True); price valid and finite source only|否|否|
|log_mean_amount_20d_rank_pct|same-date log_mean_amount_20d.rank(method='average', pct=True); price valid and finite source only|是|是|

shift/rolling均在单股票有序样本内计算，不补日，完整窗口min_periods=n；除历史额比分母外窗口均含当日，标准差ddof=1。零分母、缺输入、历史不足和非有限输出变NaN，不填零。十特征直接调用冻结实现，保留原float32运算；新增列对加载器float32 X转换float64中间运算，最后输出有限float32；F排名使用最终float32源值，在当日OHLC有效且源有限股票池中average/pct=True，其他行NaN。公式表不能替代这些精度/窗口约定。

模型和参数原样保存如下；训练标签为有限float32，评分标签为原始文本提取后解析的float64。训练资格为当日价格有效且标签非空并可转有限float32，不改样本规则。

```json
{
  "objective": "regression",
  "boosting_type": "gbdt",
  "n_estimators": 100,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "max_depth": -1,
  "min_child_samples": 100,
  "subsample": 1.0,
  "subsample_freq": 0,
  "colsample_bytree": 1.0,
  "reg_alpha": 0.0,
  "reg_lambda": 1.0,
  "random_state": 20260927,
  "data_random_seed": 20260927,
  "feature_fraction_seed": 20260927,
  "bagging_seed": 20260927,
  "deterministic": true,
  "force_col_wise": true,
  "n_jobs": -1,
  "verbosity": -1
}
```

在C:\fintechathon的PowerShell中，实际包装器调用的准确子进程命令：

```powershell
.\.venv\Scripts\python.exe -B scripts/run_features34_step6.py prepare
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m pip check
.\.venv\Scripts\python.exe -B scripts/run_features34_step6.py run
.\.venv\Scripts\python.exe -B artifacts/features34_step6/audit_compat.py
```

prepare仅创建一次注册，目录存在则拒绝覆盖。run按run_index复用本步已成功的四项重复，不再增加训练；最终使用audit_compat.py只读重核模型、预测、评分及诊断并重新写本步验收；原脚本audit首次空表错误已留档，不使用该已知失败路径重验收。兼容入口只为baseline汇总提供一个真实已冻结比较项，输出仍取baseline行，不新建候选、不改任何预测或分差。实执行由artifacts/features34_step6_checks/execute.py包装，准确子进程命令、退出码、耗时、日志位置在commands/；测试和依赖记录为tests_result.json、dependency_check.json。完整日志保留checks/。注册包含解释器及依赖版本，不保证其他环境逐值一致。

## 5. 官方一致性、覆盖率、有限性与可复现性

|candidate|split|train_samples|valid_prediction_rows|prediction_coverage|official_rescore_max_abs_difference|model_reload_equal|repeat_equal|
|---|---|---|---|---|---|---|---|
|S4R_lean31_minus4|primary_2023|4597785|1125300|1|0|True|True|
|S4R_lean31_minus4|oos_2024|5659954|1125300|1|0|True|True|
|S4R_full34_minus3|primary_2023|4597785|1125300|1|0|True|True|
|S4R_full34_minus3|oos_2024|5659954|1125300|1|0|True|True|
|baseline10|primary_2023|4597785|1125300|1|0|True|只读对照|
|baseline10|oos_2024|5659954|1125300|1|0|True|只读对照|

四项重复与各自原运行的预测数组逐值一致，SHA-256一致；全部官方指标、诊断、月度/逐日CSV、模型文件、原始评分输入和特征缺失统计等切分文件哈希全部一致。每项从原始X重建特征、加载保存模型，再对全部验证键复算；检查训练资格及purge，有限预测和评分，无丢键/重键。双方重新读取同一保存CSV，官方评分所有标量最大差0（合同≤1e-12）。

十特征2023/2024仅只读重载和重评分，不增加对照训练；预测仍为冻结参考哈希。39个冻结文件、此前执行来源、阶段小型证据、既有未跟踪文件及Git config/pre-push保持哈希。106项回归、2项汇总兼容测试及pip check通过。

|版本/年份|原运行目录|本步重复目录|预测SHA-256|
|---|---|---|---|
|S4R_lean31_minus4 / primary_2023|artifacts/experiments/S4R_lean31_minus4_2023/20261004T090750752259Z_95be596d|artifacts/experiments/S6_S4R_lean31_minus4_2023/20261004T182620942911Z_24ab2629|b63426a94e5c8d1c2beb65e0d5569c57825c075f5e1a3dc9e072056afefcdc68|
|S4R_lean31_minus4 / oos_2024|artifacts/experiments/S5_S4R_lean31_minus4_2024/20261004T153205837882Z_7694eca8|artifacts/experiments/S6_S4R_lean31_minus4_2024/20261004T182650280569Z_8e2d2881|8287f594cc59e19a9a062f7f2e5da4c3647dd40989af48183b28b486da7a88ea|
|S4R_full34_minus3 / primary_2023|artifacts/experiments/S4R_full34_minus3_2023/20261004T090648724514Z_0e18c478|artifacts/experiments/S6_S4R_full34_minus3_2023/20261004T182722043263Z_7bef86a4|ec6ba42755d54bdb24b7a6daf28ca895c5743de19a612e7d2f99a3105dde1564|
|S4R_full34_minus3 / oos_2024|artifacts/experiments/S5_S4R_full34_minus3_2024/20261004T153236927517Z_ba7c2aec|artifacts/experiments/S6_S4R_full34_minus3_2024/20261004T182752251485Z_1d21e639|2c7211490e773e1981c2d8214e719d948f9aa04bf2f52dac7d9f9d7a2370c750|

新增模型运行失败0。失败共3项（模型训练失败0）：首次测试包装器在测试退出0后显示日志触发GBK UnicodeEncodeError；原日志未改，失败及耗时记在failures.json，使用-X utf8继续，不重跑已通过测试；另一次交付生成器的辅助编辑命令在PowerShell解析阶段失败，改为字面here-string后完成，不涉及训练或评分。首次独立验收已完成六项重载/重评分，但baseline单独汇总复用函数的空区间表触发KeyError；未生成成功摘要，保留失败状态/日志，增加audit_compat.py兼容入口及2项测试，只重新验收、不重训。成功总摘要仅在四项重复及独立重核通过后生成；失败历史不删除。前五步的失败记录仍在原目录，本步没有将其改写为成功。

## 6. 月度表现与缺失样本影响

完整月度180行和缺失诊断360行包含十特征、完整34及两个候选已运行年份。月度得分仅诊断；年化超额为当月日均×252，月首换手含前一个有效交易日，不平均月分替代全年。候选对同年十特征的月度概要：

|候选|年|正月数/12|最差月|最差分差|
|---|---:|---:|---:|---:|
|S4R_full34_minus3|2021|12|202110|+0.002535637|
|S4R_full34_minus3|2022|12|202206|+0.007319632|
|S4R_full34_minus3|2023|10|202311|-0.024801350|
|S4R_full34_minus3|2024|11|202409|-0.221704692|
|S4R_lean31_minus4|2021|11|202110|-0.041460863|
|S4R_lean31_minus4|2022|12|202206|+0.007932335|
|S4R_lean31_minus4|2023|10|202311|-0.062691823|
|S4R_lean31_minus4|2024|11|202409|-0.126002584|

2024两候选各11个正月；9月都落后十特征，27列−0.126002584、31列−0.221704692。联合删除相对原背景在开发三年虽均提高全年分数，仍有负月份和条件性区间跨零；详细见第4步修订报告和joint_results.csv、monthly_candidate_results.csv。

|版本|year|top_missing_label|top_invalid_price|top_candidate_all_missing|price_valid_turnover|
|---|---|---|---|---|---|
|冻结31列|2021|1.000000000|1.000000000|0.000000000|0.881444617|
|冻结27列|2021|1.000000000|1.000000000|0.000000000|0.874102104|
|十特征|2021|1.000000000|1.000000000|1.000000000|0.841112474|
|完整34|2021|1.000000000|1.000000000|0.000000000|0.877025471|
|冻结31列|2022|0.853363581|0.853300314|0.000000000|0.878293225|
|冻结27列|2022|0.848229861|0.848121402|0.000000000|0.870390059|
|十特征|2022|0.864751765|0.864697535|0.864697535|0.857092446|
|完整34|2022|0.855117000|0.855035656|0.000000000|0.876790802|
|冻结31列|2023|0.561521755|0.561396109|0.000000000|0.874939638|
|冻结27列|2023|0.561575603|0.561396109|0.000000000|0.875395071|
|十特征|2023|0.561503805|0.561396109|0.561396109|0.852760646|
|完整34|2023|0.561548679|0.561396109|0.000000000|0.875968188|
|冻结31列|2024|0.262252748|0.262089650|0.000000000|0.867215676|
|冻结27列|2024|0.156556092|0.156266140|0.000000000|0.858511940|
|十特征|2024|0.279686127|0.279332747|0.279332747|0.845878325|

换手Top占比按股票—日期入选次数加权。官方换手保留非涨停缺标签行，2021 Top几乎全部缺标签/价格无效，2022仍约85%，不可解释成可交易股票低换手。候选含原始标记使全候选输入缺失占比为0，也不代表价格有效。价格有效换手依据当日OHLC和涨停筛选，不依据未来标签，只作诊断。

候选四年低换手贡献相对十特征均下降；2024的27列IC/收益/低换手贡献增量分别+0.016974150/+0.143670621/−0.030531817，31列+0.016068461/+0.113676890/−0.010729219。因此不支持把通过缺失排名降低换手作为主要涨分来源，正增益由IC和收益支持；不排除缺失样本确实影响官方Top集合。

第5步Top成员变化证据：2024缺标签成员变化占总变化27列9.3371%、31列1.5639%，价格无效分别9.3076%/1.5293%；2021的换手Top成员变化全在缺失/无效子集。集合对称差是描述性证据，不能精确分离换手的因果贡献，也不能把收益Top/IC有效标签集合与换手Top集合混为一谈。完整月度缺失和价格有效换手见monthly_missing_diagnostics.csv，成员变化与配对区间继续引用features34_step5/top_membership_comparison.csv、paired_intervals.csv，不追加干预或候选。

## 7. 是否支持替代十特征及局限

支持：在固定官方评分、参数和资格的历史四年研究比较中，27列主候选和31列备选都比十特征有正年度增量，并完成原始输入/保存模型重核和全量重复。第6步交付的是这两个已冻结研究版本及其证据；十特征冻结代码、参考哈希、对照产物继续保留，不自动升级为比赛最终版本。

局限：开发三年经过大量内部筛选，修订规则已观察旧结果，2024也是曾查看的受限复核集；条件性20日块区间（2000次，种子20261004）不涵盖训练、选择、多重检验或未来分布风险。逐列结论依背景，联合删除不证明每列必要或普遍可删。原full34/lean31的2024未运行；9月风险、收益集中、官方缺失样本换手偏差仍存在。本阶段未验证费用、容量、可交易收益、比赛测试期或其他运行环境，也未做最终submission。

## 8. 产物与新对话接续

- 交付结论：docs/features34/RESULTS.md；当前状态：PROGRESS.md、PLAN.md；前五步报告及冻结文件不改。
- 第6步小型证据：artifacts/features34_step6/registration.json、preflight.json、run_index.json、acceptance.json、summary.json、status.json、annual_comparison.csv、monthly_comparison.csv、monthly_missing_diagnostics.csv、feature_decisions.csv、repeat_comparison.csv、tests_result.json、dependency_check.json、compat_tests_result.json、audit_compat.py、test_audit_compat.py、audit_attempt01_status.json、failures.json及commands/。
- 完整模型、预测、历史官方评分输入和逐日诊断由本步及原run_index.json索引至artifacts/experiments/；留本地，不提交大型运行产物。executed_sources/保存实际执行文件；checks/保存包装器/完整日志。
- 当前分支ivor-work，本地提交只含本阶段相关代码、配置、测试、文档及小型证据；无远程写入、无push保护变化。提交哈希由最终交付说明和git log提供，运行注册记录执行时父提交，不冒称父提交为最终代码版本。

新对话先读取实际AGENTS.md/AGENTS.override.md、RESULTS.md、PROGRESS.md、PLAN.md、features34_step6/acceptance.json、registration.json及features34_step5/FROZEN_CANDIDATES.json；从run_index检查真实文件及源码哈希。当前第1—6步已完成，原实验不是待重做清单。必须等待新步骤明确授权，不能沿用计划文字进入调参、平滑、其他特征、2024搜索或最终submission。
