# 第一版正式十特征 LightGBM Baseline 审查交接报告

## 1. 给新对话的审查任务

请在本地仓库 `C:\fintechathon` 中对提交 `d8af845` 做一次严格、只读的代码与结果审查。

审查要求：

- 先阅读 `AGENTS.md` 和 `AGENTS.override.md`，不得修改文件、提交或 push。
- 必须检查实际代码、测试和实验摘要，不要只复述本报告。
- 重点检查特征公式、分组与排序、窗口方向、索引对齐、时间切分、训练筛选、验证覆盖、评分口径和可重复性。
- 如发现问题，请给出文件、行号、影响范围及复现依据；没有证据时不要推测为已确认问题。
- 将“已确认正确”“已确认缺陷”“仍无法验证”分开报告。

建议首先执行：

```powershell
cd C:\fintechathon
git branch --show-current
git status --short --branch
git show --stat --oneline d8af845
git diff f66dc57..d8af845
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\run_lightgbm_baseline.py
```

## 2. 审查对象与版本边界

当前正式 baseline 提交：

```text
d8af845 baseline: add ten-feature LightGBM baseline v1
```

它建立在原始字段工程链路检查版本之上：

```text
f66dc57 baseline: add reproducible raw-field LightGBM pipeline
```

核心文件：

| 文件 | 作用 |
|---|---|
| `src/features/baseline_v1.py` | 十个特征的唯一实现与字段合同 |
| `scripts/run_lightgbm_baseline.py` | 数据读取、特征生成、冻结切分、训练、预测、评分和摘要输出 |
| `tests/test_lightgbm_baseline.py` | 特征数值、防泄漏、异常输入和评分往返测试 |
| `artifacts/lightgbm_baseline_v1/summary.json` | 第二次完整正式运行的机器可读结果 |

未修改的团队公共接口：

- `src/validation/splits.py`
- `src/metrics/official.py`
- `src/validation/submission.py`
- `configs/splits.yaml`
- `scripts/run_baseline_2024.py`

## 3. 数据合同

本次运行只读：

```text
C:\fintechathon\赛题五\赛题五数据\训练集_clean.csv
```

实际读取结果：

| 项目 | 数值 |
|---|---:|
| 文件大小 | 959,043,633 bytes |
| 行数 | 7,900,350 |
| 股票数 | 4,650 |
| 交易日数 | 1,699 |

读取时使用明确 dtype：主键日期为 `int32`，原始连续字段和标签为 `float32`，标记字段为 `int8`，股票代码为分类类型。脚本检查：

- 必要字段全部存在；
- `(ts_code, trade_date)` 无缺失且唯一；
- `flag_limit_up`、`flag_limit_down`、`is_price_valid`、`is_trainable` 只包含 0/1；
- 特征计算输入按 `ts_code, trade_date` 排序；
- 特征计算前后行数和索引完全一致。

## 4. 十个正式特征合同

所有 `shift` 和 `rolling` 均在单只股票内部计算，不允许跨股票共享历史。

| 特征 | 固定定义 |
|---|---|
| `ret_1d` | `close / close.shift(1) - 1` |
| `ret_5d` | `close / close.shift(5) - 1` |
| `ret_20d` | `close / close.shift(20) - 1` |
| `gap_1d` | `open / close.shift(1) - 1` |
| `intraday_ret` | `close / open - 1` |
| `high_low_range` | `high / low - 1` |
| `volatility_5d` | `ret_1d.rolling(5, min_periods=5).std(ddof=1)` |
| `volatility_20d` | `ret_1d.rolling(20, min_periods=20).std(ddof=1)` |
| `volume_ratio_5d` | `vol / vol.shift(1).rolling(5, min_periods=5).mean()` |
| `amount_ratio_5d` | `amount / amount.shift(1).rolling(5, min_periods=5).mean()` |

关键语义：

- 收益、隔夜收益和滚动波动率只使用当日及历史数据。
- 波动率窗口包含当日已经可知的 `ret_1d`。
- 量比和额比的五日均值明确排除当日值，只使用此前五条记录。
- 历史不足、原始数据缺失、零分母以及计算产生的正负无穷均保留或转换为 `NaN`。
- 不对价格、成交量、成交额或生成特征做填补。
- 输出严格为十列 `float32`，不修改输入 DataFrame。

模型明确排除：

```text
ts_code, trade_date,
open, high, low, close, vol, amount,
flag_limit_up, flag_limit_down,
y_ret_1d, is_price_valid, is_trainable, row_id
```

这些原始字段仍可用于主键、标签、训练筛选、评分或生成特征，但不会直接进入 LightGBM。

## 5. 时间、训练与预测合同

脚本通过团队冻结接口调用：

```python
split = get_split(split_name)
train_period, valid_mask = split.masks(panel)
```

实际训练掩码固定为：

```python
train_period & (panel["is_trainable"] == 1) & panel["y_ret_1d"].notna()
```

额外检查训练标签全部有限。`is_trainable` 不参与验证期过滤。

| 切分 | 训练日期 | Purge | 验证日期 | 实际训练样本 | 验证预测 |
|---|---|---|---|---:|---:|
| `primary_2023` | 20180102–20221229 | 20221230 | 20230103–20231229 | 4,597,785 | 1,125,300 |
| `oos_2024` | 20180102–20231228 | 20231229 | 20240102–20241231 | 5,659,954 | 1,125,300 |

完整面板先做单向历史特征，再应用切分。这允许验证期首日使用此前已经发生的历史 X，包括 purge 日的 X；特征实现不能读取未来行。

验证期不按标签、质量标记或特征缺失删行。每个验证键都产生一条有限预测，覆盖率为 100%。预测、真值和涨停标记通过 `ts_code, trade_date` 键对齐，不依赖当前行号完成评分。

## 6. LightGBM 固定配置

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

没有随机切分、参数搜索、早停选择或分数驱动调参。

## 7. 评分合同与精度处理

每套切分先调用：

```python
score_official(pred, truth, x, return_details=True)
```

随后将相同预测、真值和涨停标记写入临时目录，调用仓库内未修改的 `赛题五/evaluate.py`。所有标量差异必须不超过 `1e-12`，否则运行失败且不写正式摘要。

严格验收首次发现过一个精度问题：`float32` 标签经默认 CSV 文本往返后，年化指标与内存评分相差约 `6.2e-9`。修复方式不是放宽阈值，而是：

- 评分真值显式转为 `float64`；
- 预测和真值以 `float_format="%.17g"` 写入官方校验夹具；
- 新增官方 CSV 往返回归测试。

修复后的最大绝对差异：

| 切分 | 本地与官方最大标量差异 |
|---|---:|
| `primary_2023` | `5.967448757360216e-16` |
| `oos_2024` | `3.469446951953614e-16` |

## 8. 正式运行结果

这些指标用于确认当前正式 baseline 的可复现结果，不构成投资含义或最低性能承诺。

| 切分 | Rank IC | 年化超额收益 | Turnover | 综合得分 |
|---|---:|---:|---:|---:|
| `primary_2023` | 0.02113411935065089 | 0.07055126501262099 | 0.4824181889098113 | 0.18489357057110323 |
| `oos_2024` | 0.05252313845234250 | 0.07403187626707203 | 0.7106195201138428 | 0.13003296222690577 |

完整流程连续运行两次，两次样本数、全部评分指标和预测数组 SHA-256 完全一致：

| 切分 | 预测 SHA-256 |
|---|---|
| `primary_2023` | `a494f88497430041c51febc0cdaebca45a93c187697412dad81be9c0c7cf890d` |
| `oos_2024` | `7b30a851843a2ec5261fd6d00b1035f5e04bad6bf8d6a99c53c0a9a97f79e254` |

第二次完整运行耗时 `35.8292` 秒，进程峰值 RSS 为 `1771.71 MB`。

## 9. 特征缺失与有限性结果

| 特征 | 缺失行数 | 无穷值行数 |
|---|---:|---:|
| `ret_1d` | 1,146,344 | 0 |
| `ret_5d` | 1,167,623 | 0 |
| `ret_20d` | 1,239,210 | 0 |
| `gap_1d` | 1,146,344 | 0 |
| `intraday_ret` | 1,139,980 | 0 |
| `high_low_range` | 1,139,980 | 0 |
| `volatility_5d` | 1,171,741 | 0 |
| `volatility_20d` | 1,266,119 | 0 |
| `volume_ratio_5d` | 1,204,393 | 0 |
| `amount_ratio_5d` | 1,204,393 | 0 |

这些缺失包括未上市、停牌、原始字段缺失、滚动历史不足和零分母情形。训练过程不因特征缺失额外删除样本，由 LightGBM 原生处理缺失值。

## 10. 已执行测试

提交后执行：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  src\features\baseline_v1.py `
  scripts\run_lightgbm_baseline.py `
  tests\test_lightgbm_baseline.py

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

结果：

- 编译检查通过；
- 19 项测试全部通过；
- `pip check`通过。

新增测试覆盖：

- 十个模型字段的精确合同；
- 单股票公式数值；
- 股票边界不共享历史；
- 历史不足和零分母；
- 量比、额比分母排除当日；
- 改动未来输入不影响过去特征；
- 未排序输入、重复主键和缺少字段明确失败；
- `is_trainable`只过滤训练样本；
- 官方评分 CSV 往返精度。

## 11. 建议重点复核的风险点

新对话应优先检查以下位置，而不是仅看测试是否通过：

1. `groupby().rolling()`产生的多级索引在降级、`reindex`后是否始终与原面板索引一一对应。
2. `ret_5d`和`ret_20d`的 `shift` 是否严格在股票组内执行。
3. 波动率窗口是否符合“包含当日收益”的约定，量比与额比是否严格排除当日成交数据。
4. 完整面板先计算特征再切分是否只使用历史方向，且没有任何验证期统计量参与训练期特征。
5. 训练样本是否只由冻结训练期、`is_trainable == 1`和非空标签决定。
6. 验证样本是否完整保留，模型预测是否按键与真值对齐。
7. `float64`真值及 `%.17g` CSV 写法是否确实使本地包装器和官方脚本在同一数值输入上比较。
8. 预测哈希是否基于固定顺序的 `float64`预测数组，重复运行是否仍然一致。
9. 新增后续特征时，是否只扩展正式特征合同，而没有把主键、标签、质量标记或未来数据带入模型。

## 12. 当前尚未覆盖的范围

以下内容不应被误报为已经完成：

- 尚未为 `测试集_X.csv`生成最终比赛 submission。
- 尚未接入 B/C 后续开发的其他正式特征。
- 尚未做参数搜索、模型选择或特征筛选。
- 尚未验证新增特征之后的内存峰值、预测哈希和指标。
- 当前结果只证明这十个特征版本在两套冻结验证切分上按既定合同运行，并不能保证未来修改自动正确。

以后每次增加特征，都应重新执行本报告第 10 节的测试和两次完整运行，并重新记录预测哈希、官方评分差异、特征缺失统计与资源消耗。
