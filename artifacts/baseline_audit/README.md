# Baseline 验收入口与证据

本目录保存十特征 `baseline_v1_1` 的比赛实验基线审计。放行仅针对记录的源码、原始数据和依赖快照；不保证绝对无缺陷，不证明策略显著优于随机模型或可以实现交易收益。

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_baseline.py --suite all
```

每次建立独立的 `audit/<UTC时间>_<随机ID>/`，不会覆盖旧结果。`full` 同样执行单元测试和依赖前置检查；`unit` 只跑前置检查，完整验收缺项时返回非零，不能据此放行。禁止使用 `-O` 或 `-OO` 跳过断言。

完整验收包含两次串行 `run_lightgbm_baseline.py --experiment-id baseline_audit --split all --verify-clean`，五个固定种子的 2023 日内标签打乱训练，以及必要的异常调查。模型、预测、原始标签文本提取结果和评分 CSV 留在被 Git 忽略的 `artifacts/experiments/`；小型审计报告记录其实际哈希和位置。本次不生成正式测试集提交。

每次结果应同时检查 `status.json`、`summary.json` 的 `decision` 和 `REPORT.md`。只有 `status=success` 且 `can_start_feature_experiments=true` 才放行。中断、缺少摘要、阻断失败或关键项未验证都不放行。

| 验收领域 | 独立预期与测试 | 全量证据 |
|---|---|---|
| 数据加载 | 原始行序和行数保留；唯一键、合法日期、标记、非负量额、OHLC；重建与 clean 逐值一致 | `raw_data_and_features.json`、两次运行的 migration 检查 |
| 标签 | float64 相邻交易日价格复算，误差 ≤1e-10；股票边界、缺失、末日接测试首日、目标日期不跨验证期 | `raw_data_and_features.json` 的 `labels` |
| 十特征 | 不调用 rolling/shift 的逐股票标量参考；float32 合同；容差 1e-6/1e-7，NaN 位置一致 | 同文件的 `real_feature_sample`；测试源码 `TestIndependentFeatures` |
| 分组与索引 | 不同长度、分类未使用类别、非连续索引按键一致；重复键/索引、缺列、乱序拒绝 | `unit_tests.log`；测试源码 |
| 因果性 | 历史截断和未来 X/Y 扰动不改变过去；股票隔离；验证首日接历史 | `prefixes`、`history_continuation`；合成回归测试 |
| 训练输入 | 实际 fit 捕获 X/Y/索引/特征名；验证标签与资格、purge 标签改变不影响原始预测 | `TestRawAndTraining.test_actual_fit_inputs_and_prediction_invariance` |
| 数值 | 零分母、缺失、极端值、无穷、转换溢出：特征只有限或 NaN；非有限训练标签/预测/评分拒绝 | 数值测试及两次全量摘要 |
| 覆盖和对齐 | 两个验证期完整且唯一、覆盖 100%；各输入独立乱序按键恢复，缺/多/重键拒绝 | `full_repeatability.json`；评分与导出测试 |
| 官方评分 | 手算秩相关、252 年化、Top 10%、Jaccard 及总分；29/30、99/100 等边界；不可评分明确失败 | 评分测试；全量及各诊断的 official 差异 ≤1e-12 |
| 并列与保存 | 无并列行序不影响；有并列复现官方；2023/2024 各五次行序扰动；CSV 同输入、Parquet 和模型重载 | `controls_and_ties.json`；两次全量摘要 |
| 解释与负对照 | 正负排序方向、收益/换手 Top 分开、缺失及月度分项；五种子日内打乱与异常调查 | `controls_and_ties.json`、`negative_control_investigation.json`、模型诊断文件 |
| 留档与失败 | 目录冲突、缺摘要、损坏、数据/源码运行中变化拒绝；验收边界重读关联产物哈希 | 失败路径测试、`artifact_integrity.json`、`snapshot_unchanged.json` |
| 后续推理 | 隔离夹具验证历史接测试首日、特征顺序、恢复原始键序导出 | 历史延续、实际训练及导出测试 |

实际单元测试名称与通过状态见每次 `unit_tests.log`；独立预期在 `tests/test_baseline_audit.py` 和已有测试中，不以冻结预测哈希替代公式核验。

## 缺陷和版本

`defects_20260930.json` 保留父提交 `71d7e65` 的修复前复现与文件哈希。修复原始加载器接受非法日历日期，以及有限 float64 结果转换 float32 后溢出重新产生无穷的问题。正常数据的样本、预测、评分和诊断必须继续匹配冻结 `baseline_v1_1`；不只更新参考哈希。如果后续合理修复改变正常预测或样本，应另外冻结并独立验收新版本。

## 负对照解释的限制

日内标签打乱仍保留每日收益分布和均值，训练出的函数也仍使用有信息的原始 X，因此不能要求每次随机模型 IC 恰为零。预设 3/5 模型达到 baseline 绝对平均 IC 时触发调查；该条件是诊断告警，不是显著性检验。

本次调查保留全部不利结果，通过每日均值标签模型、去每日均值后五次打乱、原有特征相关暴露和保持日内预测值集合的股票对应关系扰动检查来源。早先试探性的“每一组去均值对照都必须低于 baseline”曾阻断，旧报告保留；最终调查复用原来的 3/5 复发条件，并要求均值模型日内排名相关性 ≥0.8、预测对应关系扰动后的绝对平均 IC <0.005。这些经验条件不能证明统计显著 alpha，报告必须披露单组仍有较大 IC 的事实，不能把负对照当作择优实验。

## 上游边界

本地赛题说明标注后复权，但缺少原 PDF、历史复权因子/股票池/退市覆盖快照、修订记录和收盘数据可得时点证据。这些项列为非阻断的 `unverified`，不会计作已验证。交易成本和成交限制属于此次比赛验收范围之外。

旧失败或用户中断目录保留，不能改成成功。验收期间不增加特征、不调参、不根据 2024 分数改模型；放行后使用独立候选入口与冻结 baseline 对照。
