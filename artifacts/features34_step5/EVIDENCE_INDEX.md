# 第5步证据索引

- `FROZEN_CANDIDATES.json`、`registration.json`：查看本次候选2024前的两个清单、公式、参数、理由、配置/代码/开发证据与哈希。
- `run_index.json`：仅两项2024新运行；指向完整模型、全部键预测、原始标签文本官方输入、逐日/月度/缺失/特征统计及来源。
- `annual_comparison.csv`：3个组合×4年，全部官方分项、贡献、同年十特征增量与年度缺失/价格有效换手。
- `monthly_comparison.csv`：144行月度分项与对照增量；月度不平均替代全年。
- `monthly_missing_diagnostics.csv`：288行收益/换手Top计数和加权缺失占比、价格有效换手。
- `top_membership_comparison.csv`：2个候选×4年，Top成员对称差和缺失/价格无效子集变化。
- `paired_intervals.csv`：固定预测对十特征的配对区间，不代表选择后的普遍显著性。
- `feature_decisions.csv`：48条开发背景条件结论与实际冻结模型输入；2024不重判单列。
- `candidate_conclusions.csv`：下一阶段支持、负年度/月份、相对优势变化与限制。
- `acceptance.json`、`summary.json`、`status.json`：独立验收、真实运行状态与成功摘要。
- `REPORT.md`：验收脚本生成的完整报告，保留原哈希；`REPORT_NOTES.md`：仅由已验收表格导出的补充分项解释；最终文档为二者拼接的 `docs/features34/STEP5_REPORT.md`。
- `preflight.json`：既有未跟踪文件、前四步产物和Git配置/pre-push保护的哈希。
- `tests_result.json`、`dependency_check.json`、`failures.json`、`commands/`：102项回归、依赖、实际退出码、失败清单（本步无失败）。
- `executed_sources/`：精确执行副本保留本地；来源哈希在冻结文件。原始日志和交付核验脚本在 `artifacts/features34_step5_checks/`，不混入代码提交。
- `delivery_verification.json`：生成报告/补充报告/进度的交付哈希、冻结和证据保护、暂存字节检查。

仅本地ivor-work，无远程写入；仅两次固定候选训练，不再搜索，未生成最终比赛submission，第5步完成后停止。
