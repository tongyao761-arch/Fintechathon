# 第4步修订证据索引

- `registration.json`、`STEP4_REVISION_RULES.md`、`executed_sources/`：已观察旧结果后的探索性修订；新训练前固定配置、规则、数据、依赖及源码哈希。
- `preflight.json`：旧第4步84个文件、旧规则/报告、215个既有未跟踪文件、Git配置和pre-push保护的原哈希。
- `matrix.json`、`probe_evidence.csv`：两个固定背景、两版联合删除清单、三个年份、逐列准入依据及全阶段预算。
- `single_decisions.csv`：24列×两个背景的旧/新判断、直接年度分差、实验ID、统计不确定性和最终输入状态。不存在的背景证据不补造。
- 原135项单删：`../features34_step4/deletion_results.csv`、`monthly_deletion_results.csv`及`run_index.json`；复用前完整保存哈希/模型/预测核验，旧产物不覆盖。
- 原九项参照：`registration.json`中的`reference_runs`，来源为第3步同年baseline10/full34/lean31。
- `run_index.json`：六项新联合实验目录；每项保存模型、原始全部键预测、官方输入/比较、逐日/月度/缺失诊断、特征统计、日志及来源。
- `joint_results.csv`、`candidate_results.csv`、`monthly_candidate_results.csv`：删除前后官方分项、对十特征及原候选分差、月度稳定性/收益集中、四种Top缺失占比、价格有效换手及连续块区间。
- `candidate_ranking.csv`、`frozen_candidates.json`：官方分数准入、失败/冲突风险、精确排序和最多两个冻结输入清单；联合结果不证明各列必需。
- `tests_result.json/.log`、`dependency_check.json/.log`、`commands/`、各命令结果：真实执行命令、退出码及日志。
- `failures.json`：失败留档。失败版本不入候选排名；没有成功验收前不生成成功总摘要。
- `acceptance.json`、`summary.json`：独立模型重载、原始键/资格、逐日评分重建、表/区间/名单/排名及哈希/保护核验。
- `REPORT.md`与`docs/features34/STEP4_REVISION_REPORT.md`：解释性报告。2024未用于训练、评分或筛选；本阶段完成即停止。
