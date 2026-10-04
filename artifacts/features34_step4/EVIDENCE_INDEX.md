# 第4步证据索引

2026-10-04：135项新单删训练（2023：45；2021/2022：90），9项同年参照复用，48条条件性背景结论。独立审计通过；模型失败0，辅助审计失败1并留档。联合0、回补0，冻结原full34及lean31。到此停止，未使用2024或生成最终比赛预测。

|需要核查的问题|实际证据|
|---|---|
|24列各背景结论及三年分差|docs/features34/STEP4_REPORT.md；feature_decisions.csv|
|每列不确定的直接原因、触发年份和实验ID|decision_evidence_details.csv|
|删除前后所有官方分项、对原候选及10分差、缺失和价格有效换手|deletion_results.csv（135行）|
|每月IC、超额、官方换手、分数及两种参照分差|monthly_deletion_results.csv（1620行）|
|运行前规则、阈值、20日配对块/2000次及换手边界|configs/features34_step4.json；docs/features34/STEP4_PREREGISTRATION.md；preregistration.json|
|45项2023独立单删矩阵|screen_matrix.json；screen_registration.json|
|跨年复核名单、每项理由和90项预算|cross_review_list.csv；cross_plan.json；cross_registration.json|
|联合/回补预算与未新增版本的依据|joint_plan.json|
|两个冻结候选、模型列、三年表现和最差年月|frozen_candidates.json；frozen_candidate_comparison.csv|
|每项完整不可覆盖模型/预测/原始评分输入/逐日与缺失诊断|run_index.json指向artifacts/experiments/S4_*；各run summary的file_sha256|
|执行来源及旧参照核验|preregistration.json中的source/reference_runs；executed_sources/|
|135项全部键/模型/文件/评分及结果重算验收|acceptance.json；summary.json；audit_attempt_02_command.json；audit_attempt_02.log|
|测试与依赖|tests_result.json/log（92项通过）；dependency_check.json/log|
|失败记录及审计修正的精确边界|failures.json（模型0）；auxiliary_failures.json；audit.log；audit_command.json；audit_source_correction.json|
|本地分支、push保护和既有未跟踪内容|preflight.json及acceptance.json；仅审计器本任务新文件有登记的后修正|
|审计后补充解释表和文档的轻量核查|final_evidence_check.json；数值表、冻结候选和所有模型产物没有变化|

路径未加项目目录时，均位于artifacts/features34_step4/；docs/configs/scripts路径相对C:/fintechathon。完整实验、源码快照和日志留本地，小型结果及代码按项目规则作本地提交；无远程写操作。

运行来源中的Git提交是执行时父提交，不是最终第4步提交；准确执行版本以保存源码哈希为准。唯一审计器后修正有原版/修正版SHA-256，不涉及训练、特征、评分或预登记规则变更。
