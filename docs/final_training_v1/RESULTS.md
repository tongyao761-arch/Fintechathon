# 最终训练与测试预测验收

当前搜索结束。最终P+J基础模型及完整主备测试预测技术验收通过。主方案PJ_a080 [0.8,0.2]，备选PJ_a090 [0.9,0.1]；共用一个最终模型，未改选。

运行目录：`C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z`。实际全量fit=1、全量测试推断=2、合成fit=0、重试fit=0。第二次推断由独立审计器从保存模型重载完成，逐值一致。

原始历史X共7,900,350行；训练样本6,754,006行，使用原始y_ret_1d float32。训练日期20180102—20241230；20241231全部4,650个Y排除，全部4,650行X留作历史。排除原因是按顺序互斥统计，见training_input_summary.json。训练键/标签与原始行号严格对齐，模型输入29列、参数及100棵树均与冻结合同一致，实际num_threads=32。

全部测试预测均覆盖1,599,600个唯一原始键，日期20250102—20260608，保持测试X原行序；包括价格无效、缺失特征和涨停行。完整历史与测试X连接后计算因果特征，历史部分29列与仅历史X计算逐值一致。

测试首日20250102回退当日原值；2026首日20260105按跨年边界回退；其他月份保持连续，精确前一市场交易日按同股票同模型原预测匹配，非递归。回退/匹配计数：{'used_exact_previous_raw': 1590300, 'test_first_day': 4650, 'annual_boundary': 4650}。独立merge重建两份平滑及追踪逐值一致。

三份Parquet预测保留float64；主备CSV仅ts_code,trade_date,pred，使用%.17g；round_trip读取逐值一致，默认解析误差均不超过1e-15。

## 模型与预测文件

| 文件 | SHA256 | 行数 |
| --- | --- | --- |
| [final_model.txt](C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\final_model.txt) | `76d9fc38e08e797cab5b939561085bf392ce3dbcb191b17878a6b50646f38ede` | 100棵树 |
| [raw_predictions.parquet](C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\raw_predictions.parquet) | `10e1f481cf79e4bb811848efccc8a4877120ab63e9d23b26c715d8dca1076d0f` | 1599600 |
| [PJ_a080_predictions.parquet](C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\PJ_a080_predictions.parquet) | `77de51875fe8c6232233ecd870023736a4f69306fb807304a3a9d4830fbe8652` | 1599600 |
| [PJ_a080_predictions.csv](C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\PJ_a080_predictions.csv) | `96f0f30b5c52c526705c6b2dff9e89ebad730dbe15474219025e187a3c2faaa5` | 1599600 |
| [PJ_a090_predictions.parquet](C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\PJ_a090_predictions.parquet) | `1fdc736b540cfcc76649fb75fded2ebe79993d2c66ad1053dce5bde15cea8616` | 1599600 |
| [PJ_a090_predictions.csv](C:\fintechathon\artifacts\final_training_v1\20261010T082317559902Z\PJ_a090_predictions.csv) | `7b3edfec559d1b350f1a16b08e3423c54845010c3a41101614f50adeb01feef1` | 1599600 |

## 资源与失败记录

准备、训练/初次推断、独立审计实测合计736.489秒；峰值RSS=7925067776字节。各阶段资源原始收据及磁盘统计见resources.json。

失败/修复记录：[{'path': 'C:\\fintechathon\\artifacts\\final_training_v1\\20261010T082317559902Z\\signed_NaN_audit_precheck_failure.json', 'sha256': 'fc5b22a02aa34c65cc33efd1042ddb4654e36fca886d5ab02f4451449909368d'}, {'path': 'C:\\fintechathon\\artifacts\\final_training_v1\\20261010T082317559902Z\\prechecks\\initial_test_failure.json', 'sha256': '6ce308542cc7761da3656a31d3cc2a05ce884b84d88a2c5d00b97495755c1cdd'}]。合成测试首次受系统临时目录WinError 5影响，改用新建任务目录后7项通过；真实fit未失败、未重试。所有历史产物、旧冻结、旧预算账本、旧模型预测及Git保护配置经内容哈希复核未改变。

## 交接与停止

从HANDOFF.json读取登记、验收、账本、run_index和逐文件SHA256。PROJECT_PROGRESS.md更新前版本保存在本运行prior_PROJECT_PROGRESS.md，版本哈希链接见progress_version_link.json。旧v2冻结与交接保持历史语义。

正式submission.csv及比赛报告仍未完成；没有上传、Git提交或远程操作。未访问隐藏测试Y、未重建测试收益标签、未计算测试IC/超额收益/综合分/名次/提升概率。技术验收只证明训练和预测合同正确，无法验证隐藏测试成绩。授权工作完成后停止。
