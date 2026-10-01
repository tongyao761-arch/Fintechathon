# Baseline 严格验收报告

结论：允许开始独立候选特征实验。

本结论只适用于保存的源码、数据和依赖快照，不表示绝对无缺陷或可实现交易收益。

| 检查 | 状态 | 证据 |
|---|---|---|
| unit_tests | passed | [unit_tests.json](unit_tests.json) |
| dependencies | passed | [dependencies.json](dependencies.json) |
| upstream_documentation | passed | [upstream_documentation.json](upstream_documentation.json) |
| raw_data_and_features | passed | [raw_data_and_features.json](raw_data_and_features.json) |
| full_repeatability | passed | [full_repeatability.json](full_repeatability.json) |
| controls_and_ties | passed | [controls_and_ties.json](controls_and_ties.json) |
| artifact_integrity | passed | [artifact_integrity.json](artifact_integrity.json) |
| snapshot_unchanged | passed | [snapshot_unchanged.json](snapshot_unchanged.json) |

## 已修复边界缺陷

- float32 转换后的溢出统一转为 NaN；非法日历日期在原始加载时明确拒绝。
- 修复前复现及源码哈希见 ../../defects_20260930.json；正常数据结果必须匹配冻结 v1.1。

## 结果与解释

- primary_2023：IC=0.0211340195，年化超额=0.0705512646，换手=0.4824181889，总分=0.1848935305。
- oos_2024：IC=0.0525234788，年化超额=0.0740318770，换手=0.7106195201，总分=0.1300330986。
- 全量运行 `artifacts\experiments\baseline_audit\20261001T142637793268Z_a186a4cc`：57.92 秒，采样峰值 RSS 2648.82 MiB。
- 全量运行 `artifacts\experiments\baseline_audit\20261001T142737467959Z_3fb2aa73`：57.79 秒，采样峰值 RSS 2667.59 MiB。
- primary_2023 换手 Top 标签缺失比例 56.1504%，价格有效股票诊断换手 0.852761；不改变官方排名与分数。
- oos_2024 换手 Top 标签缺失比例 27.9686%，价格有效股票诊断换手 0.845878；不改变官方排名与分数。
- 五组标签打乱 IC：-0.046968、-0.043450、-0.043326、-0.042141、-0.045765。
- 原触发条件保留并确实触发调查。每日均值标签模型 IC=-0.043876；与打乱模型日内排名平均相关性为 0.9654、0.9596、0.9640、0.9677、0.9663。
- 去均值后五组打乱 IC：-0.010916、0.007670、-0.013361、0.020074、-0.036695；一组绝对 IC 高于 baseline，原始结果保留；未再次触发原来的 3/5 系统性告警。
- 最强残余对照与现有特征存在相关暴露；保持每日预测值集合、打乱预测与股票的对应关系后，IC=0.001205。
- 原始告警条件没有放宽。调查解释了市场日结构及随机拟合对现有特征的暴露；不证明随机模型 IC 必须为零，也不证明 baseline 显著优于随机对照。先前试探性地要求每个残余 IC 都低于 baseline 过于严格，已改为复用原告警条件并补充预测对应关系对照。
- 原版 2023 排名并列的五次行序扰动，指标范围：`{"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "ic_positive_ratio": 0.0, "annual_excess": 0.0002644045331915701, "top1_annual_ret": 0.00026440453319154233, "mean_turnover": 7.0146658993630595e-06, "final_score": 7.93213599574738e-05}`。冻结实际评分 CSV 行序；不声称任意行序下结果一致。
- 原版 2024 排名并列的五次行序扰动，指标范围：`{"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "ic_positive_ratio": 0.0, "annual_excess": 0.00029904063116063806, "top1_annual_ret": 0.00029904063116054785, "mean_turnover": 6.5358651835101256e-06, "final_score": 8.971218934819558e-05}`。冻结实际评分 CSV 行序；不声称任意行序下结果一致。

## 无法验证与使用边界

- 已读取本地赛题说明，其标注 OHLC 为后复权价格；尚无原 PDF 和历史复权因子快照。
- 后复权历史因子是否随未来事件或数据修订改变（本地说明标注后复权，但无历史因子快照）
- 历史股票池、退市样本覆盖与幸存者偏差
- 原始数据事后修订及其历史可得性
- 当日收盘价、成交量额的实际发布时间和可成交时点
- 交易成本、成交限制与实际可实现收益（本次比赛验收范围之外）
- 非空标签若缺少下一交易日价格证据，属于阻断项；缺失标签单列，不能算作公式核验通过。
- 默认只用 2023 筛选新特征；2024 仅作少量候选复核，不能反复选优后仍声称未参与选择。
- 运行 Git 提交为执行时父提交；精确代码版本以 provenance.json 的逐文件哈希为准。
- 大型模型、预测和评分输入仅保存在本地运行目录；本次未生成正式测试集 submission。

总验收耗时 375.33 秒；主进程采样峰值 RSS 2305.31 MiB；关联实验产物 1904.32 MiB。子进程峰值分别见全量运行摘要。