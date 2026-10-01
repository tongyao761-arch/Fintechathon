# Baseline 严格验收报告

结论：不放行增加特征。

本结论只适用于保存的源码、数据和依赖快照，不表示绝对无缺陷或可实现交易收益。

| 检查 | 状态 | 证据 |
|---|---|---|
| unit_tests | passed | [unit_tests.json](unit_tests.json) |
| dependencies | passed | [dependencies.json](dependencies.json) |
| upstream_documentation | passed | [upstream_documentation.json](upstream_documentation.json) |
| raw_data_and_features | passed | [raw_data_and_features.json](raw_data_and_features.json) |
| full_repeatability | passed | [full_repeatability.json](full_repeatability.json) |
| controls_and_ties | failed | [controls_and_ties.json](controls_and_ties.json) |
| snapshot_unchanged | passed | [snapshot_unchanged.json](snapshot_unchanged.json) |

## 已修复边界缺陷

- float32 转换后的溢出统一转为 NaN；非法日历日期在原始加载时明确拒绝。
- 修复前复现及源码哈希见 ../../defects_20260930.json；正常数据结果必须匹配冻结 v1.1。

## 结果与解释

- primary_2023：IC=0.0211340195，年化超额=0.0705512646，换手=0.4824181889，总分=0.1848935305。
- oos_2024：IC=0.0525234788，年化超额=0.0740318770，换手=0.7106195201，总分=0.1300330986。
- 全量运行 `artifacts\experiments\baseline_audit\20261001T140252232960Z_994d1a82`：58.04 秒，采样峰值 RSS 2070.44 MiB。
- 全量运行 `artifacts\experiments\baseline_audit\20261001T140352076568Z_a86d2e06`：58.08 秒，采样峰值 RSS 2647.84 MiB。
- primary_2023 换手 Top 标签缺失比例 56.1504%，价格有效股票诊断换手 0.852761；不改变官方排名与分数。
- oos_2024 换手 Top 标签缺失比例 27.9686%，价格有效股票诊断换手 0.845878；不改变官方排名与分数。

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

总验收耗时 296.11 秒；主进程采样峰值 RSS 2233.84 MiB；关联实验产物 489.48 MiB。子进程峰值分别见全量运行摘要。