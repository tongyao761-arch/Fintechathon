# DDL①：数据与验证框架

本交付包固定了全队共用的数据、时间验证与评分口径。原始 CSV 始终只读。

## 快速开始

在项目根目录执行：

```bash
python -m src.data.profile_raw
python -m unittest discover -s tests -v
python scripts/run_baseline_2024.py
```

## 组员接口

- B/C：用 `src.data.load_panel.load_panel("train", sort_for_features=True)` 读取数据。每张特征表以 `ts_code, trade_date` 为唯一键；只使用当日和历史信息。
- D：用 `src.validation.splits.get_split("primary_2023")` 获取日常验证掩码；模型预测应是 `ts_code, trade_date, pred` 三列。
- E：用 `src.metrics.official.score_official(..., return_details=True)` 获取官方分数及逐日 IC、超额收益、Top 集合和换手率明细。
- A：最终文件必须经过 `src.validation.submission.export_submission`；它会检查所有测试键并恢复原始行顺序，绝不补行、删行或改写预测。

## 已冻结规则

- 预测时点是交易日 `t` 收盘后，标签是 `close(t+1) / close(t) - 1`。
- `primary_2023`：训练到 2022-12-29，purge 2022-12-30，验证 2023 年。
- `oos_2024`：训练到 2023-12-28，purge 2023-12-29，验证 2024 年。
- 所有拟合（标准化、缺失填补、目标编码、分位点）仅能用训练掩码；禁止随机切分、后向填充和未来 `shift(-1)`。
- Top 收益按官方代码剔除涨停与标签缺失；换手率只剔除涨停。

`scripts/run_baseline_2024.py` 是一个 5 日动量的可复现基线，不是最终模型。它同时调用本地评分器和未修改的官方脚本，只有所有标量指标在 `1e-12` 内一致时才会成功。
