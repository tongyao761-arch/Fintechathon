# Fintechathon：基于原始量价数据的股票收益预测

本项目面向全市场 A 股日线量价数据，预测每只股票在每个交易日的未来一天收益率 `y_ret_1d`。仓库当前已完成数据与验证框架（DDL①），并提供一个用于验证管线的 5 日动量基线。该基线不是最终比赛模型。

## 赛题交付

最终需要交付：

- `submission.csv`：`ts_code`、`trade_date`、`pred` 三列，与测试集记录逐条对应。
- 数据分析报告：涵盖数据介绍、描述性分析、模型分析、应用验证、产品思路和总结结论。

综合评分由三部分构成：

- 40% Rank IC：逐日截面 Spearman 秩相关系数的均值。
- 30% Top 组超额收益：剔除涨停股后，预测排名前 10% 相对全市场的年化超额收益。
- 30% 低换手：相邻交易日 Top 股票集合的 Jaccard 距离，越低越好。

## 当前进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| DDL① | 数据审计、时间切分、本地评分、提交校验 | 已完成 |
| DDL② | LightGBM Baseline | 待完成 |
| DDL③ | 特征候选基本冻结 | 待完成 |
| DDL④ | 模型候选冻结 | 待完成 |
| DDL⑤ | 最终方案冻结 | 待完成 |

DDL①已确认：

- 训练集与测试集的 `(ts_code, trade_date)` 主键无重复。
- 训练集标签符合 `close(t+1) / close(t) - 1`。
- 时间切分在训练期与验证期之间显式 purge 一个边界交易日。
- 本地评分器与官方 Python 评分脚本的标量指标一致。
- 现有 7 项单元测试全部通过。

## 仓库结构

```text
configs/                  冻结的时间切分配置
scripts/                  可直接运行的实验脚本
src/data/                 数据契约、加载和审计
src/metrics/              官方口径评分与诊断
src/validation/           时间切分、泄漏检查和提交校验
tests/                    单元测试
artifacts/                小体积审计报告和实验摘要
赛题五/                   赛题说明与官方评分脚本
```

`artifacts/` 中的基线仅用于确认数据、切分和评分流程能够端到端运行。当前 5 日动量基线的 2024 样本外综合得分为 `-0.110508`，不应视为最终模型表现。

## 环境安装

建议使用 Python 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据准备

原始赛题数据不纳入 Git。请从比赛官方渠道获取数据，并按以下路径放置：

```text
赛题五/赛题五数据/训练集.csv
赛题五/赛题五数据/测试集_X.csv
```

已审计的原始数据概况：

| 数据集 | 行数 | 股票数 | 交易日数 | 日期范围 |
|---|---:|---:|---:|---|
| 训练集 | 7,900,350 | 4,650 | 1,699 | 2018-01-02 至 2024-12-31 |
| 测试集 | 1,599,600 | 4,650 | 344 | 2025-01-02 至 2026-06-08 |

详细审计结果见 `artifacts/data_profile.md`。

## 快速开始

在项目根目录执行：

```bash
# 重新生成原始数据审计报告
python -m src.data.profile_raw

# 运行单元测试
python -m unittest discover -s tests -v

# 运行 2024 年 5 日动量管线验证
python scripts/run_baseline_2024.py
```

## 数据、特征与预测接口

- 读取数据：`src.data.load_panel.load_panel("train", sort_for_features=True)`。
- 特征表必须以 `ts_code, trade_date` 为唯一键。
- 特征只能使用当日及历史信息，禁止未来 `shift(-1)`、后向填充或使用验证/测试期拟合统计量。
- 日常验证使用 `src.validation.splits.get_split("primary_2023")`。
- 第二套样本外验证使用 `src.validation.splits.get_split("oos_2024")`。
- 模型预测表必须包含 `ts_code, trade_date, pred` 三列。
- 最终预测必须通过 `src.validation.submission.export_submission`导出，由其验证键集并恢复测试集原始行顺序。

## Git 与数据边界

仓库应提交代码、配置、测试、文档以及必要的小体积实验摘要。以下内容默认不进入 Git：

- 训练集、测试集和原始数据压缩包。
- 完整 submission 和官方评分的临时输入。
- 模型权重、checkpoint 和大体积中间特征。
- 虚拟环境、缓存、日志和本地密钥。

## 资料与许可

本仓库尚未声明开源许可证。赛题数据、赛题说明和官方评分脚本的使用与再分发，应以比赛主办方的规则为准。
