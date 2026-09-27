# Fintechathon：基于原始量价数据的股票收益预测

本项目面向全市场 A 股日线量价数据，预测每只股票在每个交易日的未来一天收益率 `y_ret_1d`。当前固定对照为十特征 LightGBM `baseline_v1_1`，用于后续特征实验；最终比赛模型与测试集 submission 尚未完成。

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
| DDL② | LightGBM Baseline | 十特征对照已建立；验收见修正版交接报告 |
| DDL③ | 特征候选基本冻结 | 待完成 |
| DDL④ | 模型候选冻结 | 待完成 |
| DDL⑤ | 最终方案冻结 | 待完成 |

DDL①已确认：

- 训练集与测试集的 `(ts_code, trade_date)` 主键无重复。
- 训练集标签符合 `close(t+1) / close(t) - 1`。
- 时间切分在训练期与验证期之间显式 purge 一个边界交易日。
- 本地评分器与官方 Python 评分脚本的标量指标一致。
- 当前评分边界测试与全量复现记录见 `artifacts/baseline_v1_1/REVIEW_HANDOFF.md`。

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

`artifacts/lightgbm_baseline_v1/` 保留旧版十特征结果；`artifacts/baseline_v1_1/` 保存修正版的小型验收材料。早期 5 日动量结果仍保留作为历史管线检查。

## 环境安装

已验证环境为 Windows、Python 3.12.10。模型依赖意图保存在 `requirements-model.in`，复现使用已核对的 `requirements-model.lock`；`requirements.txt` 仅包含早期基础依赖。以下命令使用独立虚拟环境，不修改系统 Python 或 PATH。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-model.lock
.\.venv\Scripts\python.exe -m pip check
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

```powershell
# 运行单元测试
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 固定十特征基线，默认只验证 2023
.\.venv\Scripts\python.exe scripts/run_lightgbm_baseline.py

# 明确需要两年验收时才执行；--verify-clean 额外要求本机存在旧 clean 文件
.\.venv\Scripts\python.exe scripts/run_lightgbm_baseline.py --experiment-id baseline_v1_1 --split all --verify-clean
```

每次运行写入独立的 `artifacts/experiments/<experiment-id>/<run-id>/`，禁止覆盖。默认重建只需要原始训练 CSV；`--verify-clean` 用于输入迁移验收，不是日常运行依赖。重建逻辑见 `src/data/baseline_panel.py`：仅依据当日 OHLC 生成质量标记，不填补、不删行；模型 X 与训练 Y 使用原 float32 规范，评分标签保留原始精度。

运行目录保留模型、原始预测 Parquet、实际评分 CSV、逐日和月度诊断、文件哈希、代码哈希、依赖版本及状态。只有全部检查通过才写入成功状态。预测哈希和训练样本数必须匹配冻结的十特征版本；`--experiment-id` 仅改变留档名称，不会解除这项检查或启用新特征。

后续新增特征应建立独立候选入口，复用这里的评分、原始真值提取和留档辅助模块，保留此入口作为固定对照；不能把改动后的候选写回旧基线。日常筛选使用 2023，2024 用于少量候选复核，不能反复依据其分数调参后仍声称它是未参与选择的样本外检验。

官方换手率包含标签缺失股票，收益分组还会排除这些股票。诊断会分别保存两个 Top 集合，并报告标签缺失、价格无效、十特征全缺失占比。价格有效样本上的换手率仅供诊断，不替代官方分数，也不改变预测排名。

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
