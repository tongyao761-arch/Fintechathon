# 模型比较环境实际验收报告

验收日期：2026-10-11。**XGBoost 3.4.1和CatBoost 1.2.10已安装到项目 `.venv`，兼容性与最小功能验收通过。正式实验未执行。** 原25个既有包（包括pip）无版本变化，新增11个包；真实fit=0、真实predict=0、官方/独立评分=0、上传=0。

## 实际环境与安装结果

解释器：`C:\fintechathon\.venv\Scripts\python.exe`。CPython 3.12.10、64位AMD64；pip 25.0.1。原LightGBM 4.7.0、numpy 2.3.4、pandas 2.3.3、scipy 1.17.1、scikit-learn 1.9.1均保持原版本。本机32逻辑CPU（24物理核心），总内存34056667136字节、起点可用17753145344字节，磁盘起点空闲182529134592字节。起点与终点包快照、pip freeze和pip check存于[运行目录](../../artifacts/model_comparison_v1/20261011T073612105802Z/)。

| 新增发行包 | 精确版本 |
| --- | --- |
| catboost | 1.2.10 |
| contourpy | 1.4.0 |
| cycler | 0.12.1 |
| fonttools | 4.66.1 |
| graphviz | 0.21 |
| kiwisolver | 1.5.1 |
| matplotlib | 3.11.2 |
| pillow | 12.3.0 |
| plotly | 7.1.0 |
| pyparsing | 3.3.3 |
| xgboost | 3.4.1 |

XGBoost使用现有numpy/scipy。CatBoost声明graphviz、matplotlib、numpy、pandas、scipy、plotly、six；matplotlib又带入contourpy/cycler/fonttools/kiwisolver/pillow/pyparsing，其他所需依赖已存在。实际Requires-Dist完整存于`package_dependencies.json`。graphviz是Python发行包，本轮未安装系统Graphviz程序；该程序不用于当前回归验收。

按全部25个起点包精确约束运行pip `install --dry-run --only-binary=:all: --constraint ... --report ... xgboost catboost`，实际解析出两个稳定发行版及九个必要新依赖。解析未要求替换既有包；没有为追随网页版本升级或降级任何原包。XGBoost rolling stable安装文档页面显示3.4.2，实际pip可得并选择3.4.1，两者不混称。

解析后固定全部新增包版本，下载预编译wheel，核对下载字节SHA256与解析报告提供的SHA256完全一致，再以`--no-index --no-deps --only-binary=:all:`安装这些精确本地wheel。no-deps只用于已解析和校验的本地轮子，完整依赖验收通过；没有无约束升级、源码编译、nightly或额外pip升级。新增wheel标签均与本解释器兼容：两模型分别为py3-none-win_amd64与cp312-cp312-win_amd64。

原始argv、解析报告、安装报告、下载/安装stdout与stderr、退出码、来源URL及每个wheel哈希均存档为`pip_*.json/log`、`resolved_additions.json`和`wheel_manifest.json`，wheel原件也保留。未修改PATH、重建venv、安装CUDA/驱动/编译工具链或系统Python包。Windows wheel本身可能含其他设备能力，本轮全部模型调用显式CPU，没有借此调用GPU。

新增[requirements-model-compare.in](../../requirements-model-compare.in)固定模型版本，并沿用原依赖意图；[requirements-model-compare.lock](../../requirements-model-compare.lock)为本轮完整36包快照（包含pip）。起点全包约束另存`constraints_before.txt`；旧`requirements-model.in/lock`和`requirements.txt`原样保留，旧锁仍表示旧验收环境。

安装前pip check：`No broken requirements found.`。安装后：`No broken requirements found.`。前后freeze和importlib.metadata快照独立核对，旧包升级/降级/卸载数量均0；两库已实际导入并执行CPU功能，未出现需要系统级修复的DLL错误。

## 合成模型功能和旧模型保护

固定纯合成种子20261011，512行训练、29数值列、73行独立预测，原float32，含随机NaN与整行NaN，标签均有限；不读取真实赛题样本到模型。每库只做一次fit、5轮/5棵树，使用计划的同一参数，仅轮数从100降为5。初次/重载predict各一次，总合成fit=2，合成predict=4，fit重试=0。

| 模型 | 棵树 | 状态 | 保存重载最大绝对差 | fit到验收秒数 |
| --- | ---: | --- | ---: | ---: |
| XGBoost 3.4.1 | 5 | accepted | 0 | 0.203988 |
| CatBoost 1.2.10 | 5 | accepted | 0 | 0.209235 |

预测全部有限。XGBoost保存UBJ、CatBoost保存CBM，两者都实际重载并对同一73行样本推断；数组及模型文件留档。容差atol=1e-12、rtol=0，实际差0；该标准只验证序列化后推断一致性，不证明重复fit确定性、泛化或合成精度，未比较合成精度或据此选择参数。完整有效配置与每次fit进入事件存于`synthetic_*_effective.json`、`synthetic_*_result.json`及独立`ledger.json`。

原最终LightGBM模型只加载，不推断。29列列序、100轮/100棵树通过；原模型保存参数中的num_threads=32、objective=regression、learning_rate=0.05、num_leaves=31与原证据相符，模型哈希保持`76d9fc38e08e797cab5b939561085bf392ce3dbcb191b17878a6b50646f38ede`。正式submission哈希保持`96f0f30b5c52c526705c6b2dff9e89ebad730dbe15474219025e187a3c2faaa5`。旧源码、配置、账本、主备预测、依赖锁、Git配置和pre-push保护均纳入前后校验；原全量测试套件和旧训练均未重跑。

## 故障留存与证据限制

第一次预检因psutil Windows接口不接受Path对象报TypeError，在安装/fit前失败；修复为str路径。原脚本`environment_setup_before_repair01.py`、异常`failure_preflight_*.json`及[修复记录](../../artifacts/model_comparison_v1/20261011T073612105802Z/repair01.json)保留，未覆盖既有环境快照或浪费fit预算。

官方资料的直接归档中XGBoost rolling安装页返回HTTP 403；该页已通过网页工具读取，直接下载失败保留在`official_documentation_sources.json`。安装依据仍为实际pip解析/wheel和合成运行。CatBoost安装页、对应版本CatBoost Python源码、XGBoost v3.4.1参数源码及CatBoost参数页均成功归档；滚动文档用于解释，精确版本源码与实际执行用于兼容性验收。[XGBoost官方安装](https://xgboost.readthedocs.io/en/stable/install.html)、[CatBoost官方安装](https://catboost.ai/docs/en/concepts/python-installation)。

本轮正式数据仅用于文件SHA256/Parquet页脚核验和读取已有JSON/评分摘要，不输入新模型，不产生新历史评分或测试预测。新模型百万行耗时/RSS未知；[PLAN](PLAN.md)中的上限是预算。正式runner尚未实现，未来授权后需落实外部监控与账本；本轮不声称完整正式执行入口已经验收。比赛报告仍未完成，submission未上传。

## 起点既有包逐项核对

| 包 | 起点版本 | 终点版本 |
| --- | --- | --- |
| cloudpickle | 3.1.2 | 3.1.2 |
| colorama | 0.4.6 | 0.4.6 |
| iniconfig | 2.3.0 | 2.3.0 |
| joblib | 1.6.0 | 1.6.0 |
| lightgbm | 4.7.0 | 4.7.0 |
| loguru | 0.7.3 | 0.7.3 |
| narwhals | 2.26.0 | 2.26.0 |
| numpy | 2.3.4 | 2.3.4 |
| packaging | 26.3 | 26.3 |
| pandas | 2.3.3 | 2.3.3 |
| pip | 25.0.1 | 25.0.1 |
| pluggy | 1.6.0 | 1.6.0 |
| psutil | 7.2.2 | 7.2.2 |
| pyarrow | 25.0.1 | 25.0.1 |
| pygments | 2.21.0 | 2.21.0 |
| pytest | 9.1.1 | 9.1.1 |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 |
| pytz | 2026.4 | 2026.4 |
| pyyaml | 6.0.3 | 6.0.3 |
| scikit-learn | 1.9.1 | 1.9.1 |
| scipy | 1.17.1 | 1.17.1 |
| six | 1.17.0 | 1.17.0 |
| threadpoolctl | 3.7.0 | 3.7.0 |
| tzdata | 2026.4 | 2026.4 |
| win32-setctime | 1.2.0 | 1.2.0 |

最终专项只读测试、保护校验、状态文档版本链和文件哈希见[HANDOFF](HANDOFF.json)。安装辅助脚本仅供溯源，没有正式实验自动启动命令。本轮完成后停止，不提交Git、不操作远端。
