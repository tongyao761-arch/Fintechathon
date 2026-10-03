"""Render the step3 report from audited local evidence without selecting more runs."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_features34_step3 import read_json,OUTPUT
from src.validation.experiment import sha256_file,write_json


def table(frame,columns,headers=None):
    headers=headers or columns
    lines=['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        values=[]
        for c in columns:
            v=row[c]
            if pd.isna(v):
                v='不适用'
            elif isinstance(v,(float,)):
                v=f'{v:.9f}'
            values.append(str(v))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)


def main():
    acceptance=read_json(OUTPUT/'acceptance.json')
    assert acceptance['accepted']
    total=read_json(OUTPUT/'summary.json')
    frames={n:pd.read_csv(OUTPUT/f'{n}.csv',float_precision='round_trip') for n in
            ['comparison','candidate_ranking','monthly_stability','conditional_group_deletions','split_audit']}
    annual=frames['comparison']; ranking=frames['candidate_ranking']
    selected=total['proposed_next_candidates']
    text=['# 第3步：精简候选与2021/2022/2023开发验证','',
          '固定矩阵15项已完成并通过独立磁盘审计：新增12项，复用已验收2023的十特征、完整34、34−A共3项。没有使用2024，没有运行逐特征检查。','',
          '依据运行前固定的规则，提出进入下一步逐特征检查的候选：'+ '、'.join(selected)+'。这是检查候选，不是最终特征保留结论。','',
          '## 候选与切分','',
          '预先候选为lean31=10+BCDEF（31列）、lean27=10+BDEF（27列）、lean23=10+BDF（23列）；完整34和十特征作为同切分参照。各组理由、第2步实验ID和加入/删除分差在STEP3_CANDIDATES.md中已于训练前保存，哈希见preregistration.json。','',
          table(frames['split_audit'],['split','train_start','train_end','purge_date','valid_start','valid_end','train_samples','valid_prediction_rows'],
                ['切分','训练起点','末训练日','排除标签日','验证首日','验证末日','训练资格样本','全部验证键']), '',
          '日期从原始交易日确定，训练只用验证年前历史，并排除紧邻验证年的最后一个交易日标签。独立研究配置不修改冻结configs/splits.yaml。特征计算使用完整历史X面板；2024不训练、不评分、不参与选择。','',
          '## 全年官方分项与相对十特征分差','',
          table(annual,['year','candidate','feature_count','ic_mean','annual_excess','mean_turnover','final_score','score_minus_baseline10'],
                ['年','组合','列数','Rank IC','年化超额','官方换手','综合分','相对十特征分差']), '',
          '其他官方分项、ICIR、IC正日占比、Top年化收益、分项贡献、对照分差、预测哈希、耗时和采样峰值RSS均在comparison.csv。全年分数直接使用官方评分，未用月度分数平均替代。','',
          '## 预定义跨年选择结果','',
          table(ranking,['candidate','feature_count','mean_annual_delta','worst_annual_delta','negative_years','positive_months','worst_month_delta','median_month_delta','proposed_for_feature_checks'],
                ['组合','列数','平均年分差','最差年分差','负年份数','正月份/36','最差月分差','月分差中位数','进入检查']), '',
          '规则在运行前固定：先看各年对照分差，任一负年份标记冲突；平均和最差年及正月数共同排序。平均与最差年均相差≤0.003且正月份相差≤2才按列数优先。该接近阈值是研究约定，不是显著性或统计等效证明。','']
    for name in selected:
        row=ranking.set_index('candidate').loc[name]
        byyear=annual[annual.candidate==name]
        text.append(f'- {name}（{int(row.feature_count)}列）：三年分差'+ '、'.join(f'{int(r.year)} {r.score_minus_baseline10:+.9f}' for r in byyear.itertuples())+
                    f'；平均{row.mean_annual_delta:+.9f}，最差{row.worst_annual_delta:+.9f}，正月份{int(row.positive_months)}/36。'+
                    ('存在跨年退化，仅作为解释冲突的检查候选。' if row.negative_years else '三年均优于十特征；仍有负月份，未证明逐列必需。'))
    indexed=ranking.set_index('candidate')
    mean_gap=float(indexed.loc['lean31','mean_annual_delta']-indexed.loc['full34','mean_annual_delta'])
    text+=['',f'lean31相对完整34的平均年分差仅多{mean_gap:+.9f}，优势较小，不能宣称统计显著。它的最差年度对照分差更高，正月份33/36对31/36，最差月退化也更小，因此先进入检查；保留完整34作为第二候选，是因为删A的效果跨年反转。', '',
           'lean27的2023得分最高，最差年度对照分差也优于上述两个组合；但2021/2022低于二者，三年平均分差低于二者，且差距超过预登记的平均接近阈值0.003，因此按平均为主的固定方法未入选。这个风险取舍明确保留，不解释为lean27无效。lean23的平均与最差年度分差均更低，E删除在这个背景下三年都降低分数，故未入选。']
    text+=['','完整34和所有精简组合的比较按上述表中的三年证据解释；本步不以最佳单年宣布有效。若未被选中的组合某年优于所选组合，也保留该冲突，不改候选追加搜索。','',
        '## 组删除的条件性跨年证据','',
        table(frames['conditional_group_deletions'],['year','deleted_group','context_before','context_after','final_score_deletion_delta','ic_mean_deletion_delta','annual_excess_deletion_delta','mean_turnover_deletion_delta'],
              ['年','删组','原组合','删后组合','删后综合分差','IC差','年化超额差','官方换手差']), '',
        'A的删除在完整34背景下检验：2021负、2022/2023正，属于跨年冲突。C的删除在已无A的背景下检验：2021/2022负、2023正，同样冲突。E的删除在已无A/C的背景下三年均负，对该背景的组保留有支持，但尚不能确认E中每一列都有效。不同背景的效应不能相加，也不能外推到所有组合。本表结合第2步加入/删除证据使用。B/D/F在三个精简候选均保留，本步没有其跨年单删实验，因此不能据此宣布每列必须保留；A/C/E也没有逐列结论。','',
        '## 月度稳定性','',
        table(frames['monthly_stability'],['year','candidate','positive_months','min_month_delta','max_month_delta','median_month_delta','positive_excess_months','largest_positive_month_share'],
              ['年','组合','正分差月数/12','最差月差','最好月差','中位月差','超额正增量月数','最大月占正增量和']), '',
        '最大月占比仅描述正月度分差的集中度，不能还原或替代全年分数。180行逐月IC、年化超额、换手、分项贡献和分差见monthly_comparison.csv。负月份和收益集中说明跨年涨分不等于每月稳定，三年开发验证也不代表2024表现。','',
        '## 缺失样本与价格有效换手诊断','',
        table(annual,['year','candidate','top_missing_label_fraction','top_invalid_price_fraction','top_baseline_all_missing_fraction','top_candidate_all_missing_fraction','price_valid_only_turnover'],
              ['年','组合','换手Top缺标签占比','价格无效占比','十特征全缺失占比','候选全缺失占比','价格有效换手']), '',
        '占比按Top入选股票—日期次数加权，保留全部验证键。2021除lean23约99.9991%外，各组合换手Top组缺标签和价格无效占比均为100%；2022约84.64%–86.48%，2023约56.15%。官方低换手主要描述这个含大量缺失行的Top集合，不能解释为可交易股票稳定。所有新增特征组合在三个开发年中的官方换手及价格有效换手都高于同年十特征，低换手分项贡献均下降；综合分提高来自IC和超额收益贡献抵消该损失。', '',
        '涨跌停标记有值可使候选全缺失占比为0，这不等于价格有效。官方换手包含标签缺失行；价格有效换手只作诊断，不替代官方评分。360行月度收益/换手Top缺失占比、计数和价格有效换手见monthly_missing_diagnostics.csv。','',
        '## 验收与限制','',
        '- 88项回归通过、pip check通过。39个冻结文件哈希前后不变；训练资格、目标标签、参数、种子、特征公式及官方评分均保持。',
        '- 15项同年训练数一致、验证全部原始键一致、覆盖100%、预测有限、模型保存重载一致、本地与官方所有标量最大差异0。独立审计重算全部汇总表并检查实际文件哈希、模型列和原始面板键。',
        '- 已有2023实验经核验复用，未重做。新增训练无失败；留档辅助脚本的非UTF-8日志读取和Git中文路径转义失败均保留，修复后完成核查，详见checks目录的失败JSON。',
        '- 仅本地ivor-work。push保护、既有未跟踪文件保持。没有调参、换模型、预测平滑、添加新特征或生成最终比赛预测；evaluate_input中的submission.csv只是历史验证官方校验输入。',
        '- 第3步到此停止。逐特征检查、最终列名单及2024少量复核都尚未执行，不能将当前候选写成最终模型。','',
        '## 证据与复现入口','',
        '- 预登记：configs/features34_step3.json、docs/features34/STEP3_CANDIDATES.md、artifacts/features34_step3/preregistration.json。',
        '- 汇总、验收、复用：artifacts/features34_step3/summary.json、acceptance.json、reuse_verification.json、run_index.json。',
        '- 完整运行目录见run_index.json：模型、Parquet预测、原始真值/评分输入、逐日/月度/缺失诊断、特征缺失统计、status/config/provenance均保留。',
        '- 测试、依赖、辅助失败和独立审计：artifacts/features34_step3_checks/。精确执行源码及配置：artifacts/features34_step3/executed_sources/。',
        '- 已执行命令：`.\\.venv\\Scripts\\python.exe -B scripts/run_features34_step3.py`。固定汇总目录拒绝覆盖；这条是完成记录，不是下一步重跑指令。',
        '- 独立磁盘核查：`.\\.venv\\Scripts\\python.exe -B artifacts/features34_step3_checks/audit.py`。','']
    report=ROOT/'docs/features34/STEP3_REPORT.md'
    report.write_text('\n'.join(text),encoding='utf-8')
    write_json(OUTPUT/'report_evidence.json',dict(report_sha256=sha256_file(report),
        generator_sha256=sha256_file(Path(__file__)),proposed=selected,source_acceptance_sha256=sha256_file(OUTPUT/'acceptance.json')))
    print({'report':str(report),'selected':selected})


if __name__=='__main__': main()
