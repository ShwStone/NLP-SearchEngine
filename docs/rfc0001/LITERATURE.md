# Router 文献矩阵与适用边界

核验：2026-10-27。使用出版者 DOI/Crossref 元数据、arXiv 摘要、作者公开 PDF；
来源响应/文件的 SHA256 记录于 `source-verification.json`。某些 Crossref 模糊搜索返回无关题名，未计入文献。
未能获取全文的条目标明核验深度，不将摘要级阅读冒充完整复现。下面都是已有方法及本项目适配思路，不宣称原创。

| 文献与可核验来源 | 决策信息、监督和方法 | 成本、原任务基线 | 本项目可用点与限制 | 代码/许可与核验深度 |
| --- | --- | --- | --- | --- |
| Cronen-Townsend, Zhou, Croft. **Predicting query performance**, SIGIR 2002. [DOI](https://doi.org/10.1145/564376.564429) | QPP；检索后结果导出的 query language model 与 collection model 差异（clarity/KL）用于估计效果，运行时无需 qrels | 需要读取反馈文档/词分布，比较传统 QPP 指标；不是一次 df 查表的成本 | 支持以可解释检索信号判断不确定性；本轮不计算完整 clarity，避免额外扫描正文和把近似指标冒充原公式 | 未复用原代码，代码许可证未核验；题名/作者/DOI 已核验，原文未完整获取 |
| Shtok et al. **Predicting Query Performance by Query-Drift Estimation**, TOIS 2012（扩展自 ICTIR 2009）. [DOI](https://doi.org/10.1145/2180868.2180873) | 检索后头部得分标准差估计 query drift，normalized query commitment 思路；评估相关性需 qrels，但信号在线可算 | 已有结果列表上聚合分数；比较当时 QPP 方法，不直接预测“hybrid-BM25”增益 | 研究 CV、相对分差、头部集中度；**cv10 不是原文 NQC 的严格实现**，QPP 与额外召回收益并非同一个目标 | 不复制代码，许可未核验；出版者摘要与元数据已核验 |
| Cronen-Townsend, Zhou, Croft. **A framework for selective query expansion**, CIKM 2004. [DOI](https://doi.org/10.1145/1031171.1031220) | 选择性采用扩展结果，依据原/扩展检索变化与效果预测；并非所有查询都应执行相同增强 | 与原查询及非选择性扩展对照；若需要先执行扩展，应计入其成本 | 借鉴“增强可能有害、应逐查询择优”的研究问题；本轮在调用 dense 前决策，不把看过 dense 结果的规则用作零成本路由 | 未核验可复用代码/许可证；精确题名/作者/DOI 由 Crossref 首轮查询核验，后续抓取遇 429 |
| Wang, Lin, Metzler. **A cascade ranking model for efficient ranked retrieval**, SIGIR 2011. [DOI](https://doi.org/10.1145/2009916.2009934) / [作者 PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Wang_etal_SIGIR2011.pdf) | 监督学习排序级联，逐阶段使用更昂贵特征并剪枝；联合考虑效果与检索成本 | 对比传统排序模型及效率方法，按阶段计算成本；目标为文档级排序级联 | 借鉴逐阶段收益/成本权衡；本项目动作是查询级 BM25 或 hybrid，不改变候选集合/排序器冒充复现文档级 boosting | 作者 PDF 已下载并阅读摘要/方法，未复制代码；代码许可未核验 |
| Cormack, Clarke, Buettcher. **Reciprocal rank fusion outperforms condorcet and individual rank learning methods**, SIGIR 2009. [DOI](https://doi.org/10.1145/1571941.1572114) / [作者 PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | 无监督融合排名，sum 1/(c+rank)；需要已经获得两路候选 | 对比单路、Condorcet、CombMNZ；融合本身便宜，取两路候选的成本另算 | 本轮严格沿用原等权 c=60；不能用前轮 c=10、.25/1 的 hybrid 标签，也不将 RRF 论文结论当作本语料效果保证 | 原公式由项目自行实现；作者 PDF 已阅读；不复制外部代码 |
| Bruch, Gai, Ingber. **An Analysis of Fusion Functions for Hybrid Retrieval**, TOIS 2023（arXiv 2022）. [DOI](https://doi.org/10.1145/3596512) / [arXiv](https://arxiv.org/abs/2210.11934) | 比较 convex combination 与 RRF，分析归一化及参数敏感性；融合参数可由训练样本学习 | 单路和不同融合/参数；需要两路结果，非按需 dense 方案 | 说明融合选择会改变逐查询收益标签；本轮冻结融合避免混淆，不同时做融合权重优化 | 出版者及 arXiv 摘要核验；未复制代码，代码许可未核验 |
| Angelopoulos, Bates, Fisch, Lei, Schuster. **Conformal Risk Control**, ICLR 2024（arXiv 2022）. [arXiv](https://arxiv.org/abs/2208.02814) | 在满足假设的独立校准与单调损失条件下控制期望风险；使用校准损失选择阈值 | split conformal 扩展及多个视觉/NLP 风险任务，不等于任意 CV + bootstrap | 预算/校准应预先声明；本轮 signed hybrid-BM25 差值不保证单调，且 dev 已被观察，**拒绝将 mean+1.645SE 称为 conformal 保证** | arXiv 摘要及作者核验；未复制代码，代码许可未核验；需更深入理论复核后才能提出正式 CRC 适配 |
| Ong et al. **RouteLLM: Learning to Route LLMs with Preference Data**, 2024. [arXiv](https://arxiv.org/abs/2406.18665) | 用偏好监督训练强/弱模型选择器，权衡质量与成本；部分方法需查询表示 | 强模型、弱模型及路由质量/成本曲线；不属于检索排序评测 | 借鉴“预测两动作差值而非绝对质量”、跨域迁移与成本曲线；本项目只用 BM25 统计，不引入其 LLM 或 embedding 决策成本，不迁移其胜率结论 | arXiv 题名/作者/摘要已核验；未复用代码，尚未核验源码许可 |

## 直接影响本轮设计的结论

- QPP 分差或置信度只说明“词法可能好/坏”，不必然等于“dense 能补救”；主监督目标应是两动作的真实逐查询增益。
- 级联/选择性增强的方法必须按决策时可见信息和实际成本核对；dense-first 不能冒充 BM25-first。
- 固定融合、候选深度和数据范围；先建立完全一致的两个动作，再比较路由。
- 本轮的经验损失界和分组 bootstrap 只描述观察到的 dev 结果，不具有 CRC 的分布无关保证。
- 独立数据和正式预算仍需 0002/0003 与负责人确认；本地原型不能代替团队批准或最终验证。
