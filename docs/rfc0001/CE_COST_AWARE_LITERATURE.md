# CE 与成本感知重排文献矩阵

本矩阵只记录可核验的公开论文。`论文已有方法`描述原文贡献，`项目适配`描述 RFC-0001 当前可以复用的部分，`拟议改进`是后续工作假设，三者不混为原创贡献。MMarco 结果只用于方法研究，离线索引和自有数据验收仍需另行完成。

| 方向 | 论文与链接 | 决策时可见信息 | 监督信号 | 成本/延迟与基线 | 论文已有方法 | 项目适配 | 拟议改进 |
|---|---|---|---|---|---|---|---|
| CE 多阶段级联 | [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) | query、候选 passage、初始检索分数 | MS MARCO passage relevance | BERT cross-encoder 对 BM25 候选的质量/吞吐比较；BM25 是基线 | 用 BERT 对第一阶段候选重排 | CE-28/40 是同一思想的固定候选池实例 | 在固定预算下学习候选深度与 CE 对数，而不是默认重排全部候选 |
| 多阶段检索 | [A BERT-Based Model for Ranking in Web Search](https://arxiv.org/abs/1905.04044) | query、文档文本、级联阶段输出 | 点击/人工相关性（论文数据协议） | 级联阶段的离线质量与线上吞吐；传统学习排序为基线 | 将轻量检索和昂贵神经排序串成级联 | Router 的 pre-CE gate 与 CE action 是项目适配 | 加入早停置信度和 tail-latency 约束 |
| 动态候选深度 | [Learning to Rank with Selection Bias in Large-Scale Search](https://arxiv.org/abs/1809.04012) | 初始 rank、分数、query/document 特征 | 点击偏差校正后的排序监督 | 大规模线上排序成本与固定深度比较 | 讨论排序阶段和候选选择的偏差 | 项目只使用决策时可见检索统计，按 group 做嵌套验证 | 对 CE-28/40 以外增加预注册的 20/32/48 深度点 |
| 动态早停 | [FastBERT: a Self-distilling BERT with Adaptive Inference Time](https://arxiv.org/abs/2004.02178) | 中间层置信度、输入表示 | 自蒸馏分类损失与置信度目标 | 早退层数、精度和推理时间；完整 BERT 为基线 | 以置信度决定是否继续 Transformer 层 | 项目尚未改变 CE 层数，只测量候选数成本 | 训练 CE 层级早停器，并冻结 p50/p95 预算门槛 |
| CE 蒸馏 | [RocketQA: An Optimized Training Approach to Dense Passage Retrieval](https://arxiv.org/abs/2010.08191) | query、positive/negative passage、teacher scores | hard negatives、teacher/label 监督 | dense retriever 与 cross-encoder teacher 的训练/检索对照 | 用强 teacher 产生蒸馏信号和难负例 | 项目已有 BGE CE 作为 teacher，但尚未把它蒸馏成可替代 CE 的线上模型 | 只用开发训练折蒸馏 CE-28 student，并在自有数据重新验收 |
| 排序蒸馏 | [RankDistil: Knowledge Distillation for Ranking](https://arxiv.org/abs/1911.09796) | teacher 排序、候选特征、query/document | teacher listwise/pairwise 排序信号 | student 排序器延迟与 teacher 质量对比 | 将昂贵排序器知识迁移给轻量排序器 | `CE-* + learned ranker` 的学习排序器属于项目适配，当前仍需区分 teacher CE 收益 | 以 CE 成本为约束的 listwise distillation，报告质量—成本 Pareto |
| Late interaction | [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) | query/document token embeddings、MaxSim 统计 | MS MARCO relevance | late interaction 与 full cross-encoder、DPR 的质量/吞吐比较 | 将 query/document 编码与 token-level MaxSim 分离 | 项目已建立 late-interaction pilot，但未把它当作 CE 结论 | 用 late interaction 作为 CE-40 同质量成本控制，并测真实 p95 |
| Late interaction 扩展 | [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) | 压缩 token 表示、MaxSim、候选深度 | relevance + distillation | 压缩索引、检索延迟和质量；ColBERT 为基线 | residual compression 与更高效 late interaction | 可作为 CE 前的低成本候选扩展 | 预注册压缩率和候选池预算，避免用确认集调参 |
| 成本感知重排 | [Efficient Neural Ranking Using Learned Query Difficulty](https://arxiv.org/abs/2104.08067) | query difficulty、初始分数、候选质量预测 | relevance gain / query difficulty labels | 以 query-level 预算分配与固定深度比较 | 根据难度分配更多神经计算 | Router 的 gate 是该思路在 CE pairs 上的适配 | 用预算拉格朗日目标同时优化 nDCG、CE pairs 与 p95 |
| 级联预算优化 | [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) | 中间状态、置信度、剩余计算步数 | 任务损失与 ponder cost | 将计算步数惩罚加入目标；固定步数模型为基线 | 以 ponder cost 学习动态计算量 | 项目把同样思想适配为 CE pairs 与 p95 预算，但尚未改动 CE 内部层数 | 预注册预算点 0/25/50/70/90/100%，用配对 CI 比较 |

## 对 RFC-0001 的约束

当前实验把 CE-0、CE-28、CE-40、学习排序器和 Router 的收益分开报告。固定 CE 的质量提升归因于 CE 动作本身；只有在同一成本预算下逐查询选择相对最佳固定策略仍有优势时，才归因于 Router。HTTP 记录存放在本地 artifacts，仓库只提交汇总、协议、哈希和复现脚本。
