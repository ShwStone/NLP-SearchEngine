# RFC-0001 最终 Router 结果

关联 issue：[RFC-0001：搜索 Router 论文调研、方案征集与创新验证](https://github.com/ShwStone/NLP-SearchEngine/issues/3)

本次提交补充了一个可复核的 BM25-first Router 研究实现。它使用固定 equal-RRF hybrid 作为主基线，先用 BM25、dense 检索统计和查询特征选择 cheap/heavy 动作，再对选择的候选执行 BGE TensorRT 重排。默认生产路由行为不改变，研究 Router 通过独立运行时和明确的模型绑定启用。

## 最终验证

所有选择、训练和门控均使用查询组嵌套流程；最终模型在全部开发组上重新拟合，并通过独立重训核对。确认集在模型和代码冻结后按固定哈希顺序一次选择，HTTP 请求完成后才读取 qrels，确认标签没有回流训练或调参。

| 集合 | nDCG@10 相对提升 | Recall@10 相对提升 | 配对分组 CI 下界 | 动作 | HTTP p95 比值 |
| --- | ---: | ---: | --- | --- | ---: |
| 2499 条嵌套开发集 | +33.9031% | +20.0341% | +0.19845 / +0.13849 | cheap/heavy=105/2394 | 1.01018 |
| 全新 256 条确认集 | +31.9755% | +21.0235% | +0.16139 / +0.11425 | cheap/heavy=69/699 | 0.86510 |

确认集共执行 3840 次真实 HTTP 请求，并同时记录 fixed dense、previous Router 和 compiled hybrid 对照。确认集独立审计重建了 1409 个冻结文件、107 个存储文件、查询暴露清单、哈希抽样、qrels、请求平衡、动作、排名、指标和延迟。

## CE 动作消融与成本对照

维护者要求的 CE 方向补充使用同一固定 MMarco 开发协议、同一 equal-RRF hybrid 基线和冻结 BGE-28/BGE-40 logits 完成。汇总脚本是 `evaluation/study_prebge_ce_ablation.py`，结果保存在本地 `artifacts/rfc01/twenty-percent/ce-ablation-r6/summary.json`；逐查询预测和请求记录没有写入仓库。

| 动作 | nDCG@10 | Recall@10 | 平均 CE pairs | heavy 比例 | nDCG/Recall 配对 CI 下界（相对 CE-0） |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE-0 | 0.62604 | 0.76167 | 0 | 0% | 0 / 0 |
| always-cheap / CE-28 | 0.82527 | 0.90044 | 28 | 0% | +0.18070 / +0.12079 |
| always-heavy / CE-40 | 0.83094 | 0.90922 | 40 | 100% | +0.18704 / +0.12925 |
| CE-28 + 学习排序器 | 0.82819 | 0.90211 | 28 | 0% | +0.18413 / +0.12222 |
| CE-40 + 学习排序器 | 0.83588 | 0.91489 | 40 | 100% | +0.19217 / +0.13535 |
| Router | 0.83249 | 0.90844 | 33.408 | 45.07% | +0.18854 / +0.12891 |

在声明的固定深度集合中，Router 的 33.408 平均 CE pairs 预算下，质量高于成本不超过该预算的最佳固定策略 CE-28（nDCG +0.00722、Recall +0.00800）。这属于“预算内最佳固定策略”比较，不能夸大为精确同预算 Pareto 最优；Router 低于 CE-40 + 学习排序器的质量，因此同质量成本交点仍需更多预注册深度点确认。

四个固定动作和 Router 都在同一真实 HTTP harness 中各执行 48 条查询、3 次重复（每个固定动作 144 次请求；Router 复用同一运行时的 144 条动作请求）。固定 CE-28 原始分数的 CE p50/p95 为 27.82/30.10 ms、端到端为 41.06/44.42 ms；CE-28 + 学习排序器为 28.25/30.40 ms、42.13/46.24 ms；固定 CE-40 原始分数为 40.22/42.82 ms、54.22/58.28 ms；CE-40 + 学习排序器为 39.75/42.42 ms、55.09/59.67 ms；Router 为 40.01/42.79 ms、54.65/59.28 ms。强制动作运行时开关和协议字段位于 `evaluation/runtime_nested_prebge.py` 与 `evaluation/benchmark_nested_prebge_http.py`，独立聚合审计确认这些记录来自同一索引、引擎、批大小和 HTTP harness。

这份结果将固定 CE 收益、学习排序器收益和 Router 的逐查询选择分开归因。MMarco 只作为方法证据，离线索引和自有数据验收仍未替代。

## 复核入口

关键工具位于 `evaluation/`：

- `study_prebge_learned_gate_v4.py` 和 `prebge_nested.py`：嵌套动作与门控训练；
- `train_nested_prebge_final.py`：全部开发组最终拟合；
- `audit_nested_prebge_final.py`、`audit_nested_prebge_http.py`、`audit_nested_prebge_confirmation.py`：独立重放；
- `runtime_nested_prebge.py`：研究运行时；
- `stable_dense_topk.py`、`gte_document_tokens.py`：稳定候选顺序和经过全量输入核对的文档 token 缓存。

本地审计目录中的确认报告和冻结清单不提交模型权重、原始语料或 qrels；这些文件通过 SHA256 绑定保存。最终回归测试为 38 项通过。

此前无效的 legacy pre-BGE cascade 证据不属于本提交；它存在数据泄漏、错误配对基线和确认集重叠问题，已在本轮审计中明确撤回。
