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

## 复核入口

关键工具位于 `evaluation/`：

- `study_prebge_learned_gate_v4.py` 和 `prebge_nested.py`：嵌套动作与门控训练；
- `train_nested_prebge_final.py`：全部开发组最终拟合；
- `audit_nested_prebge_final.py`、`audit_nested_prebge_http.py`、`audit_nested_prebge_confirmation.py`：独立重放；
- `runtime_nested_prebge.py`：研究运行时；
- `stable_dense_topk.py`、`gte_document_tokens.py`：稳定候选顺序和经过全量输入核对的文档 token 缓存。

本地审计目录中的确认报告和冻结清单不提交模型权重、原始语料或 qrels；这些文件通过 SHA256 绑定保存。最终回归测试为 38 项通过。

此前无效的 legacy pre-BGE cascade 证据不属于本提交；它存在数据泄漏、错误配对基线和确认集重叠问题，已在本轮审计中明确撤回。
