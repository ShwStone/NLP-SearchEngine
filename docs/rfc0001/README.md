# RFC-0001：BM25-first Router 研究包

本轮结论：**不默认推广当前模型**。相对现有规则提高了检索质量，但dev仅节省11.0%向量调用，
真实HTTP无可靠加速，256条额外迁移查询上的质量损失超预算。详见 [RESULTS.md](RESULTS.md)。
这是一轮可审阅研究交付，不等于整个RFC验收完成。

- [目标与执行顺序](GOAL.md)
- [8篇文献矩阵](LITERATURE.md)及[source-verification.json](source-verification.json)
- [至少两份方案卡与选型](PROPOSALS.md)
- [完整结果、局限与后续](RESULTS.md)
- 完整逐查询证据：`evaluation/bm25_evidence/`（gzip文本，不含大语料、权重或索引）。

## 无模型/数据即可核对

```bash
python3 tests/run.py
python3 -m unittest discover -s evaluation
python3 -m evaluation.report_bm25_router --root evaluation/bm25_evidence --out /tmp/router-report.json
```

证据审计检查压缩文件SHA256、冻结模型、dev表、外层组覆盖、逐查询算术、HTTP请求与fallback及进程退出记录。
**复算检查证据自洽，不等于在新机器重新完成真实检索/训练。** 模型/数据来源及环境版本随证据提供。

## 完整研究复跑（需要历史冻结资产）

当前GitHub代码库没有大语料、索引、模型或旧候选capture。以下命令假设已有原研究工作区，
其中的相对文件名/校验和见证据provenance；不能把本文命令误当成从空目录下载全套数据的脚本。
需要Python>=3.10、`requirements-router-research.txt`；真实部分另需硬件适配torch及`requirements-offline.txt`对应依赖。
本轮实测Python3.10.8、sklearn1.5.2、torch2.1.2+cu118、transformers4.46.3、Qdrant1.16.0，详见证据runtime.json。

```bash
# 替换为包含 artifacts/、.runtime/ 的历史研究工作区
RESEARCH_HISTORY=/path/to/NLP-SearchEngine
python -m evaluation.prepare_bm25_router \
  --index "$RESEARCH_HISTORY/artifacts/duretrieval-v1/bm25" \
  --capture "$RESEARCH_HISTORY/artifacts/round3/dev-candidates.jsonl" \
  --audit "$RESEARCH_HISTORY/artifacts/round3/query-audit.json" \
  --qrels "$RESEARCH_HISTORY/artifacts/duretrieval/qrels.tsv" \
  --out artifacts/router-reproduction/dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m evaluation.train_bm25_router \
  --table artifacts/router-reproduction/dev/table.jsonl --out artifacts/router-reproduction/study
```

训练时所有模型/特征/阈值选择都在dev分组内完成。大capture中的原hybrid排名不用于生成本轮标签；
代码先重新计算BM25并逐条核对，再按c=60等权融合。JSON导出与sklearn逐输入比对。
路径/序列化顺序/数值库差异可能影响新导出文件的SHA256；原冻结hash绑定的是本轮已存档文件，不要求重新训练出的文件字节完全相同。

## 复跑当前冻结模型的真实验证

先把证据解压到本地 `artifacts/router-replay/`（保留目录结构，`.gz`解压去掉后缀），或直接使用本机原 `artifacts/rfc01/`。
以下用存档的原模型，不先重训并替换freeze中的hash。新输出目录必须不存在。
真实编码器使用RFC-0004中的共享BGE适配器；它不复制进本RFC生产代码，也不以另一个reader/优化混淆Router效果。
`--adapter-root` 可指向已准备的rfc04 checkout；如共享adapters已合入环境，可省略。
实验会启动并停止自己的Qdrant/HTTP进程；storage必须是既有版本兼容、当前未被其他进程占用的目录。
代码只查询这些既有collections，不覆盖/重建它们。Qdrant自身启动/优化仍会写storage，运行前应按日常流程备份重要产物。

```bash
RESEARCH_HISTORY=/path/to/NLP-SearchEngine
RESEARCH_EVIDENCE=artifacts/rfc01
RESEARCH_ADAPTERS=/path/to/NLP-SearchEngine-rfc04
python -m evaluation.benchmark_bm25_router \
  --index "$RESEARCH_HISTORY/artifacts/duretrieval-v1/bm25" \
  --manifest "$RESEARCH_HISTORY/artifacts/round3/du-vectors/vector_manifest.json" \
  --storage "$RESEARCH_HISTORY/.runtime/round3/storage" \
  --model "$RESEARCH_EVIDENCE/study/model.json" --freeze "$RESEARCH_EVIDENCE/FREEZE.json" \
  --model-path "$RESEARCH_HISTORY/.runtime/models/bge-small-zh-v1.5" \
  --qdrant-binary "$RESEARCH_HISTORY/.runtime/round3/qdrant" \
  --queries "$RESEARCH_HISTORY/artifacts/round3/queries.dev.jsonl" \
  --qrels "$RESEARCH_HISTORY/artifacts/duretrieval/qrels.tsv" \
  --adapter-root "$RESEARCH_ADAPTERS" --out artifacts/router-replay/http-dev

python -m evaluation.prepare_router_transfer \
  --history-root "$RESEARCH_HISTORY" --model "$RESEARCH_EVIDENCE/study/model.json" \
  --freeze "$RESEARCH_EVIDENCE/FREEZE.json" --out artifacts/router-replay/transfer-data
python -m evaluation.benchmark_bm25_router --transfer \
  --index "$RESEARCH_HISTORY/artifacts/round5/cmedqa-json" \
  --manifest "$RESEARCH_HISTORY/artifacts/round5/cmedqa-external/vectors/vector_manifest.json" \
  --storage "$RESEARCH_HISTORY/artifacts/round5/cmedqa-external/qdrant-storage" \
  --model "$RESEARCH_EVIDENCE/study/model.json" --freeze "$RESEARCH_EVIDENCE/FREEZE.json" \
  --model-path "$RESEARCH_HISTORY/.runtime/models/bge-small-zh-v1.5" \
  --qdrant-binary "$RESEARCH_HISTORY/.runtime/round3/qdrant" \
  --queries artifacts/router-replay/transfer-data/queries.jsonl \
  --qrels artifacts/router-replay/transfer-data/qrels.tsv \
  --adapter-root "$RESEARCH_ADAPTERS" --out artifacts/router-replay/http-transfer
```

不要基于这256条迁移结果反复修改模型、阈值或查询选择；新的探索需另登记dev研究，并重新冻结真正独立的验证。

## 接口与协作边界

`Router` / `QueryFeatures.statistics` 是供RFC-0003审阅的局部原型，并非已冻结跨包契约。
`SearchEngine(..., router=GainRouter(path))` 显式启用；默认仍为RuleRouter，CLI未自动启用新模型。
绑定要求与训练一致的BM25、candidate/RRF、索引manifest、embedding、HNSW参数；
模型迁移必须显式 `allow_transfer=True`，并不表示模型已验证可迁移。
只在adaptive执行统计/预测；显式约束由engine优先保护，故障按原契约回退，API仍限定BM25/无扩展。

需要团队确认：最终数据/标注划分、监督学习许可与正式质量/成本预算、0003的策略接口、
以及是否另开融合/动作集实验。本轮没有修改0004 PR，也没有擅自发布新的研究PR或认领评论。
