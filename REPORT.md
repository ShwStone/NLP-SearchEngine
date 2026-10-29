# 中文自适应混合搜索引擎实现报告

## 1. 项目目标与边界

本项目面向 NLP 课程大作业，提供一个中文检索系统：离线构建手写倒排索引，在线用 BM25 排序，并通过可选的 embedding 与 Qdrant 向量库补充词面不重合的结果。系统支持 lexical、dense、hybrid、adaptive 四种模式。自然语言查询默认使用词项并集；显式约束必须先在词法层过滤，语义召回不能放宽约束。

提案中的 DuRetrieval 全量实验需要单独下载语料、queries 和 qrels。仓库只保留 tiny fixture 和可复现代码，不把约 100,001 篇文档或模型文件提交进版本库。

资源核验见 [evaluation/RESOURCES.md](evaluation/RESOURCES.md)。已从 RocketQA 官方下载脚本指向的公开 CDN 获取并核验 baseline archive：8,096,668 个 passage、2,000 个 dev 查询和 9,839 条去重 qrels（原始答案段落链接 9,863 条）。该公开 archive 没有官方 test labels，与提案中要求的课程快照和官方 DuReader 版本不能混用；本次集群已完成固定 BGE revision 的 GPU 编码、受限前缀四模式验证，以及真实 Qdrant 1.16.0 服务端的小批快照恢复。8M 规模 HNSW/全量语义实验仍未完成，报告不伪造这些结果。

## 2. 已完成实现

### 词法与查询处理

- `contracts/analyzer.py` 提供离线、在线共用的 bigram/ASCII 分词和字符偏移。
- `offline/build.py` 写入 docs、postings、offsets、stats 和带 SHA256 的 manifest。
- BM25、TF-IDF、OR 候选、稳定 Top-K 和高亮均通过 `IndexReader` 协议工作。
- 查询解析器支持 `AND`、`OR`、`NOT`、括号和双引号连续短语，使用 NOT、AND、OR 优先级。
- 短语根据倒排位置连续匹配；布尔结果作为硬过滤传入 BM25。hybrid 在 RRF 前再次过滤向量结果，dense 对显式词法约束返回 400。

### 语义与混合检索

- `contracts/retrieval.py` 固定模型 revision、维度、归一化、前缀、截断和 cosine 约束。
- `online/embeddings.py` 提供懒加载的 Sentence-Transformers 编码器，缺少可选依赖时返回清晰的可用性错误。
- `offline/vector_build.py` 的 Qdrant writer 会创建新 collection、批量 upsert、核对点数并下载带 SHA256 的快照。
- `online/vector_index.py` 的 Qdrant reader 校验 collection 中持久化的完整构建身份、真实维度／距离和精确点数，并在命中时核对外部 doc_id 和 cosine 分数。Qdrant 与客户端均要求 1.16+；旧的无元数据 collection 必须重建。
- CLI 可通过 `--vector-url` 与 `--vector-manifest` 启用同版本向量后端；不提供这两个参数时保持零依赖词法服务。

### HTTP 与界面

- `POST /search` 返回 API v2.0 的解析树、结果、高亮、routing 和阶段耗时。
- `GET /health` 与 `GET /index/info` 提供健康状态、索引 manifest 和向量能力。
- `web/` 是原生 HTML/CSS/JavaScript 界面，支持查询、模式、Top-K、结果高亮、路由和耗时展示。服务端限制静态路径在 web 根目录内，拒绝路径穿越和符号链接逃逸。

### 评测器

`evaluation/evaluate.py` 只通过公开 HTTP API 发送查询，不导入检索内部模块。它读取 TREC qrels 和 TSV/JSONL queries，输出 TREC run，并计算 nDCG@10、Recall@10、MRR@10、MAP@100、平均延迟、p50 和 p95。已有 run 可以脱离服务离线评分。`evaluation/prepare_dataset.py` 负责外部数据校验、哈希和固定划分，`evaluation/compare.py` 提供可复现的逐查询 paired bootstrap。

## 3. 验证结果

在 tiny fixture 上执行：

```bash
python3 tests/run.py
python3 -m unittest discover -s evaluation -p 'test_*.py'
node --test tests/test_web.cjs  # 可选，Node.js 18+
python3 -m offline.build --corpus fixtures/tiny_corpus.jsonl \
  --out /tmp/nlp-tiny-bm25 --dataset tiny --k1 1.2 --b 0.75
python3 -m contracts.conformance /tmp/nlp-tiny-bm25 fixtures/tiny_corpus.jsonl
```

审阅修复后，54 项仓库契约／适配器测试、22 项评测器和数据准备测试、5 项浏览器高亮测试全部通过，tiny 索引 conformance 通过。Qdrant 测试使用 SDK double 验证发布／拒绝路径；另用 `qdrant-client==1.16.0` 的本地内存后端验证了真实 SDK 元数据读写和模型版本错配拒绝。此前本地内存后端的 snapshot API 明确不支持，因此当时没有据此宣称真实服务恢复；随后已在独立 Qdrant 1.16.0 服务端完成小批快照恢复，详见第 6 节。启动服务后，lexical 搜索、`/health`、`/index/info`、浏览器静态资源和 dense 未配置时的 503 行为均已验证。历史上对 tiny 磁盘索引重复执行 1,000 次 BM25 查询测得 p50 约 0.10 ms、p95 约 0.14 ms；这只是小样本工程冒烟，不能替代 DuRetrieval 全量结果。

历史上另用公开 C-MTEB DuRetrieval 快照（5000 篇文档、100 条查询、425 条 qrels）做了外部闭环冒烟。按固定划分取 80 条测试查询，旧版 Top-10 HTTP 请求下 BM25 得到 nDCG@10=0.8065、Recall@10=0.8194、MRR@10=0.8872，平均延迟 4.33 ms、p50 3.83 ms、p95 6.65 ms。修复审阅问题时发现旧评测器默认只召回 10 条，却标记 MAP@100，因此撤下该 MAP 数值；新的默认评测会召回 100 条，并需要重新测量 MAP 和完整请求延迟。该快照不是课程要求的 100001/2000/9839 版本，这些历史数字只验证管线，不代表最终成绩。

为验证官方资源上的闭环，另对官方 dev corpus 的固定 20,000 篇前缀运行了 19 条命中查询。BM25 冒烟结果为 nDCG@10=0.5032、Recall@10=0.5789、MRR@10=0.4781、MAP@100=0.4920，平均延迟 57.85 ms、p50 52.59 ms、p95 79.15 ms；TF-IDF 和 RM3 也分别生成了独立 TREC run。该结果只覆盖官方全量的一个文档前缀和 19 条查询，不能写成全量成绩；向量未就绪时 adaptive 的 fallback 比例为 10/19。

## 4. 全量实验流程

1. 固定 DuRetrieval 数据版本、查询划分和 qrels；先在开发集冻结 Router 阈值。
2. 用同一 BM25 参数和语料分别运行 BM25、纯向量、固定 hybrid、自适应 hybrid。
3. 对每组保存 TREC run 和每查询延迟，使用评测器报告 nDCG@10、Recall@10、MRR@10、MAP@100、p50/p95。
4. 使用 `evaluation.compare` 对自适应与固定 hybrid 的逐查询差值做 paired bootstrap，并单独报告向量触发、成功和 fallback 比例。

示例：

```bash
python3 -m evaluation.evaluate \
  --endpoint http://127.0.0.1:8000/search \
  --queries data/test.queries.tsv --qrels data/test.qrels.tsv \
  --output-run runs/bm25.trec --include-per-query
```

该命令不会下载数据或模型，也不会把评测结果写入受版本控制的源码目录，除非调用者明确指定输出路径。

## 5. 当前限制

默认分词器仍是可解释的 bigram 占位实现；公开 baseline 只有 dev 标注，真实分词对齐、Qdrant HNSW 参数和全量性能数字仍不能从本次前缀烟雾推断。集群上已经完成固定 BGE revision 的 CUDA 编码、20k 前缀上的四模式 HTTP 冒烟和 paired bootstrap；这些四模式结果使用 `memory-exact` 向量读取器，不能解释为 Qdrant HNSW 性能。真实 Qdrant server 的小批 snapshot/clean-restore/read-only 闭环已单独验收，但尚未完成 8M 规模 HNSW 与课程全量成绩。2026-10-29 已将有界 `evaluation.crawl_web` 工作流移入当前分支，并从公开中国政府网抓取 30 篇、depth=2 的 393 篇、depth=3 的 799 篇和 depth=4 的 1,029 篇同域文档；随后从公开 sitemap 增加 108 篇页面，按 URL 合并为 1,137 篇语料；另完成新华网 100 篇跨站点未标注抓取试验，作为抓取器的补充收据。对 10 个查询的 100 条候选已按用户授权进行 assistant 语义复核并生成正式 qrels，标签来源明确为 assistant，不是独立人工 gold labels。扩大抓取仍未满足 20k 自爬验收。RM3 已作为确定性的本地伪相关反馈扩展接入；SLM 仍需要外部固定模型与离线缓存，返回 501。

在 1,137 篇 assistant-reviewed 语料上，BM25 lexical 的 nDCG@10/Recall@10/MRR@10/MAP@100 为 `0.950197/1.000000/1.000000/0.918422`，RM3 为 `0.945802/0.980000/1.000000/0.916014`。10,000 次 paired bootstrap（seed `20261029`，RM3−lexical）nDCG 差值为 `-0.004394`，95% CI 为 `[-0.020271, 0.008033]`。adaptive 在无向量后端时 5/10 查询发生 fallback；dense 请求按契约返回 503。该结果只用于验证自爬语料上的评测闭环，不代表 20k 或 8M 全规模质量。

## 6. 2026-10-29 集群验收：路由比较与证据边界

本次验收固定使用官方 RocketQA DuReader baseline archive（8,096,668 passages、2,000 dev queries、9,839 dev qrels；archive SHA256 为 `b051d3d966f7b0288aed95a923fc3b706176767d8bdcff8aa8fe5fff84b20662`）。为避免把 v1 全量内存 builder 当成 8M 生产构建，只取 dev corpus 前 20,000 篇文档，seed `20261028` 得到 train/dev/test=3/7/9；该前缀 corpus SHA256 为 `7e2a20ca5f7a75ece81efffbb45f10983c90ebaf500f60f60a4a53f9c775a55bfc8`。官方资源与固定划分的完整哈希清单见 [evaluation/RESOURCES.md](evaluation/RESOURCES.md)。

语义侧使用 `BAAI/bge-small-zh-v1.5` revision `7999e1d3359715c523056ef9478215996d62a620`，512 维、CLS pooling、512 token 截断、L2 归一化，运行在 NVIDIA A800 80GB、PyTorch 2.3.0、Transformers 4.57.1。模型权重和 tokenizer 的哈希、CUDA 编码有限性与重复编码检查已记录在资源文档；模型没有进入 Git。

先在 dev 上冻结 `candidate_k=200`、`rrf_c=60`、`min_vocab_coverage=0.7`、`min_relative_gap=0.1` 和 `rules-v1`，再只在 test 上比较四种模式。test 的 9 条查询结果为：

| mode | nDCG@10 | Recall@10 | MRR@10 | MAP@100 | mean/p50/p95 ms | vector success / fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| lexical | 0.625659 | 0.666667 | 0.611111 | 0.626455 | 50.90 / 49.29 / 61.09 | 0 / 0 |
| dense | 0.958992 | 1.000000 | 0.944444 | 0.944444 | 198.48 / 202.67 / 218.39 | 9 / 0 |
| hybrid | 0.791378 | 1.000000 | 0.730688 | 0.730688 | 200.81 / 200.91 / 223.86 | 9 / 0 |
| adaptive | 0.713333 | 0.888889 | 0.659259 | 0.666667 | 133.46 / 169.74 / 229.49 | 5 / 0 |

这些请求通过公开 HTTP evaluator 生成了四份 TREC run。dense/hybrid/adaptive 的向量后端是带真实 BGE 查询编码的 NumPy `memory-exact` 全扫，因此结果用于验证 Router、RRF、routing 和评测闭环，不用于声称 Qdrant HNSW 性能。adaptive 的 5/9 向量触发全部成功，未发生 fallback。

配对 bootstrap 的差值定义为后者减前者，样本数 10,000、seed `20261028`。lexical→hybrid 的 nDCG/Recall/MRR/MAP 差值及 95% CI 分别为 `+0.165719 [0.037037, 0.294400]`、`+0.333333 [0.000000, 0.666667]`、`+0.119577 [0.015873, 0.238095]`、`+0.104233 [0.011111, 0.222222]`；lexical→adaptive 分别为 `+0.087674 [0.000000, 0.207683]`、`+0.222222 [0.000000, 0.555556]`、`+0.048148 [0.000000, 0.122222]`、`+0.040212 [0.000000, 0.098413]`。只有 9 个 test queries，CI 不能外推到课程全量。

Qdrant server 侧已完成真实小批验收：Qdrant `1.16.0` 服务端与 `qdrant-client==1.16.0` 由仓库自身 `QdrantSnapshotWriter` 创建 `rfc4_writer_20261029b` collection，写入 3 个 4 维 cosine 向量及完整 build metadata，HNSW 参数记录为 `m=16`、`ef_construct=100`。生成快照 SHA256 为 `899eb64518664a726612a3e6e3e9a13037c4160ae9342ba21f08468fb5c53ac0`；随后在空存储目录、独立服务端口用 `--snapshot` 恢复，服务端版本仍为 `1.16.0`，exact count=3，内部 ID 为 `[0, 1, 2]`，payload 外部 ID 为 `doc-a/doc-b/doc-c`，恢复 metadata 与 manifest 完全一致，仓库 `QdrantVectorIndex` 只读查询返回 `(0, doc-a), (2, doc-c), (1, doc-b)`。此前未归一化向量也被 `validate_vector` 正确拒绝。该证据满足真实小批 server snapshot/clean-restore/read-only 闭环，但不代表 8M 规模性能。课程目标版本的独立 test labels、团队自爬测试集和 8M 规模索引仍是后续工作，不能把本节前缀 smoke 写成正式课程成绩。
