# NLP-SearchEngine

基于 **BM25 + 向量数据库 + 两阶段 Router** 的中文自适应混合搜索引擎（NLP 大作业）。离线构建 BM25 与向量数据库，在线按需触发向量检索，用 RRF 融合排名。

- [设计与分工](DESIGN.md)
- [跨组 API 契约](contracts/RETRIEVAL.md)
- [开题汇报 PDF](slides/proposal.pdf) / [Typst 源文件](slides/proposal.typ)
- [实施计划与历史讨论](PLAN.md)
- [当前 TODO、证据边界与下一步](TODO.md)

## 目录结构

```text
contracts/    双方共有：analyzer、IndexReader、向量/编码 Protocol、HTTP schema、conformance
  retrieval.py          A2：EmbeddingSpec / VectorManifest / TextEncoder / VectorWriter / VectorReader
  api.py                B：四种 mode、路由回显、阶段耗时与 JSON Schema
  RETRIEVAL.md          字段、发布、错误和版本兼容说明
offline/      离线侧：BM25 与向量数据库都在这里构建
  build.py              倒排表、tf/df、长度/位置与 BM25 参数
  build_hybrid.py       统一双索引构建编排（依赖注入）
  vector_build.py       文档批量编码、Qdrant upsert/快照校验与 manifest 发布
  disk_index.py         IndexReader 的真实磁盘实现
online/       在线侧：查询编码、BM25、Router、向量召回、RRF、高亮、HTTP
  embeddings.py         可选 Sentence-Transformers 编码器（固定 revision）
  router.py             规则选择器，不读 qrels、不调用大模型
  fusion.py             RRF 排名融合
  vector_index.py       内存精确向量替身与 Qdrant 只读适配器
  engine.py             lexical / dense / hybrid / adaptive 四模式编排
web/          原生浏览器演示界面
evaluation/   只通过 HTTP /search 的 TREC 评测器
fixtures/     tiny_corpus.jsonl 与既有 tiny_index（仅词法产物）
tests/        标准库 runner；安装 pytest 后也可直接运行 pytest
```

纪律：跨组只依赖 contracts/；在线不 import offline.*（除 DiskIndex），离线不 import online.*。BM25 离线“构建”指索引/统计，查询打分仍在在线侧。

本地数据、下载权重、生成索引和评测输出分别建议放在 `data/`、`models/`、`index/`、`runs/`。
这些目录、Qdrant 存储／快照、模型权重和 `.env` 凭据均由 `.gitignore` 排除；小型 JSON/JSONL 测试语料仍保留在受版本控制的 `fixtures/` 中。
可提交 `.env.example` 等不含真实凭据的配置模板。

## 快速开始：词法闭环与契约测试（无额外依赖）

```bash
# 1. 构建词法索引（保留旧目录，不在框架更新中修改既有 fixtures）
python3 -m offline.build --corpus fixtures/tiny_corpus.jsonl \
    --out /tmp/nlp-tiny-bm25 --dataset tiny --k1 1.2 --b 0.75

# 2. 词法 conformance
python3 -m contracts.conformance /tmp/nlp-tiny-bm25 fixtures/tiny_corpus.jsonl

# 3. 全部测试：包含离线双索引、假向量、路由与真实本地 HTTP 请求
python3 tests/run.py
# 可选：浏览器高亮逻辑的 Unicode/转义回归测试（Node.js 18+，无需 npm 依赖）
node --test tests/test_web.cjs

# 4. 启动词法服务
python3 -m online.server --index /tmp/nlp-tiny-bm25 --port 8000

curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
    -d '{"query": "心肌梗死 症状", "mode": "lexical", "k": 5}'
curl -s localhost:8000/index/info
# 浏览器打开 http://127.0.0.1:8000/ 查看演示界面
```

默认只监听 `127.0.0.1`。需要从其他机器访问时，显式添加 `--host 0.0.0.0`，
再通过 `http://<服务器IP>:8000/` 打开界面，并确认防火墙允许该端口。
演示服务没有认证，请勿直接暴露到不可信网络。

当前 CLI 的 vector_ready=false。dense/hybrid 请求返回 503；adaptive 如需语义分支会明确标记 fallback，不能把降级结果当成成功的混合评测。

## 扩充后的构建/在线接线

真实 embedding 和 Qdrant 适配器是可选依赖，默认安装不下载模型。安装
`sentence-transformers` 与 `qdrant-client>=1.16`，并运行 Qdrant 1.16+ 后，可以用固定 manifest 启动完整向量服务：

```python
# 离线：writer 配置相同 out_dir 下的快照导出目录
bundle = build_hybrid(corpus_path, out_dir, dataset,
                      encoder=document_encoder, writer=vector_writer,
                      collection="dataset-build-id")

# 在线：仅加载/查询同版产物，不建库
engine = SearchEngine(lexical_reader, encoder=query_encoder, vectors=vector_reader,
                      config=RetrievalConfig(candidate_k=200, rrf_c=60))
response = engine.search(SearchRequest(query="手机掉水里怎么办", mode="adaptive"))
```

在线启动时传入同一版本的向量 manifest：

```bash
python3 -m online.server --index /path/to/bm25 \
  --vector-url http://127.0.0.1:6333 \
  --vector-manifest /path/to/vector_manifest.json
```

服务会从 manifest 读取模型 ID、固定 revision、维度、前缀和截断配置，并校验
Qdrant collection 中持久化的完整构建元数据、真实维度／距离和精确文档数，随后校验与倒排索引的语料指纹。
缺少构建元数据的旧 collection 必须用新 writer 重建；不会只凭同维度或外部 manifest 放行。

四种请求模式：

| mode | 行为 |
|---|---|
| lexical（默认） | 仅 BM25；兼容 TF-IDF |
| dense | 仅向量检索 |
| hybrid | 每次 BM25 + 向量，RRF 融合 |
| adaptive | 先 BM25，规则 Router 决定是否语义补充与融合 |

HTTP 响应 v2.0 新增 routing / timings_ms；普通自然语言查询使用 OR 词项候选。
显式 `AND`、`OR`、`NOT`、括号和双引号短语在词法层过滤，hybrid 会在融合前保留硬约束，
dense 对显式约束返回 400。详细参数、分页和兼容规则见契约文档。

## HTTP 评测

评测器只调用公开 HTTP API，不导入检索实现。查询文件支持 `qid<TAB>query` 或 JSONL，
qrels 使用 TREC 四列格式：

```bash
python3 -m evaluation.evaluate \
  --endpoint http://127.0.0.1:8000/search \
  --queries queries.tsv --qrels qrels.tsv \
  --output-run runs/bm25.trec --include-per-query
```

它默认召回 100 条结果，分别计算 nDCG@10、Recall@10、MRR@10、MAP@100，以及完整请求的平均、p50、p95 延迟。
输出包含向量尝试、成功和 fallback 计数／比例；正式实验可加 `--fail-on-fallback` 拒绝降级结果。已有 TREC run
也可以离线评分，但不推断其中不存在的路由或延迟信息。`evaluation.prepare_dataset` 可校验外部 DuRetrieval 文件并生成固定划分，
`evaluation.compare` 可对两组 TREC run 做可复现的 query-level paired bootstrap。详见
[evaluation/README.md](evaluation/README.md)。

### 教师建议：路由对比与自建测试集

Router 是本项目的核心实验变量：固定 hybrid 每次访问两路，adaptive 只改变是否触发向量分支，模型、候选池、RRF 参数和数据划分必须保持一致。先在 dev 冻结规则阈值，再在独立 test 上保存四种模式的 TREC run、逐查询路由原因和向量触发/成功/fallback 统计；用 `evaluation.compare` 对 fixed-hybrid 与 adaptive 做 paired bootstrap。当前 prefix20k 的真实结果和区间只属于工程冒烟，详见 [evaluation/RESOURCES.md](evaluation/RESOURCES.md)。

为响应“自行构造网络测试集”的要求，仓库提供了有边界的 crawler，但不会伪造 qrels。它默认只抓 HTTP(S)、尊重 robots、限制同域、深度、文档数、响应大小和请求间隔，先生成未标注 corpus 与人工标注模板：

```bash
python -m evaluation.crawl_web crawl \
  --seed https://example.org/ \
  --out-corpus data/crawl/corpus.jsonl \
  --annotation-template data/crawl/annotations.template.jsonl
```

组员人工填写 `query` 和 `relevance`（0/1/2）后，再生成评测器可读的 queries/qrels：

```bash
python -m evaluation.crawl_web finalize \
  --corpus data/crawl/corpus.jsonl \
  --annotations data/crawl/annotations.jsonl \
  --queries-out data/crawl/queries.jsonl \
  --qrels-out data/crawl/qrels.tsv
```

模板本身不是评测集；没有人工正相关标注时命令会拒绝输出 qrels。数据和生成结果位于被 `.gitignore` 排除的目录，不应提交到仓库。

外部数据版本、下载限制和课程规模校验见 [evaluation/RESOURCES.md](evaluation/RESOURCES.md)。

## 当前状态与后续工作

**已可用**：bigram 占位分词、词法构建/读取、BM25/TF-IDF、普通 OR 查询、布尔/短语过滤、Top-K/高亮/HTTP；双索引构建编排、Qdrant upsert/快照发布检查；规则 Router、RRF、四模式编排、降级和阶段耗时；可选 Sentence-Transformers/Qdrant 适配器；浏览器 UI、HTTP-only 评测器与标准库测试。

**已完成的资源准备**：已核验 RocketQA 官方 DuReader-Retrieval baseline archive，确认其包含
8,096,668 个 passage、2,000 个 dev 查询和 9,839 条去重 qrels，并提供专用流式适配器、SHA256
清单和 dev-only 边界。另已在集群上用固定的 20,000 文档前缀、19 条有标注查询、真实 384 维
Sentence-Transformers 模型和 qdrant-client local embedded collection 完成 lexical、dense、hybrid、
adaptive 四模式及 20,000 次 paired bootstrap。该公开 archive 没有官方 test labels，prefix smoke
不能代表课程全量；该次真实模型运行使用 qdrant-client 1.16.0 local embedded 的精确／全扫描配置，
没有宣称 HTTP Qdrant/HNSW 或 8M 规模语义实验。数据、模型和索引产物不随
仓库提交，仓库已提供校验、固定划分和区间计算入口。

RM3 已实现为确定性的本地伪相关反馈扩展；SLM 仍需要外部固定模型与离线缓存，返回 501。扩展不改变显式布尔/短语约束。内存测试通过不代表真实语义质量或 HNSW 性能已经验收。
