# NLP-SearchEngine

基于 **BM25 + 向量数据库 + 两阶段 Router** 的中文自适应混合搜索引擎（NLP 大作业）。离线构建 BM25 与向量数据库，在线按需触发向量检索，用 RRF 融合排名。

- [设计与分工](DESIGN.md)
- [跨组 API 契约](contracts/RETRIEVAL.md)
- [实验报告](REPORT.md) / [当前交付与待办](TODO.md)
- [RFC-0001 等向量预算与学习型 Router 对照](docs/contributions/rfc0001-router-budget.md)
- [RFC-0004 离线产物交付补充](docs/contributions/rfc0004-delivery-validation.md)
- [RFC-0004 接收机实测与合并修正](docs/contributions/rfc0004-receiver-acceptance.md)
- [在线服务部署与验收](docs/online-serving.md)
- [RFC-0003 在线服务验收记录](docs/contributions/rfc0003-online-acceptance.md)
- [开题汇报 PDF](slides/proposal.pdf) / [Typst 源文件](slides/proposal.typ)
- [实施计划与历史讨论](PLAN.md)

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
  provenance.py         无凭据的输入/语料/模型/产物 provenance 与成本报告
  validate.py            已发布双索引的严格哈希、指纹和快照校验器
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

下方为 Bash 示例。Windows 可将 `python3` 换成已安装解释器对应的 `python`，
将 `/tmp/...` 换成版本库外的绝对目录，并将带 `\` 换行的命令合并为一行。Windows PowerShell 的 `curl` 可能是 `Invoke-WebRequest` 别名，HTTP 请求请使用下方的 `Invoke-RestMethod` 示例。

```bash
# 1. 构建词法索引（保留旧目录，不在框架更新中修改既有 fixtures）
python3 -m offline.build --corpus fixtures/tiny_corpus.jsonl \
    --out /tmp/nlp-tiny-bm25 --dataset tiny --k1 1.2 --b 0.75

# 2. 词法 conformance
python3 -m contracts.conformance /tmp/nlp-tiny-bm25 fixtures/tiny_corpus.jsonl

# 3. 核心契约与评测器
python3 tests/run.py
# 扩展回归需要 Router 研究依赖、pytest、PyTorch（CPU 版即可）和 ijson
python3 -m pip install -r requirements-router-research.txt pytest torch ijson==3.4.0.post0
python3 -m unittest discover -s evaluation -p 'test_*.py'
# 同时收集函数式和 unittest 评测用例
python3 -m pytest evaluation tests/test_gain_router.py tests/test_replay_router_gain.py
python3 -m unittest discover -s tests -p 'test_*.py'
# 可选：浏览器高亮逻辑的 Unicode/转义回归测试（Node.js 18+，无需 npm 依赖）
node --test tests/test_web.cjs

# 4. 启动词法服务
python3 -m online.server --index /tmp/nlp-tiny-bm25 --port 8000

curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
    -d '{"query": "心肌梗死 症状", "mode": "lexical", "k": 5}'
curl -s localhost:8000/index/info
# 浏览器打开 http://127.0.0.1:8000/ 查看演示界面
```

Windows PowerShell 请求示例（服务已启动）：

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/search' -Method Post -ContentType 'application/json; charset=utf-8' -Body (@{query='心肌梗死 症状';mode='lexical';k=5} | ConvertTo-Json -Compress)
```

默认只监听 `127.0.0.1`。需要从其他机器访问时，显式添加 `--host 0.0.0.0`，
再通过 `http://<服务器IP>:8000/` 打开界面，并确认防火墙允许该端口。
演示服务没有认证，请勿直接暴露到不可信网络。

当前 CLI 的 vector_ready=false。dense/hybrid 请求返回 503；adaptive 如需语义分支会明确标记 fallback，不能把降级结果当成成功的混合评测。

## RFC-0004 离线交付校验

`offline.build_hybrid` 使用同文件系统临时目录构建，完成后才发布 BM25 与向量 manifest；
新构建失败会清理本次目录，已有非空产物不会被覆盖。成功交付会额外写出
`build_provenance.json` 和 `build_report.json`，记录输入文件 SHA256、ordered corpus
fingerprint、完整 embedding 配置、collection/index 参数、阶段耗时及产物哈希。报告不会
伪造峰值 RAM/GPU；没有作业监控证据时对应字段为空。

```bash
python3 -m offline.validate --build-dir /path/to/released-dual-index
```

严格 validator 要求 A1 的四个 JSONL/统计文件都带 SHA256，并核对 BM25 与向量的文档数、
corpus fingerprint、快照文件和 provenance/report。旧 v1 产物仍可由普通 `DiskIndex` 兼容
读取，但缺少文件摘要的旧产物不能作为 RFC-0004 的生产交付。Qdrant writer 逐批消费向量，
已确认本次创建的新 collection 后续失败时才尝试清理；创建冲突或响应未知时保留远端资源。
快照使用独占 `.part`、大小/哈希校验和不覆盖目标的 hard link 发布，manifest 同样不覆盖；
输出文件系统必须支持 hard link，否则安全拒绝发布。2026-10-29 已在
Qdrant 1.16.0 服务端上完成 3 点小批快照的干净实例恢复和只读适配器验收；这不等于
8M 规模的 HNSW 构建或性能验收。

8M 规模路径和最新资源 pilot 见 [RFC-0004 scale path](docs/RFC0004_SCALE.md)。
`offline.stream_build --work-dir` 可将 SQLite 临时库放到独立磁盘；可选
`offline.compact --streaming` 使用固定 `ijson==3.4.0.post0` 逐事件转换高频词倒排，
保持 compact-v1 与默认零依赖路径兼容。包含 `()` 的零 token 文档会保留其 ID、
空 offsets 和原文，计入全部文档数及 avgdl，不被静默丢弃。

需要跨进程中断恢复时使用 `offline.checkpoint_build --state-dir /scratch/checkpoint`。
它将已落盘的 docs/offsets 长度、输入字节位置和 SQL 记录一同提交，恢复时校验输入、
配置与源码身份，并支持按完整词项边界继续导出 postings。旧 `stream_build` 的临时库
不能作为这种检查点导入。完整 v1 发布后，使用 `offline.checkpoint_compact` 的独立
`--state-dir` 继续可恢复的 compact 转换。`online.server --index /path/to/compact`
经现有 `DiskIndex` 入口自动使用 packed reader，始终核验 compact 文件 SHA256，
同时存在两种 manifest 的目录会被拒绝。详见同一份规模说明及相应专项测试。

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

固定 BGE 构建的正式接线应显式复用已验 SHA 的本地模型目录：

```bash
python3 -m online.server --index /path/to/compact \
  --vector-url http://127.0.0.1:6333 \
  --vector-manifest /path/to/vector_manifest.json \
  --bge-model-path /path/to/verified-bge-model --encoder-device cpu
```

集群 GPU 可将 device 改为 `cuda:0`。该路径与离线共用 `adapters.embedding.BGEEncoder`，
固定 revision、CLS、右截断和 FP32，启动前校验同一 model inventory；缺依赖、错模型或
坏哈希明确拒绝。未传 BGE 参数时仍使用通用 SentenceTransformer 路径。无模型接线测试
不替代真实模型的长短文本、padding/batch 一致性和运行环境记录。

版本化部署改用单个 profile 文件：`python3 -m online.server --config /path/to/news20k.json`，
示例见 `deploy/config/`。`--config` 与上面所有 index/vector/retrieval/host/port flag 互斥，
`dataset` 必须与产物 manifest 完全一致；配置路径相对配置文件解析（私有部署配置也可写绝对路径）、
不做环境变量插值。
就绪语义、日志隐私、前台启动与三阶段 HTTP 验收见[在线服务部署与验收](docs/online-serving.md)。

向量构建可以直接读取冻结的标准 corpus，与词法构建并行；必须显式提供原始文件的
SHA256 和文档数。内部 ID 保持物理行序，空行、坏字段、文件变化和错哈希都会拒绝。
这条路径只提供文档，不制造词法 manifest；最终双索引仍须核对相同的有序语料指纹。

```bash
python3 scripts/build_vectors.py --corpus /data/corpus.jsonl \
  --corpus-sha256 <64位小写SHA256> --corpus-n-docs <文档数> \
  --out /output/vectors --collection <新的版本化collection> \
  --model-path /path/to/verified-bge-model --device cuda:0 \
  --hnsw-full-scan-threshold-kb 10 --hnsw-max-indexing-threads 2 --hnsw-on-disk
```

已有词法产物可继续使用 `--index /path/to/bm25`，两种来源二选一。
Qdrant 1.16 REST 要求 `ef_construct >= 4`、`full_scan_threshold >= 10 KiB`；
实际服务对阈值 0 返回 HTTP 422，CLI 在读取数据/加载模型之前拒绝这些无效值。
建图默认仍为 `--hnsw-m 16 --hnsw-ef-construct 100`；其余可选设置会写入 manifest，
writer 会核对服务实际配置。现有普通文件或被普通文件阻断的输出路径会在语料扫描及
模型加载前拒绝。线程数应与实际分配一致；显式参数不替代真实图就绪与快照恢复验收。
在线 `--hnsw-ef 256` 可显式记录搜索 beam（须同时提供向量参数）；它传入
`exact=false`，但是否实际走 HNSW 仍取决于 collection 索引状态及 Qdrant planner。
该参数本身不是 HNSW 已完成或已有性能结果的证据。

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

外部数据版本、下载限制和课程规模校验见 [evaluation/RESOURCES.md](evaluation/RESOURCES.md)。

## 当前状态与后续工作

**已可用**：bigram 占位分词、词法构建/读取、BM25/TF-IDF、普通 OR 查询、布尔/短语过滤、Top-K/高亮/HTTP；双索引构建编排、Qdrant upsert/快照发布检查；规则 Router、RRF、四模式编排、降级和阶段耗时；可选 Sentence-Transformers/Qdrant 适配器；浏览器 UI、HTTP-only 评测器与标准库测试。

**资源与实验进度**：团队自爬语料已冻结 20,000 篇，完成 150 条测试查询的真实 BGE/Qdrant
四模式 HTTP 评测和 paired bootstrap；池化判定的来源明确记录为 assistant 全文复核，
不是独立人工 gold。官方 RocketQA 基线包已核验 8,096,668 个 passage、2,000 个 dev 查询、
9,839 条去重 qrels；固定 BGE revision 的全量向量和 Qdrant 1.16.0 HNSW 快照已通过
两次独立空实例恢复。词法全量合并已交付 8,578,589 个词项，13 个 A1/compact 文件及
正常 `DiskIndex` 读取验收通过；冻结的 808 条内部 holdout 四模式 HTTP 评测及四项
paired bootstrap 已完成并独立复算。Dense 四项质量最高，当前规则 Adaptive 在该集上
低于固定 Hybrid；完整结果见报告第 8 节。该包没有
官方 test labels，内部 holdout 不冒称官方 test。
另在协作版 DuReader 100k 的 1,020 条开发查询上，复验了四种在线模式、BM25S 和同源精确稠密参考。
规则 Adaptive 减少了向量调用，质量低于固定 Hybrid；详见[报告第 9 节](REPORT.md#9-协作开发集与-router-对照)。
数据、模型、快照、索引和生成 run 均留在 Git 外；具体结果与边界见 [报告](REPORT.md)
和 [资源记录](evaluation/RESOURCES.md)。公开演示站点当前仍使用 tiny lexical 索引。

RM3 已实现为确定性的本地伪相关反馈扩展；SLM 仍需要外部固定模型与离线缓存，返回 501。扩展不改变显式布尔/短语约束。内存测试通过不代表真实语义质量或 HNSW 性能已经验收。
