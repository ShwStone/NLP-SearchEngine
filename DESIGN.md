# 中文搜索引擎大作业 · 自适应混合检索设计

> 当前设计取代早期“BM25 + RM3 / SLM 扩展”主线。PPT、代码契约与本文同步；历史讨论保留在 PLAN.md 后半部分。

## 1. 定位与边界

**手写倒排索引与 BM25，离线维护向量数据库，在线用轻量 Router 按需触发语义检索，再用 RRF 融合排名的中文搜索引擎。**

研究问题：与每次都调用向量分支的固定混合检索相比，能否减少在线查询编码/向量搜索，在可接受的质量变化下降低平均耗时？不预设路由一定有效或一定更省。

- 必做：BM25、向量检索、固定混合、自适应混合；查询高亮、HTTP、数据集评测。
- 布尔/短语：显式约束绝不放松；parser 已支持 `AND`、`OR`、`NOT`、括号和双引号连续短语。约束先在词法层过滤，hybrid 在 RRF 前保留过滤结果，dense 对显式约束返回 400。
- 使用预训练 embedding 与现成向量库（拟用单机 Qdrant / HNSW）；不训练神经网络、不生成答案。这是 RAG 的检索层，不是完整 RAG。
- 不做生产级持续爬虫、分布式、增量更新、PageRank、神经重排。为满足课程自建测试集要求，`evaluation.crawl_web` 只提供有边界的一次性 corpus 构造和人工标注模板；它不自动生成 qrels，也不进入在线检索服务。RM3 已作为可选的确定性附加实验接入；SLM 仍需外部模型，不计入必做主线。
- UI 和评测器已实现并通过人工可运行检查；评测器只能通过 HTTP 调用引擎。

## 2. 检索原理与算法分工

### BM25

离线构建词项倒排表、位置、tf/df、文档长度和平均长度；记录分词器哈希与 BM25 参数。**BM25 的“构建”指索引和统计，不是提前计算所有查询的分数。查询相关打分仍在在线侧。**

在线用 IDF、词频饱和与文档长度归一化计算相关性。适合术语、型号和精确关键词，可能漏掉词面不同的同义表达。

### 向量数据库

离线将 `title + "\n" + text` 送入固定版本编码器，应用 document_prefix、按模型 tokenizer 右截断至 max_tokens，输出 L2 归一化向量。一篇文档一个向量，不切块，避免引入块到文档聚合变量。在线查询应用 query_prefix，使用同版模型、tokenizer 和归一化规则。

向量库保存内部 docid、外部 doc_id 与向量，拟用 HNSW：上层图快速导航，下层扩大局部近邻搜索。用余弦相似度取 Top-N；近似搜索不保证精确 Top-N。模型、截断、索引和搜索参数必须在评测中冻结。

向量检索可能补充同义表达，但也可能找回语义接近、实体/事实却不符的文档；效果和速度都必须实测。

## 3. 系统流程与两组工作范围

```text
离线组（两套索引均归离线组）
  语料加载 / 文档 ID 统一 / 分词
    ├─ 倒排索引 + BM25 统计 + 原文/位置/高亮 offsets
    └─ 文档 embedding 批量编码 + 向量数据库构建 + 快照
  产物版本 / 语料指纹 / 校验和 / 自检查
                         │ 接口 A1 / A2：只读产物与共享 Protocol
在线组                   ▼
  查询解析 / 显式约束识别
    → 首轮 BM25（Top-N 候选）
    → Router（查询特征 + 首轮结果）
        ├─ 词法足够：直接返回
        └─ 需语义补充：查询编码 → 向量数据库只读检索
                         → 与首轮 BM25 排名做 RRF → Top-K
    → 原文摘要 / 高亮 / 路由与耗时回显 / HTTP
                         │ 接口 B
                  UI / 评测器（独立 HTTP 客户端）
```

### Router 与融合

- 初版规则特征：唯一查询词数、词表覆盖、首轮候选数、归一化前两名分差。IDF/词项覆盖等是后续可选特征，不宣称已实现。
- 优先保护显式约束；普通查询无词法命中、词表覆盖偏低或头部分差较小时，尝试语义补充。
- 当前阈值只是可执行骨架默认值。开发集设定规则与阈值后冻结，测试时不能读取 qrels。BM25 原始分数不是跨查询置信度。
- RRF：`score(d) = Σ_s 1 / (c + rank_s(d))`；名次从 1 起，未入某路候选贡献 0。先按 docid 合并，再按融合分降序、内部 docid 升序排列。
- 不直接相加 BM25 分数和余弦相似度。每路候选 N、RRF c 和分页候选池为服务配置，不由单次查询偷偷改变。
- dense / hybrid 强制语义分支；依赖未就绪或失败返回 503。adaptive 可明确降级到 BM25，响应必须标记 fallback，评测报告要单列失败/降级率。
- 路由只减少部分在线查询编码、向量搜索与融合；不消除文档编码、向量存储和索引维护成本，也不保证 p95 下降。

## 4. 冻结接口 A：两套离线产物

完整字段和方法见 [contracts/RETRIEVAL.md](contracts/RETRIEVAL.md)。

```text
index/<dataset>/<build-id>/
  bm25/
    manifest.json       # lexical schema v1；分词器、参数、文件 SHA256
    docs.jsonl          # docid、doc_id、标题、正文
    postings.jsonl      # term → [(docid, tf, positions)]，含 df
    offsets.jsonl       # 高亮字符偏移
    stats.json          # doc_lengths、avgdl
  vector_manifest.json  # vector schema v1；成功发布才表示双索引完成
  <snapshot-file>       # Qdrant 快照，文件名与 SHA256 写入 vector manifest
```

- A1：`IndexReader`，由离线 `DiskIndex` 和在线 `MemoryIndex` 实现。保留原有索引格式；新建 manifest 增补 `bm25` 参数，旧 v1 文件默认 k1=1.2、b=0.75。
- A2：`EmbeddingSpec`、`VectorManifest`、`TextEncoder`、`VectorWriter`、`VectorReader`，集中在 `contracts/retrieval.py`。
- `docid` 固定为语料行序 0..N-1，向量库 point ID 与之一致；外部 `doc_id` 必须唯一，作为 payload 和评测 ID 使用。不能在两侧各自重排编号。
- 语料指纹包含有序 docid/doc_id/title/text。启动时核对文档数、语料指纹与完整 embedding 配置；查询时检查命中 ID、数量、去重和分数。
- 编码器版本、维度、归一化、距离、模板、前缀、截断规则变化都要重建/重新版本化，不能只凭维度一致混用索引。
- 离线在新的 collection/目录构建；写入完毕、点数/ID/向量检查、快照校验成功后才发布 manifest。失败不得覆盖已发布产物。
- 生产适配器须校验实际 collection 元数据与发布 manifest，并核验恢复用快照的 SHA256；内存测试替身明确标记 memory-exact，无真实快照。

## 5. 冻结接口 B：HTTP v2.0

`contracts/api.py` 是请求/响应结构和 JSON Schema 的代码真源。

```json
{"query": "手机掉水里怎么办", "mode": "adaptive", "k": 10, "offset": 0}
```

- `mode`：`lexical` / `dense` / `hybrid` / `adaptive`；默认 lexical，旧调用不需要向量依赖。UI 演示自适应时显式传 adaptive。
- 非 lexical 模式只允许 scorer=bm25、expander=none。旧 TF-IDF 和扩展参数只在 lexical 下保留；SLM 无外部模型时返回 501。
- 普通查询现在使用 OR 词项候选，按相关度排序；区别于旧骨架的隐式 AND，属于 v2 的明确语义调整。显式约束交给解析/过滤层，不用扩展或向量绕过。
- 非 lexical 的 `offset + k <= candidate_k`；先产生固定候选池再分页，保证分页稳定。单次请求不能用加大 k 改变融合池。
- 响应包含原有 results/parsed/expanded，新增 api_version、routing、timings_ms。routing 回显请求/执行模式、原因、向量尝试/成功、fallback、分数类型、候选池和策略版本。
- `/index/info` 保留原 manifest，额外回显 retrieval 能力、配置和向量 manifest。`/health` 保留原基本健康信息，不代表向量分支就绪。
- 高亮字符下标标在 doc.text 原文上；语义命中不捏造“同义词高亮”。BM25、cosine、RRF 分数不能跨模式直接比较。

## 6. 交付与并行纪律

### 离线组

| 交付 | 完成判据 |
|---|---|
| 分词与稳定文档 ID | offsets 回切原文正确；外部 ID 唯一，两套索引内部 ID 一致 |
| BM25 构建 | tf/df/长度/位置与全扫 oracle 一致；参数/分词哈希可追溯，停用词保留 |
| 预训练模型文档编码 | 固定 revision、模板、前缀、截断和维度，有限归一化向量，批次不丢数据 |
| 向量数据库构建与快照 | Qdrant collection、HNSW 参数、完整 ID payload、快照 SHA256，失败不发布 |
| 统一构建入口 | `build_hybrid` 组织 BM25 与向量构建，产物指纹一致 |
| DiskIndex / 自检查 | A1 conformance；新增向量 ID/配置/快照检查；交付 tiny 双索引小样本 |
| 成本记录 | 文档编码/建库时间、索引体积、内存、加载耗时 |

### 在线组

| 交付 | 完成判据 |
|---|---|
| MemoryIndex / MemoryVectorIndex | 不依赖真实数据库/模型即可开发检索链路 |
| 查询解析与显式过滤 | 普通查询 OR 候选；支持布尔/短语位置过滤；硬约束不能被语义召回放松 |
| BM25 与向量只读召回 | 复用离线统计，查询编码版本一致，向量命中 ID/分数/维度符合契约 |
| Router | 只消费查询与首轮特征，无 qrels/语言模型依赖；规则可版本化、可单测 |
| RRF / Top-K / 分页 | 融合只看排名、去重、同分稳定、候选池固定 |
| HTTP / 可观测性 / 高亮 | schema 一致，强制模式 503、自适应降级显式标记，阶段耗时可定位 |
| UI 与评测器验收 | 四种模式都通过 HTTP 对照，不 import 检索内部代码 |

纪律：只有 contracts/ 作为跨组公共依赖；在线不 import 离线内部实现（除 DiskIndex），离线不 import online。共享产物由离线构建、在线只读；真实模型 SDK 与 Qdrant SDK 留在适配器中，不侵入 contracts/。

## 7. 评测与里程碑

数据：DuRetrieval，100,001 文档 / 2,000 查询。查询按固定种子或落盘 ID 清单划分开发/测试；固定四组使用同样划分和语料。近重复查询应尽量分组，防止泄漏。1k 子集只做 CI 冒烟，不用于最终结论。

| 方案 | 目的 |
|---|---|
| BM25 | 词法基线；与统一分词/参数/qrels 的 bm25s 参考核对，nDCG@10 相对差 < 2% |
| 纯向量 | 语义基线，固定模型与 HNSW 搜索参数 |
| 固定混合 + RRF | 每个查询都访问两路，是主要效果/成本对照 |
| 自适应混合 + RRF | 只改变触发策略，其余配置一致 |

主指标 nDCG@10，辅助 Recall@10、MAP@100。报告平均延迟、p50/p95、向量触发率、成功/降级率，另报离线构建时间、存储与内存。核心显著性分析为自适应减固定混合的逐查询质量差与 paired bootstrap 区间；“差异不显著”不等于质量等效。改变预先设定阈值集合展示权衡曲线，不在测试集挑最优阈值。

| 事件 | 完成判据 |
|---|---|
| E0 契约冻结 | A1/A2/B、分词/ID/编码配置约定、假向量库与契约测试 |
| E1 双路闭环 | 小样本真实 BM25/向量产物 → HTTP → 评测器；真实适配器不可用前仅算假数据联调 |
| E2 基线冻结 | BM25 对齐，纯向量与固定混合在全量语料有数，模型/候选/融合参数固定 |
| E3 路由与交付 | 独立测试集四组表、成本权衡、区间分析、UI 演示和报告 |

性能目标中的 p95 < 100ms 仅针对 BM25 热查询，向量与混合路径单独测量，不套用同一目标。

## 8. 当前代码状态（不是最终实现声明）

已实现：词法索引构建/读取、普通 OR 查询、BM25/TF-IDF、布尔/短语过滤、高亮与 HTTP；双索引构建编排和 manifest 发布检查；四模式编排、规则 Router、RRF、分页/降级回显；浏览器 UI、HTTP-only 评测器、可选 Sentence-Transformers 编码器与 Qdrant 建库/快照/只读适配器。

默认 CLI 不传向量参数时只启动词法服务；强制 dense/hybrid 在向量依赖未配置时返回 503，adaptive 明确降级。仓库已提供外部数据校验/固定划分、HTTP 评测、paired bootstrap 和有边界的网络测试集构造工具；prefix20k 的真实四模式结果已作为工程冒烟记录。待完成的是依赖课程实际数据、固定 test labels、真实 Qdrant/HNSW 服务的全量四组实验与性能测量。内存或 local exact 向量测试不能作为 HNSW 性能或全量语义效果结论。
