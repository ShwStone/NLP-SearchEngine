# 外部资源与实验边界

本仓库不提交课程数据、模型权重或 Qdrant 产物。开题报告中的 100001 篇文档、2000 条查询和 9839 条 qrels 只能作为团队计划，不能替代公开资源的实际清单。官方大语料先用下文的 `prepare_dureader` 流式校验数量、ID 和 qrels 引用，再划分已核验的查询与判定：

```bash
python -m evaluation.prepare_dataset \
  --queries-only \
  --queries /path/to/prepared/queries.dev.jsonl \
  --qrels /path/to/prepared/qrels.dev.tsv \
  --out /path/to/internal-splits \
  --seed 20261028 --dev-fraction 0.5 --test-fraction 0.4 \
  --expected-queries 2000 \
  --expected-qrels 9839
```

该命令默认输出 JSONL 查询和 LF 换行；不读取、复制或重新认证 corpus。它已逐字节复建本次冻结的六份 train/dev/holdout 查询与 qrels 文件，查询数量为 172/1020/808。普通 `--corpus` 模式仍适合能放入内存的小语料，默认 TSV；含制表或换行的查询自动改用 JSONL，显式要求 TSV 时拒绝。上述数量对应本次核验的公开基线包；使用其他版本须替换为实际清单。开题报告是团队计划，不是教师指定的数据快照。官方说明给出的规模约为 86K 条训练查询、2K 条开发查询、4K 条测试查询和 809 万段落，并要求在百度 AI Studio 竞赛页注册后下载：

- [DuReader-Retrieval 官方说明](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval)
- [RocketQA 官方基线](https://github.com/PaddlePaddle/RocketQA/tree/main/research/DuReader-Retrieval-Baseline)
- [基线下载脚本](https://raw.githubusercontent.com/PaddlePaddle/RocketQA/main/research/DuReader-Retrieval-Baseline/script/download.sh)

本次已从 RocketQA 官方下载脚本指向的公开 CDN 获取 baseline archive，并记录下载包 SHA256：
`b051d3d966f7b0288aed95a923fc3b706176767d8bdcff8aa8fe5fff84b20662`。
该包实际包含 8,096,668 个 passage、2,000 个 dev 查询和 9,839 条去重后的 dev qrels（原始答案段落链接为 9,863 条）；四个 passage 分片各有 2,024,167 行。它没有官方 test 查询和 test qrels，上游划分仍标为 `dev`。后文 train/dev/holdout 指从该公开 dev 按预冻规则得到的内部划分，不能冒称官方 test。

官方 baseline 格式可用适配器转换，适配器会校验四个分片、连续行号映射、查询 ID、qrels 引用并流式写出 corpus。
查询输出为 `queries.dev.jsonl`，通过 JSON 转义保留原文中的换行、制表符等字符，避免产生额外查询。
提供 `--archive-sha256` 时必须同时提供 `--archive`；缺少文件或哈希不匹配时不会创建输出目录：

```bash
python -m evaluation.prepare_dureader \
  --root /path/to/dureader-retrieval-baseline-dataset \
  --out /path/to/prepared/duretrieval-official-dev \
  --archive /path/to/dureader-retrieval-baseline-dataset.tar.gz \
  --archive-sha256 b051d3d966f7b0288aed95a923fc3b706176767d8bdcff8aa8fe5fff84b20662 \
  --expected-queries 2000 --expected-qrels 9839
```

为先验证完整评测闭环，已从官方 dev corpus 固定取前 20,000 篇文档，并只保留其中有 qrel 命中的 19 条查询／19 条 qrel。该前缀索引不代表全量结果：BM25 的 nDCG@10、Recall@10、MRR@10、MAP@100 分别为 0.5032、0.5789、0.4781、0.4920，平均延迟 57.85 ms、p50 52.59 ms、p95 79.15 ms。无向量后端时 adaptive 的 10/19 次查询明确记录为 fallback；这组结果仅是官方数据前缀上的工程冒烟。

## 2026-10-29 集群验收记录（受限前缀烟雾实验）

以下结果来自一次固定的集群任务，所有生成数据、权重、索引和 TREC run 均留在任务私有临时目录，没有提交到仓库。公开 baseline archive 为 4,214,306,577 bytes，SHA256 为
`b051d3d966f7b0288aed95a923fc3b706176767d8bdcff8aa8fe5fff84b20662`；准备后的 dev 资源为 8,096,668 篇文档、2,000 条查询和 9,839 条 qrels。其 passage map、corpus、queries.dev 和 qrels.dev 的 SHA256 分别为 `9b091ee62278b02973c6fa6de5798c3b0bee9a493613ab29a4b807efdd38741a`、`bc67f3ccfe7423c039f04c7a5e7f6eae07a7f722087edf0caa7cfc694a8c45e8`、`3be3067e73022ea5b3b919196baae29dc2d50c6aa7053ad911e2b3d5cf8d389a` 和 `150f4665e779c0c876a46102d9a00fa467759847bc01ff344ee6c7f92c43fd79`。

为控制内存和验证完整 HTTP 闭环，从该 dev corpus 流式取前 20,000 篇文档，固定 seed `20261028`，按 dev/test/train=0.5/0.4/0.1 得到 7/9/3 条查询。前缀 corpus SHA256 为 `7e2a20ca5f7a75ece81efffbb45f10983c90ebaf500f60a4a53f9c775a55bfc8`；test qrels、queries 的 SHA256 分别为 `b07146daa42cd3b53c4c55779625ecce20dca3d337d86a64eb03a1c3d288f1c2` 和 `f05c990beb04a4997b7910149d4a128f74823aae6f1af7b09c84c80ec5f20ef5`。这仍是官方 dev 的前缀烟雾数据，不能当作官方全量结果，也不是团队自爬测试集。

语义编码使用 `BAAI/bge-small-zh-v1.5`，固定 revision `7999e1d3359715c523056ef9478215996d62a620`，512 维、CLS pooling、最大 512 tokens、L2 归一化；`model.safetensors` SHA256 为 `354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`，`tokenizer.json` SHA256 为 `48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26`。运行时为 PyTorch 2.3.0、Transformers 4.57.1、NVIDIA A800 80GB；CUDA 编码结果有限、归一化范数为 1，重复编码最大绝对差为 0。标准 Hugging Face 端点在该任务中不可达，权重通过固定 revision 的公开镜像取得；镜像只用于下载，不改变模型 ID 或 revision。

路由参数在 dev 上先固定为 `candidate_k=200`、`rrf_c=60`、`min_vocab_coverage=0.7`、`min_relative_gap=0.1`、`policy_version=rules-v1`，随后只在 test 上报告结果。test 的四模式结果如下；dense/hybrid/adaptive 使用真实 BGE 查询编码和 20k 向量，但向量读取器是 `memory-exact` NumPy 全扫替身，因此不能解释为 Qdrant HNSW 性能：

| mode | nDCG@10 | Recall@10 | MRR@10 | MAP@100 | mean ms | p50 ms | p95 ms | vector success / fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lexical | 0.625659 | 0.666667 | 0.611111 | 0.626455 | 50.90 | 49.29 | 61.09 | 0 / 0 |
| dense | 0.958992 | 1.000000 | 0.944444 | 0.944444 | 198.48 | 202.67 | 218.39 | 9 / 0 |
| hybrid | 0.791378 | 1.000000 | 0.730688 | 0.730688 | 200.81 | 200.91 | 223.86 | 9 / 0 |
| adaptive | 0.713333 | 0.888889 | 0.659259 | 0.666667 | 133.46 | 169.74 | 229.49 | 5 / 0 |

配对 bootstrap 使用 test 的 9 条查询、10,000 次重采样和 seed `20261028`，差值定义为后者减前者。lexical→hybrid 的 nDCG/Recall/MRR/MAP 差值及 95% percentile CI 为 `+0.165719 [0.037037, 0.294400]`、`+0.333333 [0.000000, 0.666667]`、`+0.119577 [0.015873, 0.238095]`、`+0.104233 [0.011111, 0.222222]`；lexical→adaptive 对应为 `+0.087674 [0.000000, 0.207683]`、`+0.222222 [0.000000, 0.555556]`、`+0.048148 [0.000000, 0.122222]`、`+0.040212 [0.000000, 0.098413]`。样本量很小，CI 只证明这次前缀烟雾的配对差异，不能外推到全量。

真实 Qdrant server 小批验收已完成：服务端版本 `1.16.0`，客户端版本 `1.16.0`。仓库自身 `QdrantSnapshotWriter` 创建 `rfc4_writer_20261029b` collection，写入 3 个 4 维 cosine 向量和完整 `nlp_search_engine` metadata；HNSW 参数为 `m=16`、`ef_construct=100`。快照文件 112,640 bytes，SHA256 为 `899eb64518664a726612a3e6e3e9a13037c4160ae9342ba21f08468fb5c53ac0`。在空存储目录和独立服务端口用该快照启动后，exact count=3、内部 ID 覆盖 `[0,1,2]`、外部 payload ID 为 `doc-a/doc-b/doc-c`、metadata 与 manifest 完全一致；仓库 `QdrantVectorIndex` 成功只读查询并返回 doc-a/doc-c/doc-b 的稳定结果。未归一化向量被 `validate_vector` 拒绝。该小批证据不外推 8M 规模；stream/compact 入口已纳入，但旧 v1 builder 仍是兼容性全量内存路径，当时仍需单独完成 8M 资源与 HNSW 验收；后续全量工程终态见本文末段。

为了验证工程闭环，公开的 [C-MTEB 快照镜像](https://github.com/Lsogod/everything-ent-hybrid-rag/tree/main/data/knowledge/datasets) 可在当前网络中取得，但它只有 5000 篇文档、100 条查询和 425 条 qrels。它适合冒烟评测，不能当作所选官方版本的全量数据；本地冒烟结果也不能写成最终成绩。

RocketQA 脚本中的数据和 ERNIE 权重来自 `dataset-bj.cdn.bcebos.com`。数据包已完成哈希和结构核验。集群完成了 BGE CUDA 编码和受限前缀四模式实验，随后取得固定模型的 SHA、CPU 准入和全量 HNSW 快照回执。全量词法合并和正常 reader 已取得原生 PASS；双索引完整四模式质量与配对区间已通过独立复算，见本文“官方全量四模式最终验收”。构建 BM25、向量索引和四组实验时，模型 revision、维度、前缀、截断规则和 Qdrant HNSW 参数须随结果记录。 [评测说明](README.md)

Qdrant 适配器和快照发布逻辑已在 `offline/vector_build.py`、`online/vector_index.py` 中实现。没有可用 Qdrant 实例时，内存向量替身只能用于契约测试，不能作为 HNSW 性能结论。

## 团队自爬新闻测试集

2026-11-03，新华网 `clean-news/v1` 抓取任务达到 `TARGET_REACHED`，清洗及精确/近重复去重后共有 **20,000** 篇有效文章。该语料独立于官方 DuReader 前缀试验，采用同域、robots fail-closed、深度 6、每次请求间隔 1 秒和单响应 1,000,000 字节上限。语料与 sidecar 的最终 SHA256 已重新核验，原始响应缓存、数据和索引均不进入 Git。

| 最终产物 | 字节数 | SHA256 |
| --- | ---: | --- |
| corpus.jsonl | 84,555,357 | `3f16eaeae59ab9c0cfc80c219b0f51b8a0aeb784cca4a2be4070c0161447b851` |
| article_sidecar.jsonl | 36,800,693 | `f19c6972e6d97419a6206c47fb695757f815e8b6ef62953d4ef8d498737afc7e` |

最终回执 SHA256 为 `e73861130be4da1ee1a1eef772e4d52bbb5837126a265b6650aadebade8e99a6`。此前冻结的 200 条来源支持查询（dev/test=50/150）均与最终语料的来源 ID、标题和正文身份核验一致；查询文件 SHA256 为 `3b906bfc4841dd34f919f3b87a984a7902860348f0ce6ce40213b99b50632578`。由 lexical、dense、固定 hybrid、adaptive 和 lexical+RM3 在深度 20 汇集候选，补入来源候选后共 8,281 对。按身份复用 200 条 assistant 种子判定，其余 8,081 对逐篇全文复核。标签来源为 `assistant_semantic_review_not_human_gold`。全池 0/1/2 级分别为 7,603/360/318 条，开发集 2,067 条，测试集 6,214 条。正式质量请求使用深度 100，Recall/MAP 按池中判定计算。

同一冻结 20k 语料已完成真实 BGE/512/FP32 文档向量缓存。独立作业申请 1 张 2080 Ti、4 CPU、4 GiB，共耗时 197.097 秒，包含模型/源码准入、编码、全量缓存校验与打包；本机再次完整核验 20,000 条映射、向量和输入哈希。收据 SHA256 为 `456b3843225b0ec7b5bc819314e1846def1b51e99607cc36289d78198f674ca0`，缓存标识为 `a8402c6d95714b150ad4570baa0a61a85b1429fd02f3e1e5bd19c69647ba4c5b`。A1、compact 及 Qdrant HNSW 建图已完成，查询仍使用固定 CPU BGE。五个检索臂各完成 200 条查询；本地会话中断后逐字节复用已完整落盘的四臂，仅重跑未完整记录的 RM3，保留旧的部分结果。缓存、索引及运行数据均不进入 Git。

独立恢复验证从空存储、新 Qdrant 进程装载 97,468,928 字节快照。恢复后 exact count 和 indexed vectors 均为 20,000，两次图状态观测稳定；16 条预先固定查询的 Top-20 IDs、顺序和分数与池化 dense 基准完全一致，最大绝对差 0（预设容差 `1e-5`）。非空 segment 的 HNSW counter +16、plain/exact +0；空 segment 的 plain +16 单列。这里的计数单位是 segment searches，此验收证明真实 HNSW 与快照恢复一致性，未运行相关度质量或 bootstrap。

| 外部验收产物 | SHA256 |
| --- | --- |
| 五臂运行 manifest | `903973eef424189e508ba24934ed8da5bd195862fa72bb792d2bfe26e3293aa4` |
| 8,281 对候选池 manifest | `c02c8c18a1f65ed0c306175c558ee683b4fecf46f9a1877cc7b13dd59b5bc3ca` |
| 正式 qrels manifest | `fdcdc39b91388b7e48b58fbbbbfecab7206f756542ead1d49a6904912a1c5214` |
| test.qrels.tsv | `4448466dc6cbe41569b755ce5f817f56e8b88e927ab13b7b7b6d06a465694850` |
| 20k Qdrant 快照 | `143724e357d88a40ed5ec5f0d0a22f2f3c1e3769e8aa6d0cad35859100104036` |
| 新进程恢复验证回执 | `61fd1279b3954cd1235439f35542a53df52be98a2db5bcbfefdba1deef9ab3f7` |
| 正式测试 run-state（绑定四组请求、TREC、指标和四项区间） | `6f03d13cfd69e5984bcaf7e916d207179978962e136ecd8155af3ed65ba0eed9` |
| 正式测试启动证明（绑定线上接口与运行配置） | `8157d0f5848f9c66068cdc59c5cbe94cf34c53362e05907e055e96757ca2cf26` |
| 正式测试服务及 HNSW 恢复回执 | `9ba484730d033ed686331316b86fda623af1895da05f7b0a72b0ca7ca277dd11` |

正式测试使用冻结的 `2362689098384529728f0bfdc9c20e7dbcf9227f` 源码和 HNSW recipe。每组保留 150 条成功请求及独立 TREC run。run-state 固定每组的请求、run、指标与区间 SHA256，逐查询输出和原文留在 Git 外。复算可用 [评测入口](README.md) 的 `evaluation.evaluate` 与 `evaluation.compare`，按深度 100 请求，使用 nDCG@10、Recall@10、MRR@10、MAP@100 和 seed `20261028` 的 10,000 次配对重采样。Adaptive 与固定 Hybrid 的详细指标和延迟见 [课程报告](../REPORT.md#6-自爬新闻测试集的四模式评测)。

以下为历史小批数据与工程冒烟，保留各自的来源和规模边界。


2026-10-29 已使用仓库的 `evaluation.crawl_web` 对公开的中国政府网执行一次有界抓取：seed 为 `https://www.gov.cn/`，同域 HTTP(S) BFS，`max_documents=30`、`max_depth=1`、`max_bytes=500000`、请求间隔 0.2 秒，并要求 robots.txt 允许。抓取结果为 30 篇文档，corpus SHA256 为 `8e6cde8dfa0647deb9d55466064d951b445eb19f13740d349ef8398b86d5ab1b`；该语料和抓取收据保存在版本库外。

从这 30 篇文档构造了 10 个查询和 97 条 BM25 候选池记录，生成了带标题、URL、片段和启发式建议的 `annotation_review.md`。启发式建议文件的 SHA256 为 `61f38c72504dfbe4ff764946d4c149a6585faa9226646b9edff7b0b629303ba8`，用途为早期标注建议。后续 20k 自爬评测使用本节开头记录的独立新华网语料。

随后使用相同 seed、robots 和同域约束，将 `max_depth` 提高到 2、`max_documents=20000`，实际前沿耗尽后得到 393 篇有效文档；该 corpus 的字节数为 5,048,778，SHA256 为 `757e93bfc08f800827cff925864155032bbfe5d4be6cdcc5009b7b8996419531`。对这 393 篇文档重新生成了 10 查询、100 条 top-10 候选池和独立的 provisional suggestion manifest，标记为 `ai_suggested_needs_human_review`。随后修复分块传输中断边界并运行 depth-3 重试，得到 799 篇有效文档，corpus 字节数为 9,083,175，SHA256 为 `1f209cad5da40808b14f6249b886308e977f708dd222f2c9eaf984f3d5b9f258`，该批次只生成了未标注模板。该站点的结果保留作小批试采证据，所有 corpus、模板和建议文件均保存在版本库外。

随后运行 depth-4 重试，得到 1,029 篇有效文档；corpus 字节数为 11,681,876，SHA256 为 `463ac4eff4fdc22047da362fdc22d47f0be6bf09c1587c3c8ba318b9e8164ff3`。该批次生成了 10 个查询、100 条 top-10 候选池记录和 `annotation20k-depth4_review.md`；建议文件 SHA256 为 `5400b65c24c1a4e114df67c1cf4e9428621cde92052ec5ebe7bef5ae8beda643`，状态仍为 `ai_suggested_needs_human_review`。它仍不是人工 qrels，也没有达到 20k 有效文档。

对这 1,029 篇外部语料完成了不入库的规模路径冒烟：`offline.stream_build` 生成 1,029 文档、平均文档长度 2,945.7 的 v1 索引，耗时 66.69 秒；`offline.compact` 生成 178,504 词项的 compact-v1 产物，耗时 15.22 秒。stream manifest SHA256 为 `529b7c4576c50e59a20a031ad1e46b8d382e5455cc55a5be5f11782185574e10`，compact manifest SHA256 为 `06c71823b174f2b85ff5200eed96bfa8e0d90494ac5c7cee617bbe223fa00c07`。这些是 1,029 文档工程证据，不能外推为 8M 峰值 RSS、磁盘、吞吐或 Qdrant HNSW 性能。

2026-10-29 的集群复核未形成新的 8M/HNSW 任务收据：交互式任务表单在最长运行时间校验处拒绝提交；同一账号的训练入口可以看到自有 RFC-0004 工作空间和仓库目录，但入口文件级联选择读取私有数据时返回 504。未创建新任务，也未把该失败误写成计算资源不足或实验结果。

为扩大团队语料，读取 gov.cn robots.txt 声明的四份公开 sitemap，去重后得到 117 个 URL，实际抓取 108 篇可用页面；该批次 corpus 字节数为 602,669，SHA256 为 `49024d91182bfaf781b2b399e5a30c180edb0aa858fbe7a35e226124dea8702d`。将 depth=4 语料与 sitemap 批次按 URL 合并后得到 1,137 篇文档，corpus 字节数为 12,284,545，SHA256 为 `3bea58d3c66856001669a33f0919ddc95fabb4548535e1f3d6898c939a2f7ef9`；没有发现重复 URL。该合并语料的 `offline.stream_build` 用时 74.19 秒、avgdl 2,797.2，stream manifest SHA256 为 `e2336a9dfdd96d9070a278ccd6be7cac7bc36f5c85ff9658d4089a3a03a0b084`；`offline.compact` 用时 13.97 秒、186,263 词项，compact.json SHA256 为 `67abc2a7eb5e821666985abda1f6facb5b007e10f9b25085740e28cb73976aa2`。

作为跨站点抓取能力的补充试验，使用同一 `evaluation.crawl_web` 约束从公开新华网入口抓取 `max_documents=100`、`max_depth=2`、单响应上限 1,000,000 bytes、请求超时 5 秒和 0.3 秒间隔；得到 100 个唯一文档和 URL，corpus 大小 480,476 bytes，SHA256 为 `eafb1dee8ddc36a3d5c778883d8e379618d073014d742359623cb321db8394f5`。该批次仅生成 `needs_human_annotation` 模板，没有查询、qrels 或质量评测，因此只作为未标注候选资源收据，不能与 1,137 篇 assistant-reviewed 语料合并后宣称新的正式测试集。

在用户明确授权后，对 100 条 top-10 候选逐条进行 assistant 语义复核，标签来源明确记为 `assistant`，不是独立人工标注。复核结果为 relevance 0/1/2 = 26/56/18，生成 10 个 queries 和 100 行 qrels；corpus SHA256 为 `3bea58d3c66856001669a33f0919ddc95fabb4548535e1f3d6898c939a2f7ef9`，reviewed annotation SHA256 为 `42815d911a94477ccd112f4a397a98300fd9d51c9de6934eacd1d09b59e650bd`，queries SHA256 为 `c43b3d629af4850cced161a4123b1939a227317c94b92776f1af7d99cbb8081b`，qrels SHA256 为 `a6d47baf3cfc9f228863e1b5adcc3ba909792cc81480b7c57db3603224200bf5`。这些 qrels 可以用于工程评测，但报告必须保留 assistant-review provenance，不能改写成独立人工真值。

在这份 assistant-reviewed qrels 上，1,137 文档 BM25 lexical run 的 nDCG@10/Recall@10/MRR@10/MAP@100 为 `0.950197/1.000000/1.000000/0.918422`，平均/p50/p95 延迟为 `1014.92/869.50/1578.00 ms`。RM3 run 为 `0.945802/0.980000/1.000000/0.916014`；10,000 次 paired bootstrap（seed `20261029`，RM3−lexical）nDCG 差值为 `-0.004394`，95% CI `[-0.020271, 0.008033]`。adaptive 在没有向量后端时 5/10 查询触发向量尝试，5 次 fallback，fallback rate `0.5`；dense 请求按契约返回 HTTP 503 `vector_unavailable`。这些是 1,137 文档的 assistant-reviewed 工程结果，不是 20k 或 8M 全规模成绩。

所有 corpus、annotation、queries、qrels 和 runs 仍保存在版本库外；本次正式收据记录了 assistant 审阅者、日期、标签规则和 SHA256。若后续需要独立人工 gold labels，应另建版本并保留本次 assistant qrels 作为单独实验条件。

## 2026-11-02 全量数据恢复与规模构建

本地续传完成 4,214,306,577-byte 官方 archive 并重新核验 SHA256；经四分片、连续 ID、唯一性及 qrels 引用校验后，原子发布 8,096,668 篇 corpus、2,000 个 official dev queries 和 9,839 条 qrels。corpus 大小为 7,425,979,008 bytes，SHA256 为 `bc67f3ccfe7423c039f04c7a5e7f6eae07a7f722087edf0caa7cfc694a8c45e8`。数据仍位于 Git 外，完整准备不等于完整索引或语义实验完成。

本轮重新计算发现历史文字中 passage map、queries.dev 和 prefix20k/split 的五个摘要存在多抄字符，长度甚至超过 64 位。上文已用本轮真实 manifest 和相同 seed/行序/LF 规则重新构造的 prefix20k、test queries/qrels 摘要修正。这里只重建并核验了输入身份，没有重新运行或重新确认历史四模式实验的指标。

官方 shard 01 的 10,000-document 资源 pilot 保留了正文 `()` 的零 token 文档；索引契约和 stream/compact 均已支持该类记录，保持原文与内部行序。pilot 使用明确的合成 ID，只用于资源估计，不用于评测。stream 构建 152.93 秒、进程峰值工作集 33,714,176 bytes；事件流 compact 转换 21.24 秒、峰值 34,058,240 bytes。另一次相同输出的磁盘补测为 106.67 秒，100 ms 采样观测到 SQLite 最大 42,618,880 bytes、总 staging 最大 130,210,588 bytes；这些值不能作为 8M 实测资源成本。

全量 CPU 词法构建使用冻结源码副本和分盘临时库；初次非检查点进程异常终止且旧 SQLite 损坏，原目录完整保留。当时后续 `checkpoint_build` 使用 WAL/FULL 和输入/代码绑定重新构建（最终全量终态见本文末段）；仅实际发布并经数量、哈希、reader 验收后才登记完整交付。可恢复 compact 已在相同真实 10k pilot 上确认六个数据文件字节一致、全部 384,350 词项逻辑行一致和 28 篇 reader 抽样；此次包装脚本验证阶段 API 使用错误后补验通过，但没有保存转换成本，故不填新耗时/RAM。

完整 official-dev 的本地确定性划分固定为 train/dev/local-holdout=172/1020/808 queries（qrels=825/4960/4054），seed `20261028`；冻结文件含全部 query IDs 和 SHA256。它仍来自 official dev，且包括已观察过前缀查询，不是官方 test 或完全盲测。模型、candidate pool、RRF 和规则策略已记录；正式质量运行前另行冻结了 HNSW 建图与搜索配置：m=16、ef_construct=100、full_scan_threshold=10 KiB、max_indexing_threads=2、on_disk=true、hnsw_ef=256、exact=false。必须核验服务实际配置、索引状态和快照恢复，参数本身不证明图就绪。 原冻结的阈值 0 后经官方 1.16.0 二进制实测返回 HTTP 422（REST 最小值 10），故在任何正式质量请求前另存 v2 冻结，不覆盖旧配置；新配置 SHA256 为 `5fc1204abbd52faa5ded5f5dcfb7fd2a643c63163861213822db9644f616c899`。修正依据是接口约束，不是检索结果。

历史 IAB 导航、会话或入口观察不能扩大为“集群不可用”或“配额不足”。本轮非 GUI 内存 API 已核对既有任务、剩余配额、当前工作区节点名单和平台健康信息，并实际完成独立 CPU 任务。集群端重新验证全部 corpus/queries/qrels 的 SHA、字节数及行数，与本地冻结资源一致。该次数据准入和后续全量 Qdrant HNSW 工程验收分别保存回执。

独立文档源 `CorpusDocuments` 已实际扫描并解析全 8,096,668 行：108.55 秒、峰值工作集 218,562,560 bytes，原始 SHA/大小/数量匹配，首/中/末固定内部 ID 可读。该验收保留两 uint64 数组，总有效载荷 129,546,688 bytes；外部 ID 唯一性和最终双索引 fingerprint 仍有各自门禁。收据 SHA256 为 `12784e7d2156c756f0962eb867c8564278b7153994b26f7e6a10bec3bb558491`，不将文档访问准入称为完整索引交付。

固定 revision 的六个官方模型文件已逐 SHA 取得并传到新的隔离目录，回下载一致。旧缓存缺文件、集群直连下载超时和首次依赖安装时限超出的证据均保留。原生 Linux 从先前下载的 wheels 构造 34 包依赖闭包，每个 wheel 经官方 PyPI JSON 核对；46,498,720-byte wheelhouse SHA256 为 `d39dc737917e17ebab4f23a0085170709a6e715eca5750915a9bff4d57dac277`。不修改共享环境，Torch 保留基础镜像版本。

同一模型的真实 CPU 数值准入通过：Python3.10.14、Torch2.3.0、Transformers4.46.3，FP32/CLS/right/512/L2；覆盖短中文、标点、长输入、batch/padding、两个共享编码实例和重复编码。最大绝对差 `1.775333657860756e-7`，范数 `[0.9999999454,1.0000000735]`，峰值进程 RSS 743,702,528 bytes；worker 211.51 秒包含离线安装 147.69 秒。收据 SHA256 为 `25c9369d437f07265ef2e90a51383dfb1e7f35c55843827d34aa9d984e54ae93`。这是实际数值路径验证，仍不代表语义质量或 GPU/HNSW 实验。

独立 local-cache 2080Ti 任务已完成 `GPU_PILOT_PASS`，实测配额为 1 GPU / 4 CPU / 4 GiB；CPU/GPU 文档及查询最大绝对差 `1.6391277313232422e-7 < 1e-5`，范数范围 `[0.9999999137,1.0000000886]`。2,048 篇全文的 batch16/32/64 吞吐分别为 420.896/404.423/414.533 docs/s，计时覆盖分词、forward、归一化及 CPU list transfer，每档先 warmup 一批；worker 79.941 秒，supervisor 81.31 秒且 exit0。进程峰值 RSS 为 1,015,775,232 bytes；flat cgroup 路径缺少 exact-membership 证明，不把该 cgroup 数值当任务真实峰值。收据 SHA256 为 `ea976cf2f347c6b3b3d01b19af23101750aceced2f8c13df5d254cb32d3507dd`，supervisor 为 `9066c9eb975a4c54d5b362514bc4e56549fb344042ad06858af0176e36a84c9c`。这是数值/有界吞吐准入，不替代 8M 编码或质量评测；旧 GPU 调度任务及失败证据保留。

完整 2,000 queries 的语法预检发现 3 条显式约束（train/dev/local-holdout=0/1/2）。它们按当前 dense 契约预期返回特定 HTTP400；评测器提供显式、query SHA 绑定的 policy，仍发送全部请求，将准确匹配的预期拒绝记为空检索并单列覆盖率和延迟。默认仍严格失败，其他错误不能被吞掉；完整分母不会被共同支持子集分析替换。对应 policy 在质量运行前冻结于 Git 外。当时尚无正式请求；后续 808-query 四模式尝试因传输错误失败，未形成完整 arm seal 或可接受质量结果。

新的自爬版本使用 `evaluation.crawl_articles`，目标为清洗去重后的 20k accepted unique articles，不再以 raw URL 计数。原 raw 采集已停止并完整保留为 superseded 证据（2,015 raw docs）。新版本导入原 frontier 的 11,703 个 URL（seed 文件 SHA256 `69c807888876cba583abe372826e900ac0b90ba561c42bf07128895ebdd85f4a`，全部 depth=0），绑定固定 `news-cn-body-v1` 规则、NFKC/空白精确去重、SimHash64 距离 <=3 且长度比 >=0.9 的近重复政策。原 HTTP bytes 先持久化到 SHA 寻址 gzip，再清洗并记录完整 sidecar；不因检索分数选择文章。只读抽样前五篇的原字节/decoded SHA、正文、标题及全部元数据重算一致；150 篇 snapshot 中未发现满足当前规则的残留近重复。该轮抽样时采集仍在进行；最终 20k 语料于 2026-11-03 冻结，完整池化 qrels 于 2026-11-04 冻结，见上文。正文/缓存无再分发授权，均留在 Git 外。

该新版本已有 200 条来源支持的 assistant 查询草案：对 220 个来源独立阅读正文，初稿接受 206、拒绝 14；独立 assistant 回读及正文核对额外拒绝 1 条，再按预冻 source rank 取前 200，未使用检索结果或 qrels 选择。草案 JSONL SHA256 为 `3b906bfc4841dd34f919f3b87a984a7902860348f0ce6ce40213b99b50632578`，类型为 explanation/comparison/policy/event/fact=63/9/35/43/50，dev/test=50/150 的 query IDs 已冻结。全部选中来源已于 2026-11-03 核验最终 20k 的 source_doc_id/title/text 身份。盲化多系统池化和 assistant 全文复核均已完成，150 个 test 查询已用于正式四模式评测，另 50 个保留为 dev。设计 manifest 与来源证据均留在 Git 外。

该阶段历史代码批次核心 62/62、evaluation 76/76、checkpoint/stream/BGE/文档源/CLI/HNSW 专项 109/109，compile 和 diffcheck 通过，未改动的 Web 代码此前 Node 5/5 通过。规模路径的当前边界和可复现实验入口见 [`docs/RFC0004_SCALE.md`](../docs/RFC0004_SCALE.md)。文档源、CPU/GPU 数值及小批 HNSW 收据均不替代完整双索引和全规模四模式/paired bootstrap 评测。

### REST 配置与存储检查的边界

隔离 CPU2 / RAM2GiB / GPU0 检查确认官方 Qdrant `1.16.0`（commit `0d79528705ff04def1b52842f747e5a00ab1e338a`）对 `full_scan_threshold=0` 返回 422，并明确要求至少 10。原始请求/响应收据 SHA256 为 `3c9bf01ad297ad74fe45d3156d85c1f44f625a1ea7fa12f50e055b108bfe5307`。检查脚本原预期 400 因而停在这一负例，不能宣称正向 10、graph readiness 或查询已通过；此前 Python3.10 hash API 不兼容和 JSON 配置未被发现的失败均保留，未把失败收据升级为 PASS。

服务配置文件使用 `.yaml`，并禁止静默退回默认端口/存储；正向建库须核验服务实际配置。历史检查中的 NFS 路径不作为 Qdrant 数据目录，不能泛称全部私有存储都是 NFS：后续静态 ROOT 实测类型为 LeoFS。独占 `/tmp` 的可见类型为 overlay，其 physical backing 仍为 UNKNOWN，因此不构成物理 SSD 或 8M 存储容量/耐久性准入。小型协议检查与正式规模存储验收分开，不能把输入配置或空库建成当成 HNSW 图搜索。

历史独立正向流水线已实际通过阈值 10 的 HTTP200 建库、完整 HNSW 配置/metadata、精确 count=0 和 YAML 资源设置继承；该组件回执 SHA256 为 `b7f13044f2844104b2dca05699b578043de7579a5c3f43d4ab1a588cc82d79d5`。语料/模型/版本准入也通过，随后在编码开始前的模型初始化阶段耗尽有界预算：wall 1185.58 秒、进程 CPU 11.84 秒、峰值 RSS 491,229,184 bytes，监督器在 1193.16 秒内清理本次新进程组。最终失败回执 SHA256 为 `42db086184e3ae93ce8cc26084c3c82b00f12ee42900b8dcd78e41809516a334`，继续作为失败证据保留；该次空库准入不升级为完整 HNSW PASS。

新的独立 local-cache 流水线已完成 `REAL_BGE_HNSW_PILOT_PASS`：2 CPU / 4 GiB / GPU0，Qdrant1.16，真实固定 revision BGE、512 维 FP32。A 建库与 B 新进程/空存储恢复均 exact count=2,048、全部非空 segment indexed，两次稳定观测。A/B 各 16 个非精确查询均以 hnsw_ef=256 执行，非空 segment 的 HNSW counter +16、plain/exact +0；空 segment 的 plain +16 单列，计数单位为 segment searches。恢复前后 ANN 与 exact 各自 IDs、顺序及 scores 一致（绝对容差 `1e-5`），不以 ANN/exact 的相互一致声称召回质量。快照 10,060,800 bytes，SHA256 `69f84ccf6607cd57a280010af9a2ec900f52b676b6e911e60e0773fed9394d39`，已持久化到静态 ROOT，交付引用不依赖 `/tmp`。worker 656.764 秒、supervisor 667.318 秒，`COMPLETE`/exit0/无 timeout；模型构造 314.718 秒不用于推断旧超时根因，2,048 文档编码 90.98 秒只作此次 CPU 工程吞吐。主收据 SHA256 `800b5618a0f017e8cf918e7b559cd7660f195fb09d5e52c056f8e03c97103dbd`，supervisor `ab8ead3deb2afb5853e95773ea6378e563738710cb7726d7a5bb33a695e888ca`。未使用 qrels、未运行质量评测，不替代 8M 资源验收或四模式/paired bootstrap。

独立 10k scratch feasibility 首次因 cgroup 统计路径不可解析，在 builder 前停止；43.44 秒不作为构建成本，失败回执 SHA256 `37399802366c4ea4986fee7e35cfcff31b0aae564337c5a45d035050e7708b27` 保留。repair1 原生构建已 `PASS`：2 CPU / 2 GiB / GPU0，核心构建 31.215 秒、320.359 docs/s，cgroup exact process membership（非进程独占）峰值 858,669,056 bytes，无 OOM；发布/绑定和两次重开全量 reader 均通过。overlay physical backing 仍 UNKNOWN，不据此认定物理 SSD 或全量准入。后续 100k pilot 首次在构建前被严格运行时绑定拒绝，原因为包装脚本错误地假设 Python3.10 使用 Unicode14；已依据 [Python3.10 官方 unicodedata 文档](https://docs.python.org/3.10/library/unicodedata.html) 改为严格 Unicode13，修正版 preflight SHA256 `653f423053e5dd614c88c273c6a72881e7c5444e43ba3ea5416873a3aa357cd8`。独立 repair1 的最终收据证实 A1 构建与独立校验已完成，但 CHECKPOINT_COMPACT 在原 4 GiB cgroup 的 90% 门禁停止：current 3,881,177,088 bytes、进程 RSS 45,588,480 bytes、compact WAL 1,980,566,432 bytes，OOM/failcnt 均无增长。原始响应 SHA256 为 `1c0c2e4ea772763b7719c2309919aa09e1123deb463964c012ed95dce33b08838`；URL 脱敏后的本地收据 SHA256 为 `557e505973b53c19cd7d0e8e776caa9cf84102b6e8bff8bc575602d3dafaba7a6`。未完成静态交付，不能声称已有持久化 100k A1。

源码和真实 SQLite 对照证实 compact 同连接的长生命周期 SELECT 阻止 WAL checkpoint；有界 keyset 分页并在写入/提交前关闭游标后，tiny WAL 峰值由 659,232 降至 32,992 bytes。旧 compiler 在新增回归中 4 项失败（database table is locked），修复后通过；相同 tiny 输入的 8 个公开产物逐字节相同，分页边界 fsync/commit 硬退出恢复保持字节一致。修复保持 WAL/FULL、原提交节奏、格式、字段校验和原生资源门禁。修复后以相同 2 CPU / 4 GiB 通过 100,000 文档原生复验。A1 构建 602.202 秒，compact 358.180 秒，cgroup 峰值 2,408,161,280 bytes，OOM 与 failcnt 均为 0。13 个静态文件读回 SHA256 与源一致，packed reader 两次重开。原始收据 SHA256 为 `6fc41aa4d19e3506ac0dc00f00faa4384ef3486cdd1664696a152692906c765be`。

100,000 文档的语义 pilot 使用 1 张 2080 Ti、4 CPU、8 GiB 和固定 BGE revision。CUDA 编码 245.574 秒（407.209 docs/s），Qdrant 1.16 完成建图、三份静态交付物的 SHA256 读回和新实例恢复。16 条探针恢复前后的 ANN 与 exact top-200 各自一致，非空 segment 记录到实际 HNSW 搜索。cgroup 峰值 2,799,603,712 bytes，无 OOM。收据 SHA256 为 `edc2a955974b0457163654d9e88729a9c20841e0416eebe9f158f978eb91f59d`，快照 SHA256 为 `e9c6d8764bc39f8f0cf086c2a67325eba4ff0aafd3be834afcea79565611eacd`。该轮未使用 qrels。

2026-11-03 另申请 2 CPU / 4 GiB 验证 100k A1 产物直接写共享卷、SQLite 留本机的布局，五个文件与已有原生私有盘基准逐字节一致。A1 构建 631.627 秒，cgroup 峰值 1,387,294,720 bytes，OOM/failcnt 均为 0；私有/共享目录采样峰值分别为 779,751,424 / 811,275,616 bytes。旧私有盘构建为 602.202 秒，两者不是随机交错的性能比较。收据 SHA256 为 `1bf2c4fd513e33b297eab85078070acf190d452f4840b51fc241c9c8543bb76e`。该次观测到的 cgroup CPU quota 相当于 4 核。试验只覆盖 A1，不能外推为全量或并发容量验收；物理后端仍未确认。

同日，十分片计划中的首个 506,041 文档小分片完成 A1 与 compact 独立验收，13 个静态文件读回 SHA256 与源一致；两种格式的 docs 摘要均与预冻范围一致。请求资源为 2 CPU / 10 GiB / 0 GPU，实际 cgroup CPU quota 为 4 核；进程峰值 RSS 为 48,128,000 bytes，cgroup 峰值为 5,475,852,288 bytes，私有盘采样峰值为 3,178,532,864 bytes，OOM/failcnt 均为 0。A1、compact、compact 静态传输分别耗时 2902.918、1204.154、322.014 秒。产物收据 SHA256 为 `991bfb9218689e30b614a9f73a85f9addf261ad4f1c432894d6fcde5790c48a7`，资源回执为 `a6aa6b1c04876a9d17d35be2a7009a2f9333ecd578acacdb0d4e6e5087aac04c`。这是一份分片的工程证据，不能将进程 RSS 当作完整内存成本，也不能将单任务峰值当作并发或 8M 容量证明。

官方全量 BGE 缓存的两份原生生产任务随后各报告 `chunks_ready`，`completed_count=81`、`reused_count=0`，存储清单中的 162 个完成标记齐全。该缓存覆盖 8,096,668 篇文档，固定 `BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620`、512 维 FP32 和原有文本/截断契约；缓存配置身份为 `dbcc76ab042f9d04a68fa7cbad4fbe4f0f798e2b363416ac6c214e5b6a77b53f`，不是所有向量文件拼接后的摘要。生产任务各请求 1×2080Ti、4 CPU、12 GiB，观测到实际 CPU quota 为 8 核，但进程 affinity 限制为最多 4 核。当时独立全内容复验尚未完成；后续最终验收见下文。当时全规模 HNSW/四模式评测尚未完成，不能只凭生产日志与标记计数报告最终验收通过；后续终态见下文。

随后两个各 506,042 文档的小分片也完成 A1/compact 的 13 文件验收，三份连续小分片共覆盖 1,518,125 篇文档。新增两份产物收据 SHA256 分别为 `f29618536aa05d2495d55735301b4e56eb75c43239ecda594dba5cd4158fafa3`、`39b91bd3d77a4bc8990d8b3f2b0cf3e08f26ad578e80d6cc0f9412d11f336157`；私有盘采样峰值分别为 3,165,499,392 / 3,183,312,896 bytes，cgroup 峰值分别为 5,844,250,624 / 5,918,748,672 bytes，均无 OOM/failcnt 变化。两任务实际执行存在重叠，但这些收据没有测量其他训练任务的吞吐，不能声称并发没有性能影响。当时其余分片和全量合并尚未完成；后续终态见下文。

独立全量向量复验采用 8 MiB/s 的只读流和原内容校验器。2 GiB 原生尝试在完成 38 块后达到 90% 内存保护线，峰值 1,936,510,976 bytes，1724.422 秒后退出；没有 OOM、failcnt 变化或已报告的内容错误。回执 SHA256 为 `e1526b37a2a259343553526fcace01fe131de129347a95fae364f642e27f088a`。此时尚未完成全语料 SHA256 和最终文件身份复查，38 块不能当作全量验收；需要按实测内存重新定容，再执行完整协议。

20 万文档合并试验完成新增 10 万文档的 A1/compact 后，父进程的文件采样出现 `FileNotFoundError`。固定 CPython 3.10.14 的 `pathlib` 源码在 Linux Python 3.12 上复现了目录重命名竞争，修正版仅容忍遍历中消失的子目录，根目录丢失、权限及资源门禁仍拒绝，20 项本地回归通过。原生失败回执未记录具体路径，不能唯一确定原始系统调用。修正版原生尝试又在开工前因私有盘 115,506,016,256 bytes 低于并发预算要求的 165.5 GiB 被拒绝，未执行合并；回执 SHA256 为 `bfb5e8c810b73edb0a89919c56696998f0fb3e1d62d507e3719014c6f6252984`。后续须重新核对可用存储与并发预留，不把本地回归当成原生合并通过。

2026-11-04 取得独立全量缓存复验最终收据：`FULL_CACHE_CONTENT_VERIFIED`，162 块连续覆盖 8,096,668 篇，16,581,976,064 bytes 的 FP32 向量及 905,715,706 bytes 映射。验证器完整检查逐块 SHA/大小、内部/外部 ID、`title + "\n" + text` 输入哈希、512 维有限值/L2 归一化、全语料 SHA/EOF，以及文件身份稳定性。2 CPU / 10 GiB / 0 GPU，固定 8 MiB/s 读取上限，耗时 7,164.775 秒；cgroup 内存峰值 7,571,202,048 bytes（含文件缓存），worker RSS 23,855,104 bytes，私有文件分配采样峰值 2,228,224 bytes，无 OOM/failcnt 增长。收据 SHA256 `3852ac87d489f2fb98f5060ed6dc2fc2776681ab3a727c6cf9c5c028218a0479`；本地审计核对完整区间、固定编码配置、源码 ZIP 与验证逻辑。该内容验收不重跑模型推理，不代表 HNSW 或检索质量完成。

全量 Qdrant 1.16 在快照前通过两次稳定图检查，exact count 和已索引数量均为 8,096,668。16 条固定查询分别执行 ANN 与 exact，每组在 32 个非空分段产生 512 次对应类型的搜索。历史任务在快照阶段达到 47,884,369,920 bytes，超过 48 GiB cgroup 的 90% 保护线后退出，终态回执 SHA256 为 `c3d250a4669eb053fa714665bd1339e89c481e08293ffe99bcabfc18999106d2`。同机文件恢复随后在真实 20k 快照上通过 16 条固定查询及 SHA256 校验，回执为 `d3904273e903b67ec0e04b2d4b75fbcdadef25808c787b533ee078bd73fecde3`，为下述全量恢复提供了工程前置证据。

同日取回原两个大词法分片的最终生产收据，SHA256 分别为 `257b09e1abe357b61651ee9f22b808fac76d16ce3fe02835142112806843f9c9`、`13c0c62b39b2139f7f0466e99cee38f5243fe94fc37086fd76303817d139eae5`。各覆盖 2,024,167 篇，每份 13 个 A1/compact 文件经生产端复制哈希与目标读回验证后发布。随后分片 05 再覆盖 506,042 篇，生产回执 SHA256 为 `261747163d59ac2e134037b88b693edc9e8ddec96749b6bfb10ddf52d3d852ae`。分片 06 在固定 CPython 3.10.14、Unicode 13 和原生 ijson 运行时构建 506,041 篇，57 个传输分块经远端读回，集群原生导入器再校验 13 个发布文件。导入回执 SHA256 为 `90c030973de241c561544ad4ca406f80b152a7b519c236fcb8a7baa3a40d557d`。此时七片连续覆盖 6,578,542 篇；后续三片和全量合并验收见下文终态记录。

全量 HNSW 首次独立启动在资源采样阶段退出，worker 未启动、未生成 collection，失败收据 SHA256 为 `7b03aa0f24f93cc80ac60996fe6f8a73d9f4d3a54a8686e3e7275d5f58894256`。审查发现外部运行包装脚本继承了 pilot 的至多 8 GiB 采样门槛，与请求 48 GiB、监督器要求精确 48 GiB 的配置矛盾。真实文件型 cgroup v1/v2 fixtures 复现旧脚本拒绝 48 GiB，新副本改为精确请求内存并继续拒绝 8/64 GiB、unlimited、缺失或畸形 OOM 数据，19 项专项通过。原收据只保留未分类 ValueError，故该复现只能说明一个已发现的门槛问题。

全量 HNSW 最终保留快照为 18,477,460,992 bytes，SHA256 `2df5c889386db365e4d72d089f90bfb736c8a6a24d5bede4a10f727d9f0976e0`。初次恢复的 16 条 ANN Top-200 一致，但 7 条 exact 的原始同分返回顺序或第 200 名边界不同，原失败回执 `a5f5db6af6ce527a4bc02b883454255295934340e74059287b986604d341013c` 保留。

后续 4 CPU / 72 GiB / 0 GPU 的独立验收复用同一快照，顺序恢复到两个空实例。两个实例各在两次稳定观测中报告 green、optimizer OK、exact count=8,096,668。各 16 条 ANN 的 Top-200 ID 和顺序与原实例一致，分数按冻结的绝对容差 `1e-5` 检查，非空分段各增加 512 次 HNSW 搜索。exact 诊断各增加 512 次对应的 exact 搜索。另对全部 162 块做 FP64 归一化余弦扫描，交叉核对截断同分组完整性及原生分数，最大绝对差约 `1.93e-7`。两次恢复均通过精确排序与投影验证。采样 cgroup 峰值为 38,968,786,944 bytes。原生回执 SHA256 为 `a79d9aa0cc8de004a1d0c15621b086f9a8d900dcc23bb097573ccfcc653d16b2`，独立核验摘要 SHA256 为 `d7b4389350c680d24cddc8bd3a72fa79c82f7f12d021e5d4c8ff67083ea1f243`。该流程只使用机械探针。该机械恢复验收当时不含相关度；双索引就绪后的四模式质量、逐臂延迟与配对区间终态见本文末段。

2026-11-05，分片 07、08、09 的原生发布回执均通过，十片连续覆盖全部 8,096,668 篇。三份新增回执 SHA256 依次为 `1c6313753dd620afb97b3c55d594b13555ccfb2bd2e2df3785fc7e5212b56a28`、`9e583cc67d7c8e34b17449020e1ee4d669a1413bca54a42a7c6a569f322e3a2d`、`99a80f688ff1e42861a9696fcb4906a670bce3d954dc8064fa7ee43390681c54`。每片 13 个 A1/compact 文件均按固定摘要校验。实际 LeoFS 上的 512 文档合并探针还核对了 13 个参照文件，并通过 READY 边界硬退出恢复，回执 SHA256 为 `e68e802e7a1d0c99468e8ae761ba5afb504745af7aacb24532bf76ae4c4f930c`。首次全量合并因旧目录映射在输入阶段退出，失败回执 `57cfb9fdeaa6b5d53e5ed851b6b4c19d62e4107aa36081e0815677f25e2db273` 保留；修正后的十个目录都独立回读了收据。该探针只提供合并工程前置证据；后续全量合并与公开 reader 终态见下文，正式质量结果须独立运行。

BM25 工程核对使用 [PyPI 的 rank-bm25 0.2.2](https://pypi.org/project/rank-bm25/0.2.2/)；wheel SHA256 为 `7bd4a95571adadfc271746fa146a4bcfd89c0cf731e49c3d1ad863290adbe8ae`，其 `rank_bm25.py` SHA256 为 `2f28cc795415c01e9f3db5a8ed019774f9cba747b272c9c304271589b8081ac6`。同一 analyzer、`k1=1.2/b=0.75`、词项并集候选和稳定同分顺序下，对齐参考库 IDF 后，tiny 十篇与额外八篇边界文档上的 16 条有效查询分数和排名一致，最大绝对误差 `2.6645352591003757e-15`；另有 1 条语法拒绝。参考库原始 IDF 与默认参数的结果另行保留：只保持同一 `k1/b` 而保留原始 IDF 时，有 2 条查询排名不同。收据 SHA256 为 `bb7e15ef3451cdf7847089c88ba99c1d5a9610280df569c784dc8bab5802d090`。它验证计分实现，不是检索质量实验；没有改动 `fixtures/tiny_index`。

该检查为可选依赖，生成的语料、索引和收据要求放在版本库外的新目录：

```bash
python -m pip install rank-bm25==0.2.2
python scripts/check_bm25_reference.py --out /tmp/fresh-bm25-reference-check
```

全量向量和后续正式 HTTP 评测使用的 CPU 准入环境固定为 Python `3.10.14`、Unicode `13.0.0`、PyTorch `2.3.0`、NumPy `1.26.4`、Transformers `4.46.3`、tokenizers `0.20.3`、huggingface-hub `0.26.5`、safetensors `0.5.3`、qdrant-client `1.16.0` 和 ijson `3.4.0.post0`；隔离 site 的完整 distribution 清单和实际解析版本都按准入回执核对。上文受限前缀实验的 Transformers `4.57.1` 是历史环境，不与本轮全量环境混记。

完整文档的截断统计使用同一模型 tokenizer SHA256 `48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26`，在原生 Python3.10.14 / Unicode13 / tokenizers0.20.3 上以 162 个连续分块扫描原 corpus，并核对总计 7,425,979,008 bytes 的完整 SHA256。输入为 `title + "\n" + text`、空文档前缀且包含 CLS/SEP；超过 512 tokens 的文档为 620,314 / 8,096,668（7.6613%）。汇总收据 SHA256 `2fd8f31a8cdc564c9f45b307054b00c5e540925be56e9825f05601b129d1f2e6`。早期使用另一 tokenizers 版本的未完成统计保留但排除；本统计没有模型推理、qrels 或质量指标。

全量合并的第二次任务触发 `cgroup_memory_guard`，最后成功采样为 38,641,569,792 bytes、限额 40 GiB；当时没有记录触发瞬间的 memory.stat，不能据此断言 OOM。失败收据 SHA256 `cda9714ae72aca57b17ee6d4007bbb99bc96877dadec5d047a0e80873559c9e4` 和 1,770,000 文档封存断点保留。随后原生 LeoFS 的 256 MiB 自有文件探针核对释放缓存前后内容 SHA256 一致，实际释放 268,308,480 bytes 文件缓存，收据 SHA256 `660a2af7e1392077268ad68e7189b55856c4d7081510ee7d7b71743e906c67b3`。新恢复方案限定在本任务分片、输出和私有 SQLite 文件，保留内存保护与两轮完整源文件校验；512 文档的真实 WAL 备份、改变监督器配置但保持 core/path 身份的恢复、13 文件参照、伪造与损坏拒绝均通过。新任务申请 2 CPU / 10 GiB / 0 GPU，最终全量结果按下述独立原生终态验收。

2026-11-06 独立取回并绑定原生全量合并的三份未改写收据：监督器 SHA256 `bc9b425ccc15f1f87576e03c37d13b7ee6bae24b6445a3aebf6895070d5e2b59`、合并 `1d4704da31373db5bbb6b441a5dc23f84fe1ea3792d95057c5528ec6bb95f0f3`、公开 reader `aa008c41a3bf323e596147893d19b174db2c1b582207ed7609a8ff89756c01ad`。终态 PASS、worker exit 0，十片顺序和 13 个产物清单与配置 `7e19c0c97d442df2f188125f9730c1f5fbf4333d7b7edf05b32ec038def6b99e` 绑定。最终有 8,096,668 documents、8,578,589 terms，有序 corpus fingerprint 为 `0ae8b1174d0ee8e55f269c3c6d8e5f12d7e9c79fcb72c3966e99de18c9061019`，两个格式的 docs.jsonl 哈希相同。公开 `DiskIndex` 对内部 ID 0、4,048,334、8,096,667 的读取与预先绑定的外部 ID 一致。

本次恢复运行 wall time 为 40,563.541 秒，校验输入总字节数 107,064,297,777；cgroup 内核峰值为 2,201,608,192 bytes，worker RSS 峰值为 392,413,184 bytes，无 OOM 或 failcnt 增长。cgroup 包含文件缓存，不解释为进程独占 RAM；该 wall time 不包含此前分片、旧失败和已完成的原始断点。根审查验证收据、配置、源码、十片顺序、完整文件清单、计数、fingerprint 和 public reader 的绑定，没有另行下载约 107 GB 输入重新哈希。此前失败收据和恢复证据继续保留。该原生 PASS 完成全量 lexical 交付工程验收；后续 808-query HTTP 四模式质量的独立终态见本文末段。

早期失败边界另行保留：无检查点的 journal-OFF SQLite 全量 staging 在进程终止后出现 `database disk image is malformed`，原数据库不被接纳为可恢复断点；后续路径使用源身份绑定的 WAL/FULL 与明确提交边界。另一次对 1,029 篇 packed 产物运行通用 conformance 超过 20 分钟后停止，未生成成功收据。tiny conformance、全量文件/语义校验和公开 reader 探针各有独立范围，后来的合并 PASS 不将旧通用 conformance 追认为通过，也不证明断电或磁盘故障恢复。

2026-11-06 在另一个 2 CPU / 2 GiB / 0 GPU 的只读元数据任务中，逐项测量最终发布目录的 13 个 regular 文件，前后两轮 inode/dev/size/mtime 均稳定；该任务不读取大文件体，不改动运行中的评测或产物。文件 SHA 逐项对应已固定的 producer receipt，并非此次重新计算。原生体积收据 SHA256 为 `3c9f31b1bdaf03b433ce8f65c861edc2888fd786efdbce5d1c040b4d63afd4a8`，测量源码 SHA256 为 `613a2d584b8fb154e0bf90a821200de760368f89453580a0a689bdc04c8bf017`。

| 最终产物 | 文件数 | apparent bytes | GiB（2^30 bytes） |
| --- | ---: | ---: | ---: |
| A1 lexical | 5 | 65,243,179,664 | 60.762446 |
| compact | 8 | 41,554,639,647 | 38.700774 |
| 两套格式合计 | 13 | 106,797,819,311 | 99.463220 |

这是 13 个文件 `stat.st_size` 的合计，两份 `docs.jsonl` 按各自路径计入；不是去除共享块后的物理独占空间，也不包含输入分片、向量缓存、快照和模型。上文的 107,064,297,777 bytes 是合并输入总量，不能当作此输出体积；目录采样峰值也不能替代最终文件大小。

源码 `382a8cc` 增加保持计分与稳定排序语义的词频、外部 ID、正文读取和有序指纹快路径；旧失败请求和部分输出保留，30 秒 HTTP 超时未放宽。核心 69/69、evaluation 83/83、固定 Python3.10.14 / Unicode13 专项 153/153 通过。同一集群模型依赖环境以 2 CPU / 2 GiB / 0 GPU 完成核心 69、评测 83、计分一致性 7 项复验，wall time 17.602 秒，无 OOM/failcnt 增长；原生收据 SHA256 为 `ac15719f68fc26c64c8819e02acebf184bc6ac160fe1c076725bcbedb067ff69`。新的固定 dev32→formal808 流程先验证 32 次 HTTP200、零回退、每次不超过 20 秒及预计正式请求时间不超过 18,000 秒，再在同一服务 epoch 运行完整评测。该准入只测运行性能，不读取 dev qrels 或调优质量参数；该性能准入阶段尚无可接受的正式 8M 质量结果；后续终态见本文末段。

2026-11-06 上述正式流程通过 dev32 后，词法 808 查询整臂已封存；随后因 `supervisor_deadline_reached` 结束，监督器原始收据 SHA256 为 `11c59d1e159399e7fc8e7464723b39992b90d4ca5aa0cc5acaea57248100d828`。wall time 21,609.622 秒，worker group 已关闭。Dense 日志保留 689 次尝试、688 次 HTTP200、0 次预期拒绝、0 次已记录异常，以及一条未完成请求；它没有完整 seal，不能算作正式 Dense 结果，也不复用该部分。该轮峰值 cgroup 为 25,809,977,344 bytes；收据的 `private_bytes=19,625,379,883` 实际对应 `/tmp` 自有 scratch，`static_bytes=19,283,634` 对应共享 stage 目录。scratch 文件系统为 overlay，物理 backing 未验证，不据此称作 SSD 测量。

同一源码归档、bundle、数据、模型和检索设置的恢复任务现已终态 PASS，按原有完整臂校验合同核验并复用词法 seal，补齐其余三臂。总时限调整为 57,600 秒，CPU4 / RAM48GiB、30 秒单请求超时和内存/磁盘门禁不变；监督器源码仅改动新 stage、旧终态 SHA、总时限及其上限。原生终态 schema、完整臂来源、原 epoch 与字节相等检查的控制层 fixture 在固定 Python3.10.14 上 26/26 通过；这些是恢复工具验证，不是完整四模式质量结果。四臂完整 808 查询、原始 TREC/请求日志、指标和 10,000 次 paired bootstrap 已于 2026-11-07 通过独立复算；跨服务 epoch 的延迟按臂保留。

## 官方全量四模式最终验收

2026-11-07，完整 8,096,668 篇双索引、808 条固定内部留出查询和 4,054 条 qrels 的四模式 HTTP 评测通过 `PASS_ROOT_RECOMPUTED_ALL_ARMS_AND_BOOTSTRAP`。正式源码为 `382a8cc356c814d55e0f3fbe73b1a6a5e31296f7`。取回 34 份当前文件及旧词法臂来源证据；所有可信 seal/摘要逐字节匹配，下载端曾出现的空 HTTP200 响应被拒绝而未作为结果保存。Python3.10.14 / Unicode13 独立复算全部四臂质量、请求覆盖、路由、有效请求延迟，以及 seed `20261028`、10,000 次重采样的四项 paired bootstrap。完整结果和解释见 [报告第 8 节](../REPORT.md#8-官方全量-808-查询四模式评测2026-11-07-完成复核)。

| 模式 | nDCG@10 | Recall@10 | MRR@10 | MAP@100 | 向量尝试 / 成功 / 回退 |
| --- | ---: | ---: | ---: | ---: | --- |
| lexical | 0.126475 | 0.152901 | 0.195561 | 0.089223 | 0 / 0 / 0 |
| dense | 0.255994 | 0.285620 | 0.385472 | 0.194676 | 806 / 806 / 0 |
| hybrid | 0.209705 | 0.244810 | 0.321067 | 0.158736 | 808 / 808 / 0 |
| adaptive | 0.187638 | 0.217471 | 0.290920 | 0.140833 | 595 / 595 / 0 |

| 指标 | Adaptive − Hybrid | 95% percentile 区间 |
| --- | ---: | --- |
| ndcg | -0.022068 | [-0.028850, -0.015632] |
| recall | -0.027339 | [-0.035531, -0.019674] |
| mrr | -0.030147 | [-0.041505, -0.019787] |
| map | -0.017903 | [-0.023487, -0.012448] |

四臂各使用全部 808 条查询；Dense 为 806 次 HTTP200 与 2 次冻结的显式约束400拒绝（空排名计入质量），其余三臂各 808 次 HTTP200，均无异常。Adaptive 向量尝试/成功/回退为 595/595/0，比 Hybrid 少触发 213 次但四项质量均下降。参数是既有开发设置在正式质量前冻结，不声称在完整 dev 上优化；不因负结果调整 held-out 阈值。官方标注来源为 dev，含历史已观察查询；自爬标签为 assistant 全文复核，不是独立人工 gold。

原生任务本轮 supervisor wall=18,725.252 秒，4 CPU / 48 GiB / 0 GPU，worker exit0、group closed。采样 cgroup peak=26,491,912,192 bytes，包含子进程和文件缓存；scratch `private_bytes=19,597,561,394`、共享 stage `static_bytes=41,562,731` 均是目录表观字节，不是物理独占空间。监督器耗时不包括旧任务的词法整臂、失败与此前构建，也不是从零总成本。原 21,609.622 秒时限失败及未封存 Dense 输出继续保留，不改写为成功。固定 dev32 预计时间低估了正式执行耗时，仅属于准入预测。

Lexical seal 对应旧 epoch `56f6a8a0407643d8855e479de6c7a56f`，其余三臂对应新 epoch `5f429cef31434f20b77ab7c855269e57`；`latency_comparability=per_arm_only_mixed_epochs`。逐臂延迟见报告，Dense 仅计 806 个有效响应，其余各 808。固定顺序、缓存、跨 epoch 和单次运行不支持因果加速比或多次重复的稳定性结论。

| 文件 / 证据 | SHA256 |
| --- | --- |
| 原生终态监督器 | `25ab6ed8b92695ebe178396fb175b67627fd94b2fa3b3b65fa9e75fdb650547c` |
| 独立复算结果 | `106f5d3928473d7676fbb8978bd3948c6c0c539b9c204a26a96aacf5078326cd` |
| 冻结源码 source.zip | `95bae9c94c766b4eaac532b06e67313fc1a6c0cd8c2d6788b8c65f420ee50ee8` |
| 运行 bundle | `d25e44ba9f003346bc32d697d4a5fd5d3f89878c65ef1ece1c83a363a5c473bb` |
| 运行协议 | `90f5c629b72571cf3aa067d35450913b3b9bc95dbfa67ec5b0b05d90e485ffaa` |
| lexical.checkpoint.json | `a78c3b94b7991a4870f80bd585c090c3eff633215fb492dbe5aaf71c275f14f8` |
| lexical.trec | `29f47dfd80cec8eb694736572f263cd5f558b6959bc3ee8065325f58498ec2f4` |
| dense.checkpoint.json | `6e03ef97b93268de399367efa89fd63fc60b900ce6ef5a0ba59093f57ec5ad65` |
| dense.trec | `b81a885b80b936193619b6ccc3a3974f0c1fa814cc2e724e7e9e7dbe928d6d91` |
| hybrid.checkpoint.json | `52c581fc802f83c13051658440d80d70644a96567f2ea976d9976aec7c811238` |
| hybrid.trec | `7470e96c8f09f5a5e14ae20168283466b640fe9cc76c398de5b5482ac9bcf9ad` |
| adaptive.checkpoint.json | `b9c083f51382b62cc51aeb6c8c37289228f82cc886803c0373522210a0d6b830` |
| adaptive.trec | `26d5954065ba72c454d6ad6174e6482b1d5b3db6f39d2d6b709ae35f60cd36c4` |
| paired-adaptive-minus-hybrid.ndcg.json | `0b043a6188cd8bbe612af783d35c24c99dd41fcedc34f1808663e23f130577ab` |
| paired-adaptive-minus-hybrid.recall.json | `80c9e22e9fb05ef2e7726fdd6147e7c032a672d0c53cca925015f1cf9ff18e86` |
| paired-adaptive-minus-hybrid.mrr.json | `22c106782b0d7fbfb2808d5c699652438f410f536c920aa957d8d458d9c23373` |
| paired-adaptive-minus-hybrid.map.json | `6cfbacb9942e80a9bc902e7740129bdab03be96dc7a364dc050851ac8fc4e12f` |

以下位置均相对于授权的私有产物根目录，不包含真实集群连接信息；向组员交付时使用该目录的受控只读副本，勿用临时 scratch 冒充持久产物：

- A1/compact：`rfc4stage20261105fullmergedirect2/published/{lexical,compact}`；13 文件与体积/producer hashes 的对应见上文。
- Qdrant 持久快照：`rfc4stage20261104hnswfullc/durable-a`；18,477,460,992 bytes，SHA256 `2df5c889386db365e4d72d089f90bfb736c8a6a24d5bede4a10f727d9f0976e0`。版本化 manifest 与外部 ID 映射必须一并交付，禁止替换在线旧 collection。
- 本轮评测：`rfc4stage20261106officialquality4/quality-service/quality`，含四份 TREC、metrics、requests、线性 events journal、checkpoint 与四项 paired JSON；旧 lexical seal 来源为 `rfc4stage20261106officialquality3`。
- 本机已取回的完整证据位于项目外 `holdout-execution-ready-20261105/runtime-v2-20261106/collected-native-quality4`；其 `root-verified-results.json`、`collection-manifest.json` 和 `prior/` 支持离线审查。冻结查询/qrels、模型合同与 evaluator 源归档在同级复核输入目录，与上述摘要绑定。

使用与正式环境相同的 Python3.10.14 / Unicode13，在项目源码 `382a8cc` 上对受控结果副本复算（`QRELS` 指完整808-query冻结 test qrels，`RESULTS` 指四臂结果目录）：

```bash
python -m evaluation.evaluate --qrels "$QRELS" --run "$RESULTS/lexical.trec" --include-per-query
python -m evaluation.evaluate --qrels "$QRELS" --run "$RESULTS/dense.trec" --include-per-query
python -m evaluation.evaluate --qrels "$QRELS" --run "$RESULTS/hybrid.trec" --include-per-query
python -m evaluation.evaluate --qrels "$QRELS" --run "$RESULTS/adaptive.trec" --include-per-query
for metric in ndcg recall mrr map; do
  python -m evaluation.compare --qrels "$QRELS" \
    --run-a "$RESULTS/hybrid.trec" --run-b "$RESULTS/adaptive.trec" \
    --metric "$metric" --k 10 --map-k 100 --samples 10000 --seed 20261028
done
```

该 TREC 离线入口重算质量与区间；请求覆盖/400 policy、路由、epoch、journal 和延迟必须同时核对已封存的请求证据，不能从 TREC 行数推断。最新 Windows Python3.13 复验核心69/69、evaluation83/83、Node7/7、compileall 与 diffcheck 通过；同环境原生69/83/7及固定 Linux153 的源码身份未改变。数据、模型、索引、快照、runs、原始收据及任何凭据不提交 Git。
## DuReader 100k 协作开发集词法对照

使用维护者发布的 [DuReader 100k v1.0 重建材料](https://huggingface.co/datasets/ShwStone/dureader-mini-100k/tree/2db5eadeb4adabec463324b4ca3562b7b79fda94)，从已规范化的官方公开语料重建 100,000 篇、172 条 train 和 1,020 条 dev。五份数据文件和压缩包的 SHA-256 均与[发布记录](https://huggingface.co/datasets/ShwStone/dureader-mini-100k/blob/v1.0/verification.json)一致。此处的查询来自公开 official-dev 的内部开发划分。

在源码 `d1f2d12` 上通过本地公开 HTTP `POST /search` 执行手写 BM25，使用同一份冻结语料的 A1 索引。外部参考使用 `bm25s==0.2.14` 的 ATIRE 词频饱和、Lucene IDF 和 float64，双方使用同一 bigram 分词、`k1=1.2`、`b=0.75`。一条显式约束查询共用项目的硬过滤，BM25S 独立计分。1,020 次请求全部返回 HTTP 200；两份 run 都有 101,872 条记录。三条查询的短榜分别为 41、43 和 88 篇，经扫描全部 100,000 篇原文独立复核。其余 1,017 条查询各返回 100 篇。

| 方法 | nDCG@10 | Recall@10 | MRR@10 | MAP@100 |
| --- | ---: | ---: | ---: | ---: |
| 手写 BM25（HTTP） | 0.133959 | 0.157120 | 0.217316 | 0.131827 |
| BM25S 公式对齐 | 0.133959 | 0.157120 | 0.217316 | 0.131827 |

两份 run 的全部排名逐条一致。按 HTTP 的六位小数精度比较，101,872 个分数全部相同。使用 seed `2026` 做 10,000 次查询级配对 bootstrap，四项指标差均为 0，95% 区间均为 `[0, 0]`。这组结果用于验证词法实现的一致性。HTTP 单次顺序开发评测的平均/p50/p95 分别为 76.10/69.17/140.04 ms，期间本机有索引压缩任务；BM25S 的外部 run 无可比较的 HTTP 延迟。

HTTP 和 BM25S 两份 TREC 文件 SHA-256 分别为 `2bea488509bc591c52858c30adbe2677d5fc5fe14bb65dc3a9d6957ec76645bf`、`e46e616b962c5fccb03707feeefe4aaeb01d9060f618dcfe3ff5a39b4484945f`。逐查询证据与校验收据保存在 Git 外。复建、TREC 校验和参考入口见 [评测说明](README.md#external-trec-runs-and-the-dureader-100k-development-subset)。

## DuReader 100k 四模式与精确稠密参考

2026-11-12 在固定运行源码 `71284695a822e08088e3c393eb387ea3b8d54667` 上完成
100,000 篇、1,020 条 dev 查询、4,960 条 qrels 的四模式质量测试。
冻结协议 SHA-256 为 `26c99b6ab9cfe8ed97adddd54864450b48b5bc6e2f7cf68321e53d94aea77e64`。
四臂使用同一压缩词法索引、候选池 200、RRF 60、规则 `rules-v1`、
BM25 和关闭查询扩展的配置，评测深度 100。语义路径使用
[固定 BGE revision](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620)，
512 维 FP32、CLS、右截断 512、L2 归一化和固定查询前缀。
98 个封存的文档向量块同时供精确参考和 Qdrant 1.16.0 使用；
精确参考以 float64 计算余弦相似度。HNSW 使用 `m=16`、
`ef_construct=100`、`full_scan_threshold=10` KiB、`max_indexing_threads=2`、
`on_disk=true`、`hnsw_ef=256`，正式请求前核验精确点数 100,000，
索引向量数 100,000，collection 为 `green`。快照恢复后重复该核验。

语料的有序指纹为 `0792a807c614de14e3b57f3b8f051add2b1c4d7556eee743f4e3cd69f28d5457`，
Qdrant 快照 SHA-256 为 `4d5bf2e061a30a4be4b51711e951637851a510fec73acd9937e4c811bc3dbcfd`。
四模式的词法与向量身份及请求配置均通过逐臂前后检查。
一条显式约束查询按预先登记的规则留在 Dense 和精确参考的分母内，
以空排名计分；Hybrid 和 Adaptive 继续执行硬过滤。
质量分数、配对区间、约束边界和成本见[报告第 9 节](../REPORT.md#9-协作开发集与-router-对照)。
精确参考与 ANN 使用同一编码器，故它们的差异衡量近似检索与实现一致性。

| 方法 | TREC 行数 | TREC SHA-256 |
| --- | ---: | --- |
| HTTP BM25（compact） | 101,872 | `2bea488509bc591c52858c30adbe2677d5fc5fe14bb65dc3a9d6957ec76645bf` |
| HTTP Dense | 101,900 | `62caf89b6e77e3af0d3fc8f0de870e02e45d96df657be9051cde2508299abdc2` |
| HTTP Hybrid | 102,000 | `dae02ed96fb606d5fbeade44bffad29938565b107cb03dbfbe96c492c4c24d37` |
| HTTP Adaptive | 101,943 | `3fc535ca9eec398aacf9bf808faf5c2474cee53a8306555c29d8e752a04c080e` |
| 精确 Dense 参考 | 101,900 | `240637c36bcd50dc059a95c587c776edb1e072900740383f6120fdd5aa92e859` |

外部 FAISS 参考生成 101,900 行 TREC，SHA256 为 `344a6ba79a630d3e79daf906b186b005958cbe14c7c25d953fcf5ccdd9dcbe71`。
它在同一冻结语料、1,020 条开发查询和 4,960 条 qrels 上评分，显式约束查询保留为空排名。
采用 FAISS CPU 1.10.0，核对文档及查询原始向量摘要后，用 L2 归一化向量执行 float32 平面内积扫描。
用 `evaluation.external_run` 验证文档 ID、查询覆盖、深度 100 和四项指标；
与 NumPy 精确参考和 Qdrant ANN 都按查询配对重采样 10,000 次，seed 2026，95% 百分位区间。
FAISS 相对 Qdrant 的平均 Top-10/100 集合重合率分别为 99.98%/97.83%，MAP@100 差值为 +0.0000080，
95% 区间为 [−0.0000724, +0.0001152]，其余三项差值和区间均为 0。
汇总回执 SHA256 为 `4a0f89ccfe6bc3a7254b834f040b34976094a0edcba765c60851a963f484119f`。
复现参数见[精确向量参考](README.md#exact-dense-reference)。TREC、逐查询指标及回执保存在 Git 外。

两个服务 epoch 使用同一快照和参数。质量臂保留各自的请求与服务日志，
跨臂延迟仅使用随后单独运行的交错计时。复现步骤见
[同语料 HTTP 评测](README.md#matched-http-runs-on-the-shared-development-corpus)。
计时环境为 Intel Core Ultra 7 155H、Windows Python 3.13.5 CPU 推理、
WSL 内 Qdrant 1.16.0 和回环 HTTP。计时回执 SHA-256 为
`6dfcb325e462edee38b0335c6b6a26711c180e299953a6f9642d8e980bc3acb5`。
质量配对回执 SHA-256 为 `28ec2c9e3407003b6979bc1ce1f37ff10f874372855cfc0c237686b6138187bc`，
逐臂 TREC 重评分、ANN 重合率与单一计时服务日志的复核回执 SHA-256 为
`25b137212b918173b9eaa9fc94baed5b48c71d09e2b4d9aefd9162be19554b34`。
原始请求、逐查询结果、TREC、回执和快照保存在 Git 外。

## 新闻 20k 本地浏览器演示与恢复验收

本地恢复并核对交接清单中的 10 个索引与向量文件，验证语料指纹
`1480ab10cbefcb57bc1132871420b1f88d18ac5f260150080953f9441377a861` 和
Qdrant 快照 SHA-256 `143724e357d88a40ed5ec5f0d0a22f2f3c1e3769e8aa6d0cad35859100104036`。
从干净的 Linux Qdrant 1.16.0 实例恢复快照，精确计数为 20,000。
固定 `rules-v1`、候选池 200、RRF 60 和 `hnsw_ef=256`，使用
`scripts/online_acceptance.py` 验证健康 23/23、后端断开 10/10、
同一 HTTP 进程恢复 27/27。断开时强制 dense 和 hybrid 返回 503，
词法仍可用，自适应会明确报告回退。

Chrome 无头浏览器访问实际服务，用开发查询“抚州采茶戏如何培养青年演员并抢救散落的老剧本和曲谱？”
经表单执行四模式，并验证索引身份和路由回显。未闭合引号返回查询错误，浏览器清除先前的结果。
这些记录覆盖本地端到端演示和故障恢复。检索质量沿用冻结的 AI 池化 qrels，见[自动复评](../docs/contributions/rfc0002-news20k-dataset-card.md#自动复评)。

同一语料和快照上的另一轮本地工程验收使用冻结的 50 条开发查询。四种模式分别运行
50 次串行 HTTP 请求，200 次全部返回 HTTP 200，且客户端 journal 与同一服务进程的
标签、状态及路由逐条一致。固定 RRF 调用向量检索 50 次，规则 Router 调用 9 次（18%），
其余 41 次执行词法检索。向量调用全部成功，各模式均无 fallback。规则 Router 的服务端
决策耗时 p95 为 0.091 ms。

这轮是顺序单次工程测量，运行时同机有 BGE 文档编码，未独立预热或交错各臂。HTTP
延迟保留在原始收据中，速度比较需要独立的交错重复测量。相关性判断仍沿用冻结的 AI 标签。
完整 TREC、journal、服务日志与对账收据保存在 Git 外，
对账收据 SHA-256 为 `c5b5b167e505404b5539210e158270feb226b66479ab04db38d5573b7c2d308c`。
