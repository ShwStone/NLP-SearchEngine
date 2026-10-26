# 外部资源与实验边界

本仓库不提交课程数据、模型权重或 Qdrant 产物。开题报告中的 100001 篇文档、2000 条查询和 9839 条 qrels 只能作为团队计划，不能替代公开资源的实际清单。拿到任一版本后，先用下面的通用命令校验数量、ID、qrels 引用和固定划分：

```bash
python3 -m evaluation.prepare_dataset \
  --corpus data/raw/corpus.jsonl \
  --queries data/raw/queries.jsonl \
  --qrels data/raw/qrels.tsv \
  --out data/duretrieval \
  --expected-docs 100001 \
  --expected-queries 2000 \
  --expected-qrels 9839
```

官方公开的 DuReader-Retrieval 版本不是同一份课程快照。官方说明给出的规模约为 86K 条训练查询、2K 条开发查询、4K 条测试查询和 809 万段落，并要求在百度 AI Studio 竞赛页注册后下载：

- [DuReader-Retrieval 官方说明](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval)
- [RocketQA 官方基线](https://github.com/PaddlePaddle/RocketQA/tree/main/research/DuReader-Retrieval-Baseline)
- [基线下载脚本](https://raw.githubusercontent.com/PaddlePaddle/RocketQA/main/research/DuReader-Retrieval-Baseline/script/download.sh)

本次已从 RocketQA 官方下载脚本指向的公开 CDN 获取 baseline archive，并记录下载包 SHA256：
`b051d3d966f7b0288aed95a923fc3b706176767d8bdcff8aa8fe5fff84b20662`。
该包实际包含 8,096,668 个 passage、2,000 个 dev 查询和 9,839 条去重后的 dev qrels（原始答案段落链接为 9,863 条）；四个 passage 分片各有 2,024,167 行。它没有官方 test 查询和 test qrels，因此这些 dev 数据只能标记为 `dev`，不能改名为 `test`，也不能用来声称完成课程目标版本。

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

为先验证完整评测闭环，已从官方 dev corpus 固定取前 20,000 篇文档，并只保留其中有 qrel 命中的 19 条查询／19 条 qrel。该前缀索引不代表全量结果：BM25 的 nDCG@10、Recall@10、MRR@10、MAP@100 分别为 0.5032、0.5789、0.4781、0.4920；这组结果仅是官方数据前缀上的工程冒烟。此前记录的 HTTP 延迟来自不同的服务配置，不能与下面的集群直接调用延迟混用。

在集群上又完成了同一 prefix20k 的真实 embedding 四模式冒烟。模型固定为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，revision `ef8c211226b894fcb81acc59f3b34ba3efd5f42`，384 维、`max_tokens=256`、cosine、CUDA 编码；模型 `model.safetensors` SHA256 为 `eaa086f0ffee582aeb45b36e34cdd1fe2d6debe61f8a559a1bbc9bd9d955917b`。Qdrant 使用 `qdrant-client==1.16.0` 的 local embedded collection，并将 HNSW `m=0`、indexing threshold 设为 0，作为可复现的精确／全扫描工程配置；这不是 HTTP Qdrant 或 HNSW 性能验收。19/19 条查询的 dense 和 hybrid 向量请求成功，adaptive 触发向量 10/10 次且没有 fallback。结果如下（均为该 19 条查询的 prefix smoke；延迟由集群脚本直接调用检索引擎测量）：

| 模式 | nDCG@10 | Recall@10 | MRR@10 | MAP@100 | 平均延迟 ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | 0.503242 | 0.578947 | 0.478070 | 0.491982 | 24.726959 | 24.324300 | 35.187026 |
| dense | 0.383132 | 0.578947 | 0.326462 | 0.326462 | 182.851627 | 189.567106 | 212.146423 |
| hybrid | 0.533961 | 0.789474 | 0.455556 | 0.457929 | 183.931793 | 193.680419 | 208.295847 |
| adaptive | 0.483885 | 0.578947 | 0.453216 | 0.465164 | 124.681510 | 106.596140 | 229.564881 |

对 nDCG@10 做 20,000 次逐查询 paired bootstrap（主比较 seed=2026，hybrid−adaptive seed=2027），差值和 95% 区间为 lexical−dense `-0.120108 [-0.370730, 0.118329]`、lexical−hybrid `0.030711 [-0.110777, 0.158990]`、lexical−adaptive `-0.019357 [-0.103473, 0.038510]`、hybrid−adaptive `-0.050076 [-0.159552, 0.063509]`。这验证了真实模型、真实 SDK 和持久化向量检索在小前缀上的闭环，但没有声称已验收 HTTP Qdrant 服务、HNSW 性能或 8M 全量语义实验。运行目录、模型和向量产物均保留在集群，不提交到仓库。

为了验证工程闭环，公开的 [C-MTEB 快照镜像](https://github.com/Lsogod/everything-ent-hybrid-rag/tree/main/data/knowledge/datasets) 可在当前网络中取得，但它只有 5000 篇文档、100 条查询和 425 条 qrels。它适合冒烟评测，不能替代课程全量数据；本地实验结果也不能写成提案的最终成绩。

RocketQA 脚本中的数据和 ERNIE 权重来自 `dataset-bj.cdn.bcebos.com`。数据包已完成哈希和结构核验；上述 prefix20k 的 GPU 编码和四模式结果已经在集群完成，但 8M 规模索引、课程目标快照、HTTP Qdrant/HNSW 以及 test labels 仍未验收。按 [评测说明](README.md) 构建 BM25、向量索引和四组实验时，模型 revision、维度、前缀、截断规则和 Qdrant 后端参数必须随实验结果一并记录。

Qdrant 适配器和快照发布逻辑已在 `offline/vector_build.py`、`online/vector_index.py` 中实现。没有可用 Qdrant 实例时，内存向量替身只能用于契约测试，不能作为 HNSW 性能结论。

## 自建网络测试集冒烟（待组员复核）

为响应课程教师关于自行构造测试集的建议，2026-10-26 使用 `evaluation.crawl_web`
从公开的 [中国政府网](https://www.gov.cn/) 入口做了一次受限抓取：同域、深度 1、最多
10 篇页面、单页最多 500 KB、请求间隔 0.2 秒，并通过 robots 检查。临时 corpus 有 10 篇
文档；随后根据页面标题和主题写了 3 条**暂定**查询及 0/1/2 判断，仍需组员人工复核，不能
视为教师认可的正式 qrels。临时产物不在仓库中，便于复核的 SHA256 为：corpus
`e2bad4f46d3108af54b8a9432fb393dd203b00d3acf3e07f1f42ad1ee672ba73`、queries
`ff5c5e949870796c62b3445917eebc67edb80869b780a8114779aae9d1db1d5d`、qrels
`71723fb7d275d34014dd4fe7083fbc4b4849099ca852f1fbc03290042f8689ff`（显示时折行仅为排版，
实际摘要为连续 64 个十六进制字符）。

在这组 10 文档/3 查询的临时语料上，仅用本项目 lexical HTTP 服务做了管线冒烟：
nDCG@10=0.767010、Recall@10=1.000000、MRR@10=0.703704、MAP@100=0.703704，平均延迟
77.36 ms、p50=39.17 ms、p95=146.06 ms。该数值受页面内容、暂定标注和 tiny 规模影响，
不是网络数据集的正式结论，也不能与 DuReader 结果混合。后续应由组员复核标注、扩充查询，
再用同一划分运行四种模式和 paired bootstrap。
