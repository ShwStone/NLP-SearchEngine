# 8M 在线检索调优记录

## 最终结果

本轮使用官方 dev 中固定顺序的前 64 条查询，比较原始服务与最终服务的完整 HTTP BM25 延迟。两边均返回 10 条结果，单客户端串行请求，预热 1 条。计时入口为 `scripts/benchmark_http.py`，客户端与本机服务通过回环地址通信。

| 64 条 BM25 查询 | 原始服务端 ms | 最终服务端 ms | 延迟比值 |
| --- | --- | --- | --- |
| 平均 | 853.875 | 68.156 | 12.53× |
| p50 | 685.000 | 43.000 | 15.93× |
| p95 | 2369.000 | 167.700 | 14.13× |
| 最大 | 3720.000 | 609.000 | 6.11× |

`official8m-final-parity64.json` 对每条查询逐个比较结果。64 组 Top-10 文档 ID 与顺序全部一致，响应中六位小数分数的最大差值为 0。加速来自计算与存储路径，未牺牲这一组查询的排名。

最终热轮 Dense 使用 `ef=64`、候选池 200、Qdrant 8 工作线程和 Torch 8 工作线程。32 条完整 HTTP 查询的服务端 p50 为 15 ms，p95 为 19.7 ms。两路并发 p50 为 23.5 ms，四路并发 p50 为 48.5 ms。这里的 Dense 数字描述缓存已热的运行状态，冷启动与缓存受扰动的首轮另行保留在收据中。

## 实验环境与口径

| 项目 | 配置 |
| --- | --- |
| CPU | AMD Ryzen 9 5950X，16 核 32 线程 |
| 内存 | 62 GiB，正式运行阶段约 25 GiB 可用 |
| 词法数据 | 8,096,668 篇，8,578,589 词项，compact 约 39 GiB |
| 向量数据 | 8,096,668 条 512 维 FP32，Qdrant 快照约 18 GiB |
| 词法存储 | 本机 NVMe 上的 compact-v1 / compact-packed-v1 |
| Python | 3.10.14 |
| 数值依赖 | NumPy 1.26.4、Numba 0.60.0 |
| 向量依赖 | Qdrant 1.16.0、Torch 2.3.0+cpu、Transformers 4.46.3 |
| 查询模型 | 固定 revision 的 BGE-small-zh-v1.5，CPU 查询编码 |

每个 JSON 收据包含查询 ID、查询文本、客户端时间、服务端时间、阶段时间、Top-10 ID 与分数。p50 和 p95 使用脚本内的线性插值分位数。平均耗时包含所有正式请求，预热请求单独存放。原始收据位于 [experiments](experiments/)。

测量分为三类。同一组 64 条查询用于 BM25 严格对照，前 16 条用于观察各优化阶段的收益，前 64 条及前 32 条用于向量参数与并发实验。冷轮会受首次磁盘读取、JIT 后的页面缓存和此前任务影响，热轮用于描述持续服务状态。跨轮数据只按表中口径解释。

## 基线定位

最初 16 条 BM25 查询的服务端 p50 为 775 ms，p95 为 3462.75 ms，词法阶段 p50 为 754.33 ms。文档读取与序列化只占很小部分，主要成本来自 Python 逐 posting 计算 BM25、构建完整分数字典和 Top-K 选择。

同一批 16 条 Dense 查询在冷轮中的服务端 p50 为 917.5 ms。查询编码 p50 只有 11.65 ms，向量阶段 p50 为 904.22 ms。Qdrant 使用 on-disk HNSW，33 个 segment，首次访问不同图节点时延迟差异明显。

两条链路需要分开调。词法侧先减少 posting 级 Python 工作，向量侧先区分 HNSW 参数、磁盘缓存、候选身份校验和数据库线程限制。

## 独立微基准与稀疏方案选择

DeepSeek Flash agent 在独立目录中完成合成 posting 微基准和逐比特正确性实验，主流程负责真实 8M 实验与最终取舍。合成实验使用与 `PackedIndex` 相同的 uint32 块和文档长度数组，覆盖 8M 文档形状、重复词、零权重、负权重、全并列与分页。

| 合成 8M 查询 | 标量 ms | NumPy 稀疏 ms | 比值 |
| --- | --- | --- | --- |
| 1 词，df=10 | 0.005 | 0.010 | 0.48× |
| 1 词，df=1k | 0.389 | 0.026 | 14.97× |
| 1 词，df=100k | 46.587 | 0.635 | 73.37× |
| 1 词，df=1M | 465.188 | 12.690 | 36.65× |
| 8 词，每词 df=1k | 2.481 | 0.317 | 7.83× |
| 8 词，每词 df=10k | 25.366 | 2.019 | 12.56× |

实验发现每个词都创建 8M 评分向量会带来固定分配与全数组加法成本。小倒排表因此出现数量级回退。最终采用稀疏并集，并设置 `total_postings >= 64 * scored_terms` 的进入门限。低于门限时沿用标量路径。

多词路径先把各词的 docid 和 BM25 贡献按原顺序拼接，再通过 `np.unique(..., return_inverse=True)` 找到命中文档并集，用一次 `np.bincount` 按词项顺序累加。单词的 docid 已严格递增，无需并集排序。Top-K 先由 partition 找分数边界，再只排序边界附近的数据，同分时按 docid 升序补齐。

`PackedIndex` 新增的数组访问器只暴露已加载的只读块，不改变 compact 文件。没有 NumPy、显式布尔过滤、`match_all` 或普通内存索引时继续使用原实现。TF-IDF 保留标量路径。

## JIT 解码与缓存

稀疏打分后，16 条真实查询的服务端 p50 降到 221 ms。首次读取大倒排块仍要逐个解析 `[docid, tf, positions...]`，这一阶段成为主要成本。

`offline/packed_decode.py` 用 Numba 在机器码循环中完成相同的头部定位、边界检查和数组生成，再转换回原有 `array('I')` 块。服务监听前用合法小块完成编译，索引文件格式与校验和不变。Numba 不可用时保留原 Python 解码。

缓存预算通过 `retrieval.lexical_cache_bytes` 显式配置。正式 8M 服务使用 536,870,912 字节。该预算约束保留的块，不包含请求中的 NumPy 临时数组、文档缓存或 Qdrant 内存。一次 16 条查询与热轮复测后，缓存中有 143 个词项块，占用 445,783,692 字节，淘汰次数为 0。

| 前 16 条 BM25 阶段 | 服务端 p50 ms | 服务端 p95 ms |
| --- | --- | --- |
| 原始路径 | 775.0 | 3462.75 |
| 稀疏 NumPy | 221.0 | 1458.50 |
| 稀疏 + Numba + 512 MiB | 65.5 | 477.25 |
| 较低阈值的自适应累加 | 80.5 | 416.00 |
| 最终阈值方案的首 16 条子集 | 58.5 | 369.00 |

这些行来自不同运行轮次，用于定位成本变化。最终结论使用首页的 64 条同组对照。

## 大并集的第二次取舍

缓存全命中后，最重查询仍受 `np.unique` 排序影响。原自适应方案在 postings 超过约 50 万时切换为一次全文 `bincount`，热轮 p95 从 321.5 ms 降到 240 ms，但 p50 从 38.5 ms 升到 56.5 ms。中等查询为全文数组付出了额外成本。

逐查询统计 postings 后，两个明显受益的查询分别包含 3,337,870 和 3,795,858 条 posting。最终门限设为 `max(2 * 1024 * 1024, n_docs / 4)`。小中查询继续走稀疏并集，大并集才分配全文分数与命中掩码。两条路径使用相同的 BM25 运算顺序和 Top-K 规则，强制路径测试比较到浮点十六进制表示。

## HNSW 参数实验

向量参数实验复用同一批预计算 BGE 查询向量，直接测量 Qdrant 阶段。参照为 `ef=256、candidate_k=200` 的 ANN Top-10，再用 dev qrels 计算实际指标。64 条结果如下。

| 参数，candidate=200 | 向量 p50 ms | 参照 Top-10 平均重合 | 有序一致查询 | nDCG@10 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| ef=32 | 92.07 | 91.72% | 43 / 64 | 0.246611 | 0.285020 |
| ef=64 | 24.44 | 96.41% | 51 / 64 | 0.250629 | 0.305853 |
| ef=128 | 39.39 | 99.06% | 59 / 64 | 0.250282 | 0.305853 |
| ef=256 | 63.20 | 100% | 64 / 64 | 0.255258 | 0.308978 |

`ef=64` 与 `ef=128` 在这 64 条查询上的 Recall@10 相同，nDCG@10 接近。网站选择 `ef=64` 提供更快的交互响应，同时保留它相对 `ef=256` 的质量差异。需要按高质量口径复评时可切回 128 或 256。

早期 16 条网格还测试了候选池 50。部分小候选组合比候选池 200 更慢，因此没有缩小候选池。该现象与 33 个 segment 的搜索规划和每个 segment 的固定开销同时出现，收据保留原始数据，未据此推断普遍规律。

## 外部 ID 校验的位置

初版 `_vector_search` 对 200 个 ANN 候选逐篇读取文档并核对外部 ID，页面最终只展示 10 篇。校验时间被计入向量阶段，混合词法实验后还会触发额外随机读取。

最终版本在候选阶段检查内部 ID 范围、重复项和 cosine 分数。外部 ID 只针对最终展示页核对，并复用渲染所需的文档读取。自适应遇到展示页 ID 不一致时明确回退到词法结果，dense 和 hybrid 保留后端契约错误。完整四模式回归与错误路径测试均通过。

在相同的 64 条热轮口径下，原始 `ef=256` 与全候选校验的服务端 p50 为 34 ms，采用 `ef=64` 与展示页校验后的 p50 为 21 ms。该差值同时包含 HNSW 参数和校验位置的变化，分项证据见对应收据。

## Qdrant 与 Torch 并发调优

Qdrant 原配置把 `max_search_threads` 和 `max_workers` 都设为 2。仅将二者调整为 8 后，向量阶段在 1、2、4 路并发下都保持约 8 ms，瓶颈转移到 BGE 编码。随后把 Torch 的 OMP/MKL 工作线程从 2 调整为 8。

下表使用同一组 32 条 dense 查询。服务端列为完整 HTTP p50，向量列为其中的 Qdrant 阶段 p50。

| 并发与配置 | 服务端 p50 ms | 向量 p50 ms | 编码 p50 ms |
| --- | --- | --- | --- |
| 2 路，Qdrant 2 线程，Torch 2 线程 | 225.5 | 208.47 | 15.77 |
| 2 路，Qdrant 8 线程，Torch 2 线程 | 30.0 | 7.89 | 19.83 |
| 2 路，Qdrant 8 线程，Torch 8 线程 | 23.5 | 9.14 | 12.40 |
| 4 路，Qdrant 2 线程，Torch 2 线程 | 416.5 | 398.32 | 51.05 |
| 4 路，Qdrant 8 线程，Torch 2 线程 | 60.0 | 7.98 | 51.05 |
| 4 路，Qdrant 8 线程，Torch 8 线程 | 48.5 | 9.32 | 37.51 |

最终单路热轮的编码 p50 为 6.17 ms，向量 p50 为 7.09 ms。公网 API 连接限制设为单客户端 4 路、全站 8 路，并保留请求速率限制。超过连接上限的请求返回 429，避免在饱和后继续排队。

## 可复现命令

```bash
# 完整 HTTP 延迟
python3 scripts/benchmark_http.py \
  --endpoint http://127.0.0.1:18086/search \
  --queries /path/to/queries.dev.jsonl --limit 64 \
  --modes lexical --k 10 --warmup 1 --concurrency 1 \
  --label official8m-final-c1 --out /path/to/official8m-final-c1.json

# 排名与分数对照
python3 scripts/compare_benchmark_results.py \
  --baseline /path/to/official8m-baseline64-c1.json \
  --candidate /path/to/official8m-final-c1.json \
  --out /path/to/official8m-final-parity64.json

# 固定查询向量的 HNSW 参数网格
python3 scripts/benchmark_vector.py \
  --manifest /path/to/vector_manifest.json --url http://127.0.0.1:26333 \
  --model-path /path/to/bge --queries /path/to/queries.dev.jsonl \
  --limit 64 --ef 32 64 128 256 --candidates 200 \
  --label hnsw-validation --out /path/to/hnsw-validation.json
```

生产配置见 [fast profile](../deploy/config/duretrieval-official-fast.example.json) 与 [Qdrant 配置](../deploy/config/qdrant-official.example.yaml)。网站与本机服务的启动、隧道和回滚见 [部署文档](MULTI_DATASET_SERVING.md)。

## 研究参考

- [BM25S 的 eager sparse scoring 与 Top-K 选择](https://arxiv.org/abs/2407.03618)
- [Block-Max WAND 与 Lucene 的查询剪枝](https://ncbi.nlm.nih.gov/pmc/articles/PMC7148045/)
- [Qdrant 性能与内存配置](https://qdrant.tech/documentation/ops-optimization/optimize/)

本轮没有重建 HNSW 图或更换量化方案。现有 8M 快照继续使用 FP32 向量与 on-disk HNSW，因此实验结果可以直接对应同一批冻结文档向量和既有质量评测。
