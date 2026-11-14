# DuReader 100k 的 Qdrant ef 敏感性

在 DuReader mini100k v1.0 的 1,020 条内部开发查询和 4,960 条 qrels 上，分别比较 Dense 和固定等权 RRF 的 `hnsw_ef=128` 与 `256`。
四个 HTTP 条件复用同一份 100,000 点快照、同一 BM25 紧凑索引和固定 BGE-small-zh-v1.5。服务代码为 `e5ec95c`。
Qdrant Linux 1.16.0 运行于本机 WSL，回读确认 100,000 个已建图向量。查询编码在 Windows CPU FP32 上执行。
同一查询的编码在四个条件之间共享，条件按查询 ID 与条件标签的哈希决定执行顺序。候选池和请求深度均为 200，RRF 常数为 60，向量超时为 3 秒。

共核对 4,080 个预定条件，包含两次同一显式约束查询的预期 Dense HTTP 400，按空排名计入 1,020 查询分母。其余 4,078 次成功，回退为零。
每组 10,000 次查询级配对 bootstrap 使用相同的 seed `20261112`。共享编码与多次服务恢复仅用于质量对照，此处未测可比较延迟。

| 模式 | ef | nDCG@10 | Recall@10 | MRR@10 | MAP@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 128 | 0.264822 | 0.291235 | 0.409845 | 0.247843 |
| dense | 256 | 0.264822 | 0.291235 | 0.409845 | 0.247816 |
| hybrid | 128 | 0.228366 | 0.265769 | 0.347645 | 0.207862 |
| hybrid | 256 | 0.228366 | 0.265769 | 0.347645 | 0.207860 |

| 模式 | 指标 | 256 减 128 | 配对 95% CI |
| --- | --- | ---: | --- |
| dense | nDCG@10 | +0.000000 | [+0.000000, +0.000000] |
| dense | Recall@10 | +0.000000 | [+0.000000, +0.000000] |
| dense | MRR@10 | +0.000000 | [+0.000000, +0.000000] |
| dense | MAP@100 | -0.000027 | [-0.000051, -0.000012] |
| hybrid | nDCG@10 | +0.000000 | [+0.000000, +0.000000] |
| hybrid | Recall@10 | +0.000000 | [+0.000000, +0.000000] |
| hybrid | MRR@10 | +0.000000 | [+0.000000, +0.000000] |
| hybrid | MAP@100 | -0.000002 | [-0.000031, +0.000025] |

| 模式 | Top-200 同序查询 | Top-200 同集合查询 | 非空排名的平均集合 Jaccard | 新旧 ef=256 Top-100 同序 |
| --- | ---: | ---: | ---: | ---: |
| dense | 72/1,020 | 72/1,020 | 0.953921 | 1019/1019 |
| hybrid | 214/1,020 | 393/1,020 | 0.984648 | 1020/1020 |

Top-200 同序与同集合将 Dense 的共同空排名计入。平均 Jaccard 忽略共同空排名。旧 ef=256 参考来自同一份已验收 100k 快照的 TREC run，比较的是有序文档 ID。
本地结果的适用范围是该 CPU FP32 快照的 ANN 参数对照。PR #9 的 CUDA 快照、CE 质量与成本、808 条全量留出仍按各自的冻结协议验证。 [100k 资源记录](../../evaluation/RESOURCES.md)

## 复验

使用资源记录中的同一份紧凑词法索引、BGE 模型与 Qdrant 快照。恢复快照后，分别用 `online.server` 启动两个配置。两组都设置 `--encoder-device cpu --candidate-k 200 --rrf-c 60 --vector-timeout-s 3`，分别指定 `--hnsw-ef 128` 和 `--hnsw-ef 256`，并从 `/index/info` 核对有效参数和索引身份。 [服务配置说明](../online-serving.md)

对各端点各运行一次 `dense` 和 `hybrid`。Dense 使用冻结的 `dense-http-unsupported.json`，该文件绑定查询 SHA-256 与唯一一条允许返回 HTTP 400 的查询。采集深度 200，随后按 MAP@100 重算。

```bash
python -m evaluation.evaluate --endpoint "$ENDPOINT/search" --queries "$DEV_QUERIES" \
  --qrels "$DEV_QRELS" --mode "$MODE" --k 10 --map-k 200 \
  --output-run "$RUN" --fail-on-fallback
# Dense 在上述命令附加 --unsupported-policy "$DENSE_SKIP_POLICY"。
python -m evaluation.evaluate --run "$RUN" --qrels "$DEV_QRELS" --k 10 --map-k 100
python -m evaluation.compare --qrels "$DEV_QRELS" --run-a "$RUN_EF128" --run-b "$RUN_EF256" \
  --metric ndcg --k 10 --map-k 100 --samples 10000 --seed 20261112
```

依次替换 `--metric` 为 `recall`、`mrr` 和 `map`。逐查询排名与回读证据保存在 Git 外的 `ann-ef-sensitivity-v1`，发布只附聚合统计和输入身份。

- 查询 SHA-256 为 `ce010e6ea3a498a696d653b58bcf7976b5101d4034341983429df0a44837cfd9`。
- qrels SHA-256 为 `e75a68b89fa8a1ed1879e6703cffc1d6514ba295c82667dc4ce574e24b318f30`。
- Qdrant 快照 SHA-256 为 `4d5bf2e061a30a4be4b51711e951637851a510fec73acd9937e4c811bc3dbcfd`。
- 结果摘要 SHA-256 为 `438d6bab3576cbc4aa559c3e987d797701deacbc5142c7a33e410229e7a12d9d`，逐查询排名 SQLite SHA-256 为 `83f54ce700c8895d49071c7432ae148f3246c6294ce64413e9e303a2d5c849bb`。
