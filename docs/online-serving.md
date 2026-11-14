# 在线服务部署与验收（RFC-0003）

本文说明如何用**版本化服务 profile** 启动在线检索、如何理解启动身份回显、请求硬化与日志隐私，
以及如何用仓库自带的公开 HTTP 验收脚本分阶段核对。范围只覆盖在线服务化：
Router 算法（RFC-0001）、评测与质量结论（RFC-0002/0004）、离线构建不在本文。

本文命令是工程接线说明，不是性能或质量承诺。仓库不随源码提供数据、模型权重、Qdrant 存储和
快照；本文列出的 identity 数值来自已复核的本地产物，你重建产物后必须同步更新配置与验收期望。

## 1. 前置条件与固定版本

在线服务本身只用标准库；接真实向量后端与固定 BGE 模型时额外需要下表依赖。

| 组件 | 固定值 | 说明 |
|---|---|---|
| Python | `3.10.14` | 本仓库已复核的在线运行环境（更晚的 3.x 未在此链路复验） |
| PyTorch | `2.3.0+cpu` | 按硬件单独安装；FP32 前向 |
| Transformers / tokenizers | `4.46.3` / `0.20.3` | 见 `requirements-offline.txt` |
| NumPy | `1.26.4` | `requirements-offline.txt` 允许 `>=1.24,<2` |
| huggingface-hub / safetensors | `0.26.5` / `0.5.3` | 离线加载固定 revision |
| qdrant-client | `1.16.0` | collection metadata 依赖 1.16+ |
| Qdrant 服务端 | `1.16.0` | 只读访问，服务不会建库或写入 |
| 查询编码模型 | `BAAI/bge-small-zh-v1.5` @ `7999e1d3359715c523056ef9478215996d62a620` | 512 维、CLS、右截断、L2 归一化；query prefix `为这个句子生成表示以用于检索相关文章：` |
| 校验和清单 | `adapters/embedding.py` + `adapters/model_files.py` | 本地模型目录必须带固定 revision 的 SHA256 清单 |

```bash
python3 -m venv .venv-serving
# 按硬件安装 torch >=2.1,<2.6（例如 CPU 轮子），再安装其余固定版本
.venv-serving/bin/pip install -r requirements-offline.txt
```

在线服务只读访问已发布的产物；启动前先过 `offline.validate`（见第 5 节）。

## 2. 两个数据集 profile

`dataset` 必须与产物 manifest 中的字符串**完全一致**，不做别名映射。身份不匹配时服务拒绝启动，
而不是降级成"另一个语料的结果"。

| profile | `dataset` | 文档数 | corpus_sha256 | collection | 产物布局（示例） |
|---|---|---|---|---|---|
| news | `rfc4-clean-news-20k` | 20000 | `1480ab10cbefcb57bc1132871420b1f88d18ac5f260150080953f9441377a861` | `news_rfc4_clean20k` | `artifacts/news20k/{compact,vector}` |
| official | `dureader-official-dev-full-8096668` | 8096668 | `0ae8b1174d0ee8e55f269c3c6d8e5f12d7e9c79fcb72c3966e99de18c9061019` | `real_bge_official8096668` | `artifacts/official/{compact,vector}` |

新闻集是团队自爬 20,000 篇；官方集是 DuRetrieval 全量 passage 的冻结切片身份，
不含官方 test labels。两者不能混用同一个 Qdrant collection 或同一份配置。

## 3. 配置文件（schema_version 1）

示例文件在仓库内，只作模板：`deploy/config/news20k.example.json`、
`deploy/config/duretrieval-official.example.json`。复制到**仓库外**的部署目录再改路径，
例如 `~/nlp-serving/deploy/config/news20k.json`，并让 `artifacts/`、`models/` 相对该目录存放：

```text
~/nlp-serving/deploy/
  config/news20k.json
  artifacts/news20k/compact/…            # A1/compact 词法产物
  artifacts/news20k/vector/vector_manifest.json
  models/bge-small-zh-v1.5-7999e1d3/…    # 带 SHA256 清单的本地模型目录
```

```json
{
  "schema_version": 1,
  "dataset": "rfc4-clean-news-20k",
  "index": "../artifacts/news20k/compact",
  "vector": {
    "url": "http://127.0.0.1:6333",
    "manifest": "../artifacts/news20k/vector/vector_manifest.json",
    "hnsw_ef": 256,
    "timeout_s": 30.0
  },
  "embedding": {"model_path": "../models/bge-small-zh-v1.5-7999e1d3", "device": "cpu"},
  "retrieval": {"candidate_k": 200, "rrf_c": 60.0},
  "serve": {"host": "127.0.0.1", "port": 8000, "max_body_bytes": 65536,
            "request_timeout_s": 10.0},
  "policy": {"factory": "rule"}
}
```

| 字段 | 必填 | 约定 |
|---|---|---|
| `schema_version` | 是 | 只能是 `1` |
| `dataset` | 是 | 必须与词法 manifest 的 `dataset` 完全一致（见第 2 节） |
| `index` | 是 | 词法产物目录（普通或 compact A1 产物） |
| `vector` | 否 | 省略即纯词法 profile：dense/hybrid 返回 503，adaptive 显式回退。块内 `url`、`manifest` 必填；`hnsw_ef`、`timeout_s` 可选，示例固定为 256/30.0，实际生效值看 `/index/info` 的 `search` 回显。省略 `hnsw_ef` 时服务不传 Qdrant `SearchParams`，回显为 `null`（不代填 256），所以验收脚本要求向量 profile 显式给出 `--expect-hnsw-ef` |
| `embedding` | 否 | 整个块可省略：只给 `vector` 时用仓库通用 pinned 编码器（需本地已有对应 revision），等价于只省略 `model_path`；反过来 `embedding` 不能脱离 `vector` 单独出现。`device` 只在给出 `model_path` 时有意义（默认 `cpu`） |
| `retrieval` | 否 | `candidate_k` 默认 200、`rrf_c` 默认 60.0；非 lexical 必须 `offset+k <= candidate_k` |
| `serve` | 否 | `host` 默认 `127.0.0.1`，`port` 默认 8000，`max_body_bytes` 默认 65536，`request_timeout_s` 默认 10.0 |
| `policy` | 否 | `factory` 默认 `rule`；其他值必须是受信本地 `module:factory` 导入路径，工厂收到配置后返回带 `policy_version` 的 Router 对象 |
| `rerank` | 否 | 固定混合候选重排实验；需要 `vector`。必填 `model_path` 与 `depth`（1 至 40，且不超过 `candidate_k`）；可选 `device`（默认 `cpu`）、`batch_size`（1 至 32，默认 20）、`capture_features`（默认 `false`）及 `gate`（仅 `depth=40`） |

两个示例 profile 都用 8000 端口；同时运行两个实例时给 `port` 不同值，不要复用同一端口。

加载规则（启动即拒绝，不做静默修正）：

- 未知字段、错误类型、非有限数值（NaN/Infinity）、空路径一律报错；不做单位或类型转换。
- **所有路径相对配置文件所在目录**解析，不是相对当前工作目录。不做 shell/环境变量插值：
  `$VAR`、`~` 都按字面字符拼成路径，通常会在启动加载产物时因路径不存在而失败（这是启动加载
  报错，不是加载器按字符模式无条件拒绝）。绝对路径是允许的：仓库内模板只写相对路径，
  私有部署配置（仓库外）可以写本机绝对路径，但配置本身仍不得包含凭据。
- `vector.url` 是 Qdrant 的 `http(s)` 服务地址；带 userinfo、query 或 fragment 的 URL 直接拒绝，
  报错不回显这些片段（避免把凭据写进日志或异常）。默认只监听回环地址；显式配置
  非回环 `host` 时会打印明确警告。
- `dataset` 不做别名（`news20k`、`rfc4-clean-news20k` 都不是合法值）。

### RFC-0001 策略与固定重排对照

要加载组员的 BM25 收益 Router，设置 `policy.factory` 为
`online.gain_router:load_gain_router`，并在本机进程提供 `NLP_PEER_GAIN_MODEL`
与 `NLP_PEER_GAIN_SHA256`。工厂核对模型文件摘要，启动时再核对词法和向量
manifest、候选池、RRF 常数及向量查询实际生效的 ANN 设置。早期实验脚本声明
ef=128 时尚未实际传给 Qdrant，复验应重新捕获训练排序并生成相同配置的模型。
当前可选动作包括 lexical、dense、hybrid。评测包含 dense 动作的策略时使用
`evaluation.evaluate --mode adaptive --allow-adaptive-dense`。

使用 [BAAI 的固定 revision](https://huggingface.co/BAAI/bge-reranker-base/tree/2cfc18c9415c912f9d8155881c133215df768a70)
准备重排模型。下载器复用独立缓存，校验 safetensors 权重和分词文件后发布本地目录。
启动服务从该目录加载，推理使用 FP32，最大输入长度为 256 个 token。

```bash
python -m scripts.download_reranker --out artifacts/bge-reranker-base \
  --cache-dir artifacts/hf-download-cache
```

对照分别启动无 `rerank` 块的 RRF 基线和含 `rerank` 块的服务，三者共用同一个
Qdrant collection、词法索引、编码器和候选配置。将下面的块加入重排 profile，
CE28 与 CE40 各用自己的端口，`depth` 分别设为 28 与 40。

```json
"rerank": {"model_path": "../artifacts/bge-reranker-base", "depth": 28,
           "device": "cpu", "batch_size": 20}
```

该设置对自然语言的固定 `hybrid` 请求生效。CE 给 RRF 前缀打分，按 CE 分数降序和
外部文档 ID 升序排列，再追加原尾部，最后按请求分页。原 RRF 分数保留在 `score`。
显式布尔和短语查询沿用原结果。服务在 `/index/info` 回显模型身份，成功重排时返回
`rerank.depth`、`rerank.scored` 和 `timings_ms.rerank`。模型打分失败返回
`503 reranker_unavailable`，以便评测拒绝该请求。

```bash
python -m evaluation.ce_http \
  --baseline http://127.0.0.1:8000 --rerank 28 http://127.0.0.1:8028 \
  --rerank 40 http://127.0.0.1:8040 --queries data/queries.dev.jsonl \
  --warmup-queries data/queries.warmup.jsonl --qrels data/qrels.dev.tsv \
  --out runs/ce-fixed-controls
```

预热查询须与评测查询的 ID 和归一化文本均不重叠。每条查询的各臂全部成功后才
写入可续跑的 SQLite 记录，评分在 HTTP 采集结束后进行。结果记录实际 CE pair、
完整 HTTP 延迟和 10,000 次配对 bootstrap。中断续跑的延迟分批报告。

本机真实冒烟通过 8 次请求，其中 3 条自然查询共执行 84 个 CE pair，另 1 条
显式约束查询在两套服务的排名相同。两套服务使用同一个 100k Qdrant 恢复实例，
实际 `hnsw_ef=128`。验收回执 SHA-256 为
`30b4a5650e11ead6b59d4cc00e3261d5552495edfc2769aefbd4a18ecbcc129f`。
完整 100k 开发集的固定对照和等预算排名复算见[课程报告](../REPORT.md#同检索配置下的-100k-等预算开发对照)。全量 CE 留出复验另行验收。

### 学习门控 CE40

训练采集使用无 `rerank` 的 RRF 服务和固定 CE40 服务。在固定 CE40 的
`rerank` 块设 `"depth": 40, "capture_features": true`，不设 `gate`。
按[评测手册](../evaluation/README.md#train-and-evaluate-a-ce40-gate)采集训练集，
生成门控 JSON 和训练回执。另启一个 profile，保留固定 CE40 的模型、批大小和检索设置，
在其 `rerank` 块增加以下对象（摘要替换为回执的 `model_sha256`）：

```json
"gate": {"model_path": "../artifacts/ce40-gate.json", "sha256": "<64-hex SHA256>"}
```

服务在监听前校验 JSON 摘要、特征顺序和系数，并比对训练绑定的词法索引、向量索引、
搜索配置及 CE 模型身份。现有门控文件的启动校验仅覆盖这些字段。正式 HTTP 评测传入
`--fit-receipt`，另行核对查询编码器运行时与 BM25 加速配置，并保持各臂的 CE 设备和
其他运行条件一致。`/index/info` 的 `reranker` 回显 `gate_sha256` 和
`capture_features: true`，不需再显式开启特征采集。

门控只用于无显式约束且 RRF 候选非空的固定 `hybrid` 请求：标准化 BM25 特征后的
线性分数大于 0 时执行 CE40，否则保留 RRF 顺序并返回 `rerank.scored: 0`。
两种门控结果都回显 `rerank.features` 和 `rerank.gate_sha256`；显式约束、空候选
和其他检索模式不返回 `rerank` 对象。

### CLI 选择

```bash
# 版本化 profile：唯一入口，禁止再叠加零散 flag
python3 -m online.server --config ~/nlp-serving/deploy/config/news20k.json

# 旧词法入口仍然可用（无向量依赖）
python3 -m online.server --index /path/to/compact --port 8000
```

`--config` 与所有旧的 `--index`、`--vector-*`、`--bge-model-path`、`--encoder-device`、
`--candidate-k`、`--rrf-c`、`--hnsw-ef`、`--host`、`--port` 以及路由/`--router-factory` 等
flag **互斥**，没有任何覆盖优先级：同一个字段只有一处来源。无 `--config` 时旧的
`--vector-*` 接线照旧，另外可用 `--router-factory`、`--vector-timeout-s`、
`--max-body-bytes`、`--request-timeout-s`，默认值与上表一致。

## 4. 启动与停止（前台，单进程）

```bash
# 1) 先验产物身份（发布门禁，不启动服务）
python3 -m offline.validate \
  --lexical-index ~/nlp-serving/deploy/artifacts/news20k/compact \
  --vector-artifacts ~/nlp-serving/deploy/artifacts/news20k/vector

# 2) 前台启动；Ctrl-C 停止
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -m online.server --config ~/nlp-serving/deploy/config/news20k.json

# 3) 另一个终端确认就绪（只读）
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/index/info
```

身份校验在开始监听之前完成，失败时进程直接退出，不会留下半开的监听端口。
仓库不提供 daemon/PID 文件框架：服务是前台进程，`Ctrl-C` 即停。需要托管时由运维自行接
systemd 或容器，进程管理、日志轮转和资源上限都属于部署方职责，不在本仓库承诺范围内。
停止或重启后端口立即可复用；同一端口只应有一个实例，启动前自行检查端口占用。

## 5. `offline.validate` 与启动校验的边界

| 阶段 | 校验内容 | 不覆盖 |
|---|---|---|
| `offline.validate`（发布前） | A1 四文件与 compact 摘要、文档数、语料指纹、向量 manifest、快照文件 SHA256、provenance/report | 服务端 collection 是否真的能查询 |
| 启动（每次启动） | 配置 schema/路径/dataset；词法 manifest；向量 manifest 的 `EmbeddingSpec`、`n_docs`、`corpus_sha256` 与词法侧对齐；collection 持久化构建元数据、真实维度、Cosine 距离、`count(exact=True)`；BGE 本地模型 SHA256 清单 | **不**重新逐 GB 校验快照 SHA256 |
| 运行期 | 每次请求的正常错误包装 | **不**周期性重新核对身份 |

启动成功只说明"配置意图与产物身份对齐"，不说明后端之后一直健康。快照 SHA 复核留在
`offline.validate` 门禁；每次启动重算大快照哈希会把启动时间拉到分钟级，因此明确不做。
后端在运行期失联时服务不会崩：dense/hybrid 返回 503，adaptive 显式 `fallback=true`，
后端恢复后无需重启服务即可继续语义检索。

## 6. 就绪、身份回显与 UI

- `GET /health`：**仅**词法服务 liveness（dataset/n_docs/schema_version/analyzer_hash）。
  它不探测 Qdrant、不加载模型，因此不能当语义就绪探针。
- `vector_ready`：启动时"已配置且身份对齐"的结论，不是后端实时健康。
- `GET /index/info`：原有 `vector_ready`/`config`/`vector_manifest` 不变，新增
  `index_identity`、`vector_identity`、`search_config`、`policy`，HTTP server 另加
  `serving`。字段定义见 `contracts/RETRIEVAL.md` 第 4 节"在线服务身份与配置回显"。
  客户端必须容忍字段缺失或 `null`；`search.exact=false` 只说明允许 ANN，
  不证明本次查询实际走了 HNSW。`config.policy_version` 与 `policy.policy_version`
  都回显实际生效的策略版本（自定义 `module:factory` 策略也一致），其余检索参数不变。
- 演示 UI 读取同一份身份回显显示 dataset、文档数、语料指纹、analyzer、BM25 参数、
  collection、embedding 规格与搜索参数；缺失字段显示为不可用，不伪造数值。

## 7. 请求硬化与日志隐私

| 项 | 默认值 | 说明 |
|---|---|---|
| `max_body_bytes` | 65536 | 请求体上限；超限请求在解析前被拒绝（4xx），不进入检索 |
| `request_timeout_s` | 10.0 | socket I/O 超时（单次 body 读/连接读）；不是整请求 deadline，编码与检索耗时不受它约束 |
| `vector.timeout_s` | 30.0 | 只是 Qdrant 客户端 HTTP I/O 超时；**不覆盖**本地查询编码（BGE 前向）的耗时，两者需分别评估 |

本仓库不承诺编码器硬 deadline，也不提供并发限流/排队框架：服务仍是
`ThreadingHTTPServer` 单进程多线程，容量结论需要独立压测。真实 BGE 编码是 CPU/GPU 同步调用，
超时与限流应由前面的反向代理或进程管理器承担。

请求日志默认以 **JSONL 写 stderr**（不写访问日志），用于与评测 `run_tag` 对账：

```text
event, time, status, run_tag, requested_mode, executed_mode, reason,
vector_attempted, vector_used, fallback, policy_version, dataset,
corpus identity（短指纹）, took_ms, timings_ms 的五个阶段,
serialization_ms, http elapsed
```

隐私边界：日志**不**包含查询文本、请求头、完整异常堆栈或任何凭据；只记录工程元数据。
用 `evaluation.evaluate --request-tag` 指定单次评测的标签，再用
`evaluation.serving_cost` 对账。`evaluation.paired_http` 为每个预热和正式请求生成独立标签，
并在校验时要求所有标签恰好出现一次。未指定标签的请求在日志中保留空值。
同一份日志不含 `/search` 请求体，排查质量问题时不要指望从中重建查询。

## 8. 公开 HTTP 验收脚本

`scripts/online_acceptance.py` 只用标准库访问公开 HTTP（不 import `online.*`/`offline.*`），
把证据写到调用方指定的 JSON。`--query` 必须是目标语料上确实有词法命中的查询，
`--oov-query`（默认 `onlineacceptancezzqx2026`）用于零词法命中的语义救援场景。

```bash
# 期望身份必须由调用方显式给出；缺失不会被默默跳过
NEWS_ARGS=(--endpoint http://127.0.0.1:8000 \
  --expect-dataset rfc4-clean-news-20k --expect-n-docs 20000 \
  --expect-corpus-sha256 1480ab10cbefcb57bc1132871420b1f88d18ac5f260150080953f9441377a861 \
  --expect-collection news_rfc4_clean20k --expect-hnsw-ef 256 \
  --query '秦山核电如何在发电的同时生产医用同位素？')

# 阶段 1：后端在线的完整矩阵，记录服务实例
python3 scripts/online_acceptance.py "${NEWS_ARGS[@]}" --phase healthy --require-instance-id \
  --out /tmp/accept/healthy.json || exit 1
INSTANCE_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["instance_binding"]["verified_id"])' \
  < /tmp/accept/healthy.json)" || exit 1

# 阶段 2：停掉向量后端（不要重启服务进程），核对 503 与显式回退
python3 scripts/online_acceptance.py "${NEWS_ARGS[@]}" --phase down --expect-instance-id "$INSTANCE_ID" \
  --out /tmp/accept/down.json || exit 1

# 阶段 3：重启后端（服务进程仍不动），核对语义模式恢复
python3 scripts/online_acceptance.py "${NEWS_ARGS[@]}" --phase recovered --expect-instance-id "$INSTANCE_ID" \
  --out /tmp/accept/recovered.json
```

脚本覆盖：`/health` 的词法 liveness、`/index/info` 的 dataset/文档数/语料指纹/collection/
`hnsw_ef`/`timeout_s`/`candidate_k`/`rrf_c`/`policy` 回显、静态 UI 与路径穿越拒绝、
四模式路由回显与分页稳定性（仅 live-vector 阶段）、OOV 语义救援、显式布尔/短语约束、
八类 400 错误矩阵（含纯词法 profile）、down 阶段的 503/`vector_unavailable`/显式
fallback，以及 recovered 阶段无需重启服务的恢复。
每个检索请求都带 `--run-tag`（默认 `online-acceptance`），便于与服务 stderr 日志对账。

- 退出码 `0` 表示全部检查通过，`1` 表示至少一项失败；证据 JSON 里 `checks`/`events`/`failures`
  逐项可查，`scope` 明确写了"不是质量基准"。
- `--require-instance-id` 对每个响应校验唯一且一致的 `X-Search-Instance`，包括预期的
  400、404、503，并拒绝 HTTP 重定向。`--expect-instance-id` 自动启用同样的校验，并锁定到
  healthy 收据 `instance_binding.verified_id` 记录的实例。代理须把该响应头原样传到客户端。
  两个选项都省略时适用于尚未返回该响应头的服务，此时收据仅覆盖单次 HTTP 契约验收。
- 向量 profile 要求显式 `--expect-hnsw-ef`：先固定搜索 beam（示例为 256）再验收；
  省略 beam 的部署 `search.hnsw_ef` 回显为 `null`，没有可断言的值，无法给出 HNSW 搜索参数证据。
- 纯词法 profile 用 `--lexical-only`：脚本不再要求 collection/`hnsw_ef` 期望，改为核对
  dense/hybrid 的 503 与 adaptive 回退，并同样执行八类请求校验矩阵
  （`--lexical-only` 与 `--phase recovered` 互斥）。四模式分页只对**有活向量后端**的
  healthy/recovered 阶段适用；词法 profile 不假设目标查询一定满页命中，因此不做分页
  断言。该分支共 17 项检查，配置向量的 down 阶段仍是 10 项，healthy/recovered 仍是 23/27 项。
  这些回退预期假定默认 `rule` 策略：自定义策略若从不尝试向量，`adaptive` 不会走
  `vector_unavailable` 回退，需按该策略语义单独解释。
- 脚本只读、可重复执行；它不启动/停止服务或后端，也不写入数据库。阶段切换由操作者完成。
- 它证明的是服务接线与失败语义，不证明检索质量、HNSW 性能或容量。

回归测试（无需真实服务/Qdrant，假 HTTP 服务覆盖身份不匹配、后端不可用、恢复、词法 profile、
错误矩阵缺失校验、路径泄漏与端点完全不可达）：

```bash
python3 -B -m unittest tests.test_online_acceptance -v
```

探针的每条检查只靠 `assert` 判定，所以在启用优化（`python -O` / `PYTHONOPTIMIZE`）的
解释器下会直接拒绝运行，而不是把失败的核对静默报成 PASS。

新服务的回归入口：`python3 tests/run.py` 是默认核心 runner（已收录函数式
`tests/test_router_seam.py`）；`unittest` 类形式的新增用例用发现命令运行：

```bash
python3 -B -m unittest discover -s tests -p 'test_*.py'
# 或只跑服务相关子集：
python3 -B -m unittest tests.test_serving_config tests.test_observability \
  tests.test_online_acceptance
```

## 9. 故障排查

| 症状 | 先查 | 说明 |
|---|---|---|
| 启动即退出：dataset / corpus / n_docs 不一致 | `offline.validate`；配置 `dataset`；`/index/info` 的历史值 | 双索引不同版本不可放行，必须用同版产物 |
| 启动失败：collection 缺少构建元数据 / 点数不符 / 维度或距离不符 | Qdrant 里该 collection 的实际配置 | 旧 collection 必须用 1.16+ writer 重建，不能补填 manifest 冒充 |
| 模型加载失败 | 本地模型目录与 SHA256 清单、`HF_HUB_OFFLINE` | 缺依赖或清单不符会明确拒绝，不会退回未固定 revision |
| dense/hybrid 返回 503 `vector_unavailable` | Qdrant 是否在跑、`vector.url`、collection 是否被删/重建 | 词法仍可用；adaptive 会 `fallback=true`，评测需单列，不能当语义结果 |
| adaptive 大量 `fallback=true` | 同一场景直接请求 dense 看 503 | 回退是显式降级而非质量结论；`--fail-on-fallback` 可拒绝降级跑批 |
| `/health` 正常但语义全挂 | `/index/info` 的 `vector_ready` 与后端进程 | liveness ≠ readiness；`vector_ready` 只是启动配置结论 |
| 请求 4xx（请求体超限）或读取超时 | `max_body_bytes`、`request_timeout_s` | 超限请求在解析前被拒绝，不进入检索；大 `k` 仍受 `candidate_k` 约束 |
| 端口被占用 | `ss -ltnp` | 单进程单端口；不要杀掉不属于自己的进程 |
| 日志里 `run_tag` 为空 | 请求是否带 `run_tag` | 单模式评测通过 `--request-tag` 启用关联标签 |

## 10. 规模与证据边界

- 新闻 profile：20,000 篇；在本仓库复核环境里启动到 `/health` 可用约数秒（语料指纹 + 模型加载）。
  这是单次工程观察，不是 SLO。
- 官方 profile：8,096,668 篇；启动需要多分钟（大 manifest/collection 核对），Qdrant 实例需要
  数十 GiB 级内存（复核时按约 26 GiB 可用内存准入）。不要把 20k 的启动时间/内存外推到 8M。
- 单查询阶段耗时（编码、向量、词法）只是接线冒烟，量级随硬件和后端状态变化；本仓库没有
  吞吐、并发或 p95 的正式结论，也没有把 `exact=false` 当成 HNSW 性能证据。
- 质量结论（nDCG/Recall 等）属于 RFC-0002/0004 的评测范围；adaptive 回退数据必须单列。
- 服务是单进程 `ThreadingHTTPServer`，没有副本、选主或优雅再平衡。

## 11. 安全边界

- 默认只监听 `127.0.0.1`；配置非回环 `host` 会打印明确警告。服务**没有认证、没有 TLS、
  没有速率限制**，也不要通过它暴露内部文件或管理操作。
- 不要把配置写成公网地址；`vector.url` 带 userinfo/query/fragment 会被拒绝且不回显。
- 配置与日志都不应包含凭据；证据 JSON 只包含调用方自己给的查询与公开响应。
- 需要对外提供服务时，自行在前面加认证、TLS 和限流，并明确这是非本仓库承诺的部署形态。

## 12. 已知限制

- Router **算法**（阈值、统计量、策略训练）由 RFC-0001 负责；本文只接线 `policy.factory`、
  `policy_version` 回显与可选的 `requires_statistics` 统计钩子（耗时计入 `router` 阶段）。
- 没有 expanded-term 高亮、没有来源 URL/侧边栏字段（需要 0002/0004 的文档字段决策）。
- 没有快照 SHA 的每次启动复核（见第 5 节），也没有后端周期健康探测。
- 不提供 daemon/PID/systemd 单元；服务以前台方式运行，`Ctrl-C` 停止。
- 本文与示例配置描述的是 interface 冻结后的服务；若实现回显字段与示例不一致，
  以 `contracts/RETRIEVAL.md` 第 4 节定义为准并更新示例。
