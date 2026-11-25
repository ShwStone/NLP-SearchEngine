# 基于 BM25 与向量检索的中文自适应混合搜索引擎 — 项目结题报告

> **完整论文格式 PDF**：[`paper/report.pdf`](paper/report.pdf)（Typst 源码：[`paper/report.typ`](paper/report.typ)）
>
> **在线公网检索网站**：[https://search.shwstone.cn](https://search.shwstone.cn)（三套数据源全部支持四模式实时检索与并行对比）
>
> **团队成员**：石皓文 · 张恩瑞 · 许程 · 杜鹏宇

---

## 1. 项目概览与交付达成

本项目面向 NLP 课程大作业，完整交付了从底层双索引数据库构建到算法研究与公网产品部署的端到端中文搜索引擎：

| 开题核心目标 | 完成情况 | 关键指标 / 交付物 |
| --- | --- | --- |
| **建立多规模数据库** | ✅ 已完成 | 8,096,668 篇 DuReader 全集（39 GB 手写 Compact 词法 + 18.5 GB BGE Qdrant HNSW）、100k DuReader Mini（双索引对齐）、20k 自爬去重新华网新闻库（双索引对齐） |
| **搭建公网可用网站** | ✅ 已上线 | [search.shwstone.cn](https://search.shwstone.cn)：阿里云 HTTPS 静态前端 + 限流反代 + `systemd` 托管 SSH 反向隧道 + 本机多数据集网关，三套语料全四模式在线，支持四模式同题并行对比与 BM25/TF-IDF/RM3 切换 |
| **评测多种检索方法** | ✅ 已完成 | 统一 HTTP 评测 BM25、TF-IDF、Dense（BGE HNSW）、Fixed Hybrid（RRF）、Adaptive；与 `BM25S`（0 误差）、`rank-bm25`（$2.67\times 10^{-15}$）、`FAISS IndexFlatIP`（Top-10 100% 集合一致）逐分对齐 |
| **探索自定义 Router** | ✅ 已完成 | 四代完整演进：① 规则路由 `rules-v1` → ② 监督线性收益路由 → ③ 嵌套 Pre-BGE / CE 路由与质量护栏（PR #9/#14） → ④ 师生蒸馏与文档级损益分解 CE 预算路由（PR #16，同预算 nDCG@10 **0.4372**，超基线 **+0.0130**） |
| **系统消融实验** | ✅ 已完成 | BM25 vs TF-IDF、RM3 伪相关反馈、`Qwen2.5-1.5B-Instruct` SLM 查询扩写（100k 上全模式显著正向增益）、RRF $c$ / 候选深度 / 非对称加权 RRF / Min-Max 融合、路由阈值权衡曲线与 8M 延迟/HNSW/并发调优 |
| **自爬新闻库实测** | ✅ 已完成 | 28,914 次同域合规抓取 → 精确 + SimHash64 去重冻结 20,000 篇新闻；200 条自然语言问句（50 dev / 150 test）；5 系统 Top-20 池化 8,281 对三级判定；完整四模式 HTTP 评测与消融实测 |

---

## 2. 核心实验汇总

### 2.1 三套语料四模式主实验（统一配置：BM25 `k1=1.2, b=0.75`，`BGE-small-zh-v1.5`，`candidate_k=200`，RRF `c=60`）

| 评测集 | 模式 | nDCG@10 | Recall@10 | MRR@10 | MAP@100 | 向量触发率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **自爬新闻 20k** (Test, n=150) | Lexical (BM25) | **0.9168** | **0.8897** | 0.9669 | **0.8602** | 0.0% |
| | Dense (BGE HNSW) | 0.7513 | 0.7363 | 0.8455 | 0.6567 | 100.0% |
| | Fixed Hybrid (RRF) | 0.8511 | 0.8286 | 0.9242 | 0.7811 | 100.0% |
| | Adaptive (`rules-v1`) | 0.9082 | 0.8731 | **0.9756** | 0.8514 | 21.3% (32/150) |
| **DuReader 100k** (Dev, n=1,020) | Lexical (BM25 / BM25S) | 0.1340 | 0.1571 | 0.2173 | 0.1318 | 0.0% |
| | Dense (BGE / FAISS) | **0.2648** | **0.2912** | **0.4098** | **0.2478** | 99.9% |
| | Fixed Hybrid (RRF) | 0.2284 | 0.2658 | 0.3476 | 0.2079 | 100.0% |
| | Adaptive (`rules-v1`) | 0.2050 | 0.2410 | 0.3170 | 0.1866 | 73.3% (748/1020) |
| **DuReader 8M** (Holdout, n=808) | Lexical (BM25) | 0.1265 | 0.1529 | 0.1956 | 0.0892 | 0.0% |
| | Dense (BGE HNSW) | **0.2560** | **0.2856** | **0.3855** | **0.1947** | 99.8% |
| | Fixed Hybrid (RRF) | 0.2097 | 0.2448 | 0.3211 | 0.1587 | 100.0% |
| | Adaptive (`rules-v1`) | 0.1876 | 0.2175 | 0.2909 | 0.1408 | 73.6% (595/808) |

> **关键发现**：长篇实体新闻（`news20k`）由词法主导，`rules-v1` 相对固定 Hybrid 显著提升 **+0.0572**（95% CI `[+0.0293, +0.0888]`）；而短句网页问答（DuReader）由语义主导，Dense 领先 BM25 两倍以上。

### 2.2 词法打分器、RM3 与 SLM（`Qwen2.5-1.5B-Instruct`）查询扩写消融

详细原始数据见 [`docs/experiments/ablations-summary.json`](docs/experiments/ablations-summary.json)：

- **BM25 vs TF-IDF**：在文档长度跨度极大的新闻 20k 上，BM25 以 **0.9168 vs 0.7539**（+0.1629，95% CI `[+0.127, +0.201]`）大幅领先；在长度均一的 DuReader 100k 上二者统计持平（0.1340 vs 0.1346）。
- **RM3 伪相关反馈**：零神经网络开销，新闻 20k 上小幅提升至 **0.9175**（Recall@10 `0.8897 → 0.8977`）。
- **SLM 查询扩写（RTX 3090 FP16，p50 139.3 ms）**：在短查询 DuReader 100k 上对全线检索模式均产生统计显著正向收益——**BM25+SLM `0.1517`（+0.0178, +13.3%）**、**Dense+SLM `0.2777`（+0.0129, Recall@10 +10.2%）**、**Hybrid+SLM `0.2416`（+0.0132）**、**Adaptive+SLM `0.2196`（+0.0147，95% CI `[+0.0068, +0.0228]`）**；而在原本实体完备的长新闻问句上增益不显著。
- **非对称融合消融**：新闻 20k 上 Min-Max 插值（$\alpha_{\text{dense}}=0.25$）达 **nDCG@10 = 0.9284 / MRR@10 = 0.9878**，同时超越纯 BM25 与纯 Dense；DuReader 100k 上加权 RRF（$w_V=5.0$）达 **0.2609**（较等权 RRF +0.0325）。

### 2.3 四代自定义 Router 与 CE 预算分配（PR #9 / #14 / #16）

在 DuReader 100k 730 条报告面板（固定预算 21,676 CE pairs，相对固定 CE-33 节约 10.02%）上，PR #16 提出的**基于师生蒸馏与文档级损益分解的 CE 预算路由**（发现 66.5% 的相关文档损失来自前十降位）取得如下同预算对照结果（详述见 [`docs/rfc0005-factor-router/README.md`](docs/rfc0005-factor-router/README.md)）：

| 路由策略（21,676 CE pairs，均 29.69/条） | 100k 训练 nDCG@10 | 8M 训练 nDCG@10 | 8M 本方法差值 95% CI |
| --- | ---: | ---: | --- |
| 同预算随机分配期望 | 0.4242 | 0.4242 | +0.0131 |
| PR #9 ExtraTrees（原 7 项前置） | 0.4260 | 0.4223 | +0.0149 `[+0.0079, +0.0227]` |
| PR #9 ExtraTrees（适配 96 项前置） | 0.4276 | 0.4242 | +0.0130 `[+0.0052, +0.0211]` |
| PR #9 原 26 项探针（含 CE-28 事后参考） | 0.4325 | 0.4270 | +0.0103 `[+0.0021, +0.0187]` |
| **净收益分解 CE Router（PR #16 终版）** | **0.4305** | **0.4372** | 基准（Recall 0.4594 / MRR 0.6076） |
| 固定 CE-33（24,090 pairs, +11.1% 成本） | 0.4512 | 0.4512 | −0.0140 `[−0.0234, −0.0045]` |
| 同预算事后 Oracle 上界 | 0.4689 | 0.4689 | — |

### 2.4 8M 在线性能调优实测

详述见 [`docs/PERFORMANCE_TUNING.md`](docs/PERFORMANCE_TUNING.md)：
- **8M 词法 BM25（64 条查询）**：稀疏 NumPy + Numba JIT 解码 + 512 MiB 有界缓存 + 大并集双阈值累加 → 服务端 **p50 `685 → 43 ms`（15.93×）**、**p95 `2369 → 167.7 ms`（14.13×）**，64/64 组 Top-10 排名与六位小数分数完全一致。
- **8M 语义 Dense & 并发**：HNSW `ef=64`（96.4% Top-10 重合率）+ 展示页惰性外部 ID 校验 + Qdrant/Torch 双 8 线程 → 单路热态 **p50 15.0 ms / p95 19.7 ms**，2 路并发 `225.5 → 23.5 ms`（9.6×），4 路并发 `416.5 → 48.5 ms`（8.6×）。

---

## 3. 文档与复现索引

- **学术论文格式完整报告**：[`paper/report.pdf`](paper/report.pdf) / [`paper/report.typ`](paper/report.typ)
- **多数据集公网部署文档**：[`docs/MULTI_DATASET_SERVING.md`](docs/MULTI_DATASET_SERVING.md)
- **8M 在线调优记录**：[`docs/PERFORMANCE_TUNING.md`](docs/PERFORMANCE_TUNING.md)
- **CE 净收益分解路由专题（PR #16）**：[`docs/rfc0005-factor-router/README.md`](docs/rfc0005-factor-router/README.md)
- **两动作 Router 对照**：[`docs/contributions/rfc0001-router-budget.md`](docs/contributions/rfc0001-router-budget.md)
- **自爬新闻 20k 数据卡**：[`docs/contributions/rfc0002-news20k-dataset-card.md`](docs/contributions/rfc0002-news20k-dataset-card.md)
- **资源与哈希清单**：[`evaluation/RESOURCES.md`](evaluation/RESOURCES.md)
