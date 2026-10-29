# 交付 TODO（2026-10-26）

这个清单把已验证事实、可继续执行的工作和受外部资源阻塞的工作分开。开题报告中的数量和方向只作为团队假设；实际资源、脚本输出和标注状态以 `evaluation/RESOURCES.md` 及本清单为准。

## DONE

- [x] BM25、TF-IDF、倒排/磁盘索引、布尔/短语/位置过滤、稳定 Top-K、高亮、分页和 HTTP v2 契约。
- [x] Router、RRF、RM3、lexical/dense/hybrid/adaptive 四种模式、向量失败时的显式 503/fallback telemetry。
- [x] 官方 RocketQA DuReader-Retrieval baseline archive 的哈希、分片、ID、dev 查询和 qrels 校验；确认它不是课程报告所写的 100001/2000/9839 快照，也没有官方 test labels。
- [x] 集群上固定 prefix20k、真实 SentenceTransformer revision 和 qdrant-client local exact backend 的四模式工程冒烟及 query-level paired bootstrap；结果只代表 19 条有标注查询，不能外推为全量或 HNSW 结论。
- [x] `evaluation.evaluate` 的完整 MAP@100 召回深度、路由 telemetry 与 `evaluation.compare` 的确定性 paired bootstrap。
- [x] 公共演示站点 [search.shwstone.cn](https://search.shwstone.cn/) 的 lexical 搜索和 adaptive 缺向量时的明确 fallback；修复前端失败后清除旧路由/耗时观测的残留问题。
- [x] 新增有边界的 `evaluation.crawl_web`：HTTP(S) only、robots-aware、同域、深度/文档数/响应大小/速率限制，以及人工标注后生成 queries/qrels 的校验入口。
- [x] 执行一次可审计的公开站点抓取：gov.cn、30 篇文档、10 个查询候选池、97 条 provisional 建议；收据、文件哈希和审阅模板均保存在版本库外。
- [x] 复跑同域 depth=2 的扩大抓取并记录实际结果：393 篇有效文档、corpus SHA256 `757e93bfc08f800827cff925864155032bbfe5d4be6cdcc5009b7b8996419531`、10 查询/100 条 provisional 候选；未把请求上限误报为实际文档数。
- [x] 修复 HTTP 分块传输中断并完成 depth=3 重试：799 篇有效文档、corpus SHA256 `1f209cad5da40808b14f6249b886308e977f708dd222f2c9eaf984f3d5b9f258`；仍只有未标注模板，未形成正式 qrels。
- [x] 完成 depth=4 有界重试：1,029 篇有效文档、corpus SHA256 `463ac4eff4fdc22047da362fdc22d47f0be6bf09c1587c3c8ba318b9e8164ff3`；生成 10 查询/100 条 provisional 候选和人工复核表，仍未形成正式 qrels。
- [x] 用 SQLite 流式构建器和 compact reader 对 depth=4 的 1,029 篇语料完成规模路径冒烟；stream/compact 产物保存在版本库外，不能外推为 8M 验收。
- [x] 从公开 sitemap 增加 108 篇页面并与 depth=4 语料按 URL 合并为 1,137 篇；在用户明确授权后完成 100 条候选的 assistant 语义复核，生成 10 个 queries、100 行正式 assistant qrels，并完成 BM25、RM3、adaptive fallback 和 paired bootstrap 工程评测；结果不等同独立人工 gold 或 20k/8M 成绩。
- [x] 以相同 robots、同域、深度和速率约束对新华网完成 100 篇跨站点抓取试验；输出仅为未标注候选模板（corpus SHA256 `eafb1dee8ddc36a3d5c778883d8e379618d073014d742359623cb321db8394f5`），未将其误作正式 qrels 或质量结果。

## NEXT（需要继续做，但不应伪造结果）

- [ ] 用课程实际提供的、可追溯的 100001/2000/9839 版本执行 `prepare_dataset`，固定 dev/test ID 清单、哈希和数据分割；若仍拿不到，报告中保留阻塞证据。
- [ ] 在 dev 上冻结 Router 阈值、candidate pool、RRF `c` 和模型配置，再在独立 test 上运行 lexical、dense、固定 hybrid、adaptive 四组；逐查询保存 TREC run、路由原因分布、向量触发/成功/fallback、平均/p50/p95 延迟。
- [ ] 用 `evaluation.compare` 至少完成 fixed-hybrid vs adaptive 的 paired bootstrap，并在报告中给出 query-level 差值和区间；不要在 test 上挑阈值。
- [~] 选择允许抓取的公开站点并运行 `crawl_web crawl` 已完成 gov.cn 的 30/393/799/1,029 篇候选批次、sitemap 合并后的 1,137 篇，以及新华网 100 篇跨站点候选；assistant qrels 已按授权生成，但仍需找到来源扩展到 20k 有效文档。若要独立人工 gold labels，必须另行复核，不能把 assistant labels 改称人工真值。
- [ ] 将经过人工确认的自爬 20k corpus、queries、qrels 和 manifest 接入统一双索引构建；保留来源、ID、计数和 SHA256。
- [x] 纳入 SQLite 支持的 `offline.stream_build`、compact/packed reader、DiskDocuments 和临时 SQLite ID 校验，并以 tiny corpus conformance 验证；1,029 文档的 stream/compact 已完成，但全量 packed conformance 在本地超过 20 分钟无收据后停止，仍需进行有界验证、8M 预检、全量构建、峰值 RSS/磁盘/吞吐和快照收据。
- [ ] 若要声称 HNSW 或 Qdrant 服务性能，必须另行启动可复现的 HTTP Qdrant 服务并记录版本、索引参数、快照哈希和失败率；local exact smoke 不替代这项验收。
- [ ] 公共站点当前展示的是 tiny lexical 部署；向量模式必须在后端真正提供固定模型和向量库后再宣称可用。部署变更后重新做 `/health`、`/index/info`、四模式和错误路径检查。

## BLOCKED / 证据边界

- 官方公开 baseline 只有 dev 标注，课程目标快照和 test labels 尚未被公开资源证明存在；不能把 dev 或 prefix smoke 改名为 test/full-scale。
- 本地电脑没有 NVIDIA GPU；真实 embedding/Qdrant 的集群运行产物不提交仓库，且不代表全量语义质量。
- 网络爬取只负责构造候选 corpus；本次 qrels 的标签来源明确为 assistant 语义复核，用户已授权将其作为正式工程 qrels，但它不是独立人工 gold labels。脚本不会从网页标题自动制造真值。
- 集群交互式任务表单已确认存在 A800 配额，但当前提交动作被“最长运行时间 <= 0日6时0分”的表单校验拦截，任务列表仍显示 0 个交互式运行应用。改查同一账号的训练任务入口时，自有 RFC-0004 工作空间和仓库目录可见，但程序入口文件级联选择在读取私有数据时返回 504，未创建任务；在平台校验或入口文件服务修复前，不记录 8M/HNSW 任务已启动。

## 固定验证入口

```bash
python tests/run.py
python -m unittest discover -s evaluation -p 'test_*.py'
node --test tests/test_web.cjs
python -m compileall .
git diff --check
```
