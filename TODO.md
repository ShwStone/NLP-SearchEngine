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

## NEXT（需要继续做，但不应伪造结果）

- [ ] 用课程实际提供的、可追溯的 100001/2000/9839 版本执行 `prepare_dataset`，固定 dev/test ID 清单、哈希和数据分割；若仍拿不到，报告中保留阻塞证据。
- [ ] 在 dev 上冻结 Router 阈值、candidate pool、RRF `c` 和模型配置，再在独立 test 上运行 lexical、dense、固定 hybrid、adaptive 四组；逐查询保存 TREC run、路由原因分布、向量触发/成功/fallback、平均/p50/p95 延迟。
- [ ] 用 `evaluation.compare` 至少完成 fixed-hybrid vs adaptive 的 paired bootstrap，并在报告中给出 query-level 差值和区间；不要在 test 上挑阈值。
- [ ] 选择允许抓取的公开站点，运行 `crawl_web crawl` 生成未标注 corpus；由组员人工填写 query/relevance，再运行 `crawl_web finalize` 生成 qrels。未完成人工标注前不得把模板当评测集。
- [ ] 若要声称 HNSW 或 Qdrant 服务性能，必须另行启动可复现的 HTTP Qdrant 服务并记录版本、索引参数、快照哈希和失败率；local exact smoke 不替代这项验收。
- [ ] 公共站点当前展示的是 tiny lexical 部署；向量模式必须在后端真正提供固定模型和向量库后再宣称可用。部署变更后重新做 `/health`、`/index/info`、四模式和错误路径检查。

## BLOCKED / 证据边界

- 官方公开 baseline 只有 dev 标注，课程目标快照和 test labels 尚未被公开资源证明存在；不能把 dev 或 prefix smoke 改名为 test/full-scale。
- 本地电脑没有 NVIDIA GPU；真实 embedding/Qdrant 的集群运行产物不提交仓库，且不代表全量语义质量。
- 网络爬取只负责构造候选 corpus；相关性 qrels 必须有人审阅。脚本不会从网页标题自动制造“真值”。

## 固定验证入口

```bash
python tests/run.py
python -m unittest discover -s evaluation -p 'test_*.py'
node --test tests/test_web.cjs
python -m compileall .
git diff --check
```
