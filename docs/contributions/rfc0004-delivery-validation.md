# RFC-0004：离线产物交付与校验补充

本文件记录本分支补充的 RFC-0004 部分。它与 PR #7 的 streaming/compact/packed
个人成果互补，不复制其实现，也不宣称单独完成 RFC-0004 的 20k 自爬或标准集全量验收。

## 已落地的交付语义

1. `offline.build` 现在先在目标目录同文件系统的临时目录中构建，完成 A1 文件和
   manifest 后再原子改名；已有非空目录不会被覆盖。JSON 错误、字段类型错误和重复
   外部 ID 在写出前拒绝；空标题或空正文仍按既有 A1 兼容语义保留。
2. `offline.build_hybrid` 在新构建失败时删除本次创建的目录；已存在的发布目录不被
   这个入口覆盖。成功构建会生成 `build_provenance.json` 与 `build_report.json`。
3. provenance 记录数据文件名、字节数、SHA256、ordered corpus fingerprint、文档数、
   BM25 参数、完整 embedding 配置、collection、backend、index params 和快照摘要，
   不写入机器绝对路径、凭据或 qrels。
4. build report 记录各阶段耗时、每个产物的大小和 SHA256。峰值 RAM/GPU 若没有作业
   监控证据就保持为空，不用估算值冒充实测。
5. Qdrant writer 逐条消费记录、按 batch upsert，检查内部 docid 的连续顺序、外部 ID
   唯一性、向量契约和 exact count。创建的新 collection 在后续失败时尝试删除；已有
   collection 永不覆盖。
6. 快照下载写入 `.part` 文件，完整下载后才改名；失败不会把半截文件当作快照。
7. `python -m offline.validate --build-dir <dir>` 要求 A1 四个文件都有 SHA256，核对
   BM25/vector 的文档数和 corpus fingerprint，验证快照哈希、provenance 和 report
   中的完整必需产物集合、身份字段和产物哈希。它只依赖 offline/contracts，不导入 online。
8. writer 返回成功后若本地 manifest 发布失败，Qdrant writer 通过 `rollback(manifest)`
   尝试删除本次 collection；原始发布异常优先保留。

## 尚未宣称的能力

- 当前旧 v1 builder 仍会把 conformance oracle 放入内存；PR #7 的 SQLite/compact 方案
  是独立的规模路线，合并前需解决接口兼容、峰值内存和完整回滚证据。
- `offline.validate` 能验证已导出的快照文件完整性和本地双索引身份，但不能把没有
  干净 Qdrant 实例恢复日志的构建伪装成“已完成恢复验收”。真实恢复仍需在隔离实例中
  执行并记录 Qdrant/server/client 版本、collection metadata、exact count、ID 覆盖和
  corpus fingerprint。
- 失败后的默认语义是使用新的版本化 collection 全量重建；没有实现跨 corpus、模型或
  embedding 配置的断点续写。任何 retry/resume 都必须先通过同一 provenance 校验。
- `memory-exact` 和 SDK double 只能验证契约，不能替代 HNSW、Qdrant 服务性能或语义
  质量实验。自爬 20k 和批准的标准数据版本必须分别记录，不能用 prefix 冒充全量。

## 运行与验收

```bash
python tests/run.py
python -m unittest discover -s evaluation -p 'test_*.py'
python -m offline.validate --build-dir /path/to/released-dual-index
```

所有数据、模型、Qdrant collection、快照和生成 run 均保持在版本库之外。
