# RFC-0004：个人已有离线贡献（部分交付）

关联：<https://github.com/ShwStone/NLP-SearchEngine/issues/6>。
**本提交只交付个人已有成果，不认领整个 RFC-0004，也不声明完成其全部验收。**
基于 `dba882b079f481cf03d86aef4b51d4566d63a28a`，从原研究提交 `96e4f49`
及已保存的离线开发提交 `8e22623` 提取。其他研究成果保留在原分支。

## 本次贡献

1. 独立 SQLite 暂存的流式词法构建，替代把完整语料及所有 postings 作为 Python 对象保存在内存。
   生产构建不再调用 conformance oracle，继续输出原 A1 JSONL，保留确定性、位置及 BM25 参数。
   在同文件系统临时目录构建，完成后 rename 发布；拒绝覆盖已有非空目录。
   这是发布可见性的保证，不是断电后的 fsync 耐久性保证。
2. `offline.prepare` 对冻结的标准 JSONL 做显式预检，生成规范字段、稳定内部 ID、映射、排除记录及校验清单。
   默认遇错终止；只有显式指定 `--invalid skip` 才排除坏记录，重复外部 ID 始终失败。
   不读取 queries/qrels，不采样、不替 RFC-0002 选择版本或决定正式语料内容。
   如预检发现冻结语料异常，应先与数据负责人确认，不能静默改变评测对象。
3. `DocumentSource` / `DiskDocuments` 为文档编码提供窄接口，只保留每行字节位置并按需读取正文。
   双索引入口不再为了文档编码加载完整词法 postings 和 token offsets。
4. 固定 revision 的 `bge-small-zh-v1.5` 文档/查询编码器，CLS pooling、右侧截断及 L2 归一化，
   验证本地模型六个文件的固定 SHA256；提供离线模型/Qdrant 下载和向量构建 CLI。
   这是可注入的共享适配器，未改变上游在线服务默认编码器及其 CLI。
5. Qdrant writer 按批消费向量生成器并校验数量、唯一 ID、维度与归一化，支持 HNSW 参数。
   已存在 collection 绝不覆盖；仅明确 404 时创建；失败尝试清理本次创建的未发布 collection，保护旧库。
6. 可选 `CompactIndex` / `PackedIndex` 派生产物：校验来源 A1，保留位置，提供只读随机访问、缓存预算及并发合并加载。
   原 A1 和默认在线 reader 不变。compact-v1 是新增可选派生格式，后续正式采用需契约/规模兼容性评审。

“流式”没有承诺常数内存：长度统计、唯一 ID、文档字节位置仍随文档数增长；
PackedIndex 预算只覆盖保留块及键，不等于进程 RSS 上限，也不包括并发请求和 SQLite。
只移植 reader 不能继承原研究分支依赖在线 scorer/highlight 快路径的性能数字。

## 使用与验证

```bash
python3 tests/run.py
python3 -m unittest discover -s evaluation
python3 -m offline.prepare --corpus fixtures/tiny_corpus.jsonl --out /tmp/offline-prepared
python3 -m offline.build --corpus /tmp/offline-prepared/corpus.jsonl \
  --out /tmp/offline-lexical --dataset tiny
python3 -m contracts.conformance /tmp/offline-lexical /tmp/offline-prepared/corpus.jsonl
python3 -m offline.compact --source /tmp/offline-lexical --out /tmp/offline-compact
```

可选真实依赖（torch 按机器另装；只运行标准库测试无需下载模型）：

```bash
pip install -r requirements-offline.txt
python3 scripts/bootstrap_offline.py \
  --model-dir .runtime/models/bge-small-zh-v1.5 --qdrant-dir .runtime/qdrant
# 此脚本自己启动隔离 Qdrant，在临时目录构建/上传恢复快照，最后停止进程并清理临时数据。
python3 scripts/smoke_offline.py \
  --qdrant-binary .runtime/qdrant/qdrant \
  --model-path .runtime/models/bge-small-zh-v1.5 --device cpu
# 对自己已启动的 Qdrant 构建新的 collection，拒绝覆盖已有同名库：
python3 scripts/build_vectors.py --index /tmp/offline-lexical --out /tmp/offline-vectors \
  --collection unique_new_collection --url http://127.0.0.1:6333 \
  --model-path .runtime/models/bge-small-zh-v1.5
```

本次拆分分支的验证结果：

- 核心 53/53；评测模块 22/22。
- Compact/Packed 均通过同一 conformance oracle，与上游 DiskIndex 的 BM25、TFIDF、RM3、布尔、短语、分页、高亮结果一致。
- 覆盖坏语料不发布、重复 ID、确定性文件、checksum 拒绝、字节缓存淘汰、并发单次加载、流式消费及失败清理。
- RTX4090、固定 BGE、Qdrant 1.16.0：tiny fixture 10 篇真实编码/建库，删除后由快照恢复，ID 和精确数量一致，耗时约 3.74 秒。
  此为小规模真实适配验证，不等于正式数据交付或训练效果提升。

## 明确剩余工作

本提交未交付 RFC-0002 最终冻结的自爬 20k / DuRetrieval 双索引或 8M 规模验证，
也未完成正式发布目录切换、恢复性能报告和全套 RFC 验收。没有变更 `online/` 或前端。
writer 的失败清理是尽力而为，后端不可达时可能残留未发布 collection；
writer 返回后若上层发布校验失败，也未新增跨阶段事务回滚。
模型/大数据/快照不进入 Git，实际交付仍需负责人按最终数据清单完成。
