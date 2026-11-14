# 交付 TODO

## DONE

- 搜索引擎的核心契约已有回归覆盖，包括词法索引、布尔与位置过滤、稳定分页、HTTP API、Router/RRF/RM3，以及向量失败时的错误和 adaptive 回退。
- **自爬语料**。清洗去重后冻结 20,000 篇，dev/test 为 50/150 个查询。8,281 条池化判定含 200 条已绑定身份的种子判定和 8,081 条 assistant 全文复核，来源随数据记录。A1、compact、真实 BGE 向量和 Qdrant 快照恢复已验收。
- **自爬四模式**。150 条测试查询经公开 `/search` 契约各执行一次 lexical、dense、hybrid、adaptive，共 600 次请求。TREC、路由、延迟和配对 bootstrap 已核验，详见 [REPORT.md](REPORT.md) 与 [资源记录](evaluation/RESOURCES.md)。
- **自爬数据复验与维护**。[新华网 20k 数据卡](docs/contributions/rfc0002-news20k-dataset-card.md)记录冻结数据、划分隔离、AI 判定来源和排名覆盖。审计和撤回预演可在授权副本上只读执行。正文发布权限及公开 20k 部署仍待确认。
- **官方数据**。核验公开 RocketQA DuReader-Retrieval 资源，冻结 8,096,668 篇、2,000 条 official-dev 查询、9,839 条 qrels。内部划分 train/dev/holdout=172/1020/808，holdout 在正式质量评测前冻结。`prepare_dataset --queries-only` 已在 Windows 和固定原生运行时逐字节复建六份划分文件。
- **官方全量词法**。十片合并原生终态 PASS，8,096,668 篇、8,578,589 个词项、13 个 A1/compact 文件、有序语料指纹及正常 `DiskIndex` 读取验收通过。原生收据链已独立复核；恢复耗时和历史失败边界见资源记录。
- **官方全量向量**。固定 BGE 的 162 个向量块覆盖全部文档。Qdrant HNSW 的 8,096,668 点快照已在两个独立空实例恢复。16 条固定 ANN 查询的 Top-200 与原实例一致，exact 同分边界另经全量扫描校验。收据和资源边界见 [资源记录](evaluation/RESOURCES.md)。
- **离线交付工具**。A1 和 compact 构建支持检查点，分片合并有顺序与哈希校验，`offline.validate` 可在原位核验单目录双索引或独立发布的 compact 与 Qdrant 快照。真实 Qdrant 上三文档检查通过，覆盖快照随整包发布、哈希和混合查询。失败后的清理与重试另有回归。
- **成熟 BM25 对照**。固定 rank-bm25 0.2.2，对齐 analyzer、公式与参数后验证 16 条有效查询的分数及排名，并保留默认 IDF 的差异；复现命令和收据见资源记录。当前源码在原生 Python 3.10.14 / Unicode13 上的 153 项专项 unittest 已通过。

- **全量截断统计**。固定正式 tokenizer，8,096,668 篇中 620,314 篇（7.6613%）超过 512 tokens；全量字节身份和 162 个分块的聚合已独立核验。

- **官方全量四模式**。完整语料的 808 查询、4,054 qrels 已完成四臂 HTTP、TREC、路由/请求覆盖、逐臂延迟及 10,000 次配对 bootstrap，并在固定 Python3.10.14 / Unicode13 独立复算 PASS。Dense 两条预定义约束拒绝按空排名计入 808 分母，其余请求均成功、无异常或回退。Dense 四项质量最高；当前规则 Adaptive 相对 Hybrid 的四项区间均为负，保留结果。混合 epoch 延迟仅逐臂报告，见 [报告第 8 节](REPORT.md#8-官方全量-808-查询四模式评测2026-11-07-完成复核)。

## 当前团队任务（以 daf6e03 为基线）

- [x] **RFC-0004 离线交付**。全量词法、向量与评测证据已通过 [PR #8](https://github.com/ShwStone/NLP-SearchEngine/pull/8) 合入，关闭 [Issue #6](https://github.com/ShwStone/NLP-SearchEngine/issues/6)。
- [x] **RFC-0003 在线接线**。[PR #10](https://github.com/ShwStone/NLP-SearchEngine/pull/10) 合入启动身份、Router 接口、HTTP 请求保护、可观测性与 UI 身份面板，关闭 [Issue #5](https://github.com/ShwStone/NLP-SearchEngine/issues/5)。已有两套真实索引的[工程验收](docs/contributions/rfc0003-online-acceptance.md)。
- [x] **RFC-0002 共享集复建与词法参考**。重建 [DuReader 100k v1.0](https://huggingface.co/datasets/ShwStone/dureader-mini-100k/tree/v1.0)，逐文件复核发布哈希。手写 BM25 与 BM25S 在冻结的 1,020 条 dev 查询上逐排名一致，保存了两份深度 100 的 TREC run、输入校验和四项配对区间，详见[资源记录](evaluation/RESOURCES.md#dureader-100k-协作开发集词法对照)。
- [x] **RFC-0002 语义与规则 Router 基线**。DuReader 100k 的 1,020 条开发查询已完成 BM25、ANN Dense、固定 RRF、规则 Adaptive 及同源 NumPy、外部 FAISS CPU 精确 Dense，另有 1,152 次同服务交错计时。质量、请求与 TREC 逐条核验，保留配对区间及 Router 的负结果。见[报告第 9 节](REPORT.md#9-协作开发集与-router-对照)；808 条全量留出另列。
- [x] **RFC-0002 本地 20k 演示验收**。从已交接的快照恢复 20,000 个点，校验同一语料的压缩词法索引。真实浏览器走四模式，实测无效语法清除旧结果。HTTP 后端断开与恢复分别通过 10 项和 27 项检查。 数据卡记录固定开发问例、Router 两种分支、显式约束、失败路径与 HTTP 复验命令。
- [x] **RFC-0002 20k 路由日志对账**。50 条开发查询逐一运行四模式，200 条 HTTP 结果均匹配同一服务的日志。固定 RRF 调用向量 50 次，规则 Router 调用 9 次，无回退。单次顺序工程测量见[资源记录](evaluation/RESOURCES.md#新闻-20k-本地浏览器演示与恢复验收)。
- [x] **RFC-0002 配对 HTTP 复验入口**。冻结查询、执行顺序和服务身份，通过日志对账验收，导出同批排名以便独立复算质量。既有 1,152 次四模式交错计时与 2,040 次规则/线性收益对照沿用各自的原始计划和回执。见[评测手册](evaluation/README.md#paired-live-http-measurements)。
- [x] **RFC-0002 可选的盲审导入工具**。裁决包只包含分歧条目，导入核对准备回执、冻结计划和原始评分。见[可选人工复评](docs/contributions/rfc0002-news20k-dataset-card.md#可选的人工评分复评)。
- [x] **RFC-0002 冻结新闻语料去重复验**。核查最终 20,000 篇，按原正文规范化和 SimHash64 规则确认精确与近重复均为 0 对。独立核对冻结语料与压缩索引文档表。[语料与复验命令](docs/contributions/rfc0002-news20k-dataset-card.md#采集与文档)
- [x] **RFC-0002 原始响应复提取**。最终 20,000 篇已接受文章的原始字节哈希、标题、正文及清洗元数据全部通过回放核验。命令只读使用私有缓存。[数据卡](docs/contributions/rfc0002-news20k-dataset-card.md#采集与文档)
- [x] **RFC-0002 自爬 AI 复核与实测**。冻结 8,281 对 assistant 全文判定，核验种子和批次身份，重放 150 条留出查询的四份 HTTP 排名。自动复算四项指标、判定覆盖及配对区间。见[自动复评命令](docs/contributions/rfc0002-news20k-dataset-card.md#自动复评)。
- [ ] **RFC-0002 公开 20k 部署复验**。部署负责人切换获准发布的冻结双索引后，用[验收命令](docs/contributions/rfc0003-online-acceptance.md#公开部署切换后的发布身份复验)验证索引、实际编码器运行时、同一服务实例、四种检索模式和错误路径。
- [x] **RFC-0001 两动作对照**。完成规则与同预算随机对照、线性收益模型独立训练和等预算回放。固定训练阈值与规则在 100k 开发集另有 2,040 次真实 HTTP 对照，向量调用为 715 对 748，质量与耗时证据支持保留现有默认规则。详见 [Router 对照](docs/contributions/rfc0001-router-budget.md)。
- [x] **RFC-0001 Router 预算、消融与分组复验**。冻结检索排名，比较等向量调用预算、特征移除和按查询组重采样。保留规则、学习模型、随机期望及事后最优分配的证据。见[Router 预算报告](docs/contributions/rfc0001-router-budget.md#开发集向量调用预算曲线)。
- [x] **RFC-0001 双动作研究与域迁移分析**。[Issue #3](https://github.com/ShwStone/NLP-SearchEngine/issues/3) 的 BM25 与混合检索选择研究已有[文献矩阵](docs/rfc0001/LITERATURE.md)、[方案和选型](docs/rfc0001/PROPOSALS.md)及[冻结实验和失败分析](docs/rfc0001/RESULTS.md)。256 条 CMedQA 迁移查询的质量损失超过预设上限，保留现行默认策略。
- [x] **RFC-0002 ANN 参数敏感性**。在同一 100k 快照及开发划分上对照 `hnsw_ef=128/256`，记录一致指标、约束拒绝和延迟比较边界。见[ANN 对照](docs/contributions/rfc0002-ann-ef-sensitivity.md)。
- [x] **RFC-0002 评测与复评工具合并**。[PR #11](https://github.com/ShwStone/NLP-SearchEngine/pull/11) 已合入 `master`，包含 FAISS 对照和 [PR #12](https://github.com/ShwStone/NLP-SearchEngine/pull/12) 的新闻盲审、裁决和复评入口。已核对两个 PR 的 head 都是当前主干祖先。
- [ ] **RFC-0001 Router 合并评审**。基于最新 `master` 整理 [PR #9](https://github.com/ShwStone/NLP-SearchEngine/pull/9) 的代码、阈值与汇总证据，包括 `f6838a0` 的复核。研究权重和逐查询文件由授权证据目录提供。待作者和维护者评审。
- [x] **RFC-0001 可选 CE 接线及对照入口**。固定和学习门控共用冻结的混合检索候选，服务回显重排身份和实际打分数，配对评估校验同一服务实例与等重排预算。见[评测手册](evaluation/README.md#train-and-evaluate-a-ce40-gate)。
- [x] **RFC-0001 开发集 CE 质量与成本对照**。课程报告复用 PR #9 的 100k 与全量开发实验，保留固定深度、同量级 CE 预算、训练内排序选型、负结果及负载差异。详见 [CE 重排与逐查询成本分配](REPORT.md#10-ce-重排与逐查询成本分配)。
- [x] **RFC-0001 100k 等预算随机质量与 HTTP 成本对照**。独立复算 1,020 条实测排名、2,040 次配对 HTTP 计时及冻结的随机计划。学习门控和随机组各用 21,600 个 CE 对，四项质量区间均跨过零，延迟按两个服务批次分别报告。[开发对照](REPORT.md#同检索配置下的-100k-等预算开发对照)
- [x] **RFC-0001/0002 配对评测运行时核验**。新配对协议将查询编码器实现、实际设备、权重精度和可核验的模型清单摘要纳入服务身份。历史 journal 按原协议复验。CE 留出采集可用 `--fit-receipt` 将运行时、门控权重和检索配置绑定到训练回执，并另行记录真实硬件和负载条件。
- [ ] **RFC-0001 CE 独立留出复验**。在 PR #9 的 CE 开发集对照基础上，对词法索引、向量快照及冻结策略进行身份核验，再运行固定 CE 和等成本对照。既有全量 CE HTTP 延迟来自并发负载环境，隔离计时待按冻结策略复验。离线复算入口从封存 SQLite、原始 qrels 和拟合回执重建 TREC、配对区间与随机对照摘要，并核对封存哈希。
- [ ] **RFC-0001 CE‑33/40 成本对照复核**。[PR #9](https://github.com/ShwStone/NLP-SearchEngine/pull/9) 的最新阈值汇总复用了此前请求，原始请求与阈值选择回执仍待回读。新采集会校验两臂的候选顺序、BM25 特征及实际重排范围，并支持在读取 qrels 前停止和封存请求。复验入口已核对索引、模型身份和同一 Qdrant 恢复实例的见证格式，仍待全量原生运行。见[复核说明](docs/rfc0001/FULL_ROUTER_HTTP_QUALITY_GUARD.md)。
- [x] **Windows 回归验证**。修复 checkpoint 子进程退出后锁释放的时序断言、CLI 子进程编码、原生路径断言和 JSON 解码异常的测试稳定性。完整 Windows unittest 运行 211 项，跳过两项 POSIX 专用用例，其余通过。
- [x] **旧 PR 功能核对**。[PR #2](https://github.com/ShwStone/NLP-SearchEngine/pull/2) 的采集与界面改动已进入主干。通用 SentenceTransformer 的批次配置和 Qdrant 逐批消费与失败清理回归在 [PR #11](https://github.com/ShwStone/NLP-SearchEngine/pull/11)。联合候选复用 [PR #7](https://github.com/ShwStone/NLP-SearchEngine/pull/7) 的[三种磁盘读器端到端对照](tests/test_scale_support.py)，并覆盖 PackedIndex 并发、缓存和失败重试。
- [x] **旧 PR 处置**。维护者已关闭过期的 PR #2 和 #7，关闭留言保留了对应贡献与测试来源。旧爬虫和索引差异会撤回主干已有加固，当前分支保留主干实现及新增回归。

## 证据边界

- 官方公开基线的判定来自 dev。808 条留出查询来自该集合。开题报告中的 100001/2000/9839 是团队提出的目标快照，来源见 [资源记录](evaluation/RESOURCES.md)。
- 2026-11-14 [公开演示站点](https://search.shwstone.cn/)的 `/health` 返回 200，`/index/info` 仍回显 `tiny`、10 篇、`retrieval.vector_ready=false`。本地 20k 验收使用独立服务。
- 自爬评测的判定来源为 assistant 全文复核，报告按工程评测解释。数据、模型、快照、索引与生成结果均保存在 Git 外。

## 复验入口

```bash
python tests/run.py
python -m pip install -r requirements-router-research.txt pytest torch ijson==3.4.0.post0
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s evaluation -p 'test_*.py'
python -m pytest evaluation tests/test_gain_router.py tests/test_replay_router_gain.py
node --test tests/test_web.cjs
python -m compileall .
git diff --check
```

规模构建、恢复、合并和产物校验命令见 [RFC-0004 规模构建](docs/RFC0004_SCALE.md)。
