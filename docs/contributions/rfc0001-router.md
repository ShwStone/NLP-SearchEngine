# RFC-0001：已有 Router 贡献与研究证据（阶段提交）

关联：<https://github.com/ShwStone/NLP-SearchEngine/issues/3>。
本提交提供已有策略、研究工具和实验记录，供 RFC-0001 继续讨论；不是整个 RFC 的完成声明。
从 `dba882b079f481cf03d86aef4b51d4566d63a28a` 独立提取，历史研究来源为
`96e4f492b076bb7c6daf8c61494820e0f752b221`。其他研究成果仍保存在原研究分支。

## 可直接运行的 BM25-first 对照

默认 `rules-v1` 保持原行为。新增 `RetrievalConfig.adaptive_policy` 及服务参数
`--adaptive-policy {rules-v1,recall-v1,hybrid-v1}`，只影响 adaptive 决策：

| 策略 | 行为 | 当前结论 |
| --- | --- | --- |
| rules-v1 | 原有覆盖率、分差规则 | 默认基线 |
| recall-v1 | 原规则基础上，单个词法命中也请求向量检索 | 消除“只有一条命中就认为充分”的结构缺口；尚无普遍准确率提升证据 |
| hybrid-v1 | 普通自然语言查询总是请求向量检索 | 质量对照，须实际测量成本；并不保证融合优于 dense |

先执行 BM25，决策本身不计算 embedding。显式布尔/短语约束优先于策略；
候选深度、分页、等权 RRF、同分排序和 adaptive 后端故障回退保持原契约。
返回的 `routing.policy_version` 标识所配置策略。强制 lexical/dense/hybrid 模式继续按原语义执行。
不引入新的 API 版本。启动示例（向量依赖仍按主 README 配置）：

```bash
python3 -m online.server --index fixtures/tiny_index --adaptive-policy recall-v1
python3 tests/run.py
python3 -m unittest discover -s evaluation
python3 -m evaluation.replay_router_evidence
```

没有向量后端时，新增策略会明确回退，不能把此类运行作为混合检索质量结果。

## 备选方案卡：历史 dense-first 增益预测

先获得向量候选的分数统计，以轻量增益模型预测是否值得补充 BM25，再融合。
优势是可能节省词法检索；代价是每次都调用向量，不能作为节省 GPU 或向量调用的方案。
它不符合 RFC 当前主线的 BM25-first 顺序，因此只放在 `evaluation/`，没有接入在线服务。
原模型使用 candidate_k=200、rrf_c=10、BM25/dense 权重 .25/1，与当前默认等权 RRF 不同；
不得把旧模型直接用于本提交的默认配置。

已有工具：`train_cascade.py` 用 dev 的 5 外层/4 内层 group CV 选择模型及阈值，
检验导出模型预测与 sklearn 一致；`router_model.py` 提供冻结模型离线推断。
保留 Ridge、树及森林的原研究流程。可用原研究分支保存的 capture、分组审计和 qrels 重跑：

```bash
# 可选训练依赖，在线策略和证据复算无需这些依赖
pip install -r requirements-router-research.txt
python3 -m evaluation.train_cascade \
  --capture artifacts/round3/dev-candidates.jsonl \
  --audit artifacts/round3/query-audit.json \
  --qrels artifacts/duretrieval/qrels.tsv --out artifacts/router-training
```

大语料、原查询文本、候选 capture 和模型权重下载不纳入 Git。上述数据文件需要从原研究工作区取得；
本提交包含无需它们即可复算的逐查询决策、增益和摘要，并附 SHA256 清单。
冻结模型只是历史证据，没有在本分支训练新模型。

## 备选方案卡：增加语料统计与成本目标

`round4_router_study.py` 在已有 14 个查询/向量特征基础上增加 7 个 DF/IDF/OOV 特征，
比较 vector、lexical-only、combined 的特征选择，以及质量/成本目标和 .01/.002 质量预算。
其中 lexical-only 指预测器使用的特征，历史执行仍为 dense-first。
训练输入全部来自 dev，所有分组继承冻结审计；不读取 test/external 标签调参。
本提交将原 CompactIndex 依赖换为上游 DiskIndex 的同一 df 接口，并显式传入数据路径；
算法和历史记录保持不变。大索引的 DiskIndex 会占用较多内存。

```bash
python3 -m evaluation.round4_router_study \
  --capture artifacts/round3/dev-candidates.jsonl \
  --audit artifacts/round3/query-audit.json \
  --qrels artifacts/duretrieval/qrels.tsv --index /path/to/lexical-v1 \
  --out artifacts/router-study
# 可选 --cost-file 原 packed-64 的逐查询 trace；只使用其中 lexical 阶段时间。
```

成本代理是 sum(df) 或单次已记录的词法阶段毫秒数，未计入特征提取、路由、编码、向量、融合、渲染和 HTTP，
不是端到端节省率。旧融合参数及 dev 已被前轮观察，嵌套 CV 也不能消除整个研究过程的选择偏差。

## 历史结果及限制

`router_evidence/` 的 JSON/JSONL 是原实验记录；`inventory.json` 标记来源提交及文件校验和。
`replay_router_evidence` 校验文件并逐查询重算 997 条 dev 的损失、BM25 调用节省和成本节省。
复算验证记录自洽，不代表在当前拆分分支重新完成训练或端到端检索。

| 历史评估 | dense nDCG@10 | 固定加权 hybrid | 冻结 cascade | BM25 调用节省 |
| --- | --- | --- | --- | --- |
| 已观察的 Du test，1000 条 | 0.770882 | 0.774300 | 0.775024 | 33.6% |
| MMarco 外部迁移，999 条 | 0.645372 | 0.656156 | 0.655141 | 18.42% |

Du 上相对 hybrid 的差异区间跨 0，不能宣称显著提升准确率；MMarco 未达到 30% 节省目标。
Round4 新语料特征在 .01 预算下没有改变外层选择；.002 严格预算下所有外层方案均未通过。
没有因这些探索导出/部署新策略。负结果也保留用于避免重复试错。
`recall-v1` 在已观察的 512 条 CMedQA 外部问题上未改变路由，不能宣称它已经提高真实集准确率。

## 本提交验证与后续边界

- 标准库核心测试：50/50；评测模块测试：22/22。
- 新策略覆盖单命中、约束优先、强制模式、candidate_k、故障回退及策略标识。
- 冻结模型覆盖线性导出公式及非法输入；全部历史逐查询证据复算通过。
- 另外以小型人工特征验证 sklearn 训练/导出/加载一致性；不将其作为检索质量实验。

仍需由 RFC-0001 工作补充论文调研、主线 BM25-first 的新特征/训练与独立评测，
并和 RFC-0003 协商最终策略扩展接口。没有把概念扩展、单字索引、重排序、UI 或在线读表优化混入这里。
