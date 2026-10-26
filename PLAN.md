# 当前实施计划：自适应混合检索

> 当前共识以 [DESIGN.md](DESIGN.md) 和 [contracts/RETRIEVAL.md](contracts/RETRIEVAL.md) 为准。下方历史对话中的 RM3/SLM 四行实验、纯词法离线范围已被新设计取代。

| 顺序 | 离线组 | 在线组 | 联合验收 |
|---|---|---|---|
| E0 契约冻结 | 文档 ID、BM25 产物、向量 manifest/快照规则 | TextEncoder/VectorReader 消费方式、四模式 HTTP、Router/RRF | 假模型/假向量与 HTTP 契约测试；代码骨架已落地 |
| E1 小样本真实双路闭环 | 可选 Sentence-Transformers/Qdrant writer 已接线；tiny 词法产物和 manifest 校验可复现 | Qdrant reader、查询编码、启动版本校验、显式语法/过滤、UI 已接线 | HTTP evaluator 和内存假向量已验收；集群 prefix20k 真实模型/local exact 四模式冒烟已记录 |
| E2 三条固定基线 | 全量 BM25 + 向量库、构建/存储/内存记录 | BM25/纯向量/固定混合；固定模型、候选池与 RRF 参数 | 同一语料和查询划分，BM25 对齐 bm25s |
| E3 自适应与交付 | 排查 ID/版本/快照问题，提供只读稳定产物 | 开发集定路由、测试集四组对照、UI/报告 | nDCG/Recall、延迟、向量尝试/成功/降级率及成本权衡 |

工作边界：**离线负责 BM25 构建和向量数据库构建；在线负责查询编码、Router、向量召回与 RRF 融合**。不训练模型、不生成答案。两侧只共享 contracts/；在线只读离线产物。

当前真实模型和 Qdrant 适配器已作为可选依赖接入，集群已完成 prefix20k 的真实四模式工程冒烟；课程实际全量快照、test labels、HTTP Qdrant/HNSW 性能仍未验收。教师要求的路由比较和自建网络测试集流程已分别由 HTTP telemetry/paired bootstrap 与 `evaluation.crawl_web` 支持，详见 [TODO.md](TODO.md)。运行 `python3 tests/run.py`、`python3 -m unittest discover -s evaluation -p 'test_*.py'` 和 `node --test tests/test_web.cjs` 验证框架；不要把模板或 prefix smoke 写成正式全量成绩。

---

# 历史讨论存档（不作为现行实现要求）

> From: https://chatgpt.com/c/6ad0c360-0d20-83ea-8e5c-995142756c72

# you asked

message time: 2026-10-15 20:13:26

有没有测试搜索引擎demo的数据集

---

# chatgpt response

有，而且有不少专门用于测试搜索引擎的数据集。

如果你要做一个 **搜索引擎 Demo（倒排索引 + BM25 + 查询界面 + 结果排序）**，我建议优先选择有以下三部分的数据集：

- **Corpus**：可供检索的文档集合。
- **Queries**：真实或人工构造的搜索请求。
- **Qrels**：每个查询对应哪些相关文档的标注，可以用来评估 Recall\@k、MRR、nDCG\@k 等指标。

这样既能展示一个可交互的搜索引擎，也能定量测试搜索质量。

## 推荐的 5 个数据集

| 数据集 | 文档数 | 查询数 | 语言 | 适合场景 |
|--------|--------|--------|--------|--------|
| [Cranfield](https://ir-datasets.com/cranfield.html) | 1,400 | 225 | 英文 | 最小规模、传统 IR 教学 |
| [NFCorpus](https://github.com/beir-cellar/beir) | 3,633 | 323 | 英文 | BM25、TF-IDF 排序评估 |
| [SciFact](https://github.com/beir-cellar/beir) | 约 5,000 | 300 | 英文 | 科学文献搜索 |
| [MedicalRetrieval](https://huggingface.co/datasets/mteb/MedicalRetrieval) | 100,999 | 1,000 | 中文 | 中文检索、较大索引 |
| [T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval) | 118,605 | 22,812 | 中文 | 综合中文搜索 Demo |

以上均提供相关性标注，数量采用对应评测集的统计。 [ir_datasets](https://ir-datasets.com/cranfield.html) [GitHub](https://github.com/beir-cellar/beir/wiki/Datasets-available) [Datasets at Hugging Face](https://huggingface.co/datasets/mteb/MedicalRetrieval) [Datasets at Hugging Face](https://huggingface.co/datasets/mteb/T2Retrieval)

## 我比较推荐的两个

**NFCorpus**

(英文轻量首选)

只有约 3,600 篇文档，但每个测试查询平均有较多相关文档，非常适合观察 BM25 排序和 Top-K 搜索结果。

**T2Retrieval**

(中文首选)

涉及医疗、学术、金融、政务等领域，数据包约 159 MB，足够展示真实的中文检索体验。

## 如何快速下载

**英文：NFCorpus / SciFact**

BEIR 提供了统一格式，可直接下载：

```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
unzip nfcorpus.zip
```

解压后主要包含：

```text
nfcorpus/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
```

**中文：T2Retrieval**

```python
from datasets import load_dataset

corpus = load_dataset(
    "mteb/T2Retrieval", "corpus", split="dev"
)
queries = load_dataset(
    "mteb/T2Retrieval", "queries", split="dev"
)
qrels = load_dataset(
    "mteb/T2Retrieval", "default", split="dev"
)
```

数据可以转换成 SQLite FTS5、Lucene 或 Elasticsearch 所需的索引格式。

**如果是传统 NLP 课程大作业**，我更推荐用 NFCorpus 或 Cranfield 起步，做出倒排索引、TF-IDF、BM25 的对比；如果希望前端展示一个更像实际产品的中文搜索引擎，则选择 T2Retrieval。

---

# you asked

message time: 2026-10-15 20:15:46

搜索引擎主要分成几部分？

---

# chatgpt response

一个传统的**全文搜索引擎（Full-text Search Engine）**，通常可以分为三个核心部分：**数据获取与处理、索引构建、查询与排序**。如果考虑完整系统，还需要搜索服务、前端交互和评测模块。

## 1. 整体架构

## 2. 核心模块

| 模块 | 主要功能 | 常见技术 |
|--------|--------|--------|
| ① 数据获取 | 收集网页、文章、文档 | Web Crawler、数据集导入 |
| ② 文本预处理 | 分词、归一化、去停用词 | Tokenization、中文分词 |
| ③ 索引构建 | 建立词项到文档的映射 | 倒排索引、Posting List |
| ④ 查询处理 | 解析用户查询、查找候选文档 | 布尔检索、词项匹配 |
| ⑤ 检索排序 | 计算相关性、返回 Top-K | TF-IDF、BM25、Learning to Rank |
| ⑥ 结果展示 | 显示标题、摘要、高亮 | Web UI、REST API |

其中最有算法价值的是 **③ 索引构建、④ 查询处理和 ⑤ 检索排序**。

### 倒排索引的例子

假设有三个文档：

```text
D1: machine learning algorithm
D2: deep learning model
D3: machine learning model
```

倒排索引为：

```text
machine  → [D1, D3]
learning → [D1, D2, D3]
algorithm→ [D1]
deep     → [D2]
model    → [D2, D3]
```

查询 `machine learning` 时，可以通过两个 Posting List 的交集直接得到 `D1, D3`，不必遍历所有文档。

之后再使用 BM25 对候选文档进行打分排序。

## 3. 对于课程 Demo，我建议这样划分

**模块 A：Index Builder**

数据集读取、文本预处理、倒排索引构建、索引持久化。建议实现 Posting List、词频和文档长度存储。

---

**模块 B：Search Engine**

查询分词、候选召回、布尔运算、TF-IDF / BM25 打分、Top-K 选择。是项目的算法核心。

---

**模块 C：Web Interface**

提供搜索框、搜索结果、关键词高亮、耗时显示和不同排序算法的切换。

---

**模块 D：Evaluator**

用数据集中的 Qrels 计算 Precision\@K、Recall\@K、MRR、nDCG，并比较不同检索算法的效果和耗时。

## 4. 哪些功能值得做，哪些可以不做？

对于传统 NLP 课程项目，我建议：

- **必须实现**：分词、倒排索引、布尔检索、BM25、Top-K 排序。
- **推荐实现**：TF-IDF 对比、查询高亮、搜索性能测试、nDCG 评测。
- **进阶功能**：拼写纠错、Query Expansion、短语检索、位置索引、相关性反馈。
- **可以省略**：大规模网络爬虫、分布式索引、PageRank、神经网络重排序。

尤其不建议把重点放在爬虫上：直接使用 NFCorpus、T2Retrieval 等现成数据集，能将主要精力集中在检索算法本身。

**最终可以把项目定义为：一个基于倒排索引与 BM25 的轻量级全文搜索引擎，支持多种检索策略及标准 IR 指标评测。** 这个范围比较适合作为传统 NLP 课程的完整大作业。

---

# you asked

message time: 2026-10-15 20:19:50

我希望能够实现“必须实现”、查询高亮、数据集评测，以及一部分针对查询语句的进阶功能，主要是 Query Expansion 和短语检索

---

# chatgpt response

我建议把这个项目定位为：

**一个支持布尔检索、BM25 相关性排序、短语检索和查询扩展的传统全文搜索引擎，并使用标准 IR 数据集进行定量评测。**

这个范围很合适：既能实现搜索引擎的核心数据结构，又能体现传统 NLP 的文本处理与查询理解技术，还能通过消融实验验证 Query Expansion 的实际效果。

我建议采用 **自己实现检索核心 + 轻量级 Web UI + 标准数据集评测** 的方式，而不是直接调用 Elasticsearch、Lucene 等现成搜索引擎。

## 一、整体架构

```svg
<svg viewBox="0 0 340 352" width="100%" xmlns="http://www.w3.org/2000/svg"><rect x="9" y="8" width="322" height="39" rx="7" fill="#64748b" fill-opacity=".12" stroke="#64748b" /><text x="170" y="32" font-size="13" font-weight="600" text-anchor="middle" fill="currentColor">Corpus + Queries + Qrels</text><g stroke="#64748b" stroke-width="1.5" fill="none"><path d="M170 47V63H83V80 M170 63H256V80" /></g><g font-size="12" fill="currentColor" text-anchor="middle"><rect x="11" y="80" width="143" height="39" rx="6" fill="#3b82f6" fill-opacity=".12" stroke="#3b82f6" /><text x="82" y="104">Tokenizer</text><rect x="185" y="80" width="144" height="39" rx="6" fill="#0d9488" fill-opacity=".12" stroke="#0d9488" /><text x="257" y="104">Evaluation</text><rect x="11" y="145" width="143" height="43" rx="6" fill="#3b82f6" fill-opacity=".12" stroke="#3b82f6" /><text x="82" y="162">Positional</text><text x="82" y="178">Inverted Index</text><rect x="185" y="145" width="144" height="43" rx="6" fill="#0d9488" fill-opacity=".12" stroke="#0d9488" /><text x="257" y="162">Query Parser</text><text x="257" y="178">+ Expansion</text><rect x="67" y="216" width="207" height="40" rx="6" fill="#0d9488" fill-opacity=".12" stroke="#0d9488" /><text x="170" y="232">Retrieval + Phrase Matching</text><text x="170" y="246" font-size="10">Boolean / BM25 / Top-K</text><rect x="67" y="286" width="207" height="45" rx="6" fill="#3b82f6" fill-opacity=".12" stroke="#3b82f6" /><text x="170" y="303">Search UI</text><text x="170" y="318" font-size="10">Snippet + Highlighting</text></g><g stroke="currentColor" stroke-opacity=".55" stroke-width="1.5" fill="none"><path d="M82 119V145 M257 119V145 M82 188V199H170V216 M257 188V199H170 M170 256V286" /><path d="M256 80V63" stroke-dasharray="3 3" /></g><g font-size="9" fill="#64748b"><text x="13" y="137">离线</text><text x="187" y="137">在线</text><text x="276" y="275">结果输出</text></g></svg>
```

系统分为离线索引、在线查询、结果展示与离线评测四个模块。Evaluation 实际上会调用整个检索流程，不需要单独实现一套搜索算法。

## 二、功能范围

| 模块 | 具体功能 | 优先级 |
|--------|--------|--------|
| 文本处理 | Tokenization、大小写归一化、停用词、可选词干提取 | 必须 |
| 倒排索引 | 词频、文档长度、词项位置、索引持久化 | 必须 |
| 布尔检索 | AND、OR、NOT、括号 | 必须 |
| 排序检索 | BM25、Top-K、可选 TF-IDF 对比 | 必须 |
| 短语检索 | `"machine learning"` 精确匹配 | 必须 |
| 查询扩展 | 相关词扩展、Pseudo-Relevance Feedback | 重点 |
| 结果展示 | 标题、摘要、关键词高亮、耗时 | 必须 |
| 检索评测 | Precision\@K、Recall\@K、MRR、nDCG\@K | 必须 |

这里最重要的设计决策是：**一开始就构建带词项位置的倒排索引（Positional Inverted Index）**。这样无需后期重建索引，就可以同时支持布尔检索、BM25 和短语检索。

## 三、关键数据结构

### 1. 带位置的倒排索引

例如文档 `D1`：

```text
machine learning improves machine translation
```

建立索引：

```python
index = {
    "machine": {
        "D1": {"tf": 2, "positions": [0, 3]}
    },
    "learning": {
        "D1": {"tf": 1, "positions": [1]}
    },
    "translation": {
        "D1": {"tf": 1, "positions": [4]}
    }
}
```

还需要独立维护：

- `doc_lengths`：每篇文档的 Token 数。
- `doc_freq`：包含每个词项的文档数。
- `documents`：原始标题、正文和文档 ID。
- `token_offsets`：Token 在原文中的字符位置，用于高亮。

注意：**为了支持精确短语查询，不能在建立位置索引之前直接删除停用词**。例如 `"to be"`，如果索引时删除了 `to`，就无法正确匹配。可以同时保留完整位置索引和过滤停用词的排序词项。

### 2. 短语检索

假设搜索：

`"machine learning"`

只有当两个词项在同一文档中，且满足

$$
\operatorname{pos}(\text{learning})
=\operatorname{pos}(\text{machine})+1
$$

时，才算匹配。

多个词项则推广为：

$$
p(t_i)=p(t_0)+i
$$

实现时先对各词项的 Posting List 求文档交集，然后验证位置关系，不需要扫描整篇文档。

我建议支持以下查询语法：

| 查询 | 语义 |
|--------|--------|
| `machine learning` | 默认 BM25 相关性搜索 |
| `machine AND learning` | 同时包含两个词 |
| `machine OR learning` | 包含至少一个词 |
| `machine AND NOT learning` | 排除包含 learning 的文档 |
| `"machine learning"` | 精确短语匹配 |
| `"machine learning" OR "deep learning"` | 多短语检索 |
| `(machine OR deep) AND learning` | 混合布尔查询 |

**布尔条件和短语条件用于筛选候选集，BM25 用于对候选集排序。** 这样两种检索方式可以自然结合。

## 四、Query Expansion：项目最值得研究的部分

建议至少实现两种模式：**无扩展的 BM25 基线**和**伪相关反馈（Pseudo-Relevance Feedback, PRF）扩展**。有余力再添加 WordNet 词义扩展作为对照。

基于 PRF 的查询扩展流程

PRF 的基本思想是：假设第一次检索排名靠前的若干文档大概率与查询相关，从中提取有代表性的词项，再执行一次检索。

建议从简单的 TF-IDF 反馈词选择开始：

1. 用原始 Query 执行 BM25，取前 3 篇文档。
2. 从这些文档中提取候选词项，去除停用词和过于常见的词。
3. 根据 TF-IDF 或文档频率加权选出 5 个扩展词。
4. 原始词赋予权重 1.0，扩展词赋予较小权重，例如 0.3。
5. 使用加权 BM25 重新检索并排序。

之后可以实现更标准的 **RM3（Relevance Model 3）**，用于进一步评测。PyTerrier 有成熟的 BM25 → RM3 → BM25 实现，可以作为验证自己算法的参考。 [PyTerrier](https://pyterrier.readthedocs.io/en/latest/terrier/rewrite.html) [PyTerrier](https://pyterrier.readthedocs.io/en/stable/tuning.html)

一个重要规则：对于用户显式写出的 `AND`、`NOT` 和引号内短语，扩展词不应该改变这些硬性过滤条件。Query Expansion 主要作用于自由文本查询或排序阶段。

## 五、查询高亮与界面

machine learning

(BM25) (PRF Enabled) (Top 10)

---

An Introduction to Machine Learning Score 8.42

This paper explores machine learning algorithms and their applications to predictive models\...

Document \#124 · Matched terms: machine, learning, model (expanded)

---

Deep Learning and Neural Networks Score 7.16

Modern learning techniques leverage neural architectures and optimization algorithms\...

Document \#082 · Matched terms: learning, algorithm (expanded)

示意界面：支持区分原始匹配词与扩展词，并显示每篇文档的评分。

实现时不建议直接对 HTML 进行字符串替换，应使用索引阶段保存的原文字符偏移量生成高亮区间，合并重叠区间，并对原文进行 HTML 转义。

## 六、数据集评测和消融实验

我建议使用 [NFCorpus（BEIR）](https://github.com/beir-cellar/beir) 作为正式评测集，再用 [Cranfield](https://ir-datasets.com/cranfield.html) 做轻量功能测试。两者都有文档、查询及相关性标注。 [Datasets at Hugging Face](https://huggingface.co/datasets/BeIR/nfcorpus-qrels) [ir_datasets](https://ir-datasets.com/cranfield.html)

设计以下实验：

| 实验组 | 检索方式 | 研究问题 |
|--------|--------|--------|
| A | TF-IDF | 传统基线 |
| B | BM25 | BM25 相比 TF-IDF 的收益 |
| C | BM25 + PRF | 查询扩展是否提高相关性 |
| D | BM25 + RM3（可选） | 不同扩展算法的效果 |

各实验统一计算 `Precision@10`、`Recall@10`、`MAP@10`、`nDCG@10` 和查询耗时。

短语检索建议**单独构造功能测试集**，测量精确匹配正确率。因为普通数据集中的 Qrels 通常标注主题相关性，不保证包含特定短语，不能直接用于评判精确短语匹配是否正确。

另一个很有价值的实验是改变 PRF 的反馈文档数、扩展词数、扩展权重，观察 nDCG 的变化和 Query Drift（查询主题漂移）。在验证集上调参，最终测试集只用于报告结果。

## 七、建议项目结构

```text
search-engine/
├── data/
│   └── nfcorpus/
├── src/
│   ├── tokenizer.py
│   ├── indexer.py
│   ├── query_parser.py
│   ├── boolean_retriever.py
│   ├── phrase_retriever.py
│   ├── bm25.py
│   ├── query_expansion.py
│   ├── highlighter.py
│   └── search_engine.py
├── evaluation/
│   ├── metrics.py
│   └── benchmark.py
├── web/
│   ├── api.py
│   └── frontend/
└── tests/
```

实现顺序上，建议先完成**位置倒排索引 → 布尔/短语检索 → BM25 → 评测 → PRF → Web UI**。其中检索核心尽量只依赖 Python 标准库和基础数值计算库，其他库用于数据读取、前端和指标交叉验证即可。

最终这个 Demo 不只是展示搜索结果，还能回答一个有研究价值的问题：

**在传统词项匹配检索中，位置约束与伪相关反馈分别如何影响检索的准确性、召回率和查询开销？**

这可以成为整个课程项目的技术主线。

---

# you asked

message time: 2026-10-15 20:27:49

短语扩展部分也许可以拿个 bert 或者 slm？

---

# chatgpt response

可以把 BERT 或小语言模型作为 Query Expansion 的一个可选模块，同时保留倒排索引和 BM25 作为检索核心。我查一下适合直接使用的轻量模型，以及怎样设计对照实验。

可以，而且我认为这会成为项目里比较有特色的部分。

我的建议是：**保留传统的 BM25 + 倒排索引作为检索核心，使用一个预训练 BERT 或 Small Language Model（SLM）完成语义层面的 Query Expansion。** 不训练模型，只做推理。

不过需要区分两个概念：

- **Phrase Search（短语检索）**：查询 `"heart attack"` 时，必须找到连续出现这两个词的文档。
- **Phrase Expansion（短语扩展）**：将 `"heart attack"` 扩展为 `"myocardial infarction"` 等具有相近含义的表达，再执行检索。

它们可以很好地结合，但用户显式要求精确短语匹配时，不应默认进行语义扩展。

## 一、三种可行的模型方案

| 方案 | 模型 | 实现思路 | 推荐程度 |
|--------|--------|--------|--------|
| BERT MLM | [BERT Base](https://huggingface.co/google-bert/bert-base-uncased) | Masked Token Prediction，生成词项替换候选 | 中 |
| Sentence-BERT | [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 从语料库挖掘候选短语，再按语义相似度筛选 | **最高** |
| 生成式 SLM | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 直接生成同义短语、相关表达和查询改写 | 高 |

这些模型均有公开的预训练权重。MiniLM 支持句子相似度计算，Qwen3-0.6B 支持文本生成。 [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main) [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) [Hugging Face](https://huggingface.co/google-bert/bert-base-uncased/tree/main)

### 方案 A：BERT / MiniLM 语义短语扩展

我更倾向于使用 Sentence-BERT 而不是原始 BERT MLM。

流程是：

1. 从整个 Corpus 或首次检索的 Top-K 文档中提取 2～4 gram 候选短语。
2. 计算原始 Query 与候选短语的 Embedding。
3. 通过余弦相似度筛选语义接近的短语。
4. 使用 IDF、短语出现频率等传统统计特征进一步过滤。
5. 将选出的短语加入 BM25 查询。

可以定义：

$$
S(p,q)=\alpha\cos(E(p),E(q))+
\beta\operatorname{IDF}(p)+
\gamma\operatorname{PRF}(p)
$$

其中各项需要适当归一化，权重在验证集上确定。

这种方式不依赖生成能力，并且候选短语来自实际文档，不容易产生语料库中根本不存在的扩展词。

### 方案 B：SLM 生成查询扩展

用 Qwen3-0.6B 等小模型直接进行查询改写：

原始查询

`"heart attack"`

SLM Query Expansion

可能生成的候选表达（示意）

myocardial infarction (同义医学术语)

cardiac infarction (近义表达)

heart disease (宽泛相关词)

扩展词需要过滤；例如 heart disease 不是 heart attack 的同义词，不应直接作为等价短语。

这种方式对同义表达、缩写、专业术语很友好。但缺点是推理成本更高，而且小模型生成的表达未必精确，可能引起 Query Drift。

## 二、我推荐的混合架构

```svg
<svg viewBox="0 0 330 268" width="100%" xmlns="http://www.w3.org/2000/svg"><g font-size="11" fill="currentColor" text-anchor="middle"><rect x="85" y="5" width="160" height="32" rx="6" fill="#64748b" fill-opacity=".12" stroke="#94a3b8" /><text x="165" y="25" font-weight="600">Original Query</text><path d="M165 37V50H85V62 M165 50H245V62" stroke="currentColor" stroke-width="1.5" opacity=".5" fill="none" /><rect x="6" y="62" width="155" height="43" rx="6" fill="#0d9488" fill-opacity=".12" stroke="#0d9488" /><text x="83" y="79" font-weight="600">Traditional PRF</text><text x="83" y="94" font-size="10">TF-IDF / RM3</text><rect x="171" y="62" width="153" height="43" rx="6" fill="#3b82f6" fill-opacity=".12" stroke="#3b82f6" /><text x="247" y="79" font-weight="600">Neural Expansion</text><text x="247" y="94" font-size="10">MiniLM / SLM</text><path d="M83 105V121H165V137 M247 105V121H165" stroke="currentColor" stroke-width="1.5" opacity=".5" fill="none" /><rect x="67" y="137" width="196" height="38" rx="6" fill="#64748b" fill-opacity=".12" stroke="#94a3b8" /><text x="165" y="151" font-weight="600">Expansion Filtering</text><text x="165" y="165" font-size="10">Corpus / Similarity / Weight</text><path d="M165 175V189" stroke="currentColor" stroke-width="1.5" opacity=".5" /><rect x="67" y="189" width="196" height="33" rx="6" fill="#0d9488" fill-opacity=".12" stroke="#0d9488" /><text x="165" y="210" font-weight="600">Weighted BM25 Retrieval</text><path d="M165 222V234" stroke="currentColor" stroke-width="1.5" opacity=".5" /><rect x="85" y="234" width="160" height="30" rx="6" fill="#64748b" fill-opacity=".12" stroke="#94a3b8" /><text x="165" y="253" font-weight="600">Ranked Results</text></g></svg>
```

这里的一个技术细节值得注意：**扩展短语不能简单地拆成单词后全部加入 BM25。**

例如扩展短语 `"myocardial infarction"`，如果拆开计算，可能检索到只有 `myocardial` 或 `infarction` 的文档，甚至匹配错误组合。

因此可以在原有 BM25 上增加短语匹配奖励：

$$
S(d,q)=S_{\mathrm{BM25}}(d,q)
+\lambda\sum_{p\in P(q)}
w_p\,\mathbf{1}[p\text{ occurs in }d]
$$

其中 $P(q)$ 是生成的扩展短语集合，$w_p$ 是短语权重，通过位置倒排索引判断是否连续匹配。这个评分公式可以作为你自己设计的简单实现。

## 三、实验可以更有研究价值

既然引入了神经模型，我建议将 Query Expansion 设计为一个专门的消融实验：

| 实验 | 查询扩展方式 | 检索后端 |
|--------|--------|--------|
| Baseline | 无 | BM25 |
| Traditional | RM3 | BM25 |
| Neural Encoder | MiniLM 短语筛选 | BM25 |
| Generative | Qwen3-0.6B 短语生成 | BM25 |
| Hybrid | RM3 + MiniLM | BM25 |

统一评测 nDCG\@10、Recall\@10、MAP，以及查询时延。

特别值得分析的是：**神经查询扩展是否一定优于传统的统计方法？**

答案未必。比如在 NFCorpus 这样的生物医学领域数据集上，通用语言模型可能对医学术语的判断不够准确。RM3 利用的则是数据集自身的语料统计信息。这种差异反而能够提供很好的实验分析材料。 [Datasets at Hugging Face](https://huggingface.co/datasets/BeIR/nfcorpus) [Wiley Online Library](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/1697950)

## 四、最终建议

我会优先实现 **MiniLM + Corpus-based Phrase Expansion**，因为它既保持了传统搜索引擎的主体，也引入了一小部分真正有用的语义理解能力，而且可以完全离线运行。

之后再把 Qwen3-0.6B 做成可选的生成式扩展模块，让 Demo 支持切换：

- Original Query
- Statistical Expansion（RM3）
- Semantic Expansion（MiniLM）
- Generative Expansion（SLM）

这样可以很好地展示从传统词项匹配到神经辅助查询理解的演进，同时又不会变成一个简单封装 LLM API 的项目。

**尤其是既有短语检索的 Positional Index，又有神经短语扩展，两者之间的结合是这个项目最值得突出的设计点。**
