"use strict";

// Optional browser-logic checks: node --test tests/test_web.cjs (Node.js 18+).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const nodes = Object.fromEntries([
  "#search-form", "#query", "#mode", "#scorer", "#expander", "#k", "#submit",
  "#compare-btn", "#status", "#error", "#results", "#compare-panel",
  "#observability", "#summary", "#expanded-terms", "#routing", "#timings",
  "#index-summary", "#index-identity", "#identity-facts", "#dataset-picker",
  "#dataset-description", "#example-queries", "#pagination", "#prev-page",
  "#next-page", "#page-info", "#copy-url", "#service-state", "#benchmark-content",
  ".performance-note",
].map((selector) => [selector, {
  addEventListener() {}, innerHTML: "", textContent: "", hidden: false, value: "",
  className: "", disabled: false, attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
  getAttribute(name) { return this.attributes[name]; },
  focus() {},
}]));
nodes["#mode"].options = ["lexical", "adaptive", "hybrid", "dense"].map((value) => ({ value, disabled: false }));
nodes["#scorer"].value = "bm25";
nodes["#expander"].value = "none";
const clipboard = { written: null };
const context = vm.createContext({
  document: { querySelector: (selector) => nodes[selector] },
  fetch: async () => { throw new Error("No network in UI tests"); },
  navigator: { clipboard: { writeText: async (value) => { clipboard.written = value; } } },
  URL,
});
vm.runInContext(
  fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8"),
  context,
  { filename: "web/app.js" },
);
const { highlightedSnippet } = context;

function highlight(start, end, kind = "original") {
  return { start, end, kind };
}

test("BMP highlights retain their character offsets", () => {
  assert.equal(
    highlightedSnippet("心肌梗死症状", [highlight(0, 4)]),
    '<mark class="original">心肌梗死</mark>症状',
  );
});

test("non-BMP characters before a hit do not shift its highlight", () => {
  assert.equal(
    highlightedSnippet("😀 apple", [highlight(2, 7)]),
    '😀 <mark class="original">apple</mark>',
  );
});

test("highlight boundaries never split surrogate pairs", () => {
  assert.equal(
    highlightedSnippet("𠮷野家", [highlight(0, 1, "expanded")]),
    '<mark class="expanded">𠮷</mark>野家',
  );
});

test("highlight clipping uses the snippet's code-point length", () => {
  assert.equal(highlightedSnippet("😀 apple", [highlight(7, 20)]), "😀 apple");
  assert.equal(
    highlightedSnippet("😀 apple", [highlight(-5, 20)]),
    '<mark class="original">😀 apple</mark>',
  );
});

test("Unicode handling preserves HTML escaping and empty snippets", () => {
  assert.equal(
    highlightedSnippet("<script>&😀", [highlight(0, 8)]),
    '<mark class="original">&lt;script&gt;</mark>&amp;😀',
  );
  assert.equal(highlightedSnippet(""), '<span class="muted">无摘要</span>');
});

async function successfulSearch() {
  nodes["#query"].value = "检索";
  nodes["#mode"].value = "hybrid";
  nodes["#k"].value = "10";
  context.fetch = async () => ({
    ok: true,
    json: async () => ({
      api_version: "2.0", took_ms: 2, routing: { executed_mode: "hybrid" },
      timings_ms: { vector: 1 },
      results: [{ rank: 1, doc_id: "previous-doc", title: "旧结果", score: 1,
        snippet: "检索", highlights: [{ start: 0, end: 2 }] }],
    }),
  });
  await context.search({ preventDefault() {} });
  assert.equal(nodes["#observability"].hidden, false);
  assert.match(nodes["#routing"].innerHTML, /hybrid/);
  assert.match(nodes["#results"].innerHTML, /previous-doc/);
}

function assertFailedSearchIsCleared() {
  assert.equal(nodes["#summary"].textContent, "");
  assert.equal(nodes["#routing"].innerHTML, "");
  assert.equal(nodes["#timings"].innerHTML, "");
  assert.equal(nodes["#observability"].hidden, true);
  assert.equal(nodes["#results"].innerHTML, "");
  assert.equal(nodes["#submit"].disabled, false);
  assert.equal(nodes["#error"].hidden, false);
}

test("HTTP failures clear the previous successful search facts", async () => {
  await successfulSearch();
  context.fetch = async () => ({
    ok: false, status: 503, json: async () => ({ error: "vector unavailable" }),
  });
  await context.search({ preventDefault() {} });
  assertFailedSearchIsCleared();
  assert.equal(nodes["#error"].textContent, "vector unavailable");
});

test("non-JSON HTTP failures fall back to the HTTP status", async () => {
  await successfulSearch();
  context.fetch = async () => ({
    ok: false, status: 503, json: async () => { throw new SyntaxError("invalid JSON"); },
  });
  await context.search({ preventDefault() {} });
  assertFailedSearchIsCleared();
  assert.equal(nodes["#error"].textContent, "请求失败（HTTP 503）");
});

test("malformed successful responses clear the previous successful search facts", async () => {
  const invalidBodies = [
    async () => { throw new SyntaxError("invalid JSON"); },
    async () => ({ routing: { executed_mode: "dense" } }),
    async () => ({ results: {} }),
  ];
  for (const json of invalidBodies) {
    await successfulSearch();
    context.fetch = async () => ({ ok: true, status: 200, json });
    await context.search({ preventDefault() {} });
    assertFailedSearchIsCleared();
    assert.equal(nodes["#error"].textContent, "搜索响应格式无效");
  }
});

test("a valid empty search response displays zero results", async () => {
  await successfulSearch();
  context.fetch = async () => ({
    ok: true,
    json: async () => ({
      api_version: "2.0", took_ms: 2, parsed: {}, expanded: [],
      results: [], routing: { executed_mode: "hybrid" }, timings_ms: {},
    }),
  });
  await context.search({ preventDefault() {} });
  assert.equal(nodes["#status"].textContent, "检索完成，返回 0 条结果");
  assert.match(nodes["#results"].innerHTML, /没有找到结果/);
  assert.equal(nodes["#observability"].hidden, false);
  assert.equal(nodes["#error"].hidden, true);
});

test("render failures clear partially rendered routing and timing facts", async () => {
  await successfulSearch();
  context.fetch = async () => ({
    ok: true,
    json: async () => ({ routing: { executed_mode: "dense" }, timings_ms: { vector: 99 },
      results: [{ rank: 1, doc_id: "bad-result", snippet: "检索", highlights: {} }] }),
  });
  await context.search({ preventDefault() {} });
  assertFailedSearchIsCleared();
});

const FINGERPRINT = "1480ab10cbefcb57bc1132871420b1f88d18ac5f260150080953f9441377a861";
const ANALYZER_HASH = "12bee770ee080a665c95862486cc711c658b8d5bbacf6e06dce6d9cc01fc0926";
const REVISION = "7999e1d3359715c523056ef9478215996d62a620";

function fullIndexInfo() {
  return {
    schema_version: 1,
    dataset: "rfc4-clean-news-20k",
    n_docs: 20000,
    analyzer: { name: "simple-bigram", version: "0.1.0", config_hash: ANALYZER_HASH },
    bm25: { k1: 1.2, b: 0.75, idf: "log1p-lucene" },
    retrieval: {
      vector_ready: true,
      index_identity: {
        dataset: "rfc4-clean-news-20k", n_docs: 20000, schema_version: 1,
        analyzer_hash: ANALYZER_HASH, corpus_sha256: FINGERPRINT,
        bm25: { k1: 1.2, b: 0.75, idf: "log1p-lucene" },
      },
      vector_identity: {
        corpus_sha256: FINGERPRINT, n_docs: 20000, schema_version: 1,
        embedding: {
          model_id: "BAAI/bge-small-zh-v1.5", revision: REVISION, dimension: 512,
          query_prefix: "为这个句子生成表示以用于检索相关文章：",
        },
        collection: "news_rfc4_clean20k", backend: "qdrant-hnsw",
        index_params: { m: 16, ef_construct: 100, full_scan_threshold: 10 },
        search: { hnsw_ef: 256, exact: false, timeout_s: 30.0 },
      },
      search_config: { candidate_k: 200, rrf_c: 60.0 },
      policy: { name: "rule", policy_version: "rules-v1", requires_statistics: false },
    },
  };
}

test("index identity renders manifest, vector and serving facts", () => {
  const html = context.renderIdentityFacts(fullIndexInfo());
  assert.match(html, /rfc4-clean-news-20k/);
  assert.match(html, /news_rfc4_clean20k/);
  assert.match(html, /qdrant-hnsw/);
  assert.match(html, /BAAI\/bge-small-zh-v1\.5/);
  assert.match(html, /512/);
  assert.match(html, /k1=1\.2 · b=0\.75 · idf=log1p-lucene/);
  assert.match(html, /m=16 · ef_construct=100 · full_scan_threshold=10/);
  assert.match(html, /ef=256 · exact=否 · timeout=30 s/);
  assert.match(html, /candidate_k=200 · RRF c=60/);
  assert.match(html, /rule · 版本=rules-v1 · 需要统计=否/);
  assert.match(html, /为这个句子生成表示以用于检索相关文章：/);
});

test("vector readiness is labelled as startup configuration, not live health", () => {
  const summaryText = context.indexSummaryText(fullIndexInfo());
  assert.match(summaryText, /rfc4-clean-news-20k · 20000 篇文档/);
  assert.match(summaryText, /向量启动时已配置 · 非实时健康/);
  assert.doesNotMatch(summaryText, /已就绪|就绪/);
  assert.match(context.renderIdentityFacts(fullIndexInfo()),
    /启动时已配置（启动校验，非后端实时健康）/);

  const lexicalInfo = { dataset: "tiny", n_docs: 10, retrieval: { vector_ready: false } };
  assert.match(context.indexSummaryText(lexicalInfo), /仅词法模式/);
  assert.match(context.renderIdentityFacts(lexicalInfo), /未配置（仅词法模式）/);
  assert.doesNotMatch(context.indexSummaryText({ retrieval: {} }), /已就绪|就绪/);
  assert.match(context.indexSummaryText({ retrieval: {} }), /向量状态未知/);
});

test("index identity escapes every served value", () => {
  const info = fullIndexInfo();
  info.retrieval.index_identity.dataset = '<img src=x onerror="alert(1)">';
  info.retrieval.vector_identity.collection = "<script>alert('x')</script>";
  info.retrieval.vector_identity.embedding.query_prefix = "\"><b>&";
  const html = context.renderIdentityFacts(info);
  assert.doesNotMatch(html, /<img|<script|<b>/);
  assert.match(html, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
  assert.match(html, /&lt;script&gt;alert\(&#39;x&#39;\)&lt;\/script&gt;/);
  assert.match(html, /&quot;&gt;&lt;b&gt;&amp;/);
});

test("index identity degrades safely when optional blocks are missing", () => {
  const empty = context.renderIdentityFacts({});
  assert.doesNotMatch(empty, /undefined|null|NaN/);
  assert.match(empty, /—/);
  assert.match(empty, /未提供/);
  assert.equal(context.indexSummaryText({}), "未知数据集 · ? 篇文档 · 向量状态未知");

  // Legacy /index/info shape: lexical manifest + retrieval.config only.
  const legacy = context.renderIdentityFacts({
    dataset: "tiny", n_docs: 10,
    analyzer: { name: "simple-bigram", version: "0.1.0", config_hash: ANALYZER_HASH },
    retrieval: {
      vector_ready: false, vector_manifest: null,
      config: { candidate_k: 200, rrf_c: 60.0, policy_version: "rules-v1" },
    },
  });
  assert.doesNotMatch(legacy, /undefined|null|NaN/);
  assert.match(legacy, /tiny/);
  assert.match(legacy, /candidate_k=200 · RRF c=60/);
  assert.match(legacy, /版本=rules-v1/);
  assert.match(legacy, /12bee770ee08…/);

  // Pre-identity vector_manifest is still usable for backend/model facts.
  const legacyVector = context.renderIdentityFacts({
    dataset: "tiny", n_docs: 10,
    retrieval: {
      vector_ready: true,
      vector_manifest: {
        corpus_sha256: FINGERPRINT, n_docs: 10,
        embedding: { model_id: "BAAI/bge-small-zh-v1.5", revision: REVISION, dimension: 512 },
        collection: "news_rfc4_clean20k", backend: "qdrant-hnsw",
      },
    },
  });
  assert.match(legacyVector, /news_rfc4_clean20k/);
  assert.match(legacyVector, /BAAI\/bge-small-zh-v1\.5/);
  assert.match(legacyVector, /1480ab10cbef…/);
});

test("identity fingerprint keeps its full value for copying", async () => {
  const html = context.renderIdentityFacts(fullIndexInfo());
  const copyValues = [...html.matchAll(/data-copy="([^"]+)"/g)].map((match) => match[1]);
  assert.ok(copyValues.includes(FINGERPRINT), "full corpus fingerprint is copyable");
  assert.ok(copyValues.includes(REVISION), "full embedding revision is copyable");
  assert.doesNotMatch(html, new RegExp(`data-copy="[^"]*…`), "copy values are never truncated");

  const button = { textContent: "复制", getAttribute: () => FINGERPRINT };
  clipboard.written = null;
  await context.copyIdentityValue(button);
  assert.equal(clipboard.written, FINGERPRINT);
  assert.equal(button.textContent, "已复制");

  const failing = { textContent: "复制", getAttribute: () => FINGERPRINT };
  context.navigator.clipboard.writeText = async () => { throw new Error("denied"); };
  await context.copyIdentityValue(failing);
  assert.equal(failing.textContent, "复制失败");
  context.navigator.clipboard.writeText = async (value) => { clipboard.written = value; };
});

test("index info is rendered on load and cleared when the request fails", async () => {
  context.fetch = async () => ({ ok: true, json: async () => fullIndexInfo() });
  await context.loadIndexInfo();
  assert.equal(nodes["#index-identity"].hidden, false);
  assert.equal(nodes["#index-summary"].textContent,
    "rfc4-clean-news-20k · 20000 篇文档 · 向量启动时已配置 · 非实时健康");
  assert.match(nodes["#identity-facts"].innerHTML, /news_rfc4_clean20k/);

  context.fetch = async () => ({ ok: false, status: 503, json: async () => ({}) });
  await context.loadIndexInfo();
  assert.equal(nodes["#index-summary"].textContent, "索引信息暂不可用");
  assert.equal(nodes["#identity-facts"].innerHTML, "");
  assert.equal(nodes["#index-identity"].hidden, true);
});

const appState = vm.runInContext("state", context);

function datasetCatalog() {
  return [
    { id: "news20k", label: "20k 自爬新闻", description: "团队新闻", n_docs: 20000,
      modes: ["lexical", "dense", "hybrid", "adaptive"], examples: ["核电"] },
    { id: "mini100k", label: "100k 协作子集", description: "DuReader 子集", n_docs: 100000,
      modes: ["lexical"], examples: ["QQ 群名片"] },
  ];
}

test("dataset selection loads the matching identity and disables unavailable modes", async () => {
  appState.catalog = datasetCatalog();
  appState.dataset = "";
  const urls = [];
  context.fetch = async (url) => {
    urls.push(url);
    return { ok: true, json: async () => fullIndexInfo() };
  };
  await context.selectDataset("mini100k");
  assert.deepEqual(urls, ["/datasets/mini100k/info"]);
  assert.equal(appState.dataset, "mini100k");
  assert.match(nodes["#dataset-picker"].innerHTML, /100k 协作子集/);
  assert.match(nodes["#example-queries"].innerHTML, /QQ 群名片/);
  assert.equal(nodes["#mode"].options.find((option) => option.value === "dense").disabled, true);
  assert.equal(nodes["#mode"].options.find((option) => option.value === "lexical").disabled, false);
  assert.equal(nodes["#mode"].value, "lexical");
});

test("dataset-specific searches and pagination include dataset and offset", async () => {
  appState.catalog = datasetCatalog();
  await context.selectDataset("mini100k");
  const requests = [];
  context.fetch = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    return { ok: true, json: async () => ({
      api_version: "2.0", took_ms: 3, parsed: {}, expanded: [],
      results: Array.from({ length: 10 }, (_, index) => ({ rank: index + 1,
        doc_id: `doc-${index}`, title: "结果", score: 1, snippet: "内容", highlights: [] })),
      routing: { executed_mode: "lexical" }, timings_ms: {},
    }) };
  };
  nodes["#query"].value = "QQ 群名片";
  nodes["#mode"].value = "lexical";
  nodes["#k"].value = "10";
  await context.search({ preventDefault() {} });
  await context.changePage(1);
  assert.equal(requests[0].url, "/datasets/mini100k/search");
  assert.equal(requests[0].body.offset, 0);
  assert.equal(requests[1].body.offset, 10);
  assert.equal(nodes["#page-info"].textContent, "第 2 页");
  assert.equal(nodes["#prev-page"].disabled, false);
});

test("a stale response cannot overwrite a newly selected dataset", async () => {
  appState.catalog = datasetCatalog();
  await context.selectDataset("news20k");
  let resolveSearch;
  context.fetch = async () => new Promise((resolve) => { resolveSearch = resolve; });
  nodes["#query"].value = "核电";
  const pending = context.executeSearch(0);
  context.fetch = async () => ({ ok: true, json: async () => fullIndexInfo() });
  await context.selectDataset("mini100k");
  resolveSearch({ ok: true, json: async () => ({
    api_version: "2.0", took_ms: 1, results: [{ rank: 1, doc_id: "stale", score: 1,
      snippet: "旧结果", highlights: [] }], routing: {}, timings_ms: {},
  }) });
  await pending;
  assert.doesNotMatch(nodes["#results"].innerHTML, /stale|旧结果/);
  assert.match(nodes["#results"].innerHTML, /100k 协作子集/);
});

test("shared links restore dataset query mode page size and offset", () => {
  context.window = { location: { href: "https://search.example/?dataset=official8m&q=%E8%8A%82%E5%81%87%E6%97%A5&mode=dense&k=20&offset=20" } };
  const location = context.applyLocation();
  assert.equal(location.offset, 20);
  assert.equal(appState.requestedOffset, 20);
  assert.equal(nodes["#query"].value, "节假日");
  assert.equal(nodes["#mode"].value, "dense");
  assert.equal(nodes["#k"].value, "20");
  delete context.window;
});

test("benchmark rendering escapes evidence labels", () => {
  context.renderBenchmarks({
    cards: [{ label: "<img src=x>", value: "12.3", unit: "ms", detail: "<script>bad()</script>" }],
    rows: [{ label: "baseline", p50: "12 ms", p95: "30 ms", speedup: "1×" }],
    quality_rows: [{ label: "<b>news20k</b>", ndcg: "0.9168", recall: "0.8897", mrr: "0.9669", note: "BM25" }],
  });
  assert.doesNotMatch(nodes["#benchmark-content"].innerHTML, /<img|<script>|<b>/);
  assert.match(nodes["#benchmark-content"].innerHTML, /12\.3/);
  assert.match(nodes["#benchmark-content"].innerHTML, /0\.9168/);
});

test("lexical scorer and RM3 expander pass through on lexical mode and render expanded chips", async () => {
  appState.catalog = datasetCatalog();
  await context.selectDataset("news20k");
  const bodies = [];
  context.fetch = async (url, options) => {
    bodies.push(JSON.parse(options.body));
    return {
      ok: true,
      json: async () => ({
        api_version: "2.0", took_ms: 5, parsed: {},
        expanded: [{ term: "核能", w: 0.75, src: "rm3" }],
        results: [{ rank: 1, doc_id: "doc-1", title: "核电", score: 10.5, snippet: "核电核能", highlights: [] }],
        routing: { executed_mode: "lexical", reason: "forced_lexical" }, timings_ms: { lexical: 4 },
      }),
    };
  };
  nodes["#query"].value = "核电";
  nodes["#mode"].value = "lexical";
  nodes["#scorer"].value = "tfidf";
  nodes["#expander"].value = "rm3";
  nodes["#k"].value = "10";
  await context.search({ preventDefault() {} });
  assert.equal(bodies[0].scorer, "tfidf");
  assert.equal(bodies[0].expander, "rm3");
  assert.equal(nodes["#expanded-terms"].hidden, false);
  assert.match(nodes["#expanded-terms"].innerHTML, /核能/);
  nodes["#scorer"].value = "bm25";
  nodes["#expander"].value = "none";
});
