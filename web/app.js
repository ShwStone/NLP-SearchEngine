const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeInput = document.querySelector("#mode");
const modeTabs = document.querySelector("#mode-tabs");
const lexicalOptions = document.querySelector("#lexical-options");
const scorerInput = document.querySelector("#scorer");
const expanderInput = document.querySelector("#expander");
const kInput = document.querySelector("#k");
const submitButton = document.querySelector("#submit");
const compareButton = document.querySelector("#compare-btn");
const statusBox = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const resultBox = document.querySelector("#results");
const comparePanel = document.querySelector("#compare-panel");
const observability = document.querySelector("#observability");
const summary = document.querySelector("#summary");
const compactTimings = document.querySelector("#compact-timings");
const toggleTraceButton = document.querySelector("#toggle-trace");
const traceDetails = document.querySelector("#trace-details");
const expandedTermsBox = document.querySelector("#expanded-terms");
const routingBox = document.querySelector("#routing");
const timingsBox = document.querySelector("#timings");
const indexSummary = document.querySelector("#index-summary");
const indexDetails = document.querySelector("#index-identity");
const identityFactsBox = document.querySelector("#identity-facts");
const datasetPicker = document.querySelector("#dataset-picker");
const datasetDescription = document.querySelector("#dataset-description");
const exampleQueries = document.querySelector("#example-queries");
const recentQueriesBox = document.querySelector("#recent-queries");
const pagination = document.querySelector("#pagination");
const previousPage = document.querySelector("#prev-page");
const nextPage = document.querySelector("#next-page");
const pageInfo = document.querySelector("#page-info");
const copyUrlButton = document.querySelector("#copy-url");
const serviceState = document.querySelector("#service-state");
const benchmarkContent = document.querySelector("#benchmark-content");

const docDrawer = document.querySelector("#doc-drawer");
const docDrawerBackdrop = document.querySelector("#doc-drawer-backdrop");
const docDrawerClose = document.querySelector("#doc-drawer-close");
const docDrawerCloseFoot = document.querySelector("#doc-drawer-close-foot");
const docDrawerSource = document.querySelector("#doc-drawer-source");
const docDrawerId = document.querySelector("#doc-drawer-id");
const docDrawerCopyId = document.querySelector("#doc-drawer-copy-id");
const docDrawerTitle = document.querySelector("#doc-drawer-title");
const docDrawerMeta = document.querySelector("#doc-drawer-meta");
const docDrawerNextHit = document.querySelector("#doc-drawer-next-hit");
const docDrawerBody = document.querySelector("#doc-drawer-body");
const docDrawerCopy = document.querySelector("#doc-drawer-copy");
const docDrawerResearch = document.querySelector("#doc-drawer-research");

const RECENT_STORAGE_KEY = "zhihai_recent_queries_v1";
const RECENT_MAX = 6;

const state = {
  catalog: [],
  dataset: "",
  offset: 0,
  resultCount: 0,
  requestedOffset: 0,
  requestSerial: 0,
  controller: null,
  info: null,
  activeView: "search", // "search" | "compare"
  lastQuery: "",
  resultMap: new Map(),
  visitedDocs: new Set(),
  drawer: {
    serial: 0,
    controller: null,
    docId: "",
    title: "",
    text: "",
    hitIndex: -1,
    hitCount: 0,
  },
};

const MODE_LABELS = {
  lexical: "词法检索",
  adaptive: "自适应检索",
  hybrid: "混合召回",
  dense: "语义检索",
};

const SCORE_LABELS = {
  bm25: "BM25",
  tfidf: "TF-IDF",
  cosine: "Cosine",
  rrf: "RRF",
};

const SOURCE_META = {
  news20k: { badge: "新", name: "新华网 · news.cn", tone: "news" },
  mini100k: { badge: "问", name: "DuReader 中文问答 · 100k", tone: "qa" },
  official8m: { badge: "全", name: "DuReader 全集 · 8.1M", tone: "corpus" },
};

const ROUTING_LABELS = {
  requested_mode: "请求模式",
  executed_mode: "执行模式",
  reason: "路由原因",
  score_kind: "分数类型",
  vector_used: "使用向量",
  vector_attempted: "尝试向量",
  fallback: "发生降级",
  candidate_k: "候选池",
  policy_version: "策略版本",
};

const LEADING_NOISE_RE = /^(?:\s|(?:您好|你好|亲爱的?|朋友|楼主|患者你好|医生建议|病情分析|指导意见|意见建议|问题分析|健康指导|参考答案|最佳答案|网友回答|精选回答|回答|答|问|解析|摘要|导读|正文|来源|编辑|记者)[：:，,！!、\s]+|【[^】]{1,12}】\s*[：:]?\s*|\[图\]\s*|[（(]?\d+[）)\.、．]\s*)+/u;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function setStatus(message, kind = "") {
  statusBox.textContent = message;
  statusBox.className = `status ${kind}`.trim();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function setTraceOpen(open) {
  if (traceDetails) traceDetails.hidden = !open;
  if (toggleTraceButton) {
    toggleTraceButton.setAttribute("aria-expanded", String(Boolean(open)));
    toggleTraceButton.textContent = open ? "收起路由详情 ▴" : "路由与耗时详情 ▾";
  }
}

function clearObservability() {
  summary.textContent = "";
  if (compactTimings) compactTimings.innerHTML = "";
  if (expandedTermsBox) {
    expandedTermsBox.innerHTML = "";
    expandedTermsBox.hidden = true;
  }
  routingBox.innerHTML = "";
  timingsBox.innerHTML = "";
  setTraceOpen(false);
  observability.hidden = true;
}

function clearCompare() {
  if (!comparePanel) return;
  comparePanel.innerHTML = "";
  comparePanel.hidden = true;
  state.activeView = "search";
  syncModeTabs();
}

function syncLexicalControls() {
  const isLexical = modeInput.value === "lexical";
  if (lexicalOptions) lexicalOptions.hidden = !isLexical;
  if (scorerInput) {
    scorerInput.disabled = !isLexical;
    if (!isLexical) scorerInput.value = "bm25";
  }
  if (expanderInput) {
    expanderInput.disabled = !isLexical;
    if (!isLexical) expanderInput.value = "none";
  }
}

function syncModeTabs() {
  if (!modeTabs || typeof modeTabs.querySelectorAll !== "function") return;
  const selected = currentDataset();
  const supported = new Set(selected ? selected.modes : ["lexical", "adaptive", "hybrid", "dense"]);
  const activeKey = state.activeView === "compare" ? "compare" : modeInput.value;
  modeTabs.querySelectorAll("[data-mode-tab]").forEach((tab) => {
    const key = tab.getAttribute("data-mode-tab");
    const isActive = key === activeKey;
    const isDisabled = key === "compare" ? supported.size < 2 : !supported.has(key);
    tab.disabled = isDisabled;
    tab.className = `mode-tab${key === "compare" ? " mode-tab-compare" : ""}${isActive ? " active" : ""}`;
    tab.setAttribute("aria-selected", String(isActive));
  });
}

function formatBoolean(value) {
  return value ? "是" : "否";
}

const MISSING_VALUE = "—";

function textField(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number" && !Number.isFinite(value)) return "";
  return String(value).trim();
}

function firstField(...values) {
  for (const value of values) {
    const text = textField(value);
    if (text) return text;
  }
  return "";
}

function digestField(value) {
  const text = textField(value);
  if (!text) return null;
  return text.length > 16 ? `${text.slice(0, 12)}…` : text;
}

function joinParams(params) {
  if (!params || typeof params !== "object") return "";
  return Object.entries(params)
    .filter(([, value]) => textField(value) !== "")
    .map(([key, value]) => `${key}=${textField(value)}`)
    .join(" · ");
}

function sourceMetaFor(datasetId) {
  if (SOURCE_META[datasetId]) return SOURCE_META[datasetId];
  const dataset = currentDataset();
  return { badge: "知", name: dataset ? dataset.label : "知海检索库", tone: "default" };
}

function compactDocId(docId) {
  const id = textField(docId);
  if (!id) return "";
  if (/^[0-9a-f]{24,}$/i.test(id)) return `#${id.slice(0, 8)}`;
  return id.length > 22 ? `${id.slice(0, 18)}…` : id;
}

function formatScoreBadge(score, scoreKind) {
  const num = Number(score);
  if (!Number.isFinite(num)) return "";
  const label = SCORE_LABELS[scoreKind] || "";
  const digits = scoreKind === "rrf" || scoreKind === "cosine" ? 4 : 2;
  return label ? `${label} ${num.toFixed(digits)}` : num.toFixed(4);
}

function deriveResultTitle(item) {
  const explicit = textField(item?.title);
  if (explicit) {
    return { text: explicit, isDerived: false, consumedEnd: 0 };
  }
  const raw = String(item?.snippet || item?.text || "");
  if (!raw.trim()) {
    return { text: textField(item?.doc_id) || "未命名文档", isDerived: false, consumedEnd: 0 };
  }
  const stripped = raw.replace(LEADING_NOISE_RE, "");
  const working = stripped.trim() ? stripped : raw.trimStart();
  const fullChars = Array.from(raw);
  const chars = Array.from(working);
  const leadCp = fullChars.length - chars.length;
  const limit = Math.min(chars.length, 40);
  let cut = -1;
  let keepPunct = false;

  for (let i = 6; i < limit; i += 1) {
    const ch = chars[i];
    if ("？?！!".includes(ch)) {
      cut = i + 1;
      keepPunct = true;
      break;
    }
    if ("。；;\n\r".includes(ch)) {
      cut = i;
      break;
    }
  }
  if (cut === -1) {
    for (let i = Math.min(chars.length - 1, 38); i >= 12; i -= 1) {
      if ("，,：:、".includes(chars[i])) {
        cut = i;
        break;
      }
    }
  }
  let titleText;
  let consumedInWorking;
  if (cut !== -1) {
    titleText = chars.slice(0, cut).join("").trim();
    consumedInWorking = keepPunct ? cut : cut + 1;
  } else if (chars.length > 36) {
    titleText = `${chars.slice(0, 36).join("").trim()}…`;
    consumedInWorking = 0;
  } else {
    titleText = chars.join("").trim();
    consumedInWorking = chars.length;
  }
  titleText = titleText.replace(/[，,：:、；;。]+$/u, "").trim();
  if (!titleText) {
    return { text: textField(item?.doc_id) || "未命名文档", isDerived: false, consumedEnd: 0 };
  }
  let consumedEnd = consumedInWorking > 0 ? leadCp + consumedInWorking : 0;
  if (consumedEnd > 0) {
    while (consumedEnd < fullChars.length && /[\s：:，,。；;—\-]/u.test(fullChars[consumedEnd])) {
      consumedEnd += 1;
    }
  }
  return { text: titleText, isDerived: true, consumedEnd };
}

function vectorStatus(info) {
  // vector_ready is a startup configuration statement, not a live Qdrant probe.
  const configured = (info.retrieval || {}).vector_ready;
  if (configured === true) {
    return { summary: "向量启动时已配置 · 非实时健康", detail: "启动时已配置（启动校验，非后端实时健康）", tone: "positive" };
  }
  if (configured === false) {
    return { summary: "仅词法模式", detail: "未配置（仅词法模式）", tone: "" };
  }
  return { summary: "向量状态未知", detail: "未提供", tone: "" };
}

function indexSummaryText(info) {
  const retrieval = info.retrieval || {};
  const index = retrieval.index_identity || {};
  const dataset = firstField(index.dataset, info.dataset) || "未知数据集";
  const nDocs = firstField(index.n_docs, info.n_docs) || "?";
  return `${dataset} · ${nDocs} 篇文档 · ${vectorStatus(info).summary}`;
}

function bm25Fields(bm25) {
  if (!bm25 || typeof bm25 !== "object") return "";
  const parts = [];
  if (textField(bm25.k1)) parts.push(`k1=${textField(bm25.k1)}`);
  if (textField(bm25.b)) parts.push(`b=${textField(bm25.b)}`);
  if (textField(bm25.idf)) parts.push(`idf=${textField(bm25.idf)}`);
  return parts.join(" · ");
}

function searchParamFields(search) {
  if (!search || typeof search !== "object") return "";
  if (search.exact === true && !textField(search.hnsw_ef)) return "exact=是（全量余弦扫描）";
  const parts = [];
  if (textField(search.hnsw_ef)) parts.push(`ef=${textField(search.hnsw_ef)}`);
  if (typeof search.exact === "boolean") parts.push(`exact=${search.exact ? "是" : "否"}`);
  if (textField(search.timeout_s)) parts.push(`timeout=${textField(search.timeout_s)} s`);
  return parts.join(" · ");
}

function retrievalParamFields(searchConfig, config) {
  const parts = [];
  const candidateK = firstField(searchConfig.candidate_k, config.candidate_k);
  const rrfC = firstField(searchConfig.rrf_c, config.rrf_c);
  if (candidateK) parts.push(`candidate_k=${candidateK}`);
  if (rrfC) parts.push(`RRF c=${rrfC}`);
  return parts.join(" · ");
}

function policyFields(policy, config) {
  const parts = [];
  const name = firstField(policy.name);
  const version = firstField(policy.policy_version, config.policy_version);
  if (name) parts.push(name);
  if (version) parts.push(`版本=${version}`);
  if (typeof policy.requires_statistics === "boolean") {
    parts.push(`需要统计=${policy.requires_statistics ? "是" : "否"}`);
  }
  return parts.join(" · ");
}

function identityFacts(info) {
  const retrieval = info.retrieval || {};
  const index = retrieval.index_identity || {};
  const vector = retrieval.vector_identity || retrieval.vector_manifest || {};
  const searchConfig = retrieval.search_config || {};
  const config = retrieval.config || {};
  const policy = retrieval.policy || {};
  const analyzerInfo = info.analyzer || {};
  const embedding = vector.embedding || {};
  const bm25 = index.bm25 || info.bm25 || {};
  const status = vectorStatus(info);
  const facts = [];
  const add = (label, value, wide = false) => {
    const text = textField(value);
    facts.push({ label, value: text || MISSING_VALUE, wide });
  };
  const addDigest = (label, value) => {
    const digest = digestField(value);
    facts.push(digest
      ? { label, value: digest, copy: textField(value), wide: false }
      : { label, value: MISSING_VALUE, wide: false });
  };
  facts.push({ label: "向量能力", value: status.detail, tone: status.tone, wide: false });
  add("数据集", firstField(index.dataset, info.dataset));
  add("文档数", firstField(index.n_docs, info.n_docs));
  addDigest("语料指纹", firstField(index.corpus_sha256, vector.corpus_sha256));
  add("分词器", [textField(analyzerInfo.name), textField(analyzerInfo.version)].filter(Boolean).join(" "));
  addDigest("分词器哈希", firstField(index.analyzer_hash, analyzerInfo.config_hash));
  add("BM25 参数", bm25Fields(bm25), true);
  add("向量集合", vector.collection);
  add("向量后端", vector.backend);
  add("嵌入模型", embedding.model_id);
  addDigest("模型版本", embedding.revision);
  add("向量维度", embedding.dimension);
  add("查询前缀", embedding.query_prefix, true);
  add("构建参数", joinParams(vector.index_params), true);
  add("检索 HNSW", searchParamFields(vector.search), true);
  add("候选池 / RRF", retrievalParamFields(searchConfig, config), true);
  add("路由策略", policyFields(policy, config), true);
  return facts;
}

function renderIdentityFacts(info) {
  return identityFacts(info).map((fact) => {
    const className = fact.wide ? "fact wide" : "fact";
    const valueClass = fact.tone === "positive" ? "fact-value positive" : "fact-value";
    const title = fact.copy ? ` title="${escapeHtml(fact.copy)}"` : "";
    const copy = fact.copy
      ? `<button type="button" class="copy" data-copy="${escapeHtml(fact.copy)}" aria-label="${escapeHtml(`复制${fact.label}的完整值`)}">复制</button>`
      : "";
    return `<div class="${className}"><dt>${escapeHtml(fact.label)}</dt>`
      + `<dd class="${valueClass}"${title}><span class="fact-text">${escapeHtml(fact.value)}</span>${copy}</dd></div>`;
  }).join("");
}

async function copyIdentityValue(button) {
  const value = button.getAttribute("data-copy") || "";
  try {
    await navigator.clipboard.writeText(value);
    button.textContent = "已复制";
  } catch (_) {
    button.textContent = "复制失败";
  }
}

function renderRouting(routing) {
  routingBox.innerHTML = Object.entries(ROUTING_LABELS)
    .filter(([key]) => key in routing)
    .map(([key, label]) => {
      const value = typeof routing[key] === "boolean"
        ? formatBoolean(routing[key]) : routing[key];
      const className = routing[key] === true ? " fact-value positive" : " fact-value";
      return `<div class="fact"><dt>${escapeHtml(label)}</dt><dd class="${className}">${escapeHtml(value)}</dd></div>`;
    }).join("");
}

function renderTimings(timings) {
  const labels = { lexical: "词法", router: "路由", encode: "编码", vector: "向量", fusion: "融合" };
  timingsBox.innerHTML = Object.entries(labels).map(([key, label]) => {
    const value = Number(timings[key] || 0);
    return `<span class="timing"><b>${escapeHtml(label)}</b><span>${value.toFixed(3)} ms</span></span>`;
  }).join("");
  if (compactTimings) {
    const pills = Object.entries(labels)
      .filter(([key]) => Number(timings[key] || 0) > 0.01)
      .map(([key, label]) => `<span class="compact-timing">${escapeHtml(label)} ${Number(timings[key]).toFixed(1)}ms</span>`)
      .join("");
    compactTimings.innerHTML = pills;
  }
}

function highlightedSnippet(text, highlights = []) {
  // API offsets count Unicode code points, not JavaScript UTF-16 code units.
  const source = Array.from(String(text || ""));
  const spans = highlights.map((highlight) => ({
    start: Math.max(0, Math.min(source.length, Number(highlight.start) || 0)),
    end: Math.max(0, Math.min(source.length, Number(highlight.end) || 0)),
    kind: highlight.kind === "expanded" ? "expanded" : "original",
  })).filter(({ start, end }) => end > start).sort((left, right) => left.start - right.start);
  const pieces = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start < cursor) {
      if (span.end <= cursor) continue;
      span.start = cursor;
    }
    if (span.start > cursor) pieces.push(escapeHtml(source.slice(cursor, span.start).join("")));
    pieces.push(`<mark class="${span.kind === "expanded" ? "expanded" : "original"}">${escapeHtml(source.slice(span.start, span.end).join(""))}</mark>`);
    cursor = span.end;
  }
  pieces.push(escapeHtml(source.slice(cursor).join("")));
  return pieces.join("") || "<span class=\"muted\">无摘要</span>";
}

function renderCardSnippet(item, derived) {
  const raw = String(item.snippet || "");
  const chars = Array.from(raw);
  const highlights = item.highlights;
  if (!Array.isArray(highlights)) {
    // Preserve strict validation behavior for malformed responses.
    return highlightedSnippet(raw, highlights);
  }
  let startOffset = 0;
  if (derived.isDerived && derived.consumedEnd > 0 && chars.length - derived.consumedEnd >= 35) {
    const visibleHits = highlights.filter((h) => Number(h.start) < chars.length && Number(h.end) > 0).length;
    const afterHits = highlights.filter((h) => Number(h.end) > derived.consumedEnd && Number(h.start) < chars.length).length;
    if (visibleHits === 0 || afterHits > 0) {
      startOffset = derived.consumedEnd;
    }
  }
  const slicedText = startOffset > 0 ? chars.slice(startOffset).join("") : raw;
  const shiftedHighlights = startOffset > 0
    ? highlights
      .filter((h) => Number(h.end) > startOffset)
      .map((h) => ({ start: Number(h.start) - startOffset, end: Number(h.end) - startOffset, kind: h.kind }))
    : highlights;
  const html = highlightedSnippet(slicedText, shiftedHighlights);
  const mayContinue = chars.length >= 180 && !/[。！？?!…]$/.test(raw.trim());
  return mayContinue ? `${html}<span class="snippet-ellipsis">…</span>` : html;
}

function splitParagraphRanges(chars) {
  const ranges = [];
  const n = chars.length;
  let start = 0;
  let i = 0;
  while (i < n) {
    if (chars[i] === "\n" || chars[i] === "\r") {
      if (i > start) ranges.push([start, i]);
      while (i < n && (chars[i] === "\n" || chars[i] === "\r")) i += 1;
      start = i;
      continue;
    }
    i += 1;
  }
  if (start < n) ranges.push([start, n]);

  const refined = [];
  for (const [segStart, segEnd] of ranges) {
    if (segEnd - segStart <= 280) {
      refined.push([segStart, segEnd]);
      continue;
    }
    let cur = segStart;
    while (cur < segEnd) {
      if (segEnd - cur <= 280) {
        refined.push([cur, segEnd]);
        break;
      }
      let cut = -1;
      for (let pos = cur + 140; pos < Math.min(segEnd, cur + 280); pos += 1) {
        if ("。！？?!；;".includes(chars[pos])) {
          cut = pos + 1;
          if (pos - cur >= 180) break;
        }
      }
      if (cut === -1) {
        for (let pos = cur + 280; pos < Math.min(segEnd, cur + 380); pos += 1) {
          if ("。！？?!；;".includes(chars[pos])) {
            cut = pos + 1;
            break;
          }
        }
      }
      if (cut === -1 || cut <= cur) {
        refined.push([cur, segEnd]);
        break;
      }
      refined.push([cur, cut]);
      cur = cut;
    }
  }
  return refined;
}

function renderDocumentParagraphs(text, highlights = []) {
  const chars = Array.from(String(text || ""));
  if (!chars.length) return '<p class="muted">文档内容为空</p>';
  const spans = Array.isArray(highlights) ? highlights : [];
  const paragraphs = [];
  for (let [a, b] of splitParagraphRanges(chars)) {
    while (a < b && /\s/.test(chars[a])) a += 1;
    while (b > a && /\s/.test(chars[b - 1])) b -= 1;
    if (a >= b) continue;
    const chunk = chars.slice(a, b).join("");
    const localHighlights = spans
      .filter((h) => Number(h.end) > a && Number(h.start) < b)
      .map((h) => ({ start: Number(h.start) - a, end: Number(h.end) - a, kind: h.kind }));
    paragraphs.push(`<p class="reader-paragraph">${highlightedSnippet(chunk, localHighlights)}</p>`);
  }
  return paragraphs.join("") || '<p class="muted">文档内容为空</p>';
}

function renderResults(data) {
  const results = Array.isArray(data.results) ? data.results : [];
  state.resultMap.clear();
  if (!results.length) {
    resultBox.innerHTML = `<div class="empty-state compact"><span class="empty-index">NO MATCH</span><h2>没有找到结果</h2><p>换一个关键词，或点击上方标签切换混合/语义检索模式试试。</p></div>`;
    return;
  }
  const source = sourceMetaFor(state.dataset);
  const scoreKind = data.routing?.score_kind || "";
  resultBox.innerHTML = results.map((item) => {
    const derived = deriveResultTitle(item);
    const snippetHtml = renderCardSnippet(item, derived);
    const snippetChars = Array.from(String(item.snippet || "")).length;
    const allHighlights = Array.isArray(item.highlights) ? item.highlights : [];
    const totalHits = allHighlights.length;
    const visibleHits = allHighlights.filter((h) => Number(h.start) < snippetChars).length;
    const extraHits = Math.max(0, totalHits - visibleHits);
    const hitBadge = totalHits > 0
      ? `<span class="hit-badge${extraHits > 0 ? " has-more" : ""}">${extraHits > 0
        ? `全文 ${totalHits} 处命中 · 正文另有 ${extraHits} 处`
        : `命中 ${totalHits} 处`}</span>`
      : "";
    const visited = state.visitedDocs.has(item.doc_id);
    const scoreText = formatScoreBadge(item.score, scoreKind);
    state.resultMap.set(item.doc_id, {
      item,
      derivedTitle: derived.text,
      scoreText,
      sourceName: source.name,
    });
    return `
    <article class="result-card${visited ? " is-visited" : ""}" data-card-doc="${escapeHtml(item.doc_id)}">
      <div class="result-source-row">
        <div class="result-source-left">
          <span class="source-badge tone-${escapeHtml(source.tone)}">${escapeHtml(source.badge)}</span>
          <span class="source-name">${escapeHtml(source.name)}</span>
          <span class="source-sep" aria-hidden="true">›</span>
          <span class="doc-id" title="${escapeHtml(item.doc_id)}">${escapeHtml(compactDocId(item.doc_id))}</span>
        </div>
        <div class="result-source-right">
          <span class="result-rank">#${String(item.rank).padStart(2, "0")}</span>
          <span class="score">${escapeHtml(scoreText)}</span>
        </div>
      </div>
      <div class="result-heading">
        <h2><button type="button" class="result-title-btn" data-open-doc="${escapeHtml(item.doc_id)}">${escapeHtml(derived.text)}</button></h2>
      </div>
      <p class="snippet">${snippetHtml}</p>
      <div class="result-footer">
        <button type="button" class="read-more-btn" data-open-doc="${escapeHtml(item.doc_id)}">阅读全文 →</button>
        ${hitBadge}
        <button type="button" class="copy-doc" data-copy-doc="${escapeHtml(item.doc_id)}" title="完整 ID：${escapeHtml(item.doc_id)}">复制 ID</button>
      </div>
    </article>`;
  }).join("");
}

function currentDataset() {
  return state.catalog.find((dataset) => dataset.id === state.dataset) || null;
}

function datasetEndpoint(kind) {
  if (!state.dataset) return kind === "info" ? "/index/info" : "/search";
  return `/datasets/${encodeURIComponent(state.dataset)}/${kind}`;
}

function docEndpoint(docId, query = "") {
  const encodedId = encodeURIComponent(docId);
  const qs = query ? `?q=${encodeURIComponent(query)}` : "";
  if (!state.dataset) return `/doc/${encodedId}${qs}`;
  return `/datasets/${encodeURIComponent(state.dataset)}/doc/${encodedId}${qs}`;
}

function readRecentQueries() {
  try {
    if (typeof localStorage === "undefined") return [];
    const raw = localStorage.getItem(RECENT_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((q) => typeof q === "string" && q.trim()).slice(0, RECENT_MAX) : [];
  } catch (_) {
    return [];
  }
}

function saveRecentQuery(query) {
  const clean = textField(query);
  if (!clean) return;
  try {
    if (typeof localStorage === "undefined") return;
    const next = [clean, ...readRecentQueries().filter((q) => q !== clean)].slice(0, RECENT_MAX);
    localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(next));
  } catch (_) {
    // Ignore storage quota / private mode errors.
  }
  renderRecentQueries();
}

function clearRecentQueries() {
  try {
    if (typeof localStorage !== "undefined") localStorage.removeItem(RECENT_STORAGE_KEY);
  } catch (_) {
    // Ignore storage errors.
  }
  renderRecentQueries();
}

function renderRecentQueries() {
  if (!recentQueriesBox) return;
  const list = readRecentQueries();
  if (!list.length) {
    recentQueriesBox.innerHTML = "";
    recentQueriesBox.hidden = true;
    return;
  }
  recentQueriesBox.innerHTML = '<span class="example-label">最近</span>'
    + list.map((q) => `<button type="button" class="recent-chip" data-recent="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join("")
    + '<button type="button" class="clear-recent" data-clear-recent="1">清空</button>';
  recentQueriesBox.hidden = false;
}

function closeDocumentDrawer() {
  if (!docDrawer) return;
  if (state.drawer.controller) {
    state.drawer.controller.abort();
    state.drawer.controller = null;
  }
  docDrawer.hidden = true;
  if (typeof document !== "undefined" && document.body?.classList) {
    document.body.classList.remove("drawer-open");
  }
}

function updateDrawerHitCounter() {
  if (!docDrawerNextHit) return;
  if (state.drawer.hitCount <= 0) {
    docDrawerNextHit.hidden = true;
    return;
  }
  docDrawerNextHit.hidden = false;
  const current = state.drawer.hitIndex >= 0 ? state.drawer.hitIndex + 1 : 0;
  docDrawerNextHit.textContent = current > 0
    ? `下一处命中 (${current}/${state.drawer.hitCount}) ↓`
    : `跳到命中 (共 ${state.drawer.hitCount} 处) ↓`;
}

function jumpToNextHit() {
  if (!docDrawerBody || typeof docDrawerBody.querySelectorAll !== "function") return;
  const marks = Array.from(docDrawerBody.querySelectorAll("mark"));
  state.drawer.hitCount = marks.length;
  if (!marks.length) {
    updateDrawerHitCounter();
    return;
  }
  if (state.drawer.hitIndex >= 0 && marks[state.drawer.hitIndex]?.classList) {
    marks[state.drawer.hitIndex].classList.remove("active-hit");
  }
  state.drawer.hitIndex = (state.drawer.hitIndex + 1) % marks.length;
  const target = marks[state.drawer.hitIndex];
  if (target?.classList) target.classList.add("active-hit");
  if (typeof target?.scrollIntoView === "function") {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  updateDrawerHitCounter();
}

async function openDocument(docId) {
  if (!docDrawer || !docId) return;
  const cached = state.resultMap.get(docId);
  const query = state.lastQuery || queryInput.value.trim();
  const source = sourceMetaFor(state.dataset);

  if (state.drawer.controller) state.drawer.controller.abort();
  const serial = ++state.drawer.serial;
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  state.drawer.controller = controller;
  state.drawer.docId = docId;
  state.drawer.title = cached?.derivedTitle || compactDocId(docId) || docId;
  state.drawer.text = cached?.item?.snippet || "";
  state.drawer.hitIndex = -1;
  state.drawer.hitCount = 0;

  state.visitedDocs.add(docId);
  if (typeof resultBox?.querySelectorAll === "function") {
    resultBox.querySelectorAll("[data-card-doc]").forEach((card) => {
      if (card.getAttribute("data-card-doc") === docId && card.classList) {
        card.classList.add("is-visited");
      }
    });
  }

  if (docDrawerSource) docDrawerSource.textContent = cached?.sourceName || source.name;
  if (docDrawerId) {
    docDrawerId.textContent = compactDocId(docId);
    docDrawerId.setAttribute("title", docId);
  }
  if (docDrawerTitle) docDrawerTitle.textContent = state.drawer.title;
  if (docDrawerMeta) docDrawerMeta.textContent = cached?.scoreText ? `正在加载全文… · ${cached.scoreText}` : "正在加载全文…";
  if (docDrawerNextHit) docDrawerNextHit.hidden = true;
  if (docDrawerBody) {
    docDrawerBody.innerHTML = '<div class="reader-loading"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line"></div></div>';
  }
  docDrawer.hidden = false;
  if (typeof document !== "undefined" && document.body?.classList) {
    document.body.classList.add("drawer-open");
  }

  try {
    const options = { headers: { Accept: "application/json" } };
    if (controller) options.signal = controller.signal;
    const response = await fetch(docEndpoint(docId, query), options);
    const data = await response.json().catch(() => null);
    if (serial !== state.drawer.serial) return;
    if (!response.ok || !data || typeof data.text !== "string") {
      throw new Error(data?.error || `HTTP ${response.status}`);
    }
    const derived = deriveResultTitle({ title: data.title, snippet: data.text, doc_id: data.doc_id });
    state.drawer.title = derived.text;
    state.drawer.text = data.text;
    const highlights = Array.isArray(data.highlights) ? data.highlights : [];
    const charCount = Number(data.char_count ?? Array.from(data.text).length);
    if (docDrawerTitle) docDrawerTitle.textContent = derived.text;
    const metaParts = [`共 ${charCount.toLocaleString("zh-CN")} 字`];
    if (highlights.length) metaParts.push(`命中 ${highlights.length} 处`);
    if (cached?.scoreText) metaParts.push(cached.scoreText);
    if (docDrawerMeta) docDrawerMeta.textContent = metaParts.join(" · ");
    if (docDrawerBody) {
      docDrawerBody.innerHTML = renderDocumentParagraphs(data.text, highlights);
      docDrawerBody.scrollTop = 0;
    }
    state.drawer.hitCount = highlights.length;
    state.drawer.hitIndex = -1;
    updateDrawerHitCounter();
  } catch (error) {
    if (serial !== state.drawer.serial || error?.name === "AbortError") return;
    const fallbackSnippet = cached?.item?.snippet || "";
    const fallbackHighlights = Array.isArray(cached?.item?.highlights) ? cached.item.highlights : [];
    if (docDrawerMeta) docDrawerMeta.textContent = "暂无法读取全文，已展示检索摘要";
    if (docDrawerBody) {
      docDrawerBody.innerHTML = `<div class="reader-fallback-notice">未能从索引加载完整文档（${escapeHtml(error instanceof Error ? error.message : "请求失败")}），以下为检索摘要：</div>`
        + `<p class="reader-paragraph">${highlightedSnippet(fallbackSnippet, fallbackHighlights)}</p>`;
    }
    state.drawer.hitCount = 0;
    updateDrawerHitCounter();
  } finally {
    if (serial === state.drawer.serial) state.drawer.controller = null;
  }
}

function renderExpandedTerms(expanded) {
  if (!expandedTermsBox) return;
  if (!Array.isArray(expanded) || !expanded.length) {
    expandedTermsBox.innerHTML = "";
    expandedTermsBox.hidden = true;
    return;
  }
  const chips = expanded.map((item) =>
    `<span class="expanded-chip">${escapeHtml(item.term)} <em>${Number(item.w || 0).toFixed(2)}</em></span>`).join("");
  expandedTermsBox.innerHTML = `<span class="expanded-label">扩展词 (${escapeHtml(String(expanded[0]?.src || "rm3").toUpperCase())})</span>${chips}`;
  expandedTermsBox.hidden = false;
}

function renderResponse(data, clientMs = null) {
  const routing = data.routing || {};
  const count = Array.isArray(data.results) ? data.results.length : 0;
  const dataset = currentDataset();
  const prefix = dataset ? `${dataset.label} · ` : "";
  const execLabel = MODE_LABELS[routing.executed_mode] || routing.executed_mode || "";
  const execPart = execLabel ? ` · ${execLabel}` : "";
  const network = Number.isFinite(clientMs) ? `（往返 ${clientMs.toFixed(0)} ms）` : "";
  summary.textContent = `${prefix}${count} 条结果 · 服务端 ${Number(data.took_ms || 0)} ms${network}${execPart}`;
  renderExpandedTerms(data.expanded);
  renderRouting(routing);
  renderTimings(data.timings_ms || {});
  observability.hidden = false;
  renderResults(data);
  state.resultCount = count;
  setStatus(`检索完成，返回 ${count} 条结果`, "success");
}

function renderPagination() {
  const k = Number.parseInt(kInput.value, 10) || 10;
  pagination.hidden = state.resultCount === 0 && state.offset === 0;
  previousPage.disabled = state.offset === 0;
  nextPage.disabled = state.resultCount < k || (modeInput.value !== "lexical" && state.offset + k >= 200);
  pageInfo.textContent = `第 ${Math.floor(state.offset / k) + 1} 页`;
}

function renderLoading() {
  resultBox.innerHTML = Array.from({ length: 4 }, () => '<div class="skeleton-card" aria-hidden="true"></div>').join("");
  resultBox.setAttribute("aria-busy", "true");
  pagination.hidden = true;
}

function updateUrl(push = false) {
  if (typeof window === "undefined" || !window.history) return;
  const url = new URL(window.location.href);
  if (state.dataset) url.searchParams.set("dataset", state.dataset);
  else url.searchParams.delete("dataset");
  const query = queryInput.value.trim();
  if (query) url.searchParams.set("q", query);
  else url.searchParams.delete("q");
  url.searchParams.set("mode", modeInput.value);
  url.searchParams.set("k", kInput.value);
  if (state.offset) url.searchParams.set("offset", String(state.offset));
  else url.searchParams.delete("offset");
  const method = push ? "pushState" : "replaceState";
  window.history[method]({ dataset: state.dataset }, "", url);
}

async function executeSearch(offset = 0) {
  const query = queryInput.value.trim();
  if (!query) {
    showError("请输入搜索内容");
    queryInput.focus();
    return;
  }
  const parsedK = Number.parseInt(kInput.value, 10);
  const k = Number.isInteger(parsedK) ? Math.max(1, Math.min(200, parsedK)) : 10;
  kInput.value = String(k);
  state.lastQuery = query;
  hideError();
  clearObservability();
  clearCompare();
  resultBox.hidden = false;
  if (state.controller) state.controller.abort();
  const serial = ++state.requestSerial;
  const requestedDataset = state.dataset;
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  state.controller = controller;
  submitButton.disabled = true;
  if (compareButton) compareButton.disabled = true;
  setStatus(`正在检索${currentDataset() ? `「${currentDataset().label}」` : "…"}`);
  renderLoading();
  const started = Date.now();
  try {
    const payload = { query, mode: modeInput.value, k, offset };
    if (modeInput.value === "lexical") {
      if (scorerInput && scorerInput.value && scorerInput.value !== "bm25") payload.scorer = scorerInput.value;
      if (expanderInput && expanderInput.value && expanderInput.value !== "none") payload.expander = expanderInput.value;
    }
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    };
    if (controller) options.signal = controller.signal;
    const response = await fetch(datasetEndpoint("search"), options);
    const data = await response.json().catch(() => null);
    if (serial !== state.requestSerial || requestedDataset !== state.dataset) return;
    if (!response.ok) throw new Error(data?.error || `请求失败（HTTP ${response.status}）`);
    if (!Array.isArray(data?.results)) throw new Error("搜索响应格式无效");
    state.offset = offset;
    saveRecentQuery(query);
    renderResponse(data, Date.now() - started);
    renderPagination();
    updateUrl(false);
  } catch (error) {
    if (serial !== state.requestSerial || error?.name === "AbortError") return;
    resultBox.innerHTML = "";
    resultBox.setAttribute("aria-busy", "false");
    pagination.hidden = true;
    clearObservability();
    setStatus("检索失败", "failure");
    showError(error instanceof Error ? error.message : "请求失败，请稍后重试");
  } finally {
    if (serial === state.requestSerial) {
      submitButton.disabled = false;
      syncModeTabs();
      resultBox.setAttribute("aria-busy", "false");
      state.controller = null;
    }
  }
}

async function compareAllModes() {
  if (!comparePanel) return;
  const query = queryInput.value.trim();
  if (!query) {
    showError("请先输入查询内容，再执行四模式对比");
    queryInput.focus();
    return;
  }
  const dataset = currentDataset();
  const modes = (dataset && Array.isArray(dataset.modes) && dataset.modes.length)
    ? ["lexical", "dense", "hybrid", "adaptive"].filter((m) => dataset.modes.includes(m))
    : ["lexical", "dense", "hybrid", "adaptive"];
  state.lastQuery = query;
  state.activeView = "compare";
  syncModeTabs();
  hideError();
  clearObservability();
  pagination.hidden = true;
  if (state.controller) state.controller.abort();
  const serial = ++state.requestSerial;
  const requestedDataset = state.dataset;
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  state.controller = controller;
  submitButton.disabled = true;
  if (compareButton) compareButton.disabled = true;
  setStatus(`正在并行对比 ${modes.length} 种检索模式…`);
  comparePanel.hidden = false;
  comparePanel.innerHTML = '<div class="compare-loading">正在并行执行各检索模式，请稍候…</div>';
  resultBox.hidden = true;
  try {
    const entries = await Promise.all(modes.map(async (mode) => {
      const started = Date.now();
      const options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode, k: 5, offset: 0 }),
      };
      if (controller) options.signal = controller.signal;
      const response = await fetch(datasetEndpoint("search"), options);
      const data = await response.json().catch(() => null);
      return { mode, ok: response.ok, status: response.status, data, clientMs: Date.now() - started };
    }));
    if (serial !== state.requestSerial || requestedDataset !== state.dataset) return;
    saveRecentQuery(query);
    const source = sourceMetaFor(state.dataset);
    const docFreq = new Map();
    for (const entry of entries) {
      if (entry.ok && Array.isArray(entry.data?.results)) {
        for (const item of entry.data.results) {
          docFreq.set(item.doc_id, (docFreq.get(item.doc_id) || 0) + 1);
        }
      }
    }
    const cards = entries.map((entry) => {
      const label = MODE_LABELS[entry.mode] || entry.mode;
      if (!entry.ok || !Array.isArray(entry.data?.results)) {
        const msg = entry.data?.error || `HTTP ${entry.status}`;
        return `<article class="compare-col"><header class="compare-col-head"><div><span class="compare-mode-tag">${escapeHtml(entry.mode.toUpperCase())}</span><h3>${escapeHtml(label)}</h3></div><span class="compare-ms">不可用</span></header><p class="compare-err">${escapeHtml(msg)}</p></article>`;
      }
      const routing = entry.data.routing || {};
      const timings = entry.data.timings_ms || {};
      const scoreKind = routing.score_kind || "";
      const items = entry.data.results.map((item) => {
        const unique = (docFreq.get(item.doc_id) || 0) === 1;
        const derived = deriveResultTitle(item);
        const scoreText = formatScoreBadge(item.score, scoreKind);
        if (!state.resultMap.has(item.doc_id)) {
          state.resultMap.set(item.doc_id, { item, derivedTitle: derived.text, scoreText, sourceName: source.name });
        }
        return `<li class="compare-item" data-open-doc="${escapeHtml(item.doc_id)}">
          <div class="compare-item-top">
            <span class="compare-rank">#${item.rank}</span>
            <button type="button" class="compare-title-btn" data-open-doc="${escapeHtml(item.doc_id)}">${escapeHtml(derived.text)}</button>
            <span class="compare-score">${Number(item.score).toFixed(4)}</span>
          </div>
          <p class="compare-snippet">${renderCardSnippet(item, derived)}</p>
          <div class="compare-item-foot">
            <span title="${escapeHtml(item.doc_id)}">${escapeHtml(compactDocId(item.doc_id))}</span>
            <span>${unique ? '<span class="compare-unique">独有命中</span>' : ""}<span class="compare-open-hint">阅读全文 →</span></span>
          </div>
        </li>`;
      }).join("");
      const stageBits = [
        timings.lexical ? `词法 ${Number(timings.lexical).toFixed(1)}` : "",
        timings.encode ? `编码 ${Number(timings.encode).toFixed(1)}` : "",
        timings.vector ? `向量 ${Number(timings.vector).toFixed(1)}` : "",
      ].filter(Boolean).join(" · ");
      return `<article class="compare-col">
        <header class="compare-col-head">
          <div>
            <span class="compare-mode-tag">${escapeHtml(entry.mode.toUpperCase())}</span>
            <h3>${escapeHtml(label)}</h3>
          </div>
          <span class="compare-ms">${Number(entry.data.took_ms || 0)} ms</span>
        </header>
        <div class="compare-meta">
          <span>执行: <b>${escapeHtml(MODE_LABELS[routing.executed_mode] || routing.executed_mode || entry.mode)}</b></span>
          <span>原因: ${escapeHtml(routing.reason || "—")}</span>
        </div>
        ${stageBits ? `<div class="compare-stages">${escapeHtml(stageBits)} ms</div>` : ""}
        <ol class="compare-list">${items || '<li class="compare-empty">无命中结果</li>'}</ol>
      </article>`;
    }).join("");
    comparePanel.innerHTML = `
      <div class="compare-toolbar">
        <div>
          <p class="panel-label">SIDE-BY-SIDE COMPARISON</p>
          <h2>「${escapeHtml(query)}」四模式并行对比（Top-5 · 点击任意卡片读全文）</h2>
        </div>
        <button type="button" id="close-compare" class="page-button">返回单模式视图</button>
      </div>
      <div class="compare-grid">${cards}</div>`;
    setStatus(`已完成 ${entries.length} 种检索模式并行对比`, "success");
  } catch (error) {
    if (serial !== state.requestSerial || error?.name === "AbortError") return;
    clearCompare();
    resultBox.hidden = false;
    setStatus("对比失败", "failure");
    showError(error instanceof Error ? error.message : "对比请求失败");
  } finally {
    if (serial === state.requestSerial) {
      submitButton.disabled = false;
      syncModeTabs();
      state.controller = null;
    }
  }
}

async function search(event) {
  event.preventDefault();
  state.offset = 0;
  updateUrl(true);
  await executeSearch(0);
}

async function loadIndexInfo() {
  const requestedDataset = state.dataset;
  try {
    const response = await fetch(datasetEndpoint("info"), { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("index info unavailable");
    const data = await response.json();
    if (requestedDataset !== state.dataset) return;
    state.info = data;
    indexSummary.textContent = indexSummaryText(data);
    identityFactsBox.innerHTML = renderIdentityFacts(data);
    indexDetails.hidden = false;
  } catch (_) {
    if (requestedDataset !== state.dataset) return;
    indexSummary.textContent = "索引信息暂不可用";
    identityFactsBox.innerHTML = "";
    indexDetails.hidden = true;
  }
}

function scaleLabel(nDocs) {
  if (nDocs >= 1_000_000) return `${(nDocs / 1_000_000).toFixed(1)}M`;
  if (nDocs >= 1000) return `${Math.round(nDocs / 1000)}K`;
  return String(nDocs);
}

function renderCatalog() {
  if (!state.catalog.length) {
    datasetPicker.innerHTML = '<span class="dataset-loading">数据源目录暂不可用</span>';
    return;
  }
  datasetPicker.innerHTML = state.catalog.map((dataset) => {
    const selected = dataset.id === state.dataset;
    return `<button type="button" class="dataset-card${selected ? " selected" : ""}" data-dataset="${escapeHtml(dataset.id)}" aria-pressed="${selected}">
      <span class="dataset-card-top"><span class="dataset-name">${escapeHtml(dataset.label)}</span><span class="dataset-scale">${escapeHtml(scaleLabel(dataset.n_docs))}</span></span>
      <span class="dataset-card-description">${escapeHtml(dataset.description)}</span>
      <span class="dataset-card-meta">${Number(dataset.n_docs).toLocaleString("zh-CN")} 篇 · ${dataset.modes.length} 种模式</span>
    </button>`;
  }).join("");
  const selected = currentDataset();
  datasetDescription.textContent = selected ? selected.description : "请选择数据源";
  exampleQueries.innerHTML = selected ? '<span class="example-label">试试</span>' + selected.examples.map((example) =>
    `<button type="button" class="example-query" data-example="${escapeHtml(example)}">${escapeHtml(example)}</button>`).join("") : "";
  const supported = new Set(selected ? selected.modes : ["lexical"]);
  Array.from(modeInput.options).forEach((option) => { option.disabled = !supported.has(option.value); });
  if (!supported.has(modeInput.value)) modeInput.value = supported.has("adaptive") ? "adaptive" : "lexical";
  syncLexicalControls();
  syncModeTabs();
  if (compareButton) compareButton.disabled = supported.size < 2;
}

async function selectDataset(datasetId, { initial = false, runQuery = false, offset = 0 } = {}) {
  if (!state.catalog.some((dataset) => dataset.id === datasetId)) return;
  closeDocumentDrawer();
  if (state.controller) state.controller.abort();
  state.requestSerial += 1;
  state.controller = null;
  submitButton.disabled = false;
  state.dataset = datasetId;
  state.offset = 0;
  state.resultCount = 0;
  state.info = null;
  state.resultMap.clear();
  renderCatalog();
  clearObservability();
  clearCompare();
  resultBox.hidden = false;
  hideError();
  pagination.hidden = true;
  resultBox.setAttribute("aria-busy", "false");
  resultBox.innerHTML = `<div class="empty-state"><span class="empty-index">READY</span><h2>已选择 ${escapeHtml(currentDataset().label)}</h2><p>输入问题或点击上方示例开始检索，点击结果即可展开高亮全文。</p></div>`;
  setStatus(initial ? "" : `已切换到 ${currentDataset().label}`);
  updateUrl(false);
  await loadIndexInfo();
  if (runQuery && queryInput.value.trim()) await executeSearch(offset);
}

async function loadCatalog() {
  const response = await fetch("/datasets", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error("dataset catalog unavailable");
  const data = await response.json();
  if (!Array.isArray(data?.datasets) || !data.datasets.length) throw new Error("invalid dataset catalog");
  state.catalog = data.datasets;
  const requested = typeof window === "undefined" ? "" : new URL(window.location.href).searchParams.get("dataset");
  const selected = state.catalog.some((row) => row.id === requested) ? requested : data.default_dataset;
  await selectDataset(selected, { initial: true, runQuery: Boolean(queryInput.value.trim()), offset: state.requestedOffset });
}

function setServiceState(kind, text) {
  serviceState.className = `service-state ${kind}`.trim();
  serviceState.innerHTML = `<span class="status-dot"></span>${escapeHtml(text)}`;
}

async function checkGateway() {
  try {
    const response = await fetch("/gateway/health");
    if (!response.ok) throw new Error("offline");
    setServiceState("online", "检索服务在线");
  } catch (_) {
    setServiceState("offline", "检索服务离线");
  }
}

function applyLocation() {
  if (typeof window === "undefined") return "";
  const params = new URL(window.location.href).searchParams;
  const query = params.get("q") || "";
  queryInput.value = query;
  if (["lexical", "dense", "hybrid", "adaptive"].includes(params.get("mode"))) modeInput.value = params.get("mode");
  if (["10", "20", "50"].includes(params.get("k"))) kInput.value = params.get("k");
  const parsedOffset = Number.parseInt(params.get("offset") || "0", 10);
  state.requestedOffset = Number.isInteger(parsedOffset) && parsedOffset >= 0 ? parsedOffset : 0;
  syncLexicalControls();
  syncModeTabs();
  return { query, offset: state.requestedOffset };
}

async function copySearchUrl() {
  updateUrl(false);
  const original = copyUrlButton.textContent;
  try {
    await navigator.clipboard.writeText(window.location.href);
    copyUrlButton.textContent = "链接已复制";
  } catch (_) {
    copyUrlButton.textContent = "复制失败";
  }
  setTimeout(() => { copyUrlButton.textContent = original; }, 1600);
}

function changePage(direction) {
  const k = Number.parseInt(kInput.value, 10) || 10;
  const offset = Math.max(0, state.offset + direction * k);
  return executeSearch(offset).then(() => {
    if (typeof resultBox.scrollIntoView === "function") resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function renderBenchmarks(data) {
  const cards = Array.isArray(data.cards) ? data.cards : [];
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const qualityRows = Array.isArray(data.quality_rows) ? data.quality_rows : [];
  const qualityHtml = qualityRows.length
    ? `<h3 class="benchmark-subtitle">多数据集检索质量与 Router 评测总览</h3>
      <table class="benchmark-table"><thead><tr><th>评测集与方案</th><th>nDCG@10</th><th>Recall@10</th><th>MRR@10</th><th>备注</th></tr></thead>
      <tbody>${qualityRows.map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.ndcg)}</td><td>${escapeHtml(row.recall)}</td><td>${escapeHtml(row.mrr)}</td><td>${escapeHtml(row.note || "")}</td></tr>`).join("")}</tbody></table>`
    : "";
  benchmarkContent.innerHTML = `<div class="benchmark-grid">${cards.map((card) => `
    <article class="benchmark-card"><span class="benchmark-card-label">${escapeHtml(card.label)}</span>
    <strong class="benchmark-value">${escapeHtml(card.value)}<small>${escapeHtml(card.unit || "")}</small></strong>
    <p>${escapeHtml(card.detail || "")}</p></article>`).join("")}</div>
    ${qualityHtml}
    <h3 class="benchmark-subtitle">8M 在线延迟优化与并发实测</h3>
    <table class="benchmark-table"><thead><tr><th>实验方案</th><th>p50</th><th>p95</th><th>相对基线</th></tr></thead>
    <tbody>${rows.map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.p50)}</td><td>${escapeHtml(row.p95)}</td><td>${escapeHtml(row.speedup)}</td></tr>`).join("")}</tbody></table>`;
  const noteEl = document.querySelector(".performance-note");
  if (data.note && noteEl) noteEl.textContent = data.note;
}

async function loadBenchmarks() {
  try {
    const response = await fetch("/static/benchmark-summary.json");
    if (!response.ok) throw new Error("no benchmark summary");
    renderBenchmarks(await response.json());
  } catch (_) {
    benchmarkContent.innerHTML = '<p class="benchmark-placeholder">实验数据整理中，检索功能可正常使用。</p>';
  }
}

function selectModeTab(mode) {
  if (mode === "compare") {
    compareAllModes();
    return;
  }
  if (!["lexical", "adaptive", "hybrid", "dense"].includes(mode)) return;
  const opt = Array.from(modeInput.options || []).find((item) => item.value === mode);
  if (opt && opt.disabled) return;
  modeInput.value = mode;
  state.activeView = "search";
  state.offset = 0;
  syncLexicalControls();
  syncModeTabs();
  updateUrl(false);
  if (queryInput.value.trim()) {
    executeSearch(0);
  }
}

async function initialize() {
  applyLocation();
  renderRecentQueries();
  setServiceState("", "正在连接");
  await Promise.all([checkGateway(), loadBenchmarks()]);
  try {
    await loadCatalog();
  } catch (_) {
    renderCatalog();
    setServiceState("offline", "数据源不可用");
    indexSummary.textContent = "索引信息暂不可用";
    showError("数据源目录暂时不可用，请稍后刷新页面。检索服务恢复后可继续使用。");
  }
}

form.addEventListener("submit", search);
identityFactsBox.addEventListener("click", (event) => {
  const target = event.target;
  const button = target && typeof target.closest === "function" ? target.closest("[data-copy]") : null;
  if (button) copyIdentityValue(button);
});
datasetPicker.addEventListener("click", (event) => {
  const button = event.target?.closest?.("[data-dataset]");
  if (button && button.getAttribute("data-dataset") !== state.dataset) {
    const hasQuery = Boolean(queryInput.value.trim());
    selectDataset(button.getAttribute("data-dataset"), { runQuery: hasQuery, offset: 0 });
  }
});
exampleQueries.addEventListener("click", (event) => {
  const button = event.target?.closest?.("[data-example]");
  if (!button) return;
  queryInput.value = button.getAttribute("data-example") || "";
  state.offset = 0;
  updateUrl(true);
  if (state.activeView === "compare") compareAllModes();
  else executeSearch(0);
});
if (recentQueriesBox) {
  recentQueriesBox.addEventListener("click", (event) => {
    if (event.target?.closest?.("[data-clear-recent]")) {
      clearRecentQueries();
      return;
    }
    const chip = event.target?.closest?.("[data-recent]");
    if (!chip) return;
    queryInput.value = chip.getAttribute("data-recent") || "";
    state.offset = 0;
    updateUrl(true);
    executeSearch(0);
  });
}
if (modeTabs) {
  modeTabs.addEventListener("click", (event) => {
    const tab = event.target?.closest?.("[data-mode-tab]");
    if (!tab || tab.disabled) return;
    selectModeTab(tab.getAttribute("data-mode-tab"));
  });
}
if (toggleTraceButton) {
  toggleTraceButton.addEventListener("click", () => {
    const isOpen = traceDetails ? !traceDetails.hidden : false;
    setTraceOpen(!isOpen);
  });
}
resultBox.addEventListener("click", (event) => {
  const copyBtn = event.target?.closest?.("[data-copy-doc]");
  if (copyBtn) {
    event.stopPropagation();
    const original = copyBtn.textContent;
    navigator.clipboard.writeText(copyBtn.getAttribute("data-copy-doc") || "")
      .then(() => { copyBtn.textContent = "已复制"; })
      .catch(() => { copyBtn.textContent = "复制失败"; })
      .finally(() => { setTimeout(() => { copyBtn.textContent = original; }, 1400); });
    return;
  }
  const openBtn = event.target?.closest?.("[data-open-doc]");
  if (openBtn) {
    openDocument(openBtn.getAttribute("data-open-doc"));
    return;
  }
  // Allow clicking anywhere on a result card (unless the user is selecting text)
  const selection = typeof window !== "undefined" && typeof window.getSelection === "function"
    ? String(window.getSelection() || "") : "";
  if (selection.trim()) return;
  const card = event.target?.closest?.("[data-card-doc]");
  if (card) openDocument(card.getAttribute("data-card-doc"));
});
previousPage.addEventListener("click", () => changePage(-1));
nextPage.addEventListener("click", () => changePage(1));
copyUrlButton.addEventListener("click", copySearchUrl);
modeInput.addEventListener("change", () => {
  syncLexicalControls();
  syncModeTabs();
  updateUrl(false);
});
if (scorerInput) scorerInput.addEventListener("change", () => {
  updateUrl(false);
  if (queryInput.value.trim() && modeInput.value === "lexical") executeSearch(0);
});
if (expanderInput) expanderInput.addEventListener("change", () => {
  updateUrl(false);
  if (queryInput.value.trim() && modeInput.value === "lexical") executeSearch(0);
});
if (comparePanel) {
  comparePanel.addEventListener("click", (event) => {
    if (event.target?.closest?.("#close-compare")) {
      clearCompare();
      resultBox.hidden = false;
      return;
    }
    const openEl = event.target?.closest?.("[data-open-doc]");
    if (openEl) openDocument(openEl.getAttribute("data-open-doc"));
  });
}
kInput.addEventListener("change", () => {
  state.offset = 0;
  updateUrl(false);
  if (queryInput.value.trim() && state.activeView === "search") executeSearch(0);
});

if (docDrawerBackdrop) docDrawerBackdrop.addEventListener("click", closeDocumentDrawer);
if (docDrawerClose) docDrawerClose.addEventListener("click", closeDocumentDrawer);
if (docDrawerCloseFoot) docDrawerCloseFoot.addEventListener("click", closeDocumentDrawer);
if (docDrawerNextHit) docDrawerNextHit.addEventListener("click", jumpToNextHit);
if (docDrawerCopyId) {
  docDrawerCopyId.addEventListener("click", () => {
    const original = docDrawerCopyId.textContent;
    navigator.clipboard.writeText(state.drawer.docId || "")
      .then(() => { docDrawerCopyId.textContent = "已复制"; })
      .catch(() => { docDrawerCopyId.textContent = "失败"; })
      .finally(() => { setTimeout(() => { docDrawerCopyId.textContent = original; }, 1400); });
  });
}
if (docDrawerCopy) {
  docDrawerCopy.addEventListener("click", () => {
    const original = docDrawerCopy.textContent;
    navigator.clipboard.writeText(state.drawer.text || "")
      .then(() => { docDrawerCopy.textContent = "正文已复制"; })
      .catch(() => { docDrawerCopy.textContent = "复制失败"; })
      .finally(() => { setTimeout(() => { docDrawerCopy.textContent = original; }, 1600); });
  });
}
if (docDrawerResearch) {
  docDrawerResearch.addEventListener("click", () => {
    const nextQuery = state.drawer.title.replace(/…$/, "").trim();
    if (!nextQuery) return;
    closeDocumentDrawer();
    queryInput.value = nextQuery;
    state.offset = 0;
    updateUrl(true);
    executeSearch(0);
  });
}

if (typeof window !== "undefined") {
  window.addEventListener("popstate", () => {
    const location = applyLocation();
    const requested = new URL(window.location.href).searchParams.get("dataset");
    if (requested && requested !== state.dataset && state.catalog.some((row) => row.id === requested)) {
      selectDataset(requested, { runQuery: true, offset: location.offset });
    } else if (queryInput.value.trim()) {
      executeSearch(location.offset);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && docDrawer && !docDrawer.hidden) {
      event.preventDefault();
      closeDocumentDrawer();
      return;
    }
    const tag = event.target?.tagName?.toLowerCase();
    if (event.key === "/" && !["input", "textarea", "select"].includes(tag)
        && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      queryInput.focus();
    }
  });
  initialize();
}
