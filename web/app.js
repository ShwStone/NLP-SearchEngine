const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeInput = document.querySelector("#mode");
const kInput = document.querySelector("#k");
const submitButton = document.querySelector("#submit");
const statusBox = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const resultBox = document.querySelector("#results");
const observability = document.querySelector("#observability");
const summary = document.querySelector("#summary");
const routingBox = document.querySelector("#routing");
const timingsBox = document.querySelector("#timings");
const indexSummary = document.querySelector("#index-summary");
const indexDetails = document.querySelector("#index-identity");
const identityFactsBox = document.querySelector("#identity-facts");
const datasetPicker = document.querySelector("#dataset-picker");
const datasetDescription = document.querySelector("#dataset-description");
const exampleQueries = document.querySelector("#example-queries");
const pagination = document.querySelector("#pagination");
const previousPage = document.querySelector("#prev-page");
const nextPage = document.querySelector("#next-page");
const pageInfo = document.querySelector("#page-info");
const copyUrlButton = document.querySelector("#copy-url");
const serviceState = document.querySelector("#service-state");
const benchmarkContent = document.querySelector("#benchmark-content");

const state = {
  catalog: [],
  dataset: "",
  offset: 0,
  resultCount: 0,
  requestedOffset: 0,
  requestSerial: 0,
  controller: null,
  info: null,
};

const MODE_LABELS = {
  lexical: "词法检索",
  adaptive: "自适应检索",
  hybrid: "混合召回",
  dense: "语义检索",
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

function clearObservability() {
  summary.textContent = "";
  routingBox.innerHTML = "";
  timingsBox.innerHTML = "";
  observability.hidden = true;
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

function renderResults(data) {
  const results = Array.isArray(data.results) ? data.results : [];
  if (!results.length) {
    resultBox.innerHTML = `<div class="empty-state compact"><span class="empty-index">NO MATCH</span><h2>没有找到结果</h2><p>换一个关键词，或试试其他数据源与检索模式。</p></div>`;
    return;
  }
  resultBox.innerHTML = results.map((item) => `
    <article class="result-card">
      <div class="result-rank">${String(item.rank).padStart(2, "0")}</div>
      <div class="result-content">
        <div class="result-heading">
          <h2>${escapeHtml(item.title || item.doc_id)}</h2>
          <span class="score">${Number(item.score).toFixed(4)}</span>
        </div>
        <div class="result-meta">
          <span class="doc-id">${escapeHtml(item.doc_id)}</span>
          <button type="button" class="copy-doc" data-copy-doc="${escapeHtml(item.doc_id)}">复制 ID</button>
        </div>
        <p class="snippet">${highlightedSnippet(item.snippet, item.highlights)}</p>
      </div>
    </article>`).join("");
}

function currentDataset() {
  return state.catalog.find((dataset) => dataset.id === state.dataset) || null;
}

function datasetEndpoint(kind) {
  if (!state.dataset) return kind === "info" ? "/index/info" : "/search";
  return `/datasets/${encodeURIComponent(state.dataset)}/${kind}`;
}

function renderResponse(data, clientMs = null) {
  const routing = data.routing || {};
  const count = Array.isArray(data.results) ? data.results.length : 0;
  const dataset = currentDataset();
  const prefix = dataset ? `${dataset.label} · ` : "";
  const network = Number.isFinite(clientMs) ? ` · 客户端 ${clientMs.toFixed(0)} ms` : "";
  summary.textContent = `${prefix}${count} 条结果 · 服务端 ${Number(data.took_ms || 0)} ms${network} · ${data.api_version || "API"}`;
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
  resultBox.innerHTML = Array.from({ length: 3 }, () => '<div class="skeleton-card" aria-hidden="true"></div>').join("");
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
  hideError();
  clearObservability();
  if (state.controller) state.controller.abort();
  const serial = ++state.requestSerial;
  const requestedDataset = state.dataset;
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  state.controller = controller;
  submitButton.disabled = true;
  setStatus(`正在检索${currentDataset() ? `「${currentDataset().label}」` : "…"}`);
  renderLoading();
  const started = Date.now();
  try {
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode: modeInput.value, k, offset }),
    };
    if (controller) options.signal = controller.signal;
    const response = await fetch(datasetEndpoint("search"), options);
    const data = await response.json().catch(() => null);
    if (serial !== state.requestSerial || requestedDataset !== state.dataset) return;
    if (!response.ok) throw new Error(data?.error || `请求失败（HTTP ${response.status}）`);
    if (!Array.isArray(data?.results)) throw new Error("搜索响应格式无效");
    state.offset = offset;
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
      resultBox.setAttribute("aria-busy", "false");
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
  if (!supported.has(modeInput.value)) modeInput.value = "lexical";
}

async function selectDataset(datasetId, { initial = false, runQuery = false, offset = 0 } = {}) {
  if (!state.catalog.some((dataset) => dataset.id === datasetId)) return;
  if (state.controller) state.controller.abort();
  state.requestSerial += 1;
  state.controller = null;
  submitButton.disabled = false;
  state.dataset = datasetId;
  state.offset = 0;
  state.resultCount = 0;
  state.info = null;
  renderCatalog();
  clearObservability();
  hideError();
  pagination.hidden = true;
  resultBox.setAttribute("aria-busy", "false");
  resultBox.innerHTML = `<div class="empty-state"><span class="empty-index">READY</span><h2>已选择 ${escapeHtml(currentDataset().label)}</h2><p>输入问题开始检索，结果不会与其他数据源混合。</p></div>`;
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
  benchmarkContent.innerHTML = `<div class="benchmark-grid">${cards.map((card) => `
    <article class="benchmark-card"><span class="benchmark-card-label">${escapeHtml(card.label)}</span>
    <strong class="benchmark-value">${escapeHtml(card.value)}<small>${escapeHtml(card.unit || "")}</small></strong>
    <p>${escapeHtml(card.detail || "")}</p></article>`).join("")}</div>
    <table class="benchmark-table"><thead><tr><th>实验方案</th><th>p50</th><th>p95</th><th>相对基线</th></tr></thead>
    <tbody>${rows.map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.p50)}</td><td>${escapeHtml(row.p95)}</td><td>${escapeHtml(row.speedup)}</td></tr>`).join("")}</tbody></table>`;
  if (data.note) document.querySelector(".performance-note").textContent = data.note;
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

async function initialize() {
  applyLocation();
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
  if (button && button.getAttribute("data-dataset") !== state.dataset) selectDataset(button.getAttribute("data-dataset"));
});
exampleQueries.addEventListener("click", (event) => {
  const button = event.target?.closest?.("[data-example]");
  if (!button) return;
  queryInput.value = button.getAttribute("data-example") || "";
  state.offset = 0;
  updateUrl(true);
  executeSearch(0);
});
resultBox.addEventListener("click", (event) => {
  const button = event.target?.closest?.("[data-copy-doc]");
  if (!button) return;
  const original = button.textContent;
  navigator.clipboard.writeText(button.getAttribute("data-copy-doc") || "")
    .then(() => { button.textContent = "已复制"; })
    .catch(() => { button.textContent = "复制失败"; })
    .finally(() => { setTimeout(() => { button.textContent = original; }, 1400); });
});
previousPage.addEventListener("click", () => changePage(-1));
nextPage.addEventListener("click", () => changePage(1));
copyUrlButton.addEventListener("click", copySearchUrl);
modeInput.addEventListener("change", () => updateUrl(false));
kInput.addEventListener("change", () => { state.offset = 0; updateUrl(false); });

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
    const tag = event.target?.tagName?.toLowerCase();
    if (event.key === "/" && !["input", "textarea", "select"].includes(tag)
        && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      queryInput.focus();
    }
  });
  initialize();
}
