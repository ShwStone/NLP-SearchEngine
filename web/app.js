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
  // A failed request must not leave the previous query's routing facts visible.
  summary.textContent = "";
  routingBox.innerHTML = "";
  timingsBox.innerHTML = "";
  observability.hidden = true;
}

function formatBoolean(value) {
  return value ? "是" : "否";
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
    resultBox.innerHTML = `<div class="empty-state compact"><h2>没有找到结果</h2><p>可以尝试更换关键词或检索模式。</p></div>`;
    return;
  }
  resultBox.innerHTML = results.map((item) => `
    <article class="result-card">
      <div class="result-rank">${escapeHtml(item.rank)}</div>
      <div class="result-content">
        <div class="result-heading">
          <h2>${escapeHtml(item.title || item.doc_id)}</h2>
          <span class="score">${Number(item.score).toFixed(4)}</span>
        </div>
        <p class="doc-id">${escapeHtml(item.doc_id)}</p>
        <p class="snippet">${highlightedSnippet(item.snippet, item.highlights)}</p>
      </div>
    </article>`).join("");
}

function renderResponse(data) {
  const routing = data.routing || {};
  const count = Array.isArray(data.results) ? data.results.length : 0;
  summary.textContent = `${count} 条结果 · 总耗时 ${Number(data.took_ms || 0)} ms · ${data.api_version || "API"}`;
  renderRouting(routing);
  renderTimings(data.timings_ms || {});
  observability.hidden = false;
  renderResults(data);
  setStatus(`检索完成，返回 ${count} 条结果`, "success");
}

async function search(event) {
  event.preventDefault();
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
  submitButton.disabled = true;
  setStatus("正在检索…");
  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode: modeInput.value, k }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
    renderResponse(data);
  } catch (error) {
    resultBox.innerHTML = "";
    clearObservability();
    setStatus("检索失败", "failure");
    showError(error instanceof Error ? error.message : "请求失败，请稍后重试");
  } finally {
    submitButton.disabled = false;
  }
}

async function loadIndexInfo() {
  try {
    const response = await fetch("/index/info");
    if (!response.ok) throw new Error("index info unavailable");
    const data = await response.json();
    const retrieval = data.retrieval || {};
    const vectorState = retrieval.vector_ready ? "向量已就绪" : "词法模式可用";
    indexSummary.textContent = `${data.dataset || "index"} · ${data.n_docs ?? "?"} 篇文档 · ${vectorState}`;
  } catch (_) {
    indexSummary.textContent = "索引信息暂不可用";
  }
}

form.addEventListener("submit", search);
loadIndexInfo();
