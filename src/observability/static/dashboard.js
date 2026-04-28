const state = {
  summary: null,
  requests: [],
  failures: [],
  toolCalls: [],
};

const el = (id) => document.getElementById(id);

function fmtInt(value) {
  return new Intl.NumberFormat().format(Math.round(Number(value || 0)));
}

function fmtMs(value) {
  if (value === null || value === undefined) return "0 ms";
  return `${fmtInt(value)} ms`;
}

function fmtMoney(value, currency = "USD") {
  if (value === null || value === undefined) return "not configured";
  const prefix = currency === "USD" ? "$" : `${currency} `;
  return `${prefix}${Number(value).toFixed(6)}`;
}

function fmtRate(value) {
  if (value === null || value === undefined) return "not configured";
  return `${Number(value).toFixed(1)} tok/s`;
}

function fmtTime(value) {
  if (!value) return "";
  return new Date(value).toLocaleString();
}

function statusPill(status) {
  const cls = status === "success" ? "status" : "status error";
  return `<span class="${cls}">${escapeHtml(status || "unknown")}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function firstCurrency(summary) {
  return summary.pricing?.find((price) => price.currency)?.currency || "USD";
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function refresh() {
  const hours = el("windowSelect").value;
  const [summary, requests, failures, toolCalls] = await Promise.all([
    fetchJson(`/api/observability/summary?hours=${hours}`),
    fetchJson("/api/observability/requests?limit=80"),
    fetchJson("/api/observability/failures?limit=50"),
    fetchJson("/api/observability/tool-calls?limit=80"),
  ]);
  state.summary = summary;
  state.requests = requests.data || [];
  state.failures = failures.data || [];
  state.toolCalls = toolCalls.data || [];
  render();
}

function render() {
  renderSummary();
  renderModels();
  renderCharts();
  renderModelStats();
  renderRequests();
  renderFailures();
  renderToolCalls();
}

function renderSummary() {
  const summary = state.summary;
  const win = summary.window || {};
  const all = summary.all_time || {};
  const currency = firstCurrency(summary);
  const requests = Number(win.request_count || 0);
  const failures = Number(win.failure_count || 0);
  const input = Number(win.input_tokens || 0);
  const output = Number(win.output_tokens || 0);
  const hasPricing = Boolean(summary.pricing?.length);

  el("providerLine").textContent = `${summary.provider.base_url} · ${summary.provider.observability_enabled ? "recording enabled" : "recording disabled"}`;
  el("requestCount").textContent = fmtInt(requests);
  el("allTimeRequests").textContent = `${fmtInt(all.request_count)} all time`;
  el("costTotal").textContent = hasPricing ? fmtMoney(win.estimated_cost, currency) : "not configured";
  el("allTimeCost").textContent = hasPricing
    ? `${fmtMoney(all.estimated_cost, currency)} all time`
    : "set MODEL_PRICES_JSON";
  el("tokenTotal").textContent = fmtInt(input + output);
  el("tokenSplit").textContent = `${fmtInt(input)} in / ${fmtInt(output)} out`;
  el("failureCount").textContent = fmtInt(failures);
  el("failureRate").textContent = requests ? `${((failures / requests) * 100).toFixed(1)}%` : "0%";
  el("avgLatency").textContent = fmtMs(win.avg_latency_ms);
  el("toolCount").textContent = fmtInt(win.tool_call_count);
  el("toolArgsMode").textContent = summary.provider.store_tool_args ? "arguments stored with redaction" : "arguments disabled";
}

function renderModels() {
  const summary = state.summary;
  const prices = new Map((summary.pricing || []).map((item) => [item.model, item]));
  const cards = Object.entries(summary.configured_models || {}).map(([tier, model]) => {
    const price = prices.get(model);
    return `
      <article class="model-card">
        <b>${escapeHtml(tier)}</b>
        <code>${escapeHtml(model)}</code>
        <div class="price-line">
          <span class="pill">${price ? fmtMoney(price.input_per_1m, price.currency).replace(/0+$/, "").replace(/\.$/, "") : "price missing"} / 1M In</span>
          <span class="pill">${price ? fmtMoney(price.output_per_1m, price.currency).replace(/0+$/, "").replace(/\.$/, "") : "price missing"} / 1M Out</span>
          <span class="pill">${price ? fmtRate(price.advertised_tok_s) : "speed missing"}</span>
        </div>
      </article>
    `;
  });
  el("modelCards").innerHTML = cards.join("");
}

function renderCharts() {
  drawSeriesChart(el("tokensChart"), state.summary.series || [], {
    labelA: "Input",
    labelB: "Output",
    valueA: (row) => row.input_tokens || 0,
    valueB: (row) => row.output_tokens || 0,
    colorA: "#2563eb",
    colorB: "#0f9f6e",
    formatter: fmtInt,
  });
  drawSeriesChart(el("costChart"), state.summary.series || [], {
    labelA: "Cost",
    valueA: (row) => row.estimated_cost || 0,
    colorA: "#2563eb",
    formatter: (value) => `$${Number(value).toFixed(5)}`,
  });
}

function drawSeriesChart(canvas, rows, opts) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = Number(canvas.getAttribute("height")) || 210;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const pad = { top: 18, right: 18, bottom: 34, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const valuesA = rows.map(opts.valueA);
  const valuesB = opts.valueB ? rows.map(opts.valueB) : [];
  const maxValue = Math.max(1, ...valuesA, ...valuesB);

  ctx.strokeStyle = "#d9e0ea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#647386";
  ctx.font = "12px system-ui";
  ctx.fillText(opts.formatter(maxValue), 8, pad.top + 5);
  ctx.fillText("0", 36, pad.top + plotH);

  if (!rows.length) {
    ctx.fillText("No data yet", pad.left + 12, pad.top + 34);
    return;
  }

  drawLine(ctx, rows, opts.valueA, maxValue, pad, plotW, plotH, opts.colorA);
  if (opts.valueB) drawLine(ctx, rows, opts.valueB, maxValue, pad, plotW, plotH, opts.colorB);

  ctx.fillStyle = opts.colorA;
  ctx.fillText(opts.labelA, pad.left, height - 10);
  if (opts.labelB) {
    ctx.fillStyle = opts.colorB;
    ctx.fillText(opts.labelB, pad.left + 70, height - 10);
  }
}

function drawLine(ctx, rows, valueFn, maxValue, pad, plotW, plotH, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((row, index) => {
    const x = pad.left + (rows.length === 1 ? plotW : (index / (rows.length - 1)) * plotW);
    const y = pad.top + plotH - (Number(valueFn(row) || 0) / maxValue) * plotH;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderModelStats() {
  const rows = state.summary.model_stats || [];
  el("modelStatsBody").innerHTML = rows.length
    ? rows.map((row) => `
      <tr>
        <td><code>${escapeHtml(row.backend_model || "unknown")}</code></td>
        <td>${fmtInt(row.request_count)}</td>
        <td>${fmtInt(Number(row.input_tokens || 0) + Number(row.output_tokens || 0))}</td>
        <td>${fmtMoney(row.estimated_cost)}</td>
        <td>${fmtMs(row.avg_latency_ms)}</td>
        <td>${fmtRate(row.avg_observed_tok_s)}</td>
        <td>${fmtRate(row.advertised_tok_s)}</td>
        <td>${fmtInt(row.failure_count)}</td>
      </tr>
    `).join("")
    : `<tr><td class="empty" colspan="8">No model data yet</td></tr>`;
}

function renderRequests() {
  el("requestsBody").innerHTML = state.requests.length
    ? state.requests.map((row) => `
      <tr>
        <td>${fmtTime(row.started_at)}</td>
        <td>${statusPill(row.status)}</td>
        <td><code>${escapeHtml(row.claude_model)}</code></td>
        <td><code>${escapeHtml(row.backend_model)}</code></td>
        <td>${fmtInt(Number(row.input_tokens || 0) + Number(row.output_tokens || 0))}</td>
        <td><span class="pill">${escapeHtml(row.usage_source || "provider")}</span></td>
        <td>${fmtMoney(row.estimated_cost, row.currency || "USD")}</td>
        <td>${fmtMs(row.latency_ms)}</td>
        <td>${fmtInt(row.tool_call_count)}</td>
      </tr>
    `).join("")
    : `<tr><td class="empty" colspan="9">No requests yet</td></tr>`;
}

function renderFailures() {
  el("failuresBody").innerHTML = state.failures.length
    ? state.failures.map((row) => `
      <tr>
        <td>${fmtTime(row.started_at)}</td>
        <td><code>${escapeHtml(row.backend_model)}</code></td>
        <td>${statusPill(row.status)}</td>
        <td>${escapeHtml(row.error_message || row.error_type || "")}</td>
      </tr>
    `).join("")
    : `<tr><td class="empty" colspan="4">No failures in the selected window</td></tr>`;
}

function renderToolCalls() {
  el("toolCallsBody").innerHTML = state.toolCalls.length
    ? state.toolCalls.map((row) => `
      <tr>
        <td>${fmtTime(row.timestamp)}</td>
        <td><code>${escapeHtml(row.tool_name)}</code></td>
        <td><code>${escapeHtml(row.backend_model || "")}</code></td>
        <td><code>${escapeHtml(row.arguments_preview || "")}</code></td>
      </tr>
    `).join("")
    : `<tr><td class="empty" colspan="4">No tool calls yet</td></tr>`;
}

el("refreshBtn").addEventListener("click", refresh);
el("windowSelect").addEventListener("change", refresh);
window.addEventListener("resize", () => {
  if (state.summary) renderCharts();
});

refresh().catch((error) => {
  el("providerLine").textContent = `Dashboard failed to load: ${error.message}`;
});
setInterval(() => refresh().catch(() => {}), 5000);
