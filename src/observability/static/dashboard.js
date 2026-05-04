const state = {
  summary: null,
  requests: [],
  failures: [],
  toolCalls: [],
  refreshing: false,
  refreshQueued: false,
  tables: {
    modelStats: { sort: { column: "request_count", direction: "desc" }, filters: {} },
    requests: { sort: { column: "started_at", direction: "desc" }, filters: {} },
    failures: { sort: { column: "started_at", direction: "desc" }, filters: {} },
    toolCalls: { sort: { column: "timestamp", direction: "desc" }, filters: {} },
  },
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
  if (Number(value) === 0) return "free";
  const prefix = currency === "USD" ? "$" : `${currency} `;
  return `${prefix}${Number(value).toFixed(6)}`;
}

function fmtRequestCost(row) {
  if (row.usage_source === "local_optimization" || String(row.backend_model || "").startsWith("local/")) {
    return "free";
  }
  return fmtMoney(row.estimated_cost, row.currency || "USD");
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

function rowTotalTokens(row) {
  return Number(row.input_tokens || 0) + Number(row.output_tokens || 0);
}

const tableConfigs = {
  modelStats: {
    bodyId: "modelStatsBody",
    countId: "modelStatsCount",
    empty: "No model data yet",
    columns: [
      {
        id: "backend_model",
        label: "Model",
        type: "text",
        value: (row) => row.backend_model || "unknown",
        render: (row) => `<code>${escapeHtml(row.backend_model || "unknown")}</code>`,
      },
      {
        id: "request_count",
        label: "Requests",
        type: "number",
        value: (row) => Number(row.request_count || 0),
        render: (row) => fmtInt(row.request_count),
      },
      {
        id: "tokens",
        label: "Tokens",
        type: "number",
        value: rowTotalTokens,
        render: (row) => fmtInt(rowTotalTokens(row)),
      },
      {
        id: "estimated_cost",
        label: "Cost",
        type: "number",
        value: (row) => Number(row.estimated_cost || 0),
        render: (row) => fmtMoney(row.estimated_cost),
        filterValue: (row) => fmtMoney(row.estimated_cost),
      },
      {
        id: "avg_latency_ms",
        label: "Latency",
        type: "number",
        value: (row) => Number(row.avg_latency_ms || 0),
        render: (row) => fmtMs(row.avg_latency_ms),
      },
      {
        id: "avg_observed_tok_s",
        label: "Observed Tok/s",
        type: "number",
        value: (row) => Number(row.avg_observed_tok_s || 0),
        render: (row) => fmtRate(row.avg_observed_tok_s),
      },
      {
        id: "advertised_tok_s",
        label: "Advertised Tok/s",
        type: "number",
        value: (row) => Number(row.advertised_tok_s || 0),
        render: (row) => fmtRate(row.advertised_tok_s),
      },
      {
        id: "failure_count",
        label: "Failures",
        type: "number",
        value: (row) => Number(row.failure_count || 0),
        render: (row) => fmtInt(row.failure_count),
      },
    ],
  },
  requests: {
    bodyId: "requestsBody",
    countId: "requestsCount",
    empty: "No requests yet",
    columns: [
      {
        id: "started_at",
        label: "Time",
        type: "date",
        value: (row) => row.started_at,
        render: (row) => fmtTime(row.started_at),
      },
      {
        id: "status",
        label: "Status",
        type: "text",
        value: (row) => row.status || "unknown",
        render: (row) => statusPill(row.status),
      },
      {
        id: "claude_model",
        label: "Claude Model",
        type: "text",
        value: (row) => row.claude_model || "",
        render: (row) => `<code>${escapeHtml(row.claude_model)}</code>`,
      },
      {
        id: "backend_model",
        label: "Backend Model",
        type: "text",
        value: (row) => row.backend_model || "",
        render: (row) => `<code>${escapeHtml(row.backend_model)}</code>`,
      },
      {
        id: "tokens",
        label: "Tokens",
        type: "number",
        value: rowTotalTokens,
        render: (row) => fmtInt(rowTotalTokens(row)),
      },
      {
        id: "usage_source",
        label: "Usage",
        type: "text",
        value: (row) => row.usage_source || "provider",
        render: (row) => `<span class="pill">${escapeHtml(row.usage_source || "provider")}</span>`,
      },
      {
        id: "estimated_cost",
        label: "Cost",
        type: "number",
        value: (row) =>
          row.usage_source === "local_optimization" || String(row.backend_model || "").startsWith("local/")
            ? 0
            : Number(row.estimated_cost || 0),
        render: fmtRequestCost,
        filterValue: fmtRequestCost,
      },
      {
        id: "latency_ms",
        label: "Latency",
        type: "number",
        value: (row) => Number(row.latency_ms || 0),
        render: (row) => fmtMs(row.latency_ms),
      },
      {
        id: "tool_call_count",
        label: "Tools",
        type: "number",
        value: (row) => Number(row.tool_call_count || 0),
        render: (row) => fmtInt(row.tool_call_count),
      },
    ],
  },
  failures: {
    bodyId: "failuresBody",
    countId: "failuresCount",
    empty: "No failures in the selected window",
    columns: [
      {
        id: "started_at",
        label: "Time",
        type: "date",
        value: (row) => row.started_at,
        render: (row) => fmtTime(row.started_at),
      },
      {
        id: "backend_model",
        label: "Model",
        type: "text",
        value: (row) => row.backend_model || "",
        render: (row) => `<code>${escapeHtml(row.backend_model)}</code>`,
      },
      {
        id: "status",
        label: "Status",
        type: "text",
        value: (row) => row.status || "unknown",
        render: (row) => statusPill(row.status),
      },
      {
        id: "error",
        label: "Error",
        type: "text",
        value: (row) => row.error_message || row.error_type || "",
        render: (row) => escapeHtml(row.error_message || row.error_type || ""),
      },
    ],
  },
  toolCalls: {
    bodyId: "toolCallsBody",
    countId: "toolCallsCount",
    empty: "No tool calls yet",
    columns: [
      {
        id: "timestamp",
        label: "Time",
        type: "date",
        value: (row) => row.timestamp,
        render: (row) => fmtTime(row.timestamp),
      },
      {
        id: "tool_name",
        label: "Tool",
        type: "text",
        value: (row) => row.tool_name || "",
        render: (row) => `<code>${escapeHtml(row.tool_name)}</code>`,
      },
      {
        id: "backend_model",
        label: "Model",
        type: "text",
        value: (row) => row.backend_model || "",
        render: (row) => `<code>${escapeHtml(row.backend_model || "")}</code>`,
      },
      {
        id: "arguments_preview",
        label: "Args",
        type: "text",
        value: (row) => row.arguments_preview || "",
        render: (row) => `<code>${escapeHtml(row.arguments_preview || "")}</code>`,
      },
    ],
  },
};

function setupTableControls() {
  for (const [tableId, config] of Object.entries(tableConfigs)) {
    const table = document.querySelector(`table[data-table="${tableId}"]`);
    if (!table) continue;

    const headerRow = table.querySelector("thead tr:first-child");
    if (!headerRow || table.querySelector("thead tr.filter-row")) continue;

    headerRow.querySelectorAll("th").forEach((th, index) => {
      const column = config.columns[index];
      if (!column) return;
      th.dataset.column = column.id;
      th.innerHTML = `
        <button type="button" class="sort-button" data-table="${tableId}" data-column="${column.id}">
          <span>${escapeHtml(column.label)}</span>
          <span class="sort-indicator" aria-hidden="true"></span>
        </button>
      `;
    });

    const filterRow = document.createElement("tr");
    filterRow.className = "filter-row";
    filterRow.innerHTML = config.columns
      .map(
        (column) => `
          <th>
            <input
              class="column-filter"
              type="search"
              data-table="${tableId}"
              data-column="${column.id}"
              aria-label="Filter ${escapeHtml(column.label)}"
              placeholder="${column.type === "number" ? "Filter, >10" : "Filter"}"
            />
          </th>
        `,
      )
      .join("");
    headerRow.after(filterRow);
  }

  document.querySelectorAll(".sort-button").forEach((button) => {
    button.addEventListener("click", () => {
      const tableId = button.dataset.table;
      const column = button.dataset.column;
      const table = state.tables[tableId];
      if (!table || !column) return;

      table.sort =
        table.sort.column === column
          ? { column, direction: table.sort.direction === "asc" ? "desc" : "asc" }
          : { column, direction: "asc" };
      renderTableById(tableId);
    });
  });

  document.querySelectorAll(".column-filter").forEach((input) => {
    input.addEventListener("input", () => {
      const tableId = input.dataset.table;
      const column = input.dataset.column;
      if (!tableId || !column) return;
      state.tables[tableId].filters[column] = input.value;
      renderTableById(tableId);
    });
  });
}

function setupPanelControls() {
  document.querySelectorAll(".collapse-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const panel = button.closest(".table-panel");
      if (!panel) return;

      const collapsed = !panel.classList.contains("is-collapsed");
      panel.classList.toggle("is-collapsed", collapsed);
      button.setAttribute("aria-expanded", String(!collapsed));
      if (!collapsed && state.summary) renderCharts();
    });
  });
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function refresh() {
  if (state.refreshing) {
    state.refreshQueued = true;
    return;
  }
  state.refreshing = true;
  try {
    do {
      state.refreshQueued = false;
      const hours = el("windowSelect").value;
      const [summary, requests, failures, toolCalls] = await Promise.all([
        fetchJson(`/api/observability/summary?hours=${hours}`),
        fetchJson("/api/observability/requests?limit=500"),
        fetchJson("/api/observability/failures?limit=500"),
        fetchJson("/api/observability/tool-calls?limit=500"),
      ]);
      state.summary = summary;
      state.requests = requests.data || [];
      state.failures = failures.data || [];
      state.toolCalls = toolCalls.data || [];
      render();
    } while (state.refreshQueued);
  } finally {
    state.refreshing = false;
  }
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
  if (!canvas.dataset.chartHeight) {
    canvas.dataset.chartHeight = canvas.getAttribute("height") || "210";
  }
  const height = Number(canvas.dataset.chartHeight) || 210;
  canvas.style.height = `${height}px`;
  canvas.style.width = "100%";

  const rect = canvas.getBoundingClientRect();
  const parentWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
  const width = Math.max(1, Math.floor(rect.width || parentWidth || 600));
  const pixelWidth = Math.max(1, Math.floor(width * dpr));
  const pixelHeight = Math.max(1, Math.floor(height * dpr));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
  renderTableById("modelStats");
}

function renderRequests() {
  renderTableById("requests");
}

function renderFailures() {
  renderTableById("failures");
}

function renderToolCalls() {
  renderTableById("toolCalls");
}

function rowsForTable(tableId) {
  switch (tableId) {
    case "modelStats":
      return state.summary?.model_stats || [];
    case "requests":
      return state.requests || [];
    case "failures":
      return state.failures || [];
    case "toolCalls":
      return state.toolCalls || [];
    default:
      return [];
  }
}

function renderTableById(tableId) {
  const config = tableConfigs[tableId];
  const table = state.tables[tableId];
  if (!config || !table) return;

  const sortColumn = config.columns.find((column) => column.id === table.sort.column);
  const rows = rowsForTable(tableId)
    .filter((row) => config.columns.every((column) => matchesFilter(row, column, table.filters[column.id])))
    .sort((a, b) => {
      if (!sortColumn) return 0;
      const result = compareValues(sortColumn.value(a), sortColumn.value(b), sortColumn.type);
      return table.sort.direction === "asc" ? result : -result;
    });

  updateSortIndicators(tableId);
  updateTableCount(config, rows.length);
  el(config.bodyId).innerHTML = rows.length
    ? rows
        .map(
          (row) => `
            <tr>
              ${config.columns.map((column) => `<td>${column.render(row)}</td>`).join("")}
            </tr>
          `,
        )
        .join("")
    : `<tr><td class="empty" colspan="${config.columns.length}">${config.empty}</td></tr>`;
}

function updateTableCount(config, visibleRows) {
  if (!config.countId) return;

  const count = el(config.countId);
  if (!count) return;

  const suffix = visibleRows === 1 ? "row" : "rows";
  count.textContent = `${fmtInt(visibleRows)} ${suffix}`;
}

function matchesFilter(row, column, filter) {
  const needle = String(filter || "").trim().toLowerCase();
  if (!needle) return true;

  const value = column.value(row);
  const rendered = String(column.filterValue ? column.filterValue(row) : value ?? "").toLowerCase();
  if (column.type !== "number") {
    return rendered.includes(needle);
  }

  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    const range = needle.match(/^(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)$/);
    if (range) {
      return numeric >= Number(range[1]) && numeric <= Number(range[2]);
    }

    const comparator = needle.match(/^(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)$/);
    if (comparator) {
      const target = Number(comparator[2]);
      switch (comparator[1]) {
        case ">=":
          return numeric >= target;
        case "<=":
          return numeric <= target;
        case ">":
          return numeric > target;
        case "<":
          return numeric < target;
        case "=":
          return numeric === target;
      }
    }
  }

  return rendered.includes(needle);
}

function compareValues(a, b, type) {
  const left = normalizeSortValue(a, type);
  const right = normalizeSortValue(b, type);
  const leftMissing = left === null || left === undefined || Number.isNaN(left);
  const rightMissing = right === null || right === undefined || Number.isNaN(right);

  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  if (type === "number" || type === "date") return left - right;
  return String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
}

function normalizeSortValue(value, type) {
  if (type === "date") {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? timestamp : null;
  }
  if (type === "number") {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }
  return String(value ?? "").toLowerCase();
}

function updateSortIndicators(tableId) {
  const table = state.tables[tableId];
  document.querySelectorAll(`.sort-button[data-table="${tableId}"]`).forEach((button) => {
    const active = button.dataset.column === table.sort.column;
    button.classList.toggle("active", active);
    const indicator = button.querySelector(".sort-indicator");
    if (indicator) indicator.textContent = active ? (table.sort.direction === "asc" ? "▲" : "▼") : "";
    button.setAttribute("aria-sort", active ? (table.sort.direction === "asc" ? "ascending" : "descending") : "none");
  });
}

setupTableControls();
setupPanelControls();
el("refreshBtn").addEventListener("click", refresh);
el("windowSelect").addEventListener("change", refresh);
window.addEventListener("resize", () => {
  if (state.summary) renderCharts();
});

refresh().catch((error) => {
  el("providerLine").textContent = `Dashboard failed to load: ${error.message}`;
});
setInterval(() => refresh().catch(() => {}), 5000);
