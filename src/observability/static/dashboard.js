// Claude Proxy Observability Dashboard
// Self-contained, no external dependencies.

// ============================================
// Utilities
// ============================================

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

async function fetchJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

// ============================================
// ThemeManager: handles dark/light mode
// ============================================
class ThemeManager {
  constructor() {
    this.theme = this.loadTheme();
    this.apply();
  }

  loadTheme() {
    const saved = localStorage.getItem("dashboard-theme");
    if (saved) return saved;
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) return "dark";
    return "light";
  }

  toggle() {
    const target = this.theme === "light" ? "dark" : "light";
    // Unified cross-fade: fade body slightly while colours swap,
    // then bring it back. Total perceived time ~300 ms.
    document.body.style.opacity = "0.6";
    setTimeout(() => {
      this.theme = target;
      this.apply();
      localStorage.setItem("dashboard-theme", this.theme);
      document.dispatchEvent(new CustomEvent("themechange", { detail: { theme: this.theme } }));
      document.body.style.opacity = "1";
    }, 280);
  }

  apply() {
    document.documentElement.setAttribute("data-theme", this.theme);
    const sun = document.querySelector(".icon-sun");
    const moon = document.querySelector(".icon-moon");
    if (sun) sun.style.display = this.theme === "light" ? "block" : "none";
    if (moon) moon.style.display = this.theme === "light" ? "none" : "block";
  }
}

// ============================================
// SessionManager: fetches and renders sidebar sessions
// ============================================
class SessionManager {
  constructor(onSessionSelect, contextLimits, configuredModels) {
    this.sessions = [];
    this.filtered = [];
    this.activeSession = null;
    this.onSessionSelect = onSessionSelect;
    this.contextLimits = contextLimits || {};
    this.configuredModels = configuredModels || {};
    this.REPORTED_LIMIT = 1_048_576;
    this.searchEl = document.getElementById("sessionSearch");
    this.listEl = document.getElementById("sessionList");
    this.skeletonEl = document.getElementById("sessionSkeleton");
    this.globalBtn = document.getElementById("globalOverviewBtn");
    this.bindEvents();
  }

  updateContextLimits(contextLimits, configuredModels) {
    this.contextLimits = contextLimits || {};
    this.configuredModels = configuredModels || {};
  }

  // Resolve real context limit for a backend model
  _resolveLimit(modelName) {
    if (!modelName) return 0;
    // Reverse-lookup: find which tier maps to this model
    const tier = Object.entries(this.configuredModels).find(([_, m]) => m === modelName)?.[0];
    if (tier) return this.contextLimits[tier] || 0;
    return 0;
  }

  bindEvents() {
    this.searchEl?.addEventListener("input", () => this.filter(this.searchEl.value));
    this.globalBtn?.addEventListener("click", () => {
      this.selectGlobal();
    });
  }

  async load() {
    try {
      const resp = await fetchJson("/api/observability/sessions");
      this.sessions = resp.data || [];
      this.filtered = [...this.sessions];
      if (this.skeletonEl) this.skeletonEl.style.display = "none";
      this.render();
    } catch (e) {
      console.error("Failed to load sessions:", e);
      if (this.skeletonEl) this.skeletonEl.style.display = "none";
      if (this.listEl) this.listEl.innerHTML = '<div class="empty-state">No sessions found</div>';
    }
  }

  filter(query) {
    const q = (query || "").toLowerCase();
    this.filtered = q
      ? this.sessions.filter(
          (s) =>
            (s.session_name || "").toLowerCase().includes(q) ||
            (s.backend_model || "").toLowerCase().includes(q)
        )
      : [...this.sessions];
    this.render();
  }

  render() {
    if (!this.listEl) return;
    if (this.filtered.length === 0) {
      this.listEl.innerHTML = '<div class="empty-state">No sessions match</div>';
      return;
    }
    const now = Date.now();
    const fiveMinAgo = now - 5 * 60 * 1000;
    this.listEl.innerHTML = this.filtered
      .map((s) => {
        const lastSeen = s.last_seen ? new Date(s.last_seen).getTime() : 0;
        const isLive = lastSeen > fiveMinAgo;
        // Context bar: proportional scaling
        // If real model has 256K window and Claude assumes 1M,
        // 128K real usage -> 128K/256K = 50% -> report as 512K/1M = 50%
        const peakTokens = s.peak_context_tokens || s.total_tokens || 0;
        const modelName = s.backend_model || "";
        const realLimit = this._resolveLimit(modelName);
        let contextPct;
        if (realLimit > 0) {
          contextPct = Math.min(100, Math.round((peakTokens / realLimit) * 100));
        } else {
          contextPct = Math.min(100, Math.round((peakTokens / this.REPORTED_LIMIT) * 100));
        }
        let fillColor = "var(--accent-green)";
        if (contextPct > 80) fillColor = "var(--accent-red)";
        else if (contextPct > 50) fillColor = "var(--accent-amber)";
        const relativeTime = this.fmtRelative(s.last_seen);
        const activeClass = this.activeSession === s.session_name ? " session-card--active" : "";
        return `
          <div class="session-card${activeClass}" data-session-name="${escapeHtml(s.session_name || "")}">
            <div class="session-card-header">
              <code class="session-card-title">${escapeHtml(s.backend_model || "unknown")}</code>
              ${isLive ? '<span class="session-live-dot" aria-label="Live session"></span>' : ""}
            </div>
            <div class="session-card-subtitle">${escapeHtml(s.session_name || "Unnamed")}</div>
            <div class="session-context-bar">
              <div class="fill" style="width:${contextPct}%; background:${fillColor}"></div>
            </div>
            <div class="session-card-meta">
              ${fmtInt(s.total_requests || 0)} requests &middot; ${fmtInt(s.total_tokens || 0)} tokens &middot; ${relativeTime}
            </div>
          </div>
        `;
      })
      .join("");

    this.listEl.querySelectorAll(".session-card").forEach((card) => {
      card.addEventListener("click", () => {
        const name = card.dataset.sessionName;
        this.setActive(name);
        if (this.onSessionSelect) this.onSessionSelect(name);
      });
    });
  }

  fmtRelative(dateStr) {
    if (!dateStr) return "unknown";
    const d = new Date(dateStr);
    const diff = Date.now() - d.getTime();
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  setActive(name) {
    this.activeSession = name;
    this.render();
    if (this.globalBtn) {
      this.globalBtn.classList.toggle("session-card--active", !name);
    }
  }

  selectGlobal() {
    this.setActive(null);
    if (this.onSessionSelect) this.onSessionSelect(null);
  }
}

// ============================================
// ChartController: enhanced canvas charts
// ============================================
class ChartController {
  constructor(themeManager) {
    this.themeManager = themeManager;
    this.charts = new Map(); // canvas -> { rows, opts }
    this.tooltipEl = null;
    this.ensureTooltip();
    document.addEventListener("themechange", () => this.redrawAll());
  }

  ensureTooltip() {
    if (this.tooltipEl) return;
    this.tooltipEl = document.createElement("div");
    this.tooltipEl.className = "chart-tooltip";
    this.tooltipEl.style.cssText =
      "position:absolute;pointer-events:none;background:rgba(0,0,0,0.85);color:#fff;font:13px system-ui, -apple-system, sans-serif;padding:6px 10px;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.2);z-index:9999;opacity:0;transition:opacity 0.15s;white-space:nowrap;";
    document.body.appendChild(this.tooltipEl);
  }

  getThemeColor(varName) {
    const val = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    if (val) return val;
    // Fallbacks
    const isDark = this.themeManager.theme === "dark";
    const fallbacks = {
      "--text-secondary": isDark ? "#94a3b8" : "#647386",
      "--border": isDark ? "#334155" : "#d9e0ea",
    };
    return fallbacks[varName] || "#999";
  }

  register(canvas, rows, opts) {
    if (!canvas) return;
    this.charts.set(canvas, { rows, opts });
    // Clean stale listeners by cloning and replacing
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);
    this.charts.set(newCanvas, { rows, opts });
    this._attachListeners(newCanvas, rows, opts);
    this.drawSeriesChart(newCanvas, rows, opts);
  }

  _attachListeners(canvas, rows, opts) {
    canvas.addEventListener("mousemove", (e) => {
      this._onMouseMove(e, canvas, rows, opts);
    });
    canvas.addEventListener("mouseleave", () => {
      this.tooltipEl.style.opacity = "0";
    });
    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      this._onMouseMove(e, canvas, rows, opts);
    }, { passive: false });
    canvas.addEventListener("touchend", () => {
      this.tooltipEl.style.opacity = "0";
    });
  }

  _onMouseMove(e, canvas, rows, opts) {
    if (!rows.length) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
    const y = (e.clientY || (e.touches && e.touches[0].clientY)) - rect.top;
    const pad = { top: 18, right: 18, bottom: 34, left: 54 };
    const height = Number(canvas.dataset.chartHeight) || 210;
    const plotW = rect.width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    if (x < pad.left || x > pad.left + plotW || y < pad.top || y > pad.top + plotH) {
      this.tooltipEl.style.opacity = "0";
      return;
    }

    const relX = Math.max(0, Math.min(1, (x - pad.left) / plotW));
    const index = Math.round(relX * (rows.length - 1));
    const row = rows[index];
    if (!row) return;

    const valuesA = rows.map(opts.valueA);
    const valuesB = opts.valueB ? rows.map(opts.valueB) : [];
    const aVal = valuesA[index];
    const bVal = valuesB.length ? valuesB[index] : null;
    const ts = row.bucket || row.interval || row.timestamp || "";

    let tooltipHtml = "";
    if (ts) {
      tooltipHtml += `<div style="opacity:0.7;font-size:11px;margin-bottom:4px;">${escapeHtml(fmtTime(ts))}</div>`;
    }
    tooltipHtml += `<div style="color:${opts.colorA};font-weight:600;">${opts.labelA}: ${opts.formatter ? opts.formatter(aVal) : aVal}</div>`;
    if (bVal !== null && opts.labelB) {
      tooltipHtml += `<div style="color:${opts.colorB};font-weight:600;">${opts.labelB}: ${opts.formatter ? opts.formatter(bVal) : bVal}</div>`;
    }

    this.tooltipEl.innerHTML = tooltipHtml;
    this.tooltipEl.style.opacity = "1";

    let tipX = e.clientX + 14;
    let tipY = e.clientY - 10;
    const tipRect = this.tooltipEl.getBoundingClientRect();
    if (tipX + tipRect.width > window.innerWidth) {
      tipX = e.clientX - tipRect.width - 14;
    }
    if (tipY < 0) tipY = e.clientY + 20;
    this.tooltipEl.style.left = `${tipX}px`;
    this.tooltipEl.style.top = `${tipY}px`;
  }

  redrawAll() {
    for (const [canvas, { rows, opts }] of this.charts) {
      if (!canvas.parentNode) {
        this.charts.delete(canvas);
        continue;
      }
      this.drawSeriesChart(canvas, rows, opts);
    }
  }

  drawSeriesChart(canvas, rows, opts) {
    if (!canvas) return;
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

    // Theme-aware axis/grid colors
    const axisColor = this.getThemeColor("--border");
    const labelColor = this.getThemeColor("--text-secondary");

    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Keep existing gridline logic: horizontal line at 50%
    const gridColor = this.getThemeColor("--border");
    ctx.strokeStyle = gridColor;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH / 2);
    ctx.lineTo(pad.left + plotW, pad.top + plotH / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = labelColor;
    ctx.font = "12px system-ui";
    ctx.fillText(opts.formatter ? opts.formatter(maxValue) : fmtInt(maxValue), 8, pad.top + 5);
    ctx.fillText("0", 36, pad.top + plotH);

    if (!rows.length) {
      ctx.fillStyle = labelColor;
      ctx.fillText("No data yet", pad.left + 12, pad.top + 34);
      return;
    }

    // Draw gradient fills and lines
    this._drawSmoothLine(ctx, rows, opts.valueA, maxValue, pad, plotW, plotH, opts.colorA, true);
    if (opts.valueB) {
      this._drawSmoothLine(ctx, rows, opts.valueB, maxValue, pad, plotW, plotH, opts.colorB, true);
    }
    this._drawSmoothLine(ctx, rows, opts.valueA, maxValue, pad, plotW, plotH, opts.colorA, false);
    if (opts.valueB) {
      this._drawSmoothLine(ctx, rows, opts.valueB, maxValue, pad, plotW, plotH, opts.colorB, false);
    }

    // Legend
    const labelColorVal = this.getThemeColor("--text-secondary");
    ctx.fillStyle = opts.colorA;
    ctx.fillText(opts.labelA, pad.left, height - 10);
    if (opts.labelB) {
      ctx.fillStyle = opts.colorB;
      ctx.fillText(opts.labelB, pad.left + 70, height - 10);
    }
  }

  _drawSmoothLine(ctx, rows, valueFn, maxValue, pad, plotW, plotH, color, fillOnly) {
    if (!rows.length) return;
    const points = rows.map((row, index) => ({
      x: pad.left + (rows.length === 1 ? plotW : (index / (rows.length - 1)) * plotW),
      y: pad.top + plotH - (Number(valueFn(row) || 0) / maxValue) * plotH,
    }));

    ctx.beginPath();
    if (points.length === 1) {
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[0].x, pad.top + plotH);
    } else {
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i];
        const p1 = points[i + 1];
        const cpX = (p0.x + p1.x) / 2;
        ctx.quadraticCurveTo(p0.x, p0.y, cpX, (p0.y + p1.y) / 2);
        ctx.quadraticCurveTo(cpX, (p0.y + p1.y) / 2, p1.x, p1.y);
      }
      // Close for fill
      ctx.lineTo(points[points.length - 1].x, pad.top + plotH);
      ctx.lineTo(points[0].x, pad.top + plotH);
      ctx.closePath();
    }

    if (fillOnly) {
      // Gradient fill under the line
      const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
      gradient.addColorStop(0, this._withAlpha(color, 0.3));
      gradient.addColorStop(1, this._withAlpha(color, 0.0));
      ctx.fillStyle = gradient;
      ctx.fill();
    } else {
      // Stroke the line
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
          const p0 = points[i];
          const p1 = points[i + 1];
          const cpX = (p0.x + p1.x) / 2;
          ctx.quadraticCurveTo(cpX, p0.y, p1.x, p1.y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();
      }
    }
  }

  _withAlpha(hex, alpha) {
    // Convert hex (#RRGGBB or #RGB) to rgba
    let c = hex.replace("#", "");
    if (c.length === 3) {
      c = c[0] + c[0] + c[1] + c[1] + c[2] + c[2];
    }
    const r = parseInt(c.substring(0, 2), 16);
    const g = parseInt(c.substring(2, 4), 16);
    const b = parseInt(c.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  drawSparkline(canvas, values, color) {
    if (!canvas || !values.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = 80;
    const h = 24;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(1, ...values);
    const pad = 2;
    const points = values.map((v, i) => ({
      x: pad + (values.length === 1 ? (w - pad * 2) : (i / (values.length - 1)) * (w - pad * 2)),
      y: h - pad - (v / max) * (h - pad * 2),
    }));

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const cpX = (p0.x + p1.x) / 2;
      ctx.quadraticCurveTo(cpX, p0.y, p1.x, p1.y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
  }
}

// ============================================
// Table Configs
// ============================================
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
        render: (row) => `<code>${escapeHtml(row.claude_model || "")}</code>`,
      },
      {
        id: "backend_model",
        label: "Backend Model",
        type: "text",
        value: (row) => row.backend_model || "",
        render: (row) => `<code>${escapeHtml(row.backend_model || "")}</code>`,
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
        render: (row) => `<code>${escapeHtml(row.backend_model || "")}</code>`,
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
        render: (row) => `<code>${escapeHtml(row.tool_name || "")}</code>`,
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
  // Session tables use the same column definitions
  sessionRequests: {
    bodyId: "sessionRequestsBody",
    countId: "sessionRequestsCount",
    empty: "No requests in this session",
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
        render: (row) => `<code>${escapeHtml(row.claude_model || "")}</code>`,
      },
      {
        id: "backend_model",
        label: "Backend Model",
        type: "text",
        value: (row) => row.backend_model || "",
        render: (row) => `<code>${escapeHtml(row.backend_model || "")}</code>`,
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
  sessionFailures: {
    bodyId: "sessionFailuresBody",
    countId: "sessionFailuresCount",
    empty: "No failures in this session",
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
        render: (row) => `<code>${escapeHtml(row.backend_model || "")}</code>`,
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
  sessionToolCalls: {
    bodyId: "sessionToolCallsBody",
    countId: "sessionToolCallsCount",
    empty: "No tool calls in this session",
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
        render: (row) => `<code>${escapeHtml(row.tool_name || "")}</code>`,
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

// ============================================
// DashboardApp: main orchestrator
// ============================================
class DashboardApp {
  constructor() {
    this.themeManager = new ThemeManager();
    this.sessionManager = new SessionManager(
      (name) => this.onSessionSelect(name),
      {},  // context Limits populated after first summary fetch
      {}   // configured Models populated after first summary fetch
    );
    this.chartController = new ChartController(this.themeManager);
    this.state = {
      summary: null,
      requests: [],
      failures: [],
      toolCalls: [],
      sessionSummary: null,
      sessionRequests: [],
      sessionFailures: [],
      sessionToolCalls: [],
      tables: {
        modelStats: { sort: { column: "request_count", direction: "desc" }, filters: {} },
        requests: { sort: { column: "started_at", direction: "desc" }, filters: {} },
        failures: { sort: { column: "started_at", direction: "desc" }, filters: {} },
        toolCalls: { sort: { column: "timestamp", direction: "desc" }, filters: {} },
        sessionRequests: { sort: { column: "started_at", direction: "desc" }, filters: {} },
        sessionToolCalls: { sort: { column: "timestamp", direction: "desc" }, filters: {} },
        sessionFailures: { sort: { column: "started_at", direction: "desc" }, filters: {} },
      },
      refreshing: false,
      refreshQueued: false,
    };
    this.animatedMetrics = new Set();
    this.init();
  }

  init() {
    // Sidebar toggle
    document.getElementById("sidebarToggle")?.addEventListener("click", () => this.toggleSidebar());
    document.getElementById("sidebarBackdrop")?.addEventListener("click", () => this.closeSidebar());

    // Theme toggle
    document.getElementById("themeToggle")?.addEventListener("click", () => this.themeManager.toggle());

    // Back button
    document.getElementById("backToGlobal")?.addEventListener("click", () => this.showGlobalView());

    // Existing controls
    document.getElementById("refreshBtn")?.addEventListener("click", () => this.refresh());
    document.getElementById("windowSelect")?.addEventListener("change", () => this.refresh());

    this.setupTableControls();
    this.setupPanelControls();

    // Load sidebar
    this.sessionManager.load();

    // Initial data load
    this.refresh();

    // Auto refresh every 5s
    setInterval(() => this.refresh().catch(() => {}), 5000);
  }

  toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar?.classList.toggle("sidebar--open");
  }

  closeSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar?.classList.remove("sidebar--open");
  }

  // ===========================
  // View switching
  // ===========================
  async onSessionSelect(sessionName) {
    if (!sessionName) {
      this.showGlobalView();
      return;
    }
    await this.loadSessionView(sessionName);
  }

  showGlobalView() {
    const globalView = document.getElementById("globalView");
    const sessionView = document.getElementById("sessionView");
    if (globalView) globalView.style.display = "block";
    if (sessionView) sessionView.style.display = "none";
    this.sessionManager.setActive(null);
    this.closeSidebar();
  }

  showSessionView() {
    const globalView = document.getElementById("globalView");
    const sessionView = document.getElementById("sessionView");
    if (globalView) globalView.style.display = "none";
    if (sessionView) sessionView.style.display = "block";
    this.closeSidebar();
  }

  async loadSessionView(sessionName) {
    this.showSessionView();
    const title = document.getElementById("sessionTitle");
    if (title) title.textContent = sessionName;

    const [summary, requests, failures, toolCalls, contextUsage] = await Promise.all([
      fetchJson(`/api/observability/sessions/${encodeURIComponent(sessionName)}/summary`).catch(() => null),
      fetchJson(`/api/observability/requests?session_name=${encodeURIComponent(sessionName)}&limit=500`).catch(() => ({ data: [] })),
      fetchJson(`/api/observability/failures?session_name=${encodeURIComponent(sessionName)}&limit=500`).catch(() => ({ data: [] })),
      fetchJson(`/api/observability/tool-calls?session_name=${encodeURIComponent(sessionName)}&limit=500`).catch(() => ({ data: [] })),
      fetchJson("/api/observability/context-usage", {
        headers: { "x-session-name": sessionName },
      }).catch(() => null),
    ]);

    this.state.sessionSummary = summary;
    this.state.sessionRequests = requests.data || [];
    this.state.sessionFailures = failures.data || [];
    this.state.sessionToolCalls = toolCalls.data || [];
    this.renderSessionView(contextUsage);
  }

  renderSessionView(contextUsage) {
    const summary = this.state.sessionSummary;
    if (!summary) return;
    const win = summary.window || {};
    const currency = (this.state.summary?.pricing || []).find((p) => p.currency)?.currency || "USD";

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };
    set("sessionRequestCount", fmtInt(win.request_count || 0));
    set("sessionCostTotal", fmtMoney(win.estimated_cost, currency));
    set("sessionTokenTotal", fmtInt((win.input_tokens || 0) + (win.output_tokens || 0)));
    set("sessionTokenSplit", `${fmtInt(win.input_tokens || 0)} in / ${fmtInt(win.output_tokens || 0)} out`);
    set("sessionFailureCount", fmtInt(win.failure_count || 0));
    const reqs = win.request_count || 0;
    set("sessionFailureRate", reqs ? `${((win.failure_count / reqs) * 100).toFixed(1)}%` : "0%");
    set("sessionAvgLatency", fmtMs(win.avg_latency_ms));
    set("sessionToolCount", fmtInt(win.tool_call_count || 0));

    // Context usage bar
    const pct = contextUsage ? contextUsage.percentage_used || 0 : 0;
    const fill = document.getElementById("sessionContextFill");
    const pctEl = document.getElementById("sessionContextPercent");
    if (fill) fill.style.width = `${pct}%`;
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
    if (fill) {
      fill.style.background = pct > 80 ? "var(--accent-red)" : pct > 50 ? "var(--accent-amber)" : "var(--accent-green)";
    }

    // Session tokens chart
    this.chartController.register(document.getElementById("sessionTokensChart"), summary.series || [], {
      labelA: "Input",
      labelB: "Output",
      valueA: (row) => row.input_tokens || 0,
      valueB: (row) => row.output_tokens || 0,
      colorA: "#2563eb",
      colorB: "#0f9f6e",
      formatter: fmtInt,
    });

    // Render session tables
    this.renderTableById("sessionRequests");
    this.renderTableById("sessionToolCalls");
    this.renderTableById("sessionFailures");
  }

  // ===========================
  // Global refresh / render
  // ===========================
  async refresh() {
    if (this.state.refreshing) {
      this.state.refreshQueued = true;
      return;
    }
    this.state.refreshing = true;
    try {
      do {
        this.state.refreshQueued = false;
        const hours = el("windowSelect")?.value || "24";
        const [summary, requests, failures, toolCalls] = await Promise.all([
          fetchJson(`/api/observability/summary?hours=${hours}`),
          fetchJson("/api/observability/requests?limit=500"),
          fetchJson("/api/observability/failures?limit=500"),
          fetchJson("/api/observability/tool-calls?limit=500"),
        ]);
        this.state.summary = summary;
        this.state.requests = requests.data || [];
        this.state.failures = failures.data || [];
        this.state.toolCalls = toolCalls.data || [];
        this.render();

        // Also refresh session list sidebar
        this.sessionManager.load().catch(() => {});
      } while (this.state.refreshQueued);
    } finally {
      this.state.refreshing = false;
    }
  }

  render() {
    this.renderSummary();
    this.renderModels();
    this.renderCharts();
    this.renderTableById("modelStats");
    this.renderTableById("requests");
    this.renderTableById("failures");
    this.renderTableById("toolCalls");
  }

  renderSummary() {
    const summary = this.state.summary;
    if (!summary) return;
    const win = summary.window || {};
    const all = summary.all_time || {};
    const currency = firstCurrency(summary);
    const requests = Number(win.request_count || 0);
    const failures = Number(win.failure_count || 0);
    const input = Number(win.input_tokens || 0);
    const output = Number(win.output_tokens || 0);
    const hasPricing = Boolean(summary.pricing?.length);

    el("providerLine").textContent = `${summary.provider.base_url} · ${summary.provider.observability_enabled ? "recording enabled" : "recording disabled"}`;

    // Animated counters for metric values on first load only
    this._animatedSet("requestCount", fmtInt(requests), "requestCount");
    el("allTimeRequests").textContent = `${fmtInt(all.request_count)} all time`;
    this._animatedSet("costTotal", hasPricing ? fmtMoney(win.estimated_cost, currency) : "not configured", "costTotal");
    el("allTimeCost").textContent = hasPricing
      ? `${fmtMoney(all.estimated_cost, currency)} all time`
      : "set MODEL_PRICES_JSON";
    this._animatedSet("tokenTotal", fmtInt(input + output), "tokenTotal");
    el("tokenSplit").textContent = `${fmtInt(input)} in / ${fmtInt(output)} out`;
    this._animatedSet("failureCount", fmtInt(failures), "failureCount");
    el("failureRate").textContent = requests ? `${((failures / requests) * 100).toFixed(1)}%` : "0%";
    el("avgLatency").textContent = fmtMs(win.avg_latency_ms);
    el("toolCount").textContent = fmtInt(win.tool_call_count);
    el("toolArgsMode").textContent = summary.provider.store_tool_args ? "arguments stored with redaction" : "arguments disabled";
  }

  _animatedSet(elementId, targetValue, key) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Only animate on first load, not on refresh
    if (this.animatedMetrics.has(key)) {
      element.textContent = targetValue;
      return;
    }

    this.animatedMetrics.add(key);

    // Numeric values only: extract numeric part for animation
    const numericMatch = targetValue.match(/[\d,]+/);
    if (!numericMatch) {
      element.textContent = targetValue;
      return;
    }

    const digitsOnly = numericMatch[0].replace(/,/g, "");
    const targetNum = Number(digitsOnly);
    if (!Number.isFinite(targetNum) || targetNum === 0) {
      element.textContent = targetValue;
      return;
    }

    const prefix = targetValue.substring(0, numericMatch.index);
    const suffix = targetValue.substring(numericMatch.index + numericMatch[0].length);
    const duration = 800;
    const start = performance.now();

    const tick = (now) => {
      const elapsed = now - start;
      const progress = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - progress, 3); // easeOut cubic
      const current = Math.round(targetNum * eased);
      // Re-apply comma formatting
      element.textContent = prefix + fmtInt(current) + suffix;
      if (progress < 1) {
        requestAnimationFrame(tick);
      } else {
        element.textContent = targetValue;
      }
    };

    element.textContent = prefix + "0" + suffix;
    requestAnimationFrame(tick);
  }

  renderModels() {
    const summary = this.state.summary;
    if (!summary) return;
    // Pass config context limits to session manager for proportional sidebar bars
    this.sessionManager?.updateContextLimits(
      summary.context_limits || {},
      summary.configured_models || {}
    );
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

  renderCharts() {
    this.chartController.register(document.getElementById("tokensChart"), this.state.summary?.series || [], {
      labelA: "Input",
      labelB: "Output",
      valueA: (row) => row.input_tokens || 0,
      valueB: (row) => row.output_tokens || 0,
      colorA: "#2563eb",
      colorB: "#0f9f6e",
      formatter: fmtInt,
    });
    this.chartController.register(document.getElementById("costChart"), this.state.summary?.series || [], {
      labelA: "Cost",
      valueA: (row) => row.estimated_cost || 0,
      colorA: "#2563eb",
      formatter: (value) => `$${Number(value).toFixed(5)}`,
    });

    // Cumulative cost (always increasing)
    const series = this.state.summary?.series || [];
    let runningCost = 0;
    const cumulativeCost = series.map((row) => {
      runningCost += Number(row.estimated_cost || 0);
      return {
        ...row,
        cumulative_cost: runningCost,
      };
    });
    this.chartController.register(document.getElementById("cumulativeCostChart"), cumulativeCost, {
      labelA: "Cumulative Cost",
      valueA: (row) => row.cumulative_cost || 0,
      colorA: "#0f9f6e",
      formatter: (value) => `$${Number(value).toFixed(5)}`,
    });

    // Cumulative tokens (always increasing)
    let runningTokens = 0;
    const cumulativeTokens = series.map((row) => {
      runningTokens += Number(row.input_tokens || 0) + Number(row.output_tokens || 0);
      return {
        ...row,
        cumulative_tokens: runningTokens,
      };
    });
    this.chartController.register(document.getElementById("cumulativeTokensChart"), cumulativeTokens, {
      labelA: "Cumulative Tokens",
      valueA: (row) => row.cumulative_tokens || 0,
      colorA: "#0f9f6e",
      formatter: fmtInt,
    });
  }

  // ===========================
  // Table controls
  // ===========================
  setupTableControls() {
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
          `
        )
        .join("");
      headerRow.after(filterRow);
    }

    document.querySelectorAll(".sort-button").forEach((button) => {
      button.addEventListener("click", () => {
        const tableId = button.dataset.table;
        const column = button.dataset.column;
        const table = this.state.tables[tableId];
        if (!table || !column) return;

        table.sort =
          table.sort.column === column
            ? { column, direction: table.sort.direction === "asc" ? "desc" : "asc" }
            : { column, direction: "asc" };
        this.renderTableById(tableId);
      });
    });

    document.querySelectorAll(".column-filter").forEach((input) => {
      input.addEventListener("input", () => {
        const tableId = input.dataset.table;
        const column = input.dataset.column;
        if (!tableId || !column) return;
        this.state.tables[tableId].filters[column] = input.value;
        this.renderTableById(tableId);
      });
    });
  }

  setupPanelControls() {
    document.querySelectorAll(".collapse-toggle").forEach((button) => {
      button.addEventListener("click", () => {
        const panel = button.closest(".table-panel");
        if (!panel) return;
        const collapsed = !panel.classList.contains("is-collapsed");
        panel.classList.toggle("is-collapsed", collapsed);
        button.setAttribute("aria-expanded", String(!collapsed));
        // Refresh the charts when expanding the global view after resize
        if (!collapsed && this.state.summary) {
          requestAnimationFrame(() => this.renderCharts());
        }
      });
    });
  }

  rowsForTable(tableId) {
    switch (tableId) {
      case "modelStats":
        return this.state.summary?.model_stats || [];
      case "requests":
        return this.state.requests || [];
      case "failures":
        return this.state.failures || [];
      case "toolCalls":
        return this.state.toolCalls || [];
      case "sessionRequests":
        return this.state.sessionRequests || [];
      case "sessionFailures":
        return this.state.sessionFailures || [];
      case "sessionToolCalls":
        return this.state.sessionToolCalls || [];
      default:
        return [];
    }
  }

  renderTableById(tableId) {
    const config = tableConfigs[tableId];
    const table = this.state.tables[tableId];
    if (!config || !table) return;

    const sortColumn = config.columns.find((column) => column.id === table.sort.column);
    const rows = this.rowsForTable(tableId)
      .filter((row) => config.columns.every((column) => this.matchesFilter(row, column, table.filters[column.id])))
      .sort((a, b) => {
        if (!sortColumn) return 0;
        const result = this.compareValues(sortColumn.value(a), sortColumn.value(b), sortColumn.type);
        return table.sort.direction === "asc" ? result : -result;
      });

    this.updateSortIndicators(tableId);
    this.updateTableCount(config, rows.length);
    const body = el(config.bodyId);
    if (body) {
      body.innerHTML = rows.length
        ? rows
            .map(
              (row) => `
                <tr>
                  ${config.columns.map((column) => `<td>${column.render(row)}</td>`).join("")}
                </tr>
              `
            )
            .join("")
        : `<tr><td class="empty" colspan="${config.columns.length}">${config.empty}</td></tr>`;
    }
  }

  updateTableCount(config, visibleRows) {
    if (!config.countId) return;
    const count = el(config.countId);
    if (!count) return;
    const suffix = visibleRows === 1 ? "row" : "rows";
    count.textContent = `${fmtInt(visibleRows)} ${suffix}`;
  }

  matchesFilter(row, column, filter) {
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

  compareValues(a, b, type) {
    const left = this.normalizeSortValue(a, type);
    const right = this.normalizeSortValue(b, type);
    const leftMissing = left === null || left === undefined || Number.isNaN(left);
    const rightMissing = right === null || right === undefined || Number.isNaN(right);

    if (leftMissing && rightMissing) return 0;
    if (leftMissing) return 1;
    if (rightMissing) return -1;
    if (type === "number" || type === "date") return left - right;
    return String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
  }

  normalizeSortValue(value, type) {
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

  updateSortIndicators(tableId) {
    const table = this.state.tables[tableId];
    document.querySelectorAll(`.sort-button[data-table="${tableId}"]`).forEach((button) => {
      const active = button.dataset.column === table.sort.column;
      button.classList.toggle("active", active);
      const indicator = button.querySelector(".sort-indicator");
      if (indicator) indicator.textContent = active ? (table.sort.direction === "asc" ? "▲" : "▼") : "";
      button.setAttribute("aria-sort", active ? (table.sort.direction === "asc" ? "ascending" : "descending") : "none");
    });
  }
}

// ============================================
// Bootstrap
// ============================================
const app = new DashboardApp();
window.addEventListener("resize", () => {
  // Debounce chart redraw on resize
  if (app._resizeTimer) clearTimeout(app._resizeTimer);
  app._resizeTimer = setTimeout(() => {
    if (app.state.summary) app.renderCharts();
  }, 150);
});
