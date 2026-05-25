# Dashboard Revamp PRD

## Overview & Goals

Redesign the Claude Code Proxy observability dashboard from a single-page metrics view into a modern, session-aware application with a per-session sidebar, dark/light mode theming, and a responsive, polished UI.

## User Stories

1. **Session Awareness**: As a developer running multiple Claude Code sessions through the proxy, I want to see a sidebar listing all my sessions (with names, ports, and timestamps) so I can switch between them and inspect per-session metrics.

2. **Theme Preference**: As a user, I want to toggle between dark and light modes so the dashboard matches my environment and reduces eye strain.

3. **Modern UX**: As a user, I want a visually polished, responsive dashboard that feels like a contemporary SaaS monitoring tool (Datadog / Grafana-lite vibes) rather than a bare-bones admin page.

## Functional Requirements

### 1. Per-Session Sidebar

**Session Discovery:**
- A collapsible left sidebar that lists all recorded sessions from the observability database.
- Each session card shows: session name, backend model, total requests, tokens used, context percentage used (colored indicator), last active time, and the forwarder port.
- Sessions are sorted by most recent activity first.
- Active sessions (last request within 5 minutes) get a green "live" dot.

**Session Navigation:**
- Clicking a session switches the main content area to show that session's data.
- A "Global Overview" option at the top of the sidebar shows the current full-dashboard view (aggregated across all sessions).

**Session Detail View:**
- When a session is selected, the main area shows:
  - Session metadata (name, model, start time, duration, port)
  - A "context usage" gauge bar (similar to Claude Code's own /context output)
  - Request count, token count, estimated cost for just that session
  - A table of requests filtered to that session
  - A table of tool calls filtered to that session
  - A table of failures filtered to that session
  - A sparkline chart showing token usage over time for that session

**Data Source:**
- Backend needs a new API endpoint: `GET /api/observability/sessions`
  - Returns distinct sessions with metadata (session_name, session_id, first_seen, last_seen, total_requests, total_tokens, backend_model, port inferred from request timestamps/frequency).
- Existing endpoints `requests`, `failures`, `tool-calls` need optional `session_name` or `session_id` query params for filtering.
- Existing `/context-usage` already supports session.

### 2. Dark / Light Mode Toggle

**Theme System:**
- A toggle button in the top-right toolbar (sun/moon icons).
- Theme preference is stored in `localStorage` and persists across page reloads.
- System preference is respected on first visit via `prefers-color-scheme`.

**CSS Architecture:**
- Replace hardcoded colors with CSS custom properties defined on `:root`.
- Provide a `[data-theme="dark"]` variant on `<html>` or `<body>` that overrides the variables.
- Ensure all colors in dashboard.css use variables: background, surface, text, muted, borders, accent blues/greens/reds.
- Chart.js (or custom canvas) colors must also adapt to the active theme.

**Dark Mode Palette:**
- Background: `#0d1117` (GitHub dark surface bg)
- Surface: `#161b22`
- Surface subtle: `#1c2128`
- Text: `#c9d1d9`
- Muted: `#8b949e`
- Border: `#30363d`
- Accent blue: `#58a6ff`
- Accent green: `#3fb950`
- Accent red: `#f85149`
- Accent amber: `#d29922`
- Shadows: subtle, low-opacity

**Light Mode Palette:** (mostly the current palette, refined)
- Background: `#f6f7f9`
- Surface: `#ffffff`
- Surface subtle: `#f0f3f7`
- Text: `#17202a`
- Muted: `#647386`
- Border: `#d9e0ea`
- Accent blue: `#2563eb`
- Accent green: `#0f9f6e`
- Accent red: `#c24141`
- Accent amber: `#b7791f`
- Shadows: current

### 3. Modern, Beautiful, Responsive UI

**Overall Layout:**
- Left sidebar (collapsible on mobile) for session navigation.
- Top app bar with: title, global time window selector, theme toggle, refresh button.
- Main content area with card-based layout.
- Right-click or hamburger menu for additional actions (export, settings, etc).

**Visual Design Principles:**
- **Card美学**: Every metric panel, table, and chart should be a distinct card with rounded corners, subtle shadow, and clean typography.
- **Typography**: Inter font (already in use), with a clear hierarchy — large numbers for metrics, medium weight for headers, monospace for model names and code.
- **Spacing**: Generous whitespace with consistent 16px, 24px, and 32px gutters.
- **Transitions**: Smooth 0.2s CSS transitions for hover states, theme toggle, sidebar collapse, and panel expand/collapse.
- **Responsive**: Single-column layout on mobile, two-column on tablet, full layout on desktop. Tables should be horizontally scrollable with sticky headers.

**Metric Cards:**
- Animated number counters on load.
- Colored icons/symbols next to each metric label (e.g., a bolt for tokens, a dollar for cost).
- Small sparklines (mini charts) in each metric card showing the metric's trend over the selected time window.

**Charts:**
- Replace custom Canvas charts with a lightweight charting library (e.g., **Chart.js** via CDN, or **ApexCharts** if we want richer interactivity) to get tooltips, smooth curves, and area fills.
- Tooltip on hover shows exact values and timestamps.
- Dark mode: axes/gridlines use muted color, area fills have subtle opacity.

**Tables:**
- Sticky headers with subtle bottom border.
- Alternating row backgrounds (zebra striping) in light mode, subtle hover highlight in both modes.
- Status pills: success (green solid), error (red), pending (amber), local optimization (blue).
- Expandable rows: click a row to see request/response JSON payload.
- Pagination or virtual scrolling for the requests table (up to 500 rows).

**Sidebar Design:**
- Width: 280px on desktop, 100% overlay on mobile.
- Toggle button (hamburger or chevron) in the app bar.
- Session cards have: model name as title, session name as subtitle, a horizontal "context bar" showing usage percentage (green -> orange -> red), request count badge, live dot pulse animation.
- A search/filter input at the top of the sidebar to find sessions by name.

**Loading States:**
- Skeleton loaders (shimmer blocks) while data is fetching.
- Spinning refresh icon on the refresh button.
- "No sessions yet" empty state with a friendly illustration or icon.

**Error States:**
- Retry button for failed fetches.
- Toast notifications for errors (brief, dismissible).
- Offline indicator if the proxy is unreachable.

## Technical Requirements

### New / Modified Backend Endpoints

1. `GET /api/observability/sessions`
   - Returns array of session summaries from SQLite DISTINCT queries on `session_name` / `session_id`.
   - Fields: `session_name`, `session_id`, `first_seen`, `last_seen`, `total_requests`, `total_tokens`, `total_cost`, `backend_model`, `context_percentage`.

2. Modify `GET /api/observability/requests`
   - Add optional query params: `session_id` and `session_name`.
   - When present, filter `WHERE session_id = ?` or `WHERE session_name = ?`.

3. Modify `GET /api/observability/failures`
   - Same session filtering.

4. Modify `GET /api/observability/tool-calls`
   - Join via request_id, same session filtering.

5. `GET /api/observability/session/{session_id}/summary`
   - Returns per-session aggregated metrics (token count, cost, latency averages) AND the bucketed time series data for sparklines.

### Frontend Architecture Changes

1. **HTML Structure:**
   Introduce a sidebar `<aside>` and a `<main>` content wrapper.
   The current `<main class="shell">` becomes the global view inside the main area.

2. **CSS:**
   - Migrate all hardcoded colors to CSS variables under `:root` and `[data-theme="dark"]`.
   - Add a `dashboard-theme.css` or inline the theme block in the existing CSS to keep file count managed.
   - Add sidebar styles: `position: fixed`, `z-index`, transition for `transform` on collapse.
   - Add responsive media queries: sidebar collapses to drawer on <768px.
   - Add smooth transitions for theme toggle and interactive states.

3. **JavaScript:**
   - Refactor the monolithic `dashboard.js` into discrete responsibilities (even if still one file, use clear sections):
     - `ThemeManager`: handles localStorage, system preference, CSS var toggle.
     - `SessionManager`: fetches `/api/observability/sessions`, renders sidebar, handles search/filter, click navigation.
     - `DashboardApp`: orchestrates fetch, render, and route between global view and session view.
   - Add a lightweight chart library. Chart.js (CDN) is ~70KB gzipped and handles everything we need. Or stick with custom Canvas if bundle size is critical — but add smooth bezier curves, gradient fills, and tooltip logic.
   - Add an IntersectionObserver to animate metric counters when they scroll into view.
   - Add a ResizeObserver on charts so they redraw cleanly.

4. **State Management:**
   - Add to the `state` object: `currentView: 'global' | 'session'`, `activeSession: null | sessionName`, `theme: 'light' | 'dark'`, `sessions: []`.

### Data Model Changes (Backend)

No schema changes needed to the SQLite tables. The existing `session_name`, `session_id` columns on `requests` are sufficient. We just need new SELECT queries:

```sql
-- /api/observability/sessions
SELECT
    session_name,
    session_id,
    MIN(started_at) AS first_seen,
    MAX(started_at) AS last_seen,
    COUNT(*) AS total_requests,
    COALESCE(SUM(total_tokens), 0) AS total_tokens,
    COALESCE(SUM(estimated_cost), 0) AS total_cost,
    MAX(backend_model) AS backend_model
FROM requests
WHERE session_name IS NOT NULL
GROUP BY session_name, session_id
ORDER BY MAX(started_at_unix) DESC;
```

For the per-session sparkline/time-series (used in the session detail view and the sidebar mini-charts):

```sql
-- Reuse _fetch_series logic but with WHERE session_name = ? and larger time window (all time for the session)
SELECT
    CAST(started_at_unix / ? AS INTEGER) * ? AS bucket,
    SUM(input_tokens) AS input_tokens,
    SUM(output_tokens) AS output_tokens
FROM requests
WHERE session_name = ?
GROUP BY bucket
ORDER BY bucket ASC;
```

## UI/UX Design Specifications

### Color System (Light Theme)

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#f6f7f9` | Page background |
| `--surface` | `#ffffff` | Cards, panels, sidebar |
| `--surface-hover` | `#f8fafc` | Table row hover, button hover |
| `--surface-subtle` | `#f0f3f7` | Alternate rows, badges |
| `--text-primary` | `#17202a` | Headings, primary text |
| `--text-secondary` | `#647386` | Labels, metadata, muted text |
| `--text-tertiary` | `#8b949e` | Disabled, hints |
| `--border` | `#d9e0ea` | Card borders, dividers |
| `--accent-blue` | `#2563eb` | Primary actions, links, active states |
| `--accent-blue-light` | `#dbeafe` | Blue backgrounds |
| `--accent-green` | `#0f9f6e` | Success states |
| `--accent-green-bg` | `#dcfce7` | Success pill backgrounds |
| `--accent-red` | `#c24141` | Errors |
| `--accent-red-bg` | `#fee2e2` | Error pill backgrounds |
| `--accent-amber` | `#b7791f` | Warnings |
| `--shadow-card` | `0 4px 12px rgba(30, 41, 59, 0.06)` | Card shadows |
| `--shadow-elevated` | `0 8px 24px rgba(30, 41, 59, 0.08)` | Elevated elements |

### Color System (Dark Theme)

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#0d1117` | Page background |
| `--surface` | `#161b22` | Cards, panels, sidebar |
| `--surface-hover` | `#1c2128` | Table row hover |
| `--surface-subtle` | `#1c2128` | Alternate rows, badges |
| `--text-primary` | `#c9d1d9` | Headings, primary text |
| `--text-secondary` | `#8b949e` | Labels, metadata |
| `--text-tertiary` | `#6e7681` | Disabled, hints |
| `--border` | `#30363d` | Card borders, dividers |
| `--accent-blue` | `#58a6ff` | Primary actions, links |
| `--accent-blue-light` | `#0d1117` | Blue backgrounds |
| `--accent-green` | `#3fb950` | Success states |
| `--accent-green-bg` | `rgba(46, 160, 67, 0.15)` | Success pill backgrounds |
| `--accent-red` | `#f85149` | Errors |
| `--accent-red-bg` | `rgba(248, 81, 73, 0.15)` | Error pill backgrounds |
| `--accent-amber` | `#d29922` | Warnings |
| `--shadow-card` | `0 4px 12px rgba(0, 0, 0, 0.3)` | Card shadows |
| `--shadow-elevated` | `0 8px 24px rgba(0, 0, 0, 0.4)` | Elevated elements |

### Layout Specs

- **Sidebar**:
  - Desktop: fixed left, 280px width, full viewport height.
  - Mobile: fixed left, 280px width, full viewport height, `transform: translateX(-100%)` when hidden. Toggle pushes content or overlays with backdrop.
  - Background: `--surface`, border-right: 1px solid `--border`.
  - Top: Title "Sessions" + search input + "Global Overview" card.
  - Middle: Scrollable session list.
  - Session card: padding 12px 16px, border-bottom 1px solid `--border`, hover background `--surface-hover`, active state has a 2px `--accent-blue` left border.
  - Session card content:
    - Line 1: `backend_model` in monospace, truncated with ellipsis.
    - Line 2: `session_name` in small secondary text.
    - Line 3: Horizontal progress bar for context usage (green 0-50%, amber 50-80%, red 80-100%). Width = 100px, height = 4px, border-radius 2px.
    - Line 4: `total_requests` requests · `total_tokens` tokens · `last_seen` relative time (e.g., "2m ago").
    - Live dot: 6px circle, green (#`--accent-green`), pulse animation when last_seen < 5 min.

- **Top Bar**:
  - Height: 64px.
  - Background: `--surface`.
  - Border-bottom: 1px solid `--border`.
  - Left: Hamburger menu (opens sidebar) + title "Claude Proxy".
  - Right: Time window selector (same dropdown, styled), theme toggle button (sun/moon), refresh button (circular, icon only).

- **Metric Cards**:
  - Grid: 6 columns on desktop, 3 on tablet, 2 on mobile.
  - Card: padding 20px, border-radius 12px, background `--surface`, border 1px solid `--border`, box-shadow `--shadow-card`.
  - Label: font-size 13px, color `--text-secondary`, uppercase, letter-spacing 0.5px.
  - Value: font-size 32px, font-weight 600, color `--text-primary`, margin-top 8px.
  - Mini-sparkline: 80px wide x 24px tall, aligned right, at top of card. Stroke color matches metric theme.
  - Sub-label: font-size 12px, color `--text-tertiary`, margin-top 4px.

- **Tables**:
  - Container: border-radius 8px, border 1px solid `--border`, overflow hidden.
  - Header: background `--surface-subtle`, sticky top, font-weight 600, font-size 13px, color `--text-secondary`, padding 10px 12px.
  - Rows: padding 12px, alternating backgrounds (transparent / `--surface-subtle` at 50% opacity).
  - Hover: background `--surface-hover`.
  - Cell text: 13px, monospace for code/model columns.
  - Status pills: padding 2px 8px, border-radius 999px, font-size 11px, font-weight 500.

- **Charts**:
  - Container: same card style as tables.
  - Height: 260px.
  - Area fill: gradient from line color (30% opacity) to transparent.
  - Axes: gridlines dashed, color `--border`, labels `--text-tertiary`.
  - Tooltip: background `--surface`, border 1px solid `--border`, border-radius 6px, padding 8px 12px, shadow `--shadow-card`.

## Animation & Interaction Specs

- **Theme Toggle**: `0.3s` ease on all color-related properties. Sun/moon icons rotate 180deg and swap with a fade.
- **Sidebar Toggle**: `0.25s` ease-out on `transform`. Backdrop on mobile fades in at `opacity 0 -> 0.5` over `0.2s`.
- **Metric Counters**: Numbers count up from 0 over `0.8s` with `ease-out`.
- **Table Row Hover**: `0.15s` background-color transition.
- **Live Dot Pulse**: CSS `@keyframes` — scale 1 -> 1.5 -> 1, opacity 1 -> 0.6 -> 1, duration 2s, infinite.
- **Card Hover**: Subtle shadow increase (`0.2s` transition), translateY(-1px).
- **Panel Collapse**: Height transition `0.25s` ease. Chevron rotates 90deg.
- **Page Load**: Staggered fade-in for cards — each delays by `0.05s`, opacity 0 -> 1, translateY(8px) -> translateY(0).

## Responsive Breakpoints

| Breakpoint | Layout |
|------------|--------|
| >= 1280px (xl) | Full layout: sidebar always visible, 6 metric columns, 2 chart columns, full tables |
| >= 1024px (lg) | Sidebar always visible, 4 metric columns, 2 chart columns |
| >= 768px (md) | Sidebar collapsible (drawer), 3 metric columns, 1 chart column |
| < 768px (sm) | Sidebar hidden by default (drawer), 2 metric columns, stacked everything |
| < 640px (xs) | Single column all, metric cards stack vertically, tables scroll horizontally |

## Non-Goals (Explicitly Out of Scope)

1. **No real-time WebSocket updates** — polling every 5s is sufficient. WebSockets add significant backend complexity.
2. **No backend auth changes** — the existing `x_api_key` header validation remains unchanged.
3. **No export to PDF/CSV** — can be added later as a feature.
4. **No multi-provider support** — this dashboard remains Nebius-focused as per project scope.
5. **No historical comparison charts** (e.g., today vs yesterday) — sparklines provide enough trend context.

## Implementation Phases

### Phase 1: Backend API Enhancement (Foundation)
- [ ] Add `GET /api/observability/sessions` endpoint to `src/observability/routes.py`.
- [ ] Add optional `session_id` / `session_name` query params to existing list endpoints (`requests`, `failures`, `tool-calls`).
- [ ] Add `GET /api/observability/session/{session_name}/summary` endpoint (reuse `_fetch_series`, `_fetch_model_stats` but filtered).
- [ ] Update `tests/test_observability.py` with coverage for new endpoints.
- [ ] Verify endpoint responses are correct via manual curl/`pytest`.

### Phase 2: CSS Theme System (Infrastructure)
- [ ] Refactor `dashboard.css` into CSS variable-based theming.
- [ ] Define all tokens for both light and dark themes.
- [ ] Add `[data-theme="dark"]` overrides.
- [ ] Add smooth transitions for theme property changes.
- [ ] Test both themes manually by toggling the attribute.

### Phase 3: Frontend Layout & Sidebar (Structure)
- [ ] Restructure `dashboard.html` to have sidebar + main content wrapper.
- [ ] Implement sidebar HTML with session cards, search input, global overview button.
- [ ] Add sidebar CSS (fixed positioning, collapsible states, responsive drawer).
- [ ] Add top bar with hamburger, title, controls.
- [ ] Add sidebar JS: fetch `/api/observability/sessions`, render list, handle click, highlight active.
- [ ] Add session search/filter in sidebar.
- [ ] Add live dot and relative time formatting.

### Phase 4: Global View Polish (Presentational)
- [ ] Add metric card sparklines (mini charts from summary series data).
- [ ] Replace custom canvas charts with Chart.js (CDN) or enhance custom canvas with bezier curves, gradients, and tooltips.
- [ ] Add animated number counters for metrics.
- [ ] Add table hover states, zebra striping, and responsive horizontal scroll containers.
- [ ] Polish status pills, empty states, and loading skeletons.
- [ ] Update `setupPanelControls` to use smooth height transitions.

### Phase 5: Session Detail View (Feature)
- [ ] Add session detail HTML template/section (hidden by default).
- [ ] Implement session selection state change: hide global view, show session detail.
- [ ] Fetch filtered data when session is selected.
- [ ] Render session-specific metrics, charts, and tables.
- [ ] Add a "Back to Global View" button / breadcrumb.
- [ ] Add context usage gauge bar (semicircular or horizontal bar with percentage).

### Phase 6: Theme Toggle & Polish (Finishing)
- [ ] Add theme toggle button to top bar with sun/moon SVG icons.
- [ ] Implement `ThemeManager` class: read `localStorage`, detect `prefers-color-scheme`, toggle handler.
- [ ] Ensure Chart.js or canvas charts redraw with theme-appropriate colors on toggle.
- [ ] Ensure all elements respect theme (scrollbars, canvas, tables, inputs).
- [ ] Add responsive handling for all breakpoints: sidebar drawer on mobile, metric grid collapse, table scrolling.
- [ ] Final QA: test on Safari, Chrome, Firefox. Test dark mode, light mode, mobile viewport.

## Acceptance Criteria

1. **Sidebar Sessions**: A user opens the dashboard and sees a sidebar with all recorded sessions. Clicking a session switches the view to show only that session's data. The "Global Overview" option returns to the aggregate view.

2. **Session Context Bar**: Each session card in the sidebar shows a horizontal progress bar for context usage percentage (green for <50%, amber for 50-80%, red for >80%).

3. **Dark Mode**: Clicking the theme toggle instantly switches the entire dashboard to dark mode. All colors, charts, tables, and inputs adapt. Preference persists across reloads. First-time users get their OS preference.

4. **Responsive**: On a 375px wide mobile device, the dashboard is usable: sidebar is a drawer accessible via hamburger, metrics stack vertically, tables scroll horizontally with sticky headers.

5. **Live Sessions**: Sessions with activity in the last 5 minutes show a pulsing green dot.

6. **Performance**: Dashboard loads and renders within 2 seconds for up to 1000 requests / 50 sessions. Auto-refresh every 5 seconds does not cause layout shift.

7. **Chart Tooltips**: Hovering over any chart point shows a tooltip with the exact value and timestamp.

8. **Accessibility**: All interactive elements are keyboard-navigable. Tables have `aria-label`s. Color contrast meets WCAG AA for both themes.

## Appendix: File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/observability/routes.py` | Modify | Add sessions endpoint, query params for filtering, session summary endpoint |
| `src/observability/store.py` | Modify | Add `fetch_sessions()`, `fetch_session_summary()`, prepare filtered versions of `_fetch_series` etc |
| `src/observability/static/dashboard.html` | Rewrite | Restructure with sidebar, top bar, theme toggle, session detail view |
| `src/observability/static/dashboard.css` | Rewrite | Full CSS variable theming, sidebar styles, responsive breakpoints, animations |
| `src/observability/static/dashboard.js` | Rewrite | Refactor into ThemeManager, SessionManager, ChartController, DashboardApp |
| `tests/test_observability.py` | Modify | Add tests for new endpoints |

---
*End of PRD*
