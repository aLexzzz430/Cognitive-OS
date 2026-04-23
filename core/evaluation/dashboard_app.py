from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from core.evaluation.metrics_panel import build_eval_metrics_panel_from_paths


DASHBOARD_UI_VERSION = "conos.dashboard_ui/v1"


class DashboardHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _default_input_paths() -> list[Path]:
    root = Path.cwd()
    return [root / "runtime", root / "reports", root / "audit"]


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _metric_label(name: str) -> str:
    labels = {
        "verified_success_rate": "Verified Success",
        "human_intervention_rate": "Human Intervention",
        "recovery_rate": "Recovery",
        "verifier_coverage": "Verifier Coverage",
    }
    return labels.get(str(name), str(name).replace("_", " ").title())


def _panel_payload(paths: Iterable[str | Path]) -> Dict[str, Any]:
    panel = build_eval_metrics_panel_from_paths(paths)
    panel["dashboard_ui_version"] = DASHBOARD_UI_VERSION
    panel["generated_at"] = _generated_at()
    return panel


def render_dashboard_html(panel: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(panel), ensure_ascii=False, default=str)
    escaped_payload = html.escape(payload, quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cognitive OS Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8f5;
      --surface: #ffffff;
      --surface-alt: #eef3ed;
      --ink: #17201b;
      --muted: #637168;
      --line: #d8ded7;
      --green: #277b5d;
      --teal: #237485;
      --amber: #9b6b14;
      --red: #a34545;
      --blue: #356fa5;
      --shadow: 0 10px 28px rgba(23, 32, 27, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-width: 320px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    button, input, select {{ font: inherit; }}
    .shell {{ min-height: 100vh; }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 248, 245, 0.94);
      backdrop-filter: blur(16px);
    }}
    .topbar-inner {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      width: min(1280px, calc(100% - 32px));
      margin: 0 auto;
      padding: 14px 0;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .mark {{
      width: 34px;
      height: 34px;
      border: 1px solid #b8c7bd;
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      display: grid;
      place-items: center;
      flex: 0 0 auto;
    }}
    .mark svg {{ width: 23px; height: 23px; }}
    .brand h1 {{
      font-size: 17px;
      line-height: 1.1;
      margin: 0;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .brand .sub {{
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .actions {{ display: flex; align-items: center; gap: 8px; }}
    .icon-button {{
      width: 36px;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      display: grid;
      place-items: center;
      cursor: pointer;
    }}
    .icon-button:hover {{ border-color: #a9b7ad; box-shadow: 0 4px 12px rgba(23, 32, 27, 0.08); }}
    .icon-button svg {{ width: 18px; height: 18px; }}
    main {{
      width: min(1280px, calc(100% - 32px));
      margin: 0 auto;
      padding: 22px 0 40px;
    }}
    .status-band {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
      gap: 16px;
      align-items: stretch;
      margin-bottom: 16px;
    }}
    .overview, .health {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .overview h2, .health h2, .section-title {{
      margin: 0 0 12px;
      font-size: 14px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .fact {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfa;
      min-height: 74px;
    }}
    .fact .label {{ color: var(--muted); font-size: 12px; }}
    .fact .value {{ margin-top: 9px; font-size: 22px; font-weight: 760; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      position: relative;
      overflow: hidden;
      min-height: 154px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .metric::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 5px;
      background: var(--accent, var(--green));
    }}
    .metric-header {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }}
    .metric-name {{ font-size: 13px; font-weight: 720; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      background: var(--surface-alt);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .metric-value {{
      margin-top: 18px;
      font-size: 32px;
      line-height: 1;
      font-weight: 780;
      letter-spacing: 0;
    }}
    .metric-foot {{
      margin-top: 13px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .bar {{
      height: 8px;
      margin-top: 15px;
      border-radius: 999px;
      background: #e5e9e3;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      width: var(--bar-width, 0%);
      background: var(--accent, var(--green));
    }}
    .content-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
    }}
    .table-wrap, .side-panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .table-head, .side-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}
    th, td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th {{
      color: var(--muted);
      font-weight: 640;
      background: #fbfcfa;
      font-size: 12px;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .empty {{
      padding: 34px 16px;
      color: var(--muted);
      text-align: center;
      border-top: 1px solid var(--line);
    }}
    .side-body {{ padding: 16px; display: grid; gap: 12px; }}
    .route-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfcfa;
    }}
    .route-row code {{
      color: var(--ink);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .status-dot {{
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--green);
      flex: 0 0 auto;
    }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    @media (max-width: 980px) {{
      .status-band, .content-grid {{ grid-template-columns: 1fr; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
      .topbar-inner, main {{ width: min(100% - 20px, 1280px); }}
      .topbar-inner {{ align-items: flex-start; }}
      .actions {{ align-self: center; }}
      .overview-grid, .metric-grid {{ grid-template-columns: 1fr; }}
      .metric-value {{ font-size: 28px; }}
      th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <div class="mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <path d="M4 12h4l2-7 4 14 2-7h4"></path>
              <circle cx="5" cy="12" r="1.4" fill="currentColor" stroke="none"></circle>
              <circle cx="19" cy="12" r="1.4" fill="currentColor" stroke="none"></circle>
            </svg>
          </div>
          <div>
            <h1>Cognitive OS</h1>
            <div class="sub" id="generatedAt">Dashboard</div>
          </div>
        </div>
        <div class="actions">
          <button class="icon-button" id="refreshButton" title="Refresh" aria-label="Refresh">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M20 11a8 8 0 0 0-14.5-4.5L4 8"></path>
              <path d="M4 4v4h4"></path>
              <path d="M4 13a8 8 0 0 0 14.5 4.5L20 16"></path>
              <path d="M20 20v-4h-4"></path>
            </svg>
          </button>
        </div>
      </div>
    </header>

    <main>
      <section class="status-band">
        <div class="overview">
          <h2>Runtime Overview</h2>
          <div class="overview-grid">
            <div class="fact"><div class="label">Runs</div><div class="value" id="runCount">0</div></div>
            <div class="fact"><div class="label">Sources</div><div class="value" id="sourceCount">0</div></div>
            <div class="fact"><div class="label">Schema</div><div class="value mono" id="schemaValue">-</div></div>
          </div>
        </div>
        <div class="health">
          <h2>Entrypoints</h2>
          <div class="side-body" style="padding:0">
            <div class="route-row"><code>conos run</code><span class="status-dot"></span></div>
            <div class="route-row"><code>conos eval</code><span class="status-dot"></span></div>
            <div class="route-row"><code>conos app</code><span class="status-dot"></span></div>
            <div class="route-row"><code>conos ui</code><span class="status-dot"></span></div>
            <div class="route-row"><code>conos preflight</code><span class="status-dot"></span></div>
          </div>
        </div>
      </section>

      <section class="metric-grid" id="metricGrid"></section>

      <section class="content-grid">
        <div class="table-wrap">
          <div class="table-head">
            <h2 class="section-title" style="margin:0">Runs</h2>
            <span class="pill" id="runBadge">0 total</span>
          </div>
          <table aria-label="Run summaries">
            <thead>
              <tr>
                <th>Run</th>
                <th>Reward</th>
                <th>Verifier</th>
                <th>Recovery</th>
                <th>Human</th>
              </tr>
            </thead>
            <tbody id="runRows"></tbody>
          </table>
          <div class="empty" id="emptyRuns">No audit runs found.</div>
        </div>
        <aside class="side-panel">
          <div class="side-head">
            <h2 class="section-title" style="margin:0">Audit Sources</h2>
            <span class="pill" id="sourceBadge">0</span>
          </div>
          <div class="side-body" id="sourceList"></div>
        </aside>
      </section>
    </main>
  </div>

  <script type="application/json" id="initialPanel">{escaped_payload}</script>
  <script>
    const initialPanel = JSON.parse(document.getElementById('initialPanel').textContent);
    const metricOrder = ['verified_success_rate', 'human_intervention_rate', 'recovery_rate', 'verifier_coverage'];
    const metricLabels = {{
      verified_success_rate: 'Verified Success',
      human_intervention_rate: 'Human Intervention',
      recovery_rate: 'Recovery',
      verifier_coverage: 'Verifier Coverage'
    }};
    const accents = {{
      verified_success_rate: '#277b5d',
      human_intervention_rate: '#9b6b14',
      recovery_rate: '#237485',
      verifier_coverage: '#356fa5'
    }};

    function percent(value) {{
      return value === null || value === undefined ? 'n/a' : `${{(value * 100).toFixed(1)}}%`;
    }}

    function setText(id, value) {{
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }}

    function render(panel) {{
      const metrics = panel.metrics || {{}};
      const runs = panel.runs || [];
      const sources = panel.source_files || [];
      setText('generatedAt', panel.generated_at ? `Generated ${{panel.generated_at}}` : 'Dashboard');
      setText('runCount', String(panel.run_count || 0));
      setText('sourceCount', String(sources.length));
      setText('schemaValue', String(panel.schema_version || '-').replace('conos.', ''));
      setText('runBadge', `${{runs.length}} total`);
      setText('sourceBadge', String(sources.length));

      const metricGrid = document.getElementById('metricGrid');
      metricGrid.innerHTML = '';
      metricOrder.forEach((name) => {{
        const metric = metrics[name] || {{}};
        const value = metric.value;
        const width = value === null || value === undefined ? 0 : Math.max(0, Math.min(100, value * 100));
        const node = document.createElement('article');
        node.className = 'metric';
        node.style.setProperty('--accent', accents[name] || '#277b5d');
        node.style.setProperty('--bar-width', `${{width}}%`);
        node.innerHTML = `
          <div class="metric-header">
            <div class="metric-name">${{metricLabels[name] || name}}</div>
            <span class="pill">${{metric.status || 'ok'}}</span>
          </div>
          <div class="metric-value">${{percent(value)}}</div>
          <div class="bar"><span></span></div>
          <div class="metric-foot">${{metric.numerator || 0}} / ${{metric.denominator || 0}}</div>
        `;
        metricGrid.appendChild(node);
      }});

      const rows = document.getElementById('runRows');
      rows.innerHTML = '';
      runs.slice(0, 80).forEach((run) => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono" title="${{run.source_path || ''}}">${{run.run_id || 'unknown'}}</td>
          <td>${{Number(run.total_reward || 0).toFixed(2)}}</td>
          <td>${{run.verification_passed ? 'passed' : (run.verification_failed ? 'failed' : (run.verifier_covered ? 'covered' : 'open'))}}</td>
          <td>${{run.recovery_events || 0}}</td>
          <td>${{run.human_intervention_events || 0}}</td>
        `;
        rows.appendChild(tr);
      }});
      document.getElementById('emptyRuns').style.display = runs.length ? 'none' : 'block';

      const sourceList = document.getElementById('sourceList');
      sourceList.innerHTML = '';
      if (!sources.length) {{
        const empty = document.createElement('div');
        empty.className = 'muted';
        empty.textContent = 'No source files.';
        sourceList.appendChild(empty);
      }} else {{
        sources.slice(0, 20).forEach((source) => {{
          const item = document.createElement('div');
          item.className = 'route-row';
          item.innerHTML = `<code title="${{source}}">${{source}}</code><span class="status-dot"></span>`;
          sourceList.appendChild(item);
        }});
      }}
    }}

    async function refresh() {{
      if (location.protocol.startsWith('http')) {{
        try {{
          const response = await fetch('/api/panel', {{cache: 'no-store'}});
          if (response.ok) {{
            render(await response.json());
            return;
          }}
        }} catch (error) {{}}
      }}
      render(initialPanel);
    }}

    document.getElementById('refreshButton').addEventListener('click', refresh);
    render(initialPanel);
  </script>
</body>
</html>
"""


def write_dashboard_html(paths: Iterable[str | Path], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard_html(_panel_payload(paths)), encoding="utf-8")
    return output


def serve_dashboard(paths: Iterable[str | Path], *, host: str = "127.0.0.1", port: int = 8766) -> int:
    source_paths = [Path(path) for path in paths]

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/api/panel":
                body = json.dumps(_panel_payload(source_paths), ensure_ascii=False, default=str).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if route in {"/", "/index.html"}:
                body = render_dashboard_html(_panel_payload(source_paths)).encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if route == "/health":
                self._send(200, b"ok", "text/plain; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    try:
        server = DashboardHTTPServer((host, int(port)), DashboardHandler)
    except OSError as exc:
        print(f"dashboard_error=failed_to_bind host={host} port={int(port)} detail={exc}", flush=True)
        return 1
    url = f"http://{host}:{int(port)}"
    print(f"Cognitive OS dashboard: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard stopped.", flush=True)
    finally:
        server.server_close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conos ui",
        description="Render or serve the Cognitive OS local dashboard.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Audit JSON/JSONL files or directories. Defaults to runtime/, reports/, and audit/.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", default="", help="Write a static dashboard HTML file and exit.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = [Path(path) for path in args.paths] if args.paths else _default_input_paths()
    if args.output:
        output = write_dashboard_html(paths, args.output)
        print(f"dashboard_html={output}")
        return 0
    return serve_dashboard(paths, host=str(args.host), port=int(args.port))
