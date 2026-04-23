from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, Sequence
import webbrowser

from core.evaluation.dashboard_app import write_dashboard_html
from core.evaluation.metrics_panel import build_eval_metrics_panel_from_paths


APP_VERSION = "conos.desktop_app/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_DASHBOARD_PATH = Path("runtime/ui/dashboard.html")
METRIC_ORDER = (
    "verified_success_rate",
    "human_intervention_rate",
    "recovery_rate",
    "verifier_coverage",
)
METRIC_LABELS = {
    "verified_success_rate": "Verified Success",
    "human_intervention_rate": "Human Intervention",
    "recovery_rate": "Recovery",
    "verifier_coverage": "Verifier Coverage",
}


def default_input_paths(root: str | Path | None = None) -> list[Path]:
    base = Path(root) if root is not None else Path.cwd()
    return [base / "runtime", base / "reports", base / "audit"]


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _metric_display(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _paths(paths: Iterable[str | Path] | None) -> list[Path]:
    return [Path(path) for path in paths] if paths else default_input_paths()


def build_app_state(paths: Iterable[str | Path] | None = None) -> Dict[str, Any]:
    input_paths = _paths(paths)
    panel = build_eval_metrics_panel_from_paths(input_paths)
    raw_metrics = dict(panel.get("metrics", {})) if isinstance(panel.get("metrics"), dict) else {}
    metrics: Dict[str, Dict[str, Any]] = {}
    for name in METRIC_ORDER:
        metric = dict(raw_metrics.get(name, {})) if isinstance(raw_metrics.get(name), dict) else {}
        value = metric.get("value")
        metrics[name] = {
            "label": METRIC_LABELS.get(name, name.replace("_", " ").title()),
            "value": value,
            "display": _metric_display(value),
            "count": f"{metric.get('numerator', 0)} / {metric.get('denominator', 0)}",
            "status": metric.get("status", "ok"),
        }
    source_files = panel.get("source_files", [])
    if not isinstance(source_files, list):
        source_files = []
    return {
        "schema_version": APP_VERSION,
        "generated_at": _generated_at(),
        "project_root": str(Path.cwd()),
        "input_paths": [str(path) for path in input_paths],
        "run_count": int(panel.get("run_count", 0) or 0),
        "source_count": len(source_files),
        "metrics": metrics,
        "panel": panel,
    }


def write_dashboard_snapshot(
    paths: Iterable[str | Path] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    output = Path(output_path) if output_path is not None else Path.cwd() / DEFAULT_DASHBOARD_PATH
    return write_dashboard_html(_paths(paths), output)


def run_preflight(*, strict_dev: bool = True) -> tuple[int, str]:
    script = Path.cwd() / "scripts" / "check_runtime_preflight.py"
    cmd = [sys.executable, str(script)]
    if strict_dev:
        cmd.append("--strict-dev")
    completed = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, check=False)
    return int(completed.returncode), completed.stdout + completed.stderr


class DesktopApp:
    def __init__(self, root: Any, paths: Iterable[str | Path] | None = None, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.paths = _paths(paths)
        self.host = str(host)
        self.port = int(port)
        self.dashboard_process: subprocess.Popen[str] | None = None
        self.metric_vars: Dict[str, Any] = {}
        self.status_var = tk.StringVar(value="Ready")
        self.path_var = tk.StringVar(value="  ".join(str(path) for path in self.paths))

        self.root.title("Cognitive OS")
        self.root.geometry("1080x720")
        self.root.minsize(900, 620)
        self.root.configure(bg="#f7f8f5")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._style()
        self._build()
        self.refresh()

    def _style(self) -> None:
        style = self.ttk.Style()
        style.theme_use("clam")
        style.configure("Shell.TFrame", background="#f7f8f5")
        style.configure("Surface.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Title.TLabel", background="#f7f8f5", foreground="#17201b", font=("Helvetica", 22, "bold"))
        style.configure("Sub.TLabel", background="#f7f8f5", foreground="#637168", font=("Helvetica", 12))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#17201b", font=("Helvetica", 12, "bold"))
        style.configure("MetricValue.TLabel", background="#ffffff", foreground="#17201b", font=("Helvetica", 23, "bold"))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#637168", font=("Helvetica", 11))
        style.configure("TButton", padding=(12, 8))

    def _build(self) -> None:
        shell = self.ttk.Frame(self.root, style="Shell.TFrame", padding=18)
        shell.pack(fill="both", expand=True)

        header = self.ttk.Frame(shell, style="Shell.TFrame")
        header.pack(fill="x")
        title_box = self.ttk.Frame(header, style="Shell.TFrame")
        title_box.pack(side="left", fill="x", expand=True)
        self.ttk.Label(title_box, text="Cognitive OS", style="Title.TLabel").pack(anchor="w")
        self.ttk.Label(title_box, text="Local control app for runtime health, evaluation, and dashboard access.", style="Sub.TLabel").pack(anchor="w", pady=(4, 0))
        actions = self.ttk.Frame(header, style="Shell.TFrame")
        actions.pack(side="right")
        self.ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left", padx=(0, 8))
        self.ttk.Button(actions, text="Preflight", command=self.preflight).pack(side="left", padx=(0, 8))
        self.ttk.Button(actions, text="Open Dashboard", command=self.open_dashboard).pack(side="left")

        path_card = self.ttk.Frame(shell, style="Surface.TFrame", padding=14)
        path_card.pack(fill="x", pady=(18, 12))
        self.ttk.Label(path_card, text="Input Sources", style="CardTitle.TLabel").pack(anchor="w")
        self.ttk.Label(path_card, textvariable=self.path_var, style="Muted.TLabel", wraplength=980).pack(anchor="w", pady=(8, 0))

        metrics = self.ttk.Frame(shell, style="Shell.TFrame")
        metrics.pack(fill="x", pady=(0, 12))
        for index, name in enumerate(METRIC_ORDER):
            card = self.ttk.Frame(metrics, style="Surface.TFrame", padding=14)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            metrics.columnconfigure(index, weight=1)
            value_var = self.tk.StringVar(value="n/a")
            count_var = self.tk.StringVar(value="0 / 0")
            status_var = self.tk.StringVar(value="not_applicable")
            self.metric_vars[name] = {"value": value_var, "count": count_var, "status": status_var}
            self.ttk.Label(card, text=METRIC_LABELS[name], style="CardTitle.TLabel").pack(anchor="w")
            self.ttk.Label(card, textvariable=value_var, style="MetricValue.TLabel").pack(anchor="w", pady=(12, 0))
            self.ttk.Label(card, textvariable=count_var, style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
            self.ttk.Label(card, textvariable=status_var, style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

        body = self.ttk.PanedWindow(shell, orient="horizontal")
        body.pack(fill="both", expand=True)
        runs_panel = self.ttk.Frame(body, style="Surface.TFrame", padding=12)
        log_panel = self.ttk.Frame(body, style="Surface.TFrame", padding=12)
        body.add(runs_panel, weight=3)
        body.add(log_panel, weight=2)

        self.ttk.Label(runs_panel, text="Runs", style="CardTitle.TLabel").pack(anchor="w")
        self.runs = self.ttk.Treeview(
            runs_panel,
            columns=("reward", "verifier", "recovery", "human"),
            show="tree headings",
            height=15,
        )
        self.runs.heading("#0", text="Run")
        self.runs.heading("reward", text="Reward")
        self.runs.heading("verifier", text="Verifier")
        self.runs.heading("recovery", text="Recovery")
        self.runs.heading("human", text="Human")
        self.runs.column("#0", width=260, anchor="w")
        self.runs.column("reward", width=90, anchor="center")
        self.runs.column("verifier", width=110, anchor="center")
        self.runs.column("recovery", width=90, anchor="center")
        self.runs.column("human", width=80, anchor="center")
        self.runs.pack(fill="both", expand=True, pady=(10, 0))

        tools = self.ttk.Frame(log_panel, style="Surface.TFrame")
        tools.pack(fill="x")
        self.ttk.Label(tools, text="Actions", style="CardTitle.TLabel").pack(anchor="w")
        self.ttk.Button(tools, text="Export HTML", command=self.export_dashboard).pack(fill="x", pady=(10, 0))
        self.ttk.Button(tools, text="Start Live Dashboard", command=self.start_live_dashboard).pack(fill="x", pady=(8, 0))
        self.ttk.Button(tools, text="Stop Live Dashboard", command=self.stop_live_dashboard).pack(fill="x", pady=(8, 12))
        self.ttk.Label(log_panel, textvariable=self.status_var, style="Muted.TLabel").pack(anchor="w")
        self.log = self.tk.Text(log_panel, height=14, bg="#fbfcfa", fg="#17201b", relief="solid", borderwidth=1, wrap="word")
        self.log.pack(fill="both", expand=True, pady=(10, 0))

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")

    def refresh(self) -> None:
        state = build_app_state(self.paths)
        for name, metric in state["metrics"].items():
            vars_for_metric = self.metric_vars[name]
            vars_for_metric["value"].set(str(metric["display"]))
            vars_for_metric["count"].set(str(metric["count"]))
            vars_for_metric["status"].set(str(metric["status"]))
        self.runs.delete(*self.runs.get_children())
        runs = state["panel"].get("runs", [])
        if isinstance(runs, list):
            for run in runs[:100]:
                if not isinstance(run, dict):
                    continue
                verifier = "passed" if run.get("verification_passed") else "covered" if run.get("verifier_covered") else "open"
                self.runs.insert(
                    "",
                    "end",
                    text=str(run.get("run_id") or "unknown"),
                    values=(
                        f"{float(run.get('total_reward') or 0.0):.2f}",
                        verifier,
                        str(run.get("recovery_events", 0)),
                        str(run.get("human_intervention_events", 0)),
                    ),
                )
        self.status_var.set(f"Runs: {state['run_count']}   Sources: {state['source_count']}   Updated: {state['generated_at']}")
        self._append_log(f"Refreshed app state from {len(self.paths)} source path(s).")

    def preflight(self) -> None:
        code, output = run_preflight(strict_dev=True)
        self._append_log(output)
        self.status_var.set("Preflight passed." if code == 0 else "Preflight needs attention.")

    def export_dashboard(self) -> Path:
        output = write_dashboard_snapshot(self.paths)
        self._append_log(f"dashboard_html={output}")
        self.status_var.set(f"Dashboard exported: {output}")
        return output

    def open_dashboard(self) -> None:
        output = self.export_dashboard()
        webbrowser.open(output.resolve().as_uri())

    def start_live_dashboard(self) -> None:
        if self.dashboard_process is not None and self.dashboard_process.poll() is None:
            webbrowser.open(f"http://{self.host}:{self.port}")
            self.status_var.set("Live dashboard already running.")
            return
        cmd = [
            sys.executable,
            str(Path.cwd() / "scripts" / "conos.py"),
            "ui",
            *[str(path) for path in self.paths],
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self.dashboard_process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        url = f"http://{self.host}:{self.port}"
        webbrowser.open(url)
        self._append_log(f"live_dashboard={url}")
        self.status_var.set(f"Live dashboard started: {url}")

    def stop_live_dashboard(self) -> None:
        if self.dashboard_process is None or self.dashboard_process.poll() is not None:
            self.status_var.set("No live dashboard process is running.")
            return
        self.dashboard_process.terminate()
        self.dashboard_process = None
        self.status_var.set("Live dashboard stopped.")
        self._append_log("Live dashboard stopped.")

    def _on_close(self) -> None:
        self.stop_live_dashboard()
        self.root.destroy()


def launch_desktop_app(paths: Iterable[str | Path] | None = None, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    import tkinter as tk

    root = tk.Tk()
    DesktopApp(root, paths, host=host, port=port)
    root.mainloop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conos app",
        description="Launch the Cognitive OS local desktop app.",
    )
    parser.add_argument("paths", nargs="*", help="Audit JSON/JSONL files or directories. Defaults to runtime/, reports/, and audit/.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--summary-json", action="store_true", help="Print app state JSON and exit without opening the desktop UI.")
    parser.add_argument("--write-dashboard", default="", help="Write a static dashboard HTML file and exit without opening the desktop UI.")
    parser.add_argument("--preflight", action="store_true", help="Run strict preflight and exit without opening the desktop UI.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = _paths(args.paths)
    if args.summary_json:
        print(json.dumps(build_app_state(paths), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.write_dashboard:
        output = write_dashboard_snapshot(paths, args.write_dashboard)
        print(f"dashboard_html={output}")
        return 0
    if args.preflight:
        code, output = run_preflight(strict_dev=True)
        print(output, end="")
        return code
    return launch_desktop_app(paths, host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    raise SystemExit(main())
