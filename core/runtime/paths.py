from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


RUNTIME_PATHS_VERSION = "conos.runtime_paths/v1"

ENV_RUNTIME_HOME = "CONOS_RUNTIME_HOME"
ENV_STATE_DB = "CONOS_STATE_DB"
ENV_RUNS_ROOT = "CONOS_RUNS_ROOT"
ENV_LOG_DIR = "CONOS_LOG_DIR"

DEFAULT_RUNTIME_HOME = Path.home() / ".conos" / "runtime"
DEFAULT_STATE_DB_NAME = "conos.sqlite"
DEFAULT_SERVICE_LABEL = "com.conos.runtime"


def _path_from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "")
    if value:
        return Path(value).expanduser()
    return fallback.expanduser()


@dataclass(frozen=True)
class RuntimePaths:
    """Filesystem layout for the local-first Con OS runtime service."""

    runtime_home: Path = DEFAULT_RUNTIME_HOME
    state_db: Optional[Path] = None
    runs_root: Optional[Path] = None
    logs_dir: Optional[Path] = None
    snapshots_dir: Optional[Path] = None
    soak_dir: Optional[Path] = None

    @classmethod
    def from_env(cls, runtime_home: str | Path | None = None) -> "RuntimePaths":
        home = Path(runtime_home).expanduser() if runtime_home is not None else _path_from_env(ENV_RUNTIME_HOME, DEFAULT_RUNTIME_HOME)
        return cls(
            runtime_home=home,
            state_db=_path_from_env(ENV_STATE_DB, home / DEFAULT_STATE_DB_NAME),
            runs_root=_path_from_env(ENV_RUNS_ROOT, home / "runs"),
            logs_dir=_path_from_env(ENV_LOG_DIR, home / "logs"),
            snapshots_dir=home / "snapshots",
            soak_dir=home / "soak",
        )

    def resolved(self) -> "RuntimePaths":
        home = self.runtime_home.expanduser()
        return RuntimePaths(
            runtime_home=home,
            state_db=(self.state_db or home / DEFAULT_STATE_DB_NAME).expanduser(),
            runs_root=(self.runs_root or home / "runs").expanduser(),
            logs_dir=(self.logs_dir or home / "logs").expanduser(),
            snapshots_dir=(self.snapshots_dir or home / "snapshots").expanduser(),
            soak_dir=(self.soak_dir or home / "soak").expanduser(),
        )

    def ensure(self) -> "RuntimePaths":
        paths = self.resolved()
        paths.runtime_home.mkdir(parents=True, exist_ok=True)
        paths.runs_root.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
        paths.soak_dir.mkdir(parents=True, exist_ok=True)
        paths.state_db.parent.mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def stdout_log(self) -> Path:
        return self.resolved().logs_dir / "conos.out.log"

    @property
    def stderr_log(self) -> Path:
        return self.resolved().logs_dir / "conos.err.log"

    @property
    def service_status_log(self) -> Path:
        return self.resolved().snapshots_dir / "status.jsonl"

    def as_env(self) -> Dict[str, str]:
        paths = self.resolved()
        return {
            ENV_RUNTIME_HOME: str(paths.runtime_home),
            ENV_STATE_DB: str(paths.state_db),
            ENV_RUNS_ROOT: str(paths.runs_root),
            ENV_LOG_DIR: str(paths.logs_dir),
        }

    def as_dict(self) -> Dict[str, str]:
        paths = self.resolved()
        return {
            "schema_version": RUNTIME_PATHS_VERSION,
            "runtime_home": str(paths.runtime_home),
            "state_db": str(paths.state_db),
            "runs_root": str(paths.runs_root),
            "logs_dir": str(paths.logs_dir),
            "snapshots_dir": str(paths.snapshots_dir),
            "soak_dir": str(paths.soak_dir),
            "stdout_log": str(paths.stdout_log),
            "stderr_log": str(paths.stderr_log),
            "service_status_log": str(paths.service_status_log),
        }


def runtime_paths_from_mapping(payload: Mapping[str, str] | None = None) -> RuntimePaths:
    values = dict(payload or {})
    return RuntimePaths(
        runtime_home=Path(values.get("runtime_home") or values.get(ENV_RUNTIME_HOME) or DEFAULT_RUNTIME_HOME).expanduser(),
        state_db=Path(values["state_db"]).expanduser() if values.get("state_db") else None,
        runs_root=Path(values["runs_root"]).expanduser() if values.get("runs_root") else None,
        logs_dir=Path(values["logs_dir"]).expanduser() if values.get("logs_dir") else None,
    )
