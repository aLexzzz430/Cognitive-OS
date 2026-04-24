from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from core.runtime.state_store import RuntimeStateStore


DEFAULT_RUNS_ROOT = Path("runtime/runs")
DEFAULT_EVENT_JSONL_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_EVENT_JSONL_RETAINED_FILES = 5


class EventJournal:
    """Durable event journal with SQLite and JSONL mirrors."""

    def __init__(
        self,
        state_store: RuntimeStateStore,
        runs_root: str | Path = DEFAULT_RUNS_ROOT,
        *,
        max_jsonl_bytes: int = DEFAULT_EVENT_JSONL_MAX_BYTES,
        retained_jsonl_files: int = DEFAULT_EVENT_JSONL_RETAINED_FILES,
    ) -> None:
        self.state_store = state_store
        self.runs_root = Path(runs_root)
        self.max_jsonl_bytes = int(max_jsonl_bytes)
        self.retained_jsonl_files = int(retained_jsonl_files)

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Optional[Mapping[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = self.state_store.append_event(
            run_id=run_id,
            task_id=task_id,
            event_type=event_type,
            payload=payload,
        )
        run_dir = self.runs_root / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        event_path = run_dir / "events.jsonl"
        self._rotate_if_needed(event_path)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return event

    def status(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.runs_root / str(run_id)
        event_path = run_dir / "events.jsonl"
        rotated = sorted(run_dir.glob("events.*.jsonl")) if run_dir.exists() else []
        return {
            "path": str(event_path),
            "exists": event_path.exists(),
            "size_bytes": event_path.stat().st_size if event_path.exists() else 0,
            "rotated_files": [str(path) for path in rotated],
            "max_jsonl_bytes": self.max_jsonl_bytes,
            "retained_jsonl_files": self.retained_jsonl_files,
        }

    def _rotate_if_needed(self, event_path: Path) -> None:
        if self.max_jsonl_bytes <= 0:
            return
        if not event_path.exists() or event_path.stat().st_size < self.max_jsonl_bytes:
            return
        retained = max(0, self.retained_jsonl_files)
        if retained <= 0:
            event_path.unlink(missing_ok=True)
            return
        oldest = event_path.with_name(f"events.{retained}.jsonl")
        oldest.unlink(missing_ok=True)
        for index in range(retained - 1, 0, -1):
            source = event_path.with_name(f"events.{index}.jsonl")
            if source.exists():
                source.rename(event_path.with_name(f"events.{index + 1}.jsonl"))
        event_path.rename(event_path.with_name("events.1.jsonl"))
