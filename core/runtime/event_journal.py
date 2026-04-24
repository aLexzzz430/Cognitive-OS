from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from core.runtime.state_store import RuntimeStateStore


DEFAULT_RUNS_ROOT = Path("runtime/runs")


class EventJournal:
    """Durable event journal with SQLite and JSONL mirrors."""

    def __init__(self, state_store: RuntimeStateStore, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> None:
        self.state_store = state_store
        self.runs_root = Path(runs_root)

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
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return event
