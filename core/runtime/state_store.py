from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional
import uuid


STATE_STORE_SCHEMA_VERSION = "conos.runtime_state_store/v1"
DEFAULT_STATE_DB = Path("runtime/long_run/state.sqlite3")
TERMINAL_RUN_STATUSES = {"COMPLETED", "STOPPED", "FAILED"}


def utc_ts() -> float:
    return float(time.time())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload if payload is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(text: Any) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(str(text))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


class RuntimeStateStore:
    """SQLite-backed durable state for local long-run supervision."""

    def __init__(self, db_path: str | Path = DEFAULT_STATE_DB) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self._conn.close()

    def _initialize(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                heartbeat_status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                heartbeat_updated_at REAL NOT NULL,
                paused_reason TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                verifier_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                result_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_run_status_priority
            ON tasks(run_id, status, priority DESC, created_at ASC);

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_events_run_created
            ON events(run_id, created_at ASC);

            CREATE TABLE IF NOT EXISTS approvals (
                approval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL DEFAULT '{}',
                response_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_approvals_run_status
            ON approvals(run_id, status, created_at ASC);

            CREATE TABLE IF NOT EXISTS leases (
                run_id TEXT PRIMARY KEY,
                lease_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                acquired_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS soak_sessions (
                soak_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                started_at REAL NOT NULL,
                expected_end_at REAL NOT NULL,
                current_run_id TEXT NOT NULL DEFAULT '',
                last_snapshot_at REAL NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                summary_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_soak_sessions_status_started
            ON soak_sessions(status, started_at DESC);

            CREATE TABLE IF NOT EXISTS learning_lessons (
                lesson_id TEXT PRIMARY KEY,
                task_family TEXT NOT NULL,
                trigger TEXT NOT NULL,
                lesson_hash TEXT NOT NULL DEFAULT '',
                lesson_json TEXT NOT NULL DEFAULT '{}',
                source_run_id TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL DEFAULT 0,
                use_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_lessons_dedupe
            ON learning_lessons(task_family, trigger, lesson_hash);

            CREATE INDEX IF NOT EXISTS idx_learning_lessons_family_created
            ON learning_lessons(task_family, created_at DESC);

            CREATE TABLE IF NOT EXISTS failure_learning_objects (
                failure_id TEXT PRIMARY KEY,
                task_family TEXT NOT NULL,
                failure_mode TEXT NOT NULL,
                failure_hash TEXT NOT NULL DEFAULT '',
                object_json TEXT NOT NULL DEFAULT '{}',
                source_run_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL DEFAULT 0,
                use_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_failure_learning_dedupe
            ON failure_learning_objects(task_family, failure_mode, failure_hash);

            CREATE INDEX IF NOT EXISTS idx_failure_learning_family_created
            ON failure_learning_objects(task_family, status, created_at DESC);

            CREATE TABLE IF NOT EXISTS formal_evidence_ledger (
                evidence_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL DEFAULT '',
                task_family TEXT NOT NULL DEFAULT '',
                evidence_type TEXT NOT NULL DEFAULT '',
                claim TEXT NOT NULL DEFAULT '',
                evidence_json TEXT NOT NULL DEFAULT '{}',
                hypotheses_json TEXT NOT NULL DEFAULT '{}',
                action_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                update_json TEXT NOT NULL DEFAULT '{}',
                source_refs_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'recorded',
                ledger_hash TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_formal_evidence_run_created
            ON formal_evidence_ledger(run_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_formal_evidence_family_type_created
            ON formal_evidence_ledger(task_family, evidence_type, created_at DESC);

            CREATE TABLE IF NOT EXISTS hypothesis_lifecycle (
                hypothesis_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL DEFAULT '',
                task_family TEXT NOT NULL DEFAULT '',
                family TEXT NOT NULL DEFAULT '',
                claim TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 0.5,
                prior REAL NOT NULL DEFAULT 0.5,
                posterior REAL NOT NULL DEFAULT 0.5,
                support_count INTEGER NOT NULL DEFAULT 0,
                contradiction_count INTEGER NOT NULL DEFAULT 0,
                evidence_refs_json TEXT NOT NULL DEFAULT '{}',
                competing_with_json TEXT NOT NULL DEFAULT '{}',
                predictions_json TEXT NOT NULL DEFAULT '{}',
                falsifiers_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_hypothesis_lifecycle_run_status
            ON hypothesis_lifecycle(run_id, status, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_hypothesis_lifecycle_family
            ON hypothesis_lifecycle(task_family, family, updated_at DESC);

            CREATE TABLE IF NOT EXISTS hypothesis_lifecycle_events (
                event_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                run_id TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                evidence_ref TEXT NOT NULL DEFAULT '',
                delta REAL NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_hypothesis_events_hyp_created
            ON hypothesis_lifecycle_events(hypothesis_id, created_at DESC);
            """
        )

    def create_run(
        self,
        goal: str,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        status: str = "RUNNING",
    ) -> str:
        resolved_run_id = str(run_id or new_id("run"))
        now = utc_ts()
        self._conn.execute(
            """
            INSERT INTO runs(run_id, goal, status, heartbeat_status, created_at, updated_at, heartbeat_updated_at, paused_reason, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, '', ?)
            """,
            (
                resolved_run_id,
                str(goal or ""),
                str(status or "RUNNING"),
                str(status or "RUNNING"),
                now,
                now,
                now,
                _json_dumps(dict(metadata or {})),
            ),
        )
        return resolved_run_id

    def get_run(self, run_id: str) -> Dict[str, Any]:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()
        return self._row_to_run(row) if row is not None else {}

    def list_runs(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY updated_at DESC").fetchall()
        return [self._row_to_run(row) for row in rows]

    def update_run_status(self, run_id: str, status: str, *, paused_reason: str = "") -> None:
        now = utc_ts()
        status_value = str(status)
        self._conn.execute(
            """
            UPDATE runs
            SET status = ?, heartbeat_status = ?, paused_reason = ?, updated_at = ?, heartbeat_updated_at = ?
            WHERE run_id = ?
            """,
            (status_value, status_value, str(paused_reason or ""), now, now, str(run_id)),
        )
        if status_value in TERMINAL_RUN_STATUSES:
            self._conn.execute(
                """
                UPDATE approvals
                SET status = 'CANCELLED', response_json = ?, updated_at = ?
                WHERE run_id = ? AND status = 'WAITING'
                """,
                (
                    _json_dumps(
                        {
                            "cancelled": True,
                            "reason": f"run_status:{status_value}",
                            "cancelled_at": now,
                        }
                    ),
                    now,
                    str(run_id),
                ),
            )

    def update_heartbeat_status(self, run_id: str, heartbeat_status: str) -> None:
        now = utc_ts()
        self._conn.execute(
            "UPDATE runs SET heartbeat_status = ?, heartbeat_updated_at = ?, updated_at = ? WHERE run_id = ?",
            (str(heartbeat_status), now, now, str(run_id)),
        )

    def update_run_metadata(self, run_id: str, metadata_patch: Mapping[str, Any]) -> Dict[str, Any]:
        current = self.get_run(run_id)
        if not current:
            return {}
        metadata = dict(current.get("metadata", {}) or {})
        metadata.update(dict(metadata_patch or {}))
        self._conn.execute(
            "UPDATE runs SET metadata_json = ?, updated_at = ? WHERE run_id = ?",
            (_json_dumps(metadata), utc_ts(), str(run_id)),
        )
        return self.get_run(run_id)

    def add_task(
        self,
        run_id: str,
        objective: str,
        *,
        priority: int = 0,
        verifier: Optional[Mapping[str, Any]] = None,
        task_id: Optional[str] = None,
        status: str = "PENDING",
    ) -> str:
        resolved_task_id = str(task_id or new_id("task"))
        now = utc_ts()
        self._conn.execute(
            """
            INSERT INTO tasks(task_id, run_id, objective, priority, verifier_json, status, created_at, updated_at, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            (
                resolved_task_id,
                str(run_id),
                str(objective or ""),
                int(priority),
                _json_dumps(dict(verifier or {})),
                str(status or "PENDING"),
                now,
                now,
            ),
        )
        return resolved_task_id

    def get_task(self, task_id: str) -> Dict[str, Any]:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (str(task_id),)).fetchone()
        return self._row_to_task(row) if row is not None else {}

    def list_tasks(self, run_id: str, *, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        params: List[Any] = [str(run_id)]
        where = "run_id = ?"
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where += f" AND status IN ({placeholders})"
            params.extend(str(item) for item in statuses)
        rows = self._conn.execute(
            f"SELECT * FROM tasks WHERE {where} ORDER BY priority DESC, created_at ASC",
            params,
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def next_pending_task(self, run_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT * FROM tasks
            WHERE run_id = ? AND status = 'PENDING'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """,
            (str(run_id),),
        ).fetchone()
        return self._row_to_task(row) if row is not None else {}

    def first_active_task(self, run_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT * FROM tasks
            WHERE run_id = ? AND status = 'RUNNING'
            ORDER BY updated_at ASC
            LIMIT 1
            """,
            (str(run_id),),
        ).fetchone()
        return self._row_to_task(row) if row is not None else {}

    def update_task_status(
        self,
        task_id: str,
        status: str,
        *,
        result: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now = utc_ts()
        extra_started = now if str(status) == "RUNNING" else None
        extra_completed = now if str(status) == "COMPLETED" else None
        current = self.get_task(task_id)
        result_json = _json_dumps(dict(result if result is not None else current.get("result", {})))
        self._conn.execute(
            """
            UPDATE tasks
            SET status = ?,
                updated_at = ?,
                started_at = COALESCE(started_at, ?),
                completed_at = COALESCE(?, completed_at),
                result_json = ?
            WHERE task_id = ?
            """,
            (str(status), now, extra_started, extra_completed, result_json, str(task_id)),
        )

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Optional[Mapping[str, Any]] = None,
        task_id: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_event_id = str(event_id or new_id("event"))
        now = utc_ts()
        event = {
            "event_id": resolved_event_id,
            "run_id": str(run_id),
            "task_id": str(task_id or ""),
            "event_type": str(event_type),
            "payload": dict(payload or {}),
            "created_at": now,
        }
        self._conn.execute(
            """
            INSERT INTO events(event_id, run_id, task_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["run_id"],
                event["task_id"] or None,
                event["event_type"],
                _json_dumps(event["payload"]),
                now,
            ),
        )
        return event

    def list_events(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY created_at ASC",
            (str(run_id),),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def create_approval(
        self,
        run_id: str,
        request: Mapping[str, Any],
        *,
        task_id: Optional[str] = None,
        approval_id: Optional[str] = None,
    ) -> str:
        resolved_approval_id = str(approval_id or new_id("approval"))
        now = utc_ts()
        self._conn.execute(
            """
            INSERT INTO approvals(approval_id, run_id, task_id, status, request_json, response_json, created_at, updated_at)
            VALUES (?, ?, ?, 'WAITING', ?, '{}', ?, ?)
            """,
            (resolved_approval_id, str(run_id), str(task_id) if task_id else None, _json_dumps(dict(request)), now, now),
        )
        return resolved_approval_id

    def get_latest_approval(self, run_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (str(run_id),),
        ).fetchone()
        return self._row_to_approval(row) if row is not None else {}

    def get_approval(self, approval_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (str(approval_id),),
        ).fetchone()
        return self._row_to_approval(row) if row is not None else {}

    def list_approvals(self, run_id: Optional[str] = None, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(str(run_id))
        if status is not None:
            where.append("status = ?")
            params.append(str(status))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(
            f"SELECT * FROM approvals {clause} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [self._row_to_approval(row) for row in rows]

    def update_approval_status(
        self,
        approval_id: str,
        status: str,
        *,
        response: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = utc_ts()
        current = self.get_approval(approval_id)
        if not current:
            return {}
        response_json = _json_dumps(dict(response if response is not None else current.get("response", {})))
        self._conn.execute(
            """
            UPDATE approvals
            SET status = ?, response_json = ?, updated_at = ?
            WHERE approval_id = ?
            """,
            (str(status), response_json, now, str(approval_id)),
        )
        return self.get_approval(approval_id)

    def approve_approval(self, approval_id: str, *, approved_by: str = "operator") -> Dict[str, Any]:
        current = self.get_approval(approval_id)
        request = dict(current.get("request", {}) or {}) if current else {}
        approval_effect = dict(request.get("approval_effect", {}) or {}) if isinstance(request.get("approval_effect", {}), Mapping) else {}
        approved_capability_layers = (
            approval_effect.get("approved_capability_layers")
            or request.get("required_capability_layers")
            or request.get("capability_layers")
            or []
        )
        return self.update_approval_status(
            approval_id,
            "APPROVED",
            response={
                "approved": True,
                "approved_by": str(approved_by or "operator"),
                "approved_at": utc_ts(),
                "approved_capability_layers": list(approved_capability_layers or []),
            },
        )

    def prune_events(self, *, max_events_per_run: int = 5000) -> Dict[str, Any]:
        limit = int(max_events_per_run)
        if limit <= 0:
            cursor = self._conn.execute("DELETE FROM events")
            return {"deleted": int(cursor.rowcount or 0), "max_events_per_run": limit}
        deleted = 0
        run_rows = self._conn.execute("SELECT DISTINCT run_id FROM events").fetchall()
        for row in run_rows:
            event_ids = [
                str(item["event_id"])
                for item in self._conn.execute(
                    """
                    SELECT event_id FROM events
                    WHERE run_id = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (str(row["run_id"]), limit),
                ).fetchall()
            ]
            if not event_ids:
                continue
            placeholders = ", ".join("?" for _ in event_ids)
            cursor = self._conn.execute(f"DELETE FROM events WHERE event_id IN ({placeholders})", event_ids)
            deleted += int(cursor.rowcount or 0)
        return {"deleted": deleted, "max_events_per_run": limit, "run_count": len(run_rows)}

    def create_soak_session(
        self,
        *,
        soak_id: Optional[str] = None,
        mode: str,
        expected_end_at: float,
        current_run_id: str = "",
        status: str = "RUNNING",
        summary: Optional[Mapping[str, Any]] = None,
    ) -> str:
        resolved_soak_id = str(soak_id or new_id("soak"))
        now = utc_ts()
        self._conn.execute(
            """
            INSERT INTO soak_sessions(
                soak_id, status, mode, started_at, expected_end_at,
                current_run_id, last_snapshot_at, failure_count, summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (
                resolved_soak_id,
                str(status or "RUNNING"),
                str(mode or "infrastructure"),
                now,
                float(expected_end_at),
                str(current_run_id or ""),
                _json_dumps(dict(summary or {})),
            ),
        )
        return resolved_soak_id

    def get_soak_session(self, soak_id: str) -> Dict[str, Any]:
        row = self._conn.execute("SELECT * FROM soak_sessions WHERE soak_id = ?", (str(soak_id),)).fetchone()
        return self._row_to_soak_session(row) if row is not None else {}

    def list_soak_sessions(self, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params: List[Any] = []
        clause = ""
        if status is not None:
            clause = "WHERE status = ?"
            params.append(str(status))
        rows = self._conn.execute(
            f"SELECT * FROM soak_sessions {clause} ORDER BY started_at DESC",
            params,
        ).fetchall()
        return [self._row_to_soak_session(row) for row in rows]

    def update_soak_session(
        self,
        soak_id: str,
        *,
        status: Optional[str] = None,
        current_run_id: Optional[str] = None,
        last_snapshot_at: Optional[float] = None,
        failure_count: Optional[int] = None,
        summary: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = self.get_soak_session(soak_id)
        if not current:
            return {}
        values = {
            "status": str(status if status is not None else current["status"]),
            "current_run_id": str(current_run_id if current_run_id is not None else current["current_run_id"]),
            "last_snapshot_at": float(last_snapshot_at if last_snapshot_at is not None else current["last_snapshot_at"]),
            "failure_count": int(failure_count if failure_count is not None else current["failure_count"]),
            "summary_json": _json_dumps(dict(summary if summary is not None else current.get("summary", {}))),
            "soak_id": str(soak_id),
        }
        self._conn.execute(
            """
            UPDATE soak_sessions
            SET status = :status,
                current_run_id = :current_run_id,
                last_snapshot_at = :last_snapshot_at,
                failure_count = :failure_count,
                summary_json = :summary_json
            WHERE soak_id = :soak_id
            """,
            values,
        )
        return self.get_soak_session(soak_id)

    def record_learning_lesson(
        self,
        *,
        task_family: str,
        trigger: str,
        lesson: Mapping[str, Any],
        source_run_id: str = "",
        confidence: float = 0.5,
        lesson_id: Optional[str] = None,
    ) -> str:
        family = str(task_family or "generic")
        trigger_value = str(trigger or "generic")
        lesson_payload = dict(lesson or {})
        lesson_hash = hashlib.sha256(_json_dumps(lesson_payload).encode("utf-8")).hexdigest()
        existing = self._conn.execute(
            """
            SELECT * FROM learning_lessons
            WHERE task_family = ? AND trigger = ? AND lesson_hash = ?
            LIMIT 1
            """,
            (family, trigger_value, lesson_hash),
        ).fetchone()
        now = utc_ts()
        if existing is not None:
            resolved_lesson_id = str(existing["lesson_id"])
            self._conn.execute(
                """
                UPDATE learning_lessons
                SET confidence = MAX(confidence, ?),
                    source_run_id = CASE WHEN ? != '' THEN ? ELSE source_run_id END
                WHERE lesson_id = ?
                """,
                (float(confidence), str(source_run_id or ""), str(source_run_id or ""), resolved_lesson_id),
            )
            return resolved_lesson_id
        resolved_lesson_id = str(lesson_id or new_id("lesson"))
        self._conn.execute(
            """
            INSERT INTO learning_lessons(
                lesson_id, task_family, trigger, lesson_hash, lesson_json,
                source_run_id, confidence, created_at, last_used_at, use_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (
                resolved_lesson_id,
                family,
                trigger_value,
                lesson_hash,
                _json_dumps(lesson_payload),
                str(source_run_id or ""),
                float(confidence),
                now,
            ),
        )
        return resolved_lesson_id

    def list_learning_lessons(
        self,
        *,
        task_family: Optional[str] = None,
        trigger: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = []
        if task_family is not None:
            where.append("task_family = ?")
            params.append(str(task_family))
        if trigger is not None:
            where.append("trigger = ?")
            params.append(str(trigger))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit or 20)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM learning_lessons
            {clause}
            ORDER BY confidence DESC, last_used_at ASC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_learning_lesson(row) for row in rows]

    def mark_learning_lesson_used(self, lesson_id: str) -> None:
        self._conn.execute(
            """
            UPDATE learning_lessons
            SET last_used_at = ?, use_count = use_count + 1
            WHERE lesson_id = ?
            """,
            (utc_ts(), str(lesson_id)),
        )

    def record_failure_learning_object(
        self,
        *,
        task_family: str,
        failure_mode: str,
        failure_object: Mapping[str, Any],
        source_run_id: str = "",
        confidence: float = 0.5,
        status: str = "active",
        failure_id: Optional[str] = None,
    ) -> str:
        family = str(task_family or "generic")
        mode = str(failure_mode or "unknown_failure")
        object_payload = dict(failure_object or {})
        comparable = {
            "failure_mode": object_payload.get("failure_mode", mode),
            "violated_assumption": object_payload.get("violated_assumption", ""),
            "missing_tool": object_payload.get("missing_tool", ""),
            "bad_policy": object_payload.get("bad_policy", ""),
            "new_regression_test": object_payload.get("new_regression_test", {}),
            "new_governance_rule": object_payload.get("new_governance_rule", {}),
            "retrieval_tags": object_payload.get("retrieval_tags", []),
        }
        object_hash = hashlib.sha256(_json_dumps(comparable).encode("utf-8")).hexdigest()
        existing = self._conn.execute(
            """
            SELECT * FROM failure_learning_objects
            WHERE task_family = ? AND failure_mode = ? AND failure_hash = ?
            LIMIT 1
            """,
            (family, mode, object_hash),
        ).fetchone()
        now = utc_ts()
        if existing is not None:
            resolved_failure_id = str(existing["failure_id"])
            self._conn.execute(
                """
                UPDATE failure_learning_objects
                SET confidence = MAX(confidence, ?),
                    source_run_id = CASE WHEN ? != '' THEN ? ELSE source_run_id END,
                    status = CASE WHEN ? != '' THEN ? ELSE status END
                WHERE failure_id = ?
                """,
                (
                    float(confidence),
                    str(source_run_id or ""),
                    str(source_run_id or ""),
                    str(status or ""),
                    str(status or ""),
                    resolved_failure_id,
                ),
            )
            return resolved_failure_id
        resolved_failure_id = str(failure_id or new_id("failure"))
        self._conn.execute(
            """
            INSERT INTO failure_learning_objects(
                failure_id, task_family, failure_mode, failure_hash, object_json,
                source_run_id, status, confidence, created_at, last_used_at, use_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (
                resolved_failure_id,
                family,
                mode,
                object_hash,
                _json_dumps(object_payload),
                str(source_run_id or ""),
                str(status or "active"),
                float(confidence),
                now,
            ),
        )
        return resolved_failure_id

    def list_failure_learning_objects(
        self,
        *,
        task_family: Optional[str] = None,
        failure_mode: Optional[str] = None,
        status: Optional[str] = "active",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = []
        if task_family is not None:
            where.append("task_family = ?")
            params.append(str(task_family))
        if failure_mode is not None:
            where.append("failure_mode = ?")
            params.append(str(failure_mode))
        if status is not None:
            where.append("status = ?")
            params.append(str(status))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit or 20)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM failure_learning_objects
            {clause}
            ORDER BY confidence DESC, last_used_at ASC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_failure_learning_object(row) for row in rows]

    def mark_failure_learning_object_used(self, failure_id: str) -> None:
        self._conn.execute(
            """
            UPDATE failure_learning_objects
            SET last_used_at = ?, use_count = use_count + 1
            WHERE failure_id = ?
            """,
            (utc_ts(), str(failure_id)),
        )

    def record_evidence_entry(self, entry: Mapping[str, Any]) -> str:
        payload = dict(entry or {})
        evidence_id = str(payload.get("evidence_id") or new_id("evidence"))
        created_at = float(payload.get("created_at") or utc_ts())
        source_refs = list(payload.get("source_refs", []) or [])
        hypotheses = list(payload.get("hypotheses", []) or [])
        self._conn.execute(
            """
            INSERT INTO formal_evidence_ledger(
                evidence_id,
                run_id,
                task_family,
                evidence_type,
                claim,
                evidence_json,
                hypotheses_json,
                action_json,
                result_json,
                update_json,
                source_refs_json,
                confidence,
                status,
                ledger_hash,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id) DO UPDATE SET
                run_id = excluded.run_id,
                task_family = excluded.task_family,
                evidence_type = excluded.evidence_type,
                claim = excluded.claim,
                evidence_json = excluded.evidence_json,
                hypotheses_json = excluded.hypotheses_json,
                action_json = excluded.action_json,
                result_json = excluded.result_json,
                update_json = excluded.update_json,
                source_refs_json = excluded.source_refs_json,
                confidence = excluded.confidence,
                status = excluded.status,
                ledger_hash = excluded.ledger_hash,
                created_at = excluded.created_at
            """,
            (
                evidence_id,
                str(payload.get("run_id") or ""),
                str(payload.get("task_family") or ""),
                str(payload.get("evidence_type") or ""),
                str(payload.get("claim") or ""),
                _json_dumps(dict(payload.get("evidence", {}) or {})),
                _json_dumps({"items": hypotheses}),
                _json_dumps(dict(payload.get("action", {}) or {})),
                _json_dumps(dict(payload.get("result", {}) or {})),
                _json_dumps(dict(payload.get("update", {}) or {})),
                _json_dumps({"refs": source_refs}),
                float(payload.get("confidence") or 0.0),
                str(payload.get("status") or "recorded"),
                str(payload.get("ledger_hash") or ""),
                created_at,
            ),
        )
        return evidence_id

    def list_evidence_entries(
        self,
        *,
        run_id: Optional[str] = None,
        task_family: Optional[str] = None,
        evidence_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(str(run_id))
        if task_family is not None:
            where.append("task_family = ?")
            params.append(str(task_family))
        if evidence_type is not None:
            where.append("evidence_type = ?")
            params.append(str(evidence_type))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit or 50)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM formal_evidence_ledger
            {clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_evidence_entry(row) for row in rows]

    def upsert_hypothesis_lifecycle(self, row: Mapping[str, Any]) -> str:
        payload = dict(row or {})
        hypothesis_id = str(payload.get("hypothesis_id") or new_id("hyp"))
        now = utc_ts()
        evidence_refs = list(payload.get("evidence_refs", []) or [])
        competing_with = list(payload.get("competing_with", []) or [])
        created_at = float(payload.get("created_at") or now)
        updated_at = float(payload.get("updated_at") or now)
        confidence = float(payload.get("confidence", payload.get("posterior", 0.5)) or 0.5)
        posterior = float(payload.get("posterior", confidence) or confidence)
        prior = float(payload.get("prior", confidence) or confidence)
        self._conn.execute(
            """
            INSERT INTO hypothesis_lifecycle(
                hypothesis_id,
                run_id,
                task_family,
                family,
                claim,
                status,
                confidence,
                prior,
                posterior,
                support_count,
                contradiction_count,
                evidence_refs_json,
                competing_with_json,
                predictions_json,
                falsifiers_json,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET
                run_id = excluded.run_id,
                task_family = excluded.task_family,
                family = excluded.family,
                claim = excluded.claim,
                status = excluded.status,
                confidence = excluded.confidence,
                prior = excluded.prior,
                posterior = excluded.posterior,
                support_count = excluded.support_count,
                contradiction_count = excluded.contradiction_count,
                evidence_refs_json = excluded.evidence_refs_json,
                competing_with_json = excluded.competing_with_json,
                predictions_json = excluded.predictions_json,
                falsifiers_json = excluded.falsifiers_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                hypothesis_id,
                str(payload.get("run_id") or ""),
                str(payload.get("task_family") or ""),
                str(payload.get("family") or ""),
                str(payload.get("claim") or ""),
                str(payload.get("status") or "active"),
                confidence,
                prior,
                posterior,
                int(payload.get("support_count", 0) or 0),
                int(payload.get("contradiction_count", 0) or 0),
                _json_dumps({"refs": evidence_refs}),
                _json_dumps({"items": competing_with}),
                _json_dumps(dict(payload.get("predictions", {}) or {})),
                _json_dumps(dict(payload.get("falsifiers", {}) or {})),
                _json_dumps(dict(payload.get("metadata", {}) or {})),
                created_at,
                updated_at,
            ),
        )
        return hypothesis_id

    def list_hypothesis_lifecycle(
        self,
        *,
        run_id: Optional[str] = None,
        task_family: Optional[str] = None,
        family: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(str(run_id))
        if task_family is not None:
            where.append("task_family = ?")
            params.append(str(task_family))
        if family is not None:
            where.append("family = ?")
            params.append(str(family))
        if status is not None:
            where.append("status = ?")
            params.append(str(status))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit or 50)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM hypothesis_lifecycle
            {clause}
            ORDER BY posterior DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_hypothesis_lifecycle(row) for row in rows]

    def record_hypothesis_lifecycle_event(
        self,
        *,
        hypothesis_id: str,
        run_id: str = "",
        event_type: str,
        evidence_ref: str = "",
        delta: float = 0.0,
        payload: Optional[Mapping[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        resolved_event_id = str(event_id or new_id("hyp_event"))
        self._conn.execute(
            """
            INSERT INTO hypothesis_lifecycle_events(
                event_id, hypothesis_id, run_id, event_type, evidence_ref, delta, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_event_id,
                str(hypothesis_id or ""),
                str(run_id or ""),
                str(event_type or ""),
                str(evidence_ref or ""),
                float(delta or 0.0),
                _json_dumps(dict(payload or {})),
                utc_ts(),
            ),
        )
        return resolved_event_id

    def list_hypothesis_lifecycle_events(
        self,
        *,
        hypothesis_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = []
        if hypothesis_id is not None:
            where.append("hypothesis_id = ?")
            params.append(str(hypothesis_id))
        if run_id is not None:
            where.append("run_id = ?")
            params.append(str(run_id))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit or 50)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM hypothesis_lifecycle_events
            {clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_hypothesis_lifecycle_event(row) for row in rows]

    def count_events(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM events WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        return int(row["count"] if row is not None else 0)

    def latest_event_created_at(self, run_id: str) -> float:
        row = self._conn.execute(
            "SELECT MAX(created_at) AS created_at FROM events WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        return float(row["created_at"] or 0.0) if row is not None else 0.0

    def latest_task_progress_at(self, run_id: str) -> float:
        row = self._conn.execute(
            "SELECT MAX(updated_at) AS updated_at FROM tasks WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        return float(row["updated_at"] or 0.0) if row is not None else 0.0

    def acquire_lease(self, run_id: str, *, worker_id: str, ttl_seconds: float) -> Dict[str, Any]:
        now = utc_ts()
        expires_at = now + max(0.1, float(ttl_seconds))
        lease_id = new_id("lease")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute("SELECT * FROM leases WHERE run_id = ?", (str(run_id),)).fetchone()
            if row is not None and float(row["expires_at"]) > now and str(row["worker_id"]) != str(worker_id):
                self._conn.execute("COMMIT")
                return {
                    "acquired": False,
                    "run_id": str(run_id),
                    "lease_id": str(row["lease_id"]),
                    "worker_id": str(row["worker_id"]),
                    "expires_at": float(row["expires_at"]),
                }
            self._conn.execute(
                """
                INSERT INTO leases(run_id, lease_id, worker_id, acquired_at, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    lease_id = excluded.lease_id,
                    worker_id = excluded.worker_id,
                    acquired_at = excluded.acquired_at,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (str(run_id), lease_id, str(worker_id), now, expires_at, now),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return {
            "acquired": True,
            "run_id": str(run_id),
            "lease_id": lease_id,
            "worker_id": str(worker_id),
            "expires_at": expires_at,
        }

    def count_leases(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM leases").fetchone()
        return int(row["count"] if row is not None else 0)

    def release_lease(self, run_id: str, *, worker_id: str) -> None:
        self._conn.execute(
            "DELETE FROM leases WHERE run_id = ? AND worker_id = ?",
            (str(run_id), str(worker_id)),
        )

    def clear_expired_lease(self, run_id: str) -> bool:
        now = utc_ts()
        cursor = self._conn.execute(
            "DELETE FROM leases WHERE run_id = ? AND expires_at <= ?",
            (str(run_id), now),
        )
        return int(cursor.rowcount or 0) > 0

    def clear_expired_leases(self) -> Dict[str, Any]:
        now = utc_ts()
        cursor = self._conn.execute("DELETE FROM leases WHERE expires_at <= ?", (now,))
        return {"deleted": int(cursor.rowcount or 0), "cleared_at": now}

    def checkpoint_wal(self, *, mode: str = "PASSIVE") -> Dict[str, Any]:
        if str(self.db_path) == ":memory:":
            return {"status": "SKIPPED", "reason": "memory_database"}
        normalized = str(mode or "PASSIVE").upper()
        if normalized not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            normalized = "PASSIVE"
        wal_path = Path(f"{self.db_path}-wal")
        wal_size_before = wal_path.stat().st_size if wal_path.exists() else 0
        row = self._conn.execute(f"PRAGMA wal_checkpoint({normalized})").fetchone()
        wal_size_after = wal_path.stat().st_size if wal_path.exists() else 0
        busy = int(row[0]) if row is not None else 0
        log_frames = int(row[1]) if row is not None else 0
        checkpointed_frames = int(row[2]) if row is not None else 0
        return {
            "status": "OK" if busy == 0 else "BUSY",
            "mode": normalized,
            "busy": busy,
            "log_frames": log_frames,
            "checkpointed_frames": checkpointed_frames,
            "wal_size_before_bytes": wal_size_before,
            "wal_size_after_bytes": wal_size_after,
        }

    def quick_check(self) -> Dict[str, Any]:
        try:
            row = self._conn.execute("PRAGMA quick_check").fetchone()
            result = str(row[0] if row is not None else "")
        except Exception as exc:
            return {
                "status": "FAILED",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        return {"status": "OK" if result.lower() == "ok" else "FAILED", "result": result}

    def _row_to_run(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "goal": str(row["goal"]),
            "status": str(row["status"]),
            "heartbeat_status": str(row["heartbeat_status"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "heartbeat_updated_at": float(row["heartbeat_updated_at"]),
            "paused_reason": str(row["paused_reason"] or ""),
            "metadata": _json_loads(row["metadata_json"]),
        }

    def _row_to_task(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "task_id": str(row["task_id"]),
            "run_id": str(row["run_id"]),
            "objective": str(row["objective"]),
            "priority": int(row["priority"]),
            "verifier": _json_loads(row["verifier_json"]),
            "status": str(row["status"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "started_at": float(row["started_at"]) if row["started_at"] is not None else None,
            "completed_at": float(row["completed_at"]) if row["completed_at"] is not None else None,
            "result": _json_loads(row["result_json"]),
        }

    def _row_to_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "event_id": str(row["event_id"]),
            "run_id": str(row["run_id"]),
            "task_id": str(row["task_id"] or ""),
            "event_type": str(row["event_type"]),
            "payload": _json_loads(row["payload_json"]),
            "created_at": float(row["created_at"]),
        }

    def _row_to_approval(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "approval_id": str(row["approval_id"]),
            "run_id": str(row["run_id"]),
            "task_id": str(row["task_id"] or ""),
            "status": str(row["status"]),
            "request": _json_loads(row["request_json"]),
            "response": _json_loads(row["response_json"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def _row_to_soak_session(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "soak_id": str(row["soak_id"]),
            "status": str(row["status"]),
            "mode": str(row["mode"]),
            "started_at": float(row["started_at"]),
            "expected_end_at": float(row["expected_end_at"]),
            "current_run_id": str(row["current_run_id"] or ""),
            "last_snapshot_at": float(row["last_snapshot_at"]),
            "failure_count": int(row["failure_count"]),
            "summary": _json_loads(row["summary_json"]),
        }

    def _row_to_learning_lesson(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "lesson_id": str(row["lesson_id"]),
            "task_family": str(row["task_family"]),
            "trigger": str(row["trigger"]),
            "lesson_hash": str(row["lesson_hash"] or ""),
            "lesson": _json_loads(row["lesson_json"]),
            "source_run_id": str(row["source_run_id"] or ""),
            "confidence": float(row["confidence"]),
            "created_at": float(row["created_at"]),
            "last_used_at": float(row["last_used_at"]),
            "use_count": int(row["use_count"]),
        }

    def _row_to_failure_learning_object(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "failure_id": str(row["failure_id"]),
            "task_family": str(row["task_family"]),
            "failure_mode": str(row["failure_mode"]),
            "failure_hash": str(row["failure_hash"] or ""),
            "failure_object": _json_loads(row["object_json"]),
            "source_run_id": str(row["source_run_id"] or ""),
            "status": str(row["status"] or ""),
            "confidence": float(row["confidence"]),
            "created_at": float(row["created_at"]),
            "last_used_at": float(row["last_used_at"]),
            "use_count": int(row["use_count"]),
        }

    def _row_to_evidence_entry(self, row: sqlite3.Row) -> Dict[str, Any]:
        source_refs = _json_loads(row["source_refs_json"]).get("refs", [])
        hypotheses = _json_loads(row["hypotheses_json"]).get("items", [])
        return {
            "evidence_id": str(row["evidence_id"]),
            "run_id": str(row["run_id"] or ""),
            "task_family": str(row["task_family"] or ""),
            "evidence_type": str(row["evidence_type"] or ""),
            "claim": str(row["claim"] or ""),
            "evidence": _json_loads(row["evidence_json"]),
            "hypotheses": list(hypotheses or []) if isinstance(hypotheses, list) else [],
            "action": _json_loads(row["action_json"]),
            "result": _json_loads(row["result_json"]),
            "update": _json_loads(row["update_json"]),
            "source_refs": list(source_refs or []) if isinstance(source_refs, list) else [],
            "confidence": float(row["confidence"]),
            "status": str(row["status"] or ""),
            "ledger_hash": str(row["ledger_hash"] or ""),
            "created_at": float(row["created_at"]),
        }

    def _row_to_hypothesis_lifecycle(self, row: sqlite3.Row) -> Dict[str, Any]:
        evidence_refs = _json_loads(row["evidence_refs_json"]).get("refs", [])
        competing_with = _json_loads(row["competing_with_json"]).get("items", [])
        return {
            "hypothesis_id": str(row["hypothesis_id"]),
            "run_id": str(row["run_id"] or ""),
            "task_family": str(row["task_family"] or ""),
            "family": str(row["family"] or ""),
            "claim": str(row["claim"] or ""),
            "status": str(row["status"] or ""),
            "confidence": float(row["confidence"]),
            "prior": float(row["prior"]),
            "posterior": float(row["posterior"]),
            "support_count": int(row["support_count"]),
            "contradiction_count": int(row["contradiction_count"]),
            "evidence_refs": list(evidence_refs or []) if isinstance(evidence_refs, list) else [],
            "competing_with": list(competing_with or []) if isinstance(competing_with, list) else [],
            "predictions": _json_loads(row["predictions_json"]),
            "falsifiers": _json_loads(row["falsifiers_json"]),
            "metadata": _json_loads(row["metadata_json"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def _row_to_hypothesis_lifecycle_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "event_id": str(row["event_id"]),
            "hypothesis_id": str(row["hypothesis_id"]),
            "run_id": str(row["run_id"] or ""),
            "event_type": str(row["event_type"] or ""),
            "evidence_ref": str(row["evidence_ref"] or ""),
            "delta": float(row["delta"]),
            "payload": _json_loads(row["payload_json"]),
            "created_at": float(row["created_at"]),
        }
