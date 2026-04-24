from __future__ import annotations

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
        return self.update_approval_status(
            approval_id,
            "APPROVED",
            response={"approved": True, "approved_by": str(approved_by or "operator"), "approved_at": utc_ts()},
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

    def count_events(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM events WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        return int(row["count"] if row is not None else 0)

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
