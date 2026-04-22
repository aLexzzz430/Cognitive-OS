from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from modules.governance.object_store import AssetStatus
from modules.memory.schema import MemoryType


DURABLE_OBJECT_RECORDS_STATE_PATH = "object_workspace.durable_object_records"
_MAX_PROMOTION_EPISODE_SEEDS = 12


def _episode_content(record: Dict[str, Any]) -> Dict[str, Any]:
    content = record.get("content", {})
    return dict(content) if isinstance(content, dict) else {}


def _is_promotion_episode_seed(record: Dict[str, Any]) -> bool:
    if str(record.get("memory_type") or "").strip() != MemoryType.EPISODE_RECORD.value:
        return False
    content = _episode_content(record)
    if not content:
        return False
    has_mechanism_signal = bool(list(content.get("mechanism_signals", []) or []))
    has_discoveries = bool(list(content.get("key_discoveries", []) or []))
    reward_trend = str(content.get("reward_trend") or "").strip().lower()
    total_reward = float(content.get("total_reward", 0.0) or 0.0)
    is_positive = reward_trend != "negative" and total_reward >= 0.0
    return is_positive and (has_mechanism_signal or has_discoveries)


def _compact_promotion_episode_seed(record: Dict[str, Any]) -> Dict[str, Any]:
    compact = copy.deepcopy(record)
    content = _episode_content(record)
    compact["retrieval_tags"] = ["promotion_seed", "durable"]
    compact["content"] = {
        "episode_id": int(content.get("episode_id", record.get("trigger_episode", 0)) or 0),
        "total_reward": float(content.get("total_reward", 0.0) or 0.0),
        "reward_trend": str(content.get("reward_trend") or "").strip() or "neutral",
        "teacher_present": bool(content.get("teacher_present", False)),
        "key_discoveries": [str(item) for item in list(content.get("key_discoveries", []) or []) if str(item or "")],
        "callable_functions": [str(item) for item in list(content.get("callable_functions", []) or []) if str(item or "")],
        "mechanism_signals": [
            dict(item) for item in list(content.get("mechanism_signals", []) or []) if isinstance(item, dict)
        ],
        "promotion_seed_only": True,
    }
    memory_metadata = compact.get("memory_metadata", {}) if isinstance(compact.get("memory_metadata", {}), dict) else {}
    memory_metadata = dict(memory_metadata)
    memory_metadata["retrieval_tags"] = list(compact["retrieval_tags"])
    memory_metadata["promotion_seed_only"] = True
    compact["memory_metadata"] = memory_metadata
    return compact


def _should_persist_object(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    object_id = str(record.get("object_id") or "").strip()
    if not object_id:
        return False
    if str(record.get("status") or "").strip() == "invalidated":
        return False
    if str(record.get("asset_status") or "").strip() == AssetStatus.GARBAGE.value:
        return False
    if str(record.get("memory_type") or "").strip() == MemoryType.EPISODE_RECORD.value:
        return _is_promotion_episode_seed(record)
    return True


def build_durable_object_records(shared_store: Any) -> List[Dict[str, Any]]:
    durable_rows: List[Dict[str, Any]] = []
    promotion_episode_rows: List[Dict[str, Any]] = []

    for record in shared_store.iter_objects(limit=None):
        if not _should_persist_object(record):
            continue
        if str(record.get("memory_type") or "").strip() == MemoryType.EPISODE_RECORD.value:
            promotion_episode_rows.append(_compact_promotion_episode_seed(record))
        else:
            durable_rows.append(copy.deepcopy(record))

    durable_rows.sort(
        key=lambda row: (
            float(row.get("confidence", 0.0) or 0.0),
            str(row.get("updated_at", "") or ""),
            str(row.get("object_id", "") or ""),
        ),
        reverse=True,
    )
    promotion_episode_rows.sort(
        key=lambda row: (
            int(row.get("trigger_episode", 0) or 0),
            str(row.get("updated_at", "") or ""),
            str(row.get("object_id", "") or ""),
        ),
        reverse=True,
    )
    return durable_rows + promotion_episode_rows[:_MAX_PROMOTION_EPISODE_SEEDS]


def persist_durable_object_records(loop: Any, *, reason: str) -> List[str]:
    records = build_durable_object_records(loop._shared_store)
    loop._state_sync.sync(
        loop._state_sync_input_cls(
            updates={DURABLE_OBJECT_RECORDS_STATE_PATH: records},
            reason=reason,
        )
    )
    return [str(record.get("object_id") or "") for record in records if str(record.get("object_id") or "")]


def restore_durable_object_records(loop: Any) -> List[str]:
    raw = loop._state_mgr.get(DURABLE_OBJECT_RECORDS_STATE_PATH, default=[])
    records = [dict(record) for record in raw if isinstance(record, dict)] if isinstance(raw, list) else []
    if not records:
        return []
    return loop._shared_store.restore_records(records, replace=False)


def load_persisted_durable_object_records(state_mgr: Any) -> List[Dict[str, Any]]:
    state_path = Path(str(getattr(state_mgr, "_state_path", "") or "").strip())
    if not state_path.exists():
        return []
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []
    node: Any = payload
    for part in DURABLE_OBJECT_RECORDS_STATE_PATH.split("."):
        if not isinstance(node, dict):
            return []
        node = node.get(part, [])
    if not isinstance(node, list):
        return []
    return [copy.deepcopy(record) for record in node if isinstance(record, dict)]
