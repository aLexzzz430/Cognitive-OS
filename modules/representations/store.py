#!/usr/bin/env python3
from __future__ import annotations
"""
representations/store.py

Card warehouse: loads seed cards from JSON.
RuntimeUpdateStore: persists evidence/lifecycle updates to JSONL.
"""

import copy
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from core.runtime_paths import default_representation_updates_path

from .schema import RepresentationCard


# ============================================================
# Card Warehouse
# ============================================================

class CardWarehouse:
    """
    Loads and provides read access to RepresentationCards.
    Cards are loaded from JSON files in the cards/ directory.
    Cards are IMMUTABLE after loading — never modified at runtime.
    """

    def __init__(self, cards_dir: Optional[str] = None):
        if cards_dir is None:
            cards_dir = Path(__file__).parent / "cards"
        self.cards_dir = Path(cards_dir)
        self._cards: dict[str, RepresentationCard] = {}
        self._loaded = False
        # Private storage: _cards is module-internal only.
        # External code should query through get_card/get_all_cards/iter_cards/snapshot_for_audit.

    def _convert(self, obj):
        """Recursively convert nested containers to plain dict/list."""
        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert(item) for item in obj]
        return obj

    def load(self) -> None:
        """Load all cards from JSON files in cards_dir."""

        if self._loaded:
            return

        self._cards = {}

        seed_file = self.cards_dir / "seed_cards.json"
        if seed_file.exists():
            with open(seed_file, "r") as f:
                raw = json.load(f)

            raw = self._convert(raw)

            for card_dict in raw.get("cards", []):
                card = RepresentationCard.from_dict(card_dict)
                self._cards[card.rep_id] = card

        self._loaded = True

    def get_card(self, rep_id: str) -> Optional[RepresentationCard]:
        """Get a single card by rep_id."""
        if not self._loaded:
            self.load()
        return self._cards.get(rep_id)

    def get_all_cards(self) -> list[RepresentationCard]:
        """Get all loaded cards."""
        if not self._loaded:
            self.load()
        return list(self._cards.values())

    def iter_cards(self) -> Iterator[RepresentationCard]:
        """Read-only iteration over cards."""
        if not self._loaded:
            self.load()
        return iter(self._cards.values())

    def snapshot_for_audit(self) -> Tuple[dict, ...]:
        """Return detached immutable snapshot for batch read/audit flows."""
        if not self._loaded:
            self.load()
        return tuple(copy.deepcopy(card.to_dict()) for card in self._cards.values())

    def get_cards_by_family(self, family: str) -> list[RepresentationCard]:
        """Get all cards of a given family."""
        if not self._loaded:
            self.load()
        return [c for c in self._cards.values() if c.family == family]

    def get_cards_by_origin(self, origin_type: str) -> list[RepresentationCard]:
        """Get all cards of a given origin type."""
        if not self._loaded:
            self.load()
        return [c for c in self._cards.values() if c.origin_type == origin_type]

    def get_cards_by_status(self, lifecycle_status: str) -> list[RepresentationCard]:
        """Get all cards with a given lifecycle status."""
        if not self._loaded:
            self.load()
        return [c for c in self._cards.values()
                if c.lifecycle.current_status == lifecycle_status]

    def get_relevant_card_ids(
        self,
        observation_keys: list[str],
        regime: str = "nominal",
        planner_style: str = "any",
    ) -> list[str]:
        """
        Fast filter: return card_ids whose structural_signature
        mentions any of the given observation_keys and whose scope
        includes the given regime and planner_style.
        """
        if not self._loaded:
            self.load()

        relevant = []
        for card in self._cards.values():
            # Check scope: valid_in_regimes
            if card.scope.valid_in_regimes:
                if regime not in card.scope.valid_in_regimes:
                    continue

            # Check scope: invalid_in_regimes
            if card.scope.invalid_in_regimes:
                if regime in card.scope.invalid_in_regimes:
                    continue

            # Check scope: planner_styles
            if card.scope.planner_styles and card.scope.planner_styles != ["any"]:
                if planner_style not in card.scope.planner_styles:
                    continue

            # Check observation_keys overlap
            sig_keys = set(card.structural_signature.observation_keys)
            obs_keys = set(observation_keys)
            if sig_keys & obs_keys:  # intersection non-empty
                relevant.append(card.rep_id)

        return relevant

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._cards)


# ============================================================
# Runtime Update Store
# ============================================================

class UpdateType:
    SUPPORT = "support"
    COUNTEREXAMPLE = "counterexample"
    ACTIVATION = "activation"
    HELPFULNESS = "helpfulness"
    STATUS_CHANGE = "status_change"
    SNAPSHOT = "snapshot"


@dataclass
class UpdateRecord:
    """Single update record written to the runtime updates log."""
    rep_id: str
    tick: int
    update_type: str
    detail: dict
    snapshot: dict  # written-at-moment state

    def to_dict(self) -> dict:
        return {
            "rep_id": self.rep_id,
            "tick": self.tick,
            "update_type": self.update_type,
            "detail": self.detail,
            "snapshot": self.snapshot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UpdateRecord":
        return cls(
            rep_id=d["rep_id"],
            tick=d["tick"],
            update_type=d["update_type"],
            detail=d.get("detail", {}),
            snapshot=d.get("snapshot", {}),
        )


@dataclass
class RuntimeSnapshot:
    """Current runtime state for a card, loaded from updates log."""
    support_count: int = 0
    counterexample_count: int = 0
    times_activated: int = 0
    times_helpful: int = 0
    times_harmful: int = 0
    last_support_tick: int = 0
    last_activated_tick: int = 0
    lifecycle_status: str = "candidate"


class RuntimeUpdateStore:
    """
    Persists evidence and lifecycle updates for RepresentationCards.

    File format: JSONL (append-only)
    Path: runtime/representations/runtime_updates.jsonl

    Each line is one UpdateRecord (JSON).
    This allows replay: to compute current state, replay all updates.

    Public API:
      record_support(rep_id, tick, observation, outcome)
      record_counterexample(rep_id, tick, observation, outcome)
      record_activation(rep_id, tick, helpful: bool)
      update_status(rep_id, new_status, tick)
      load_all_updates() -> list[UpdateRecord]
      get_card_runtime_state(rep_id) -> RuntimeSnapshot
      replay_state(rep_id) -> RepresentationCard with updated evidence/lifecycle
    """

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = default_representation_updates_path(store_path)
        self._summary_path = self.store_path.with_suffix(self.store_path.suffix + ".summary.json")
        self._updates: Optional[list[UpdateRecord]] = None
        self._state_index: Optional[dict[str, RuntimeSnapshot]] = None
        self._record_count: Optional[int] = None
        self._log_mode: str = "incremental"

    # ---- Persistence ----

    def _ensure_path(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.touch()

    def _json_safe(self, obj: Any, _seen: Optional[set[int]] = None) -> Any:
        """
        Convert arbitrary runtime objects into JSON-safe structures.

        This is the critical boundary for Stage 6 post-commit logging:
        observations/results may contain toolkit-native objects such as ActionInput.
        Those must never reach json.dumps() raw.
        """
        if _seen is None:
            _seen = set()

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        obj_id = id(obj)
        if obj_id in _seen:
            return "<recursive_ref>"
        _seen.add(obj_id)

        try:
            if isinstance(obj, Path):
                return str(obj)

            if isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8")
                except UnicodeDecodeError:
                    return repr(obj)

            if isinstance(obj, dict):
                safe_dict = {}
                for key, value in obj.items():
                    safe_key = str(self._json_safe(key, _seen))
                    safe_dict[safe_key] = self._json_safe(value, _seen)
                return safe_dict

            if isinstance(obj, (list, tuple, set)):
                return [self._json_safe(item, _seen) for item in obj]

            if is_dataclass(obj):
                return self._json_safe(asdict(obj), _seen)

            if hasattr(obj, "_asdict"):
                try:
                    return self._json_safe(obj._asdict(), _seen)
                except Exception:
                    pass

            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                try:
                    return self._json_safe(obj.to_dict(), _seen)
                except Exception:
                    pass

            if hasattr(obj, "__dict__"):
                try:
                    raw = {
                        k: v
                        for k, v in vars(obj).items()
                        if not k.startswith("_")
                    }
                    if raw:
                        raw["__class__"] = obj.__class__.__name__
                        return self._json_safe(raw, _seen)
                except Exception:
                    pass

            if hasattr(obj, "name") and isinstance(getattr(obj, "name"), str):
                maybe_value = getattr(obj, "value", None)
                return {
                    "__enum__": obj.__class__.__name__,
                    "name": getattr(obj, "name"),
                    "value": self._json_safe(maybe_value, _seen),
                }

            return repr(obj)
        finally:
            _seen.discard(obj_id)

    def _append(self, record: UpdateRecord) -> None:
        self._ensure_path()
        safe_record = self._json_safe(record.to_dict())
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
        if self._updates is not None:
            self._updates.append(record)
        if self._record_count is not None:
            self._record_count += 1
        self._log_mode = "incremental"
        if self._state_index is not None:
            snap = self._state_index.setdefault(record.rep_id, RuntimeSnapshot())
            self._apply_update_record(snap, record)
            self._save_state_index()
            self._maybe_auto_compact()
        else:
            self._updates = None

    # ---- Recording ----

    def record_support(
        self,
        rep_id: str,
        tick: int,
        observation: dict,
        outcome: dict,
        current_snapshot: Optional[RuntimeSnapshot] = None,
    ) -> None:
        """Record that a card was supported by a positive example."""
        snap = self._snapshot_dict(current_snapshot)
        record = UpdateRecord(
            rep_id=rep_id,
            tick=tick,
            update_type=UpdateType.SUPPORT,
            detail={"observation": observation, "outcome": outcome},
            snapshot=snap,
        )
        self._append(record)

    def record_counterexample(
        self,
        rep_id: str,
        tick: int,
        observation: dict,
        outcome: dict,
        current_snapshot: Optional[RuntimeSnapshot] = None,
    ) -> None:
        """Record that a card was contradicted by a negative example."""
        snap = self._snapshot_dict(current_snapshot)
        record = UpdateRecord(
            rep_id=rep_id,
            tick=tick,
            update_type=UpdateType.COUNTEREXAMPLE,
            detail={"observation": observation, "outcome": outcome},
            snapshot=snap,
        )
        self._append(record)

    def record_activation(
        self,
        rep_id: str,
        tick: int,
        helpful: bool,
        current_snapshot: Optional[RuntimeSnapshot] = None,
    ) -> None:
        """Record that a card was activated and whether it helped."""
        snap = self._snapshot_dict(current_snapshot)
        record = UpdateRecord(
            rep_id=rep_id,
            tick=tick,
            update_type=UpdateType.ACTIVATION,
            detail={"helpful": helpful},
            snapshot=snap,
        )
        self._append(record)

    def update_status(
        self,
        rep_id: str,
        new_status: str,
        tick: int,
        current_snapshot: Optional[RuntimeSnapshot] = None,
    ) -> None:
        """Record a lifecycle status transition."""
        snap = self._snapshot_dict(current_snapshot)
        record = UpdateRecord(
            rep_id=rep_id,
            tick=tick,
            update_type=UpdateType.STATUS_CHANGE,
            detail={"new_status": new_status},
            snapshot=snap,
        )
        self._append(record)

    # ---- Loading ----

    def load_all_updates(self) -> list[UpdateRecord]:
        """Load all update records from disk (cached)."""
        if self._updates is not None:
            return self._updates

        self._updates = []
        if not self.store_path.exists():
            return self._updates

        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._updates.append(UpdateRecord.from_dict(d))
                except (json.JSONDecodeError, KeyError):
                    continue

        self._record_count = len(self._updates)
        self._log_mode = (
            "snapshot_compacted"
            if self._updates and all(update.update_type == UpdateType.SNAPSHOT for update in self._updates)
            else "incremental"
        )

        return self._updates

    def _log_signature(self) -> dict[str, int]:
        if not self.store_path.exists():
            return {"size": 0, "mtime_ns": 0}
        stat = self.store_path.stat()
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        }

    def _load_state_index_from_summary(self) -> Optional[dict[str, RuntimeSnapshot]]:
        if not self._summary_path.exists():
            return None
        try:
            payload = json.loads(self._summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        signature = payload.get("log_signature", {})
        if not isinstance(signature, dict) or signature != self._log_signature():
            return None
        stats = payload.get("stats", {})
        if isinstance(stats, dict):
            self._record_count = int(stats.get("record_count", 0) or 0)
            self._log_mode = str(stats.get("log_mode", "incremental") or "incremental")
        raw_snapshots = payload.get("snapshots", {})
        if not isinstance(raw_snapshots, dict):
            return None
        index: dict[str, RuntimeSnapshot] = {}
        for rep_id, raw in raw_snapshots.items():
            if not isinstance(rep_id, str) or not isinstance(raw, dict):
                continue
            index[rep_id] = RuntimeSnapshot(
                support_count=int(raw.get("support_count", 0) or 0),
                counterexample_count=int(raw.get("counterexample_count", 0) or 0),
                times_activated=int(raw.get("times_activated", 0) or 0),
                times_helpful=int(raw.get("times_helpful", 0) or 0),
                times_harmful=int(raw.get("times_harmful", 0) or 0),
                last_support_tick=int(raw.get("last_support_tick", 0) or 0),
                last_activated_tick=int(raw.get("last_activated_tick", 0) or 0),
                lifecycle_status=str(raw.get("lifecycle_status", "candidate") or "candidate"),
            )
        return index

    def _save_state_index(self) -> None:
        if self._state_index is None:
            return
        payload = {
            "log_signature": self._log_signature(),
            "stats": {
                "record_count": int(self._record_count or 0),
                "representation_count": len(self._state_index),
                "log_mode": str(self._log_mode or "incremental"),
            },
            "snapshots": {
                rep_id: self._snapshot_dict(snapshot)
                for rep_id, snapshot in sorted(self._state_index.items())
            },
        }
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._summary_path.with_suffix(self._summary_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self._summary_path)

    @staticmethod
    def _apply_update_record(snap: RuntimeSnapshot, update: UpdateRecord) -> None:
        if update.update_type == UpdateType.SNAPSHOT:
            snapshot = update.snapshot if isinstance(update.snapshot, dict) else {}
            snap.support_count = int(snapshot.get("support_count", 0) or 0)
            snap.counterexample_count = int(snapshot.get("counterexample_count", 0) or 0)
            snap.times_activated = int(snapshot.get("times_activated", 0) or 0)
            snap.times_helpful = int(snapshot.get("times_helpful", 0) or 0)
            snap.times_harmful = int(snapshot.get("times_harmful", 0) or 0)
            snap.last_support_tick = int(snapshot.get("last_support_tick", 0) or 0)
            snap.last_activated_tick = int(snapshot.get("last_activated_tick", 0) or 0)
            snap.lifecycle_status = str(snapshot.get("lifecycle_status", "candidate") or "candidate")
            return
        if update.update_type == UpdateType.SUPPORT:
            snap.support_count += 1
            snap.last_support_tick = max(snap.last_support_tick, update.tick)
            return
        if update.update_type == UpdateType.COUNTEREXAMPLE:
            snap.counterexample_count += 1
            return
        if update.update_type == UpdateType.ACTIVATION:
            snap.times_activated += 1
            snap.last_activated_tick = max(snap.last_activated_tick, update.tick)
            if update.detail.get("helpful"):
                snap.times_helpful += 1
            else:
                snap.times_harmful += 1
            return
        if update.update_type == UpdateType.STATUS_CHANGE:
            snap.lifecycle_status = str(update.detail.get("new_status", snap.lifecycle_status) or snap.lifecycle_status)

    def _ensure_state_index(self) -> dict[str, RuntimeSnapshot]:
        if self._state_index is not None:
            return self._state_index
        cached_index = self._load_state_index_from_summary()
        if cached_index is not None:
            self._state_index = cached_index
            self._maybe_auto_compact()
            return self._state_index

        index: dict[str, RuntimeSnapshot] = {}
        for update in self.load_all_updates():
            snap = index.setdefault(update.rep_id, RuntimeSnapshot())
            self._apply_update_record(snap, update)
        self._state_index = index
        self._save_state_index()
        self._maybe_auto_compact()
        return self._state_index

    @staticmethod
    def _copy_snapshot(snap: RuntimeSnapshot) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            support_count=snap.support_count,
            counterexample_count=snap.counterexample_count,
            times_activated=snap.times_activated,
            times_helpful=snap.times_helpful,
            times_harmful=snap.times_harmful,
            last_support_tick=snap.last_support_tick,
            last_activated_tick=snap.last_activated_tick,
            lifecycle_status=snap.lifecycle_status,
        )

    @staticmethod
    def _resolve_auto_compact_threshold(env_name: str, default: int) -> int:
        raw = str(os.getenv(env_name, "") or "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    def compact(self, *, force: bool = False) -> dict[str, int | bool]:
        state_index = self._ensure_state_index()
        current_record_count = self._record_count if self._record_count is not None else len(self.load_all_updates())
        compact_records = [
            UpdateRecord(
                rep_id=rep_id,
                tick=max(snapshot.last_support_tick, snapshot.last_activated_tick),
                update_type=UpdateType.SNAPSHOT,
                detail={"source": "runtime_compaction"},
                snapshot=self._snapshot_dict(snapshot),
            )
            for rep_id, snapshot in sorted(state_index.items())
        ]

        if not force and current_record_count <= len(compact_records):
            return {
                "compacted": False,
                "records_before": int(current_record_count),
                "records_after": len(compact_records),
            }

        self._ensure_path()
        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            for record in compact_records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        temp_path.replace(self.store_path)

        self._updates = compact_records
        self._record_count = len(compact_records)
        self._log_mode = "snapshot_compacted"
        self._save_state_index()
        return {
            "compacted": True,
            "records_before": int(current_record_count),
            "records_after": len(compact_records),
        }

    def _maybe_auto_compact(self) -> None:
        if self._state_index is None or not self.store_path.exists():
            return

        min_bytes = self._resolve_auto_compact_threshold(
            "THE_AGI_RUNTIME_UPDATES_AUTO_COMPACT_MIN_BYTES",
            8 * 1024 * 1024,
        )
        min_records = self._resolve_auto_compact_threshold(
            "THE_AGI_RUNTIME_UPDATES_AUTO_COMPACT_MIN_RECORDS",
            10_000,
        )
        current_record_count = self._record_count if self._record_count is not None else 0
        if min_bytes > 0 and self.store_path.stat().st_size < min_bytes and current_record_count < min_records:
            return
        if current_record_count <= len(self._state_index):
            return
        if self._log_mode == "snapshot_compacted" and current_record_count <= len(self._state_index) + 1:
            return
        self.compact()

    # ---- Replay ----

    def get_card_runtime_state(self, rep_id: str) -> RuntimeSnapshot:
        """
        Replay all updates for a card to compute current RuntimeSnapshot.
        """
        snap = self._ensure_state_index().get(rep_id)
        if snap is None:
            return RuntimeSnapshot()
        return self._copy_snapshot(snap)

    def replay_card(
        self,
        card: RepresentationCard,
        rep_id: str,
    ) -> RepresentationCard:
        """
        Return a copy of the card with evidence and lifecycle
        updated to reflect all persisted runtime updates.
        """
        updated = copy.deepcopy(card)

        snap = self.get_card_runtime_state(rep_id)
        updated.evidence.support_count = snap.support_count
        updated.evidence.counterexample_count = snap.counterexample_count
        updated.lifecycle.times_activated = snap.times_activated
        updated.lifecycle.times_helpful = snap.times_helpful
        updated.lifecycle.times_harmful = snap.times_harmful
        updated.lifecycle.last_support_tick = snap.last_support_tick
        updated.lifecycle.last_activated_tick = snap.last_activated_tick
        updated.lifecycle.current_status = snap.lifecycle_status

        return updated

    # ---- Helpers ----

    @staticmethod
    def _snapshot_dict(snap: Optional[RuntimeSnapshot]) -> dict:
        if snap is None:
            return {}
        return {
            "support_count": snap.support_count,
            "counterexample_count": snap.counterexample_count,
            "times_activated": snap.times_activated,
            "times_helpful": snap.times_helpful,
            "times_harmful": snap.times_harmful,
            "last_support_tick": snap.last_support_tick,
            "last_activated_tick": snap.last_activated_tick,
            "lifecycle_status": snap.lifecycle_status,
        }

    def clear(self) -> None:
        """Delete all runtime updates (for testing only)."""
        if self.store_path.exists():
            self.store_path.unlink()
        if self._summary_path.exists():
            self._summary_path.unlink()
        self._updates = None
        self._state_index = None
        self._record_count = None
        self._log_mode = "incremental"


# ============================================================
# Module-level convenience
# ============================================================

# Singleton instances (lazily initialized)
_warehouse: Optional[CardWarehouse] = None
_runtime_store: Optional[RuntimeUpdateStore] = None


def get_warehouse() -> CardWarehouse:
    global _warehouse
    if _warehouse is None:
        _warehouse = CardWarehouse()
        _warehouse.load()
    return _warehouse


def get_runtime_store() -> RuntimeUpdateStore:
    global _runtime_store
    if _runtime_store is None:
        _runtime_store = RuntimeUpdateStore()
    return _runtime_store
