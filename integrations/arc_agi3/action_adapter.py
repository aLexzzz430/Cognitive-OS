from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

MAX_REASONING_BYTES = 16000  # API limit is 16384, stay safely under


@dataclass(frozen=True)
class ARCActionChoice:
    action_name: str
    action_id: int
    data: Optional[Dict[str, int]] = None
    reasoning: Optional[Dict[str, Any]] = None
    requires_env_step: bool = True
    external_cost: int = 1
    internal_action_kind: Optional[str] = None


class ARCActionAdapter:
    """
    Map AGI_WORLD_V2 action dictionaries onto ARC-AGI-3's standardized actions.

    Important semantic rule:
    - Internal actions such as inspect / wait / reflect are NEVER sent to ARC as
      real surface actions.
    - They become internal-only no-ops that still pay external cost in the bridge.
    """

    ACTION_NAME_TO_ID = {
        "RESET": 0,
        "ACTION1": 1,
        "ACTION2": 2,
        "ACTION3": 3,
        "ACTION4": 4,
        "ACTION5": 5,
        "ACTION6": 6,
        "ACTION7": 7,
    }

    ALIASES = {
        "UP": "ACTION1",
        "DOWN": "ACTION2",
        "LEFT": "ACTION3",
        "RIGHT": "ACTION4",
        "INTERACT": "ACTION5",
        "EXECUTE": "ACTION5",
        "CLICK": "ACTION6",
        "POINT": "ACTION6",
        "UNDO": "ACTION7",
    }

    INTERNAL_ONLY_ACTIONS = {
        "inspect",
        "wait",
        "reflect",
        "replan",
        "retrieve",
        "deliberate",
    }

    def to_arc_action(
        self,
        action: Dict[str, Any],
        *,
        available_action_names: Sequence[str],
        perception: Optional[Dict[str, Any]] = None,
    ) -> ARCActionChoice:
        internal_kind = self._extract_internal_action_kind(action)
        if internal_kind:
            return self._build_internal_noop_choice(internal_kind, action)

        fn_name = self._extract_function_name(action).upper()
        available = [str(x).upper() for x in available_action_names]

        canonical = self.ALIASES.get(fn_name, fn_name)
        if canonical not in self.ACTION_NAME_TO_ID:
            canonical = self._fallback_non_wait(available)

        if available and canonical not in available:
            canonical = self._fallback_non_wait(available)

        return self._build_choice(canonical, action, perception)

    def _build_internal_noop_choice(
        self,
        internal_kind: str,
        action: Dict[str, Any],
    ) -> ARCActionChoice:
        reasoning_raw = {
            "agi_world_source": str(action.get("_source", "") or ""),
            "internal_action_kind": internal_kind,
            "noop_policy": "paid_external_noop",
            "candidate_meta": dict(action.get("_candidate_meta", {}) or {}) if isinstance(action.get("_candidate_meta", {}), dict) else {},
        }
        reasoning = self._json_safe(reasoning_raw)
        return ARCActionChoice(
            action_name="WAIT_INTERNAL",
            action_id=-1,
            data=None,
            reasoning=reasoning,
            requires_env_step=False,
            external_cost=1,
            internal_action_kind=internal_kind,
        )

    def _build_choice(
        self,
        canonical: str,
        action: Dict[str, Any],
        perception: Optional[Dict[str, Any]],
    ) -> ARCActionChoice:
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
        kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args, dict) else {}
        kwargs = kwargs if isinstance(kwargs, dict) else {}

        reasoning_raw = {
            "agi_world_source": str(action.get("_source", "") or ""),
            "candidate_meta": dict(action.get("_candidate_meta", {}) or {}) if isinstance(action.get("_candidate_meta", {}), dict) else {},
        }
        reasoning = self._json_safe(reasoning_raw)
        reasoning = self._truncate_reasoning(reasoning)

        data = None
        if canonical == "ACTION6":
            x, y = self._resolve_xy(kwargs, perception)
            data = {"x": x, "y": y}

        return ARCActionChoice(
            action_name=canonical,
            action_id=self.ACTION_NAME_TO_ID[canonical],
            data=self._json_safe(data),
            reasoning=reasoning,
            requires_env_step=True,
            external_cost=1,
            internal_action_kind=None,
        )

    def _resolve_xy(
        self,
        kwargs: Dict[str, Any],
        perception: Optional[Dict[str, Any]],
    ) -> Tuple[int, int]:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if self._valid_coord(x) and self._valid_coord(y):
            return int(x), int(y)
        raise ValueError(
            "ACTION6 requires explicit x/y before ARCActionAdapter.to_arc_action; "
            f"received kwargs keys={sorted(kwargs.keys())}"
        )

    def _fallback_non_wait(self, available: Sequence[str]) -> str:
        for candidate in ("ACTION5", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6", "ACTION7"):
            if not available or candidate in available:
                return candidate
        return "ACTION1"

    def _extract_internal_action_kind(self, action: Dict[str, Any]) -> Optional[str]:
        if not isinstance(action, dict):
            return None

        kind = str(action.get("kind", "") or "").strip().lower()
        if kind in self.INTERNAL_ONLY_ACTIONS:
            return kind

        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
        for value in (
            tool_args.get("function_name"),
            payload.get("function_name"),
            action.get("function_name"),
            action.get("selected_name"),
        ):
            if isinstance(value, str) and value.strip().lower() in self.INTERNAL_ONLY_ACTIONS:
                return value.strip().lower()

        return None

    def _json_safe(self, obj: Any, _seen: Optional[set[int]] = None) -> Any:
        """
        Convert AGI_WORLD_V2 internal objects into JSON-safe payloads before
        handing them to ARC remote wrappers.
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
                safe_dict: Dict[str, Any] = {}
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

    @staticmethod
    def _extract_function_name(action: Dict[str, Any]) -> str:
        if not isinstance(action, dict):
            return "WAIT"
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
        for value in (
            tool_args.get("function_name"),
            payload.get("function_name"),
            action.get("function_name"),
            action.get("selected_name"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "WAIT"

    @staticmethod
    def _valid_coord(value: Any) -> bool:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= iv <= 63

    @staticmethod
    def _clip(value: int) -> int:
        return max(0, min(63, int(value)))

    def _truncate_reasoning(self, reasoning: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Truncate reasoning dict to stay under MAX_REASONING_BYTES when JSON-serialized.
        The API rejects requests where the reasoning field exceeds 16384 bytes.
        """
        if reasoning is None:
            return None

        def _truncate_str_fields(obj: Any, max_len: int = 2000) -> Any:
            """Recursively truncate string fields in nested structures."""
            if obj is None or isinstance(obj, (bool, int, float)):
                return obj
            if isinstance(obj, str):
                return obj[:max_len] + "..." if len(obj) > max_len else obj
            if isinstance(obj, dict):
                return {k: _truncate_str_fields(v, max_len) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_truncate_str_fields(item, max_len) for item in obj]
            return obj

        # Iteratively truncate until we fit
        truncated = reasoning
        for max_len in (4000, 2000, 1000, 500, 200):
            truncated = _truncate_str_fields(truncated, max_len=max_len)
            serialized = json.dumps(truncated, separators=(',', ':'))
            if len(serialized.encode('utf-8')) <= MAX_REASONING_BYTES:
                return truncated

        # Last resort: return minimal reasoning
        return {
            "agi_world_source": str(reasoning.get("agi_world_source", "")),
            "_truncated": True,
        }
