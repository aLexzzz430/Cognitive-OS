from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class WebArenaActionChoice:
    action_name: str
    native_action: Any
    reasoning: Optional[Dict[str, Any]] = None
    requires_env_step: bool = True
    external_cost: int = 1
    internal_action_kind: Optional[str] = None


class WebArenaActionAdapter:
    INTERNAL_ONLY_ACTIONS = {
        "inspect",
        "wait",
        "reflect",
        "replan",
        "retrieve",
        "deliberate",
    }

    ACTION_ALIASES = {
        "CLICK": "click",
        "CLICK_ELEMENT": "click",
        "CLICK_ID": "click",
        "TYPE": "type",
        "TYPE_TEXT": "type",
        "SELECT": "select",
        "HOVER": "hover",
        "PRESS": "press",
        "SCROLL": "scroll",
        "GO_BACK": "go_back",
        "BACK": "go_back",
        "OPEN_URL": "goto",
        "GOTO": "goto",
        "OPEN_TAB": "new_tab",
        "NEW_TAB": "new_tab",
        "SWITCH_TAB": "tab_focus",
        "TAB_FOCUS": "tab_focus",
        "SUBMIT": "submit",
    }

    def __init__(self, action_factory: Any | None = None) -> None:
        self._action_factory = action_factory

    def to_webarena_action(
        self,
        action: Dict[str, Any],
        *,
        available_action_names: Sequence[str] = (),
    ) -> WebArenaActionChoice:
        internal_kind = self._extract_internal_action_kind(action)
        if internal_kind:
            return WebArenaActionChoice(
                action_name="WAIT_INTERNAL",
                native_action=None,
                reasoning=self._reasoning_payload(action, internal_kind=internal_kind),
                requires_env_step=False,
                external_cost=1,
                internal_action_kind=internal_kind,
            )

        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        kwargs = dict(tool_args.get("kwargs", {}) or {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
        native_action = tool_args.get("native_action")
        if native_action is not None:
            return WebArenaActionChoice(
                action_name=self._canonical_action_name(action),
                native_action=native_action,
                reasoning=self._reasoning_payload(action),
            )

        canonical = self._canonical_action_name(action)
        allowed = {str(name or "").strip().lower() for name in list(available_action_names or []) if str(name or "").strip()}
        if allowed and canonical.lower() not in allowed:
            raise ValueError(f"WebArena action {canonical!r} not available; visible={sorted(allowed)}")

        command = self._build_command(canonical, kwargs)
        native = self._action_factory(command) if callable(self._action_factory) else command
        return WebArenaActionChoice(
            action_name=canonical,
            native_action=native,
            reasoning=self._reasoning_payload(action),
        )

    def _canonical_action_name(self, action: Dict[str, Any]) -> str:
        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        for value in (
            tool_args.get("function_name"),
            payload.get("function_name"),
            action.get("function_name"),
            action.get("selected_name"),
            action.get("kind"),
        ):
            text = str(value or "").strip()
            if text:
                upper = text.upper()
                return self.ACTION_ALIASES.get(upper, text.lower())
        return "wait"

    def _extract_internal_action_kind(self, action: Dict[str, Any]) -> Optional[str]:
        for value in (
            action.get("kind"),
            action.get("function_name"),
            action.get("selected_name"),
            ((action.get("payload", {}) or {}).get("tool_args", {}) or {}).get("function_name")
            if isinstance((action.get("payload", {}) or {}).get("tool_args", {}), dict)
            else None,
        ):
            text = str(value or "").strip().lower()
            if text in self.INTERNAL_ONLY_ACTIONS:
                return text
        return None

    def _reasoning_payload(self, action: Dict[str, Any], *, internal_kind: str = "") -> Dict[str, Any]:
        meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
        return {
            "agi_world_source": str(action.get("_source", "") or ""),
            "candidate_meta": dict(meta),
            "internal_action_kind": internal_kind,
        }

    def _build_command(self, canonical: str, kwargs: Dict[str, Any]) -> str:
        name = str(canonical or "").strip().lower()
        if name == "click":
            element_id = self._required(kwargs, ("element_id", "id", "element_ref"))
            return f"click [{element_id}]"
        if name == "hover":
            element_id = self._required(kwargs, ("element_id", "id", "element_ref"))
            return f"hover [{element_id}]"
        if name == "type":
            element_id = self._required(kwargs, ("element_id", "id", "element_ref"))
            text = self._required(kwargs, ("text", "value"))
            return f"type [{element_id}] [{text}]"
        if name == "select":
            element_id = self._required(kwargs, ("element_id", "id", "element_ref"))
            option = self._required(kwargs, ("option", "value", "text"))
            return f"select [{element_id}] [{option}]"
        if name == "press":
            key = self._required(kwargs, ("key", "keys"))
            return f"press [{key}]"
        if name == "scroll":
            direction = str(kwargs.get("direction", kwargs.get("scroll_direction", "down")) or "down").strip()
            amount = kwargs.get("amount")
            return f"scroll [{direction}]" if amount is None else f"scroll [{direction}] [{amount}]"
        if name == "goto":
            url = self._required(kwargs, ("url", "href"))
            return f"goto [{url}]"
        if name == "go_back":
            return "go_back"
        if name == "new_tab":
            url = str(kwargs.get("url", "") or "").strip()
            return f"new_tab [{url}]" if url else "new_tab"
        if name == "tab_focus":
            tab_id = self._required(kwargs, ("tab_id", "index"))
            return f"tab_focus [{tab_id}]"
        if name == "submit":
            element_id = str(kwargs.get("element_id", kwargs.get("id", "")) or "").strip()
            if element_id:
                return f"click [{element_id}]"
            return "press [ENTER]"
        if name == "wait":
            return "wait"
        raise ValueError(f"Unsupported WebArena action {canonical!r}")

    @staticmethod
    def _required(kwargs: Dict[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            value = kwargs.get(key)
            text = str(value or "").strip()
            if text:
                return text
        raise ValueError(f"Missing required WebArena action parameter; expected one of {list(keys)}")
