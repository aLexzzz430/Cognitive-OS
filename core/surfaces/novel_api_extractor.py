"""
core/surfaces/novel_api_extractor.py

NovelAPIRawEvidenceExtractor: extracts raw evidence from NovelAPI interactions.

Evidence kinds:
  - visible_function_signature: API reveals function name(s)
  - error_missing_required_argument: tool call failed due to missing arg
  - error_type_mismatch: tool call failed due to type error
  - error_hidden_precondition: tool call failed due to hidden precondition
  - error_permission_or_scope_constraint: tool call blocked by permission/scope
  - successful_tool_invocation: tool call succeeded
  - failed_tool_invocation: tool call failed (general)
  - ordering_dependency_hint: hint suggests ordering between operations
  - state_change_after_call: environment state changed after tool call
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any

from core.surfaces.base import EvidencePacket, RawEvidenceExtractor


NOVEL_API_EVIDENCE_KINDS = [
    "visible_function_signature",
    "visible_parameter_constraint",
    "error_missing_required_argument",
    "error_type_mismatch",
    "error_hidden_precondition",
    "error_permission_or_scope_constraint",
    "successful_tool_invocation",
    "failed_tool_invocation",
    "ordering_dependency_hint",
    "state_change_after_call",
]

logger = logging.getLogger(__name__)


class NovelAPIRawEvidenceExtractor:
    """
    Extracts raw EvidencePackets from NovelAPI surface interactions.

    This is C1/C5-agnostic — it only looks at:
      - Tool call result (success/error)
      - Visible docs (function names)
      - Error messages
      - Hints from the environment
      - State deltas
    """

    def __init__(self):
        self._call_history: list[dict] = []

    @staticmethod
    def _normalize_function_names(value: Any) -> list[str]:
        """Normalize visible/discovered function payloads into an ordered list."""
        if isinstance(value, dict):
            candidates = value.keys()
        elif isinstance(value, (list, tuple)):
            candidates = value
        elif value:
            candidates = [value]
        else:
            candidates = []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            name = str(item).strip()
            if name and name not in seen:
                seen.add(name)
                normalized.append(name)
        return normalized

    def _extract_visible_functions(
        self,
        obs_structured: dict[str, Any],
        raw_result: dict[str, Any],
        tool_name: str,
    ) -> list[str]:
        prioritized_sources = [
            ("obs_structured.visible_functions", obs_structured.get("visible_functions")),
            ("raw_result.visible_functions", raw_result.get("visible_functions")),
            ("obs_structured.discovered_functions", obs_structured.get("discovered_functions")),
            ("raw_result.discovered_functions", raw_result.get("discovered_functions")),
        ]

        merged: list[str] = []
        seen: set[str] = set()
        used_sources: list[str] = []
        for source_name, raw_value in prioritized_sources:
            current = self._normalize_function_names(raw_value)
            if not current:
                continue
            used_sources.append(source_name)
            for fn in current:
                if fn not in seen:
                    seen.add(fn)
                    merged.append(fn)

        if not merged:
            logger.warning(
                "No visible surface functions found. no_visible_surface=true tool_name=%s",
                tool_name or "<unknown>",
            )
        else:
            logger.debug(
                "Resolved visible surface functions from sources=%s count=%d",
                ",".join(used_sources),
                len(merged),
            )
        return merged

    def extract(
        self,
        observation_before: dict[str, Any],
        action_taken: dict[str, Any],
        action_result: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> list[EvidencePacket]:
        """
        Extract raw evidence from a single NovelAPI interaction.

        Args:
            observation_before: The observation before the action was taken
            action_taken: The action selected (contains tool_name, tool_args)
            action_result: The result dict from _step8_act (may contain env_result)
            runtime_state: Current runtime state

        Returns:
            List of EvidencePackets extracted from this interaction
        """
        packets: list[EvidencePacket] = []

        # Unwrap env_result if present (CoreMainLoop._step8_act wraps ActionResult in dict)
        raw_result: dict[str, Any] = {}
        if isinstance(action_result, dict):
            if 'env_result' in action_result:
                er = action_result['env_result']
                if hasattr(er, 'raw'):
                    raw_result = er.raw
                elif isinstance(er, dict):
                    raw_result = er
            elif 'raw' in action_result:
                raw_result = action_result['raw']
            else:
                raw_result = action_result

        if hasattr(action_result, 'observation') and hasattr(action_result.observation, 'structured'):
            obs_structured = action_result.observation.structured
        elif isinstance(observation_before, dict):
            obs_structured = observation_before.get('structured', observation_before)
        else:
            obs_structured = {}

        # Extract tool call info
        if isinstance(action_taken, dict):
            tool_name = action_taken.get('tool_name', '') or action_taken.get('payload', {}).get('tool_name', '')
            tool_args = action_taken.get('tool_args', {}) or action_taken.get('payload', {}).get('tool_args', {})
            if not tool_args and isinstance(action_taken.get('payload'), dict):
                tool_args = {k: v for k, v in action_taken['payload'].items() if k not in ('tool_name', 'tool_args')}
        else:
            tool_name = ''
            tool_args = {}

        # Track this call
        self._call_history.append({
            'tool_name': tool_name,
            'tool_args': tool_args,
            'result': raw_result,
        })

        # 1. visible_function_signature
        visible_functions = self._extract_visible_functions(obs_structured, raw_result, tool_name)
        for fn in visible_functions:
            if fn and not self._already_extracted('visible_function_signature', fn):
                packets.append(EvidencePacket(
                    type="visible_function_signature",
                    kind="visible_function_signature",
                    claim=f"Function '{fn}' is visible on the API surface.",
                    content={
                        "function_name": fn,
                        "discovery_step": len(self._call_history),
                    },
                    payload={
                        "function_name": fn,
                    },
                    source_refs=[f"discovered:{fn}"],
                    confidence=0.9,
                ))

        # 2. successful_tool_invocation
        ok = raw_result.get('ok', False)
        error = raw_result.get('error')
        called = raw_result.get('called', False)
        result_val = raw_result.get('result')

        if ok or called:
            packets.append(EvidencePacket(
                type="successful_tool_invocation",
                kind="successful_tool_invocation",
                claim=f"Tool invocation succeeded: {tool_name}.",
                content={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": str(result_val)[:200] if result_val is not None else None,
                },
                payload={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result_val,
                },
                source_refs=[f"tool_call:{tool_name}"],
                confidence=0.95,
            ))

            # 3. state_change_after_call
            if result_val is not None:
                packets.append(EvidencePacket(
                    type="state_change_after_call",
                    kind="state_change_after_call",
                    claim=f"Environment state changed after calling {tool_name}.",
                    content={
                        "tool_name": tool_name,
                        "state_delta": str(result_val)[:200],
                    },
                    payload={
                        "tool_name": tool_name,
                        "state_delta": result_val,
                    },
                    source_refs=[f"state_change:{tool_name}"],
                    confidence=0.8,
                ))

        # 4. failed_tool_invocation
        if error and not ok:
            packets.append(EvidencePacket(
                type="failed_tool_invocation",
                kind="failed_tool_invocation",
                claim=f"Tool invocation failed: {tool_name}. Error: {error}",
                content={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "error": error,
                },
                payload={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "error": error,
                },
                source_refs=[f"error:{tool_name}"],
                confidence=0.95,
            ))

            # 5. Error classification
            error_lower = str(error).lower() if error else ''

            if 'missing' in error_lower and 'argument' in error_lower:
                packets.append(EvidencePacket(
                    type="error_missing_required_argument",
                    kind="error_missing_required_argument",
                    claim="A tool invocation failed because a required argument was missing.",
                    content={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    payload={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    source_refs=[f"error_type:missing_arg"],
                    confidence=0.9,
                ))

            if 'type' in error_lower or 'invalid type' in error_lower or 'typeerror' in error_lower:
                packets.append(EvidencePacket(
                    type="error_type_mismatch",
                    kind="error_type_mismatch",
                    claim="A tool invocation failed due to argument type mismatch.",
                    content={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    payload={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    source_refs=[f"error_type:type_mismatch"],
                    confidence=0.85,
                ))

            if 'precondition' in error_lower or 'must first' in error_lower or 'not ready' in error_lower or 'unknown' in error_lower:
                packets.append(EvidencePacket(
                    type="error_hidden_precondition",
                    kind="error_hidden_precondition",
                    claim="A hidden precondition may exist before this tool can be used successfully.",
                    content={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    payload={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    source_refs=[f"error_type:hidden_precondition"],
                    confidence=0.75,
                ))

            if 'permission' in error_lower or 'unauthorized' in error_lower or 'scope' in error_lower:
                packets.append(EvidencePacket(
                    type="error_permission_or_scope_constraint",
                    kind="error_permission_or_scope_constraint",
                    claim="A permission or scope restriction blocked the tool invocation.",
                    content={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    payload={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error,
                    },
                    source_refs=[f"error_type:permission_scope"],
                    confidence=0.8,
                ))

        # 6. ordering_dependency_hint (from hints in observation)
        hints = obs_structured.get('hints', []) or raw_result.get('hints', [])
        for hint in hints:
            hint_text = str(hint).lower()
            if any(word in hint_text for word in ['before', 'first', 'after', 'then', 'order']):
                packets.append(EvidencePacket(
                    type="ordering_dependency_hint",
                    kind="ordering_dependency_hint",
                    claim="There may be an ordering dependency between API operations.",
                    content={
                        "hint": str(hint),
                        "tool_name": tool_name,
                    },
                    payload={
                        "hint": str(hint),
                        "tool_name": tool_name,
                    },
                    source_refs=["hint:ordering"],
                    confidence=0.65,
                ))

        return packets

    def _already_extracted(self, kind: str, identifier: str) -> bool:
        """Check if we already extracted this evidence in a previous call."""
        for call in self._call_history[:-1]:
            if kind == "visible_function_signature":
                historical_visible = self._normalize_function_names(call.get("result", {}).get("visible_functions"))
                if identifier in historical_visible:
                    return True
        return False
