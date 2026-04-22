from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.objects import (
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
    infer_object_type,
)
from modules.memory.schema import MemoryLayer, MemoryType, RetrievalTag


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, (tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _episode_content(episode_record: Dict[str, Any]) -> Dict[str, Any]:
    content = episode_record.get("content", {})
    return dict(content) if isinstance(content, dict) else {}


def _episode_id(episode_record: Dict[str, Any]) -> int:
    content = _episode_content(episode_record)
    try:
        return int(content.get("episode_id", episode_record.get("trigger_episode", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _episode_discoveries(episode_record: Dict[str, Any]) -> List[str]:
    content = _episode_content(episode_record)
    return _normalize_list(content.get("key_discoveries", []))


def _episode_callable_functions(episode_record: Dict[str, Any]) -> List[str]:
    content = _episode_content(episode_record)
    actions = content.get("actions", [])
    out: List[str] = []
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, dict):
            continue
        name = str(action.get("function_name") or "").strip()
        if name:
            out.append(name)
    return _normalize_list(out)


def _episode_mechanism_signals(episode_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = _episode_content(episode_record)
    signals = content.get("mechanism_signals", [])
    return [dict(item) for item in signals if isinstance(item, dict)] if isinstance(signals, list) else []


def _mechanism_signal_signature(signal: Dict[str, Any]) -> str:
    goal_family = str(signal.get("goal_family") or "").strip()
    action_function = str(signal.get("action_function") or "").strip()
    mechanism_kind = str(signal.get("mechanism_kind") or "").strip() or "generic"
    supported_anchor_count = len(_normalize_list(signal.get("supported_goal_anchor_refs", [])))
    supported_color_count = len(_normalize_list(signal.get("supported_goal_colors", [])))
    return "|".join(
        [
            mechanism_kind,
            goal_family or "unknown_goal",
            action_function or "any_function",
            str(supported_anchor_count),
            str(supported_color_count),
        ]
    )


def _positive_episode(episode_record: Dict[str, Any]) -> bool:
    content = _episode_content(episode_record)
    reward_trend = str(content.get("reward_trend") or "").strip().lower()
    total_reward = float(content.get("total_reward", 0.0) or 0.0)
    return reward_trend != "negative" and total_reward >= 0.0


def _callable_invocation(record: Dict[str, Any]) -> Dict[str, Any]:
    content = record.get("content", {})
    if not isinstance(content, dict):
        return {}
    explicit = content.get("invocation_schema", {})
    if isinstance(explicit, dict) and explicit.get("function_name"):
        return dict(explicit)
    tool_args = content.get("tool_args", {})
    if isinstance(tool_args, dict) and tool_args.get("function_name"):
        return {
            "callable_form": "call_tool",
            "function_name": str(tool_args.get("function_name") or ""),
            "kwargs_schema": dict(tool_args.get("kwargs_schema", {})) if isinstance(tool_args.get("kwargs_schema"), dict) else {},
        }
    function_name = str(content.get("function_name") or "").strip()
    if function_name:
        return {
            "callable_form": str(content.get("callable_form") or "call_tool"),
            "function_name": function_name,
        }
    return {}


class MemoryPromotionRules:
    """Formal promotion heuristics for semantic, skill, transfer, and autobiographical objects."""

    def __init__(self, object_store):
        self._store = object_store

    def proposals_from_episode(self, episode_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        semantic = self.propose_semantic_from_episode(episode_record)
        if semantic:
            proposals.append(semantic)
        mechanism = self.propose_mechanism_from_episode(episode_record)
        if mechanism:
            proposals.append(mechanism)
        autobiographical = self.propose_autobiographical_from_episode(episode_record)
        if autobiographical:
            proposals.append(autobiographical)
        return proposals

    def propose_semantic_from_episode(self, episode_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        discoveries = _episode_discoveries(episode_record)
        functions = _episode_callable_functions(episode_record)
        evidence_key = discoveries[0] if discoveries else (f"successful_call:{functions[0]}" if functions else "")
        if not evidence_key or not _positive_episode(episode_record):
            return None

        supporting_records: List[Dict[str, Any]] = []
        for candidate in self._store.get_by_memory_type(MemoryType.EPISODE_RECORD.value):
            if candidate.get("status") == "invalidated" or not _positive_episode(candidate):
                continue
            candidate_discoveries = _episode_discoveries(candidate)
            candidate_functions = _episode_callable_functions(candidate)
            if evidence_key in candidate_discoveries or evidence_key in [f"successful_call:{name}" for name in candidate_functions]:
                supporting_records.append(candidate)

        if len(supporting_records) < 2:
            return None

        source_episode_ids = [_episode_id(record) for record in supporting_records if _episode_id(record) > 0]
        source_object_ids = _normalize_list([record.get("object_id", "") for record in supporting_records])
        evidence_ids: List[str] = []
        for record in supporting_records:
            evidence_ids.extend(_normalize_list(record.get("evidence_ids", [])))
        evidence_chain = _normalize_list(source_object_ids + evidence_ids)

        return {
            "object_type": OBJECT_TYPE_REPRESENTATION,
            "memory_type": MemoryType.FACT_CARD.value,
            "memory_layer": MemoryLayer.SEMANTIC.value,
            "family": evidence_key.replace(":", "_"),
            "summary": f"validated:{evidence_key}",
            "content": {
                "type": MemoryType.FACT_CARD.value,
                "assertion": evidence_key,
                "validation_mode": "repeated_validated_episode_support",
                "support_count": len(supporting_records),
                "source_episode_ids": source_episode_ids,
                "source_object_ids": source_object_ids,
            },
            "confidence": min(0.95, 0.45 + 0.1 * len(supporting_records)),
            "retrieval_tags": [
                RetrievalTag.CONSOLIDATION.value,
                RetrievalTag.EXPLORATION.value,
                RetrievalTag.AGENT.value,
            ],
            "evidence_ids": evidence_chain,
            "memory_metadata": {
                "promotion_kind": "semantic",
                "source_episode_ids": source_episode_ids,
                "source_object_ids": source_object_ids,
                "source_evidence_chain": evidence_chain,
                "validation_count": len(supporting_records),
            },
            "trigger_source": "memory_promotion_rules",
            "trigger_episode": _episode_id(episode_record),
        }

    def propose_skill_from_object(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if infer_object_type(record) != OBJECT_TYPE_SKILL:
            return None

        invocation_schema = _callable_invocation(record)
        function_name = str(invocation_schema.get("function_name") or "").strip()
        if not function_name:
            return None

        beneficial = [item for item in record.get("reuse_history", []) or [] if item.get("was_beneficial", False)]
        if len(beneficial) < 2:
            return None

        memory_metadata = record.get("memory_metadata", {}) if isinstance(record.get("memory_metadata"), dict) else {}
        owner_scope = str(record.get("owner_scope") or memory_metadata.get("owner_scope") or "system").strip().lower()
        post_teacher_beneficial = sum(1 for item in beneficial if item.get("teacher_present") is False)
        if owner_scope == "teacher" and post_teacher_beneficial <= 0:
            return None

        content = record.get("content", {}) if isinstance(record.get("content"), dict) else {}
        applicability = record.get("applicability", {}) if isinstance(record.get("applicability"), dict) else {}
        failure_conditions = _normalize_list(record.get("failure_conditions") or content.get("failure_conditions", []))
        negative_examples = _normalize_list(content.get("negative_examples", []))
        if not negative_examples:
            negative_examples = _normalize_list(
                item.get("reason") or item.get("failure") or "non_beneficial_reuse"
                for item in record.get("reuse_history", []) or []
                if not item.get("was_beneficial", False)
            )
        expected_gains = _normalize_list(content.get("expected_gains", [])) or [f"beneficial_reuse:{len(beneficial)}"]
        evidence_chain = _normalize_list(
            list(record.get("evidence_ids", []) or [])
            + list(content.get("source_evidence_chain", []) or [])
        )

        return {
            "object_type": OBJECT_TYPE_SKILL,
            "memory_type": MemoryType.SKILL_CARD.value,
            "memory_layer": MemoryLayer.PROCEDURAL.value,
            "family": str(record.get("family") or function_name),
            "summary": str(record.get("summary") or f"validated skill:{function_name}"),
            "content": {
                "type": MemoryType.SKILL_CARD.value,
                "callable_form": str(invocation_schema.get("callable_form") or "call_tool"),
                "invocation_schema": invocation_schema,
                "expected_gains": expected_gains,
                "negative_examples": negative_examples,
                "contradiction_hooks": failure_conditions,
                "source_evidence_chain": evidence_chain,
                "rewrite_hints": dict(content.get("rewrite_hints", {})) if isinstance(content.get("rewrite_hints"), dict) else {},
            },
            "applicability": applicability,
            "failure_conditions": failure_conditions,
            "confidence": min(0.97, float(record.get("confidence", 0.5) or 0.5) + 0.05),
            "retrieval_tags": _normalize_list(list(record.get("retrieval_tags", []) or []) + [RetrievalTag.EXPLOITATION.value, RetrievalTag.AGENT.value]),
            "evidence_ids": evidence_chain,
            "memory_metadata": {
                "promotion_kind": "skill",
                "source_object_ids": [str(record.get("object_id") or "")],
                "source_evidence_chain": evidence_chain,
                "beneficial_reuse_count": len(beneficial),
                "post_teacher_beneficial_count": post_teacher_beneficial,
            },
            "trigger_source": "skill_compiler",
            "trigger_episode": int(record.get("trigger_episode", 0) or 0),
        }

    def propose_mechanism_from_episode(self, episode_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _positive_episode(episode_record):
            return None

        signals = _episode_mechanism_signals(episode_record)
        if not signals:
            return None

        best_signal: Optional[Dict[str, Any]] = None
        best_supporting_records: List[Dict[str, Any]] = []
        for signal in signals:
            signature = _mechanism_signal_signature(signal)
            if not signature:
                continue

            supporting_records: List[Dict[str, Any]] = []
            for candidate in self._store.get_by_memory_type(MemoryType.EPISODE_RECORD.value):
                if candidate.get("status") == "invalidated" or not _positive_episode(candidate):
                    continue
                candidate_signals = _episode_mechanism_signals(candidate)
                if any(_mechanism_signal_signature(row) == signature for row in candidate_signals):
                    supporting_records.append(candidate)

            if len(supporting_records) < 2:
                continue
            if best_signal is None or len(supporting_records) > len(best_supporting_records):
                best_signal = signal
                best_supporting_records = supporting_records

        if best_signal is None or len(best_supporting_records) < 2:
            return None

        goal_family = str(best_signal.get("goal_family") or "").strip() or "unknown_goal"
        action_function = str(best_signal.get("action_function") or "").strip()
        controller_anchor_refs: List[str] = []
        supported_goal_anchor_refs: List[str] = []
        supported_goal_colors: List[int] = []
        for record in best_supporting_records:
            for candidate_signal in _episode_mechanism_signals(record):
                if _mechanism_signal_signature(candidate_signal) != _mechanism_signal_signature(best_signal):
                    continue
                for ref in _normalize_list(candidate_signal.get("controller_anchor_refs", [])):
                    if ref not in controller_anchor_refs:
                        controller_anchor_refs.append(ref)
                for ref in _normalize_list(candidate_signal.get("supported_goal_anchor_refs", [])):
                    if ref not in supported_goal_anchor_refs:
                        supported_goal_anchor_refs.append(ref)
                for value in list(candidate_signal.get("supported_goal_colors", []) or []):
                    try:
                        color_int = int(value)
                    except (TypeError, ValueError):
                        continue
                    if color_int not in supported_goal_colors:
                        supported_goal_colors.append(color_int)
        supported_goal_count = max(
            len(supported_goal_anchor_refs),
            1 if supported_goal_colors else 0,
        )
        source_episode_ids = [_episode_id(record) for record in best_supporting_records if _episode_id(record) > 0]
        source_object_ids = _normalize_list([record.get("object_id", "") for record in best_supporting_records])
        evidence_ids: List[str] = []
        for record in best_supporting_records:
            evidence_ids.extend(_normalize_list(record.get("evidence_ids", [])))
        evidence_chain = _normalize_list(source_object_ids + evidence_ids)
        mechanism_family = f"controller_support::{goal_family}::{action_function or 'any_function'}"

        return {
            "object_type": OBJECT_TYPE_REPRESENTATION,
            "memory_type": MemoryType.MECHANISM_SUMMARY.value,
            "memory_layer": MemoryLayer.MECHANISM.value,
            "family": mechanism_family.replace(":", "_"),
            "summary": f"validated mechanism:{goal_family}:{action_function or 'any_function'}",
            "content": {
                "type": MemoryType.MECHANISM_SUMMARY.value,
                "mechanism_kind": str(best_signal.get("mechanism_kind") or "controller_support"),
                "goal_family": goal_family,
                "action_function": action_function,
                "controller_anchor_refs": controller_anchor_refs,
                "supported_goal_anchor_refs": supported_goal_anchor_refs,
                "supported_goal_count": supported_goal_count,
                "supported_goal_colors": supported_goal_colors,
                "preferred_progress_mode": str(
                    best_signal.get("preferred_progress_mode") or "expand_anchor_coverage"
                ),
                "requires_multi_anchor_coordination": bool(
                    best_signal.get("requires_multi_anchor_coordination", False)
                    or supported_goal_count > 1
                ),
                "validation_mode": "repeated_controller_support",
                "support_count": len(best_supporting_records),
                "source_episode_ids": source_episode_ids,
                "source_object_ids": source_object_ids,
            },
            "confidence": min(0.96, 0.5 + 0.08 * len(best_supporting_records)),
            "retrieval_tags": [
                RetrievalTag.CONSOLIDATION.value,
                RetrievalTag.EXPLORATION.value,
                RetrievalTag.EXPLOITATION.value,
                RetrievalTag.AGENT.value,
            ],
            "evidence_ids": evidence_chain,
            "memory_metadata": {
                "promotion_kind": "mechanism",
                "mechanism_kind": str(best_signal.get("mechanism_kind") or "controller_support"),
                "controller_anchor_refs": controller_anchor_refs,
                "supported_goal_anchor_refs": supported_goal_anchor_refs,
                "supported_goal_colors": supported_goal_colors,
                "source_episode_ids": source_episode_ids,
                "source_object_ids": source_object_ids,
                "source_evidence_chain": evidence_chain,
                "validation_count": len(best_supporting_records),
            },
            "trigger_source": "memory_promotion_rules",
            "trigger_episode": _episode_id(episode_record),
        }

    def propose_transfer_from_object(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if infer_object_type(record) != OBJECT_TYPE_SKILL:
            return None

        beneficial = [item for item in record.get("reuse_history", []) or [] if item.get("was_beneficial", False)]
        if len(beneficial) < 2:
            return None

        content = record.get("content", {}) if isinstance(record.get("content"), dict) else {}
        applicability = record.get("applicability", {}) if isinstance(record.get("applicability"), dict) else {}
        families = []
        families.extend(_normalize_list(applicability.get("task_family", [])))
        families.extend(_normalize_list([record.get("source_family"), record.get("target_family")]))
        for item in beneficial:
            families.extend(
                _normalize_list(
                    [
                        item.get("task_family"),
                        item.get("source_family"),
                        item.get("target_family"),
                        item.get("context_family"),
                    ]
                )
            )
        distinct_families = _normalize_list(families)
        if len(distinct_families) < 2:
            return None

        invocation_schema = _callable_invocation(record)
        source_family = str(record.get("source_family") or distinct_families[0])
        target_family = str(record.get("target_family") or distinct_families[-1])
        reuse_evidence = _normalize_list(
            list(record.get("reuse_evidence", []) or [])
            + [item.get("evidence_id") for item in beneficial if item.get("evidence_id")]
            + list(record.get("evidence_ids", []) or [])
        )
        expected_gains = _normalize_list(content.get("expected_gains", [])) or [f"cross_family_beneficial_reuse:{len(beneficial)}"]

        return {
            "object_type": OBJECT_TYPE_TRANSFER,
            "memory_type": MemoryType.GENERIC_OBJECT.value,
            "memory_layer": MemoryLayer.PROCEDURAL.value,
            "family": f"{source_family}_to_{target_family}",
            "summary": f"validated transfer:{source_family}->{target_family}",
            "content": {
                "type": "transfer",
                "source_family": source_family,
                "target_family": target_family,
                "transfer_mechanism": str(record.get("summary") or content.get("summary") or ""),
                "invocation_schema": invocation_schema,
                "expected_gains": expected_gains,
                "source_evidence_chain": reuse_evidence,
            },
            "source_family": source_family,
            "target_family": target_family,
            "reuse_evidence": reuse_evidence,
            "confidence": min(0.97, float(record.get("confidence", 0.5) or 0.5) + 0.04),
            "retrieval_tags": _normalize_list(list(record.get("retrieval_tags", []) or []) + [RetrievalTag.EXPLOITATION.value, RetrievalTag.AGENT.value]),
            "evidence_ids": reuse_evidence,
            "memory_metadata": {
                "promotion_kind": "transfer",
                "source_object_ids": [str(record.get("object_id") or "")],
                "source_evidence_chain": reuse_evidence,
                "family_count": len(distinct_families),
            },
            "trigger_source": "skill_compiler",
            "trigger_episode": int(record.get("trigger_episode", 0) or 0),
        }

    def propose_autobiographical_from_episode(self, episode_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = _episode_content(episode_record)
        retention = content.get("retention_competition_summary", {})
        retention = retention if isinstance(retention, dict) else {}
        discoveries = _episode_discoveries(episode_record)
        failures = _normalize_list(content.get("failures", []))

        continuity_markers: Dict[str, Any] = {}
        dominant_failure = str(retention.get("dominant_retention_failure_type") or "").strip()
        if dominant_failure:
            continuity_markers["dominant_retention_failure_type"] = dominant_failure
        if int(retention.get("required_probe_preserved_count", 0) or 0) > 0:
            continuity_markers["probe_preserved"] = True
        if discoveries:
            continuity_markers["discovery"] = discoveries[0]
        if failures:
            continuity_markers["failure"] = failures[0]
        if not continuity_markers:
            return None

        episode_id = _episode_id(episode_record)
        episode_ref = f"ep-{episode_id}" if episode_id > 0 else ""
        evidence_chain = _normalize_list(
            [str(episode_record.get("object_id") or "")]
            + list(episode_record.get("evidence_ids", []) or [])
        )
        summary_bits = [str(value) for value in continuity_markers.values() if str(value)]
        summary = " | ".join(summary_bits[:2]) if summary_bits else "identity-impacting episode"

        return {
            "object_type": OBJECT_TYPE_AUTOBIOGRAPHICAL,
            "memory_type": MemoryType.GENERIC_OBJECT.value,
            "memory_layer": MemoryLayer.CONTINUITY.value,
            "family": "autobiographical_memory",
            "summary": summary,
            "content": {
                "type": "autobiographical",
                "episode_refs": [episode_ref] if episode_ref else [],
                "continuity_markers": continuity_markers,
                "identity_implications": summary_bits,
                "source_evidence_chain": evidence_chain,
            },
            "episode_refs": [episode_ref] if episode_ref else [],
            "continuity_markers": continuity_markers,
            "confidence": min(0.92, float(episode_record.get("confidence", 0.5) or 0.5) + 0.03),
            "retrieval_tags": [RetrievalTag.CONSOLIDATION.value, RetrievalTag.AGENT.value],
            "evidence_ids": evidence_chain,
            "memory_metadata": {
                "promotion_kind": "autobiographical",
                "source_episode_ids": [episode_id] if episode_id > 0 else [],
                "source_object_ids": [str(episode_record.get("object_id") or "")],
                "source_evidence_chain": evidence_chain,
            },
            "trigger_source": "memory_promotion_rules",
            "trigger_episode": episode_id,
        }
