from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def serialize_arc_candidate_spec(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "name": str(candidate.get("name", "") or ""),
        "kind": str(candidate.get("kind", "") or ""),
        "terminal_only": bool(candidate.get("terminal_only", False)),
    }
    if candidate.get("generalization_bias") is not None:
        spec["generalization_bias"] = int(candidate.get("generalization_bias", 0) or 0)
    if candidate.get("complexity") is not None:
        spec["complexity"] = int(candidate.get("complexity", 0) or 0)
    builder_kind = str(candidate.get("_builder_kind", "") or "")
    if builder_kind:
        spec["_builder_kind"] = builder_kind
    builder_params = candidate.get("_builder_params", {})
    if isinstance(builder_params, dict) and builder_params:
        spec["_builder_params"] = dict(builder_params)
    steps = candidate.get("steps", [])
    if isinstance(steps, list) and steps:
        spec["steps"] = [
            serialize_arc_candidate_spec(step)
            for step in steps
            if isinstance(step, Mapping)
        ]
    return spec


def infer_arc_program_tags(name: str, kind: str = "") -> List[str]:
    haystack = f"{name} {kind}".lower()
    tags: List[str] = []
    if "identity" in haystack:
        tags.append("identity")
    if "constant" in haystack:
        tags.append("constant")
    if any(token in haystack for token in ("crop", "component")):
        tags.append("crop")
    if any(token in haystack for token in ("flip", "rotate", "transpose")):
        tags.append("transform")
    if "component_layout" in haystack or "layout" in haystack:
        tags.append("layout")
        tags.append("component_reorder")
    if "color" in haystack or "remap" in haystack:
        tags.append("color_remap")
    if "local_rule" in haystack:
        tags.append("local_rule")
        tags.append("conditional_rule")
    if "panel_trace" in haystack or "marker" in haystack:
        tags.append("marker_guided")
    if "component" in haystack:
        tags.append("object_centric")
    if "sim:" in haystack or "->" in haystack:
        tags.append("composed")
    deduped: List[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def normalize_arc_program_row(row: Mapping[str, Any], index: int = 0) -> Optional[Dict[str, Any]]:
    name = str(row.get("name", row.get("program_name", "")) or "")
    if not name:
        return None
    normalized: Dict[str, Any] = {
        "program_id": str(row.get("program_id", f"arc_program_{index}") or f"arc_program_{index}"),
        "name": name,
        "kind": str(row.get("kind", row.get("program_kind", "")) or ""),
        "score": float(row.get("score", 0.0) or 0.0),
        "generalization_bias": int(row.get("generalization_bias", 0) or 0),
        "complexity": int(row.get("complexity", row.get("program_complexity", 0)) or 0),
        "terminal_only": bool(row.get("terminal_only", False)),
        "program_tags": list(row.get("program_tags", [])) if isinstance(row.get("program_tags", []), list) else [],
    }
    if not normalized["program_tags"]:
        normalized["program_tags"] = infer_arc_program_tags(normalized["name"], normalized["kind"])
    score_key = row.get("score_key", [])
    if isinstance(score_key, list):
        normalized["score_key"] = list(score_key)
    program_spec = row.get("program_spec", row.get("candidate_spec", {}))
    if isinstance(program_spec, dict) and program_spec:
        normalized["program_spec"] = dict(program_spec)
    return normalized


def normalize_arc_program_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    seen_names = set()
    for index, row in enumerate(rows):
        normalized = normalize_arc_program_row(row, index=index)
        if normalized is None:
            continue
        name = normalized["name"]
        if name in seen_names:
            continue
        normalized_rows.append(normalized)
        seen_names.add(name)
        if limit is not None and limit > 0 and len(normalized_rows) >= limit:
            break
    return normalized_rows
