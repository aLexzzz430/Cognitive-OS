from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from integrations.local_machine.target_binding import latest_failed_run_output, latest_failure_text
from modules.llm.capabilities import GENERAL_REASONING
from modules.llm.gateway import ensure_llm_gateway
from modules.llm.reliability_adapter import LLMReliabilityPolicy, normalize_reliable_llm_output
from modules.llm.thinking_policy import apply_thinking_policy, thinking_policy_for_route


PATCH_PROPOSAL_VERSION = "conos.local_machine.patch_proposal/v1"
PATCH_INTENT_ADAPTER_VERSION = "conos.local_machine.patch_intent_adapter/v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _state(context: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(context.get("investigation_state"))


def _source_root(context: Mapping[str, Any]) -> Path:
    return Path(_text(context.get("source_root")) or ".").resolve()


def _workspace_root(context: Mapping[str, Any]) -> Path:
    return Path(_text(context.get("workspace_root")) or ".").resolve()


def _file_content(context: Mapping[str, Any], relative: str) -> tuple[str, str]:
    workspace = (_workspace_root(context) / relative).resolve()
    source = (_source_root(context) / relative).resolve()
    for label, path in (("workspace", workspace), ("source", source)):
        try:
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8"), label
        except OSError:
            continue
    return "", ""


def _unified_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )


def _changed_line_count(diff: str) -> int:
    count = 0
    for line in diff.splitlines():
        if not line or line.startswith(("+++", "---", "@@")):
            continue
        if line[0] in {"+", "-"}:
            count += 1
    return count


def _diff_targets(diff: str) -> list[str]:
    targets: list[str] = []
    for line in str(diff or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        for prefix in ("a/", "b/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        if raw and raw not in {"/dev/null", "dev/null"} and raw not in targets:
            targets.append(raw)
    return targets


def _diff_old_lines_match(diff: str, before: str) -> bool:
    before_lines = str(before or "").splitlines()
    diff_lines = str(diff or "").splitlines()
    index = 0
    hunk_seen = False
    hunk_old_lines: list[str] = []

    def flush_hunk() -> bool:
        nonlocal index, hunk_old_lines
        if not hunk_old_lines:
            return True
        exact_end = index + len(hunk_old_lines)
        if before_lines[index:exact_end] == hunk_old_lines:
            index = exact_end
            hunk_old_lines = []
            return True
        search_start = max(0, index - 20)
        for candidate in range(search_start, len(before_lines) - len(hunk_old_lines) + 1):
            if before_lines[candidate : candidate + len(hunk_old_lines)] == hunk_old_lines:
                index = candidate + len(hunk_old_lines)
                hunk_old_lines = []
                return True
        hunk_old_lines = []
        return False

    for line in diff_lines:
        if line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith("@@"):
            if hunk_seen and not flush_hunk():
                return False
            hunk_seen = True
            match = re.search(r"@@\s+-(\d+)(?:,\d+)?\s+\+", line)
            if match:
                index = max(0, int(match.group(1)) - 1)
            continue
        if not hunk_seen or not line:
            continue
        marker = line[0]
        text = line[1:]
        if marker == "+":
            continue
        if marker in {" ", "-"}:
            hunk_old_lines.append(text)
    return hunk_seen and flush_hunk()


def _latest_failed_test_target(context: Mapping[str, Any]) -> str:
    payload = latest_failed_run_output(context)
    command = [str(part) for part in _as_list(payload.get("command"))]
    if command:
        target = _text(command[-1])
        if target and target != ".":
            return target.split("::", 1)[0]
    return ""


def _failure_lower(context: Mapping[str, Any]) -> str:
    return latest_failure_text(context).lower()


def _refusal_reason(context: Mapping[str, Any]) -> str:
    failure = _failure_lower(context)
    if any(token in failure for token in ("ambiguous", "conflicting", "contradictory", "underspecified", "unspecified")):
        return "ambiguous_spec"
    return "evidence_insufficient"


def _looks_like_overview_path(path: str) -> bool:
    lowered = str(path or "").lower().strip()
    name = lowered.rsplit("/", 1)[-1]
    return name in {
        "readme",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "package.json",
    }


def _looks_like_test_path(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").strip()
    lowered = normalized.lower()
    name = lowered.rsplit("/", 1)[-1]
    if lowered.startswith(("tests/", "testing/")) or "/tests/" in lowered or "/testing/" in lowered:
        return True
    return name == "conftest.py" or (name.startswith("test_") and name.endswith(".py"))


def _compact_read_files(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _as_list(state.get("read_files")):
        payload = _as_dict(row)
        path = _text(payload.get("path") or payload.get("file"))
        if not path or path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "path": path,
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
            }
        )
    return rows[-12:]


def _compact_validation_runs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(state.get("validation_runs")):
        payload = _as_dict(row)
        run_ref = _text(payload.get("run_ref"))
        if not run_ref:
            continue
        rows.append(
            {
                "run_ref": run_ref,
                "success": bool(payload.get("success", False)),
                "returncode": payload.get("returncode"),
                "target": _text(payload.get("target") or payload.get("test_target")),
            }
        )
    return rows[-8:]


def _proposal(
    *,
    proposal_id: str,
    target_file: str,
    diff: str,
    rationale: str,
    evidence_refs: Sequence[str],
    expected_tests: Sequence[str],
    risk: float,
    source: str,
) -> dict[str, Any]:
    return {
        "schema_version": PATCH_PROPOSAL_VERSION,
        "proposal_id": proposal_id,
        "target_file": target_file,
        "unified_diff": diff,
        "rationale": rationale,
        "evidence_refs": list(evidence_refs),
        "expected_tests": list(dict.fromkeys(str(item) for item in expected_tests if str(item))),
        "risk": float(risk),
        "proposal_source": source,
        "patch_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else "",
    }


def _proposal_id(target_file: str, diff: str) -> str:
    return f"proposal_{hashlib.sha256((target_file + diff).encode('utf-8')).hexdigest()[:12]}"


def _failed_test_refs(context: Mapping[str, Any], target_file: str, line_no: int) -> tuple[list[str], list[str]]:
    failed_test = _latest_failed_test_target(context)
    refs = [f"file:{target_file}:{line_no}"]
    run_ref = _text(latest_failed_run_output(context).get("run_ref"))
    if run_ref:
        refs.insert(0, f"failure:{run_ref}")
    tests = [failed_test, "."] if failed_test else ["."]
    return refs, tests


def _compact_related_context(context: Mapping[str, Any], target_file: str) -> dict[str, Any]:
    state = _state(context)
    binding = _as_dict(state.get("target_binding"))
    hypotheses = _as_list(state.get("hypotheses"))
    read_files = _compact_read_files(state)
    read_file_paths = [str(row.get("path") or "") for row in read_files if str(row.get("path") or "")]
    validation_runs = _compact_validation_runs(state)
    successful_validation_runs = [row for row in validation_runs if bool(row.get("success"))]
    overview_read = [path for path in read_file_paths if _looks_like_overview_path(path)]
    test_files_read = [path for path in read_file_paths if _looks_like_test_path(path)]
    source_files_read = [
        path
        for path in read_file_paths
        if path.endswith(".py") and not _looks_like_test_path(path) and not _looks_like_overview_path(path)
    ]
    leading = {}
    for row in hypotheses:
        payload = _as_dict(row)
        if payload.get("status") == "leading" or str(payload.get("target_file") or "") == target_file:
            leading = payload
            break
    evidence_summary_parts: list[str] = []
    if overview_read:
        evidence_summary_parts.append(f"overview_read={overview_read[0]}")
    if test_files_read:
        evidence_summary_parts.append(f"test_source_read={test_files_read[0]}")
    if source_files_read:
        evidence_summary_parts.append(f"source_files_read={len(source_files_read)}")
    if successful_validation_runs:
        evidence_summary_parts.append(f"successful_validation_runs={len(successful_validation_runs)}")
    elif validation_runs:
        evidence_summary_parts.append(f"failed_validation_runs={len(validation_runs)}")
    return {
        "goal": _text(context.get("goal") or context.get("instruction")),
        "evidence_mode": "test_failure" if latest_failure_text(context) else "open_improvement",
        "latest_failure_excerpt": latest_failure_text(context)[-4000:],
        "latest_failed_test": _latest_failed_test_target(context),
        "read_files": read_files,
        "read_file_paths": read_file_paths,
        "overview_read": overview_read,
        "test_files_read": test_files_read,
        "source_files_read": source_files_read,
        "validation_runs": validation_runs,
        "successful_validation_count": len(successful_validation_runs),
        "latest_successful_validation_ref": _text(successful_validation_runs[-1].get("run_ref")) if successful_validation_runs else "",
        "evidence_summary": "; ".join(evidence_summary_parts),
        "target_binding": binding,
        "leading_hypothesis": {
            "hypothesis_id": str(leading.get("hypothesis_id") or ""),
            "summary": str(leading.get("summary") or "")[:500],
            "target_file": str(leading.get("target_file") or ""),
            "predicted_action_effects": _as_list(leading.get("predicted_action_effects"))[:5],
            "evidence_refs": _as_list(leading.get("evidence_refs"))[:8],
        },
    }


def _parse_llm_patch_response(
    text: str,
    *,
    output_kind: str = "patch_proposal",
    expected_prefixes: Sequence[str] = ("PATCH_JSON:", "PROPOSAL_JSON:"),
) -> dict[str, Any]:
    normalized = normalize_reliable_llm_output(
        text,
        policy=LLMReliabilityPolicy(
            output_kind=output_kind,
            expected_type="dict",
            fallback_on_timeout_allowed=False,
        ),
        expected_prefixes=expected_prefixes,
    )
    return normalized.parsed_dict() if normalized.ok else {}


def _snippet_text(value: Any) -> str:
    if isinstance(value, list):
        raw = "\n".join(str(item) for item in value)
    else:
        raw = str(value or "")
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
        elif lines:
            raw = "\n".join(lines[1:]).strip()
    # Common JSON/markdown artifact from small models: a closing object brace leaks into a one-line edit.
    if raw.rstrip().endswith("}") and raw.count("{") < raw.count("}"):
        raw = raw.rstrip()[:-1].rstrip()
    return raw


def _intent_from_mapping(parsed: Mapping[str, Any]) -> tuple[str, str, str]:
    payload = dict(parsed or {})
    edits = _as_list(payload.get("edits"))
    if edits:
        payload = _as_dict(edits[0])
    pairs = (
        ("old_snippet", "new_snippet"),
        ("old", "new"),
        ("search", "replace"),
        ("find", "replace_with"),
    )
    for old_key, new_key in pairs:
        old = _snippet_text(payload.get(old_key))
        new = _snippet_text(payload.get(new_key))
        if old and new:
            return old, new, f"{old_key}/{new_key}"
    return "", "", ""


def _intent_from_diff_fragment(text: str) -> tuple[str, str, str]:
    old_lines: list[str] = []
    new_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith(("---", "+++", "@@")) or not line:
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
    old = _snippet_text(old_lines)
    new = _snippet_text(new_lines)
    if old and new:
        return old, new, "diff_fragment"
    return "", "", ""


def _replace_unique_snippet(content: str, old: str, new: str) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {"exact_match_count": 0, "line_match_count": 0}
    exact_count = content.count(old)
    meta["exact_match_count"] = exact_count
    if exact_count == 1:
        return content.replace(old, new, 1), {**meta, "match_strategy": "exact_snippet"}
    if exact_count > 1:
        return "", {**meta, "error": "old_snippet_not_unique"}

    old_stripped = old.strip()
    if not old_stripped or "\n" in old_stripped:
        return "", {**meta, "error": "old_snippet_not_found"}
    lines = content.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.strip() == old_stripped]
    meta["line_match_count"] = len(matches)
    if len(matches) != 1:
        return "", {**meta, "error": "old_snippet_not_unique" if len(matches) > 1 else "old_snippet_not_found"}
    index = matches[0]
    original = lines[index]
    indentation = original[: len(original) - len(original.lstrip())]
    newline = "\n" if original.endswith("\n") else ""
    replacement = new
    if "\n" not in replacement and not replacement.startswith(indentation):
        replacement = f"{indentation}{replacement.lstrip()}"
    lines[index] = f"{replacement.rstrip()}{newline}"
    return "".join(lines), {**meta, "match_strategy": "single_line_stripped"}


def _compile_patch_intent_to_diff(
    parsed: Mapping[str, Any],
    *,
    raw_response: str,
    target_file: str,
    content: str,
) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {
        "schema_version": PATCH_INTENT_ADAPTER_VERSION,
        "attempted": False,
        "status": "not_needed",
        "source": "",
        "reason": "",
    }
    diff = _text(_as_dict(parsed).get("unified_diff"))
    if diff:
        targets = _diff_targets(diff)
        if targets == [target_file]:
            if _diff_old_lines_match(diff, content):
                return diff, {**meta, "status": "unified_diff_valid", "source": "unified_diff"}
            meta = {
                **meta,
                "attempted": True,
                "status": "unified_diff_context_mismatch",
                "source": "unified_diff",
                "reason": "falling_back_to_diff_fragment_intent",
            }
        if targets and targets != [target_file]:
            return "", {
                **meta,
                "attempted": True,
                "status": "rejected",
                "source": "unified_diff",
                "reason": "diff_target_out_of_bounds",
                "targets": targets,
            }

    old, new, source = _intent_from_mapping(parsed)
    if not old or not new:
        old, new, source = _intent_from_diff_fragment(diff or raw_response)
    if not old or not new:
        return "", {
            **meta,
            "attempted": True,
            "status": "rejected",
            "source": source or "unparsed",
            "reason": "patch_intent_uncompilable",
        }
    updated, replace_meta = _replace_unique_snippet(content, old, new)
    if not updated:
        return "", {
            **meta,
            "attempted": True,
            "status": "rejected",
            "source": source,
            "reason": str(replace_meta.get("error") or "patch_intent_uncompilable"),
            **replace_meta,
        }
    compiled = _unified_diff(target_file, content, updated)
    if not compiled:
        return "", {
            **meta,
            "attempted": True,
            "status": "rejected",
            "source": source,
            "reason": "compiled_diff_empty",
            **replace_meta,
        }
    return compiled, {
        **meta,
        "attempted": True,
        "status": "compiled",
        "source": source,
        "reason": "compiled_patch_intent",
        "old_snippet_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
        "new_snippet_sha256": hashlib.sha256(new.encode("utf-8")).hexdigest(),
        **replace_meta,
    }


def _extract_thinking_and_visible_text(text: str) -> tuple[str, str]:
    raw = str(text or "")
    thinking = "\n".join(
        match.group(1).strip()
        for match in re.finditer(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
        if match.group(1).strip()
    )
    visible = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).replace("</think>", "").strip()
    if not thinking and not visible:
        return "", ""
    return thinking, visible


def _request_kwargs_trace(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "think": kwargs.get("think"),
        "thinking_budget": kwargs.get("thinking_budget"),
        "timeout_sec": kwargs.get("timeout_sec"),
        "max_tokens": kwargs.get("max_tokens"),
    }


def _is_timeout_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return "timeout" in lowered or "timed out" in lowered


def _llm_failure_reason_from_error(text: str, *, default: str) -> str:
    lowered = str(text or "").lower()
    if _is_timeout_error(lowered):
        return "timeout"
    if "llm_budget_exceeded" in lowered or "budget_exceeded" in lowered:
        return "llm_budget_exceeded"
    return default


def _normalize_reasoning_state(parsed: Mapping[str, Any], *, target_file: str, related: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(parsed or {})
    state = dict(payload.get("reasoning_state", {}) or {}) if isinstance(payload.get("reasoning_state"), Mapping) else payload
    evidence = _as_list(state.get("evidence")) or [_text(related.get("latest_failure_excerpt"))[-800:]]
    next_action = state.get("next_action") if isinstance(state.get("next_action"), Mapping) else {}
    if not next_action:
        next_action = {"action": "propose_bounded_diff", "target_file": target_file}
    confidence = state.get("confidence", 0.5)
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = 0.5
    return {
        "schema_version": "conos.reasoning_state/v1",
        "evidence": [str(item)[:500] for item in evidence if str(item).strip()][:6],
        "hypothesis": _as_dict(state.get("hypothesis")) or {
            "summary": _text(_as_dict(related.get("leading_hypothesis")).get("summary")),
            "target_file": target_file,
        },
        "decision": _text(state.get("decision")) or "patch",
        "next_action": dict(next_action),
        "confidence": confidence_value,
        "failure_boundary": [str(item)[:240] for item in _as_list(state.get("failure_boundary")) if str(item).strip()][:6],
        "patch_intent": _text(state.get("patch_intent") or state.get("rationale"))[:500],
    }


def _looks_like_reasoning_state(parsed: Mapping[str, Any]) -> bool:
    payload = dict(parsed or {})
    if isinstance(payload.get("reasoning_state"), Mapping):
        return True
    return any(key in payload for key in ("evidence", "hypothesis", "decision", "next_action", "confidence", "patch_intent"))


def _loose_json_string_field(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', str(text or ""), flags=re.DOTALL)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _loose_json_number_field(text: str, key: str) -> float | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _loose_reasoning_state_from_text(text: str, *, target_file: str) -> dict[str, Any]:
    raw = str(text or "")
    if "reasoning_state" not in raw:
        return {}
    lower = raw.lower()
    target = _loose_json_string_field(raw, "target_file") or _loose_json_string_field(raw, "file") or target_file
    summary = (
        _loose_json_string_field(raw, "summary")
        or _loose_json_string_field(raw, "detail")
        or "loose parsed reasoning state from malformed distill JSON"
    )
    change = _loose_json_string_field(raw, "change")
    patch_intent = _loose_json_string_field(raw, "patch_intent") or change or _loose_json_string_field(raw, "action")
    human_review_true = bool(re.search(r'"need_human_review"\s*:\s*true|"human_review_required"\s*:\s*true', lower))
    human_review_false = bool(re.search(r'"need_human_review"\s*:\s*false|"human_review_required"\s*:\s*false', lower))
    patch_safe = '"patch_safe":true' in lower or '"type":"proceed"' in lower or '"type": "proceed"' in lower
    decision = "patch" if (patch_safe or human_review_false or patch_intent) and not human_review_true else "requires_further_evidence"
    confidence = _loose_json_number_field(raw, "confidence")
    if confidence is None:
        confidence = 0.5
    return {
        "reasoning_state": {
            "evidence": [
                "distill output was malformed but contained auditable evidence/hypothesis/action fields",
                _loose_json_string_field(raw, "goal")[:300],
            ],
            "hypothesis": {
                "summary": summary,
                "target_file": target,
            },
            "decision": decision,
            "next_action": {"action": "propose_bounded_diff", "target_file": target},
            "confidence": confidence,
            "failure_boundary": [_loose_json_string_field(raw, "reject_if")[:240]],
            "patch_intent": patch_intent,
        }
    }


def _reasoning_refusal_reason(reasoning_state: Mapping[str, Any]) -> str:
    decision = _text(reasoning_state.get("decision")).lower()
    patch_intent = _text(reasoning_state.get("patch_intent")).lower()
    next_action = _as_dict(reasoning_state.get("next_action"))
    next_action_text = " ".join(str(value).lower() for value in next_action.values())
    boundary_text = " ".join(str(item).lower() for item in _as_list(reasoning_state.get("failure_boundary")))
    joined = " ".join([decision, patch_intent, next_action_text, boundary_text])
    if any(token in joined for token in ("ambiguous", "conflicting", "contradictory", "underspecified", "unspecified", "human review")):
        return "ambiguous_spec"
    if any(
        token in joined
        for token in (
            "requires_further_evidence",
            "further evidence",
            "needs_evidence",
            "insufficient_evidence",
            "evidence insufficient",
            "more evidence",
            "needs_human_review",
            "refuse",
            "refusal",
        )
    ):
        return "evidence_insufficient"
    return ""


def _llm_bounded_diff_proposals(
    context: Mapping[str, Any],
    *,
    target_file: str,
    content: str,
    llm_client: Any,
    max_changed_lines: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    think_gateway = ensure_llm_gateway(llm_client, route_name="patch_proposal", capability_prefix="general")
    act_gateway = ensure_llm_gateway(llm_client, route_name="patch_proposal", capability_prefix="structured_output")
    related = _compact_related_context(context, target_file)
    think_prompt = (
        "Think about the likely patch target and failure mechanism.\n"
        "Do not produce a final patch. Do not output a final answer. "
        "Focus on evidence, hypothesis, decision boundary, and what a later no-thinking action should do.\n\n"
        f"Context:\n{json.dumps(related, ensure_ascii=False, default=str)}\n\n"
        f"Target file {target_file}:\n{content[:9000]}\n"
    )
    think_system_prompt = (
        "You are the Think Pass of a local-first runtime. Reason briefly. "
        "Do not emit a final patch or final JSON answer."
    )
    if think_gateway is None or act_gateway is None:
        return [], [{
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "patch_proposal",
            "stage": "gateway",
            "prompt": "",
            "system_prompt": "",
            "response": "",
            "parsed_kwargs": {},
            "error": "llm_gateway_unavailable",
        }]
    thinking_mode = _text(context.get("llm_thinking_mode")) or "auto"
    think_kwargs = apply_thinking_policy(
        "patch_proposal",
        {
            "max_tokens": 512,
            "temperature": 0.0,
            "system_prompt": think_system_prompt,
        },
        mode=thinking_mode,
    )
    if isinstance(think_kwargs.get("thinking_budget"), int):
        think_kwargs["thinking_budget"] = min(512, int(think_kwargs["thinking_budget"]))
    policy = thinking_policy_for_route("patch_proposal", mode=thinking_mode)
    traces: list[dict[str, Any]] = []
    try:
        think_response = think_gateway.request_raw(
            GENERAL_REASONING,
            think_prompt,
            **think_kwargs,
        )
        gateway_error = str(getattr(think_gateway, "last_error", "") or "")
        if gateway_error:
            failure_reason = _llm_failure_reason_from_error(gateway_error, default="llm_request_failed")
            return [], [{
                "function_name": "propose_patch",
                "capability": str(GENERAL_REASONING),
                "route_name": "patch_proposal",
                "stage": "think_pass",
                "prompt": think_prompt,
                "system_prompt": think_system_prompt,
                "response": "",
                "parsed_kwargs": {},
                "thinking_policy": policy.to_dict(),
                "request_kwargs": _request_kwargs_trace(think_kwargs),
                "error": gateway_error,
                "failure_reason": failure_reason,
                "llm_timeout": failure_reason == "timeout",
            }]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure_reason = _llm_failure_reason_from_error(error, default="llm_request_failed")
        return [], [{
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "patch_proposal",
            "stage": "think_pass",
            "prompt": think_prompt,
            "system_prompt": think_system_prompt,
            "response": "",
            "parsed_kwargs": {},
            "thinking_policy": policy.to_dict(),
            "request_kwargs": _request_kwargs_trace(think_kwargs),
            "error": error,
            "failure_reason": failure_reason,
            "llm_timeout": failure_reason == "timeout",
        }]
    thinking_text, visible_think_text = _extract_thinking_and_visible_text(think_response)
    raw_think_material = (thinking_text or visible_think_text or str(think_response or ""))[:4000]
    traces.append({
        "function_name": "propose_patch",
        "capability": str(GENERAL_REASONING),
        "route_name": "patch_proposal",
        "stage": "think_pass",
        "prompt": think_prompt,
        "system_prompt": think_system_prompt,
        "response": "",
        "parsed_kwargs": {},
        "raw_response_discarded": True,
        "thinking_chars": len(thinking_text),
        "visible_think_chars": len(visible_think_text),
        "thinking_policy": policy.to_dict(),
        "request_kwargs": _request_kwargs_trace(think_kwargs),
        "error": "",
    })
    distill_prompt = (
        "Distill the raw thinking and evidence into compact executable reasoning_state JSON.\n"
        "Discard private wording. Preserve only auditable state: evidence, hypothesis, decision, "
        "next_action, confidence, failure_boundary, patch_intent.\n"
        "Return exactly one line: REASONING_STATE_JSON: {\"reasoning_state\": {...}}\n\n"
        f"Evidence context:\n{json.dumps(related, ensure_ascii=False, default=str)}\n\n"
        f"Raw think material to distill, not to preserve verbatim:\n{raw_think_material}\n"
    )
    distill_trace_prompt = "DISTILL_PASS_PROMPT_REDACTED_RAW_THINKING_DISCARDED"
    distill_system_prompt = "You are the Distill Pass. Return compact JSON only. No markdown."
    distill_kwargs = apply_thinking_policy(
        "structured_answer",
        {
            "max_tokens": 384,
            "temperature": 0.0,
            "system_prompt": distill_system_prompt,
        },
        mode="off",
    )
    distill_kwargs["timeout_sec"] = max(float(distill_kwargs.get("timeout_sec", 0.0) or 0.0), 60.0)
    try:
        distill_response = act_gateway.request_raw(
            GENERAL_REASONING,
            distill_prompt,
            **distill_kwargs,
        )
        distill_error = str(getattr(act_gateway, "last_error", "") or "")
        if distill_error:
            failure_reason = _llm_failure_reason_from_error(distill_error, default="distill_failed")
            traces.append({
                "function_name": "propose_patch",
                "capability": str(GENERAL_REASONING),
                "route_name": "patch_proposal",
                "stage": "distill_pass",
                "prompt": distill_trace_prompt,
                "system_prompt": distill_system_prompt,
                "response": "",
                "parsed_kwargs": {},
                "request_kwargs": _request_kwargs_trace(distill_kwargs),
                "error": distill_error,
                "failure_reason": failure_reason,
                "llm_timeout": failure_reason == "timeout",
            })
            return [], traces
        distill_normalized = normalize_reliable_llm_output(
            distill_response,
            policy=LLMReliabilityPolicy(
                output_kind="reasoning_state",
                expected_type="dict",
                required_fields=("reasoning_state",),
                fallback_on_timeout_allowed=False,
            ),
            expected_prefixes=("REASONING_STATE_JSON:",),
        )
        parsed_state = distill_normalized.parsed_dict()
        loose_parse_used = False
        if not parsed_state or not _looks_like_reasoning_state(parsed_state):
            parsed_state = _loose_reasoning_state_from_text(str(distill_response or ""), target_file=target_file)
            loose_parse_used = bool(parsed_state)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure_reason = _llm_failure_reason_from_error(error, default="distill_failed")
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "patch_proposal",
            "stage": "distill_pass",
            "prompt": distill_trace_prompt,
            "system_prompt": distill_system_prompt,
            "response": "",
            "parsed_kwargs": {},
            "request_kwargs": _request_kwargs_trace(distill_kwargs),
            "error": error,
            "failure_reason": failure_reason,
            "llm_timeout": failure_reason == "timeout",
        })
        return [], traces
    if "loose_parse_used" not in locals():
        loose_parse_used = False
    parsed_state_looks_valid = _looks_like_reasoning_state(parsed_state) if parsed_state else False
    if (not distill_normalized.ok and not parsed_state_looks_valid) or not parsed_state:
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "patch_proposal",
            "stage": "distill_pass",
            "prompt": distill_trace_prompt,
            "system_prompt": distill_system_prompt,
            "response": str(distill_response or ""),
            "parsed_kwargs": {},
            "request_kwargs": _request_kwargs_trace(distill_kwargs),
            "output_adapter": distill_normalized.output_adapter,
            "reliability_adapter": distill_normalized.to_trace(),
            "error": distill_normalized.error or "missing_reasoning_state",
        })
        return [], traces
    reasoning_state = _normalize_reasoning_state(parsed_state, target_file=target_file, related=related)
    traces.append({
        "function_name": "propose_patch",
        "capability": str(GENERAL_REASONING),
        "route_name": "patch_proposal",
        "stage": "distill_pass",
        "prompt": distill_trace_prompt,
        "system_prompt": distill_system_prompt,
        "response": json.dumps({"reasoning_state": reasoning_state}, ensure_ascii=False),
        "parsed_kwargs": {"reasoning_state": reasoning_state},
        "request_kwargs": _request_kwargs_trace(distill_kwargs),
        "output_adapter": distill_normalized.output_adapter,
        "reliability_adapter": distill_normalized.to_trace(),
        "raw_thinking_discarded": True,
        "distill_acceptance_override": (
            "loose_reasoning_state"
            if loose_parse_used
            else ("parsed_reasoning_state" if not distill_normalized.ok and parsed_state_looks_valid else "")
        ),
        "error": "",
    })
    reasoning_refusal = _reasoning_refusal_reason(reasoning_state)
    if reasoning_refusal:
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "patch_proposal",
            "stage": "reasoning_gate",
            "prompt": "",
            "system_prompt": "",
            "response": "",
            "parsed_kwargs": {"reasoning_state": reasoning_state},
            "reasoning_state": reasoning_state,
            "error": "reasoning_state_declined_patch",
            "failure_reason": reasoning_refusal,
            "needs_human_review": True,
            "raw_thinking_discarded": True,
        })
        return [], traces
    act_prompt = (
        "Generate one minimal unified diff for the target file only.\n"
        "Do not think. Do not modify tests. Use only reasoning_state, evidence context, and target content. "
        f"Touch only {target_file}. Change at most {int(max_changed_lines)} lines. "
        "Return exactly one line: PATCH_JSON: {\"unified_diff\": \"...\", "
        "\"rationale\": \"...\", \"expected_tests\": [\"...\"], \"risk\": 0.0}\n\n"
        f"Reasoning state:\n{json.dumps(reasoning_state, ensure_ascii=False, default=str)}\n\n"
        f"Evidence context:\n{json.dumps(related, ensure_ascii=False, default=str)}\n\n"
        f"Target file {target_file}:\n{content[:9000]}\n"
    )
    act_system_prompt = (
        "You are the Act Pass. No private thinking. Return only PATCH_JSON with a standard unified diff."
    )
    act_kwargs = apply_thinking_policy(
        "structured_answer",
        {
            "max_tokens": 900,
            "temperature": 0.0,
            "system_prompt": act_system_prompt,
        },
        mode="off",
    )
    act_kwargs["timeout_sec"] = max(float(act_kwargs.get("timeout_sec", 0.0) or 0.0), 60.0)
    try:
        act_response = act_gateway.request_raw(
            GENERAL_REASONING,
            act_prompt,
            **act_kwargs,
        )
        act_error = str(getattr(act_gateway, "last_error", "") or "")
        if act_error:
            failure_reason = _llm_failure_reason_from_error(act_error, default="act_failed")
            traces.append({
                "function_name": "propose_patch",
                "capability": str(GENERAL_REASONING),
                "route_name": "patch_proposal",
                "stage": "act_pass",
                "prompt": act_prompt,
                "system_prompt": act_system_prompt,
                "response": "",
                "parsed_kwargs": {},
                "request_kwargs": _request_kwargs_trace(act_kwargs),
                "reasoning_state": reasoning_state,
                "error": act_error,
                "failure_reason": failure_reason,
                "llm_timeout": failure_reason == "timeout",
            })
            return [], traces
        act_normalized = normalize_reliable_llm_output(
            act_response,
            policy=LLMReliabilityPolicy(
                output_kind="patch_proposal",
                expected_type="dict",
                fallback_on_timeout_allowed=False,
            ),
            expected_prefixes=("PATCH_JSON:", "PROPOSAL_JSON:"),
        )
        parsed = act_normalized.parsed_dict()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure_reason = _llm_failure_reason_from_error(error, default="act_failed")
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "patch_proposal",
            "stage": "act_pass",
            "prompt": act_prompt,
            "system_prompt": act_system_prompt,
            "response": "",
            "parsed_kwargs": {},
            "request_kwargs": _request_kwargs_trace(act_kwargs),
            "reasoning_state": reasoning_state,
            "error": error,
            "failure_reason": failure_reason,
            "llm_timeout": failure_reason == "timeout",
        })
        return [], traces
    trace = {
        "function_name": "propose_patch",
        "capability": str(GENERAL_REASONING),
        "route_name": "patch_proposal",
        "stage": "act_pass",
        "prompt": act_prompt,
        "system_prompt": act_system_prompt,
        "response": str(act_response or ""),
        "parsed_kwargs": dict(parsed),
        "reasoning_state": reasoning_state,
        "request_kwargs": _request_kwargs_trace(act_kwargs),
        "output_adapter": act_normalized.output_adapter,
        "reliability_adapter": act_normalized.to_trace(),
        "error": "",
    }
    if not act_normalized.ok:
        trace["error"] = act_normalized.error or act_normalized.status or "patch_proposal_not_accepted"
        traces.append(trace)
        return [], traces
    diff, adapter_meta = _compile_patch_intent_to_diff(
        parsed,
        raw_response=str(act_response or ""),
        target_file=target_file,
        content=content,
    )
    trace["patch_intent_adapter"] = adapter_meta
    if not diff:
        trace["error"] = _text(adapter_meta.get("reason")) or "missing_unified_diff"
        traces.append(trace)
        return [], traces
    targets = _diff_targets(diff)
    if targets != [target_file]:
        trace["error"] = "diff_target_out_of_bounds"
        traces.append(trace)
        return [], traces
    if target_file.startswith("tests/") or "/tests/" in target_file:
        trace["error"] = "test_modification_not_allowed"
        traces.append(trace)
        return [], traces
    if _changed_line_count(diff) > max(1, int(max_changed_lines)):
        trace["error"] = "diff_too_large"
        traces.append(trace)
        return [], traces
    if not _diff_old_lines_match(diff, content):
        trace["error"] = "diff_does_not_match_current_content"
        traces.append(trace)
        return [], traces
    proposal = _proposal(
        proposal_id=_proposal_id(target_file, diff),
        target_file=target_file,
        diff=diff,
        rationale=_text(parsed.get("rationale")) or "bounded LLM diff proposal from failure evidence",
        evidence_refs=[f"failure:{_text(latest_failed_run_output(context).get('run_ref'))}", f"file:{target_file}"],
        expected_tests=_as_list(parsed.get("expected_tests")) or [_latest_failed_test_target(context), "."],
        risk=float(parsed.get("risk", 0.45) or 0.45),
        source="bounded_llm_diff" if str(adapter_meta.get("status") or "") == "unified_diff_valid" else "bounded_llm_intent_diff",
    )
    proposal["reasoning_state"] = reasoning_state
    proposal["pipeline"] = "think_distill_act"
    proposal["patch_intent_adapter"] = adapter_meta
    traces.append(trace)
    return [proposal], traces


def _lossy_attribute_proposals(context: Mapping[str, Any], target_file: str, content: str) -> list[dict[str, Any]]:
    failure = latest_failure_text(context)
    if not failure:
        return []
    proposals: list[dict[str, Any]] = []
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if ".name" not in line or "/" not in line:
            continue
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\.name\b", line)
        if not match:
            continue
        variable = match.group(1)
        updated_lines = list(lines)
        updated_lines[index] = line.replace(f"{variable}.name", variable, 1)
        updated = "".join(updated_lines)
        diff = _unified_diff(target_file, content, updated)
        if not diff or _changed_line_count(diff) > 20:
            continue
        evidence_refs, expected_tests = _failed_test_refs(context, target_file, index + 1)
        proposals.append(
            _proposal(
                proposal_id=_proposal_id(target_file, diff),
                target_file=target_file,
                diff=diff,
                rationale=(
                    "failure evidence shows a value-losing transformation; replace a lossy attribute access "
                    "with the original path-like object and let verifier tests decide"
                ),
                evidence_refs=evidence_refs,
                expected_tests=expected_tests,
                risk=0.28,
                source="conservative_textual_diff",
            )
        )
    return proposals


def _function_param_stack(line: str, active_params: list[str]) -> list[str]:
    match = re.match(r"\s*def\s+[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)", line)
    if not match:
        return active_params
    params: list[str] = []
    for raw in match.group(1).split(","):
        name = raw.strip().split(":", 1)[0].split("=", 1)[0].strip()
        if name and name not in {"self", "cls"} and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            params.append(name)
    return params


def _key_reference_consistency_proposals(context: Mapping[str, Any], target_file: str, content: str) -> list[dict[str, Any]]:
    failure = _failure_lower(context)
    if not any(token in failure for token in ("key", "group", "bucket", "aggregate", "aggregation", "keyerror")):
        return []
    proposals: list[dict[str, Any]] = []
    active_params: list[str] = []
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        active_params = _function_param_stack(line, active_params)
        candidate_params = [
            param
            for param in active_params
            if param == "key" or param.endswith("_key") or param.endswith("_field") or param in {"field", "group_field"}
        ]
        if not candidate_params:
            continue
        match = re.search(r"(\[[\"'][A-Za-z_][A-Za-z0-9_-]*[\"']\])", line)
        if not match:
            continue
        replacement = f"[{candidate_params[0]}]"
        updated_lines = list(lines)
        updated_lines[index] = line[: match.start(1)] + replacement + line[match.end(1) :]
        updated = "".join(updated_lines)
        diff = _unified_diff(target_file, content, updated)
        if not diff or _changed_line_count(diff) > 20:
            continue
        evidence_refs, expected_tests = _failed_test_refs(context, target_file, index + 1)
        proposals.append(
            _proposal(
                proposal_id=_proposal_id(target_file, diff),
                target_file=target_file,
                diff=diff,
                rationale=(
                    "failure evidence points at grouping/key selection; replace a fixed mapping key with "
                    "the function's key parameter and let verifier tests decide"
                ),
                evidence_refs=evidence_refs,
                expected_tests=expected_tests,
                risk=0.32,
                source="key_reference_consistency",
            )
        )
    return proposals


def _attribute_container_kinds(content: str) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for match in re.finditer(r"self\.(?P<attr>_[A-Za-z0-9_]+)\s*=\s*(?P<expr>[^\n]+)", content):
        attr = match.group("attr")
        expr = match.group("expr")
        if re.search(r"\blist\s*\(", expr) or "[]" in expr or attr.endswith(("s", "items", "rows", "values")):
            kinds[attr] = "list"
        elif re.search(r"\bdict\s*\(", expr) or "{}" in expr or attr.endswith(("map", "dict", "by_key")):
            kinds[attr] = "dict"
        elif re.search(r"\bset\s*\(", expr) or attr.endswith(("set", "ids")):
            kinds[attr] = "set"
    return kinds


def _defensive_copy_proposals(context: Mapping[str, Any], target_file: str, content: str) -> list[dict[str, Any]]:
    failure = _failure_lower(context)
    if not any(token in failure for token in ("mutat", "alias", "copy", "leak", "external", "internal state", "defensive")):
        return []
    kinds = _attribute_container_kinds(content)
    if not kinds:
        return []
    proposals: list[dict[str, Any]] = []
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = re.match(r"(?P<indent>\s*)return\s+self\.(?P<attr>_[A-Za-z0-9_]+)\s*$", line.rstrip("\n"))
        if not match:
            continue
        attr = match.group("attr")
        kind = kinds.get(attr)
        if kind not in {"list", "dict", "set"}:
            continue
        updated_lines = list(lines)
        newline = "\n" if line.endswith("\n") else ""
        updated_lines[index] = f"{match.group('indent')}return {kind}(self.{attr}){newline}"
        updated = "".join(updated_lines)
        diff = _unified_diff(target_file, content, updated)
        if not diff or _changed_line_count(diff) > 20:
            continue
        evidence_refs, expected_tests = _failed_test_refs(context, target_file, index + 1)
        proposals.append(
            _proposal(
                proposal_id=_proposal_id(target_file, diff),
                target_file=target_file,
                diff=diff,
                rationale=(
                    "failure evidence points at mutable state exposure; return a shallow defensive copy "
                    "of the stored container and let verifier tests decide"
                ),
                evidence_refs=evidence_refs,
                expected_tests=expected_tests,
                risk=0.3,
                source="defensive_copy",
            )
        )
    return proposals


def _minimal_operator_correction_proposals(context: Mapping[str, Any], target_file: str, content: str) -> list[dict[str, Any]]:
    failure = _failure_lower(context)
    if not any(token in failure for token in ("boundary", "threshold", "inclusive", "exclusive", "exact", "equal")):
        return []
    proposals: list[dict[str, Any]] = []
    lines = content.splitlines(keepends=True)
    replacements = ((r"(?<![<>=!])>(?!=)", ">=", "inclusive lower boundary"), (r"(?<![<>=!])<(?!=)", "<=", "inclusive upper boundary"))
    for index, line in enumerate(lines):
        if not re.match(r"\s*if\s+", line):
            continue
        for pattern, replacement, rationale_suffix in replacements:
            if not re.search(pattern, line):
                continue
            updated_lines = list(lines)
            updated_lines[index] = re.sub(pattern, replacement, line, count=1)
            updated = "".join(updated_lines)
            diff = _unified_diff(target_file, content, updated)
            if not diff or _changed_line_count(diff) > 20:
                continue
            evidence_refs, expected_tests = _failed_test_refs(context, target_file, index + 1)
            proposals.append(
                _proposal(
                    proposal_id=_proposal_id(target_file, diff),
                    target_file=target_file,
                    diff=diff,
                    rationale=f"failure evidence suggests a boundary condition mismatch; try {rationale_suffix}",
                    evidence_refs=evidence_refs,
                    expected_tests=expected_tests,
                    risk=0.34,
                    source="minimal_operator_correction",
                )
            )
    return proposals


def generate_patch_proposals(
    context: Mapping[str, Any],
    *,
    top_target_file: str,
    max_changed_lines: int = 20,
    llm_client: Any = None,
    prefer_llm: bool = False,
    allow_fallback_patch: bool = False,
) -> dict[str, Any]:
    target_file = _text(top_target_file)
    if not target_file or target_file.startswith("tests/") or "/tests/" in target_file:
        return {
            "schema_version": PATCH_PROPOSAL_VERSION,
            "patch_proposals": [],
            "rejection_reason": "target file is empty or points at tests",
        }
    content, source = _file_content(context, target_file)
    if not content:
        return {
            "schema_version": PATCH_PROPOSAL_VERSION,
            "patch_proposals": [],
            "refusal_reason": "evidence_insufficient",
            "rejection_reason": "target file content is unavailable",
        }
    proposals = []
    llm_trace: list[dict[str, Any]] = []
    llm_proposals: list[dict[str, Any]] = []
    if llm_client is not None:
        llm_proposals, llm_trace = _llm_bounded_diff_proposals(
            context,
            target_file=target_file,
            content=content,
            llm_client=llm_client,
            max_changed_lines=max_changed_lines,
        )
    llm_timeout = any(bool(_as_dict(row).get("llm_timeout")) for row in llm_trace)
    llm_failure_reason = ""
    for row in llm_trace:
        reason = _text(_as_dict(row).get("failure_reason"))
        if reason:
            llm_failure_reason = reason
            break
    fallback_patch_enabled = llm_client is None or bool(allow_fallback_patch)
    deterministic_proposals: list[dict[str, Any]] = []
    if fallback_patch_enabled:
        deterministic_proposals.extend(_lossy_attribute_proposals(context, target_file, content))
        deterministic_proposals.extend(_key_reference_consistency_proposals(context, target_file, content))
        deterministic_proposals.extend(_defensive_copy_proposals(context, target_file, content))
        deterministic_proposals.extend(_minimal_operator_correction_proposals(context, target_file, content))
    if prefer_llm:
        proposals = [*llm_proposals, *deterministic_proposals]
    else:
        proposals = [*deterministic_proposals, *llm_proposals]
    bounded = [
        proposal
        for proposal in proposals
        if _changed_line_count(str(proposal.get("unified_diff") or "")) <= max(1, int(max_changed_lines))
    ]
    if bounded:
        refusal_reason = ""
    elif llm_timeout:
        refusal_reason = "timeout"
    elif llm_failure_reason in {"evidence_insufficient", "ambiguous_spec", "llm_budget_exceeded"}:
        refusal_reason = llm_failure_reason
    elif llm_client is not None and not fallback_patch_enabled:
        refusal_reason = "llm_patch_proposal_unavailable"
    else:
        refusal_reason = _refusal_reason(context)
    return {
        "schema_version": PATCH_PROPOSAL_VERSION,
        "patch_proposals": bounded[:3],
        "proposal_count": len(bounded[:3]),
        "target_file": target_file,
        "target_content_source": source,
        "llm_trace": llm_trace,
        "llm_proposal_count": len(llm_proposals),
        "llm_timeout": bool(llm_timeout),
        "fallback_patch_enabled": bool(fallback_patch_enabled),
        "deterministic_fallback_proposal_count": len(deterministic_proposals),
        "needs_human_review": bool(not bounded),
        "refusal_reason": refusal_reason,
        "rejection_reason": refusal_reason if not bounded else "",
    }
