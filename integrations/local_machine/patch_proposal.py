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
from modules.llm.thinking_policy import apply_thinking_policy, thinking_policy_for_route


PATCH_PROPOSAL_VERSION = "conos.local_machine.patch_proposal/v1"


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
        for candidate in range(index, len(before_lines) - len(hunk_old_lines) + 1):
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
    return (
        [f"failure:{_text(latest_failed_run_output(context).get('run_ref'))}", f"file:{target_file}:{line_no}"],
        [failed_test, "."],
    )


def _compact_related_context(context: Mapping[str, Any], target_file: str) -> dict[str, Any]:
    state = _state(context)
    binding = _as_dict(state.get("target_binding"))
    hypotheses = _as_list(state.get("hypotheses"))
    leading = {}
    for row in hypotheses:
        payload = _as_dict(row)
        if payload.get("status") == "leading" or str(payload.get("target_file") or "") == target_file:
            leading = payload
            break
    return {
        "goal": _text(context.get("goal") or context.get("instruction")),
        "latest_failure_excerpt": latest_failure_text(context)[-4000:],
        "latest_failed_test": _latest_failed_test_target(context),
        "target_binding": binding,
        "leading_hypothesis": {
            "hypothesis_id": str(leading.get("hypothesis_id") or ""),
            "summary": str(leading.get("summary") or "")[:500],
            "target_file": str(leading.get("target_file") or ""),
            "predicted_action_effects": _as_list(leading.get("predicted_action_effects"))[:5],
            "evidence_refs": _as_list(leading.get("evidence_refs"))[:8],
        },
    }


def _parse_llm_patch_response(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    for prefix in ("PATCH_JSON:", "PROPOSAL_JSON:"):
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                raw = stripped[len(prefix):].strip()
                break
    start = raw.find("{")
    end = raw.rfind("}") + 1
    candidate = raw[start:end] if start >= 0 and end > start else raw
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _llm_bounded_diff_proposals(
    context: Mapping[str, Any],
    *,
    target_file: str,
    content: str,
    llm_client: Any,
    max_changed_lines: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    think_gateway = ensure_llm_gateway(llm_client, route_name="patch_proposal", capability_prefix="general")
    act_gateway = ensure_llm_gateway(llm_client, route_name="structured_answer", capability_prefix="structured_output")
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
            is_timeout = "timeout" in gateway_error.lower()
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
                "failure_reason": "timeout" if is_timeout else "llm_request_failed",
                "llm_timeout": bool(is_timeout),
            }]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        is_timeout = "timeout" in error.lower()
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
            "failure_reason": "timeout" if is_timeout else "llm_request_failed",
            "llm_timeout": bool(is_timeout),
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
            is_timeout = _is_timeout_error(distill_error)
            traces.append({
                "function_name": "propose_patch",
                "capability": str(GENERAL_REASONING),
                "route_name": "structured_answer",
                "stage": "distill_pass",
                "prompt": distill_trace_prompt,
                "system_prompt": distill_system_prompt,
                "response": "",
                "parsed_kwargs": {},
                "request_kwargs": _request_kwargs_trace(distill_kwargs),
                "error": distill_error,
                "failure_reason": "timeout" if is_timeout else "distill_failed",
                "llm_timeout": bool(is_timeout),
            })
            return [], traces
        parsed_state = _parse_llm_patch_response(distill_response)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        is_timeout = _is_timeout_error(error)
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "structured_answer",
            "stage": "distill_pass",
            "prompt": distill_trace_prompt,
            "system_prompt": distill_system_prompt,
            "response": "",
            "parsed_kwargs": {},
            "request_kwargs": _request_kwargs_trace(distill_kwargs),
            "error": error,
            "failure_reason": "timeout" if is_timeout else "distill_failed",
            "llm_timeout": bool(is_timeout),
        })
        return [], traces
    if not parsed_state:
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "structured_answer",
            "stage": "distill_pass",
            "prompt": distill_trace_prompt,
            "system_prompt": distill_system_prompt,
            "response": str(distill_response or ""),
            "parsed_kwargs": {},
            "request_kwargs": _request_kwargs_trace(distill_kwargs),
            "error": "missing_reasoning_state",
        })
        return [], traces
    reasoning_state = _normalize_reasoning_state(parsed_state, target_file=target_file, related=related)
    traces.append({
        "function_name": "propose_patch",
        "capability": str(GENERAL_REASONING),
        "route_name": "structured_answer",
        "stage": "distill_pass",
        "prompt": distill_trace_prompt,
        "system_prompt": distill_system_prompt,
        "response": json.dumps({"reasoning_state": reasoning_state}, ensure_ascii=False),
        "parsed_kwargs": {"reasoning_state": reasoning_state},
        "request_kwargs": _request_kwargs_trace(distill_kwargs),
        "raw_thinking_discarded": True,
        "error": "",
    })
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
            is_timeout = _is_timeout_error(act_error)
            traces.append({
                "function_name": "propose_patch",
                "capability": str(GENERAL_REASONING),
                "route_name": "structured_answer",
                "stage": "act_pass",
                "prompt": act_prompt,
                "system_prompt": act_system_prompt,
                "response": "",
                "parsed_kwargs": {},
                "request_kwargs": _request_kwargs_trace(act_kwargs),
                "reasoning_state": reasoning_state,
                "error": act_error,
                "failure_reason": "timeout" if is_timeout else "act_failed",
                "llm_timeout": bool(is_timeout),
            })
            return [], traces
        parsed = _parse_llm_patch_response(act_response)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        is_timeout = _is_timeout_error(error)
        traces.append({
            "function_name": "propose_patch",
            "capability": str(GENERAL_REASONING),
            "route_name": "structured_answer",
            "stage": "act_pass",
            "prompt": act_prompt,
            "system_prompt": act_system_prompt,
            "response": "",
            "parsed_kwargs": {},
            "request_kwargs": _request_kwargs_trace(act_kwargs),
            "reasoning_state": reasoning_state,
            "error": error,
            "failure_reason": "timeout" if is_timeout else "act_failed",
            "llm_timeout": bool(is_timeout),
        })
        return [], traces
    trace = {
        "function_name": "propose_patch",
        "capability": str(GENERAL_REASONING),
        "route_name": "structured_answer",
        "stage": "act_pass",
        "prompt": act_prompt,
        "system_prompt": act_system_prompt,
        "response": str(act_response or ""),
        "parsed_kwargs": dict(parsed),
        "reasoning_state": reasoning_state,
        "request_kwargs": _request_kwargs_trace(act_kwargs),
        "error": "",
    }
    diff = _text(parsed.get("unified_diff"))
    if not diff:
        trace["error"] = "missing_unified_diff"
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
        source="bounded_llm_diff",
    )
    proposal["reasoning_state"] = reasoning_state
    proposal["pipeline"] = "think_distill_act"
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
