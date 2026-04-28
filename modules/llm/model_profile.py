from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from modules.llm.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient
from modules.llm.openai_client import DEFAULT_OPENAI_BASE_URL, OpenAIClient
from modules.llm.provider_inventory import VisibleModel, list_visible_provider_models
from modules.llm.route_runtime_policy import ROUTE_RUNTIME_POLICY_VERSION, route_runtime_policies_for_routes


MODEL_SELF_PROFILE_VERSION = "conos.model_self_profile/v1"
MODEL_PROFILE_VERSION = "conos.model_profile/v1"
MODEL_PROFILE_REPORT_VERSION = "conos.model_profile_report/v1"
MODEL_ROUTE_SUMMARY_VERSION = "conos.model_route_summary/v1"

CAPABILITY_KEYS: tuple[str, ...] = (
    "reasoning",
    "planning",
    "structured_output",
    "tool_use",
    "verification",
    "speed",
    "instruction_following",
    "retrieval",
    "coding",
    "long_context",
)

ROUTE_MODEL_CAPABILITY_MAP: Dict[str, tuple[str, ...]] = {
    "general": ("reasoning", "instruction_following"),
    "deliberation": ("reasoning", "planning"),
    "planning": ("reasoning", "planning"),
    "planner": ("reasoning", "planning"),
    "plan_generation": ("reasoning", "planning"),
    "hypothesis": ("reasoning", "planning"),
    "recovery": ("reasoning", "verification"),
    "root_cause": ("reasoning", "verification"),
    "test_failure": ("reasoning", "verification"),
    "patch_proposal": ("reasoning", "coding", "verification"),
    "final_audit": ("verification", "reasoning"),
    "retrieval": ("retrieval", "speed"),
    "skill": ("instruction_following", "speed"),
    "probe": ("verification", "structured_output"),
    "structured_answer": ("structured_output", "reasoning"),
    "analyst": ("verification", "reasoning"),
}

STRICT_ROUTE_CAPABILITY_MINIMUMS: Dict[str, Dict[str, float]] = {
    "probe": {
        "verification": 0.55,
        "structured_output": 0.55,
        "instruction_following": 0.50,
    },
    "structured_answer": {
        "structured_output": 0.65,
        "instruction_following": 0.50,
    },
    "analyst": {
        "verification": 0.55,
        "reasoning": 0.55,
        "instruction_following": 0.50,
    },
}

ROUTE_SUMMARY_CONTEXTS: Dict[str, Dict[str, Any]] = {
    "general": {"required_capabilities": ["reasoning"], "prefer_high_trust": 0.4},
    "deliberation": {"required_capabilities": ["reasoning", "planning"], "uncertainty_level": 0.6, "prefer_high_trust": 0.7},
    "planning": {"required_capabilities": ["reasoning", "planning"], "uncertainty_level": 0.9, "prefer_high_trust": 0.98, "prefer_low_latency": 0.0, "prefer_low_cost": 0.0},
    "planner": {"required_capabilities": ["reasoning", "planning"], "uncertainty_level": 0.9, "prefer_high_trust": 0.98, "prefer_low_latency": 0.0, "prefer_low_cost": 0.0},
    "plan_generation": {"required_capabilities": ["reasoning", "planning"], "uncertainty_level": 0.9, "prefer_high_trust": 0.98, "prefer_low_latency": 0.0, "prefer_low_cost": 0.0},
    "hypothesis": {"required_capabilities": ["reasoning", "planning"], "uncertainty_level": 0.7, "prefer_high_trust": 0.7},
    "recovery": {"required_capabilities": ["reasoning", "verification"], "verification_pressure": 0.7, "prefer_high_trust": 0.8},
    "root_cause": {"required_capabilities": ["reasoning", "verification"], "verification_pressure": 0.8, "uncertainty_level": 0.8, "prefer_high_trust": 0.85},
    "test_failure": {"required_capabilities": ["reasoning", "verification"], "verification_pressure": 0.8, "prefer_high_trust": 0.85},
    "patch_proposal": {"required_capabilities": ["reasoning", "coding", "verification"], "verification_pressure": 0.8, "uncertainty_level": 0.7, "prefer_high_trust": 0.85},
    "final_audit": {"required_capabilities": ["verification", "reasoning"], "verification_pressure": 0.95, "prefer_high_trust": 0.95},
    "retrieval": {"required_capabilities": ["retrieval"], "prefer_low_latency": 1.0, "prefer_low_cost": 0.8},
    "skill": {"required_capabilities": ["instruction_following"], "prefer_low_latency": 0.8, "prefer_low_cost": 0.6},
    "probe": {"required_capabilities": ["verification", "structured_output"], "verification_pressure": 0.8, "prefer_structured_output": 0.7},
    "structured_answer": {"required_capabilities": ["structured_output"], "prefer_structured_output": 1.0, "verification_pressure": 0.5},
    "analyst": {"required_capabilities": ["verification", "reasoning"], "verification_pressure": 0.9, "prefer_high_trust": 0.9},
}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _json_object_from_text(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    candidate = cleaned[start:end] if start >= 0 and end > start else "{}"
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _complete_json(client: Any, prompt: str, *, max_tokens: int = 256) -> tuple[Dict[str, Any], str, float]:
    started_at = time.perf_counter()
    try:
        raw = client.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            system_prompt="Return one compact JSON object only. No markdown.",
            think=False,
        )
    except TypeError:
        raw = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    raw_text = str(raw or "")
    return _json_object_from_text(raw_text), raw_text, elapsed_ms


def _scores_from_mapping(value: Any) -> Dict[str, float]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    return {key: _clamp01(payload.get(key), 0.0) for key in CAPABILITY_KEYS}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def default_model_profile_store_path() -> Path:
    return Path.home() / ".conos" / "runtime" / "model_profiles.json"


def default_model_route_policy_path() -> Path:
    return Path.home() / ".conos" / "runtime" / "llm_route_policies.json"


def sanitize_route_model_name(model: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(model or "").strip()).strip("_").lower()
    return clean or "model"


def _split_model_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = re.split(r"[,;\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        chunks = [str(item) for item in value]
    else:
        chunks = [str(value)]
    seen: set[str] = set()
    names: list[str] = []
    for chunk in chunks:
        name = str(chunk or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _profile_report(
    *,
    provider: str,
    base_url: str,
    selected_models: Sequence[str],
    generated: int,
    reused: int,
    store: "ModelProfileStore",
    profiles: Sequence[Mapping[str, Any]],
    route_policies: Mapping[str, Any],
    errors: Optional[Sequence[Mapping[str, Any]]] = None,
    skipped_reason: str = "",
    listed_model_count: int = 0,
) -> Dict[str, Any]:
    return {
        "schema_version": MODEL_PROFILE_REPORT_VERSION,
        "provider": provider,
        "base_url": base_url,
        "model_count": len(list(selected_models or [])),
        "listed_model_count": int(listed_model_count or 0),
        "generated_count": int(generated),
        "reused_count": int(reused),
        "error_count": len(list(errors or [])),
        "skipped_reason": str(skipped_reason or ""),
        "store_path": str(store.path),
        "profiles": [dict(profile) for profile in list(profiles or [])],
        "route_policies": dict(route_policies or {}),
        "errors": [dict(error) for error in list(errors or [])],
    }


def build_self_profile_prompt(*, provider: str, model: str) -> str:
    keys = ", ".join(CAPABILITY_KEYS)
    return "\n".join(
        [
            "SELF_PROFILE_PROBE",
            "You are being connected as one model inside Cognitive OS.",
            "Self-assess your likely capabilities. This is only a prior and will be calibrated.",
            f"Provider: {provider}",
            f"Model: {model}",
            f"Capability keys: {keys}",
            (
                "Return exactly JSON: "
                '{"schema_version":"conos.model_self_profile/v1",'
                '"declared_strengths":["..."],"declared_weaknesses":["..."],'
                '"capability_scores":{"reasoning":0.5,"planning":0.5,'
                '"structured_output":0.5,"tool_use":0.5,"verification":0.5,'
                '"speed":0.5,"instruction_following":0.5,"retrieval":0.5,'
                '"coding":0.5,"long_context":0.5},'
                '"preferred_task_types":["..."],"avoid_task_types":["..."],'
                '"confidence_in_self_report":0.5}'
            ),
        ]
    )


def normalize_self_profile(
    payload: Mapping[str, Any],
    *,
    provider: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    scores = _scores_from_mapping(payload.get("capability_scores", {}) if isinstance(payload, Mapping) else {})
    return {
        "schema_version": MODEL_SELF_PROFILE_VERSION,
        "provider": str(provider or ""),
        "base_url": str(base_url or ""),
        "model": str(model or ""),
        "declared_strengths": _string_list(payload.get("declared_strengths", [])),
        "declared_weaknesses": _string_list(payload.get("declared_weaknesses", [])),
        "capability_scores": scores,
        "preferred_task_types": _string_list(payload.get("preferred_task_types", [])),
        "avoid_task_types": _string_list(payload.get("avoid_task_types", [])),
        "confidence_in_self_report": _clamp01(payload.get("confidence_in_self_report"), 0.5),
    }


def _probe_result(name: str, ok: bool, *, elapsed_ms: float, raw_response: str = "", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "score": 1.0 if ok else 0.0,
        "latency_ms": round(float(elapsed_ms), 3),
        "raw_response": str(raw_response or "")[:1000],
        "details": dict(details or {}),
    }


def run_calibration_probes(client: Any) -> Dict[str, Any]:
    probes: list[Dict[str, Any]] = []

    payload, raw, elapsed = _complete_json(
        client,
        'STRUCTURED_JSON_PROBE Return exactly {"answer":"ok","items":[1,2,3]}',
        max_tokens=96,
    )
    probes.append(
        _probe_result(
            "structured_json",
            payload.get("answer") == "ok" and payload.get("items") == [1, 2, 3],
            elapsed_ms=elapsed,
            raw_response=raw,
        )
    )

    payload, raw, elapsed = _complete_json(
        client,
        'INSTRUCTION_FOLLOWING_PROBE Allowed ids are alpha and beta. Return {"selected_id":"beta"}',
        max_tokens=96,
    )
    probes.append(
        _probe_result(
            "instruction_following",
            payload.get("selected_id") == "beta",
            elapsed_ms=elapsed,
            raw_response=raw,
        )
    )

    payload, raw, elapsed = _complete_json(
        client,
        'REASONING_PROBE If A beats B and B beats C, which option is strongest? Return {"strongest":"A"}',
        max_tokens=96,
    )
    probes.append(
        _probe_result(
            "reasoning",
            str(payload.get("strongest", "")).strip().upper() == "A",
            elapsed_ms=elapsed,
            raw_response=raw,
        )
    )

    payload, raw, elapsed = _complete_json(
        client,
        'VERIFICATION_PROBE Claims: claim_1 says 2+2=4, claim_2 says 2+2=5. Return {"false_claim":"claim_2"}',
        max_tokens=96,
    )
    probes.append(
        _probe_result(
            "verification",
            payload.get("false_claim") == "claim_2",
            elapsed_ms=elapsed,
            raw_response=raw,
        )
    )

    latencies = [float(row.get("latency_ms", 0.0) or 0.0) for row in probes]
    avg_latency_ms = _mean(latencies)
    if avg_latency_ms <= 1000.0:
        speed_score = 0.95
    elif avg_latency_ms <= 3000.0:
        speed_score = 0.75
    elif avg_latency_ms <= 8000.0:
        speed_score = 0.5
    else:
        speed_score = 0.25

    structured = float(probes[0]["score"])
    instruction = float(probes[1]["score"])
    reasoning = float(probes[2]["score"])
    verification = float(probes[3]["score"])
    scores = {
        "reasoning": reasoning,
        "planning": reasoning,
        "structured_output": structured,
        "tool_use": instruction,
        "verification": verification,
        "speed": speed_score,
        "instruction_following": instruction,
        "retrieval": max(instruction, speed_score),
        "coding": _mean([reasoning, structured]),
        "long_context": 0.5,
    }
    return {
        "schema_version": "conos.model_calibration/v1",
        "probe_count": len(probes),
        "passed_count": sum(1 for row in probes if row.get("ok")),
        "avg_latency_ms": round(avg_latency_ms, 3),
        "capability_scores": scores,
        "probes": probes,
    }


def combine_profile_scores(
    self_scores: Mapping[str, Any],
    calibration_scores: Mapping[str, Any],
    *,
    feedback_scores: Optional[Mapping[str, Any]] = None,
    self_weight: float = 0.35,
    calibration_weight: float = 0.45,
    feedback_weight: float = 0.20,
) -> Dict[str, float]:
    feedback = dict(feedback_scores or {})
    total_weight = float(self_weight) + float(calibration_weight) + (float(feedback_weight) if feedback else 0.0)
    if total_weight <= 0:
        total_weight = 1.0
    merged: Dict[str, float] = {}
    for key in CAPABILITY_KEYS:
        value = (
            _clamp01(self_scores.get(key), 0.0) * float(self_weight)
            + _clamp01(calibration_scores.get(key), 0.0) * float(calibration_weight)
            + (_clamp01(feedback.get(key), 0.0) * float(feedback_weight) if feedback else 0.0)
        ) / total_weight
        merged[key] = round(_clamp01(value), 6)
    return merged


def build_model_profile(
    *,
    client: Any,
    provider: str = "ollama",
    base_url: str = "",
    model: str,
    feedback_scores: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = build_self_profile_prompt(provider=provider, model=model)
    self_payload, self_raw, self_latency_ms = _complete_json(client, prompt, max_tokens=512)
    self_profile = normalize_self_profile(
        self_payload,
        provider=provider,
        base_url=base_url,
        model=model,
    )
    calibration = run_calibration_probes(client)
    final_scores = combine_profile_scores(
        self_profile["capability_scores"],
        calibration["capability_scores"],
        feedback_scores=feedback_scores,
    )
    return {
        "schema_version": MODEL_PROFILE_VERSION,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": str(provider or ""),
        "base_url": str(base_url or ""),
        "model": str(model or ""),
        "self_profile": self_profile,
        "self_profile_raw_response": str(self_raw or "")[:2000],
        "self_profile_latency_ms": round(float(self_latency_ms), 3),
        "calibration": calibration,
        "feedback_scores": dict(feedback_scores or {}),
        "capability_scores": final_scores,
        "score_weights": {
            "self_report": 0.35,
            "calibration": 0.45,
            "route_feedback": 0.20 if feedback_scores else 0.0,
        },
    }


def _catalog_prior_scores(model: str, metadata: Mapping[str, Any]) -> Dict[str, float]:
    name = str(model or "").strip().lower()
    display = str(metadata.get("display_name", "") or "").strip().lower()
    joined = f"{name} {display}"
    scores = {
        "reasoning": 0.58,
        "planning": 0.55,
        "structured_output": 0.58,
        "tool_use": 0.55,
        "verification": 0.55,
        "speed": 0.50,
        "instruction_following": 0.58,
        "retrieval": 0.50,
        "coding": 0.55,
        "long_context": 0.50,
    }
    if "gpt-5.5" in joined:
        scores.update(reasoning=0.92, planning=0.90, verification=0.90, coding=0.90, long_context=0.86, speed=0.45)
    elif "gpt-5.4" in joined and "mini" not in joined:
        scores.update(reasoning=0.86, planning=0.84, verification=0.84, coding=0.86, long_context=0.80, speed=0.55)
    elif "gpt-5.4-mini" in joined or "mini" in joined:
        scores.update(reasoning=0.70, planning=0.66, verification=0.68, coding=0.70, speed=0.82, long_context=0.64)
    elif "gpt-5.3-codex" in joined:
        scores.update(reasoning=0.82, planning=0.82, verification=0.82, coding=0.92, structured_output=0.78, long_context=0.76, speed=0.58)
    elif "spark" in joined:
        scores.update(reasoning=0.74, planning=0.70, verification=0.72, coding=0.86, structured_output=0.76, speed=0.92, long_context=0.62)
    elif "gpt-5.2" in joined:
        scores.update(reasoning=0.78, planning=0.76, verification=0.76, coding=0.78, long_context=0.72, speed=0.55)
    elif name.startswith("gpt-"):
        scores.update(reasoning=0.72, planning=0.70, verification=0.70, coding=0.72, structured_output=0.70)
    if "codex" in joined:
        scores["coding"] = max(scores["coding"], 0.86)
        scores["tool_use"] = max(scores["tool_use"], 0.74)
    if "fast" in joined:
        scores["speed"] = max(scores["speed"], 0.82)
    return {key: _clamp01(scores.get(key), 0.5) for key in CAPABILITY_KEYS}


def build_catalog_model_profile(model: VisibleModel | Mapping[str, Any]) -> Dict[str, Any]:
    payload = model.to_dict() if isinstance(model, VisibleModel) else dict(model or {})
    metadata = dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {}
    if payload.get("display_name"):
        metadata.setdefault("display_name", payload.get("display_name"))
    if payload.get("visibility"):
        metadata.setdefault("visibility", payload.get("visibility"))
    if payload.get("supported_in_api") is not None:
        metadata.setdefault("supported_in_api", payload.get("supported_in_api"))
    provider = str(payload.get("provider", "") or "")
    base_url = str(payload.get("base_url", "") or "")
    model_name = str(payload.get("model", "") or "")
    scores = _catalog_prior_scores(model_name, metadata)
    return {
        "schema_version": MODEL_PROFILE_VERSION,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "base_url": base_url,
        "model": model_name,
        "profile_source": "provider_catalog",
        "self_profile": {
            "schema_version": MODEL_SELF_PROFILE_VERSION,
            "provider": provider,
            "base_url": base_url,
            "model": model_name,
            "declared_strengths": _capability_names_from_scores(scores),
            "declared_weaknesses": [],
            "capability_scores": scores,
            "preferred_task_types": [],
            "avoid_task_types": [],
            "confidence_in_self_report": 0.35,
        },
        "calibration": {
            "schema_version": "conos.model_calibration/v1",
            "probe_count": 0,
            "passed_count": 0,
            "avg_latency_ms": 0.0,
            "capability_scores": scores,
            "probes": [],
            "skipped_reason": "catalog_profile_no_live_probe",
        },
        "feedback_scores": {},
        "capability_scores": scores,
        "score_weights": {"provider_catalog": 1.0, "calibration": 0.0, "route_feedback": 0.0},
        "provider_catalog": dict(payload),
    }


def _route_score(scores: Mapping[str, Any], route_name: str) -> float:
    keys = ROUTE_MODEL_CAPABILITY_MAP.get(route_name, ("reasoning",))
    return _mean([_clamp01(scores.get(key), 0.0) for key in keys])


def _capability_names_from_scores(scores: Mapping[str, Any]) -> list[str]:
    capabilities: list[str] = []
    for key, value in dict(scores or {}).items():
        if _clamp01(value, 0.0) >= 0.55:
            capabilities.append(str(key))
    if not capabilities:
        capabilities.append("low_confidence")
    return capabilities


def _strict_route_block_reason(scores: Mapping[str, Any], route_name: str) -> str:
    minimums = STRICT_ROUTE_CAPABILITY_MINIMUMS.get(str(route_name or ""))
    if not minimums:
        return ""
    failures = [
        f"{capability}<{minimum:g}"
        for capability, minimum in minimums.items()
        if _clamp01(scores.get(capability), 0.0) < float(minimum)
    ]
    return "strict_route_minimum_failed:" + ",".join(failures) if failures else ""


def route_policies_from_profiles(
    profiles: Sequence[Mapping[str, Any]],
    *,
    provider: str = "ollama",
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    min_route_score: float = 0.45,
) -> Dict[str, Dict[str, Any]]:
    policies: Dict[str, Dict[str, Any]] = {}
    for profile in list(profiles or []):
        model = str(profile.get("model", "") or "").strip()
        if not model:
            continue
        profile_provider = str(profile.get("provider", "") or provider or "ollama")
        profile_base_url = str(profile.get("base_url", "") or base_url or DEFAULT_OLLAMA_BASE_URL)
        scores = dict(profile.get("capability_scores", {}) or {})
        route_scores = {
            route_name: _route_score(scores, route_name)
            for route_name in ROUTE_MODEL_CAPABILITY_MAP.keys()
        }
        route_eligibility: Dict[str, Dict[str, Any]] = {}
        served_routes: list[str] = []
        for route_name, score in route_scores.items():
            block_reason = _strict_route_block_reason(scores, route_name)
            eligible = float(score) >= float(min_route_score) and not block_reason
            route_eligibility[route_name] = {
                "score": float(score),
                "eligible": bool(eligible),
                "blocked_reason": block_reason,
            }
            if eligible:
                served_routes.append(route_name)
        route_name = f"{sanitize_route_model_name(profile_provider)}_{sanitize_route_model_name(model)}"
        ineligible_routes = [
            name
            for name, row in route_eligibility.items()
            if str(row.get("blocked_reason", "") or "")
        ]
        disabled_reason = "" if served_routes else "profile_route_scores_below_threshold_or_strict_minimums"
        policies[route_name] = {
            "served_routes": served_routes,
            "provider": profile_provider,
            "base_url": profile_base_url,
            "model": model,
            "capability_profile": {
                "capabilities": _capability_names_from_scores(scores),
                "trust_score": round(_mean([_clamp01(scores.get("reasoning"), 0.0), _clamp01(scores.get("verification"), 0.0)]), 6),
                "cost_efficiency": _clamp01(scores.get("speed"), 0.5),
                "latency_efficiency": _clamp01(scores.get("speed"), 0.5),
                "uncertainty_tolerance": round(_mean([_clamp01(scores.get("reasoning"), 0.0), _clamp01(scores.get("planning"), 0.0)]), 6),
                "verification_strength": _clamp01(scores.get("verification"), 0.0),
                "structured_output_strength": _clamp01(scores.get("structured_output"), 0.0),
                "metadata": {
                    "profile_schema_version": str(profile.get("schema_version", "") or ""),
                    "profiled_at": str(profile.get("profiled_at", "") or ""),
                    "route_scores": route_scores,
                    "route_eligibility": route_eligibility,
                    "ineligible_routes": ineligible_routes,
                },
            },
            "metadata": {
                "policy_source": "model_profile",
                "model_profile_provider": profile_provider,
                "model_profile_model": model,
                "disabled_reason": disabled_reason,
                "route_runtime_policy_version": ROUTE_RUNTIME_POLICY_VERSION,
                "route_runtime_policies": route_runtime_policies_for_routes(served_routes),
            },
        }
    return policies


def load_model_route_policies(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    policy_path = Path(path) if path else default_model_route_policy_path()
    if not policy_path.exists():
        return {}
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    raw_policies = payload.get("route_policies") if isinstance(payload.get("route_policies"), Mapping) else payload
    return {
        str(name): dict(policy)
        for name, policy in dict(raw_policies or {}).items()
        if str(name).strip() and isinstance(policy, Mapping)
    }


def write_model_route_policies(policies: Mapping[str, Any], path: str | Path | None = None) -> Path:
    policy_path = Path(path) if path else default_model_route_policy_path()
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(dict(policies or {}), indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return policy_path


def load_profile_backed_route_policies(
    *,
    store_path: str | Path | None = None,
    route_policy_path: str | Path | None = None,
    base_url: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    policies = load_model_route_policies(route_policy_path)
    if policies:
        return policies
    profiles = ModelProfileStore(store_path).list_profiles()
    if not profiles:
        return {}
    return route_policies_from_profiles(
        profiles,
        base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
    )


def build_model_route_summary(
    policies: Mapping[str, Any],
    *,
    routes: Optional[Sequence[str]] = None,
    explain: bool = False,
) -> Dict[str, Any]:
    from modules.llm.model_router import ModelRouter

    route_specs = {
        str(name): dict(policy)
        for name, policy in dict(policies or {}).items()
        if isinstance(policy, Mapping)
    }
    router = ModelRouter(route_specs=route_specs)
    route_names = list(routes or ROUTE_SUMMARY_CONTEXTS.keys())
    rows: list[Dict[str, Any]] = []
    for route_name in route_names:
        context = dict(ROUTE_SUMMARY_CONTEXTS.get(route_name, {"required_capabilities": ["reasoning"]}))
        decision = router.decide(route_name, context=context)
        selected_policy = dict(route_specs.get(str(decision.route_name or ""), {}) or {})
        row = {
            "route": route_name,
            "selected_route": str(decision.route_name or ""),
            "selected_model": str(selected_policy.get("model", "") or dict(decision.metadata or {}).get("model", "") or ""),
            "score": round(float(decision.score or 0.0), 6),
            "reason": list(decision.explanation or []),
        }
        if explain:
            candidates = []
            for candidate in list(decision.candidate_routes or []):
                candidate_policy = dict(route_specs.get(str(candidate.get("route_name", "") or ""), {}) or {})
                candidates.append(
                    {
                        "route_name": str(candidate.get("route_name", "") or ""),
                        "model": str(candidate_policy.get("model", "") or ""),
                        "score": float(candidate.get("score", 0.0) or 0.0),
                        "matched_capabilities": list(candidate.get("matched_capabilities", []) or []),
                        "explanation": list(candidate.get("explanation", []) or []),
                    }
                )
            row["candidates"] = candidates
            row["context"] = context
        rows.append(row)

    deprioritized = []
    for policy_name, raw_policy in route_specs.items():
        policy = dict(raw_policy or {})
        served_routes = list(policy.get("served_routes", []) or [])
        metadata = dict(policy.get("metadata", {}) or {})
        if served_routes:
            continue
        deprioritized.append(
            {
                "route_policy": policy_name,
                "model": str(policy.get("model", "") or ""),
                "reason": str(metadata.get("disabled_reason", "") or "no_eligible_routes"),
            }
        )
    return {
        "schema_version": MODEL_ROUTE_SUMMARY_VERSION,
        "route_policy_count": len(route_specs),
        "routes": rows,
        "deprioritized_models": deprioritized,
    }


def render_model_route_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "Con OS model routes",
        f"schema: {summary.get('schema_version', MODEL_ROUTE_SUMMARY_VERSION)}",
        f"route policies: {int(summary.get('route_policy_count', 0) or 0)}",
        "",
        f"{'route':<18} {'selected_model':<28} {'score':>7} reason",
        "-" * 92,
    ]
    for row in list(summary.get("routes", []) or []):
        if not isinstance(row, Mapping):
            continue
        reason = "; ".join(str(item) for item in list(row.get("reason", []) or []))
        lines.append(
            f"{str(row.get('route', '') or ''):<18} "
            f"{str(row.get('selected_model', '') or ''):<28} "
            f"{float(row.get('score', 0.0) or 0.0):>7.3f} "
            f"{reason}"
        )
    deprioritized = [row for row in list(summary.get("deprioritized_models", []) or []) if isinstance(row, Mapping)]
    if deprioritized:
        lines.extend(["", "Deprioritized models"])
        for row in deprioritized:
            lines.append(
                f"- {row.get('model', '')}: {row.get('reason', '')}"
            )
    return "\n".join(lines)


@dataclass
class ModelProfileStore:
    path: Path

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_model_profile_store_path()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": "conos.model_profile_store/v1", "profiles": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"schema_version": "conos.model_profile_store/v1", "profiles": []}
        if not isinstance(payload, dict):
            return {"schema_version": "conos.model_profile_store/v1", "profiles": []}
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list):
            profiles = []
        return {"schema_version": "conos.model_profile_store/v1", "profiles": profiles}

    def save(self, payload: Mapping[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized = {
            "schema_version": "conos.model_profile_store/v1",
            "profiles": list(payload.get("profiles", []) or []) if isinstance(payload.get("profiles", []), list) else [],
        }
        self.path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
        return self.path

    def list_profiles(self) -> list[Dict[str, Any]]:
        return [dict(row) for row in self.load().get("profiles", []) if isinstance(row, Mapping)]

    def get_profile(self, *, provider: str, base_url: str, model: str) -> Optional[Dict[str, Any]]:
        for profile in self.list_profiles():
            if (
                str(profile.get("provider", "") or "") == str(provider or "")
                and str(profile.get("base_url", "") or "") == str(base_url or "")
                and str(profile.get("model", "") or "") == str(model or "")
            ):
                return profile
        return None

    def upsert_profile(self, profile: Mapping[str, Any]) -> Path:
        payload = self.load()
        profiles = [
            dict(row)
            for row in payload.get("profiles", [])
            if isinstance(row, Mapping)
            and not (
                str(row.get("provider", "") or "") == str(profile.get("provider", "") or "")
                and str(row.get("base_url", "") or "") == str(profile.get("base_url", "") or "")
                and str(row.get("model", "") or "") == str(profile.get("model", "") or "")
            )
        ]
        profiles.append(dict(profile))
        payload["profiles"] = profiles
        return self.save(payload)


def profile_ollama_models(
    *,
    base_url: str | None = None,
    models: Optional[Iterable[str]] = None,
    timeout_sec: float = 30.0,
    store_path: str | Path | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    provider = "ollama"
    inventory_client = OllamaClient(base_url=base_url, auto_select_model=False, timeout_sec=timeout_sec)
    resolved_base_url = inventory_client.base_url
    selected_models = [str(model).strip() for model in list(models or []) if str(model).strip()]
    if not selected_models:
        selected_models = inventory_client.list_models()
    store = ModelProfileStore(store_path)
    profiles: list[Dict[str, Any]] = []
    reused = 0
    generated = 0
    for model in selected_models:
        existing = store.get_profile(provider=provider, base_url=resolved_base_url, model=model)
        if existing is not None and not force:
            profiles.append(existing)
            reused += 1
            continue
        client = OllamaClient(base_url=resolved_base_url, model=model, timeout_sec=timeout_sec)
        profile = build_model_profile(
            client=client,
            provider=provider,
            base_url=resolved_base_url,
            model=model,
        )
        store.upsert_profile(profile)
        profiles.append(profile)
        generated += 1
    route_policies = route_policies_from_profiles(
        profiles,
        provider=provider,
        base_url=resolved_base_url,
    )
    return {
        "schema_version": MODEL_PROFILE_REPORT_VERSION,
        "provider": provider,
        "base_url": resolved_base_url,
        "model_count": len(selected_models),
        "generated_count": generated,
        "reused_count": reused,
        "store_path": str(store.path),
        "profiles": profiles,
        "route_policies": route_policies,
    }


def _looks_like_openai_text_model(model_id: str) -> bool:
    name = str(model_id or "").strip().lower()
    if not name:
        return False
    excluded_fragments = (
        "embedding",
        "moderation",
        "whisper",
        "tts",
        "audio",
        "realtime",
        "transcribe",
        "dall-e",
        "image",
        "speech",
    )
    if any(fragment in name for fragment in excluded_fragments):
        return False
    return name.startswith(("gpt-", "chatgpt-", "o1", "o2", "o3", "o4", "o5", "codex"))


def list_openai_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_sec: float = 30.0,
    text_only: bool = True,
) -> list[str]:
    client = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        model="",
        timeout_sec=timeout_sec,
        require_model=False,
    )
    models = client.list_models()
    if text_only:
        models = [model for model in models if _looks_like_openai_text_model(model)]
    return _split_model_names(models)


def _openai_configured_models(models: Optional[Iterable[str]] = None) -> list[str]:
    selected_models = _split_model_names(models)
    if selected_models:
        return selected_models
    selected_models = _split_model_names(os.getenv("OPENAI_MODELS"))
    if selected_models:
        return selected_models
    return _split_model_names(os.getenv("OPENAI_MODEL"))


def profile_openai_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    models: Optional[Iterable[str]] = None,
    timeout_sec: float = 60.0,
    store_path: str | Path | None = None,
    force: bool = False,
    all_cloud_models: bool = False,
    max_models: int | None = None,
    text_only: bool = True,
) -> Dict[str, Any]:
    provider = "openai"
    resolved_base_url = str(base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    selected_models = _openai_configured_models(models)
    store = ModelProfileStore(store_path)
    errors: list[Dict[str, Any]] = []
    listed_model_count = 0

    if all_cloud_models:
        try:
            listed_models = list_openai_models(
                api_key=api_key,
                base_url=resolved_base_url,
                timeout_sec=timeout_sec,
                text_only=text_only,
            )
            listed_model_count = len(listed_models)
            selected_models = listed_models
        except Exception as exc:
            errors.append({"provider": provider, "model": "", "stage": "list_models", "error": str(exc)})

    if max_models is not None and int(max_models or 0) > 0:
        selected_models = selected_models[: int(max_models or 0)]

    profiles: list[Dict[str, Any]] = []
    reused = 0
    generated = 0
    if not selected_models:
        return _profile_report(
            provider=provider,
            base_url=resolved_base_url,
            selected_models=[],
            generated=0,
            reused=0,
            store=store,
            profiles=[],
            route_policies={},
            errors=errors,
            skipped_reason="no_openai_models_configured",
            listed_model_count=listed_model_count,
        )

    for model in selected_models:
        existing = store.get_profile(provider=provider, base_url=resolved_base_url, model=model)
        if existing is not None and not force:
            profiles.append(existing)
            reused += 1
            continue
        try:
            client = OpenAIClient(
                api_key=api_key,
                base_url=resolved_base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
            profile = build_model_profile(
                client=client,
                provider=provider,
                base_url=resolved_base_url,
                model=model,
            )
        except Exception as exc:
            errors.append({"provider": provider, "model": model, "stage": "profile", "error": str(exc)})
            continue
        store.upsert_profile(profile)
        profiles.append(profile)
        generated += 1

    route_policies = route_policies_from_profiles(
        profiles,
        provider=provider,
        base_url=resolved_base_url,
    )
    return _profile_report(
        provider=provider,
        base_url=resolved_base_url,
        selected_models=selected_models,
        generated=generated,
        reused=reused,
        store=store,
        profiles=profiles,
        route_policies=route_policies,
        errors=errors,
        listed_model_count=listed_model_count,
    )


def profile_visible_provider_catalog(
    *,
    provider: str,
    base_url: str | None = None,
    models: Optional[Iterable[str]] = None,
    timeout_sec: float = 30.0,
    store_path: str | Path | None = None,
    force: bool = False,
    include_hidden: bool = False,
) -> Dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    if normalized == "codex":
        normalized = "codex-cli"
    visible = list_visible_provider_models(
        provider=normalized,
        base_url=base_url,
        models=models,
        timeout_sec=timeout_sec,
        include_hidden=include_hidden,
    )
    resolved_base_url = str(visible[0].base_url if visible else (base_url or ""))
    selected_models = [row.model for row in visible]
    store = ModelProfileStore(store_path)
    profiles: list[Dict[str, Any]] = []
    reused = 0
    generated = 0
    for row in visible:
        existing = store.get_profile(provider=row.provider, base_url=row.base_url, model=row.model)
        if existing is not None and not force:
            profiles.append(existing)
            reused += 1
            continue
        profile = build_catalog_model_profile(row)
        store.upsert_profile(profile)
        profiles.append(profile)
        generated += 1
    route_policies = route_policies_from_profiles(
        profiles,
        provider=normalized,
        base_url=resolved_base_url,
    )
    report = _profile_report(
        provider=normalized,
        base_url=resolved_base_url,
        selected_models=selected_models,
        generated=generated,
        reused=reused,
        store=store,
        profiles=profiles,
        route_policies=route_policies,
        listed_model_count=len(visible),
    )
    report["profile_mode"] = "provider_catalog"
    report["visible_models"] = [row.to_dict() for row in visible]
    return report


def profile_provider_models(
    *,
    provider: str,
    base_url: str | None = None,
    models: Optional[Iterable[str]] = None,
    timeout_sec: float = 60.0,
    store_path: str | Path | None = None,
    force: bool = False,
    all_cloud_models: bool = False,
    max_cloud_models: int | None = None,
    catalog_only: bool = False,
    discover_visible: bool = False,
    include_hidden: bool = False,
) -> Dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    if normalized in {"codex", "codex-cli", "openai-oauth-codex"}:
        return profile_visible_provider_catalog(
            provider="codex-cli",
            base_url="codex-cli://chatgpt-oauth",
            models=models,
            timeout_sec=timeout_sec,
            store_path=store_path,
            force=force,
            include_hidden=include_hidden,
        )
    if catalog_only or discover_visible:
        return profile_visible_provider_catalog(
            provider=normalized,
            base_url=base_url,
            models=models,
            timeout_sec=timeout_sec,
            store_path=store_path,
            force=force,
            include_hidden=include_hidden,
        )
    if normalized == "ollama":
        return profile_ollama_models(
            base_url=base_url,
            models=models,
            timeout_sec=timeout_sec,
            store_path=store_path,
            force=force,
        )
    if normalized == "openai":
        return profile_openai_models(
            base_url=base_url,
            models=models,
            timeout_sec=timeout_sec,
            store_path=store_path,
            force=force,
            all_cloud_models=all_cloud_models,
            max_models=max_cloud_models,
        )
    raise ValueError(f"Unsupported model profile provider: {provider}")


def profile_all_configured_models(
    *,
    ollama_base_url: str | None = None,
    openai_base_url: str | None = None,
    ollama_models: Optional[Iterable[str]] = None,
    openai_models: Optional[Iterable[str]] = None,
    timeout_sec: float = 60.0,
    store_path: str | Path | None = None,
    force: bool = False,
    include_ollama: bool = True,
    include_openai: bool = True,
    include_codex: bool = False,
    all_cloud_models: bool = False,
    max_cloud_models: int | None = None,
    catalog_only: bool = False,
) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    errors: list[Dict[str, Any]] = []
    profiles: list[Dict[str, Any]] = []
    route_policies: Dict[str, Any] = {}
    if include_ollama:
        try:
            if catalog_only:
                reports["ollama"] = profile_visible_provider_catalog(
                    provider="ollama",
                    base_url=ollama_base_url,
                    models=ollama_models,
                    timeout_sec=timeout_sec,
                    store_path=store_path,
                    force=force,
                )
            else:
                reports["ollama"] = profile_ollama_models(
                    base_url=ollama_base_url,
                    models=ollama_models,
                    timeout_sec=timeout_sec,
                    store_path=store_path,
                    force=force,
                )
        except Exception as exc:
            reports["ollama"] = {
                "schema_version": MODEL_PROFILE_REPORT_VERSION,
                "provider": "ollama",
                "base_url": str(ollama_base_url or ""),
                "model_count": 0,
                "generated_count": 0,
                "reused_count": 0,
                "error_count": 1,
                "skipped_reason": "provider_unreachable",
                "profiles": [],
                "route_policies": {},
                "errors": [{"provider": "ollama", "stage": "profile_all", "error": str(exc)}],
            }
    if include_openai:
        try:
            if catalog_only:
                reports["openai"] = profile_visible_provider_catalog(
                    provider="openai",
                    base_url=openai_base_url,
                    models=openai_models,
                    timeout_sec=timeout_sec,
                    store_path=store_path,
                    force=force,
                )
            else:
                reports["openai"] = profile_openai_models(
                    base_url=openai_base_url,
                    models=openai_models,
                    timeout_sec=timeout_sec,
                    store_path=store_path,
                    force=force,
                    all_cloud_models=all_cloud_models,
                    max_models=max_cloud_models,
                )
        except Exception as exc:
            reports["openai"] = {
                "schema_version": MODEL_PROFILE_REPORT_VERSION,
                "provider": "openai",
                "base_url": str(openai_base_url or ""),
                "model_count": 0,
                "generated_count": 0,
                "reused_count": 0,
                "error_count": 1,
                "skipped_reason": "provider_unreachable",
                "profiles": [],
                "route_policies": {},
                "errors": [{"provider": "openai", "stage": "profile_all", "error": str(exc)}],
            }
    if include_codex:
        try:
            reports["codex-cli"] = profile_visible_provider_catalog(
                provider="codex-cli",
                base_url="codex-cli://chatgpt-oauth",
                timeout_sec=timeout_sec,
                store_path=store_path,
                force=force,
            )
        except Exception as exc:
            reports["codex-cli"] = {
                "schema_version": MODEL_PROFILE_REPORT_VERSION,
                "provider": "codex-cli",
                "base_url": "codex-cli://chatgpt-oauth",
                "model_count": 0,
                "generated_count": 0,
                "reused_count": 0,
                "error_count": 1,
                "skipped_reason": "provider_unreachable",
                "profiles": [],
                "route_policies": {},
                "errors": [{"provider": "codex-cli", "stage": "profile_all", "error": str(exc)}],
            }

    for report in reports.values():
        if not isinstance(report, Mapping):
            continue
        profiles.extend([dict(profile) for profile in list(report.get("profiles", []) or []) if isinstance(profile, Mapping)])
        route_policies.update(dict(report.get("route_policies", {}) or {}))
        errors.extend([dict(error) for error in list(report.get("errors", []) or []) if isinstance(error, Mapping)])

    return {
        "schema_version": MODEL_PROFILE_REPORT_VERSION,
        "provider": "all",
        "model_count": len(profiles),
        "generated_count": sum(int(dict(report).get("generated_count", 0) or 0) for report in reports.values() if isinstance(report, Mapping)),
        "reused_count": sum(int(dict(report).get("reused_count", 0) or 0) for report in reports.values() if isinstance(report, Mapping)),
        "error_count": len(errors),
        "store_path": str(ModelProfileStore(store_path).path),
        "provider_reports": reports,
        "profiles": profiles,
        "route_policies": route_policies,
        "errors": errors,
    }
