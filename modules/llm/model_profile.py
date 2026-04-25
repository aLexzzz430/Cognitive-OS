from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from modules.llm.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient


MODEL_SELF_PROFILE_VERSION = "conos.model_self_profile/v1"
MODEL_PROFILE_VERSION = "conos.model_profile/v1"
MODEL_PROFILE_REPORT_VERSION = "conos.model_profile_report/v1"

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
    "hypothesis": ("reasoning", "planning"),
    "recovery": ("reasoning", "verification"),
    "retrieval": ("retrieval", "speed"),
    "skill": ("instruction_following", "speed"),
    "probe": ("verification", "structured_output"),
    "structured_answer": ("structured_output", "reasoning"),
    "analyst": ("verification", "reasoning"),
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


def sanitize_route_model_name(model: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(model or "").strip()).strip("_").lower()
    return clean or "model"


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


def _route_score(scores: Mapping[str, Any], route_name: str) -> float:
    keys = ROUTE_MODEL_CAPABILITY_MAP.get(route_name, ("reasoning",))
    return _mean([_clamp01(scores.get(key), 0.0) for key in keys])


def _capability_names_from_scores(scores: Mapping[str, Any]) -> list[str]:
    capabilities: list[str] = []
    for key, value in dict(scores or {}).items():
        if _clamp01(value, 0.0) >= 0.55:
            capabilities.append(str(key))
    if "reasoning" not in capabilities:
        capabilities.append("reasoning")
    return capabilities


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
        served_routes = [
            route_name
            for route_name, score in route_scores.items()
            if score >= float(min_route_score)
        ]
        if not served_routes:
            served_routes = ["general"]
        route_name = f"ollama_{sanitize_route_model_name(model)}"
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
                },
            },
            "metadata": {
                "policy_source": "model_profile",
                "model_profile_provider": profile_provider,
                "model_profile_model": model,
            },
        }
    return policies


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
