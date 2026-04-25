from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.orchestration.action_utils import action_semantic_signature_key


COGNITIVE_LOOP_ABLATION_VERSION = "conos.cognitive_loop_ablation/v1"
BASELINE_LLM_DECISIONS_VERSION = "conos.cognitive_loop_ablation.baseline_decisions/v1"

ARM_FULL = "Full"
ARM_NO_POSTERIOR = "NoPosteriorUpdate"
ARM_NO_EXPERIMENT = "NoDiscriminatingExperiment"
ARM_NO_HYPOTHESIS = "NoHypothesisCompetition"
ARM_NO_SEMANTIC = "NoSemanticStrictness"
ARM_RANDOM = "RandomProbe"
ARM_BASELINE_LLM = "BaselineLLM"
ARM_NO_MEMORY = "NoMemory"
ARM_NO_EVIDENCE = "NoEvidence"
ARM_NO_STEP9 = "NoStep9"
ARM_NO_STEP10 = "NoStep10"
ARM_SHUFFLED_EVIDENCE = "ShuffledEvidence"
ARM_WRONG_BINDING = "WrongBinding"
ARM_FRESH_BASELINE = "FreshBaseline"

CORE_ABLATION_ARMS: Tuple[str, ...] = (
    ARM_FULL,
    ARM_NO_POSTERIOR,
    ARM_NO_EXPERIMENT,
    ARM_NO_HYPOTHESIS,
    ARM_NO_SEMANTIC,
    ARM_RANDOM,
    ARM_BASELINE_LLM,
)

ANTI_CHEAT_ARMS: Tuple[str, ...] = (
    ARM_NO_MEMORY,
    ARM_NO_EVIDENCE,
    ARM_NO_STEP9,
    ARM_NO_STEP10,
    ARM_SHUFFLED_EVIDENCE,
    ARM_WRONG_BINDING,
    ARM_FRESH_BASELINE,
)

DEFAULT_ARMS: Tuple[str, ...] = (
    *CORE_ABLATION_ARMS,
    *ANTI_CHEAT_ARMS,
)

TASK_CATEGORIES: Tuple[str, ...] = (
    "hidden_rule",
    "hypothesis_conflict",
    "wrong_hypothesis_recovery",
    "noisy_observation",
    "semantic_action",
)

PASSING_THRESHOLDS = {
    "full_success_min_margin_vs_baseline_llm": 0.20,
    "full_success_min_margin_vs_no_posterior": 0.10,
    "full_success_min_margin_vs_no_experiment": 0.10,
    "wrong_commit_rate_max": 0.10,
    "false_rejection_rate_max": 0.15,
    "posterior_leading_hypothesis_accuracy_min": 0.75,
    "semantic_mismatch_false_experiment_rate_max": 0.0,
    "full_success_min_margin_vs_anti_cheat": 0.10,
    "full_evidence_binding_error_rate_max": 0.0,
    "full_formal_commit_rate_min": 1.0,
}

ARM_DESCRIPTIONS = {
    ARM_FULL: "complete closed-loop controller",
    ARM_NO_POSTERIOR: "posterior writes disabled after probe evidence",
    ARM_NO_EXPERIMENT: "discriminating experiment bonus disabled",
    ARM_NO_HYPOTHESIS: "ordinary action ranking without hypothesis competition",
    ARM_NO_SEMANTIC: "function-name-only matching for semantic actions",
    ARM_RANDOM: "random probe selection before commit",
    ARM_BASELINE_LLM: "bare-model baseline hook, no Con OS posterior or experiment loop",
    ARM_NO_MEMORY: "probe traces and posterior state are not retained across steps",
    ARM_NO_EVIDENCE: "tool outcomes are executed but never admitted as evidence",
    ARM_NO_STEP9: "execution outcomes are not converted into evidence objects",
    ARM_NO_STEP10: "evidence updates are not formally committed before final decision",
    ARM_SHUFFLED_EVIDENCE: "valid evidence is intentionally shuffled onto the wrong hypothesis",
    ARM_WRONG_BINDING: "evidence is bound to the wrong action/hypothesis key",
    ARM_FRESH_BASELINE: "fresh no-history controller that commits from priors only",
}


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    label: str
    prior: float
    is_correct: bool = False


@dataclass(frozen=True)
class ProbeSpec:
    action_id: str
    function_name: str
    kwargs: Dict[str, Any]
    supports_hypothesis_id: str
    expected_information_gain: float
    risk: float
    discriminating: bool = True
    semantic_sensitive: bool = False
    ordinary_rank: float = 0.0

    def to_action(self) -> Dict[str, Any]:
        return {
            "kind": "probe",
            "payload": {
                "tool_args": {
                    "function_name": self.function_name,
                    "kwargs": dict(self.kwargs),
                }
            },
            "_candidate_meta": {
                "role": "discriminate" if self.discriminating else "probe",
                "expected_information_gain": float(self.expected_information_gain),
                "risk": float(self.risk),
                "benchmark_action_id": self.action_id,
                "supports_hypothesis_id": self.supports_hypothesis_id,
                "semantic_sensitive": bool(self.semantic_sensitive),
            },
            "risk": float(self.risk),
            "estimated_cost": 1.0,
        }


@dataclass(frozen=True)
class CognitiveLoopTask:
    task_id: str
    category: str
    correct_hypothesis_id: str
    hypotheses: List[HypothesisSpec]
    probes: List[ProbeSpec]
    max_steps: int = 4
    misleading_initial: bool = False
    noisy_first_probe: bool = False
    semantic_task: bool = False


@dataclass
class TaskRunResult:
    task_id: str
    category: str
    arm: str
    success: bool
    steps: int
    committed_hypothesis_id: str
    correct_hypothesis_id: str
    wrong_commit: bool
    false_rejection: bool
    initial_leading_hypothesis_id: str
    final_leading_hypothesis_id: str
    posterior_leading_correct: bool
    flipped_to_correct: bool
    useful_experiment_count: int = 0
    experiment_count: int = 0
    semantic_mismatch_false_experiment_count: int = 0
    recovery_after_wrong_hypothesis: bool = False
    posterior_calibration_error: float = 0.0
    evidence_conversion_count: int = 0
    evidence_binding_error_count: int = 0
    formal_commit_count: int = 0
    memory_reset_count: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _revise_belief(prior: float, raw_delta: float, *, max_revision_rate: float = 0.35) -> float:
    prior = _clamp01(prior, 0.0)
    delta = float(raw_delta or 0.0)
    if delta == 0.0:
        return prior
    rate = min(max(0.0, float(max_revision_rate)), abs(delta))
    if delta > 0.0:
        return _clamp01(prior + (1.0 - prior) * rate)
    return _clamp01(prior - prior * rate)


def _make_probe(
    *,
    task_index: int,
    action_id: str,
    supports: str,
    info_gain: float,
    risk: float,
    x: int,
    y: int,
    semantic_sensitive: bool = False,
    ordinary_rank: float = 0.0,
) -> ProbeSpec:
    return ProbeSpec(
        action_id=f"task{task_index}_{action_id}",
        function_name="ACTION6",
        kwargs={
            "x": x,
            "y": y,
            "target_family": "color_target" if not semantic_sensitive else "coordinate_lock",
            "anchor_ref": f"cell_{task_index}",
        },
        supports_hypothesis_id=supports,
        expected_information_gain=info_gain,
        risk=risk,
        discriminating=info_gain >= 0.5,
        semantic_sensitive=semantic_sensitive,
        ordinary_rank=ordinary_rank,
    )


def build_cognitive_loop_benchmark_tasks(*, task_count: int = 25, seed: int = 17) -> List[CognitiveLoopTask]:
    count = max(20, min(50, int(task_count or 25)))
    rng = random.Random(seed)
    tasks: List[CognitiveLoopTask] = []
    for index in range(count):
        category = TASK_CATEGORIES[index % len(TASK_CATEGORIES)]
        correct_is_a = (index % 2 == 0)
        correct = f"t{index}_h_a" if correct_is_a else f"t{index}_h_b"
        wrong = f"t{index}_h_b" if correct_is_a else f"t{index}_h_a"
        misleading = category in {"wrong_hypothesis_recovery", "noisy_observation"}
        prior_correct = 0.30 if misleading else (0.52 if correct_is_a else 0.48)
        prior_wrong = 0.70 if misleading else (0.48 if correct_is_a else 0.52)
        if category == "hypothesis_conflict":
            prior_correct = 0.50
            prior_wrong = 0.50
        hypotheses = [
            HypothesisSpec(hypothesis_id=f"t{index}_h_a", label="red_rule", prior=prior_correct if correct_is_a else prior_wrong, is_correct=correct_is_a),
            HypothesisSpec(hypothesis_id=f"t{index}_h_b", label="blue_rule", prior=prior_wrong if correct_is_a else prior_correct, is_correct=not correct_is_a),
        ]
        useful_rank = 0.20 if category != "semantic_action" else 0.35
        decoy_rank = 0.92 if category != "semantic_action" else 0.96
        semantic = category == "semantic_action"
        probes = [
            _make_probe(
                task_index=index,
                action_id="decoy",
                supports=wrong,
                info_gain=0.18 if category != "semantic_action" else 0.86,
                risk=0.02,
                x=9 + (index % 3),
                y=9 + (index % 4),
                semantic_sensitive=semantic,
                ordinary_rank=decoy_rank,
            ),
            _make_probe(
                task_index=index,
                action_id="useful",
                supports=correct,
                info_gain=0.88 + rng.random() * 0.05,
                risk=0.08,
                x=1 + (index % 3),
                y=2 + (index % 4),
                semantic_sensitive=semantic,
                ordinary_rank=useful_rank,
            ),
            _make_probe(
                task_index=index,
                action_id="ambiguous",
                supports="",
                info_gain=0.12,
                risk=0.01,
                x=4 + (index % 2),
                y=5 + (index % 3),
                semantic_sensitive=False,
                ordinary_rank=0.70,
            ),
        ]
        tasks.append(
            CognitiveLoopTask(
                task_id=f"{category}_{index:02d}",
                category=category,
                correct_hypothesis_id=correct,
                hypotheses=hypotheses,
                probes=probes,
                max_steps=5 if category == "noisy_observation" else 4,
                misleading_initial=misleading,
                noisy_first_probe=category == "noisy_observation",
                semantic_task=semantic,
            )
        )
    return tasks


def _arm_config(arm: str) -> Dict[str, bool]:
    return {
        "posterior_update": arm not in {ARM_NO_POSTERIOR, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "discriminating_experiment": arm not in {ARM_NO_EXPERIMENT, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "hypothesis_competition": arm not in {ARM_NO_HYPOTHESIS, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "semantic_strictness": arm != ARM_NO_SEMANTIC,
        "random_probe": arm == ARM_RANDOM,
        "baseline_llm": arm == ARM_BASELINE_LLM,
        "memory": arm not in {ARM_NO_MEMORY, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "evidence": arm not in {ARM_NO_EVIDENCE, ARM_NO_STEP9, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "step9": arm not in {ARM_NO_STEP9, ARM_NO_EVIDENCE, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "step10": arm not in {ARM_NO_STEP10, ARM_BASELINE_LLM, ARM_FRESH_BASELINE},
        "shuffled_evidence": arm == ARM_SHUFFLED_EVIDENCE,
        "wrong_binding": arm == ARM_WRONG_BINDING,
        "fresh_baseline": arm == ARM_FRESH_BASELINE,
    }


def _leading(posteriors: Mapping[str, float]) -> str:
    return sorted(posteriors.items(), key=lambda item: (-float(item[1]), item[0]))[0][0]


def _commit_threshold(task: CognitiveLoopTask, arm: str) -> float:
    if arm == ARM_BASELINE_LLM:
        return 0.0
    if task.category == "noisy_observation":
        return 0.76
    if task.misleading_initial:
        return 0.80
    return 0.68


def _choose_probe(
    task: CognitiveLoopTask,
    *,
    arm: str,
    posteriors: Mapping[str, float],
    tried: Sequence[str],
    rng: random.Random,
) -> ProbeSpec:
    remaining = [probe for probe in task.probes if probe.action_id not in set(tried)] or list(task.probes)
    config = _arm_config(arm)
    if config["random_probe"]:
        return rng.choice(remaining)
    if not config["semantic_strictness"] and task.semantic_task:
        return sorted(remaining, key=lambda probe: (-probe.ordinary_rank, probe.action_id))[0]
    if not config["hypothesis_competition"]:
        top_id = _leading(posteriors)
        favored = [probe for probe in remaining if probe.supports_hypothesis_id == top_id]
        if favored:
            return sorted(favored, key=lambda probe: (-probe.ordinary_rank, probe.risk, probe.action_id))[0]
        return sorted(remaining, key=lambda probe: (-probe.ordinary_rank, probe.risk, probe.action_id))[0]
    if not config["discriminating_experiment"]:
        return sorted(remaining, key=lambda probe: (-probe.ordinary_rank, probe.risk, probe.action_id))[0]
    runner_ids = [item[0] for item in sorted(posteriors.items(), key=lambda item: (-float(item[1]), item[0]))[:2]]

    def score(probe: ProbeSpec) -> Tuple[float, float, str]:
        supports_contender = 1.0 if probe.supports_hypothesis_id in runner_ids else 0.0
        semantic_bonus = 0.08 if probe.semantic_sensitive else 0.0
        return (
            probe.expected_information_gain + supports_contender * 0.18 + semantic_bonus - probe.risk * 0.12,
            -probe.risk,
            probe.action_id,
        )

    return sorted(remaining, key=score, reverse=True)[0]


def _probe_support_target(task: CognitiveLoopTask, probe: ProbeSpec, *, useful_attempt_index: int) -> Tuple[str, bool]:
    if task.noisy_first_probe and "useful" in probe.action_id and useful_attempt_index <= 0:
        wrong = next(h.hypothesis_id for h in task.hypotheses if h.hypothesis_id != task.correct_hypothesis_id)
        return wrong, True
    return probe.supports_hypothesis_id, False


def _apply_probe_evidence(
    posteriors: Dict[str, float],
    *,
    support_target: str,
    posterior_update: bool,
) -> Dict[str, float]:
    if not posterior_update or not support_target:
        return dict(posteriors)
    updated: Dict[str, float] = {}
    for hypothesis_id, prior in posteriors.items():
        raw_delta = 0.58 if hypothesis_id == support_target else -0.48
        updated[hypothesis_id] = _revise_belief(prior, raw_delta)
    return updated


def _load_baseline_decisions(path: str | Path | None) -> Dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), dict) else payload
    return {str(key): str(value) for key, value in decisions.items() if str(key)}


def _build_baseline_llm_client(
    *,
    provider: str,
    base_url: str | None,
    model: str | None,
    timeout_sec: float,
) -> Any:
    from modules.llm.factory import build_llm_client

    client = build_llm_client(
        provider=provider,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
    )
    if client is None:
        raise ValueError(f"baseline provider is disabled: {provider}")
    return client


def _render_baseline_llm_prompt(task: CognitiveLoopTask) -> str:
    hypothesis_lines = [
        f"- {hyp.hypothesis_id}: label={hyp.label}, prior={round(_clamp01(hyp.prior), 4)}"
        for hyp in task.hypotheses
    ]
    probe_lines = [
        (
            f"- probe_{index}: {probe.function_name}"
            f"(x={probe.kwargs.get('x')}, y={probe.kwargs.get('y')}, "
            f"target_family={probe.kwargs.get('target_family')}, risk={round(float(probe.risk), 4)})"
        )
        for index, probe in enumerate(task.probes)
    ]
    allowed_ids = ", ".join(hyp.hypothesis_id for hyp in task.hypotheses)
    return "\n".join(
        [
            "You are the naked-model baseline in a controlled Cognitive OS benchmark.",
            "Commit to one hypothesis now. You cannot run probes, update posterior beliefs, or ask for observations.",
            "Do not copy this task text.",
            f"Task id: {task.task_id}",
            f"Task category: {task.category}",
            "Hypothesis options:",
            *hypothesis_lines,
            "Visible but unexecuted probe affordances:",
            *probe_lines,
            f"Allowed hypothesis_id values: {allowed_ids}",
            'Return exactly one compact JSON object: {"hypothesis_id":"<allowed id>","confidence":0.50,"reason":"short reason"}',
        ]
    )


def _initial_leading_for_task(task: CognitiveLoopTask) -> str:
    return _leading({hyp.hypothesis_id: _clamp01(hyp.prior, 0.5) for hyp in task.hypotheses})


def _normalize_baseline_choice(task: CognitiveLoopTask, payload: Mapping[str, Any]) -> Tuple[str, bool]:
    valid_ids = {hyp.hypothesis_id for hyp in task.hypotheses}
    label_to_id = {hyp.label: hyp.hypothesis_id for hyp in task.hypotheses}
    payloads: List[Mapping[str, Any]] = [payload]
    for key in ("response", "decision", "result", "output", "final", "response_schema"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payloads.append(nested)
    candidates = [
        item.get(key, "")
        for item in payloads
        for key in ("hypothesis_id", "selected_hypothesis_id", "selected_id", "choice", "answer")
    ]
    for candidate in candidates:
        selected = str(candidate or "").strip()
        if selected in valid_ids:
            return selected, True
        if selected in label_to_id:
            return label_to_id[selected], True
    return _initial_leading_for_task(task), False


def _json_object_from_text(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
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


def _complete_baseline_prompt(
    client: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str]:
    if hasattr(client, "complete"):
        raw_text = str(
            client.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt="Return one compact JSON object only. No markdown.",
                think=False,
            )
            or ""
        )
        return _json_object_from_text(raw_text), raw_text
    payload = client.complete_json(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        think=False,
    )
    normalized = payload if isinstance(payload, dict) else {}
    return normalized, json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _choice_from_raw_text(task: CognitiveLoopTask, raw_text: str) -> Tuple[str, bool]:
    text = str(raw_text or "")
    if '"hypotheses"' in text or '"available_but_unexecuted_probes"' in text:
        return _initial_leading_for_task(task), False
    valid_ids = {hyp.hypothesis_id for hyp in task.hypotheses}
    label_to_id = {hyp.label: hyp.hypothesis_id for hyp in task.hypotheses}
    patterns = (
        r'"(?:hypothesis_id|selected_hypothesis_id|selected_id|choice|answer)"\s*:\s*"([^"]+)"',
        r"\b(?:hypothesis_id|selected_hypothesis_id|selected_id|choice|answer)\b\s*(?:is|=|:)\s*[\"'`]?([A-Za-z0-9_:-]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            selected = match.group(1).strip()
            if selected in valid_ids:
                return selected, True
            if selected in label_to_id:
                return label_to_id[selected], True
    return _initial_leading_for_task(task), False


def collect_baseline_llm_decisions(
    tasks: Sequence[CognitiveLoopTask],
    *,
    provider: str = "ollama",
    base_url: str | None = None,
    model: str | None = None,
    timeout_sec: float = 30.0,
    max_tokens: int = 128,
    temperature: float = 0.0,
    fail_on_error: bool = True,
) -> Dict[str, Any]:
    client = _build_baseline_llm_client(
        provider=provider,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
    )
    decisions: Dict[str, str] = {}
    responses: List[Dict[str, Any]] = []
    for task in tasks:
        prompt = _render_baseline_llm_prompt(task)
        try:
            payload, raw_text = _complete_baseline_prompt(
                client,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            selected, valid = _normalize_baseline_choice(task, payload)
            if not valid:
                selected, valid = _choice_from_raw_text(task, raw_text)
            error = "" if valid else "invalid_or_missing_hypothesis_id"
        except Exception as exc:
            if fail_on_error:
                raise
            payload = {}
            raw_text = ""
            selected, valid = _initial_leading_for_task(task), False
            error = str(exc)
        decisions[task.task_id] = selected
        responses.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "selected_hypothesis_id": selected,
                "valid_model_choice": bool(valid),
                "fallback_used": not bool(valid),
                "confidence": _clamp01(payload.get("confidence"), 0.0) if isinstance(payload, dict) else 0.0,
                "reason": str(payload.get("reason", "") or "")[:500] if isinstance(payload, dict) else "",
                "error": error,
                "raw_response": raw_text[:2000],
            }
        )
    return {
        "schema_version": BASELINE_LLM_DECISIONS_VERSION,
        "provider": str(provider or ""),
        "base_url": str(base_url or getattr(client, "_base_url", "") or ""),
        "model": str(model or getattr(client, "_model", "") or getattr(client, "model", "") or ""),
        "task_count": len(tasks),
        "decisions": decisions,
        "responses": responses,
    }


def run_task_arm(
    task: CognitiveLoopTask,
    *,
    arm: str,
    seed: int = 17,
    baseline_decisions: Optional[Mapping[str, str]] = None,
) -> TaskRunResult:
    config = _arm_config(arm)
    rng = random.Random(f"{seed}:{task.task_id}:{arm}")
    initial_posteriors = {hyp.hypothesis_id: _clamp01(hyp.prior, 0.5) for hyp in task.hypotheses}
    posteriors = dict(initial_posteriors)
    initial_leading = _leading(posteriors)
    tried: List[str] = []
    trace: List[Dict[str, Any]] = []
    useful_experiments = 0
    experiment_count = 0
    semantic_mismatch_count = 0
    evidence_conversions = 0
    evidence_binding_errors = 0
    memory_resets = 0
    useful_attempts = 0
    committed = ""
    steps = 0

    if config["baseline_llm"] or config["fresh_baseline"]:
        committed = str((baseline_decisions or {}).get(task.task_id, "")) or initial_leading if config["baseline_llm"] else initial_leading
        steps = 1
        trace.append(
            {
                "step": 1,
                "event": "baseline_commit" if config["baseline_llm"] else "fresh_baseline_commit",
                "committed_hypothesis_id": committed,
                "uses_persistent_evidence": False,
            }
        )
    else:
        for step in range(1, task.max_steps + 1):
            steps = step
            top_id = _leading(posteriors)
            has_probe_evidence = bool(experiment_count and config["evidence"] and config["step9"] and config["memory"])
            enough_noise_evidence = (not task.noisy_first_probe) or useful_attempts >= 2
            if (
                config["step10"]
                and has_probe_evidence
                and enough_noise_evidence
                and posteriors[top_id] >= _commit_threshold(task, arm)
            ):
                committed = top_id
                trace.append(
                    {
                        "step": step,
                        "event": "commit",
                        "committed_hypothesis_id": committed,
                        "posterior": posteriors[top_id],
                        "formal_step10_commit": True,
                    }
                )
                break
            probe = _choose_probe(
                task,
                arm=arm,
                posteriors=posteriors,
                tried=tried if config["memory"] else [],
                rng=rng,
            )
            if config["memory"]:
                tried.append(probe.action_id)
            action = probe.to_action()
            semantic_key = action_semantic_signature_key(action)
            false_semantic = bool(
                task.semantic_task
                and not config["semantic_strictness"]
                and "useful" not in probe.action_id
            )
            if false_semantic:
                semantic_mismatch_count += 1
            if "useful" in probe.action_id:
                support_target, misleading = _probe_support_target(task, probe, useful_attempt_index=useful_attempts)
                useful_attempts += 1
            else:
                support_target, misleading = _probe_support_target(task, probe, useful_attempt_index=useful_attempts)
            before = dict(posteriors)
            evidence_converted = bool(config["evidence"] and config["step9"] and support_target)
            bound_support_target = support_target if evidence_converted else ""
            if evidence_converted and (config["shuffled_evidence"] or config["wrong_binding"]):
                wrong = next(h.hypothesis_id for h in task.hypotheses if h.hypothesis_id != support_target)
                bound_support_target = wrong
                evidence_binding_errors += 1
            if evidence_converted:
                evidence_conversions += 1
            posteriors = _apply_probe_evidence(
                posteriors,
                support_target=bound_support_target,
                posterior_update=config["posterior_update"],
            )
            experiment_count += 1
            useful = bool(
                probe.discriminating
                and support_target == task.correct_hypothesis_id
                and bound_support_target == task.correct_hypothesis_id
                and not false_semantic
            )
            if useful:
                useful_experiments += 1
            trace.append(
                {
                    "step": step,
                    "event": "probe",
                    "action_id": probe.action_id,
                    "semantic_signature": semantic_key,
                    "support_target": support_target,
                    "bound_support_target": bound_support_target,
                    "step9_evidence_converted": evidence_converted,
                    "step10_formal_commit_available": bool(config["step10"]),
                    "evidence_binding_error": bool(evidence_converted and bound_support_target != support_target),
                    "misleading": misleading,
                    "useful": useful,
                    "semantic_mismatch_false_experiment": false_semantic,
                    "posterior_before": before,
                    "posterior_after": dict(posteriors),
                }
            )
            if not config["memory"]:
                memory_resets += 1
                posteriors = dict(initial_posteriors)
                tried = []
                trace.append({"step": step, "event": "memory_reset", "posterior_restored": dict(posteriors)})
        if not committed:
            if config["step10"]:
                committed = _leading(posteriors)
                event_name = "forced_final_commit"
            else:
                committed = initial_leading
                event_name = "formal_step10_missing_prior_commit"
            trace.append(
                {
                    "step": steps,
                    "event": event_name,
                    "committed_hypothesis_id": committed,
                    "formal_step10_commit": bool(config["step10"]),
                }
            )

    final_leading = _leading(posteriors)
    correct = task.correct_hypothesis_id
    success = committed == correct
    rejected_correct = posteriors.get(correct, 0.0) <= 0.15
    final_confidence = _clamp01(posteriors.get(final_leading, 0.0))
    final_leading_correct = final_leading == correct
    calibration_error = abs(final_confidence - (1.0 if final_leading_correct else 0.0))
    flipped_to_correct = initial_leading != correct and final_leading == correct
    return TaskRunResult(
        task_id=task.task_id,
        category=task.category,
        arm=arm,
        success=success,
        steps=steps,
        committed_hypothesis_id=committed,
        correct_hypothesis_id=correct,
        wrong_commit=bool(committed and committed != correct),
        false_rejection=rejected_correct,
        initial_leading_hypothesis_id=initial_leading,
        final_leading_hypothesis_id=final_leading,
        posterior_leading_correct=final_leading_correct,
        flipped_to_correct=flipped_to_correct,
        useful_experiment_count=useful_experiments,
        experiment_count=experiment_count,
        semantic_mismatch_false_experiment_count=semantic_mismatch_count,
        recovery_after_wrong_hypothesis=bool(task.misleading_initial and success and flipped_to_correct),
        posterior_calibration_error=round(calibration_error, 6),
        evidence_conversion_count=evidence_conversions,
        evidence_binding_error_count=evidence_binding_errors,
        formal_commit_count=1 if committed and config["step10"] else 0,
        memory_reset_count=memory_resets,
        trace=trace,
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def summarize_arm_results(results: Sequence[TaskRunResult]) -> Dict[str, Any]:
    rows = list(results or [])
    total = len(rows)
    successes = sum(1 for row in rows if row.success)
    committed = sum(1 for row in rows if row.committed_hypothesis_id)
    wrong_commits = sum(1 for row in rows if row.wrong_commit)
    false_rejections = sum(1 for row in rows if row.false_rejection)
    leading_correct = sum(1 for row in rows if row.posterior_leading_correct)
    experiment_count = sum(row.experiment_count for row in rows)
    useful_experiments = sum(row.useful_experiment_count for row in rows)
    evidence_conversions = sum(row.evidence_conversion_count for row in rows)
    evidence_binding_errors = sum(row.evidence_binding_error_count for row in rows)
    formal_commits = sum(row.formal_commit_count for row in rows)
    memory_resets = sum(row.memory_reset_count for row in rows)
    semantic_tasks = [row for row in rows if row.category == "semantic_action"]
    semantic_mismatches = sum(row.semantic_mismatch_false_experiment_count for row in semantic_tasks)
    misleading_rows = [row for row in rows if row.initial_leading_hypothesis_id != row.correct_hypothesis_id]
    recovered = sum(1 for row in misleading_rows if row.recovery_after_wrong_hypothesis)
    steps_to_success = [row.steps for row in rows if row.success]
    calibration_errors = [row.posterior_calibration_error for row in rows]
    return {
        "task_count": total,
        "success_rate": round(_safe_rate(successes, total), 6),
        "mean_steps_to_success": round(sum(steps_to_success) / max(1, len(steps_to_success)), 6) if steps_to_success else None,
        "wrong_commit_rate": round(_safe_rate(wrong_commits, committed), 6),
        "false_rejection_rate": round(_safe_rate(false_rejections, total), 6),
        "hypothesis_flip_accuracy": round(_safe_rate(recovered, len(misleading_rows)), 6),
        "experiment_usefulness_rate": round(_safe_rate(useful_experiments, experiment_count), 6),
        "evidence_conversion_rate": round(_safe_rate(evidence_conversions, experiment_count), 6),
        "evidence_binding_error_rate": round(_safe_rate(evidence_binding_errors, max(1, evidence_conversions)), 6),
        "formal_commit_rate": round(_safe_rate(formal_commits, committed), 6),
        "memory_reset_rate": round(_safe_rate(memory_resets, experiment_count), 6),
        "posterior_calibration_error": round(sum(calibration_errors) / max(1, len(calibration_errors)), 6),
        "recovery_after_wrong_hypothesis": round(_safe_rate(recovered, len(misleading_rows)), 6),
        "posterior_leading_hypothesis_accuracy": round(_safe_rate(leading_correct, total), 6),
        "semantic_mismatch_false_experiment_rate": round(_safe_rate(semantic_mismatches, max(1, len(semantic_tasks))), 6),
    }


def evaluate_passing_lines(arm_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    full = dict(arm_metrics.get(ARM_FULL, {}))
    baseline = dict(arm_metrics.get(ARM_BASELINE_LLM, {}))
    no_posterior = dict(arm_metrics.get(ARM_NO_POSTERIOR, {}))
    no_experiment = dict(arm_metrics.get(ARM_NO_EXPERIMENT, {}))

    def metric(arm: Mapping[str, Any], name: str) -> float:
        value = arm.get(name, 0.0)
        return float(value or 0.0)

    checks = {
        "full_beats_baseline_llm_by_20pct": {
            "passed": metric(full, "success_rate") >= metric(baseline, "success_rate") + PASSING_THRESHOLDS["full_success_min_margin_vs_baseline_llm"],
            "actual_margin": round(metric(full, "success_rate") - metric(baseline, "success_rate"), 6),
            "required_margin": PASSING_THRESHOLDS["full_success_min_margin_vs_baseline_llm"],
        },
        "full_beats_no_posterior_by_10pct": {
            "passed": metric(full, "success_rate") >= metric(no_posterior, "success_rate") + PASSING_THRESHOLDS["full_success_min_margin_vs_no_posterior"],
            "actual_margin": round(metric(full, "success_rate") - metric(no_posterior, "success_rate"), 6),
            "required_margin": PASSING_THRESHOLDS["full_success_min_margin_vs_no_posterior"],
        },
        "full_beats_no_experiment_by_10pct": {
            "passed": metric(full, "success_rate") >= metric(no_experiment, "success_rate") + PASSING_THRESHOLDS["full_success_min_margin_vs_no_experiment"],
            "actual_margin": round(metric(full, "success_rate") - metric(no_experiment, "success_rate"), 6),
            "required_margin": PASSING_THRESHOLDS["full_success_min_margin_vs_no_experiment"],
        },
        "wrong_commit_rate_at_or_below_10pct": {
            "passed": metric(full, "wrong_commit_rate") <= PASSING_THRESHOLDS["wrong_commit_rate_max"],
            "actual": metric(full, "wrong_commit_rate"),
            "maximum": PASSING_THRESHOLDS["wrong_commit_rate_max"],
        },
        "false_rejection_rate_at_or_below_15pct": {
            "passed": metric(full, "false_rejection_rate") <= PASSING_THRESHOLDS["false_rejection_rate_max"],
            "actual": metric(full, "false_rejection_rate"),
            "maximum": PASSING_THRESHOLDS["false_rejection_rate_max"],
        },
        "posterior_leading_accuracy_at_or_above_75pct": {
            "passed": metric(full, "posterior_leading_hypothesis_accuracy") >= PASSING_THRESHOLDS["posterior_leading_hypothesis_accuracy_min"],
            "actual": metric(full, "posterior_leading_hypothesis_accuracy"),
            "minimum": PASSING_THRESHOLDS["posterior_leading_hypothesis_accuracy_min"],
        },
        "semantic_mismatch_false_experiment_rate_zero": {
            "passed": metric(full, "semantic_mismatch_false_experiment_rate") <= PASSING_THRESHOLDS["semantic_mismatch_false_experiment_rate_max"],
            "actual": metric(full, "semantic_mismatch_false_experiment_rate"),
            "maximum": PASSING_THRESHOLDS["semantic_mismatch_false_experiment_rate_max"],
        },
        "full_evidence_binding_error_rate_zero": {
            "passed": metric(full, "evidence_binding_error_rate") <= PASSING_THRESHOLDS["full_evidence_binding_error_rate_max"],
            "actual": metric(full, "evidence_binding_error_rate"),
            "maximum": PASSING_THRESHOLDS["full_evidence_binding_error_rate_max"],
        },
        "full_formal_commit_rate_is_complete": {
            "passed": metric(full, "formal_commit_rate") >= PASSING_THRESHOLDS["full_formal_commit_rate_min"],
            "actual": metric(full, "formal_commit_rate"),
            "minimum": PASSING_THRESHOLDS["full_formal_commit_rate_min"],
        },
    }
    for arm in ANTI_CHEAT_ARMS:
        if arm not in arm_metrics:
            continue
        checks[f"full_beats_{arm}_by_anti_cheat_margin"] = {
            "passed": metric(full, "success_rate") >= metric(dict(arm_metrics.get(arm, {})), "success_rate") + PASSING_THRESHOLDS["full_success_min_margin_vs_anti_cheat"],
            "actual_margin": round(metric(full, "success_rate") - metric(dict(arm_metrics.get(arm, {})), "success_rate"), 6),
            "required_margin": PASSING_THRESHOLDS["full_success_min_margin_vs_anti_cheat"],
        }
    return {
        "thresholds": dict(PASSING_THRESHOLDS),
        "checks": checks,
        "anti_cheat_arms": list(ANTI_CHEAT_ARMS),
        "passed": all(bool(row.get("passed", False)) for row in checks.values()),
    }


def run_cognitive_loop_ablation(
    *,
    task_count: int = 25,
    seed: int = 17,
    arms: Sequence[str] = DEFAULT_ARMS,
    baseline_decisions_path: str | Path | None = None,
    baseline_decisions: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    tasks = build_cognitive_loop_benchmark_tasks(task_count=task_count, seed=seed)
    loaded_baseline_decisions = _load_baseline_decisions(baseline_decisions_path)
    merged_baseline_decisions = dict(loaded_baseline_decisions)
    if baseline_decisions:
        merged_baseline_decisions.update({str(key): str(value) for key, value in baseline_decisions.items()})
    arm_results: Dict[str, List[TaskRunResult]] = {}
    for arm in list(arms or DEFAULT_ARMS):
        if arm not in ARM_DESCRIPTIONS:
            raise ValueError(f"unknown ablation arm: {arm}")
        arm_results[arm] = [
            run_task_arm(task, arm=arm, seed=seed, baseline_decisions=merged_baseline_decisions)
            for task in tasks
        ]
    metrics = {
        arm: summarize_arm_results(rows)
        for arm, rows in arm_results.items()
    }
    pass_fail = evaluate_passing_lines(metrics)
    return {
        "schema_version": COGNITIVE_LOOP_ABLATION_VERSION,
        "task_count": len(tasks),
        "seed": int(seed),
        "arms": {arm: ARM_DESCRIPTIONS[arm] for arm in arm_results.keys()},
        "core_ablation_arms": list(CORE_ABLATION_ARMS),
        "anti_cheat_arms": [arm for arm in ANTI_CHEAT_ARMS if arm in arm_results],
        "task_categories": list(TASK_CATEGORIES),
        "metrics": metrics,
        "passing_lines": pass_fail,
        "status": "PASSED" if pass_fail["passed"] else "FAILED",
        "baseline_llm_mode": "external_decisions" if merged_baseline_decisions else "deterministic_naked_model_baseline",
        "baseline_decision_count": len(merged_baseline_decisions),
        "task_results": {
            arm: [row.to_dict() for row in rows]
            for arm, rows in arm_results.items()
        },
    }


def render_cognitive_loop_ablation_report(report: Mapping[str, Any]) -> str:
    metrics = dict(report.get("metrics", {}) or {})
    lines = [
        "Cognitive loop ablation benchmark",
        f"schema: {report.get('schema_version', COGNITIVE_LOOP_ABLATION_VERSION)}",
        f"tasks: {int(report.get('task_count', 0) or 0)}",
        f"status: {report.get('status', 'UNKNOWN')}",
        "",
        f"{'arm':<34} {'success':>8} {'steps':>8} {'wrong':>8} {'flip':>8} {'useful':>8} {'calib':>8}",
        "-" * 90,
    ]
    for arm in DEFAULT_ARMS:
        row = dict(metrics.get(arm, {}) or {})
        if not row:
            continue
        steps = row.get("mean_steps_to_success")
        lines.append(
            f"{arm:<34} "
            f"{float(row.get('success_rate', 0.0) or 0.0):>8.2f} "
            f"{(float(steps) if steps is not None else 0.0):>8.2f} "
            f"{float(row.get('wrong_commit_rate', 0.0) or 0.0):>8.2f} "
            f"{float(row.get('hypothesis_flip_accuracy', 0.0) or 0.0):>8.2f} "
            f"{float(row.get('experiment_usefulness_rate', 0.0) or 0.0):>8.2f} "
            f"{float(row.get('posterior_calibration_error', 0.0) or 0.0):>8.2f}"
        )
    lines.append("")
    lines.append("Passing lines")
    checks = dict(dict(report.get("passing_lines", {}) or {}).get("checks", {}) or {})
    for name, payload in checks.items():
        row = dict(payload or {})
        mark = "PASS" if row.get("passed", False) else "FAIL"
        details = {key: value for key, value in row.items() if key != "passed"}
        lines.append(f"{mark:<4} {name}: {json.dumps(details, sort_keys=True)}")
    return "\n".join(lines)


def write_ablation_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    return output


def write_baseline_llm_decisions(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    return output
