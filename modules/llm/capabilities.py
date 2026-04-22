from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): raw for key, raw in dict(value).items()}


@dataclass(frozen=True)
class LLMCapabilitySpec:
    full_name: str
    route_name: str
    required_capabilities: tuple[str, ...] = ()
    schema_name: str = ""
    fallback_route: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        if "." not in self.full_name:
            return ""
        return self.full_name.split(".", 1)[0]

    @property
    def short_name(self) -> str:
        if "." not in self.full_name:
            return self.full_name
        return self.full_name.split(".", 1)[1]

    def to_policy(self, *, policy_source: str = "default_exact") -> Dict[str, Any]:
        payload = {
            "route_name": self.route_name,
            "required_capabilities": list(self.required_capabilities),
            "schema_name": self.schema_name,
            "fallback_route": self.fallback_route,
            "metadata": {
                **_dict_or_empty(self.metadata),
                "policy_source": str(policy_source or "default_exact"),
            },
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", [], {})
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capability_name(value: Any) -> str:
    if isinstance(value, LLMCapabilitySpec):
        return str(value.full_name or "").strip()
    return str(value or "").strip()


def capability_spec(value: Any, *, namespace: str = "") -> LLMCapabilitySpec | None:
    return LLM_CAPABILITY_CATALOG.get(canonical_capability_name(value, namespace=namespace))


def canonical_capability_name(value: Any, *, fallback: str = "", namespace: str = "") -> str:
    if isinstance(value, Mapping):
        payload = dict(value)
        return canonical_capability_name(
            payload.get("capability", "")
            or payload.get("capability_name", "")
            or payload.get("full_name", "")
            or fallback,
            fallback=fallback,
            namespace=str(payload.get("namespace", "") or payload.get("capability_namespace", "") or namespace),
        )
    if isinstance(value, LLMCapabilitySpec):
        return str(value.full_name or "").strip()

    raw_name = str(value or "").strip() or str(fallback or "").strip()
    if not raw_name:
        return ""
    if raw_name in LLM_CAPABILITY_CATALOG:
        return raw_name

    namespace_hint = str(namespace or "").strip().strip(".")
    if namespace_hint and "." not in raw_name:
        namespaced = f"{namespace_hint}.{raw_name}"
        if namespaced in LLM_CAPABILITY_CATALOG:
            return namespaced

    if "." not in raw_name:
        matches = [spec.full_name for spec in LLM_CAPABILITY_SPECS if spec.short_name == raw_name]
        if len(matches) == 1:
            return matches[0]

    return raw_name


GENERAL_REASONING = LLMCapabilitySpec(
    full_name="general",
    route_name="general",
    required_capabilities=("reasoning",),
    description="General reasoning fallback.",
)

RETRIEVAL_QUERY_REWRITE = LLMCapabilitySpec(
    full_name="retrieval.query_rewrite",
    route_name="retrieval",
    required_capabilities=("retrieval", "grounding"),
    description="Rewrite retrieval queries against the current context.",
)
RETRIEVAL_CANDIDATE_RERANK = LLMCapabilitySpec(
    full_name="retrieval.candidate_rerank",
    route_name="retrieval",
    required_capabilities=("retrieval", "grounding"),
    description="Re-rank retrieved episodic candidates.",
)
RETRIEVAL_EPISODE_SUMMARIZATION = LLMCapabilitySpec(
    full_name="retrieval.episode_summarization",
    route_name="retrieval",
    required_capabilities=("retrieval", "grounding", "reasoning"),
    description="Summarize episodes for later retrieval.",
)
RETRIEVAL_GATE_ADVICE = LLMCapabilitySpec(
    full_name="retrieval.gate_advice",
    route_name="retrieval",
    required_capabilities=("retrieval", "verification"),
    description="Advise whether retrieval should be opened.",
)

REASONING_HYPOTHESIS_GENERATION = LLMCapabilitySpec(
    full_name="reasoning.hypothesis_generation",
    route_name="hypothesis",
    required_capabilities=("reasoning", "hypothesis", "uncertainty"),
    description="Generate competing hypothesis candidates.",
)
REASONING_HYPOTHESIS_COMPETITOR_EXPANSION = LLMCapabilitySpec(
    full_name="reasoning.hypothesis_competitor_expansion",
    route_name="hypothesis",
    required_capabilities=("reasoning", "hypothesis"),
    description="Expand a hypothesis's competitor set.",
)
REASONING_PROBE_DESIGN = LLMCapabilitySpec(
    full_name="reasoning.probe_design",
    route_name="probe",
    required_capabilities=("reasoning", "probe", "verification"),
    description="Design discriminating probes between competing hypotheses.",
)
REASONING_PROBE_OUTCOME_PREDICTION = LLMCapabilitySpec(
    full_name="reasoning.probe_outcome_prediction",
    route_name="probe",
    required_capabilities=("reasoning", "probe", "verification"),
    description="Predict expected outcomes for a proposed probe.",
)
REASONING_INFORMATION_GAIN_EXPLANATION = LLMCapabilitySpec(
    full_name="reasoning.information_gain_explanation",
    route_name="probe",
    required_capabilities=("reasoning", "verification"),
    description="Explain why a probe is informative.",
)
REASONING_PROBE_URGENCY_ADVICE = LLMCapabilitySpec(
    full_name="reasoning.probe_urgency_advice",
    route_name="probe",
    required_capabilities=("reasoning", "verification"),
    description="Advise whether a probe should run now.",
)

RECOVERY_ERROR_DIAGNOSIS = LLMCapabilitySpec(
    full_name="recovery.error_diagnosis",
    route_name="recovery",
    required_capabilities=("recovery", "reasoning"),
    description="Diagnose runtime failures and recovery classes.",
)
RECOVERY_PLAN_SYNTHESIS = LLMCapabilitySpec(
    full_name="recovery.plan_synthesis",
    route_name="recovery",
    required_capabilities=("recovery", "reasoning"),
    description="Synthesize ranked recovery plans.",
)
RECOVERY_FAILURE_SUMMARY = LLMCapabilitySpec(
    full_name="recovery.failure_summary",
    route_name="recovery",
    required_capabilities=("recovery", "reasoning"),
    description="Summarize failures for later recovery loops.",
)
RECOVERY_GATE_ADVICE = LLMCapabilitySpec(
    full_name="recovery.recovery_gate_advice",
    route_name="recovery",
    required_capabilities=("recovery", "verification"),
    description="Advise whether recovery paths should open.",
)

SKILL_CONTEXT_COMPRESSION = LLMCapabilitySpec(
    full_name="skill.context_compression",
    route_name="skill",
    required_capabilities=("instruction_following", "rewriting"),
    description="Compress current context for skill generation.",
)
SKILL_CANDIDATE_GENERATION = LLMCapabilitySpec(
    full_name="skill.candidate_generation",
    route_name="skill",
    required_capabilities=("instruction_following", "rewriting"),
    description="Generate candidate skill rewrites.",
)
SKILL_PARAMETER_DRAFTING = LLMCapabilitySpec(
    full_name="skill.parameter_drafting",
    route_name="skill",
    required_capabilities=("instruction_following", "rewriting"),
    description="Draft skill parameters from context.",
)
SKILL_BACKEND_SELECTION = LLMCapabilitySpec(
    full_name="skill.backend_selection",
    route_name="skill",
    required_capabilities=("instruction_following", "rewriting"),
    description="Select the best symbolic skill backend.",
)

REPRESENTATION_CARD_PROPOSAL = LLMCapabilitySpec(
    full_name="representation.card_proposal",
    route_name="representation",
    required_capabilities=("representation", "creative"),
    description="Propose representation cards from trajectories.",
)
REPRESENTATION_FALSE_ABSTRACTION_CHECK = LLMCapabilitySpec(
    full_name="representation.false_abstraction_check",
    route_name="representation",
    required_capabilities=("representation", "verification"),
    description="Detect false abstractions in proposed cards.",
)

STRUCTURED_OUTPUT_ARC_GRID = LLMCapabilitySpec(
    full_name="structured_output.arc_grid",
    route_name="structured_answer",
    required_capabilities=("structured_output", "reasoning"),
    schema_name="arc_grid",
    description="Return structured ARC grid output.",
)
STRUCTURED_OUTPUT_ACTION_KWARGS = LLMCapabilitySpec(
    full_name="structured_output.action_kwargs",
    route_name="structured_answer",
    required_capabilities=("structured_output", "reasoning"),
    schema_name="tool_kwargs",
    description="Draft structured kwargs for a visible action.",
)

ANALYSIS_SHADOW_REVIEW = LLMCapabilitySpec(
    full_name="analysis.shadow_review",
    route_name="shadow",
    required_capabilities=("analysis", "reasoning"),
    description="Shadow review of a model decision.",
)
ANALYSIS_VERIFICATION_REVIEW = LLMCapabilitySpec(
    full_name="analysis.verification_review",
    route_name="analyst",
    required_capabilities=("analysis", "verification", "reasoning"),
    description="Verification-oriented analysis review.",
)

LLM_CAPABILITY_SPECS = (
    GENERAL_REASONING,
    RETRIEVAL_QUERY_REWRITE,
    RETRIEVAL_CANDIDATE_RERANK,
    RETRIEVAL_EPISODE_SUMMARIZATION,
    RETRIEVAL_GATE_ADVICE,
    REASONING_HYPOTHESIS_GENERATION,
    REASONING_HYPOTHESIS_COMPETITOR_EXPANSION,
    REASONING_PROBE_DESIGN,
    REASONING_PROBE_OUTCOME_PREDICTION,
    REASONING_INFORMATION_GAIN_EXPLANATION,
    REASONING_PROBE_URGENCY_ADVICE,
    RECOVERY_ERROR_DIAGNOSIS,
    RECOVERY_PLAN_SYNTHESIS,
    RECOVERY_FAILURE_SUMMARY,
    RECOVERY_GATE_ADVICE,
    SKILL_CONTEXT_COMPRESSION,
    SKILL_CANDIDATE_GENERATION,
    SKILL_PARAMETER_DRAFTING,
    SKILL_BACKEND_SELECTION,
    REPRESENTATION_CARD_PROPOSAL,
    REPRESENTATION_FALSE_ABSTRACTION_CHECK,
    STRUCTURED_OUTPUT_ARC_GRID,
    STRUCTURED_OUTPUT_ACTION_KWARGS,
    ANALYSIS_SHADOW_REVIEW,
    ANALYSIS_VERIFICATION_REVIEW,
)

LLM_CAPABILITY_CATALOG: Dict[str, LLMCapabilitySpec] = {
    spec.full_name: spec for spec in LLM_CAPABILITY_SPECS
}

LLM_DEFAULT_CAPABILITY_POLICIES: Dict[str, Dict[str, Any]] = {
    name: spec.to_policy(policy_source="default_exact")
    for name, spec in LLM_CAPABILITY_CATALOG.items()
}

LLM_DEFAULT_CAPABILITY_PREFIX_POLICIES: Dict[str, Dict[str, Any]] = {
    "retrieval.*": {
        "route_name": "retrieval",
        "required_capabilities": ["retrieval"],
        "metadata": {"policy_source": "default_prefix"},
    },
    "reasoning.*": {
        "route_name": "deliberation",
        "required_capabilities": ["reasoning"],
        "metadata": {"policy_source": "default_prefix"},
    },
    "recovery.*": {
        "route_name": "recovery",
        "required_capabilities": ["recovery"],
        "metadata": {"policy_source": "default_prefix"},
    },
    "skill.*": {
        "route_name": "skill",
        "required_capabilities": ["instruction_following"],
        "metadata": {"policy_source": "default_prefix"},
    },
    "representation.*": {
        "route_name": "representation",
        "required_capabilities": ["representation"],
        "metadata": {"policy_source": "default_prefix"},
    },
    "structured_output.*": {
        "route_name": "structured_answer",
        "required_capabilities": ["structured_output"],
        "metadata": {"policy_source": "default_prefix"},
    },
    "analysis.*": {
        "route_name": "analyst",
        "required_capabilities": ["analysis"],
        "metadata": {"policy_source": "default_prefix"},
    },
}
