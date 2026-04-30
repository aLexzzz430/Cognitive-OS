"""
modules/recovery/llm_interface.py

A12: C2 错误恢复 — LLM 接口（policy 级能力不整块改）

C2 是 policy 级能力，不只是分类文本。

LLM 能力：
- 错误类型建议
- 恢复路径候选
- 故障摘要

不交给 LLM：
- 恢复路由
- 正式状态更新（StateManager.update_state）
- Step 9/10 记账

接口模式：propose_with_llm() + validate_in_core() + commit_via_step10()
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.state.manager import StateManager
    from modules.governance.object_store import ObjectStore

from dataclasses import dataclass, field
from enum import Enum

from modules.llm.capabilities import (
    RECOVERY_ERROR_DIAGNOSIS,
    RECOVERY_FAILURE_SUMMARY,
    RECOVERY_GATE_ADVICE,
    RECOVERY_PLAN_SYNTHESIS,
)
from modules.llm.gateway import ensure_llm_gateway
from modules.llm.json_adaptor import normalize_llm_output


class ErrorType(Enum):
    """错误类型枚举（与 C2ErrorInjector 对齐）"""
    SKILL_APPLICABILITY = "skill_applicability"
    HYPOTHESIS_CONFLICT = "hypothesis_conflict"
    REPRESENTATION_INCONSISTENCY = "representation_inconsistency"
    RECOVERY_FAILURE = "recovery_failure"
    UNKNOWN = "unknown"


class RecoveryType(Enum):
    """恢复路由类型（与 C2ErrorInjector 对齐）"""
    REQUEST_PROBE = "request_probe"
    REQUEST_REPLAN = "request_replan"
    RAISE_REVIEW = "raise_review"
    FALLBACK_REVIEW = "fallback_review"


@dataclass
class ErrorDiagnosis:
    """错误诊断结果"""
    error_type: ErrorType
    confidence: float  # LLM 对诊断的置信度
    description: str   # 错误描述
    affected_components: List[str]  # 受影响的组件
    root_cause_hypothesis: str  # 根本原因假设
    llm_reasoning: str = ""  # LLM 推理过程（供审计）


@dataclass
class RecoveryPath:
    """恢复路径候选"""
    path_id: str
    recovery_type: RecoveryType
    description: str
    estimated_success_probability: float
    required_components: List[str]  # 需要激活的组件
    estimated_cost: float  # 估计的资源消耗
    llm_advice: str = ""  # LLM 的建议理由


class LLMErrorRecoveryInterface:
    """
    C2 错误恢复：LLM 做辅助，不接管 policy。

    使用方式：
    1. CoreMainLoop 检测到错误/异常
    2. 调用 diagnose_error() 获取 LLM 诊断
    3. 调用 suggest_recovery_paths() 获取恢复路径候选
    4. CoreMainLoop 路由恢复（validate_in_core）
    5. 恢复后调用 summarize_failure() 生成故障摘要

    不交给 LLM：
    - 恢复路由选择（CoreMainLoop 决策）
    - StateManager.update_state（正式状态更新）
    - Step 9/10 记账
    """

    LLM_ROUTE_NAME = "recovery"
    LLM_CAPABILITY_NAMESPACE = "recovery"

    def __init__(self, state_manager, object_store=None, llm_client=None):
        """
        Args:
            state_manager: StateManager instance
            object_store: ObjectStore instance (for Step 10 commit)
            llm_client: LLM API client. If None, uses rule-based fallback.
        """
        self._state_mgr = state_manager
        self._obj_store = object_store
        self._llm_gateway = ensure_llm_gateway(
            llm_client,
            route_name=self.LLM_ROUTE_NAME,
            capability_prefix=self.LLM_CAPABILITY_NAMESPACE,
        )
        self._llm = self._llm_gateway

    def _llm_available(self) -> bool:
        return self._llm_gateway is not None and bool(self._llm_gateway.is_available())

    def _request_text(self, capability: str, prompt: str, **kwargs: Any) -> str:
        if self._llm_gateway is None:
            return ""
        return self._llm_gateway.request_text(capability, prompt, **kwargs)

    # ─────────────────────────────────────────────────
    # 1. Diagnose Error
    # ─────────────────────────────────────────────────

    def diagnose_error(self, error_context: Dict[str, Any]) -> ErrorDiagnosis:
        """
        LLM 诊断错误类型。

        输入：error_context 包含：
        - failed_action: 失败的动作
        - error_result: 错误结果
        - hypotheses_state: 当前假设状态
        - recent_commits: 最近的提交
        - state_snapshot: 状态快照

        输出：ErrorDiagnosis（不直接写入状态，由 CoreMainLoop 路由）
        """
        if not self._llm_available():
            return self._rule_based_diagnose(error_context)

        failed_action = error_context.get('failed_action', {})
        action_fn = failed_action.get('payload', {}).get('tool_args', {}).get('function_name', 'unknown')
        error_result = error_context.get('error_result', {})
        err_msg = str(error_result.get('error', error_result))[:200]

        hypotheses_active = error_context.get('hypotheses_active', 0)
        hypotheses_confirmed = error_context.get('hypotheses_confirmed', 0)
        recent_commits = error_context.get('recent_commits', 0)

        prompt = f"""You are diagnosing an error in an agent system.

Failed action: {action_fn}
Error message: {err_msg}
Active hypotheses: {hypotheses_active}, Confirmed: {hypotheses_confirmed}
Recent commits: {recent_commits}

Possible error types:
1. SKILL_APPLICABILITY: skill was called but doesn't apply in current context
2. HYPOTHESIS_CONFLICT: two hypotheses are competing for the same action
3. REPRESENTATION_INCONSISTENCY: representation card contradicts actual behavior
4. RECOVERY_FAILURE: previous recovery attempt also failed
5. UNKNOWN: none of the above

Diagnose the error type and explain your reasoning.

Return JSON:
{{"error_type": "SKILL_APPLICABILITY/HYPOTHESIS_CONFLICT/...", "confidence": 0.X, "description": "...", "affected_components": ["..."], "root_cause_hypothesis": "...", "llm_reasoning": "..."}}

Return ONLY the JSON, nothing else."""

        response = self._request_text(RECOVERY_ERROR_DIAGNOSIS, prompt)
        try:
            raw = normalize_llm_output(
                response,
                output_kind="recovery_error_diagnosis",
                expected_type="dict",
            ).parsed_dict()
            return ErrorDiagnosis(
                error_type=ErrorType(raw.get('error_type', 'UNKNOWN')),
                confidence=raw.get('confidence', 0.5),
                description=raw.get('description', ''),
                affected_components=raw.get('affected_components', []),
                root_cause_hypothesis=raw.get('root_cause_hypothesis', ''),
                llm_reasoning=raw.get('llm_reasoning', ''),
            )
        except Exception:
            return self._rule_based_diagnose(error_context)

    # ─────────────────────────────────────────────────
    # 2. Suggest Recovery Paths
    # ─────────────────────────────────────────────────

    def suggest_recovery_paths(
        self,
        error_diagnosis: ErrorDiagnosis,
        current_state: Dict[str, Any],
        top_k: int = 3,
    ) -> List[RecoveryPath]:
        """
        LLM 建议恢复路径候选。

        输出：RecoveryPath 列表（由 CoreMainLoop 路由选择）
        """
        if not self._llm_available():
            return self._rule_based_recovery(error_diagnosis)

        state_summary = f"active_hyps={current_state.get('active_hypotheses', 0)}, confirmed={current_state.get('hypotheses_confirmed', 0)}, entropy={current_state.get('entropy', 0):.3f}"

        prompt = f"""You are suggesting recovery paths for an agent system error.

Error diagnosis:
- Type: {error_diagnosis.error_type.value}
- Description: {error_diagnosis.description}
- Root cause hypothesis: {error_diagnosis.root_cause_hypothesis}
- Affected components: {error_diagnosis.affected_components}

Current state: {state_summary}

Recovery types available:
1. REQUEST_PROBE: request a discriminating probe to resolve uncertainty
2. REQUEST_REPLAN: request a replan (abandon current plan, regenerate)
3. RAISE_REVIEW: escalate to review (pause current execution, review state)
4. FALLBACK_REVIEW: fallback to review (soft pause, allow retry)

For each recovery path, estimate:
- estimated_success_probability: 0.0-1.0
- required_components: what needs to be activated
- estimated_cost: resource cost (0.0-1.0, higher = more expensive)

Generate {top_k} recovery paths ranked by estimated success / cost ratio.

Return JSON list:
[
  {{"recovery_type": "REQUEST_PROBE/REQUEST_REPLAN/RAISE_REVIEW/FALLBACK_REVIEW", "description": "...", "estimated_success_probability": 0.X, "required_components": ["..."], "estimated_cost": 0.X}},
  ...
]

Return ONLY the JSON list, nothing else."""

        response = self._request_text(RECOVERY_PLAN_SYNTHESIS, prompt)
        try:
            import time
            raw_paths = normalize_llm_output(
                response,
                output_kind="recovery_plan_synthesis",
                expected_type="list",
            ).parsed_list()
            paths = []
            for i, raw in enumerate(raw_paths[:top_k]):
                if not isinstance(raw, dict):
                    continue
                paths.append(RecoveryPath(
                    path_id=f"recovery_{int(time.time()*1000)%100000}_{i}",
                    recovery_type=RecoveryType(raw.get('recovery_type', 'FALLBACK_REVIEW')),
                    description=raw.get('description', ''),
                    estimated_success_probability=raw.get('estimated_success_probability', 0.5),
                    required_components=raw.get('required_components', []),
                    estimated_cost=raw.get('estimated_cost', 0.5),
                    llm_advice=f"Based on {error_diagnosis.error_type.value} diagnosis",
                ))
            return paths
        except Exception:
            return self._rule_based_recovery(error_diagnosis)

    # ─────────────────────────────────────────────────
    # 3. Summarize Failure
    # ─────────────────────────────────────────────────

    def summarize_failure(
        self,
        failure_log: Dict[str, Any],
        error_diagnosis: ErrorDiagnosis,
        recovery_attempted: RecoveryType = None,
        recovery_success: bool = None,
    ) -> str:
        """
        LLM 生成故障摘要（供审计用）。

        在恢复尝试完成后调用，生成故障总结。
        不影响系统状态，仅供记录和审计。
        """
        if not self._llm_available():
            return f"Error {error_diagnosis.error_type.value}: {error_diagnosis.description} — recovery {'succeeded' if recovery_success else 'failed'}"

        err_type = error_diagnosis.error_type.value
        err_desc = error_diagnosis.description
        recovery_str = recovery_attempted.value if recovery_attempted else 'none'
        outcome = 'SUCCEEDED' if recovery_success else 'FAILED'

        prompt = f"""You are summarizing a failure and recovery attempt in an agent system.

Error:
- Type: {err_type}
- Description: {err_desc}
- Root cause: {error_diagnosis.root_cause_hypothesis}

Recovery attempted: {recovery_str}
Recovery outcome: {outcome}

Write a concise 2-3 sentence failure summary that captures:
1. What went wrong
2. What was tried to recover
3. What the outcome was

This summary will be used for audit logs.

Return ONLY the summary, nothing else."""

        summary = self._request_text(RECOVERY_FAILURE_SUMMARY, prompt).strip()
        return summary or f"Error {error_diagnosis.error_type.value}: {error_diagnosis.description} — recovery {'succeeded' if recovery_success else 'failed'}"

    # ─────────────────────────────────────────────────
    # 4. Should trigger recovery? (decision advisory)
    # ─────────────────────────────────────────────────

    def should_trigger_recovery(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 判断是否应该触发恢复流程（decision advisory）。

        原则：决定由 CoreMainLoop 做出，LLM 只提供建议。
        """
        if not self._llm_available():
            # Rule-based fallback: trigger on error_result present
            has_error = bool(error_context.get('error_result'))
            return {
                'should_recover': has_error,
                'urgency': 'high' if has_error else 'none',
                'reason': 'error_result present' if has_error else 'no error',
            }

        failed_action = error_context.get('failed_action', {})
        action_fn = failed_action.get('payload', {}).get('tool_args', {}).get('function_name', 'unknown')
        err_msg = str(error_context.get('error_result', {}).get('error', ''))[:150]

        prompt = f"""An action failed in an agent system:

Action: {action_fn}
Error: {err_msg}

Should we trigger a recovery flow? Consider:
- High urgency: clear error that can be recovered
- Medium urgency: ambiguous failure, might need review
- Low urgency: minor issue, continue without recovery
- None: not an error, normal operation

Answer with JSON:
{{"should_recover": true/false, "urgency": "high/medium/low/none", "reason": "..."}}

Return ONLY the JSON, nothing else."""

        response = self._request_text(RECOVERY_GATE_ADVICE, prompt)
        try:
            parsed = normalize_llm_output(
                response,
                output_kind="recovery_gate_advice",
                expected_type="dict",
            ).parsed_dict()
            return parsed if parsed else {'should_recover': True, 'urgency': 'medium', 'reason': 'parse failed'}
        except Exception:
            return {'should_recover': True, 'urgency': 'medium', 'reason': 'parse failed'}

    # ─────────────────────────────────────────────────
    # 5. Rule-based fallback
    # ─────────────────────────────────────────────────

    def _rule_based_diagnose(self, error_context: Dict[str, Any]) -> ErrorDiagnosis:
        """没有 LLM 时的基于规则的诊断"""
        err_result = error_context.get('error_result', {})
        err_msg = str(err_result.get('error', ''))

        # Simple heuristic rules
        if 'skill' in err_msg.lower():
            error_type = ErrorType.SKILL_APPLICABILITY
        elif 'hypothesis' in err_msg.lower() or 'conflict' in err_msg.lower():
            error_type = ErrorType.HYPOTHESIS_CONFLICT
        elif 'representation' in err_msg.lower() or 'inconsistency' in err_msg.lower():
            error_type = ErrorType.REPRESENTATION_INCONSISTENCY
        elif 'recovery' in err_msg.lower() or 'failed' in err_msg.lower():
            error_type = ErrorType.RECOVERY_FAILURE
        else:
            error_type = ErrorType.UNKNOWN

        return ErrorDiagnosis(
            error_type=error_type,
            confidence=0.5,
            description=f"Rule-based diagnosis: {err_msg[:100]}",
            affected_components=['CoreMainLoop'],
            root_cause_hypothesis=f"Error type: {error_type.value}",
            llm_reasoning='rule_based_fallback',
        )

    def _rule_based_recovery(self, error_diagnosis: ErrorDiagnosis) -> List[RecoveryPath]:
        """没有 LLM 时的基于规则的恢复路径"""
        import time

        # Simple mapping from error type to recovery
        type_to_recovery = {
            ErrorType.SKILL_APPLICABILITY: RecoveryType.REQUEST_REPLAN,
            ErrorType.HYPOTHESIS_CONFLICT: RecoveryType.REQUEST_PROBE,
            ErrorType.REPRESENTATION_INCONSISTENCY: RecoveryType.RAISE_REVIEW,
            ErrorType.RECOVERY_FAILURE: RecoveryType.FALLBACK_REVIEW,
            ErrorType.UNKNOWN: RecoveryType.FALLBACK_REVIEW,
        }

        recovery_type = type_to_recovery.get(error_diagnosis.error_type, RecoveryType.FALLBACK_REVIEW)

        return [RecoveryPath(
            path_id=f"recovery_rb_{int(time.time()*1000)%100000}",
            recovery_type=recovery_type,
            description=f"Rule-based recovery for {error_diagnosis.error_type.value}",
            estimated_success_probability=0.5,
            required_components=[error_diagnosis.error_type.value],
            estimated_cost=0.5,
            llm_advice='rule_based_fallback',
        )]
