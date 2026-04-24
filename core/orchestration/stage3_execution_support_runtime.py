from __future__ import annotations

from typing import Any, Dict, Set

from core.orchestration.action_execution_resolver import SelectedActionExecutionResolver
from core.orchestration.retrieval_runtime_helpers import RetrievalRuntimeHelpers


def resolve_action_for_execution(loop: Any, action_to_use: Dict[str, Any], obs_before: Dict[str, Any]) -> Dict[str, Any]:
    resolver = getattr(loop, "_selected_action_execution_resolver", None)
    if resolver is None:
        resolver = SelectedActionExecutionResolver()
        setattr(loop, "_selected_action_execution_resolver", resolver)
    resolved = resolver.resolve(action_to_use, obs_before)
    fn_name = loop._extract_action_function_name(resolved, default="wait")
    if fn_name == "wait":
        decorator = getattr(getattr(loop, "_world", None), "decorate_candidate_action", None)
        if callable(decorator):
            decorated = decorator(resolved)
            if isinstance(decorated, dict):
                return decorated
    return resolved


def collect_executable_function_names(obs_before: Dict[str, Any]) -> Set[str]:
    known: Set[str] = set()
    api_raw = obs_before.get("novel_api", {}) if isinstance(obs_before, dict) else {}
    if hasattr(api_raw, "raw"):
        api_raw = api_raw.raw
    if isinstance(api_raw, dict):
        for key in ("visible_functions", "discovered_functions"):
            values = api_raw.get(key, [])
            if isinstance(values, list):
                known.update(value for value in values if isinstance(value, str) and value)
    signatures = obs_before.get("function_signatures", {}) if isinstance(obs_before, dict) else {}
    if isinstance(signatures, dict):
        known.update(key for key in signatures.keys() if isinstance(key, str) and key)
    return known


def is_trackable_executable_function(function_name: str, executable_functions: Set[str]) -> bool:
    if not isinstance(function_name, str) or not function_name:
        return False
    lowered = function_name.lower()
    if lowered.startswith("hyp_") or lowered.startswith("obj_"):
        return False
    if not executable_functions:
        return False
    return function_name in executable_functions


def record_memory_consumption_proof(loop: Any, action_to_use: Dict[str, Any], query: Any, result: Dict[str, Any]) -> None:
    influence = RetrievalRuntimeHelpers.build_memory_consumption_proof(
        tick=loop._tick,
        episode=loop._episode,
        action_to_use=action_to_use,
        query=query,
        result_reward=loop._get_reward(result),
        plan_target=loop._plan_state.get_target_function_for_step(),
        active_beliefs=[belief.variable_name for belief in loop._belief_ledger.get_active_beliefs()],
    )
    loop._governance_log.append(
        {
            "tick": loop._tick,
            "episode": loop._episode,
            "entry": "memory_consumption_proof",
            **influence,
        }
    )
