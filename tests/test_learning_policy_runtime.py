from types import SimpleNamespace

from pytest import approx

from core.orchestration.learning_policy_runtime import (
    clamp_learning_signal,
    is_learning_verification_function,
    learning_merge_ordered_lists,
    learning_resource_band,
    learning_world_model_competition_profile,
    merge_learned_failure_strategy_profile,
    refresh_learning_policy_snapshot,
)


class _SharedStore:
    def __init__(self, rows):
        self.rows = list(rows)
        self.limits = []

    def iter_objects(self, *, limit):
        self.limits.append(limit)
        return list(self.rows)


class _ReliabilityTracker:
    def __init__(self):
        self.synced = []

    def synchronize_failure_preference_learning(self, policy):
        self.synced.append(policy)


def test_refresh_learning_policy_snapshot_filters_and_syncs_failure_policy():
    rows = [
        {
            "memory_type": "learning_update",
            "confidence": 1.0,
            "content": {
                "type": "learning_update",
                "update_type": "selector_bias",
                "key": "inspect",
                "delta": 0.5,
                "confidence": 1.0,
            },
        },
        {
            "memory_type": "learning_update",
            "status": "invalidated",
            "content": {
                "type": "learning_update",
                "update_type": "selector_bias",
                "key": "move",
                "delta": 0.5,
            },
        },
        {"memory_type": "semantic", "content": {"type": "learning_update"}},
    ]
    reliability_tracker = _ReliabilityTracker()
    loop = SimpleNamespace(
        _shared_store=_SharedStore(rows),
        _reliability_tracker=reliability_tracker,
        _learning_policy_snapshot={},
    )

    refresh_learning_policy_snapshot(loop)

    assert loop._shared_store.limits == [500]
    assert loop._learning_policy_snapshot["selector_bias"] == {"inspect": 0.5}
    assert "move" not in loop._learning_policy_snapshot["selector_bias"]
    assert reliability_tracker.synced == [{}]


def test_learning_scalar_and_list_helpers_preserve_existing_contracts():
    assert clamp_learning_signal("2.0", 0.0, 1.0) == 1.0
    assert clamp_learning_signal("bad", 0.0, 1.0, default=0.4) == 0.4
    assert learning_merge_ordered_lists(["inspect", "move"], ["move", "verify"], "ignored") == [
        "inspect",
        "move",
        "verify",
    ]
    assert is_learning_verification_function("inspect_door") is True
    assert is_learning_verification_function("MOVE") is False


def test_learning_resource_band_resolves_explicit_pressure_and_self_model_fallback():
    assert learning_resource_band({"resource_band": "tight"}) == "tight"
    assert learning_resource_band({"resource_pressure": "critical"}) == "tight"
    assert learning_resource_band({"resource_pressure": "low"}) == "normal"
    assert learning_resource_band({"self_model_summary": {"resource_tightness": "high"}}) == "tight"
    assert learning_resource_band(None) == "normal"


def test_learning_world_model_competition_profile_extracts_dominant_branch_and_pressure():
    profile = learning_world_model_competition_profile(
        {
            "world_model_control": {
                "required_probes": ["inspect_door", "verify_key"],
                "dominant_branch_id": "branch-b",
                "transition_confidence": 0.4,
                "control_trust": 0.4,
                "hidden_drift_score": 0.6,
                "hidden_uncertainty_score": 0.7,
                "state_shift_risk": 0.8,
                "latent_branches": [
                    {
                        "branch_id": "branch-a",
                        "confidence": 0.9,
                        "anchor_functions": ["noop"],
                    },
                    {
                        "branch_id": "branch-b",
                        "confidence": 0.2,
                        "anchor_functions": ["move", "inspect_door"],
                        "anchored_functions": ["verify_key"],
                        "risky_functions": ["jump"],
                    },
                ],
            }
        },
        candidate_function_universe=["inspect_door", "verify_key", "jump"],
    )

    assert profile["required_probes"] == ["inspect_door", "verify_key"]
    assert profile["probe_pressure"] == approx(2 / 3)
    assert profile["probe_pressure_active"] is True
    assert profile["latent_instability"] == approx(0.722)
    assert profile["dominant_branch_id"] == "branch-b"
    assert profile["dominant_anchor_functions"] == ["inspect_door"]
    assert profile["dominant_risky_functions"] == ["jump"]


def test_merge_learned_failure_strategy_profile_combines_learning_and_competition():
    merged = merge_learned_failure_strategy_profile(
        {
            "strategy_mode_hint": "balanced",
            "branch_budget_hint": 1,
            "preferred_verification_functions": ["old_probe"],
            "blocked_action_classes": ["old_blocked"],
        },
        {
            "strategy_mode_hint": "verify",
            "verification_budget_hint": 2,
            "safe_fallback_class": "wait",
            "preferred_verification_functions": ["learned_probe"],
            "preferred_fallback_functions": ["fallback"],
            "blocked_action_classes": ["learned_blocked"],
            "required_probes": ["learned_required"],
            "dominant_branch_id": "learned-branch",
            "source_action": "bad_action",
        },
        competition={
            "required_probes": ["required_probe"],
            "probe_pressure_active": True,
            "probe_pressure": 0.7,
            "latent_instability": 0.6,
            "dominant_anchor_functions": ["inspect_door", "move"],
            "dominant_risky_functions": ["jump"],
            "dominant_branch_id": "branch-b",
        },
    )

    assert merged["strategy_mode_hint"] == "verify"
    assert merged["branch_budget_hint"] == 2
    assert merged["verification_budget_hint"] == 2
    assert merged["safe_fallback_class"] == "wait"
    assert merged["preferred_verification_functions"] == [
        "required_probe",
        "inspect_door",
        "learned_probe",
        "old_probe",
    ]
    assert merged["preferred_fallback_functions"] == ["fallback", "inspect_door", "move"]
    assert merged["blocked_action_classes"] == ["learned_blocked", "jump", "old_blocked"]
    assert merged["required_probes"] == ["learned_required", "required_probe"]
    assert merged["dominant_branch_id"] == "learned-branch"
    assert merged["source_action"] == "bad_action"
