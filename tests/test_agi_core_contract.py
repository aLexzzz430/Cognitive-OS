from __future__ import annotations

from core.cognition import (
    ActionIntent,
    CognitiveCycleFrame,
    CognitiveExperiment,
    CognitiveGoal,
    CognitiveHypothesis,
    CognitiveOutcome,
    CognitiveSituation,
    validate_domain_neutral_contract,
)


def test_agi_core_contract_serializes_domain_neutral_cycle_frame() -> None:
    frame = CognitiveCycleFrame(
        frame_id="frame-1",
        goal=CognitiveGoal(
            goal_id="goal-1",
            statement="Reduce uncertainty about the current environment.",
            constraints=["do not perform side effects without permission"],
        ),
        situation=CognitiveSituation(
            situation_id="sit-1",
            observations=[{"kind": "anomaly", "summary": "Repeated outcome mismatch"}],
            salient_entities=["entity:a"],
            self_state_refs=["self:budget"],
            world_state_refs=["world:object_graph"],
        ),
        hypotheses=[
            CognitiveHypothesis(
                hypothesis_id="hyp-1",
                claim="The mismatch is caused by an outdated belief.",
                predictions=["A targeted observation should contradict the old belief."],
                falsifiers=["No contradiction is observed."],
            )
        ],
        experiments=[
            CognitiveExperiment(
                experiment_id="exp-1",
                question="Which belief explains the anomaly?",
                discriminates_between=["hyp-1", "hyp-2"],
                expected_observations={"hyp-1": "contradiction observed"},
                information_gain=0.7,
            )
        ],
        action_intents=[
            ActionIntent(
                intent_id="intent-1",
                verb="observe",
                target="entity:a",
                expected_effect="collect disambiguating evidence",
                capability_required=["read"],
            )
        ],
        outcomes=[
            CognitiveOutcome(
                outcome_id="outcome-1",
                action_intent_id="intent-1",
                observations=[{"kind": "evidence", "summary": "Contradiction observed"}],
                verified=True,
                success=True,
                evidence_refs=["evidence:1"],
            )
        ],
    )

    payload = frame.to_dict()

    assert payload["schema_version"] == "conos.agi_core_contract/v1"
    assert payload["goal"]["goal_id"] == "goal-1"
    assert payload["hypotheses"][0]["hypothesis_id"] == "hyp-1"
    assert validate_domain_neutral_contract(payload) == []


def test_agi_core_contract_rejects_domain_specific_leakage() -> None:
    payload = {
        "frame_id": "bad",
        "goal": {"statement": "Run pytest and apply_patch in local_machine mirror"},
    }

    violations = validate_domain_neutral_contract(payload)

    assert "local_machine" in violations
    assert "mirror" in violations
    assert "pytest" in violations
    assert "apply_patch" in violations
