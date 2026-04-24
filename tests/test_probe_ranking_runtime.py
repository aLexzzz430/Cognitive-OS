from types import SimpleNamespace

from pytest import approx

from core.orchestration.probe_ranking_runtime import (
    annotate_probe_ranking,
    estimate_probe_disambiguation_gain,
    extract_uncertain_high_impact_beliefs,
    rank_probe_candidates_by_prediction,
)


class _Probe:
    def __init__(self, target_function, test_params=None, hypothesis_a="", hypothesis_b=""):
        self.target_function = target_function
        self.test_params = dict(test_params or {})
        self.hypothesis_a = hypothesis_a
        self.hypothesis_b = hypothesis_b


class _PredictionEngine:
    def __init__(self, gains):
        self.gains = dict(gains)
        self.calls = []

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        action = kwargs["action"]
        tool_args = action["payload"]["tool_args"]
        function_name = tool_args["function_name"]
        return SimpleNamespace(
            information_gain=SimpleNamespace(value=self.gains.get(function_name, 0.0))
        )


class _Loop:
    def __init__(self, *, active=True):
        self._active = active
        self._episode = 2
        self._tick = 5
        self._prediction_engine = _PredictionEngine(
            {
                "inspect_door": 0.2,
                "wait": 0.5,
            }
        )
        self._hypotheses = SimpleNamespace(get_active=lambda: ["hypothesis"])
        self._plan_state = SimpleNamespace(
            get_plan_summary=lambda: {"plan": "summary"},
            get_intent_for_step=lambda: "inspect",
        )
        self._episode_trace = [{"tick": index} for index in range(8)]

    def _prediction_runtime_active(self):
        return self._active

    def _build_action_id(self, action):
        action["_action_id"] = f"action:{action['payload']['tool_args']['function_name']}"

    def _get_policy_profile(self):
        return {"policy": "profile"}

    def _build_recovery_prediction_context(self):
        return {"pending_replan": False}


def test_extract_uncertain_high_impact_beliefs_filters_and_prioritizes():
    beliefs = extract_uncertain_high_impact_beliefs(
        {
            "beliefs": {
                "door_color": {"confidence": 0.4},
                "button_state": {"confidence": 0.5},
                "already_known": {"confidence": 0.9},
                "bad_payload": "ignore",
            },
            "high_value_beliefs": [
                {"variable": "door_color", "impact_scope": "planner+decision"},
                {"variable": "button_state", "impact_scope": "local"},
            ],
        }
    )

    assert beliefs == [
        {
            "variable": "door_color",
            "confidence": 0.4,
            "impact": 1.0,
            "uncertainty": 0.6,
            "priority": 0.6,
        },
        {
            "variable": "button_state",
            "confidence": 0.5,
            "impact": 0.7,
            "uncertainty": 0.5,
            "priority": 0.35,
        },
    ]
    assert extract_uncertain_high_impact_beliefs(None) == []


def test_estimate_probe_disambiguation_gain_uses_params_and_hypothesis_text():
    probe = _Probe(
        "inspect_door",
        {"door_color": "red"},
        hypothesis_a="door_color determines reward",
    )

    gain = estimate_probe_disambiguation_gain(
        probe,
        [
            {"variable": "door_color", "priority": 0.6},
            {"variable": "button_state", "priority": 0.35},
        ],
    )

    assert gain == approx(0.6)
    assert estimate_probe_disambiguation_gain(probe, []) == 0.0


def test_annotate_probe_ranking_sets_score_metadata():
    probe = _Probe("inspect_door")

    annotate_probe_ranking(probe, {"score": 0.42, "reason": "test"})

    assert probe.ranking_details == {"score": 0.42, "reason": "test"}
    assert probe.expected_information_gain == 0.42


def test_rank_probe_candidates_by_prediction_combines_prediction_and_belief_uncertainty():
    loop = _Loop(active=True)
    door_probe = _Probe("inspect_door", {"door_color": "red"})
    wait_probe = _Probe("wait", {})
    frame = SimpleNamespace(
        world_model_summary={
            "beliefs": {"door_color": {"confidence": 0.4}},
            "high_value_beliefs": [
                {"variable": "door_color", "impact_scope": "planner+decision"}
            ],
        },
        self_model_summary={"self": "summary"},
    )

    ranked = rank_probe_candidates_by_prediction(
        loop,
        [wait_probe, door_probe],
        obs_before={"obs": True},
        surfaced=[{"surface": True}],
        frame=frame,
    )

    assert ranked == [door_probe, wait_probe]
    assert door_probe.ranking_details == {
        "prediction_info_gain": 0.2,
        "disambiguation_gain": 0.6,
        "uncertainty_focus": 0.2,
        "score": 0.36,
        "target_beliefs": ["door_color"],
    }
    assert wait_probe.ranking_details == {
        "prediction_info_gain": 0.5,
        "disambiguation_gain": 0.0,
        "uncertainty_focus": 0.2,
        "score": 0.3,
        "target_beliefs": ["door_color"],
    }
    assert [call["action"]["_action_id"] for call in loop._prediction_engine.calls] == [
        "action:wait",
        "action:inspect_door",
    ]
    assert loop._prediction_engine.calls[0]["recent_trace"] == [{"tick": 3}, {"tick": 4}, {"tick": 5}, {"tick": 6}, {"tick": 7}]
    assert loop._prediction_engine.calls[0]["self_model_summary"] == {"self": "summary"}


def test_rank_probe_candidates_by_prediction_noops_when_prediction_runtime_inactive():
    loop = _Loop(active=False)
    probes = [_Probe("inspect_door")]

    ranked = rank_probe_candidates_by_prediction(
        loop,
        probes,
        obs_before={},
        surfaced=[],
        frame=None,
    )

    assert ranked is probes
    assert loop._prediction_engine.calls == []
    assert not hasattr(probes[0], "ranking_details")
