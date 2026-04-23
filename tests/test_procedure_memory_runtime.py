from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.procedure_memory_runtime import (
    load_procedure_objects,
    maybe_commit_procedure_chain,
    procedure_observed_functions,
    procedure_task_signature,
    procedure_text_tokens,
)


class _RawNovelAPI:
    raw = {
        "visible_functions": [{"name": "ACTION1"}, "ACTION2"],
        "discovered_functions": ["ACTION3"],
        "available_functions": ["ACTION4"],
    }


class _Store:
    def __init__(self, rows):
        self.rows = list(rows)

    def retrieve(self, *, sort_by: str, limit: int):
        assert sort_by == "confidence"
        assert limit == 400
        return list(self.rows)


def test_procedure_signature_tokens_and_observed_functions_are_stable() -> None:
    obs = {
        "task": "Find Door",
        "goal": "Open Door",
        "instruction": "Use the red key",
        "query": "Door puzzle",
        "perception": {"goal": "Reach exit", "summary": "red key visible"},
        "world_state": {"task_family": "grid", "phase": "locked", "active_functions": ["ACTION5"]},
        "novel_api": _RawNovelAPI(),
        "available_functions": ["ACTION6"],
        "backend_functions": {"ACTION7": {}},
    }

    assert procedure_task_signature(obs) == (
        "find door|open door|use the red key|door puzzle|reach exit|red key visible|grid|locked"
    )
    assert {"red", "key", "visible"} <= procedure_text_tokens("red-key visible!")
    assert procedure_observed_functions(obs) == [
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION6",
        "ACTION5",
        "ACTION7",
    ]


def test_load_procedure_objects_ranks_filters_and_deduplicates() -> None:
    obs = {
        "task": "open red door",
        "novel_api": {"visible_functions": ["ACTION1", "ACTION2"]},
    }
    exact = {
        "memory_type": "procedure_chain",
        "confidence": 0.7,
        "content_hash": "exact",
        "content": {
            "task_signature": "open red door",
            "action_chain": ["ACTION1", "ACTION2"],
        },
    }
    duplicate = {
        "memory_type": "procedure_chain",
        "confidence": 0.95,
        "content_hash": "exact",
        "content": {
            "task_signature": "open red door",
            "action_chain": ["ACTION1"],
        },
    }
    latent = {
        "memory_type": "procedure_chain",
        "confidence": 0.2,
        "content_hash": "latent",
        "content": {
            "task_signature": "unrelated",
            "action_chain": ["ACTION9"],
            "latent_mechanism_key": "door_unlock",
            "mechanism_roles": ["key", "door"],
            "source_domain": "grid",
        },
    }
    unrelated = {
        "memory_type": "procedure_chain",
        "confidence": 0.9,
        "content_hash": "unrelated",
        "content": {
            "task_signature": "paint wall",
            "action_chain": ["ACTION8"],
        },
    }
    non_procedure = {"memory_type": "semantic", "confidence": 1.0, "content": {}}

    selected = load_procedure_objects(_Store([unrelated, latent, duplicate, exact, non_procedure]), obs)

    assert selected[0] is exact
    assert duplicate not in selected
    assert latent in selected
    assert unrelated not in selected
    assert non_procedure not in selected


class _CommitStore:
    def __init__(self):
        self.rows = {
            "first": {"content": {"tool_args": {"function_name": "ACTION1"}}},
            "skip": {"memory_type": "procedure_chain", "content": {"function_name": "OLD"}},
            "second": {"content": {"function_name": "ACTION2"}},
            "dup": {"function_name": "ACTION2"},
        }

    def get(self, object_id: str):
        return self.rows.get(object_id)


class _Validator:
    def __init__(self, decision: str = "accept"):
        self.decision = decision
        self.proposals = []

    def validate(self, proposal):
        self.proposals.append(proposal)
        return SimpleNamespace(decision=self.decision)


class _Committer:
    def __init__(self):
        self.payloads = []

    def commit(self, payload):
        self.payloads.append(payload)
        return ["procedure-1"]


def test_maybe_commit_procedure_chain_builds_validated_post_commit_procedure() -> None:
    store = _CommitStore()
    validator = _Validator()
    committer = _Committer()
    proposal_log = []

    maybe_commit_procedure_chain(
        committed_ids=["first", "skip", "second", "dup"],
        obs_before={"task": "Open Door"},
        reward=2.0,
        shared_store=store,
        validator=validator,
        committer=committer,
        procedure_proposal_log=proposal_log,
        episode=3,
        tick=4,
        reject_decision="reject",
    )

    proposal = validator.proposals[0]
    assert proposal["confidence"] == 0.55 + 2.0 * 0.15
    assert proposal["content"]["task_signature"] == "open door"
    assert proposal["content"]["action_chain"] == ["ACTION1", "ACTION2"]
    assert proposal["content_hash"] == "open door|ACTION1->ACTION2"
    assert proposal_log == [
        {
            "episode": 3,
            "tick": 4,
            "procedure_object_id": "procedure-1",
            "task_signature": "open door",
            "action_chain": ["ACTION1", "ACTION2"],
        }
    ]


def test_maybe_commit_procedure_chain_respects_reward_and_reject_gate() -> None:
    store = _CommitStore()
    rejected = _Validator(decision="reject")
    committer = _Committer()
    proposal_log = []

    maybe_commit_procedure_chain(
        committed_ids=["first", "second"],
        obs_before={"task": "Open Door"},
        reward=0.0,
        shared_store=store,
        validator=rejected,
        committer=committer,
        procedure_proposal_log=proposal_log,
        episode=1,
        tick=1,
        reject_decision="reject",
    )
    assert rejected.proposals == []

    maybe_commit_procedure_chain(
        committed_ids=["first", "second"],
        obs_before={"task": "Open Door"},
        reward=1.0,
        shared_store=store,
        validator=rejected,
        committer=committer,
        procedure_proposal_log=proposal_log,
        episode=1,
        tick=1,
        reject_decision="reject",
    )
    assert rejected.proposals
    assert committer.payloads == []
    assert proposal_log == []
