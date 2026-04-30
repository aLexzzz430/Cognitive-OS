from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import conos_cli
from core.task_discovery.creative import CreativeTaskGenerator
from core.task_discovery import TaskDiscoveryEngine
from core.task_discovery.detectors import DiscoveryContext
from core.task_discovery.models import GoalLedger, TaskCandidate


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_task_discovery_generates_scored_double_queue_from_real_signals(tmp_path: Path) -> None:
    active_goals = tmp_path / "active_goals.json"
    evidence = tmp_path / "evidence_ledger.jsonl"
    feedback = tmp_path / "user_feedback_log.jsonl"
    repo_scan = tmp_path / "repo_scan_summary.json"
    hypotheses = tmp_path / "hypothesis_registry.json"
    output_dir = tmp_path / "out"

    _write_json(
        active_goals,
        {
            "north_star": "Con OS 要成为本地优先、证据治理、可持续运行的通用智能系统",
            "active_goals": [
                {
                    "goal_id": "local_model_success",
                    "description": "local model task_success >= 0.70",
                    "metric": "task_success",
                    "current": 0.35,
                    "target": 0.70,
                }
            ],
            "constraints": ["不准未经审核改主 repo"],
            "open_gaps": [{"gap_id": "open-1", "description": "任务搜索能力弱"}],
        },
    )
    _write_jsonl(
        evidence,
        [
            {
                "evidence_id": "ev-timeout",
                "event_type": "timeout",
                "message": "Qwen3:8B timed out after Step 7 with no valid response",
            },
            {
                "evidence_id": "ev-loop",
                "status": "needs_human_review",
                "reason": "PATCH_PROPOSAL_NOT_GENERATED because diff_too_large",
            },
        ],
    )
    _write_jsonl(
        feedback,
        [
            {
                "feedback_id": "fb-1",
                "message": "现在它经常超时，而且无法闭环。",
            },
            {
                "feedback_id": "fb-2",
                "message": "mirror_exec 直接写很长命令不对，应该拆成原子动作。",
            },
        ],
    )
    _write_json(
        repo_scan,
        {
            "todo_count": 9,
            "fixme_count": 2,
            "large_files": ["core/main_loop.py"],
            "opportunities": ["已有日志但没有统一报告"],
        },
    )
    _write_json(
        hypotheses,
        {
            "hypotheses": [
                {
                    "hypothesis_id": "h-token-budget",
                    "summary": "后期失败主要由 thinking token 吞掉输出预算造成",
                    "status": "unverified",
                }
            ]
        },
    )

    engine = TaskDiscoveryEngine()
    result = engine.discover_from_paths(
        active_goals_path=active_goals,
        evidence_ledger_path=evidence,
        user_feedback_log_path=feedback,
        repo_scan_summary_path=repo_scan,
        hypothesis_registry_path=hypotheses,
    )
    outputs = engine.write_outputs(result, output_dir)

    sources = {candidate.source for candidate in result.candidates}
    assert {"failure_residue", "goal_gap", "user_feedback", "code_health", "hypothesis", "opportunity"} <= sources
    assert result.task_queue
    assert all(candidate.priority >= 0.65 for candidate in result.task_queue)
    assert any(candidate.status == "deferred" and candidate.source == "code_health" for candidate in result.candidates)
    assert any(candidate.metadata.get("low_evidence_limited_to_investigation") for candidate in result.candidates)
    assert result.report["top_candidate"]["source"] in sources
    assert Path(outputs["task_candidates"]).exists()
    assert Path(outputs["task_queue"]).exists()

    queued_rows = [json.loads(line) for line in Path(outputs["task_queue"]).read_text(encoding="utf-8").splitlines()]
    assert queued_rows
    assert queued_rows[0]["schema_version"] == "conos.task_discovery/v1"
    assert queued_rows[0]["allowed_actions"]
    assert "forbidden_actions" in queued_rows[0]


def test_task_discovery_scores_and_gates_risky_tasks() -> None:
    risky = TaskCandidate(
        task_id="",
        source="failure_residue",
        observation="credential write path failure",
        gap="sync-back may touch credentials",
        proposed_task="Investigate credential boundary before sync-back",
        expected_value=0.9,
        goal_alignment=0.9,
        evidence_strength=0.9,
        feasibility=0.9,
        risk=0.7,
        cost=0.1,
        reversibility=0.9,
        distraction_penalty=0.0,
        success_condition="risk classified",
        allowed_actions=["read_logs"],
        forbidden_actions=["sync_back"],
    )
    engine = TaskDiscoveryEngine()
    context = DiscoveryContext(
        goal_ledger=GoalLedger(),
        evidence_records=[],
        user_feedback_records=[],
        run_trace_records=[],
        repo_scan_summary={},
        hypothesis_records=[],
    )
    result = engine.discover(context)
    prepared = engine._prepare_candidate(risky)

    assert result.candidates == []
    assert prepared.priority >= 0.65
    assert prepared.status == "needs_approval"
    assert prepared.requires_human_approval is True


def test_conos_discover_tasks_cli_writes_candidate_and_queue_files(tmp_path: Path, capsys) -> None:
    active_goals = tmp_path / "active_goals.json"
    evidence = tmp_path / "evidence.jsonl"
    output_dir = tmp_path / "task_discovery"
    _write_json(
        active_goals,
        {
            "active_goals": [
                {
                    "goal_id": "close_loop",
                    "description": "increase closure rate",
                    "metric": "closure_rate",
                    "current": 0.2,
                    "target": 0.8,
                }
            ]
        },
    )
    _write_jsonl(evidence, [{"evidence_id": "ev-1", "message": "verifier failed after run_test"}])

    code = conos_cli.main(
        [
            "discover-tasks",
            "--active-goals",
            str(active_goals),
            "--evidence-ledger",
            str(evidence),
            "--user-feedback-log",
            str(tmp_path / "missing_feedback.jsonl"),
            "--run-traces",
            str(tmp_path / "missing_runs"),
            "--repo-scan-summary",
            str(tmp_path / "missing_scan.json"),
            "--hypothesis-registry",
            str(tmp_path / "missing_hypotheses.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] >= 2
    assert payload["queued_count"] >= 1
    assert Path(payload["outputs"]["task_candidates"]).exists()
    assert Path(payload["outputs"]["task_queue"]).exists()


def test_goal_gap_detector_treats_zero_current_as_real_gap(tmp_path: Path) -> None:
    active_goals = tmp_path / "active_goals.json"
    _write_json(
        active_goals,
        {
            "active_goals": [
                {
                    "goal_id": "accepted_patch_rate",
                    "description": "真实开放任务应能产生 verifier-gated 最小补丁",
                    "metric": "accepted_patch_rate",
                    "current": 0.0,
                    "target": 0.7,
                }
            ]
        },
    )

    result = TaskDiscoveryEngine().discover_from_paths(active_goals_path=active_goals)

    assert any(candidate.source == "goal_gap" for candidate in result.candidates)
    top = result.candidates[0]
    assert top.metadata["goal"]["current"] == 0.0
    assert top.status == "queued"


def test_task_discovery_turns_self_model_outcomes_into_goal_pressure_and_skill_candidates(tmp_path: Path) -> None:
    evidence = tmp_path / "learning_evidence.jsonl"
    _write_jsonl(
        evidence,
        [
            {
                "event_id": "learn-fail-1",
                "event_type": "outcome_model_update",
                "data": {
                    "action": "file_read",
                    "outcome": "failure",
                    "verified": False,
                },
            },
            {
                "event_id": "learn-state-1",
                "self_summary": {
                    "capability_estimate": {
                        "file_read": {
                            "attempts": 3,
                            "successes": 1,
                            "failures": 2,
                            "verified_successes": 0,
                            "reliability": 0.3333,
                        }
                    }
                },
            },
            {
                "event_id": "learn-success-1",
                "event_type": "outcome_model_update",
                "data": {
                    "action": "run_test",
                    "outcome": "success",
                    "verified": True,
                },
            },
            {
                "event_id": "learn-success-2",
                "learning_context": {
                    "belief_updates": [
                        {
                            "kind": "outcome_model_update",
                            "action": "run_test",
                            "outcome": "success",
                            "verified": True,
                        }
                    ]
                },
            },
        ],
    )

    result = TaskDiscoveryEngine().discover_from_paths(evidence_ledger_path=evidence)

    self_candidates = [candidate for candidate in result.candidates if candidate.source == "self_model"]
    skill_candidates = [candidate for candidate in result.candidates if candidate.source == "skill_learning"]
    assert self_candidates
    assert skill_candidates
    assert self_candidates[0].status == "queued"
    assert self_candidates[0].metadata["action_name"] == "file_read"
    assert self_candidates[0].metadata["learning_pressure"] == "capability_reliability_low"
    assert "promote_failed_skill_without_verifier" in self_candidates[0].forbidden_actions
    assert skill_candidates[0].status == "queued"
    assert skill_candidates[0].metadata["action_name"] == "run_test"
    assert skill_candidates[0].metadata["learning_pressure"] == "skill_candidate_from_verified_success"
    assert "install_skill_without_review" in skill_candidates[0].forbidden_actions


class _FakeCreativeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.response, ensure_ascii=False)


def test_creative_task_generation_adds_evidence_grounded_candidate(tmp_path: Path) -> None:
    active_goals = tmp_path / "active_goals.json"
    evidence = tmp_path / "evidence.jsonl"
    _write_json(
        active_goals,
        {
            "active_goals": [
                {
                    "goal_id": "open_loop",
                    "description": "open task closure",
                    "metric": "accepted_patch_rate",
                    "current": 0.0,
                    "target": 0.7,
                }
            ]
        },
    )
    _write_jsonl(
        evidence,
        [
            {
                "evidence_id": "ev-loop",
                "status": "needs_human_review",
                "message": "open task report: commit_log=0 unknown fallback recovery failed",
            }
        ],
    )
    fake = _FakeCreativeClient(
        [
            {
                "source": "hypothesis",
                "observation": "Open task reports and goal gap both point to an unmeasured recovery boundary.",
                "gap": "The system cannot tell whether failures come from recovery diagnosis or patch proposal gating.",
                "proposed_task": "Construct a two-run diagnostic comparing recovery classification before and after patch proposal refusal.",
                "expected_value": 0.88,
                "goal_alignment": 0.93,
                "evidence_strength": 0.74,
                "feasibility": 0.82,
                "risk": 0.18,
                "cost": 0.28,
                "reversibility": 0.90,
                "distraction_penalty": 0.05,
                "evidence_needed": ["open task report", "recovery log", "patch refusal trace"],
                "success_condition": "The diagnostic separates recovery classifier failure from patch proposal refusal in at least one real trace.",
                "allowed_actions": ["read_reports", "read_logs", "run_eval", "write_report"],
                "forbidden_actions": ["modify_core_runtime_without_approval"],
                "evidence_refs": ["failure:ev-loop"],
                "permission_level": "L1",
            }
        ]
    )

    engine = TaskDiscoveryEngine(creative_generator=CreativeTaskGenerator(fake))
    result = engine.discover_from_paths(
        active_goals_path=active_goals,
        evidence_ledger_path=evidence,
        enable_creative=True,
    )

    creative = [candidate for candidate in result.candidates if candidate.metadata.get("creative_generation")]
    assert creative
    assert creative[0].status == "queued"
    assert "modify_core_runtime_without_approval" in creative[0].forbidden_actions
    assert result.report["creative_generation"]["accepted_count"] == 1
    assert fake.prompts and "Seed candidates" in fake.prompts[0]


def test_creative_task_generation_rejects_ungrounded_candidates(tmp_path: Path) -> None:
    active_goals = tmp_path / "active_goals.json"
    evidence = tmp_path / "evidence.jsonl"
    _write_json(
        active_goals,
        {
            "active_goals": [
                {
                    "goal_id": "open_loop",
                    "description": "open task closure",
                    "metric": "accepted_patch_rate",
                    "current": 0.0,
                    "target": 0.7,
                }
            ]
        },
    )
    _write_jsonl(evidence, [{"evidence_id": "ev-loop", "status": "needs_human_review"}])
    fake = _FakeCreativeClient(
        [
            {
                "source": "opportunity",
                "observation": "Maybe build a shiny dashboard.",
                "gap": "No cited evidence.",
                "proposed_task": "Build a dashboard.",
                "expected_value": 1,
                "goal_alignment": 1,
                "evidence_strength": 1,
                "feasibility": 1,
                "risk": 0,
                "cost": 0,
                "reversibility": 1,
                "distraction_penalty": 0,
                "evidence_needed": ["none"],
                "success_condition": "dashboard exists",
                "allowed_actions": ["write_report"],
                "forbidden_actions": [],
                "evidence_refs": ["not_real_ref"],
                "permission_level": "L1",
            }
        ]
    )

    engine = TaskDiscoveryEngine(creative_generator=CreativeTaskGenerator(fake))
    result = engine.discover_from_paths(
        active_goals_path=active_goals,
        evidence_ledger_path=evidence,
        enable_creative=True,
    )

    assert not [candidate for candidate in result.candidates if candidate.metadata.get("creative_generation")]
    assert result.report["creative_generation"]["accepted_count"] == 0
    assert result.report["creative_generation"]["rejected_count"] == 1
