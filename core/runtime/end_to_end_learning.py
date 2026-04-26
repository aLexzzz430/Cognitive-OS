from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from core.runtime.api_surface import extract_attribute_errors
from core.runtime.failure_learning import (
    FAILURE_LEARNING_VERSION,
    build_failure_learning_hint_text,
    failure_object_matches_tags,
    failure_objects_to_context_entries,
    normalize_failure_learning_object,
)
from core.runtime.state_store import DEFAULT_STATE_DB, RuntimeStateStore


END_TO_END_LEARNING_VERSION = "conos.end_to_end_learning/v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _str_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _excerpt(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_tags(tags: Any) -> set[str]:
    return {str(item).strip().lower() for item in _as_list(tags) if str(item).strip()}


def _infer_objective_tags(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tags: set[str] = set()
    if _contains_any(
        lowered,
        (
            "github-ready",
            "ai program",
            "ai app",
            "ai project",
            "ai程序",
            "ai 项目",
            "可以上线github",
            "上线github",
            "autonomous build",
        ),
    ):
        tags.update({"ai_project_generation", "python_generation"})
    if _contains_any(lowered, ("test", "pytest", "unittest", "测试")):
        tags.add("requires_tests")
    if _contains_any(lowered, ("python", "pyproject", "pip install", "src/")):
        tags.add("python_generation")
    if _contains_any(lowered, ("ollama", "qwen", "gemma", "batiai", "llm", "模型", "model")):
        tags.add("llm_generation")
    if _contains_any(
        lowered,
        (
            "small project",
            "小型项目",
            "任意选",
            "improve",
            "改进",
            "maintenance",
            "维护",
            "website",
            "site",
            "readme",
            "contributing",
            ".editorconfig",
            "文档",
        ),
    ):
        tags.update({"project_maintenance", "docs_tooling"})
    return tags


def _infer_lesson_tags(row: Mapping[str, Any]) -> set[str]:
    lesson = _as_dict(row.get("lesson"))
    explicit = _normalize_tags(lesson.get("tags"))
    if explicit:
        return explicit
    trigger = str(row.get("trigger", "") or "").lower()
    text = " ".join(
        [
            trigger,
            str(lesson.get("title", "") or "").lower(),
            str(lesson.get("hint", "") or "").lower(),
            str(lesson.get("objective_excerpt", "") or "").lower(),
        ]
    )
    tags = _infer_objective_tags(text)
    if "api_surface" in trigger or "attribute" in text:
        tags.update({"api_surface", "python_generation", "requires_tests"})
    if any(token in trigger for token in ("placeholder", "missing_required_tests", "missing_autonomous", "builder_command_failed")):
        tags.update({"ai_project_generation", "python_generation"})
        if "tests" in trigger or "test" in text:
            tags.add("requires_tests")
    if "ollama" in trigger:
        tags.update({"llm_generation", "ollama"})
    if "successful_artifact_contract" in trigger:
        tags.add("local_machine")
    return tags or {"local_machine", "general"}


def _lesson_matches_objective(row: Mapping[str, Any], objective_tags: set[str]) -> bool:
    lesson_tags = _infer_lesson_tags(row)
    specific_lesson_tags = lesson_tags - {"general", "local_machine"}
    if not objective_tags:
        return not specific_lesson_tags
    if "project_maintenance" in objective_tags and "ai_project_generation" in lesson_tags and not (lesson_tags & objective_tags):
        return False
    if lesson_tags & objective_tags:
        return True
    if "ai_project_generation" in objective_tags and lesson_tags & {"api_surface", "python_generation", "requires_tests", "llm_generation"}:
        return True
    if "python_generation" in objective_tags and lesson_tags & {"api_surface", "requires_tests"}:
        return True
    return bool(lesson_tags <= {"general", "local_machine"})


def _mirror_audit_events(audit: Mapping[str, Any]) -> list[Dict[str, Any]]:
    final_raw = _as_dict(audit.get("final_surface_raw"))
    mirror = _as_dict(final_raw.get("local_mirror"))
    return [dict(item) for item in _as_list(mirror.get("audit_events")) if isinstance(item, dict)]


def _mirror_command_text(audit: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for event in _mirror_audit_events(audit):
        if str(event.get("event_type", "") or "") != "mirror_command_executed":
            continue
        payload = _as_dict(event.get("payload"))
        chunks.extend(
            [
                str(payload.get("stdout_tail", "") or ""),
                str(payload.get("stderr_tail", "") or ""),
                str(payload.get("returncode", "") or ""),
            ]
        )
    return "\n".join(part for part in chunks if part)


def build_learning_hint_text(lessons: Sequence[Mapping[str, Any]], *, limit: int = 5) -> str:
    selected = [dict(item) for item in list(lessons or [])[: max(1, int(limit or 5))]]
    if not selected:
        return ""
    lines = ["Con OS learned constraints from prior runs:"]
    for index, row in enumerate(selected, start=1):
        lesson = _as_dict(row.get("lesson"))
        title = str(lesson.get("title") or row.get("trigger") or "lesson").strip()
        hint = str(lesson.get("hint") or lesson.get("summary") or "").strip()
        if not hint:
            continue
        lines.append(f"{index}. {title}: {hint}")
    return "\n".join(lines)


def apply_lessons_to_unified_context(unified_context: Any, lessons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Write persisted lessons into the canonical context fields used by planning.

    This helper deliberately uses existing context fields instead of widening the
    public UnifiedCognitiveContext schema: failures go to recent_failure_profile,
    positive lessons go to recent_progress_markers, and all lessons leave compact
    evidence entries.
    """
    selected = [dict(item) for item in list(lessons or []) if isinstance(item, Mapping)]
    if unified_context is None or not selected:
        return {"schema_version": END_TO_END_LEARNING_VERSION, "applied": 0, "lesson_ids": []}

    failure_profile = list(getattr(unified_context, "recent_failure_profile", []) or [])
    progress_markers = list(getattr(unified_context, "recent_progress_markers", []) or [])
    evidence_queue = list(getattr(unified_context, "evidence_queue", []) or [])
    lesson_ids: list[str] = []

    for row in selected:
        lesson = _as_dict(row.get("lesson"))
        lesson_id = str(row.get("lesson_id", "") or "")
        lesson_ids.append(lesson_id)
        compact = {
            "source": "end_to_end_learning",
            "lesson_id": lesson_id,
            "trigger": str(row.get("trigger", "") or ""),
            "kind": str(lesson.get("kind", "") or ""),
            "title": str(lesson.get("title", "") or ""),
            "hint": str(lesson.get("hint", "") or ""),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
        }
        if str(lesson.get("kind", "") or "").lower() in {"positive", "success"}:
            progress_markers.append(compact)
        else:
            failure_profile.append(compact)
        evidence_queue.append(dict(compact))

    setattr(unified_context, "recent_failure_profile", failure_profile[-20:])
    setattr(unified_context, "recent_progress_markers", progress_markers[-20:])
    setattr(unified_context, "evidence_queue", evidence_queue[-50:])
    posterior_summary = _as_dict(getattr(unified_context, "posterior_summary", {}))
    posterior_summary["end_to_end_learning"] = {
        "schema_version": END_TO_END_LEARNING_VERSION,
        "lesson_count": len(selected),
        "latest_lesson_ids": [item for item in lesson_ids if item][-10:],
    }
    setattr(unified_context, "posterior_summary", posterior_summary)
    return {
        "schema_version": END_TO_END_LEARNING_VERSION,
        "applied": len(selected),
        "lesson_ids": [item for item in lesson_ids if item],
    }


def apply_learning_context_to_unified_context(unified_context: Any, learning_context: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply both compact lessons and structured failure objects to context."""
    payload = _as_dict(learning_context)
    lesson_result = apply_lessons_to_unified_context(unified_context, list(payload.get("lessons", []) or []))
    failure_objects = [
        dict(item)
        for item in list(payload.get("failure_objects", []) or [])
        if isinstance(item, Mapping)
    ]
    if unified_context is None or not failure_objects:
        return {
            "schema_version": END_TO_END_LEARNING_VERSION,
            "lessons": lesson_result,
            "failure_learning": {
                "schema_version": FAILURE_LEARNING_VERSION,
                "applied": 0,
                "failure_ids": [],
            },
        }

    entries = failure_objects_to_context_entries(failure_objects)
    failure_profile = list(getattr(unified_context, "recent_failure_profile", []) or [])
    evidence_queue = list(getattr(unified_context, "evidence_queue", []) or [])
    failure_profile.extend(entries)
    evidence_queue.extend(entries)
    setattr(unified_context, "recent_failure_profile", failure_profile[-30:])
    setattr(unified_context, "evidence_queue", evidence_queue[-80:])
    posterior_summary = _as_dict(getattr(unified_context, "posterior_summary", {}))
    posterior_summary["failure_learning"] = {
        "schema_version": FAILURE_LEARNING_VERSION,
        "failure_object_count": len(entries),
        "latest_failure_ids": [item.get("failure_id", "") for item in entries if item.get("failure_id")][-10:],
        "failure_modes": sorted({item.get("failure_mode", "") for item in entries if item.get("failure_mode")}),
    }
    setattr(unified_context, "posterior_summary", posterior_summary)
    return {
        "schema_version": END_TO_END_LEARNING_VERSION,
        "lessons": lesson_result,
        "failure_learning": {
            "schema_version": FAILURE_LEARNING_VERSION,
            "applied": len(entries),
            "failure_ids": [item.get("failure_id", "") for item in entries if item.get("failure_id")],
        },
    }


class EndToEndLearningRuntime:
    """Cross-run learning bridge backed by the local runtime state store."""

    def __init__(
        self,
        *,
        state_store: Optional[RuntimeStateStore] = None,
        db_path: str | Path = DEFAULT_STATE_DB,
    ) -> None:
        self.state_store = state_store or RuntimeStateStore(db_path)

    def learning_context_for_task(
        self,
        *,
        task_family: str,
        objective: str = "",
        limit: int = 5,
        mark_used: bool = True,
    ) -> Dict[str, Any]:
        lessons = self._select_lessons(task_family=task_family, objective=objective, limit=limit)
        failure_objects = self._select_failure_objects(task_family=task_family, objective=objective, limit=limit)
        if mark_used:
            for lesson in lessons:
                lesson_id = str(lesson.get("lesson_id", "") or "")
                if lesson_id:
                    self.state_store.mark_learning_lesson_used(lesson_id)
            for failure in failure_objects:
                failure_id = str(failure.get("failure_id", "") or "")
                if failure_id:
                    self.state_store.mark_failure_learning_object_used(failure_id)
        lesson_hint = build_learning_hint_text(lessons, limit=limit)
        failure_hint = build_failure_learning_hint_text(failure_objects, limit=min(limit, 4))
        return {
            "schema_version": END_TO_END_LEARNING_VERSION,
            "task_family": str(task_family or "generic"),
            "objective_excerpt": _excerpt(objective),
            "objective_tags": sorted(_infer_objective_tags(objective)),
            "lesson_count": len(lessons),
            "lessons": lessons,
            "failure_learning_schema_version": FAILURE_LEARNING_VERSION,
            "failure_object_count": len(failure_objects),
            "failure_objects": failure_objects,
            "failure_hint_text": failure_hint,
            "hint_text": lesson_hint,
        }

    def record_lesson(
        self,
        *,
        task_family: str,
        trigger: str,
        lesson: Mapping[str, Any],
        source_run_id: str = "",
        confidence: float = 0.5,
    ) -> str:
        return self.state_store.record_learning_lesson(
            task_family=task_family,
            trigger=trigger,
            lesson=dict(lesson or {}),
            source_run_id=source_run_id,
            confidence=confidence,
        )

    def record_failure_object(
        self,
        *,
        task_family: str,
        failure_mode: str,
        failure_object: Mapping[str, Any],
        source_run_id: str = "",
        confidence: float = 0.5,
    ) -> str:
        return self.state_store.record_failure_learning_object(
            task_family=task_family,
            failure_mode=failure_mode,
            failure_object=dict(failure_object or {}),
            source_run_id=source_run_id,
            confidence=confidence,
        )

    def learn_from_local_machine_audit(
        self,
        *,
        run_id: str,
        instruction: str,
        audit: Mapping[str, Any],
    ) -> Dict[str, Any]:
        lessons = self._extract_local_machine_lessons(
            run_id=str(run_id or ""),
            instruction=str(instruction or ""),
            audit=audit,
        )
        failure_objects = self._extract_local_machine_failure_objects(
            run_id=str(run_id or ""),
            instruction=str(instruction or ""),
            audit=audit,
        )
        recorded: list[Dict[str, Any]] = []
        for row in lessons:
            lesson_id = self.record_lesson(
                task_family="local_machine",
                trigger=str(row.get("trigger", "") or "local_machine:generic"),
                lesson=_as_dict(row.get("lesson")),
                source_run_id=str(run_id or ""),
                confidence=float(row.get("confidence", 0.5) or 0.5),
            )
            stored = dict(row)
            stored["lesson_id"] = lesson_id
            recorded.append(stored)
        recorded_failures: list[Dict[str, Any]] = []
        for row in failure_objects:
            failure_object = _as_dict(row.get("failure_object"))
            failure_id = self.record_failure_object(
                task_family="local_machine",
                failure_mode=str(failure_object.get("failure_mode") or row.get("failure_mode") or "unknown_failure"),
                failure_object=failure_object,
                source_run_id=str(run_id or ""),
                confidence=float(row.get("confidence", failure_object.get("confidence", 0.5)) or 0.5),
            )
            stored_failure = dict(row)
            stored_failure["failure_id"] = failure_id
            recorded_failures.append(stored_failure)
        return {
            "schema_version": END_TO_END_LEARNING_VERSION,
            "source_run_id": str(run_id or ""),
            "recorded_count": len(recorded),
            "lessons": recorded,
            "failure_learning_schema_version": FAILURE_LEARNING_VERSION,
            "recorded_failure_object_count": len(recorded_failures),
            "failure_objects": recorded_failures,
        }

    def _select_lessons(self, *, task_family: str, objective: str, limit: int) -> list[Dict[str, Any]]:
        family = str(task_family or "generic")
        objective_tags = _infer_objective_tags(objective)
        query_limit = max(20, max(1, int(limit or 5)) * 6)
        selected: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for candidate_family in (family, "global"):
            for lesson in self.state_store.list_learning_lessons(task_family=candidate_family, limit=query_limit):
                key = str(lesson.get("lesson_hash") or lesson.get("lesson_id") or "")
                if key and key in seen:
                    continue
                if not _lesson_matches_objective(lesson, objective_tags):
                    continue
                seen.add(key)
                selected.append(lesson)
                if len(selected) >= max(1, int(limit or 5)):
                    return selected
        return selected

    def _select_failure_objects(self, *, task_family: str, objective: str, limit: int) -> list[Dict[str, Any]]:
        family = str(task_family or "generic")
        objective_tags = _infer_objective_tags(objective)
        query_limit = max(20, max(1, int(limit or 5)) * 6)
        selected: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for candidate_family in (family, "global"):
            for row in self.state_store.list_failure_learning_objects(task_family=candidate_family, limit=query_limit):
                key = str(row.get("failure_hash") or row.get("failure_id") or "")
                if key and key in seen:
                    continue
                if not failure_object_matches_tags(row, objective_tags):
                    continue
                seen.add(key)
                selected.append(row)
                if len(selected) >= max(1, int(limit or 5)):
                    return selected
        return selected

    def _extract_local_machine_failure_objects(
        self,
        *,
        run_id: str,
        instruction: str,
        audit: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        final_raw = _as_dict(audit.get("final_surface_raw"))
        mirror = _as_dict(final_raw.get("local_mirror"))
        artifact_check = _as_dict(audit.get("local_machine_artifact_check"))
        failures = _str_list(artifact_check.get("failures"))
        command_text = _mirror_command_text(audit)
        command_text_lower = command_text.lower()
        objective_tags = _infer_objective_tags(instruction) or {"local_machine"}
        objects: list[Dict[str, Any]] = []

        def add(
            *,
            failure_mode: str,
            title: str,
            summary: str,
            violated_assumption: str,
            confidence: float,
            evidence_refs: Sequence[Any] = (),
            failed_action: Mapping[str, Any] | None = None,
            failure_result: Mapping[str, Any] | None = None,
            missing_tool: str = "",
            bad_policy: str = "",
            regression: Mapping[str, Any] | None = None,
            governance_rule: Mapping[str, Any] | None = None,
            tags: Sequence[str] = (),
        ) -> None:
            failure_object = normalize_failure_learning_object(
                failure_mode=failure_mode,
                title=title,
                summary=summary,
                violated_assumption=violated_assumption,
                failed_action=failed_action or {},
                failure_result=failure_result or {},
                evidence_refs=evidence_refs,
                missing_tool=missing_tool,
                bad_policy=bad_policy,
                new_regression_test=regression or {},
                new_governance_rule=governance_rule or {},
                retrieval_tags=sorted(set(objective_tags) | _normalize_tags(tags)),
                source_run_id=run_id,
                source_task_family="local_machine",
                confidence=confidence,
            )
            objects.append(
                {
                    "failure_mode": failure_object["failure_mode"],
                    "confidence": confidence,
                    "failure_object": failure_object,
                }
            )

        command_failed = bool(failures) or "traceback" in command_text_lower or "error" in command_text_lower
        if command_failed:
            add(
                failure_mode="validation_or_execution_failure",
                title="Failed execution must become repair evidence",
                summary="A local-machine run failed validation, execution, or artifact contract checks.",
                violated_assumption="A failed mirror command or artifact contract can be treated as task completion.",
                confidence=0.7,
                evidence_refs=["artifact_check:failures"] + [f"failure:{item}" for item in failures[:6]],
                failed_action={"action": "mirror_exec_or_artifact_check"},
                failure_result={"failures": failures, "command_excerpt": _excerpt(command_text, limit=420)},
                missing_tool="read_run_output",
                regression={
                    "description": "Assert failed mirror commands are inspected and repaired before sync planning.",
                    "suggested_target": "tests/test_local_machine_failure_learning.py",
                },
                governance_rule={
                    "description": "Block source sync when latest validation/execution failed or artifact contract is false.",
                    "required_signal": "passing_validation_and_artifact_contract",
                },
                tags=("build_repair", "requires_tests"),
            )

        if "placeholder content detected" in command_text_lower or any("placeholder" in item.lower() for item in failures):
            add(
                failure_mode="placeholder_generation",
                title="Placeholder generation is a failed artifact",
                summary="Generated content included placeholders, pass-only tests, TODOs, or template residue.",
                violated_assumption="A syntactically present generated file is equivalent to a product-quality artifact.",
                confidence=0.86,
                evidence_refs=["command:stderr_tail", "artifact_check:placeholder"],
                failed_action={"action": "generate_or_validate_artifact"},
                failure_result={"failures": failures, "command_excerpt": _excerpt(command_text, limit=420)},
                missing_tool="artifact_contract_check",
                bad_policy="sync_after_placeholder_artifact",
                regression={
                    "description": "Scan generated source/tests for placeholder markers, TODO-only bodies, ellipses, and pass-only tests.",
                    "forbid_markers": ["placeholder", "TODO", "...", "pass-only"],
                },
                governance_rule={
                    "description": "Refuse mirror_plan or mirror_apply for generated products when placeholder markers are detected.",
                    "required_signal": "non_template_artifact_verified",
                },
                tags=("ai_project_generation", "python_generation", "requires_tests"),
            )

        if any(item.startswith("required_workspace_path:") for item in failures):
            required = [item.split(":", 1)[1] for item in failures if item.startswith("required_workspace_path:")]
            add(
                failure_mode="missing_required_artifact",
                title="Required artifacts must exist before completion",
                summary="The run reached a completion path while required workspace artifacts were missing.",
                violated_assumption="A plan can be considered complete before the requested files/tests/reports exist.",
                confidence=0.8,
                evidence_refs=[f"artifact_contract:{item}" for item in required],
                failed_action={"action": "artifact_contract_check"},
                failure_result={"missing_required_paths": required},
                missing_tool="repo_find",
                bad_policy="completion_without_required_artifacts",
                regression={
                    "description": "Fail the run when required workspace globs are absent at completion.",
                    "required_globs": required,
                },
                governance_rule={
                    "description": "Completion and source sync require required artifact globs to resolve to real files.",
                    "required_signal": "required_artifact_paths_present",
                },
                tags=("ai_project_generation", "artifact_contract", "requires_tests"),
            )

        attribute_errors = extract_attribute_errors(command_text)
        if attribute_errors:
            add(
                failure_mode="api_surface_mismatch",
                title="Generated tests must match the real API surface",
                summary="Validation failed because tests or code referenced missing attributes.",
                violated_assumption="The generated or assumed API method exists without inspecting the source surface.",
                confidence=0.84,
                evidence_refs=["validation:attribute_error"],
                failed_action={"action": "run_test"},
                failure_result={"attribute_errors": attribute_errors, "command_excerpt": _excerpt(command_text, limit=420)},
                missing_tool="file_outline",
                bad_policy="invent_api_without_source_outline",
                regression={
                    "description": "Before repairing AttributeError failures, read/outline both the failing test and target source file.",
                    "attribute_errors": attribute_errors,
                },
                governance_rule={
                    "description": "Do not accept tests that call methods not present in file_outline/file_read evidence.",
                    "required_signal": "api_surface_evidence_ref",
                },
                tags=("api_surface", "python_generation", "requires_tests"),
            )

        investigation = _as_dict(mirror.get("investigation"))
        governance = _as_dict(mirror.get("action_governance") or investigation.get("action_governance"))
        last_decision = _as_dict(governance.get("last_decision"))
        if str(last_decision.get("status") or "") in {"BLOCKED", "DOWNGRADED", "WAITING_APPROVAL"}:
            request = _as_dict(last_decision.get("request"))
            add(
                failure_mode="governance_block",
                title="Governance block should become the next investigation step",
                summary="Action governance blocked or downgraded an action because authority prerequisites were missing.",
                violated_assumption="A selected tool action can proceed without satisfying evidence, validation, approval, or failure-budget gates.",
                confidence=0.78,
                evidence_refs=["action_governance:last_decision"],
                failed_action={"action": request.get("action_name", ""), "permissions": request.get("permissions_required", [])},
                failure_result={
                    "status": last_decision.get("status"),
                    "blocked_reason": last_decision.get("blocked_reason"),
                    "required_evidence": last_decision.get("required_evidence"),
                    "required_tests": last_decision.get("required_tests"),
                },
                missing_tool="investigation_status",
                bad_policy=str(last_decision.get("blocked_reason") or "authority_prerequisite_missing"),
                regression={
                    "description": "Verify blocked governance decisions are converted into evidence-gathering or validation actions.",
                    "blocked_reason": last_decision.get("blocked_reason"),
                },
                governance_rule={
                    "description": "After a governance block, select the missing prerequisite action before retrying the blocked action.",
                    "required_signal": "blocked_prerequisite_satisfied",
                },
                tags=("agent_capability_governance", "local_machine"),
            )

        return objects

    def _extract_local_machine_lessons(
        self,
        *,
        run_id: str,
        instruction: str,
        audit: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        final_raw = _as_dict(audit.get("final_surface_raw"))
        mirror = _as_dict(final_raw.get("local_mirror"))
        artifact_check = _as_dict(audit.get("local_machine_artifact_check"))
        failures = _str_list(artifact_check.get("failures"))
        command_text = _mirror_command_text(audit)
        command_text_lower = command_text.lower()
        objective_tags = _infer_objective_tags(instruction)
        if not objective_tags:
            objective_tags = {"local_machine"}
        lessons: list[Dict[str, Any]] = []

        def add(
            trigger: str,
            *,
            kind: str,
            title: str,
            hint: str,
            confidence: float,
            evidence: Mapping[str, Any],
            tags: Sequence[str] = (),
        ) -> None:
            lessons.append(
                {
                    "trigger": trigger,
                    "confidence": confidence,
                    "lesson": {
                        "schema_version": END_TO_END_LEARNING_VERSION,
                        "kind": kind,
                        "title": title,
                        "hint": hint,
                        "source": "local_machine_audit",
                        "source_run_id": run_id,
                        "objective_excerpt": _excerpt(instruction),
                        "tags": sorted(set(objective_tags) | _normalize_tags(tags)),
                        "evidence": dict(evidence),
                    },
                }
            )

        if "placeholder content detected" in command_text_lower or any("placeholder" in item.lower() for item in failures):
            add(
                "local_machine:placeholder_test_generation",
                kind="avoidance",
                title="Avoid placeholder tests",
                hint=(
                    "When generating tests, produce concrete assertions against the actual source API. "
                    "Do not include ellipses, pass-only bodies, TODOs, placeholder wording, fallback imports, "
                    "or assumption comments."
                ),
                confidence=0.86,
                evidence={"failures": failures, "command_excerpt": _excerpt(command_text, limit=360)},
                tags=("ai_project_generation", "python_generation", "requires_tests"),
            )

        if any(item.startswith("required_workspace_path:") and "tests" in item for item in failures):
            add(
                "local_machine:missing_required_tests",
                kind="recovery",
                title="Required tests must be real artifacts",
                hint=(
                    "If the task asks for a GitHub-ready program, create importable tests under the requested tests path "
                    "before planning sync-back."
                ),
                confidence=0.78,
                evidence={"failures": failures},
                tags=("ai_project_generation", "python_generation", "requires_tests"),
            )

        if any("AUTONOMOUS_BUILD_SUMMARY.json" in item for item in failures):
            add(
                "local_machine:missing_autonomous_build_summary",
                kind="recovery",
                title="Write the autonomous build summary",
                hint=(
                    "Finish autonomous builds by writing AUTONOMOUS_BUILD_SUMMARY.json with generated files, "
                    "validation results, and unresolved risks."
                ),
                confidence=0.76,
                evidence={"failures": failures},
                tags=("ai_project_generation",),
            )

        if "latest_command_succeeded" in failures:
            add(
                "local_machine:builder_command_failed",
                kind="recovery",
                title="Repair before sync planning",
                hint=(
                    "A failed mirror command is not a finished build. Inspect stderr, repair the generated workspace, "
                    "rerun validation, and only then build the sync plan."
                ),
                confidence=0.74,
                evidence={"failures": failures, "command_excerpt": _excerpt(command_text, limit=360)},
                tags=("build_repair", "python_generation"),
            )

        attribute_errors = extract_attribute_errors(command_text)
        if attribute_errors:
            add(
                "local_machine:api_surface_mismatch",
                kind="recovery",
                title="Repair source and tests against the real API surface",
                hint=(
                    "When validation fails with a missing attribute, extract the generated Python API surface, "
                    "include both the class source file and failing tests as repair targets, and do not invent a "
                    "renamed method without confirming it exists."
                ),
                confidence=0.82,
                evidence={"attribute_errors": attribute_errors, "command_excerpt": _excerpt(command_text, limit=360)},
                tags=("api_surface", "python_generation", "requires_tests"),
            )

        if "timed out" in command_text_lower or "operation timed out" in command_text_lower:
            add(
                "local_machine:ollama_timeout",
                kind="recovery",
                title="Handle remote model timeout",
                hint=(
                    "For remote Ollama timeouts, run a short health probe first, reduce prompt size or file count, "
                    "and prefer incremental retries over regenerating the whole project."
                ),
                confidence=0.72,
                evidence={"command_excerpt": _excerpt(command_text, limit=360)},
                tags=("llm_generation", "ollama"),
            )

        actionable_count = len(_as_list(_as_dict(mirror.get("sync_plan")).get("actionable_changes")))
        if bool(artifact_check.get("ok", False)) and actionable_count > 0:
            add(
                "local_machine:successful_artifact_contract",
                kind="positive",
                title="Preserve artifact-first build order",
                hint=(
                    "The reliable path is execute builder, verify required artifacts exist, then produce a sync plan "
                    "with actionable changes."
                ),
                confidence=0.68,
                evidence={"actionable_change_count": actionable_count},
                tags=("local_machine", "artifact_contract"),
            )

        return lessons
