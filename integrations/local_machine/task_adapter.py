from __future__ import annotations

import ast
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from core.environment.types import (
    GenericActionDescriptor,
    GenericObservation,
    GenericTaskSpec,
)
from core.runtime.evidence_ledger import (
    FORMAL_EVIDENCE_LEDGER_VERSION,
    FormalEvidenceLedger,
    build_local_machine_evidence_entry,
)
from core.runtime.hypothesis_lifecycle import (
    HYPOTHESIS_LIFECYCLE_VERSION,
    apply_hypothesis_evidence,
    build_discriminating_test,
    hypothesis_lifecycle_summary,
    mark_competing,
    normalize_hypothesis,
)
from core.runtime.state_store import RuntimeStateStore
from core.surfaces.base import ActionResult, SurfaceObservation, ToolSpec
from integrations.local_machine.action_grounding import (
    action_schema_registry_payload,
    extract_grounding_target_file,
    is_completed_verified_context,
    is_local_machine_side_effect_action,
    pytest_context_paths_from_tree,
    side_effect_after_completion_event,
    validate_local_machine_action,
)
from integrations.local_machine.budget_policy import budget_policy_report
from integrations.local_machine.patch_proposal import generate_patch_proposals
from integrations.local_machine.target_binding import bind_target
from modules.control_plane.action_governance import (
    ActionGovernanceDecision,
    ActionGovernancePolicy,
    coerce_action_governance_policy,
    derive_action_governance_request,
    evaluate_action_governance,
    governance_state_from_local_machine_investigation,
    record_action_governance_result,
)
from modules.internet import (
    InternetIngressError,
    InternetIngressPolicy,
    fetch_project,
    fetch_url,
    load_manifest as load_internet_manifest,
)
from modules.local_mirror.mirror import (
    DEFAULT_ALLOWED_COMMANDS,
    MirrorScopeError,
    acquire_relevant_files,
    apply_sync_plan,
    build_sync_plan,
    compute_mirror_diff,
    create_empty_mirror,
    execution_boundary_report,
    is_generated_mirror_artifact,
    materialize_files,
    open_mirror,
    run_mirror_command,
)


LOCAL_MACHINE_ADAPTER_VERSION = "conos.local_machine_adapter/v1"
LOCAL_MACHINE_RAW_DIFF_ARTIFACT_VERSION = "conos.local_machine.raw_diff_artifact/v1"
LOCAL_MACHINE_INVESTIGATION_VERSION = "conos.local_machine.investigation/v1"
LOCAL_MACHINE_RUN_OUTPUT_VERSION = "conos.local_machine.run_output/v1"

DEFAULT_REPO_EXCLUDES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "__pycache__",
        ".venv",
    }
)
EDIT_INTENT_TOKENS = frozenset(
    {
        "fix",
        "improve",
        "implement",
        "modify",
        "patch",
        "repair",
        "update",
        "修复",
        "改进",
        "实现",
        "修改",
        "补丁",
        "完善",
    }
)
ATOMIC_LOCAL_MACHINE_TOOLS = frozenset(
    {
        "repo_tree",
        "repo_find",
        "repo_grep",
        "file_read",
        "file_outline",
        "file_summary",
        "note_write",
        "hypothesis_add",
        "hypothesis_update",
        "hypothesis_compete",
        "discriminating_test_add",
        "candidate_files_set",
        "candidate_files_update",
        "investigation_status",
        "propose_patch",
        "apply_patch",
        "edit_replace_range",
        "edit_insert_after",
        "create_file",
        "delete_file",
        "run_test",
        "run_lint",
        "run_typecheck",
        "run_build",
        "read_run_output",
        "read_test_failure",
    }
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


class LocalMachineSurfaceAdapter:
    """
    CoreMainLoop adapter for a local-machine mirror.

    The adapter presents the host machine through an empty-first workspace.
    Source files are only materialized by explicit tool calls, command execution
    runs from the mirror workspace, and source writes require an approved sync
    plan.
    """

    def __init__(
        self,
        *,
        instruction: str = "",
        source_root: str | Path = ".",
        mirror_root: str | Path = "runtime/mirrors/local-machine",
        candidate_paths: Optional[Iterable[str | Path]] = None,
        fetch_paths: Optional[Iterable[str | Path]] = None,
        default_command: Optional[Sequence[str] | str] = None,
        allowed_commands: Optional[Iterable[str]] = None,
        reset_mirror: bool = False,
        terminal_after_plan: bool = True,
        expose_apply_tool: bool = False,
        allow_empty_exec: bool = False,
        default_command_timeout_seconds: int = 30,
        execution_backend: str = "local",
        docker_image: str = "python:3.10-slim",
        vm_provider: str = "auto",
        vm_name: str = "",
        vm_host: str = "",
        vm_workdir: str = "/workspace",
        vm_network_mode: str = "provider_default",
        vm_sync_mode: str = "none",
        internet_enabled: bool = False,
        internet_output_root: str | Path | None = None,
        internet_max_bytes: int = 2 * 1024 * 1024,
        internet_timeout_seconds: float = 20.0,
        internet_allow_private_networks: bool = False,
        deterministic_fallback_enabled: bool = True,
        prefer_llm_kwargs: bool = False,
        prefer_llm_patch_proposals: bool = False,
        llm_thinking_mode: str = "auto",
        llm_client: Any = None,
        require_llm_generation: bool = False,
        require_market_evidence_reference: bool = False,
        require_non_template_product: bool = False,
        extra_env: Optional[Mapping[str, str]] = None,
        learning_context: Optional[Mapping[str, Any]] = None,
        evidence_db_path: str | Path | None = None,
        task_id: str = "local_machine",
        action_governance_enabled: bool = True,
        action_governance_policy: Optional[Mapping[str, Any] | ActionGovernancePolicy] = None,
        llm_cost_ledger: Any = None,
    ) -> None:
        self.instruction = str(instruction or "")
        self.source_root = Path(source_root).resolve()
        self.mirror_root = Path(mirror_root).resolve()
        if self.mirror_root == self.source_root or self.source_root.is_relative_to(self.mirror_root):
            raise MirrorScopeError("mirror_root must not be the source root or a parent of the source root")
        self.candidate_paths = [str(path) for path in list(candidate_paths or [])]
        self.fetch_paths = [str(path) for path in list(fetch_paths or [])]
        self.default_command = self._normalize_command(default_command)
        self.allowed_commands = [str(item) for item in list(allowed_commands or DEFAULT_ALLOWED_COMMANDS)]
        self.reset_mirror_on_first_observe = bool(reset_mirror)
        self.terminal_after_plan = bool(terminal_after_plan)
        self.expose_apply_tool = bool(expose_apply_tool)
        self.allow_empty_exec = bool(allow_empty_exec)
        self.default_command_timeout_seconds = max(1, int(default_command_timeout_seconds or 30))
        self.execution_backend = str(execution_backend or "local").strip().lower() or "local"
        self.docker_image = str(docker_image or "python:3.10-slim")
        self.vm_provider = str(vm_provider or "auto")
        self.vm_name = str(vm_name or "")
        self.vm_host = str(vm_host or "")
        self.vm_workdir = str(vm_workdir or "/workspace")
        self.vm_network_mode = str(vm_network_mode or "provider_default")
        self.vm_sync_mode = str(vm_sync_mode or "none").strip().lower() or "none"
        self.internet_enabled = bool(internet_enabled)
        self.internet_output_root = Path(internet_output_root).resolve() if internet_output_root else None
        self.internet_max_bytes = max(1, int(internet_max_bytes or 1))
        self.internet_timeout_seconds = max(1.0, float(internet_timeout_seconds or 1.0))
        self.internet_allow_private_networks = bool(internet_allow_private_networks)
        self.deterministic_fallback_enabled = bool(deterministic_fallback_enabled)
        self.prefer_llm_kwargs = bool(prefer_llm_kwargs)
        self.prefer_llm_patch_proposals = bool(prefer_llm_patch_proposals)
        self.llm_thinking_mode = str(llm_thinking_mode or "auto").strip().lower() or "auto"
        self.llm_client = llm_client
        self.require_llm_generation = bool(require_llm_generation)
        self.require_market_evidence_reference = bool(require_market_evidence_reference)
        self.require_non_template_product = bool(require_non_template_product)
        self.extra_env = {str(key): str(value) for key, value in dict(extra_env or {}).items()}
        self.learning_context = dict(learning_context or {})
        self.evidence_db_path = Path(evidence_db_path).resolve() if evidence_db_path else None
        self.task_id = str(task_id or "local_machine")
        self.action_governance_enabled = bool(action_governance_enabled)
        self.action_governance_policy = coerce_action_governance_policy(action_governance_policy)
        self.llm_cost_ledger = llm_cost_ledger

        self._initialized = False
        self._terminal = False
        self._last_plan: Dict[str, Any] = {}
        self._last_action: Dict[str, Any] = {}
        self._command_executed = False
        self._last_command_returncode: int | None = None
        self._command_failed = False
        self._acquire_attempted = False
        self._last_acquire_selected_count = -1
        self._applied = False
        self._episode = 0
        self._last_internet_artifact: Dict[str, Any] = {}

    @staticmethod
    def _normalize_command(command: Optional[Sequence[str] | str]) -> list[str]:
        if command is None:
            return []
        if isinstance(command, str):
            return shlex.split(command)
        return [str(part) for part in list(command) if str(part)]

    def reset(self, seed: int | None = None, episode: int | None = None, **_: Any) -> SurfaceObservation:
        self._episode = int(episode or self._episode or 1)
        self._terminal = False
        self._last_action = {}
        self._last_plan = {}
        self._command_executed = False
        self._last_command_returncode = None
        self._command_failed = False
        self._acquire_attempted = False
        self._last_acquire_selected_count = -1
        self._applied = False
        self._last_internet_artifact = {}
        create_empty_mirror(
            self.source_root,
            self.mirror_root,
            reset=bool(self.reset_mirror_on_first_observe),
        )
        self._initialized = True
        return self.observe()

    def next_episode(self) -> None:
        self.reset(episode=self._episode + 1)

    def observe(self) -> SurfaceObservation:
        if not self._initialized:
            return self.reset()
        mirror = open_mirror(self.source_root, self.mirror_root)
        manifest = mirror.to_manifest()
        sync_plan = _load_json(mirror.sync_plan_path)
        if sync_plan:
            self._last_plan = sync_plan
        tools = self._available_tools(manifest, sync_plan)
        raw = self._observation_payload(manifest, sync_plan, tools)
        text = (
            f"Local mirror workspace contains {manifest.get('workspace_file_count', 0)} file(s). "
            f"Source writes require an approved sync plan."
        )
        return SurfaceObservation(
            text=text,
            structured={"local_mirror": raw["local_mirror"]},
            available_tools=tools,
            terminal=bool(self._terminal),
            reward=None,
            raw=raw,
        )

    def act(self, action: Any) -> ActionResult:
        if not self._initialized:
            self.reset()
        action_dict = self._coerce_action(action)
        function_name, kwargs = self._extract_tool_call(action_dict)
        original_function_name = function_name
        original_kwargs = dict(kwargs or {})
        grounding_context = self._action_grounding_context()
        if (
            is_completed_verified_context(grounding_context)
            and is_local_machine_side_effect_action(function_name)
        ):
            grounding_result = {
                "status": "blocked_after_completion",
                "function_name": function_name,
                "kwargs": original_kwargs,
                "event": side_effect_after_completion_event(
                    function_name,
                    original_kwargs,
                    grounding_context,
                ),
            }
        else:
            grounding_result = validate_local_machine_action(
                function_name,
                kwargs,
                grounding_context,
            )
        grounding_status = str(grounding_result.get("status") or "valid")
        grounding_event = (
            dict(grounding_result.get("event") or {})
            if isinstance(grounding_result.get("event"), Mapping)
            else {}
        )
        if grounding_status == "repaired":
            function_name = str(grounding_result.get("function_name") or function_name)
            kwargs = dict(grounding_result.get("kwargs") or {})
            action_dict = self._replace_tool_call_kwargs(action_dict, function_name, kwargs)
        events: list[Dict[str, Any]] = []
        raw_result: Dict[str, Any]
        governance_request: Any | None = None
        governance_decision: ActionGovernanceDecision | None = None

        try:
            if grounding_status == "blocked_after_completion":
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.0,
                    state="SIDE_EFFECT_BLOCKED_AFTER_VERIFIED_COMPLETION",
                    success=False,
                    event_type="side_effect_after_verified_completion",
                    local_machine_action_grounding=dict(grounding_event),
                    action_grounding_status=grounding_status,
                )
            elif grounding_status == "invalid":
                raise MirrorScopeError(
                    str(
                        grounding_event.get("suggested_replan_reason")
                        or f"invalid kwargs for local-machine tool: {original_function_name}"
                    )
                )
            elif function_name in {"", "wait"}:
                raw_result = self._raw_success(function_name="wait", reward=0.0, state="WAIT")
            elif function_name in {"no_op_complete", "emit_final_report", "task_done"}:
                state = self._load_investigation_state()
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.0,
                    state="COMPLETED_VERIFIED_NOOP",
                    terminal_state=str(state.get("terminal_state") or ""),
                    completion_reason=str(state.get("completion_reason") or ""),
                    terminal_tick=state.get("terminal_tick"),
                    verified_completion=bool(state.get("verified_completion", False)),
                )
            elif function_name == "repo_tree":
                raw_result = self._act_repo_tree(kwargs)
            elif function_name == "repo_find":
                raw_result = self._act_repo_find(kwargs)
            elif function_name == "repo_grep":
                raw_result = self._act_repo_grep(kwargs)
            elif function_name == "file_read":
                raw_result = self._act_file_read(kwargs)
            elif function_name == "file_outline":
                raw_result = self._act_file_outline(kwargs)
            elif function_name == "file_summary":
                raw_result = self._act_file_summary(kwargs)
            elif function_name == "note_write":
                raw_result = self._act_note_write(kwargs)
            elif function_name == "hypothesis_add":
                raw_result = self._act_hypothesis_add(kwargs)
            elif function_name == "hypothesis_update":
                raw_result = self._act_hypothesis_update(kwargs)
            elif function_name == "hypothesis_compete":
                raw_result = self._act_hypothesis_compete(kwargs)
            elif function_name == "discriminating_test_add":
                raw_result = self._act_discriminating_test_add(kwargs)
            elif function_name in {"candidate_files_set", "candidate_files_update"}:
                raw_result = self._act_candidate_files_set(kwargs)
            elif function_name == "investigation_status":
                raw_result = self._act_investigation_status(kwargs)
            elif function_name == "propose_patch":
                governance_request, governance_decision = self._evaluate_action_governance(function_name, kwargs)
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                raw_result = self._act_propose_patch(kwargs)
            elif function_name == "apply_patch":
                governance_request, governance_decision = self._evaluate_action_governance(function_name, kwargs)
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                raw_result = self._act_apply_patch(kwargs)
            elif function_name == "edit_replace_range":
                governance_request, governance_decision = self._evaluate_action_governance(function_name, kwargs)
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                raw_result = self._act_edit_replace_range(kwargs)
            elif function_name == "edit_insert_after":
                governance_request, governance_decision = self._evaluate_action_governance(function_name, kwargs)
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                raw_result = self._act_edit_insert_after(kwargs)
            elif function_name == "create_file":
                governance_request, governance_decision = self._evaluate_action_governance(function_name, kwargs)
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                raw_result = self._act_create_file(kwargs)
            elif function_name == "delete_file":
                governance_request, governance_decision = self._evaluate_action_governance(function_name, kwargs)
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                raw_result = self._act_delete_file(kwargs)
            elif function_name == "run_test":
                raw_result = self._act_run_test(kwargs)
            elif function_name in {"run_lint", "run_typecheck", "run_build"}:
                raw_result = self._act_run_lint(kwargs, function_name=function_name)
            elif function_name in {"read_run_output", "read_test_failure"}:
                raw_result = self._act_read_run_output(kwargs)
            elif function_name == "mirror_acquire":
                mirror = acquire_relevant_files(
                    self.source_root,
                    self.mirror_root,
                    instruction=str(kwargs.get("instruction", "") or self.instruction),
                    candidate_paths=_string_list(kwargs.get("candidate_paths") or kwargs.get("candidates") or self.candidate_paths),
                    limit=int(kwargs.get("limit", 20) or 20),
                )
                selected = [
                    event.get("payload", {}).get("selected_paths", [])
                    for event in mirror.audit_events
                    if event.get("event_type") == "instruction_scoped_acquisition"
                ]
                selected_paths = list(selected[-1] if selected else [])
                self._acquire_attempted = True
                self._last_acquire_selected_count = len(selected_paths)
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.25 if int(mirror.to_manifest().get("workspace_file_count", 0) or 0) else 0.0,
                    state="FILES_ACQUIRED",
                    mirror_manifest=mirror.to_manifest(),
                    selected_paths=selected_paths,
                )
            elif function_name == "mirror_fetch":
                paths = _string_list(kwargs.get("paths") or kwargs.get("relative_paths") or kwargs.get("path") or self.fetch_paths)
                mirror = materialize_files(self.source_root, self.mirror_root, paths)
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.2 if paths else 0.0,
                    state="FILES_FETCHED",
                    mirror_manifest=mirror.to_manifest(),
                    fetched_paths=paths,
                )
            elif function_name == "internet_fetch":
                if not self.internet_enabled:
                    raise InternetIngressError("internet_fetch is disabled for this local-machine run")
                url = str(kwargs.get("url") or kwargs.get("source_url") or "").strip()
                if not url:
                    raise InternetIngressError("internet_fetch requires a url")
                governance_request, governance_decision = self._evaluate_action_governance(
                    function_name,
                    kwargs,
                    metadata={
                        "risk_level": "medium",
                        "network_kind": "http_fetch",
                        "private_networks_allowed": bool(self.internet_allow_private_networks),
                    },
                )
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise InternetIngressError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                filename = str(kwargs.get("filename") or kwargs.get("name") or "").strip() or None
                mirror = open_mirror(self.source_root, self.mirror_root)
                artifact = fetch_url(
                    url,
                    self._internet_output_root(mirror.to_manifest()),
                    policy=InternetIngressPolicy(
                        max_bytes=self.internet_max_bytes,
                        timeout_seconds=self.internet_timeout_seconds,
                        allow_private_networks=self.internet_allow_private_networks,
                    ),
                    filename=filename,
                )
                artifact_payload = artifact.to_dict()
                self._last_internet_artifact = artifact_payload
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.2,
                    state="INTERNET_ARTIFACT_FETCHED",
                    internet_artifact=artifact_payload,
                    internet_manifest=load_internet_manifest(self._internet_output_root(mirror.to_manifest())),
                )
            elif function_name == "internet_fetch_project":
                if not self.internet_enabled:
                    raise InternetIngressError("internet_fetch_project is disabled for this local-machine run")
                url = str(kwargs.get("url") or kwargs.get("source_url") or "").strip()
                if not url:
                    raise InternetIngressError("internet_fetch_project requires a url")
                governance_request, governance_decision = self._evaluate_action_governance(
                    function_name,
                    kwargs,
                    metadata={
                        "risk_level": "medium",
                        "network_kind": "project_fetch",
                        "private_networks_allowed": bool(self.internet_allow_private_networks),
                    },
                )
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise InternetIngressError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                mirror = open_mirror(self.source_root, self.mirror_root)
                output_root = self._internet_output_root(mirror.to_manifest())
                artifact = fetch_project(
                    url,
                    output_root,
                    policy=InternetIngressPolicy(
                        max_bytes=self.internet_max_bytes,
                        timeout_seconds=self.internet_timeout_seconds,
                        allow_private_networks=self.internet_allow_private_networks,
                    ),
                    source_type=str(kwargs.get("source_type") or kwargs.get("kind") or "auto"),
                    ref=str(kwargs.get("ref") or "").strip() or None,
                    depth=int(kwargs.get("depth", 1) or 1),
                    directory_name=str(kwargs.get("directory_name") or kwargs.get("name") or "").strip() or None,
                    timeout_seconds=float(kwargs.get("timeout_seconds", self.internet_timeout_seconds) or self.internet_timeout_seconds),
                )
                artifact_payload = artifact.to_dict()
                self._last_internet_artifact = artifact_payload
                materialized_payload = self._materialize_project_artifact(artifact_payload, mirror.to_manifest())
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.25,
                    state="INTERNET_PROJECT_FETCHED",
                    internet_artifact=artifact_payload,
                    materialized_project=materialized_payload,
                    internet_manifest=load_internet_manifest(output_root),
                )
            elif function_name == "mirror_exec":
                command = self._normalize_command(kwargs.get("command") or self.default_command)
                self._validate_mirror_exec_request(command, kwargs, generated_command=bool(kwargs.get("command")))
                governance_request, governance_decision = self._evaluate_action_governance(
                    function_name,
                    kwargs,
                    metadata={
                        "risk_level": "high" if kwargs.get("command") else "medium",
                        "generated_command": bool(kwargs.get("command")),
                        "purpose": str(kwargs.get("purpose") or ""),
                        "timeout_seconds_present": "timeout_seconds" in kwargs,
                        "bounded_target_present": bool(
                            str(kwargs.get("target") or kwargs.get("path") or kwargs.get("root") or "").strip()
                        ),
                    },
                )
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                allowed = _string_list(kwargs.get("allowed_commands") or self.allowed_commands)
                result = run_mirror_command(
                    self.source_root,
                    self.mirror_root,
                    command,
                    allowed_commands=allowed,
                    timeout_seconds=int(kwargs.get("timeout_seconds", self.default_command_timeout_seconds) or self.default_command_timeout_seconds),
                    backend=str(kwargs.get("backend") or self.execution_backend),
                    docker_image=str(kwargs.get("docker_image") or self.docker_image),
                    vm_provider=str(kwargs.get("vm_provider") or self.vm_provider),
                    vm_name=str(kwargs.get("vm_name") or self.vm_name),
                    vm_host=str(kwargs.get("vm_host") or self.vm_host),
                    vm_workdir=str(kwargs.get("vm_workdir") or self.vm_workdir),
                    vm_network_mode=str(kwargs.get("vm_network_mode") or self.vm_network_mode),
                    vm_sync_mode=str(kwargs.get("vm_sync_mode") or self.vm_sync_mode),
                    extra_env=self.extra_env,
                )
                self._command_executed = True
                self._last_command_returncode = int(result.returncode)
                self._command_failed = result.returncode != 0
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.35 if result.returncode == 0 else 0.0,
                    state="COMMAND_EXECUTED",
                    success=result.returncode == 0,
                    mirror_command=result.to_dict(),
                    sandbox_label=str(result.security_boundary or "best_effort_local_mirror"),
                    not_os_security_sandbox=not bool(result.real_vm_boundary),
                    execution_boundary=dict(result.execution_boundary),
                )
            elif function_name == "mirror_plan":
                plan_gate_state = self._load_investigation_state()
                meaningful_diff_count = self._meaningful_actionable_diff_count()
                has_verified_changes = (
                    meaningful_diff_count > 0
                    and str(plan_gate_state.get("terminal_state") or "") == "completed_verified"
                    and bool(plan_gate_state.get("verified_completion", False))
                )
                if not has_verified_changes and not self._command_failed:
                    raw_result = self._raw_success(
                        function_name=function_name,
                        reward=0.0,
                        state="MIRROR_PLAN_BLOCKED",
                        success=False,
                        event_type="mirror_plan_blocked",
                        mirror_plan_blocked_reason="no_verified_changes",
                        terminal_state=str(plan_gate_state.get("terminal_state") or ""),
                        verified_completion=bool(plan_gate_state.get("verified_completion", False)),
                        meaningful_actionable_diff_count=meaningful_diff_count,
                    )
                else:
                    plan = build_sync_plan(self.source_root, self.mirror_root)
                    self._last_plan = dict(plan)
                    actionable_count = len(list(plan.get("actionable_changes", []) or []))
                    waiting_approval = (
                        not bool(self.terminal_after_plan)
                        and actionable_count > 0
                        and not bool(self._command_failed)
                    )
                    if self.terminal_after_plan or not waiting_approval:
                        self._terminal = True
                    state = "WAITING_APPROVAL" if waiting_approval else "SYNC_PLAN_BUILT"
                    if self._command_failed:
                        state = "COMMAND_FAILED"
                    raw_result = self._raw_success(
                        function_name=function_name,
                        reward=0.0 if self._command_failed else (0.75 if actionable_count else 0.1),
                        state=state,
                        sync_plan=plan,
                        approval_status=str(plan.get("approval", {}).get("status", "") or ""),
                        waiting_approval=waiting_approval,
                        approval_request={
                            "type": "local_mirror_sync_plan",
                            "plan_id": str(plan.get("plan_id", "") or ""),
                            "approval_status": str(plan.get("approval", {}).get("status", "") or ""),
                            "actionable_change_count": actionable_count,
                            "source_root": str(self.source_root),
                            "mirror_root": str(self.mirror_root),
                        } if waiting_approval else {},
                    )
            elif function_name == "mirror_apply":
                plan_id = str(kwargs.get("plan_id", "") or self._last_plan.get("plan_id", "") or "")
                approved_by = str(kwargs.get("approved_by", "machine") or "machine")
                plan_for_governance = dict(self._last_plan or {})
                approval_payload = plan_for_governance.get("approval", {})
                approval_status = (
                    str(approval_payload.get("status", "") or "")
                    if isinstance(approval_payload, Mapping)
                    else ""
                )
                governance_request, governance_decision = self._evaluate_action_governance(
                    function_name,
                    kwargs,
                    metadata={
                        "changed_paths": self._sync_plan_changed_paths(plan_for_governance),
                        "approval_status": approval_status,
                        "approved_by": approved_by,
                        "risk_level": "high",
                    },
                )
                if governance_decision and governance_decision.status != "ALLOWED":
                    raise MirrorScopeError(
                        f"action governance blocked {function_name}: "
                        f"{governance_decision.blocked_reason or governance_decision.status}"
                    )
                result = apply_sync_plan(
                    self.source_root,
                    self.mirror_root,
                    plan_id=plan_id,
                    approved_by=approved_by,
                )
                self._applied = True
                self._terminal = True
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=1.0 if result.get("synced_files") else 0.2,
                    state="SYNC_PLAN_APPLIED",
                    sync_result=result,
                )
            else:
                raise MirrorScopeError(f"unsupported local-machine tool: {function_name}")
        except Exception as exc:
            raw_result = self._raw_failure(function_name=function_name, exc=exc, action=action_dict)
            events.append({"type": "local_machine_error", "function_name": function_name, "reason": str(exc)})
            if grounding_status == "invalid":
                raw_result["state"] = "INVALID_ACTION_KWARGS"
                raw_result["event_type"] = "invalid_action_kwargs"

        if grounding_status in {"repaired", "invalid", "blocked_after_completion"} and grounding_event:
            raw_result["local_machine_action_grounding"] = dict(grounding_event)
            raw_result["action_grounding_status"] = grounding_status
            events.append({"type": str(grounding_event.get("event_type") or grounding_status), **dict(grounding_event)})
        phase_event = self._update_action_grounding_state(
            function_name=function_name,
            kwargs=kwargs,
            raw_result=raw_result,
            grounding_event=grounding_event,
            original_function_name=original_function_name,
            original_kwargs=original_kwargs,
        )
        if phase_event:
            raw_result["local_machine_investigation_phase"] = phase_event
        if governance_decision is not None:
            raw_result["action_governance"] = governance_decision.to_dict()
            events.append(dict(governance_decision.audit_event))
            self._finalize_action_governance(governance_request, governance_decision, raw_result)
        evidence_event = self._record_formal_evidence(function_name, kwargs, raw_result, action_dict)
        if evidence_event:
            events.append(evidence_event)
        self._last_action = dict(action_dict)
        observation = self.observe()
        raw_result["terminal"] = bool(observation.terminal)
        raw_result["done"] = bool(observation.terminal)
        return ActionResult(
            ok=bool(raw_result.get("success", False)),
            observation=observation,
            raw=raw_result,
            events=events,
        )

    def decorate_candidate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        tools = self.observe().available_tools
        if not tools:
            return dict(action)
        tool_name = tools[0].name
        return {
            "kind": "call_tool",
            "function_name": tool_name,
            "kwargs": {},
            "payload": {
                "tool_name": "call_hidden_function",
                "tool_args": {"function_name": tool_name, "kwargs": {}},
            },
            "_source": "local_machine_default_tool",
            "_candidate_meta": {"decorated_from_wait": True},
        }

    def get_generic_task_spec(self) -> GenericTaskSpec:
        available = [tool.name for tool in self.observe().available_tools]
        return GenericTaskSpec(
            task_id=self.task_id,
            environment_family="local_machine",
            instruction=self.instruction,
            success_criteria=[
                "files are materialized only after explicit request",
                "source writes occur only through an approved sync plan",
            ],
            available_action_names=available,
            metadata={
                "source_root": str(self.source_root),
                "mirror_root": str(self.mirror_root),
                "adapter_version": LOCAL_MACHINE_ADAPTER_VERSION,
                "instruction": self.instruction,
                "default_command_present": bool(self.default_command),
                "allow_empty_exec": bool(self.allow_empty_exec),
                "default_command_timeout_seconds": int(self.default_command_timeout_seconds),
                "execution_backend": str(self.execution_backend),
                "docker_image": str(self.docker_image),
                "vm_provider": str(self.vm_provider),
                "vm_name": str(self.vm_name),
                "vm_host": str(self.vm_host),
                "vm_workdir": str(self.vm_workdir),
                "vm_network_mode": str(self.vm_network_mode),
                "vm_sync_mode": str(self.vm_sync_mode),
                "execution_boundary": self._execution_boundary_report(),
                "terminal_after_plan": bool(self.terminal_after_plan),
                "internet_enabled": bool(self.internet_enabled),
                "internet_max_bytes": int(self.internet_max_bytes),
                "internet_allow_private_networks": bool(self.internet_allow_private_networks),
                "learning_hints_present": bool(self.learning_context.get("hint_text")),
                "end_to_end_learning": dict(self.learning_context),
                "formal_evidence_db_path": str(self.evidence_db_path or ""),
                "action_governance_enabled": bool(self.action_governance_enabled),
                "action_governance_policy": self.action_governance_policy.to_dict(),
            },
        )

    def get_generic_observation(self) -> GenericObservation:
        obs = self.observe()
        mirror_state = dict(obs.raw.get("local_mirror", {}) if isinstance(obs.raw, dict) else {})
        return GenericObservation(
            observation_id=f"{self.task_id}:episode-{self._episode}",
            environment_family="local_machine",
            task_id=self.task_id,
            text=obs.text,
            available_actions=[
                GenericActionDescriptor(
                    name=tool.name,
                    action_family=str(tool.capability_class or "local_machine"),
                    parameter_schema=dict(tool.input_schema or {}),
                    enabled=True,
                    source="local_machine_adapter",
                    attributes={
                        "risk_level": tool.risk_level,
                        "approval_required": tool.approval_required,
                    },
                )
                for tool in obs.available_tools
            ],
            state=mirror_state,
            metadata={
                "source_root": str(self.source_root),
                "mirror_root": str(self.mirror_root),
                "instruction": self.instruction,
                "end_to_end_learning": dict(self.learning_context),
                "internet_enabled": bool(self.internet_enabled),
                "formal_evidence_db_path": str(self.evidence_db_path or ""),
            },
            raw=dict(obs.raw),
        )

    def _with_internet_tool(self, tools: list[ToolSpec]) -> list[ToolSpec]:
        if self.internet_enabled:
            existing = {tool.name for tool in tools}
            if "internet_fetch" not in existing:
                tools.append(self._tool_internet_fetch())
            if "internet_fetch_project" not in existing:
                tools.append(self._tool_internet_fetch_project())
        return tools

    def _execution_boundary_report(self) -> Dict[str, Any]:
        return execution_boundary_report(
            backend=self.execution_backend,
            docker_image=self.docker_image,
            vm_provider=self.vm_provider,
            vm_name=self.vm_name,
            vm_host=self.vm_host,
            vm_workdir=self.vm_workdir,
            vm_network_mode=self.vm_network_mode,
        )

    def _atomic_workflow_enabled(self) -> bool:
        return bool((self.allow_empty_exec or self._empty_first_open_investigation_enabled()) and not self.default_command)

    def _empty_first_open_investigation_enabled(self) -> bool:
        return bool(
            self.instruction
            and not self.candidate_paths
            and not self.fetch_paths
            and not self.default_command
        )

    @staticmethod
    def _has_investigation_progress(state: Mapping[str, Any]) -> bool:
        return any(
            bool(state.get(key))
            for key in (
                "last_tree",
                "last_search",
                "last_read",
                "notes",
                "hypotheses",
                "candidate_files",
                "last_run_ref",
            )
        )

    def _requires_meaningful_change_before_plan(self) -> bool:
        text = " ".join(
            str(value or "")
            for value in (
                self.instruction,
                " ".join(self.candidate_paths),
                " ".join(self.fetch_paths),
            )
        ).lower()
        return any(token in text for token in EDIT_INTENT_TOKENS)

    def _meaningful_actionable_diff_count(self) -> int:
        count = 0
        for entry in compute_mirror_diff(self.source_root, self.mirror_root):
            relative = str(getattr(entry, "relative_path", "") or "")
            status = str(getattr(entry, "status", "") or "")
            if status not in {"added", "modified"}:
                continue
            if is_generated_mirror_artifact(relative):
                continue
            count += 1
        return count

    def _atomic_tools(self, *, sync_plan: Mapping[str, Any] | None = None) -> list[ToolSpec]:
        tools = [
            self._tool_repo_tree(),
            self._tool_repo_find(),
            self._tool_repo_grep(),
            self._tool_file_read(),
            self._tool_file_outline(),
            self._tool_file_summary(),
            self._tool_note_write(),
            self._tool_hypothesis_add(),
            self._tool_hypothesis_update(),
            self._tool_hypothesis_compete(),
            self._tool_discriminating_test_add(),
            self._tool_candidate_files_set(),
            self._tool_investigation_status(),
            self._tool_propose_patch(),
            self._tool_apply_patch_atomic(),
            self._tool_edit_replace_range(),
            self._tool_edit_insert_after(),
            self._tool_create_file(),
            self._tool_delete_file(),
            self._tool_run_test(),
            self._tool_run_lint("run_lint"),
            self._tool_run_lint("run_typecheck"),
            self._tool_run_lint("run_build"),
            self._tool_read_run_output(),
            self._tool_read_test_failure(),
            self._tool_no_op_complete(),
        ]
        return self._filter_atomic_tools_for_state(tools, sync_plan=sync_plan or {})

    def _filter_atomic_tools_for_state(
        self,
        tools: Sequence[ToolSpec],
        *,
        sync_plan: Mapping[str, Any],
    ) -> list[ToolSpec]:
        state = self._load_investigation_state()
        if str(state.get("terminal_state") or "") in {"completed_verified", "needs_human_review"}:
            return [tool for tool in tools if tool.name == "no_op_complete"]
        hypotheses = [row for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        distinct_hypotheses = {
            str(row.get("hypothesis_id") or "").strip()
            for row in hypotheses
            if str(row.get("hypothesis_id") or "").strip()
        }
        can_compare_hypotheses = len(distinct_hypotheses) >= 2
        require_change = self._requires_meaningful_change_before_plan()
        has_meaningful_change = self._meaningful_actionable_diff_count() > 0
        filtered: list[ToolSpec] = []
        for tool in tools:
            if tool.name == "no_op_complete":
                continue
            if tool.name in {"read_run_output", "read_test_failure"} and not str(state.get("last_run_ref") or ""):
                continue
            if tool.name in {"hypothesis_compete", "discriminating_test_add"} and not can_compare_hypotheses:
                continue
            if (
                tool.name == "mirror_plan"
                and require_change
                and not has_meaningful_change
                and not self._command_failed
                and not sync_plan
            ):
                continue
            filtered.append(tool)
        return filtered

    def _available_tools(self, manifest: Dict[str, Any], sync_plan: Dict[str, Any]) -> list[ToolSpec]:
        if self._terminal and not (self.expose_apply_tool and sync_plan and not self._applied):
            return []

        if self._atomic_workflow_enabled():
            tools: list[ToolSpec] = []
            instruction = self.instruction.lower()
            research_first = self.internet_enabled and any(
                token in instruction
                for token in ("market", "research", "trend", "competitor", "调研", "市场", "竞品")
            )
            if research_first:
                tools = self._with_internet_tool(tools)
            if (
                self._empty_first_open_investigation_enabled()
                and int(manifest.get("workspace_file_count", 0) or 0) <= 0
                and not self._has_investigation_progress(self._load_investigation_state())
            ):
                tools.append(self._tool_repo_tree())
                if self.allow_empty_exec and not self._command_executed:
                    tools.append(self._tool_exec())
                return self._with_internet_tool(tools)
            tools.extend(self._atomic_tools(sync_plan=sync_plan))
            if not research_first:
                tools = self._with_internet_tool(tools)
            if (self.default_command or self.allow_empty_exec) and not self._command_executed:
                tools.append(self._tool_exec())
            if not sync_plan:
                tools.append(self._tool_plan())
            if self.expose_apply_tool and sync_plan and not self._applied:
                tools.append(self._tool_apply())
            return tools

        workspace_count = int(manifest.get("workspace_file_count", 0) or 0)
        if workspace_count <= 0:
            tools: list[ToolSpec] = []
            can_empty_exec = bool(self.allow_empty_exec and not self._command_executed)
            if can_empty_exec:
                tools.append(self._tool_exec())
            if self.instruction or self.candidate_paths:
                skip_repeated_empty_acquire = (
                    self._acquire_attempted
                    and self._last_acquire_selected_count <= 0
                    and bool(self.default_command)
                )
                if not skip_repeated_empty_acquire:
                    tools.append(self._tool_acquire())
            if self.fetch_paths and not tools:
                tools.append(self._tool_fetch())
            if self._command_executed and not sync_plan:
                tools.append(self._tool_plan())
            return self._with_internet_tool(tools)

        if (self.default_command or self.internet_enabled or self.allow_empty_exec) and not self._command_executed:
            return self._with_internet_tool([self._tool_exec()])
        if not sync_plan:
            return self._with_internet_tool([self._tool_plan()])
        if self.expose_apply_tool and not self._applied:
            return self._with_internet_tool([self._tool_apply()])
        return self._with_internet_tool([])

    def _internet_output_root(self, manifest: Dict[str, Any]) -> Path:
        if self.internet_output_root is not None:
            return self.internet_output_root
        control_root = str(manifest.get("control_root", "") or "")
        root = Path(control_root) if control_root else self.mirror_root / "control"
        return root / "internet"

    def _materialize_project_artifact(self, artifact: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
        source_path = Path(str(artifact.get("local_path", "") or ""))
        if not source_path.exists() or not source_path.is_dir():
            return {}
        workspace_root = Path(str(manifest.get("workspace_root", "") or self.mirror_root / "workspace"))
        destination = workspace_root / "internet_projects" / source_path.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, destination, ignore=shutil.ignore_patterns(".git"))
        mirror = open_mirror(self.source_root, self.mirror_root)
        relative_path = destination.relative_to(mirror.workspace_root).as_posix()
        mirror.external_baselines[relative_path] = {
            "schema_version": "conos.local_machine.external_baseline/v1",
            "baseline_id": _json_hash(
                {
                    "artifact_id": artifact.get("artifact_id", ""),
                    "workspace_relative_path": relative_path,
                    "baseline_path": str(source_path.resolve()),
                }
            )[:16],
            "artifact_id": str(artifact.get("artifact_id", "") or ""),
            "fetch_kind": str(artifact.get("fetch_kind", "") or ""),
            "workspace_relative_path": relative_path,
            "baseline_path": str(source_path.resolve()),
            "created_at": _now(),
        }
        mirror.audit_events.append(
            {
                "schema_version": LOCAL_MACHINE_ADAPTER_VERSION,
                "event_type": "internet_project_materialized",
                "created_at": _now(),
                "payload": {
                    "artifact_id": str(artifact.get("artifact_id", "") or ""),
                    "fetch_kind": str(artifact.get("fetch_kind", "") or ""),
                    "workspace_relative_path": relative_path,
                    "workspace_path": str(destination.resolve()),
                },
            }
        )
        mirror.save_manifest()
        return {
            "artifact_id": str(artifact.get("artifact_id", "") or ""),
            "workspace_relative_path": relative_path,
            "workspace_path": str(destination.resolve()),
        }

    def _control_root(self) -> Path:
        mirror = open_mirror(self.source_root, self.mirror_root)
        mirror.control_root.mkdir(parents=True, exist_ok=True)
        return mirror.control_root

    def _workspace_root(self) -> Path:
        mirror = open_mirror(self.source_root, self.mirror_root)
        mirror.workspace_root.mkdir(parents=True, exist_ok=True)
        return mirror.workspace_root

    def _investigation_state_path(self) -> Path:
        return self._control_root() / "investigation_state.json"

    def _run_output_root(self) -> Path:
        root = self._control_root() / "run_outputs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _formal_evidence_ledger_path(self) -> Path:
        return self._control_root() / "formal_evidence_ledger.jsonl"

    def _formal_evidence_summary(self) -> Dict[str, Any]:
        path = self._formal_evidence_ledger_path()
        if not path.exists():
            return {
                "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
                "path": str(path.resolve()),
                "entry_count": 0,
                "last_evidence_id": "",
                "recent": [],
            }
        recent_rows: deque[Dict[str, Any]] = deque(maxlen=5)
        count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    count += 1
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        recent_rows.append(
                            {
                                "evidence_id": str(payload.get("evidence_id") or ""),
                                "claim": str(payload.get("claim") or "")[:240],
                                "evidence_type": str(payload.get("evidence_type") or ""),
                                "status": str(payload.get("status") or ""),
                                "source_refs": list(payload.get("source_refs", []) or [])[:10],
                                "ledger_hash": str(payload.get("ledger_hash") or ""),
                            }
                        )
        except OSError:
            return {
                "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
                "path": str(path.resolve()),
                "entry_count": 0,
                "last_evidence_id": "",
                "recent": [],
                "error": "ledger_read_failed",
            }
        recent = list(recent_rows)
        return {
            "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
            "path": str(path.resolve()),
            "entry_count": count,
            "last_evidence_id": recent[-1]["evidence_id"] if recent else "",
            "recent": recent,
        }

    def _record_formal_evidence(
        self,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Dict[str, Any],
        action_dict: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state_store: RuntimeStateStore | None = None
        try:
            hypotheses = list(self._load_investigation_state().get("hypotheses", []) or [])
            if self.evidence_db_path is not None:
                state_store = RuntimeStateStore(self.evidence_db_path)
            entry = build_local_machine_evidence_entry(
                run_id=self.task_id,
                task_family="local_machine",
                instruction=self.instruction,
                action_name=function_name,
                action_args=dict(kwargs or {}),
                action_input=dict(action_dict or {}),
                result=dict(raw_result),
                hypotheses=[row for row in hypotheses if isinstance(row, Mapping)],
            )
            recorded = FormalEvidenceLedger(
                self._formal_evidence_ledger_path(),
                state_store=state_store,
            ).record(entry)
            raw_result["formal_evidence_id"] = recorded["evidence_id"]
            raw_result["formal_evidence_ref"] = {
                "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
                "evidence_id": recorded["evidence_id"],
                "ledger_hash": recorded["ledger_hash"],
                "path": str(self._formal_evidence_ledger_path().resolve()),
            }
            return {
                "type": "formal_evidence_recorded",
                "evidence_id": recorded["evidence_id"],
                "evidence_type": recorded["evidence_type"],
                "ledger_hash": recorded["ledger_hash"],
            }
        except Exception as exc:
            raw_result["formal_evidence_error"] = str(exc)
            return {
                "type": "formal_evidence_error",
                "function_name": str(function_name),
                "reason": str(exc),
            }
        finally:
            if state_store is not None:
                state_store.close()

    def _load_investigation_state(self) -> Dict[str, Any]:
        state = _load_json(self._investigation_state_path())
        if not state:
            state = {
                "schema_version": LOCAL_MACHINE_INVESTIGATION_VERSION,
                "created_at": _now(),
                "updated_at": _now(),
                "notes": [],
                "hypotheses": [],
                "hypothesis_events": [],
                "hypothesis_lifecycle": {},
                "discriminating_tests": [],
                "candidate_files": [],
                "candidate_reason": "",
                "last_tree": {},
                "last_search": {},
                "last_read": {},
                "last_run_ref": "",
                "validation_runs": [],
                "action_governance": {},
                "investigation_phase": "discover",
                "terminal_state": "",
                "completion_reason": "",
                "terminal_tick": None,
                "verified_completion": False,
                "action_count": 0,
                "grounding": {},
                "target_binding": {},
                "patch_proposals": [],
                "action_history": [],
                "stalled_events": [],
            }
        state.setdefault("schema_version", LOCAL_MACHINE_INVESTIGATION_VERSION)
        state.setdefault("notes", [])
        state.setdefault("hypotheses", [])
        state.setdefault("hypothesis_events", [])
        state.setdefault("hypothesis_lifecycle", {})
        state.setdefault("discriminating_tests", [])
        state.setdefault("candidate_files", [])
        state.setdefault("last_tree", {})
        state.setdefault("last_search", {})
        state.setdefault("last_read", {})
        state.setdefault("read_files", [])
        state.setdefault("validation_runs", [])
        state.setdefault("action_governance", {})
        state.setdefault("investigation_phase", "discover")
        state.setdefault("terminal_state", "")
        state.setdefault("completion_reason", "")
        state.setdefault("terminal_tick", None)
        state.setdefault("verified_completion", False)
        state.setdefault("action_count", 0)
        state.setdefault("grounding", {})
        state.setdefault("target_binding", {})
        state.setdefault("patch_proposals", [])
        state.setdefault("action_history", [])
        state.setdefault("stalled_events", [])
        return state

    def _save_investigation_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(state)
        payload["schema_version"] = LOCAL_MACHINE_INVESTIGATION_VERSION
        payload["updated_at"] = _now()
        path = self._investigation_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return payload

    def _action_grounding_context(self, *, state_override: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        state = dict(state_override or self._load_investigation_state())
        return {
            "instruction": self.instruction,
            "source_root": str(self.source_root),
            "mirror_root": str(self.mirror_root),
            "workspace_root": str(self._workspace_root()),
            "run_output_root": str(self._run_output_root()),
            "investigation_state": state,
            "llm_budget": self._llm_budget_report(state),
        }

    def _llm_budget_summary(self) -> Dict[str, Any]:
        ledger = self.llm_cost_ledger
        if ledger is not None and hasattr(ledger, "summary"):
            try:
                return dict(ledger.summary())
            except Exception:
                return {}
        return {}

    def _llm_budget_report(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        return budget_policy_report(state, budget_summary=self._llm_budget_summary())

    @staticmethod
    def _replace_tool_call_kwargs(action: Dict[str, Any], function_name: str, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        updated = dict(action)
        updated["function_name"] = str(function_name)
        updated["kwargs"] = dict(kwargs)
        payload = dict(updated.get("payload", {}) if isinstance(updated.get("payload", {}), dict) else {})
        tool_args = dict(payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {})
        tool_args["function_name"] = str(function_name)
        tool_args["kwargs"] = dict(kwargs)
        payload["tool_name"] = str(payload.get("tool_name") or "call_hidden_function")
        payload["tool_args"] = tool_args
        updated["payload"] = payload
        return updated

    def _detect_investigation_stall(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        history = [
            dict(row)
            for row in list(state.get("action_history", []) or [])
            if isinstance(row, dict)
        ]
        if len(history) < 4 or self._meaningful_actionable_diff_count() > 0:
            return {}
        recent = history[-4:]
        repeated_actions = {"repo_grep", "read_test_failure", "run_test"}
        repeated_count = sum(1 for row in recent if str(row.get("function_name") or "") in repeated_actions)
        new_file_read = any(str(row.get("function_name") or "") == "file_read" for row in recent)
        if repeated_count < 3 or new_file_read:
            return {}
        binding = dict(state.get("target_binding", {}) or {})
        recommended = "target_binding_or_patch_proposal"
        top_target = str(binding.get("top_target_file") or "")
        if top_target:
            read_paths = {
                str(row.get("path") or "")
                for row in list(state.get("read_files", []) or [])
                if isinstance(row, dict)
            }
            recommended = "propose_patch" if top_target in read_paths else "file_read"
        return {
            "schema_version": LOCAL_MACHINE_INVESTIGATION_VERSION,
            "event_type": "investigation_stalled",
            "recommended_action": recommended,
            "top_target_file": top_target,
            "target_confidence": binding.get("target_confidence", 0.0),
            "recent_actions": [str(row.get("function_name") or "") for row in recent],
            "created_at": _now(),
        }

    def _closed_loop_probe_variant(self) -> str:
        match = re.search(r"\[closed_loop_probe_variant=([A-Za-z0-9_\-]+)\]", self.instruction)
        return str(match.group(1)) if match else "full"

    @staticmethod
    def _latest_failed_validation_run(state: Mapping[str, Any]) -> Dict[str, Any]:
        for row in reversed(list(state.get("validation_runs", []) or [])):
            if isinstance(row, Mapping) and not bool(row.get("success", False)):
                return dict(row)
        return {}

    @staticmethod
    def _candidate_rows_from_binding(binding: Mapping[str, Any]) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for item in list(binding.get("target_file_candidates", []) or []):
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("target_file") or "").strip()
            if not path or path.startswith("tests/") or not path.endswith(".py"):
                continue
            rows.append(
                {
                    "target_file": path,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "reasons": _string_list(item.get("reasons")),
                }
            )
        return rows

    @staticmethod
    def _surface_and_root_targets(binding: Mapping[str, Any], target_file: str) -> tuple[str, str, bool]:
        candidates = LocalMachineSurfaceAdapter._candidate_rows_from_binding(binding)
        top_target = str(target_file or binding.get("top_target_file") or "").strip()
        if not top_target and candidates:
            top_target = str(candidates[0].get("target_file") or "")
        traceback_files = [
            str(path)
            for path in _string_list(binding.get("traceback_files"))
            if str(path).endswith(".py") and not str(path).startswith("tests/")
        ]
        surface = ""
        for path in traceback_files:
            if path and path != top_target:
                surface = path
                break
        if not surface:
            for row in candidates:
                reasons = " ".join(_string_list(row.get("reasons"))).lower()
                path = str(row.get("target_file") or "")
                if path and path != top_target and "traceback" in reasons:
                    surface = path
                    break
        if not surface:
            for row in candidates:
                path = str(row.get("target_file") or "")
                if path and path != top_target:
                    surface = path
                    break
        unresolved_needed = False
        if not surface and top_target:
            surface = top_target
            unresolved_needed = True
        return surface, top_target, unresolved_needed

    def _explicit_hypothesis_id(self, hypothesis_type: str, target_file: str) -> str:
        return f"hyp_{hypothesis_type}_{_json_hash({'task_id': self.task_id, 'target_file': target_file})[:12]}"

    @staticmethod
    def _augment_explicit_hypothesis_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
        hypothesis = dict(row)
        metadata = dict(hypothesis.get("metadata", {}) or {})
        target_file = str(metadata.get("target_file") or hypothesis.get("target_file") or "")
        hypothesis_type = str(metadata.get("hypothesis_type") or hypothesis.get("hypothesis_type") or hypothesis.get("family") or "codebase")
        summary = str(hypothesis.get("summary") or hypothesis.get("claim") or "")
        hypothesis["summary"] = summary
        hypothesis["target_file"] = target_file
        hypothesis["hypothesis_type"] = hypothesis_type
        hypothesis["status"] = str(hypothesis.get("status") or "active")
        hypothesis["posterior"] = float(hypothesis.get("posterior", hypothesis.get("confidence", 0.5)) or 0.5)
        hypothesis["predictions"] = dict(hypothesis.get("predictions", {}) or {})
        hypothesis["predicted_observation_tokens"] = _string_list(
            hypothesis.get("predicted_observation_tokens")
            or metadata.get("predicted_observation_tokens")
            or hypothesis["predictions"].get("predicted_observation_tokens")
        )
        predicted_effects = hypothesis.get("predicted_action_effects") or metadata.get("predicted_action_effects")
        hypothesis["predicted_action_effects"] = dict(predicted_effects or {})
        falsifiers = hypothesis.get("falsifiers", {})
        hypothesis["falsifiers"] = dict(falsifiers or {}) if isinstance(falsifiers, Mapping) else {"observations": _string_list(falsifiers)}
        conflicts = _string_list(hypothesis.get("conflicts_with") or hypothesis.get("competing_with"))
        hypothesis["conflicts_with"] = conflicts
        hypothesis["competing_with"] = conflicts
        hypothesis["evidence_refs"] = _string_list(hypothesis.get("evidence_refs"))
        hypothesis["metadata"] = metadata
        return hypothesis

    def _make_explicit_hypothesis(
        self,
        *,
        hypothesis_type: str,
        target_file: str,
        binding: Mapping[str, Any],
        evidence_refs: Sequence[str],
        confidence: float,
    ) -> Dict[str, Any]:
        failed_test = str(binding.get("latest_failed_test_target") or "")
        failure_symbols = _string_list(binding.get("failure_symbols"))[:6]
        target = str(target_file or "").strip()
        if hypothesis_type == "surface_wrapper":
            summary = f"Surface wrapper {target or '<unknown>'} misroutes or transforms data before the observed failing test."
            observation_tokens = [Path(target).stem, "traceback", "wrapper", "surface"]
            action_effects = {
                "file_read": f"Reading {target} should reveal the wrapper call path or lossy transform.",
                "run_test": "A direct downstream test may continue to fail if the root cause is deeper.",
            }
            falsifiers = {
                "observations": [
                    "A downstream direct test fails without implicating this wrapper.",
                    "A bounded patch to this wrapper does not pass the full verifier suite.",
                ],
                "tests": [failed_test],
            }
            prior = confidence
        elif hypothesis_type == "downstream_root_cause":
            summary = f"Downstream implementation {target or '<unknown>'} contains the root cause behind the failing behavior."
            observation_tokens = [Path(target).stem, "downstream", "root", *failure_symbols[:3]]
            action_effects = {
                "file_read": f"Reading {target} should expose logic tied to the failing assertion or symbol.",
                "propose_patch": "A bounded diff on this target should be verifier-gated by targeted and full tests.",
            }
            falsifiers = {
                "observations": [
                    "The target has no relevant symbol, import path, or assertion-linked behavior.",
                    "A verifier-gated bounded patch to this target is rejected.",
                ],
                "tests": [failed_test],
            }
            prior = confidence
        else:
            summary = "The current evidence is underspecified; no source patch is safe until more disambiguating evidence is collected."
            observation_tokens = ["ambiguous", "underspecified", *failure_symbols[:3]]
            action_effects = {
                "file_read": "Additional source or test reads should clarify which hypothesis is actionable.",
                "propose_patch": "Patch proposal should refuse if evidence remains insufficient.",
            }
            falsifiers = {
                "observations": [
                    "A target binding reaches sufficient confidence with direct evidence.",
                    "A discriminating test separates competing source-level explanations.",
                ],
                "tests": [failed_test],
            }
            prior = confidence
        hypothesis = normalize_hypothesis(
            hypothesis_id=self._explicit_hypothesis_id(hypothesis_type, target),
            run_id=self.task_id,
            task_family="local_machine",
            family=hypothesis_type,
            claim=summary,
            confidence=prior,
            evidence_refs=evidence_refs,
            predictions={
                "target_file": target,
                "failed_test": failed_test,
                "failure_symbols": failure_symbols,
                "predicted_observation_tokens": observation_tokens,
            },
            falsifiers=falsifiers,
            metadata={
                "source": "local_machine_explicit_hypothesis_lifecycle",
                "created_at_iso": _now(),
                "target_file": target,
                "hypothesis_type": hypothesis_type,
                "binding_reasons": _string_list(binding.get("binding_reasons")),
                "predicted_action_effects": action_effects,
                "predicted_observation_tokens": observation_tokens,
            },
        )
        hypothesis["summary"] = summary
        hypothesis["target_file"] = target
        hypothesis["hypothesis_type"] = hypothesis_type
        hypothesis["predicted_observation_tokens"] = observation_tokens
        hypothesis["predicted_action_effects"] = action_effects
        hypothesis["conflicts_with"] = []
        return self._augment_explicit_hypothesis_fields(hypothesis)

    def _append_hypothesis_lifecycle_event(self, state: Dict[str, Any], event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("schema_version", HYPOTHESIS_LIFECYCLE_VERSION)
        payload.setdefault("run_id", self.task_id)
        payload.setdefault("created_at", _now())
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        event_key = _json_hash(payload)
        existing_keys = {
            str(row.get("_event_key") or _json_hash(row))
            for row in events[-200:]
            if isinstance(row, Mapping)
        }
        if event_key in existing_keys:
            return
        payload["_event_key"] = event_key
        events.append(payload)
        state["hypothesis_events"] = events[-200:]

    def _sync_hypothesis_conflicts(self, hypotheses: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
        rows = [self._augment_explicit_hypothesis_fields(row) for row in list(hypotheses or []) if isinstance(row, Mapping)]
        ids = [str(row.get("hypothesis_id") or "") for row in rows if str(row.get("hypothesis_id") or "")]
        for left in ids:
            for right in ids:
                if left and right and left != right:
                    rows = mark_competing(rows, left, right)
        return [self._augment_explicit_hypothesis_fields(row) for row in rows]

    def _ensure_bound_discriminating_test(
        self,
        state: Dict[str, Any],
        hypotheses: Sequence[Mapping[str, Any]],
        binding: Mapping[str, Any],
    ) -> None:
        if self._closed_loop_probe_variant() == "no_discriminating_experiment":
            return
        rows = [self._augment_explicit_hypothesis_fields(row) for row in hypotheses if isinstance(row, Mapping)]
        if len(rows) < 2:
            return
        left, right = rows[0], rows[1]
        left_id = str(left.get("hypothesis_id") or "")
        right_id = str(right.get("hypothesis_id") or "")
        if not left_id or not right_id or left_id == right_id:
            return
        tests = [dict(row) for row in list(state.get("discriminating_tests", []) or []) if isinstance(row, dict)]
        for test in tests:
            refs = _string_list(test.get("discriminates_between") or test.get("hypotheses") or [test.get("hypothesis_a"), test.get("hypothesis_b")])
            if {left_id, right_id}.issubset(set(refs)):
                return
        failed_target = str(binding.get("latest_failed_test_target") or ".") or "."
        action = {"action": "run_test", "args": {"target": failed_target, "timeout_seconds": 30}}
        test = build_discriminating_test(
            hypothesis_a=left,
            hypothesis_b=right,
            action=action,
            expected_if_a=f"Evidence continues to implicate {left.get('target_file') or 'the surface path'} and weakens {right_id}.",
            expected_if_b=f"Evidence implicates {right.get('target_file') or 'the downstream path'} and weakens {left_id}.",
            why="The same failing behavior has at least two source-level explanations; the targeted failing test and related reads separate wrapper versus root-cause predictions.",
        )
        test["discriminates_between"] = [left_id, right_id]
        test["expected_outcomes_by_hypothesis"] = {
            left_id: str(test.get("expected_if_a") or ""),
            right_id: str(test.get("expected_if_b") or ""),
        }
        test["expected_information_gain"] = 0.42
        tests.append(test)
        state["discriminating_tests"] = tests[-100:]
        self._append_hypothesis_lifecycle_event(
            state,
            {
                "event_type": "discriminating_test_bound_to_hypotheses",
                "hypotheses": [left_id, right_id],
                "discriminates_between": [left_id, right_id],
                "test_id": test["test_id"],
                "delta": 0.0,
            },
        )

    def _update_explicit_hypothesis_posteriors(
        self,
        state: Dict[str, Any],
        *,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        binding: Mapping[str, Any],
        root_target: str,
        surface_target: str,
        evidence_refs: Sequence[str],
    ) -> list[Dict[str, Any]]:
        hypotheses = [
            self._augment_explicit_hypothesis_fields(row)
            for row in list(state.get("hypotheses", []) or [])
            if isinstance(row, Mapping)
        ]
        if self._closed_loop_probe_variant() == "no_posterior":
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return hypotheses
        success = bool(raw_result.get("success", False))
        state_name = str(raw_result.get("state", "") or "")
        run_ref = str(raw_result.get("run_ref") or state.get("last_run_ref") or "")
        evidence_key = _json_hash(
            {
                "function": function_name,
                "success": success,
                "state": state_name,
                "run_ref": run_ref,
                "patch_sha256": raw_result.get("patch_sha256", ""),
                "root_target": root_target,
                "surface_target": surface_target,
            }
        )
        applied_keys = set(_string_list(state.get("hypothesis_evidence_keys")))
        if evidence_key in applied_keys:
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return hypotheses
        should_update = bool(
            (function_name == "run_test" and not success)
            or (function_name in {"propose_patch", "apply_patch"} and success)
            or (function_name == "propose_patch" and bool(raw_result.get("needs_human_review", False)))
        )
        if not should_update:
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return hypotheses
        updated_rows: list[Dict[str, Any]] = []
        for hypothesis in hypotheses:
            htype = str(hypothesis.get("hypothesis_type") or "")
            target = str(hypothesis.get("target_file") or "")
            signal = "neutral"
            strength = 0.08
            rationale = "Evidence observed but did not change this hypothesis."
            if function_name == "run_test" and not success:
                if htype == "downstream_root_cause" and target and target == root_target:
                    signal = "support"
                    strength = 0.24
                    rationale = "Failure evidence plus target binding favors the downstream/root-cause explanation."
                elif htype == "surface_wrapper" and root_target and surface_target and root_target != surface_target:
                    signal = "contradict"
                    strength = 0.22
                    rationale = "Failure evidence points past the surface wrapper toward a downstream target."
                elif htype == "unresolved_spec":
                    signal = "contradict" if root_target else "support"
                    strength = 0.1
                    rationale = "Target binding reduced ambiguity." if root_target else "Evidence remains underspecified."
            elif function_name in {"propose_patch", "apply_patch"} and success:
                touched = set(_string_list(raw_result.get("touched_files")))
                if target and target in touched:
                    signal = "support"
                    strength = 0.2
                    rationale = "Verifier-gated patch touched this hypothesis target and succeeded."
                elif htype == "unresolved_spec":
                    signal = "contradict"
                    strength = 0.12
                    rationale = "A verified source patch means evidence was sufficient."
            elif function_name == "propose_patch" and bool(raw_result.get("needs_human_review", False)):
                signal = "support" if htype == "unresolved_spec" else "contradict"
                strength = 0.14
                rationale = "Patch proposal refused because evidence remained insufficient."
            if signal == "neutral":
                updated_rows.append(hypothesis)
                continue
            updated, event = apply_hypothesis_evidence(
                hypothesis,
                signal=signal,
                evidence_refs=evidence_refs,
                strength=strength,
                rationale=rationale,
            )
            event["run_id"] = self.task_id
            event["target_file"] = target
            event["binding_top_target_file"] = root_target
            event["source_function"] = str(function_name)
            updated = self._augment_explicit_hypothesis_fields(updated)
            updated_rows.append(updated)
            self._append_hypothesis_lifecycle_event(state, event)
            self._persist_hypothesis_lifecycle(updated, event)
        applied_keys.add(evidence_key)
        state["hypothesis_evidence_keys"] = sorted(applied_keys)[-200:]
        state["hypotheses"] = updated_rows[-100:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(updated_rows)
        return updated_rows

    def _ensure_explicit_hypothesis_lifecycle(
        self,
        state: Dict[str, Any],
        *,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        binding: Mapping[str, Any],
        target_file: str,
    ) -> Dict[str, Any]:
        if not isinstance(binding, Mapping) or not (
            binding.get("target_file_candidates")
            or binding.get("top_target_file")
            or binding.get("traceback_files")
        ):
            hypotheses = [
                self._augment_explicit_hypothesis_fields(row)
                for row in list(state.get("hypotheses", []) or [])
                if isinstance(row, Mapping)
            ]
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return state
        failed_run = self._latest_failed_validation_run(state)
        run_ref = str(failed_run.get("run_ref") or raw_result.get("run_ref") or state.get("last_run_ref") or "")
        evidence_refs = [f"failure:{run_ref}"] if run_ref else []
        failed_target = str(binding.get("latest_failed_test_target") or "")
        if failed_target:
            evidence_refs.append(f"test:{failed_target}")
        surface_target, root_target, unresolved_needed = self._surface_and_root_targets(binding, target_file)
        hypotheses = [
            self._augment_explicit_hypothesis_fields(row)
            for row in list(state.get("hypotheses", []) or [])
            if isinstance(row, Mapping)
        ]
        by_id = {str(row.get("hypothesis_id") or ""): row for row in hypotheses}
        desired: list[Dict[str, Any]] = []
        if surface_target:
            desired.append(
                self._make_explicit_hypothesis(
                    hypothesis_type="surface_wrapper",
                    target_file=surface_target,
                    binding=binding,
                    evidence_refs=evidence_refs,
                    confidence=0.46 if root_target and root_target != surface_target else 0.5,
                )
            )
        if root_target and root_target != surface_target:
            desired.append(
                self._make_explicit_hypothesis(
                    hypothesis_type="downstream_root_cause",
                    target_file=root_target,
                    binding=binding,
                    evidence_refs=evidence_refs,
                    confidence=0.54,
                )
            )
        if unresolved_needed or len(desired) < 2:
            desired.append(
                self._make_explicit_hypothesis(
                    hypothesis_type="unresolved_spec",
                    target_file="",
                    binding=binding,
                    evidence_refs=evidence_refs,
                    confidence=0.42,
                )
            )
        created_any = False
        for hypothesis in desired:
            hid = str(hypothesis.get("hypothesis_id") or "")
            if hid in by_id:
                current = self._augment_explicit_hypothesis_fields({**hypothesis, **by_id[hid], "metadata": {**dict(hypothesis.get("metadata", {}) or {}), **dict(by_id[hid].get("metadata", {}) or {})}})
                by_id[hid] = current
                continue
            by_id[hid] = hypothesis
            created_any = True
            self._append_hypothesis_lifecycle_event(
                state,
                {
                    "event_type": "hypothesis_created",
                    "hypothesis_id": hid,
                    "hypothesis_type": hypothesis.get("hypothesis_type"),
                    "target_file": hypothesis.get("target_file"),
                    "evidence_refs": list(hypothesis.get("evidence_refs", []) or []),
                    "delta": 0.0,
                },
            )
            self._persist_hypothesis_lifecycle(hypothesis, {"hypothesis_id": hid, "event_type": "hypothesis_created", "evidence_refs": evidence_refs, "delta": 0.0})
        hypotheses = self._sync_hypothesis_conflicts(list(by_id.values()))
        if created_any and len(hypotheses) >= 2:
            ids = [str(row.get("hypothesis_id") or "") for row in hypotheses[:2]]
            self._append_hypothesis_lifecycle_event(
                state,
                {
                    "event_type": "hypothesis_competition_recorded",
                    "hypotheses": ids,
                    "delta": 0.0,
                    "reason": "Automatic lifecycle construction found multiple plausible source-level explanations.",
                },
            )
        state["hypotheses"] = hypotheses[-100:]
        self._ensure_bound_discriminating_test(state, hypotheses, binding)
        hypotheses = self._update_explicit_hypothesis_posteriors(
            state,
            function_name=function_name,
            kwargs=kwargs,
            raw_result=raw_result,
            binding=binding,
            root_target=root_target,
            surface_target=surface_target,
            evidence_refs=evidence_refs,
        )
        state["hypotheses"] = hypotheses[-100:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        return state

    def _leading_hypothesis_from_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        hypotheses = [
            self._augment_explicit_hypothesis_fields(row)
            for row in list(state.get("hypotheses", []) or [])
            if isinstance(row, Mapping)
        ]
        if not hypotheses:
            return {}
        lifecycle = hypothesis_lifecycle_summary(hypotheses)
        leading_id = str(lifecycle.get("leading_hypothesis_id") or "")
        for row in hypotheses:
            if str(row.get("hypothesis_id") or "") == leading_id:
                return row
        return sorted(hypotheses, key=lambda row: float(row.get("posterior", 0.0) or 0.0), reverse=True)[0]

    def _update_action_grounding_state(
        self,
        *,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        grounding_event: Mapping[str, Any] | None,
        original_function_name: str,
        original_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state = self._load_investigation_state()
        phase_before = str(state.get("investigation_phase") or "discover")
        action_count = int(state.get("action_count", 0) or 0) + 1
        state["action_count"] = action_count
        action_history = [
            dict(row)
            for row in list(state.get("action_history", []) or [])
            if isinstance(row, dict)
        ]
        action_history.append(
            {
                "tick": action_count,
                "function_name": str(function_name or ""),
                "original_function_name": str(original_function_name or ""),
                "success": bool(raw_result.get("success", False)),
                "state": str(raw_result.get("state", "") or ""),
                "path": str(raw_result.get("path") or kwargs.get("path") or kwargs.get("target_file") or ""),
                "created_at": _now(),
            }
        )
        state["action_history"] = action_history[-100:]
        grounding = dict(state.get("grounding", {}) or {})
        grounding.setdefault("events", [])
        grounding.setdefault("invalid_action_kwargs_events", [])
        grounding.setdefault("repaired_actions", [])
        grounding.setdefault("side_effect_after_verified_completion_events", [])
        if not original_kwargs and str(original_function_name or "") not in {"", "wait"}:
            grounding["empty_kwargs_attempt_count"] = int(grounding.get("empty_kwargs_attempt_count", 0) or 0) + 1
        if grounding_event:
            event = dict(grounding_event)
            events = [dict(row) for row in list(grounding.get("events", []) or []) if isinstance(row, dict)]
            events.append(event)
            grounding["events"] = events[-100:]
            if event.get("event_type") == "invalid_action_kwargs":
                invalid = [
                    dict(row)
                    for row in list(grounding.get("invalid_action_kwargs_events", []) or [])
                    if isinstance(row, dict)
                ]
                invalid.append(event)
                grounding["invalid_action_kwargs_events"] = invalid[-50:]
            elif event.get("event_type") == "local_machine_action_kwargs_repaired":
                repaired = [
                    dict(row)
                    for row in list(grounding.get("repaired_actions", []) or [])
                    if isinstance(row, dict)
                ]
                repaired.append(event)
                grounding["repaired_actions"] = repaired[-50:]
            elif event.get("event_type") == "side_effect_after_verified_completion":
                blocked = [
                    dict(row)
                    for row in list(grounding.get("side_effect_after_verified_completion_events", []) or [])
                    if isinstance(row, dict)
                ]
                blocked.append(event)
                grounding["side_effect_after_verified_completion_events"] = blocked[-50:]

        context = self._action_grounding_context(state_override={**state, "grounding": grounding})
        target_binding = bind_target(context)
        if target_binding:
            state["target_binding"] = target_binding
        target_file = extract_grounding_target_file(context)
        bound_target = str(target_binding.get("top_target_file") or "") if isinstance(target_binding, Mapping) else ""
        try:
            bound_confidence = float(target_binding.get("target_confidence", 0.0) or 0.0) if isinstance(target_binding, Mapping) else 0.0
        except (TypeError, ValueError):
            bound_confidence = 0.0
        if bound_target and (not target_file or bound_confidence >= 0.55):
            target_file = bound_target
        if target_file:
            grounding["target_file"] = target_file
        state = self._ensure_explicit_hypothesis_lifecycle(
            state,
            function_name=function_name,
            kwargs=kwargs,
            raw_result=raw_result,
            binding=dict(target_binding or {}),
            target_file=target_file,
        )

        phase_after = phase_before
        success = bool(raw_result.get("success", False))
        state_name = str(raw_result.get("state", "") or "")
        if phase_before == "complete-ready" and not (function_name == "run_test" and not success):
            phase_after = "complete-ready"
        elif function_name == "repo_tree" and success:
            phase_after = "inspect"
        elif function_name == "run_test" and not success:
            phase_after = "localize"
        elif function_name == "file_read" and success:
            read_path = str(raw_result.get("path") or kwargs.get("path") or "")
            if read_path.endswith(".py") and not (read_path.startswith("tests/") or "/tests/" in read_path):
                current_target = str(grounding.get("target_file") or "")
                if not current_target or read_path == current_target or phase_before in {"localize", "patch"}:
                    grounding["target_file"] = read_path
                target_file = str(grounding.get("target_file") or target_file)
            if read_path and read_path == str(grounding.get("target_file") or "") and target_file:
                phase_after = "patch"
        elif function_name == "apply_patch" and success:
            phase_after = "verify"
            grounding["verification_pending"] = True
            grounding["last_patch"] = {
                "touched_files": list(raw_result.get("touched_files", []) or []),
                "patch_sha256": str(raw_result.get("patch_sha256") or ""),
                "created_at": _now(),
            }
        elif function_name == "propose_patch" and success and bool(raw_result.get("patch_proposal_verified", False)):
            phase_after = "complete-ready"
            grounding["verification_pending"] = False
            grounding["last_patch"] = {
                "touched_files": list(raw_result.get("touched_files", []) or []),
                "patch_sha256": str(raw_result.get("patch_sha256") or ""),
                "created_at": _now(),
                "source": "patch_proposal",
            }
            if str(state.get("terminal_state") or "") != "completed_verified":
                state["terminal_state"] = "completed_verified"
                state["completion_reason"] = "bounded patch proposal passed targeted and full verifier tests"
                state["terminal_tick"] = action_count
                state["verified_completion"] = True
        elif function_name == "propose_patch" and bool(raw_result.get("needs_human_review", False)):
            phase_after = "complete-ready"
            grounding["verification_pending"] = False
            if str(state.get("terminal_state") or "") != "needs_human_review":
                state["terminal_state"] = "needs_human_review"
                state["completion_reason"] = str(raw_result.get("refusal_reason") or "evidence_insufficient")
                state["terminal_tick"] = action_count
                state["verified_completion"] = False
                state["needs_human_review"] = True
                state["refusal_reason"] = str(raw_result.get("refusal_reason") or "evidence_insufficient")
        elif function_name == "run_test" and success and str(kwargs.get("target") or ".") == ".":
            phase_after = "complete-ready"
            grounding["verification_pending"] = False
            if str(state.get("terminal_state") or "") != "completed_verified":
                state["terminal_state"] = "completed_verified"
                state["completion_reason"] = "full test suite passed after patch verification"
                state["terminal_tick"] = action_count
                state["verified_completion"] = True
        elif function_name == "run_test" and success:
            grounding["last_successful_test_target"] = str(kwargs.get("target") or "")
            if phase_before == "verify":
                phase_after = "verify"
        elif function_name == "mirror_plan" and success and phase_before == "complete-ready":
            phase_after = "complete-ready"
        elif state_name == "INVALID_ACTION_KWARGS":
            phase_after = phase_before

        stall_event = self._detect_investigation_stall({**state, "grounding": grounding})
        if stall_event:
            stalled_events = [
                dict(row)
                for row in list(state.get("stalled_events", []) or [])
                if isinstance(row, dict)
            ]
            last_key = (
                str(stalled_events[-1].get("recommended_action") or ""),
                str(stalled_events[-1].get("top_target_file") or ""),
                str(stalled_events[-1].get("recent_actions") or ""),
            ) if stalled_events else ("", "", "")
            next_key = (
                str(stall_event.get("recommended_action") or ""),
                str(stall_event.get("top_target_file") or ""),
                str(stall_event.get("recent_actions") or ""),
            )
            if next_key != last_key:
                stalled_events.append(stall_event)
                state["stalled_events"] = stalled_events[-50:]
                events = [dict(row) for row in list(grounding.get("events", []) or []) if isinstance(row, dict)]
                events.append(stall_event)
                grounding["events"] = events[-100:]
        state["grounding"] = grounding
        state["investigation_phase"] = phase_after
        self._save_investigation_state(state)
        return {
            "schema_version": LOCAL_MACHINE_INVESTIGATION_VERSION,
            "phase_before": phase_before,
            "phase_after": phase_after,
            "terminal_state": str(state.get("terminal_state") or ""),
            "completion_reason": str(state.get("completion_reason") or ""),
            "terminal_tick": state.get("terminal_tick"),
            "verified_completion": bool(state.get("verified_completion", False)),
            "target_file": str(grounding.get("target_file") or ""),
            "target_binding": dict(state.get("target_binding", {}) or {}),
            "hypothesis_lifecycle": dict(state.get("hypothesis_lifecycle", {}) or {}),
            "stalled_event": dict(stall_event or {}),
            "empty_kwargs_attempt_count": int(grounding.get("empty_kwargs_attempt_count", 0) or 0),
            "repaired_action_count": len(list(grounding.get("repaired_actions", []) or [])),
            "invalid_action_kwargs_count": len(list(grounding.get("invalid_action_kwargs_events", []) or [])),
        }

    def _action_governance_state(self) -> Any:
        state = self._load_investigation_state()
        return governance_state_from_local_machine_investigation(
            state,
            source_root=str(self.source_root),
            allowed_roots=[str(self.source_root), str(self._workspace_root())],
        )

    def _persist_action_governance(
        self,
        *,
        governance_state: Any | None = None,
        decision: ActionGovernanceDecision | None = None,
    ) -> None:
        state = self._load_investigation_state()
        governance = dict(state.get("action_governance", {}) or {})
        if governance_state is not None:
            governance["failure_count_by_agent"] = dict(governance_state.failure_count_by_agent)
            governance["downgraded_agents"] = dict(governance_state.downgraded_agents)
        if decision is not None:
            decision_payload = decision.to_dict()
            governance["last_decision"] = decision_payload
            events = [
                dict(row)
                for row in list(governance.get("events", []) or [])
                if isinstance(row, dict)
            ]
            events.append(dict(decision.audit_event))
            governance["events"] = events[-100:]
        state["action_governance"] = governance
        self._save_investigation_state(state)

    @staticmethod
    def _sync_plan_changed_paths(plan: Mapping[str, Any]) -> list[str]:
        paths: list[str] = []
        for row in list(plan.get("actionable_changes", []) or []):
            if not isinstance(row, Mapping):
                continue
            relative = str(
                row.get("relative_path")
                or row.get("path")
                or row.get("target_path")
                or ""
            ).strip()
            if relative and relative not in paths:
                paths.append(relative)
        return paths

    def _evaluate_action_governance(
        self,
        function_name: str,
        kwargs: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Any | None, ActionGovernanceDecision | None]:
        if not self.action_governance_enabled:
            return None, None
        request = derive_action_governance_request(
            function_name,
            kwargs,
            agent_id=self.task_id,
            metadata=metadata or {},
        )
        decision = evaluate_action_governance(
            request,
            self._action_governance_state(),
            self.action_governance_policy,
        )
        self._persist_action_governance(decision=decision)
        return request, decision

    def _finalize_action_governance(
        self,
        request: Any | None,
        decision: ActionGovernanceDecision | None,
        raw_result: Mapping[str, Any],
    ) -> None:
        if request is None or decision is None or decision.status != "ALLOWED":
            return
        updated_state = record_action_governance_result(
            self._action_governance_state(),
            request,
            success=bool(raw_result.get("success", False)),
            failure_reason=str(raw_result.get("failure_reason") or ""),
            policy=self.action_governance_policy,
        )
        self._persist_action_governance(governance_state=updated_state, decision=decision)

    def _validate_evidence_refs(self, refs: Sequence[str]) -> list[str]:
        normalized = _string_list(refs)
        if not normalized:
            return []
        state = self._load_investigation_state()
        known_notes = {
            str(row.get("note_id") or "").strip()
            for row in list(state.get("notes", []) or [])
            if isinstance(row, dict)
        }
        for ref in normalized:
            if ref.startswith("note_") and ref not in known_notes:
                raise MirrorScopeError(f"unknown evidence note reference: {ref}")
            if ref.startswith("file:"):
                raw_path = ref[len("file:") :].split(":", 1)[0].strip()
                if not raw_path:
                    raise MirrorScopeError(f"invalid file evidence reference: {ref}")
                source_path, _ = self._resolve_source_path(raw_path)
                if not source_path.exists():
                    raise MirrorScopeError(f"file evidence reference does not exist: {ref}")
        return normalized

    def _append_note(self, *, kind: str, content: str, evidence_refs: Sequence[str] = ()) -> Dict[str, Any]:
        state = self._load_investigation_state()
        notes = [dict(row) for row in list(state.get("notes", []) or []) if isinstance(row, dict)]
        refs = self._validate_evidence_refs(evidence_refs)
        note = {
            "note_id": f"note_{len(notes) + 1:04d}",
            "kind": str(kind or "finding"),
            "content": str(content or "").strip(),
            "evidence_refs": refs,
            "created_at": _now(),
        }
        if not note["content"]:
            raise MirrorScopeError("note_write requires non-empty content")
        if note["kind"].strip().lower() in {"finding", "claim", "decision", "proposal"} and not refs:
            raise MirrorScopeError("note_write finding requires evidence_refs")
        notes.append(note)
        state["notes"] = notes[-200:]
        self._save_investigation_state(state)
        return note

    def _persist_hypothesis_lifecycle(self, row: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> None:
        if self.evidence_db_path is None:
            return
        store = RuntimeStateStore(self.evidence_db_path)
        try:
            store.upsert_hypothesis_lifecycle(dict(row or {}))
            if event:
                refs = list(event.get("evidence_refs", []) or []) if isinstance(event.get("evidence_refs", []), list) else []
                store.record_hypothesis_lifecycle_event(
                    hypothesis_id=str(event.get("hypothesis_id") or row.get("hypothesis_id") or ""),
                    run_id=self.task_id,
                    event_type=str(event.get("event_type") or "hypothesis_event"),
                    evidence_ref=str(refs[0] if refs else ""),
                    delta=float(event.get("delta", 0.0) or 0.0),
                    payload=dict(event or {}),
                )
        finally:
            store.close()

    def _source_relative_path(self, raw_path: Any, *, allow_empty: bool = False) -> Path:
        raw = str(raw_path or "").strip()
        if not raw:
            if allow_empty:
                return Path(".")
            raise MirrorScopeError("path is required")
        path = Path(raw).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
            try:
                return resolved.relative_to(self.source_root)
            except ValueError as exc:
                raise MirrorScopeError(f"path is outside source root: {raw}") from exc
        parts = list(path.parts)
        if parts and parts[0] == self.source_root.name:
            parts = parts[1:]
        if not parts:
            return Path(".") if allow_empty else Path(self.source_root.name)
        if any(part in {"", ".", ".."} for part in parts):
            raise MirrorScopeError(f"path escapes source scope: {raw}")
        return Path(*parts)

    def _workspace_relative_path_arg(self, raw_path: Any, *, allow_empty: bool = False) -> Path:
        raw = str(raw_path or "").strip()
        if not raw:
            if allow_empty:
                return Path(".")
            raise MirrorScopeError("path is required")
        path = Path(raw).expanduser()
        workspace_root = self._workspace_root().resolve()
        if path.is_absolute():
            resolved = path.resolve()
            try:
                return resolved.relative_to(workspace_root)
            except ValueError as exc:
                raise MirrorScopeError(f"path is outside mirror workspace: {raw}") from exc
        parts = list(path.parts)
        if parts and parts[0] in {self.source_root.name, "workspace"}:
            parts = parts[1:]
        if not parts:
            return Path(".") if allow_empty else Path(self.source_root.name)
        if any(part in {"", ".", ".."} for part in parts):
            raise MirrorScopeError(f"path escapes workspace scope: {raw}")
        return Path(*parts)

    def _resolve_source_path(self, raw_path: Any, *, allow_empty: bool = False) -> tuple[Path, str]:
        relative = self._source_relative_path(raw_path, allow_empty=allow_empty)
        path = (self.source_root / relative).resolve()
        try:
            path.relative_to(self.source_root)
        except ValueError as exc:
            raise MirrorScopeError(f"path is outside source root: {raw_path}") from exc
        if path.is_symlink():
            raise MirrorScopeError(f"symlink paths are not supported: {raw_path}")
        return path, "." if relative == Path(".") else relative.as_posix()

    def _repair_missing_source_relative_path(self, relative: Path) -> tuple[Path, str]:
        candidate_path = (self.source_root / relative).resolve()
        if candidate_path.exists() or relative == Path("."):
            return relative, ""
        basename = str(relative.name or "").strip()
        if not basename:
            return relative, ""

        candidates: list[tuple[int, str]] = []
        state = self._load_investigation_state()
        last_tree = state.get("last_tree", {}) if isinstance(state.get("last_tree", {}), dict) else {}
        for row in list(last_tree.get("entries", []) or []):
            if not isinstance(row, Mapping) or str(row.get("kind", "") or "") != "file":
                continue
            rel = str(row.get("path", "") or "").strip()
            if not rel or Path(rel).name != basename:
                continue
            score = 1
            rel_parts = Path(rel).parts
            raw_parts = relative.parts
            for suffix_len in range(2, min(len(rel_parts), len(raw_parts)) + 1):
                if rel_parts[-suffix_len:] == raw_parts[-suffix_len:]:
                    score = max(score, suffix_len)
            if raw_parts and rel_parts and raw_parts[0] == rel_parts[0]:
                score += 1
            candidates.append((score, rel))

        if not candidates:
            for path in self.source_root.rglob(basename):
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    rel = path.resolve().relative_to(self.source_root).as_posix()
                except ValueError:
                    continue
                if self._path_excluded(rel, DEFAULT_REPO_EXCLUDES):
                    continue
                candidates.append((1, rel))

        if not candidates:
            return relative, ""
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_rel = candidates[0]
        ambiguous = len(candidates) > 1 and candidates[1][0] == best_score
        if ambiguous:
            return relative, ""
        return Path(best_rel), f"{relative.as_posix()} -> {best_rel}"

    def _resolve_workspace_path(self, raw_path: Any, *, allow_empty: bool = False) -> tuple[Path, str]:
        workspace_root = self._workspace_root().resolve()
        relative = self._workspace_relative_path_arg(raw_path, allow_empty=allow_empty)
        path = (workspace_root / relative).resolve()
        try:
            path.relative_to(workspace_root)
        except ValueError as exc:
            raise MirrorScopeError(f"path is outside mirror workspace: {raw_path}") from exc
        return path, "." if relative == Path(".") else relative.as_posix()

    def _ensure_workspace_file(self, raw_path: Any) -> tuple[Path, str]:
        workspace_path, relative = self._resolve_workspace_path(raw_path)
        if workspace_path.exists():
            if not workspace_path.is_file():
                raise MirrorScopeError(f"workspace path is not a regular file: {relative}")
            return workspace_path, relative
        source_path = (self.source_root / relative).resolve()
        if source_path.exists() and source_path.is_file():
            materialize_files(self.source_root, self.mirror_root, [relative])
            return self._resolve_workspace_path(relative)
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        return workspace_path, relative

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _path_excluded(relative: str, excludes: Iterable[str]) -> bool:
        parts = set(Path(relative).parts)
        return any(str(item) in parts for item in excludes if str(item))

    @staticmethod
    def _read_text_file(path: Path, *, max_bytes: int = 256 * 1024) -> str:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if not path.is_file():
            raise MirrorScopeError(f"path is not a regular file: {path}")
        if path.stat().st_size > max_bytes:
            raise MirrorScopeError(f"file is too large for this atomic read: {path}")
        return path.read_text(encoding="utf-8")

    def _act_repo_tree(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        depth = self._bounded_int(kwargs.get("depth"), default=2, minimum=0, maximum=6)
        max_entries = self._bounded_int(kwargs.get("max_entries"), default=200, minimum=1, maximum=1000)
        excludes = set(DEFAULT_REPO_EXCLUDES)
        excludes.update(_string_list(kwargs.get("exclude") or kwargs.get("excludes")))
        root_path, root_relative = self._resolve_source_path(kwargs.get("path") or kwargs.get("root") or ".", allow_empty=True)
        if not root_path.exists():
            raise FileNotFoundError(str(root_path))
        entries: list[Dict[str, Any]] = []

        def walk(directory: Path, current_depth: int) -> None:
            if len(entries) >= max_entries or current_depth >= depth:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError:
                return
            for child in children:
                if len(entries) >= max_entries:
                    return
                try:
                    relative = child.resolve().relative_to(self.source_root).as_posix()
                except ValueError:
                    continue
                if self._path_excluded(relative, excludes) or child.is_symlink():
                    continue
                row = {
                    "path": relative,
                    "name": child.name,
                    "kind": "dir" if child.is_dir() else "file",
                    "size_bytes": int(child.stat().st_size) if child.is_file() else 0,
                    "depth": current_depth + 1,
                }
                entries.append(row)
                if child.is_dir():
                    walk(child, current_depth + 1)

        if root_path.is_file():
            entries.append({
                "path": root_relative,
                "name": root_path.name,
                "kind": "file",
                "size_bytes": int(root_path.stat().st_size),
                "depth": 0,
            })
        else:
            walk(root_path, 0)
        state = self._load_investigation_state()
        stored_entry_limit = min(max_entries, 500)
        state["last_tree"] = {
            "root": root_relative,
            "depth": depth,
            "entry_count": len(entries),
            "entries": entries[:stored_entry_limit],
            "entries_stored": min(len(entries), stored_entry_limit),
        }
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="repo_tree",
            reward=0.15 if entries else 0.05,
            state="REPO_TREE_READ",
            root=root_relative,
            depth=depth,
            entry_count=len(entries),
            truncated=len(entries) >= max_entries,
            entries=entries,
        )

    def _act_repo_find(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        root_path, root_relative = self._resolve_source_path(kwargs.get("root") or kwargs.get("path") or ".", allow_empty=True)
        pattern = str(kwargs.get("name_pattern") or kwargs.get("pattern") or "*").strip() or "*"
        kind = str(kwargs.get("kind") or "any").strip().lower()
        max_results = self._bounded_int(kwargs.get("max_results"), default=50, minimum=1, maximum=500)
        excludes = set(DEFAULT_REPO_EXCLUDES)
        excludes.update(_string_list(kwargs.get("exclude") or kwargs.get("excludes")))
        results: list[Dict[str, Any]] = []
        iterator = root_path.rglob("*") if root_path.is_dir() else [root_path]
        for path in iterator:
            if len(results) >= max_results:
                break
            if path.is_symlink():
                continue
            rel = path.resolve().relative_to(self.source_root).as_posix()
            if self._path_excluded(rel, excludes) or not fnmatch.fnmatch(path.name, pattern):
                continue
            row_kind = "dir" if path.is_dir() else "file"
            if kind not in {"any", row_kind}:
                continue
            results.append({"path": rel, "name": path.name, "kind": row_kind})
        state = self._load_investigation_state()
        state["last_search"] = {"action": "repo_find", "root": root_relative, "pattern": pattern, "result_count": len(results), "results": results[:50]}
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="repo_find",
            reward=0.15 if results else 0.03,
            state="REPO_FIND_READ",
            root=root_relative,
            name_pattern=pattern,
            result_count=len(results),
            results=results,
        )

    def _act_repo_grep(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        root_path, root_relative = self._resolve_source_path(kwargs.get("root") or kwargs.get("path") or ".", allow_empty=True)
        query = str(kwargs.get("query") or kwargs.get("pattern") or "").strip()
        if not query:
            raise MirrorScopeError("repo_grep requires a query")
        globs = _string_list(kwargs.get("globs")) or ["*.py", "*.ts", "*.tsx", "*.js", "*.md", "*.toml", "*.json", "*.yaml", "*.yml"]
        max_matches = self._bounded_int(kwargs.get("max_matches"), default=50, minimum=1, maximum=500)
        case_sensitive = bool(kwargs.get("case_sensitive", False))
        use_regex = bool(kwargs.get("regex", False))
        matcher = re.compile(query, 0 if case_sensitive else re.IGNORECASE) if use_regex else None
        needle = query if case_sensitive else query.lower()
        matches: list[Dict[str, Any]] = []
        for path in sorted(root_path.rglob("*") if root_path.is_dir() else [root_path]):
            if len(matches) >= max_matches:
                break
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 512 * 1024:
                continue
            rel = path.resolve().relative_to(self.source_root).as_posix()
            if self._path_excluded(rel, DEFAULT_REPO_EXCLUDES):
                continue
            if not any(fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(path.name, glob) for glob in globs):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.lower()
                found = bool(matcher.search(line)) if matcher is not None else needle in haystack
                if not found:
                    continue
                matches.append({"path": rel, "line": line_number, "text": line[:500]})
                if len(matches) >= max_matches:
                    break
        state = self._load_investigation_state()
        state["last_search"] = {"action": "repo_grep", "root": root_relative, "query": query, "match_count": len(matches), "matches": matches[:50]}
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="repo_grep",
            reward=0.18 if matches else 0.02,
            state="REPO_GREP_READ",
            root=root_relative,
            query=query,
            match_count=len(matches),
            matches=matches,
        )

    def _read_path_for_investigation(self, raw_path: Any) -> tuple[Path, str, str]:
        source_path, relative = self._resolve_source_path(raw_path)
        source_relative = Path(relative)
        path_correction = ""
        if not source_path.exists():
            repaired_relative, path_correction = self._repair_missing_source_relative_path(source_relative)
            if repaired_relative != source_relative:
                source_path = (self.source_root / repaired_relative).resolve()
                relative = repaired_relative.as_posix()
        workspace_path = (self._workspace_root() / relative).resolve()
        if workspace_path.exists() and workspace_path.is_file():
            return workspace_path, relative, "workspace"
        if path_correction:
            state = self._load_investigation_state()
            state["last_path_correction"] = {
                "requested_path": str(raw_path or ""),
                "correction": path_correction,
                "resolved_path": relative,
                "created_at": _now(),
            }
            self._save_investigation_state(state)
        return source_path, relative, "source"

    def _act_file_read(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        path, relative, location = self._read_path_for_investigation(kwargs.get("path"))
        text = self._read_text_file(path)
        lines = text.splitlines()
        start_line = self._bounded_int(kwargs.get("start_line"), default=1, minimum=1, maximum=max(1, len(lines) + 1))
        end_default = min(len(lines), start_line + 199)
        end_line = self._bounded_int(kwargs.get("end_line"), default=end_default, minimum=start_line, maximum=max(start_line, len(lines)))
        selected = lines[start_line - 1:end_line]
        numbered = [{"line": start_line + idx, "text": line} for idx, line in enumerate(selected)]
        state = self._load_investigation_state()
        state["last_read"] = {"path": relative, "location": location, "start_line": start_line, "end_line": end_line}
        read_files = [dict(row) for row in list(state.get("read_files", []) or []) if isinstance(row, dict)]
        read_files.append({"path": relative, "location": location, "start_line": start_line, "end_line": end_line, "created_at": _now()})
        state["read_files"] = read_files[-100:]
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="file_read",
            reward=0.2,
            state="FILE_READ",
            path=relative,
            location=location,
            start_line=start_line,
            end_line=end_line,
            total_lines=len(lines),
            lines=numbered,
            content="\n".join(selected),
        )

    def _outline_python(self, path: Path, text: str) -> list[Dict[str, Any]]:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return [{"kind": "syntax_error", "name": str(exc), "line": int(exc.lineno or 0)}]
        rows: list[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rows.append({
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "line": int(getattr(node, "lineno", 0) or 0),
                    "end_line": int(getattr(node, "end_lineno", 0) or 0),
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                rows.append({
                    "kind": "import",
                    "name": path.name,
                    "line": int(getattr(node, "lineno", 0) or 0),
                })
        return sorted(rows, key=lambda row: (int(row.get("line", 0) or 0), str(row.get("kind", ""))))

    def _act_file_outline(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        path, relative, location = self._read_path_for_investigation(kwargs.get("path"))
        text = self._read_text_file(path)
        if path.suffix == ".py":
            outline = self._outline_python(path, text)
        else:
            outline = [
                {"kind": "heading", "name": line.strip("# "), "line": idx}
                for idx, line in enumerate(text.splitlines(), start=1)
                if line.lstrip().startswith("#")
            ][:80]
        return self._raw_success(
            function_name="file_outline",
            reward=0.16 if outline else 0.05,
            state="FILE_OUTLINE_READ",
            path=relative,
            location=location,
            outline=outline[:200],
            outline_count=len(outline),
        )

    def _act_file_summary(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        path, relative, location = self._read_path_for_investigation(kwargs.get("path"))
        text = self._read_text_file(path)
        lines = text.splitlines()
        outline = self._outline_python(path, text)[:30] if path.suffix == ".py" else []
        return self._raw_success(
            function_name="file_summary",
            reward=0.12,
            state="FILE_SUMMARY_READ",
            path=relative,
            location=location,
            suffix=path.suffix,
            size_bytes=int(path.stat().st_size),
            total_lines=len(lines),
            first_lines=lines[:40],
            outline=outline,
        )

    def _act_note_write(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        note = self._append_note(
            kind=str(kwargs.get("kind") or "finding"),
            content=str(kwargs.get("content") or ""),
            evidence_refs=_string_list(kwargs.get("evidence_refs")),
        )
        return self._raw_success(
            function_name="note_write",
            reward=0.08,
            state="INVESTIGATION_NOTE_WRITTEN",
            note=note,
        )

    def _act_hypothesis_add(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        claim = str(kwargs.get("claim") or kwargs.get("content") or "").strip()
        if not claim:
            raise MirrorScopeError("hypothesis_add requires a claim")
        evidence_refs = self._validate_evidence_refs(kwargs.get("evidence_refs", []))
        predictions = kwargs.get("predictions", {})
        falsifiers = kwargs.get("falsifiers", {})
        hypothesis = normalize_hypothesis(
            hypothesis_id=str(kwargs.get("hypothesis_id") or f"hyp_{len(hypotheses) + 1:04d}"),
            run_id=self.task_id,
            task_family="local_machine",
            family=str(kwargs.get("family") or kwargs.get("hypothesis_type") or "codebase"),
            claim=claim,
            confidence=float(kwargs.get("confidence", 0.5) or 0.5),
            evidence_refs=evidence_refs,
            competing_with=_string_list(kwargs.get("competing_with")),
            predictions=dict(predictions) if isinstance(predictions, Mapping) else {},
            falsifiers=dict(falsifiers) if isinstance(falsifiers, Mapping) else {},
            metadata={
                "source": "local_machine_hypothesis_add",
                "created_at_iso": _now(),
            },
        )
        event = {
            "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
            "hypothesis_id": hypothesis["hypothesis_id"],
            "run_id": self.task_id,
            "event_type": "hypothesis_created",
            "evidence_refs": list(hypothesis.get("evidence_refs", []) or []),
            "delta": 0.0,
            "created_at": hypothesis["created_at"],
        }
        hypotheses.append(hypothesis)
        state["hypotheses"] = hypotheses[-100:]
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        self._persist_hypothesis_lifecycle(hypothesis, event)
        return self._raw_success(
            function_name="hypothesis_add",
            reward=0.08,
            state="INVESTIGATION_HYPOTHESIS_ADDED",
            hypothesis=hypothesis,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_hypothesis_update(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        hypothesis_id = str(kwargs.get("hypothesis_id") or kwargs.get("id") or "").strip()
        if not hypothesis_id:
            raise MirrorScopeError("hypothesis_update requires hypothesis_id")
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        index = next(
            (idx for idx, row in enumerate(hypotheses) if str(row.get("hypothesis_id") or "") == hypothesis_id),
            -1,
        )
        if index < 0:
            raise MirrorScopeError(f"unknown hypothesis_id: {hypothesis_id}")
        refs = _string_list(kwargs.get("evidence_refs"))
        single_ref = str(kwargs.get("evidence_ref") or "").strip()
        if single_ref and single_ref not in refs:
            refs.append(single_ref)
        try:
            strength = float(kwargs.get("strength", 0.2) or 0.2)
        except (TypeError, ValueError):
            strength = 0.2
        updated, event = apply_hypothesis_evidence(
            hypotheses[index],
            signal=str(kwargs.get("signal") or kwargs.get("verdict") or kwargs.get("direction") or "neutral"),
            evidence_refs=refs,
            strength=strength,
            rationale=str(kwargs.get("rationale") or kwargs.get("why") or ""),
        )
        event["run_id"] = self.task_id
        hypotheses[index] = updated
        state["hypotheses"] = hypotheses[-100:]
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        self._persist_hypothesis_lifecycle(updated, event)
        return self._raw_success(
            function_name="hypothesis_update",
            reward=0.1 if str(event.get("signal") or "") in {"support", "contradiction"} else 0.04,
            state="HYPOTHESIS_UPDATED",
            hypothesis=updated,
            hypothesis_event=event,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_hypothesis_compete(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        left = str(
            kwargs.get("hypothesis_a")
            or kwargs.get("left")
            or kwargs.get("hypothesis_id_a")
            or kwargs.get("hypothesis_id")
            or ""
        ).strip()
        right = str(
            kwargs.get("hypothesis_b")
            or kwargs.get("right")
            or kwargs.get("hypothesis_id_b")
            or kwargs.get("competes_with")
            or ""
        ).strip()
        if not left or not right or left == right:
            raise MirrorScopeError("hypothesis_compete requires two distinct hypothesis ids")
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        known = {str(row.get("hypothesis_id") or "") for row in hypotheses}
        missing = [item for item in (left, right) if item not in known]
        if missing:
            raise MirrorScopeError(f"unknown competing hypothesis id(s): {', '.join(missing)}")
        hypotheses = mark_competing(hypotheses, left, right)
        event = {
            "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
            "run_id": self.task_id,
            "event_type": "hypothesis_competition_recorded",
            "hypotheses": [left, right],
            "reason": str(kwargs.get("reason") or kwargs.get("why") or ""),
            "delta": 0.0,
            "created_at": _now(),
        }
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["hypotheses"] = hypotheses[-100:]
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        for row in hypotheses:
            if str(row.get("hypothesis_id") or "") in {left, right}:
                scoped_event = dict(event)
                scoped_event["hypothesis_id"] = str(row.get("hypothesis_id") or "")
                self._persist_hypothesis_lifecycle(row, scoped_event)
        return self._raw_success(
            function_name="hypothesis_compete",
            reward=0.08,
            state="HYPOTHESIS_COMPETITION_RECORDED",
            competition={"hypothesis_a": left, "hypothesis_b": right},
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_discriminating_test_add(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        left = str(kwargs.get("hypothesis_a") or kwargs.get("hypothesis_id_a") or "").strip()
        right = str(kwargs.get("hypothesis_b") or kwargs.get("hypothesis_id_b") or "").strip()
        if not left or not right or left == right:
            raise MirrorScopeError("discriminating_test_add requires two distinct hypothesis ids")
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        by_id = {str(row.get("hypothesis_id") or ""): row for row in hypotheses}
        if left not in by_id or right not in by_id:
            missing = [item for item in (left, right) if item not in by_id]
            raise MirrorScopeError(f"unknown discriminating-test hypothesis id(s): {', '.join(missing)}")
        raw_action = kwargs.get("action")
        if isinstance(raw_action, Mapping):
            action = dict(raw_action)
        else:
            action_name = str(kwargs.get("action_name") or kwargs.get("target_action") or "").strip()
            if not action_name:
                raise MirrorScopeError("discriminating_test_add requires action or action_name")
            raw_args = kwargs.get("args", {})
            action = {
                "action": action_name,
                "args": dict(raw_args) if isinstance(raw_args, Mapping) else {},
            }
        expected_if_a = str(kwargs.get("expected_if_a") or "").strip()
        expected_if_b = str(kwargs.get("expected_if_b") or "").strip()
        why = str(kwargs.get("why") or kwargs.get("why_discriminating") or "").strip()
        if not expected_if_a or not expected_if_b or not why:
            raise MirrorScopeError("discriminating_test_add requires expected_if_a, expected_if_b, and why")
        test = build_discriminating_test(
            hypothesis_a=by_id[left],
            hypothesis_b=by_id[right],
            action=action,
            expected_if_a=expected_if_a,
            expected_if_b=expected_if_b,
            why=why,
            test_id=str(kwargs.get("test_id") or ""),
        )
        test["discriminates_between"] = [left, right]
        test["expected_outcomes_by_hypothesis"] = {
            left: expected_if_a,
            right: expected_if_b,
        }
        try:
            test["expected_information_gain"] = float(kwargs.get("expected_information_gain", 0.42) or 0.42)
        except (TypeError, ValueError):
            test["expected_information_gain"] = 0.42
        tests = [dict(row) for row in list(state.get("discriminating_tests", []) or []) if isinstance(row, dict)]
        tests.append(test)
        event = {
            "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
            "run_id": self.task_id,
            "event_type": "discriminating_test_proposed",
            "hypotheses": [left, right],
            "discriminates_between": [left, right],
            "test_id": test["test_id"],
            "delta": 0.0,
            "created_at": test["created_at"],
        }
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["discriminating_tests"] = tests[-100:]
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        for row in (by_id[left], by_id[right]):
            scoped_event = dict(event)
            scoped_event["hypothesis_id"] = str(row.get("hypothesis_id") or "")
            self._persist_hypothesis_lifecycle(row, scoped_event)
        return self._raw_success(
            function_name="discriminating_test_add",
            reward=0.09,
            state="DISCRIMINATING_TEST_ADDED",
            discriminating_test=test,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_candidate_files_set(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        files = []
        for raw in _string_list(kwargs.get("files") or kwargs.get("paths")):
            source_path, relative = self._resolve_source_path(raw)
            if not source_path.exists() or not source_path.is_file():
                raise FileNotFoundError(str(source_path))
            files.append(relative)
        state = self._load_investigation_state()
        state["candidate_files"] = files
        state["candidate_reason"] = str(kwargs.get("reason") or "").strip()
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="candidate_files_set",
            reward=0.1 if files else 0.02,
            state="CANDIDATE_FILES_SET",
            candidate_files=files,
            reason=state["candidate_reason"],
        )

    def _act_investigation_status(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        return self._raw_success(
            function_name="investigation_status",
            reward=0.03,
            state="INVESTIGATION_STATUS_READ",
            investigation=state,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _validate_patch_bounds(self, patch: str, *, max_files: int, max_hunks: int) -> None:
        files = set()
        hunks = 0
        for line in patch.splitlines():
            if line.startswith("+++ "):
                files.add(line[4:].strip().split("\t", 1)[0])
            elif line.startswith("@@"):
                hunks += 1
        if not patch.strip():
            raise MirrorScopeError("apply_patch requires a non-empty patch")
        if len(files) > max_files:
            raise MirrorScopeError(f"patch touches too many files: {len(files)} > {max_files}")
        if hunks > max_hunks:
            raise MirrorScopeError(f"patch has too many hunks: {hunks} > {max_hunks}")

    @staticmethod
    def _patch_header_path(raw: str) -> str:
        value = str(raw or "").strip().split("\t", 1)[0]
        if value in {"/dev/null", "dev/null"}:
            return ""
        for prefix in ("a/", "b/"):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value

    def _parse_unified_patch(self, patch: str) -> list[Dict[str, Any]]:
        lines = patch.splitlines(keepends=True)
        entries: list[Dict[str, Any]] = []
        index = 0
        while index < len(lines):
            if not lines[index].startswith("--- "):
                index += 1
                continue
            old_path = self._patch_header_path(lines[index][4:])
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise MirrorScopeError("unified patch is missing +++ header")
            new_path = self._patch_header_path(lines[index][4:])
            index += 1
            hunks: list[Dict[str, Any]] = []
            while index < len(lines) and not lines[index].startswith("--- "):
                header = lines[index]
                if not header.startswith("@@"):
                    index += 1
                    continue
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
                context_only_hunk = header.strip() == "@@"
                if not match and not context_only_hunk:
                    raise MirrorScopeError(f"unsupported patch hunk header: {header.strip()}")
                index += 1
                hunk_lines: list[str] = []
                while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("--- "):
                    hunk_lines.append(lines[index])
                    index += 1
                hunks.append({
                    "old_start": int(match.group(1)) if match else 1,
                    "old_count": int(match.group(2) or "1") if match else 1,
                    "new_start": int(match.group(3)) if match else 1,
                    "new_count": int(match.group(4) or "1") if match else 1,
                    "lines": hunk_lines,
                })
            path = new_path or old_path
            if not path:
                raise MirrorScopeError("patch file path is empty")
            if not hunks:
                raise MirrorScopeError(f"unified patch has no hunks for {path}")
            entries.append({"path": path, "old_path": old_path, "new_path": new_path, "hunks": hunks})
        if not entries:
            raise MirrorScopeError("no file patches found")
        return entries

    def _rollback_workspace_files(self, backups: Mapping[str, str]) -> None:
        workspace_root = self._workspace_root()
        for relative, content in backups.items():
            path = (workspace_root / relative).resolve()
            try:
                path.relative_to(workspace_root)
            except ValueError:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")

    def _apply_patch_entry(self, entry: Mapping[str, Any]) -> str:
        workspace_path, relative = self._ensure_workspace_file(entry.get("path"))
        original_lines = workspace_path.read_text(encoding="utf-8").splitlines(keepends=True) if workspace_path.exists() else []
        output: list[str] = []
        cursor = 0
        for hunk in list(entry.get("hunks", []) or []):
            old_start = int(hunk.get("old_start", 1) or 1)
            target = max(0, old_start - 1)
            old_sequence = [
                raw_line[1:].rstrip("\n")
                for raw_line in list(hunk.get("lines", []) or [])
                if raw_line and raw_line[:1] in {" ", "-"}
            ]
            if old_sequence:
                exact_end = target + len(old_sequence)
                current_sequence = [line.rstrip("\n") for line in original_lines[target:exact_end]]
                if current_sequence != old_sequence:
                    fuzzy_target = -1
                    for candidate in range(cursor, len(original_lines) - len(old_sequence) + 1):
                        candidate_sequence = [
                            line.rstrip("\n")
                            for line in original_lines[candidate : candidate + len(old_sequence)]
                        ]
                        if candidate_sequence == old_sequence:
                            fuzzy_target = candidate
                            break
                    if fuzzy_target >= 0:
                        target = fuzzy_target
            if target < cursor:
                raise MirrorScopeError(f"overlapping patch hunks for {relative}")
            output.extend(original_lines[cursor:target])
            cursor = target
            for raw_line in list(hunk.get("lines", []) or []):
                if raw_line.startswith("\\ No newline"):
                    continue
                marker = raw_line[:1]
                content = raw_line[1:]
                if marker == " ":
                    if cursor >= len(original_lines) or original_lines[cursor].rstrip("\n") != content.rstrip("\n"):
                        raise MirrorScopeError(f"patch context mismatch for {relative}")
                    output.append(original_lines[cursor])
                    cursor += 1
                elif marker == "-":
                    if cursor >= len(original_lines) or original_lines[cursor].rstrip("\n") != content.rstrip("\n"):
                        raise MirrorScopeError(f"patch deletion mismatch for {relative}")
                    cursor += 1
                elif marker == "+":
                    output.append(content)
                else:
                    raise MirrorScopeError(f"unsupported patch line marker for {relative}: {marker}")
        output.extend(original_lines[cursor:])
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(output), encoding="utf-8")
        return relative

    def _proposal_expected_tests(self, proposal: Mapping[str, Any], kwargs: Mapping[str, Any]) -> list[str]:
        tests = _string_list(kwargs.get("expected_tests")) or _string_list(proposal.get("expected_tests"))
        tests = [self._normalize_expected_test_target(test) for test in tests]
        tests = [test for test in tests if test and test != "."]
        failed_target = ""
        state = self._load_investigation_state()
        for row in reversed(list(state.get("validation_runs", []) or [])):
            item = dict(row) if isinstance(row, dict) else {}
            if bool(item.get("success", False)):
                continue
            payload = self._load_run_output_for_state(str(item.get("run_ref") or ""))
            command = [str(part) for part in list(payload.get("command", []) or [])]
            if command:
                failed_target = str(command[-1] or "")
                break
        if failed_target and failed_target != ".":
            tests.insert(0, failed_target)
        tests.append(".")
        return list(dict.fromkeys(tests))

    @staticmethod
    def _normalize_expected_test_target(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        if not tokens:
            return ""
        if len(tokens) >= 3 and tokens[0] in {"python", "python3", sys.executable} and tokens[1:3] == ["-m", "pytest"]:
            tokens = tokens[3:]
        elif Path(tokens[0]).name == "pytest":
            tokens = tokens[1:]
        candidates: list[str] = []
        for token in tokens:
            if not token or token.startswith("-"):
                continue
            if token in {"pytest", "python", "python3"}:
                continue
            if token.startswith(("tests/", "test_")) or "/test_" in token or token.endswith(".py"):
                candidates.append(token)
        target = candidates[0] if candidates else ("." if any(token.startswith("-") for token in tokens) else tokens[0])
        if "::" in target:
            target = target.split("::", 1)[0]
        if target in {"tests", "tests/"}:
            return "."
        return target

    def _load_run_output_for_state(self, run_ref: str) -> Dict[str, Any]:
        if not re.fullmatch(r"run_[A-Za-z0-9_.-]+", str(run_ref or "")):
            return {}
        return _load_json(self._run_output_root() / f"{run_ref}.json")

    def _record_patch_proposal(self, event: Mapping[str, Any]) -> None:
        state = self._load_investigation_state()
        proposals = [
            dict(row)
            for row in list(state.get("patch_proposals", []) or [])
            if isinstance(row, dict)
        ]
        proposals.append(dict(event))
        state["patch_proposals"] = proposals[-100:]
        self._save_investigation_state(state)

    def _act_propose_patch(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_investigation_state()
        context = self._action_grounding_context(state_override=state)
        binding = bind_target(context)
        state["target_binding"] = dict(binding)
        state = self._ensure_explicit_hypothesis_lifecycle(
            state,
            function_name="propose_patch",
            kwargs=kwargs,
            raw_result={"success": False, "state": "PRE_PATCH_PROPOSAL"},
            binding=dict(binding),
            target_file=str(kwargs.get("target_file") or binding.get("top_target_file") or ""),
        )
        leading = self._leading_hypothesis_from_state(state)
        if not leading:
            event = {
                "event_type": "patch_proposal_blocked",
                "reason": "missing_leading_hypothesis",
                "target_binding": dict(binding),
                "created_at": _now(),
            }
            self._record_patch_proposal(event)
            return self._raw_success(
                function_name="propose_patch",
                reward=0.0,
                state="PATCH_PROPOSAL_NEEDS_HYPOTHESIS",
                success=False,
                patch_proposals_generated=0,
                needs_human_review=True,
                refusal_reason="missing_leading_hypothesis",
                target_binding=dict(binding),
                source_hypothesis_id="",
                leading_hypothesis_before_patch={},
                hypothesis_target_file="",
                rejected_patch_proposals=[event],
            )
        leading_id = str(leading.get("hypothesis_id") or "")
        leading_target = str(leading.get("target_file") or "")
        target_file = str(kwargs.get("target_file") or binding.get("top_target_file") or "").strip()
        if not target_file and leading_target:
            target_file = leading_target
        leading_payload = {
            "source_hypothesis_id": leading_id,
            "leading_hypothesis_before_patch": dict(leading),
            "hypothesis_target_file": leading_target,
        }
        proposal_payload = generate_patch_proposals(
            {
                **context,
                "llm_thinking_mode": self.llm_thinking_mode,
                "investigation_state": {**state, "target_binding": binding},
            },
            top_target_file=target_file,
            max_changed_lines=self._bounded_int(kwargs.get("max_changed_lines"), default=20, minimum=1, maximum=20),
            llm_client=self.llm_client if self.prefer_llm_patch_proposals else None,
            prefer_llm=False,
            allow_fallback_patch=bool(kwargs.get("allow_fallback_patch", False)),
        )
        proposals = [
            dict(row)
            for row in list(proposal_payload.get("patch_proposals", []) or [])
            if isinstance(row, Mapping)
        ]
        fallback_patch = str(kwargs.get("fallback_patch") or "").strip()
        if fallback_patch and bool(kwargs.get("allow_fallback_patch", False)):
            fallback_changed_lines = sum(
                1
                for line in fallback_patch.splitlines()
                if line
                and not line.startswith(("+++", "---", "@@"))
                and line[0] in {"+", "-"}
            )
            if fallback_changed_lines <= self._bounded_int(kwargs.get("max_changed_lines"), default=20, minimum=1, maximum=20):
                proposals.append(
                    {
                        "schema_version": "conos.local_machine.patch_proposal/v1",
                        "proposal_id": f"grounded_{hashlib.sha256(fallback_patch.encode('utf-8')).hexdigest()[:12]}",
                        "target_file": target_file,
                        "unified_diff": fallback_patch,
                        "rationale": "grounded bounded patch inferred from the current hypothesis and failure evidence; verifier gate decides acceptance",
                        "evidence_refs": _string_list(kwargs.get("fallback_evidence_refs")),
                        "expected_tests": _string_list(kwargs.get("expected_tests")),
                        "risk": 0.28,
                        "proposal_source": "grounded_patch_fallback",
                        "patch_sha256": hashlib.sha256(fallback_patch.encode("utf-8")).hexdigest(),
                    }
                )
        generated_event = {
            "event_type": "patch_proposals_generated",
            "target_binding": dict(binding),
            "proposal_payload": dict(proposal_payload),
            "patch_proposal_llm_trace": list(proposal_payload.get("llm_trace", []) or []),
            **leading_payload,
            "created_at": _now(),
        }
        self._record_patch_proposal(generated_event)
        if not proposals:
            refusal_reason = str(proposal_payload.get("refusal_reason") or proposal_payload.get("rejection_reason") or "evidence_insufficient")
            state_name = "PATCH_PROPOSAL_TIMEOUT" if refusal_reason == "timeout" else "PATCH_PROPOSAL_NOT_GENERATED"
            return self._raw_success(
                function_name="propose_patch",
                reward=0.0,
                state=state_name,
                success=False,
                patch_proposals_generated=0,
                target_binding=dict(binding),
                patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
                **leading_payload,
                needs_human_review=refusal_reason != "timeout",
                refusal_reason=refusal_reason,
                llm_timeout=bool(proposal_payload.get("llm_timeout", False)),
                suggested_replan_reason="llm_patch_proposal_timeout" if refusal_reason == "timeout" else "patch_proposal_not_generated",
                rejected_patch_proposals=[
                    {
                        "reason": refusal_reason,
                        "target_file": target_file,
                        "needs_human_review": refusal_reason != "timeout",
                        "llm_timeout": bool(proposal_payload.get("llm_timeout", False)),
                    }
                ],
            )
        selected = {**proposals[0], **leading_payload}
        patch = str(selected.get("unified_diff") or "")
        try:
            self._validate_patch_bounds(patch, max_files=1, max_hunks=5)
            entries = self._parse_unified_patch(patch)
            if len(entries) != 1:
                raise MirrorScopeError("patch proposal must touch exactly one source file")
            target = str(entries[0].get("path") or "")
            if target.startswith("tests/") or "/tests/" in target:
                raise MirrorScopeError("patch proposal may not modify tests")
        except MirrorScopeError as exc:
            rejected = {
                "event_type": "patch_proposal_rejected",
                "proposal": dict(selected),
                "reason": str(exc),
                "rollback_count": 0,
                "created_at": _now(),
            }
            self._record_patch_proposal(rejected)
            return self._raw_success(
                function_name="propose_patch",
                reward=0.0,
                state="PATCH_PROPOSAL_REJECTED",
                success=False,
                patch_proposals_generated=len(proposals),
                patch_proposal_selected=dict(selected),
                patch_proposal_source=str(selected.get("proposal_source") or ""),
                patch_proposal_rationale=str(selected.get("rationale") or ""),
                patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
                **leading_payload,
                patch_proposal_applied=False,
                patch_proposal_verified=False,
                patch_proposal_rollback_count=0,
                rejected_patch_proposals=[rejected],
                proposal_test_results=[],
                target_binding=dict(binding),
            )
        backups: dict[str, str] = {}
        touched: list[str] = []
        rollback_count = 0
        try:
            for entry in entries:
                workspace_path, relative = self._ensure_workspace_file(entry.get("path"))
                backups[relative] = workspace_path.read_text(encoding="utf-8") if workspace_path.exists() else ""
            touched = [self._apply_patch_entry(entry) for entry in entries]
            test_results: list[Dict[str, Any]] = []
            verified = True
            for test_target in self._proposal_expected_tests(selected, kwargs):
                result = self._act_run_test({"target": test_target, "timeout_seconds": 30})
                test_results.append(
                    {
                        "target": test_target,
                        "success": bool(result.get("success", False)),
                        "run_ref": str(result.get("run_ref") or ""),
                        "returncode": int(result.get("returncode", 0) or 0),
                    }
                )
                if not bool(result.get("success", False)):
                    verified = False
                    break
            if not verified:
                self._rollback_workspace_files(backups)
                rollback_count = 1
                rejected = {
                    "event_type": "patch_proposal_rejected",
                    "proposal": dict(selected),
                    "test_results": test_results,
                    "rollback_count": rollback_count,
                    "created_at": _now(),
                }
                self._record_patch_proposal(rejected)
                return self._raw_success(
                    function_name="propose_patch",
                    reward=0.0,
                    state="PATCH_PROPOSAL_REJECTED",
                    success=False,
                    patch_proposals_generated=len(proposals),
                    patch_proposal_selected=dict(selected),
                    patch_proposal_source=str(selected.get("proposal_source") or ""),
                    patch_proposal_rationale=str(selected.get("rationale") or ""),
                    patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
                    **leading_payload,
                    patch_proposal_applied=True,
                    patch_proposal_verified=False,
                    patch_proposal_rollback_count=rollback_count,
                    rejected_patch_proposals=[rejected],
                    proposal_test_results=test_results,
                    target_binding=dict(binding),
                )
        except Exception:
            if backups:
                self._rollback_workspace_files(backups)
            raise
        accepted = {
            "event_type": "patch_proposal_accepted",
            "proposal": dict(selected),
            "touched_files": touched,
            "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            **leading_payload,
            "created_at": _now(),
        }
        self._record_patch_proposal(accepted)
        return self._raw_success(
            function_name="propose_patch",
            reward=0.55,
            state="PATCH_PROPOSAL_VERIFIED",
            success=True,
            touched_files=touched,
            patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            patch_proposals_generated=len(proposals),
            patch_proposal_selected=dict(selected),
            patch_proposal_source=str(selected.get("proposal_source") or ""),
            patch_proposal_rationale=str(selected.get("rationale") or ""),
            patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
            **leading_payload,
            patch_proposal_applied=True,
            patch_proposal_verified=True,
            patch_proposal_rollback_count=0,
            rejected_patch_proposals=[],
            proposal_test_results=test_results,
            target_binding=dict(binding),
            final_tests_passed=True,
        )

    def _act_apply_patch(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        patch = str(kwargs.get("patch") or "")
        max_files = self._bounded_int(kwargs.get("max_files"), default=1, minimum=1, maximum=5)
        max_hunks = self._bounded_int(kwargs.get("max_hunks"), default=3, minimum=1, maximum=20)
        self._validate_patch_bounds(patch, max_files=max_files, max_hunks=max_hunks)
        entries = self._parse_unified_patch(patch)
        if len(entries) > max_files:
            raise MirrorScopeError(f"patch touches too many files: {len(entries)} > {max_files}")
        state = self._load_investigation_state()
        leading = self._leading_hypothesis_from_state(state)
        if not leading:
            return self._raw_success(
                function_name="apply_patch",
                reward=0.0,
                state="PATCH_NEEDS_LEADING_HYPOTHESIS",
                success=False,
                needs_human_review=True,
                refusal_reason="missing_leading_hypothesis",
                source_hypothesis_id="",
                leading_hypothesis_before_patch={},
                hypothesis_target_file="",
            )
        leading_payload = {
            "source_hypothesis_id": str(leading.get("hypothesis_id") or ""),
            "leading_hypothesis_before_patch": dict(leading),
            "hypothesis_target_file": str(leading.get("target_file") or ""),
        }
        touched = [self._apply_patch_entry(entry) for entry in entries]
        self._append_note(kind="edit", content=f"Applied bounded patch to {', '.join(touched)}", evidence_refs=[f"patch:{_json_hash(patch)[:12]}"])
        return self._raw_success(
            function_name="apply_patch",
            reward=0.28,
            state="PATCH_APPLIED_TO_MIRROR",
            touched_files=touched,
            patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            **leading_payload,
        )

    def _act_edit_replace_range(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._ensure_workspace_file(kwargs.get("path"))
        existing = workspace_path.read_text(encoding="utf-8").splitlines(keepends=True) if workspace_path.exists() else []
        start_line = self._bounded_int(kwargs.get("start_line"), default=1, minimum=1, maximum=max(1, len(existing) + 1))
        end_line = self._bounded_int(kwargs.get("end_line"), default=start_line, minimum=start_line, maximum=max(start_line, len(existing)))
        replacement = str(kwargs.get("replacement") or "")
        replacement_lines = replacement.splitlines(keepends=True)
        if replacement and not replacement.endswith(("\n", "\r")):
            replacement_lines[-1:] = [replacement_lines[-1] + "\n"] if replacement_lines else [replacement + "\n"]
        updated = existing[: start_line - 1] + replacement_lines + existing[end_line:]
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(updated), encoding="utf-8")
        return self._raw_success(
            function_name="edit_replace_range",
            reward=0.24,
            state="RANGE_REPLACED_IN_MIRROR",
            path=relative,
            start_line=start_line,
            end_line=end_line,
        )

    def _act_edit_insert_after(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._ensure_workspace_file(kwargs.get("path"))
        existing = workspace_path.read_text(encoding="utf-8").splitlines(keepends=True) if workspace_path.exists() else []
        line = self._bounded_int(kwargs.get("line"), default=len(existing), minimum=0, maximum=len(existing))
        insertion = str(kwargs.get("text") or kwargs.get("content") or "")
        insertion_lines = insertion.splitlines(keepends=True)
        if insertion and not insertion.endswith(("\n", "\r")):
            insertion_lines[-1:] = [insertion_lines[-1] + "\n"] if insertion_lines else [insertion + "\n"]
        updated = existing[:line] + insertion_lines + existing[line:]
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(updated), encoding="utf-8")
        return self._raw_success(
            function_name="edit_insert_after",
            reward=0.22,
            state="TEXT_INSERTED_IN_MIRROR",
            path=relative,
            line=line,
        )

    def _act_create_file(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._resolve_workspace_path(kwargs.get("path"))
        if workspace_path.exists() and not bool(kwargs.get("overwrite", False)):
            raise MirrorScopeError(f"workspace file already exists: {relative}")
        content = str(kwargs.get("content") or "")
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text(content, encoding="utf-8")
        return self._raw_success(
            function_name="create_file",
            reward=0.24,
            state="FILE_CREATED_IN_MIRROR",
            path=relative,
            size_bytes=int(workspace_path.stat().st_size),
        )

    def _act_delete_file(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._resolve_workspace_path(kwargs.get("path"))
        if not workspace_path.exists():
            raise FileNotFoundError(str(workspace_path))
        if not workspace_path.is_file():
            raise MirrorScopeError(f"workspace path is not a regular file: {relative}")
        workspace_path.unlink()
        return self._raw_success(
            function_name="delete_file",
            reward=0.15,
            state="FILE_DELETED_FROM_MIRROR",
            path=relative,
        )

    def _write_run_output(self, *, function_name: str, command: Sequence[str], completed: subprocess.CompletedProcess[str], timeout_seconds: int) -> Dict[str, Any]:
        payload = {
            "schema_version": LOCAL_MACHINE_RUN_OUTPUT_VERSION,
            "run_ref": "",
            "function_name": function_name,
            "command": [str(part) for part in command],
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "timeout_seconds": int(timeout_seconds),
            "created_at": _now(),
        }
        payload["run_ref"] = f"run_{_json_hash(payload)[:16]}"
        path = self._run_output_root() / f"{payload['run_ref']}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        state = self._load_investigation_state()
        state["last_run_ref"] = payload["run_ref"]
        runs = [
            dict(row)
            for row in list(state.get("validation_runs", []) or [])
            if isinstance(row, dict)
        ]
        runs.append(
            {
                "run_ref": payload["run_ref"],
                "function_name": function_name,
                "returncode": int(completed.returncode),
                "success": int(completed.returncode) == 0,
                "created_at": payload["created_at"],
                "timeout_seconds": int(timeout_seconds),
            }
        )
        state["validation_runs"] = runs[-100:]
        self._save_investigation_state(state)
        return payload

    def _run_workspace_command(self, *, function_name: str, command: Sequence[str], timeout_seconds: int) -> Dict[str, Any]:
        mirror_result = run_mirror_command(
            self.source_root,
            self.mirror_root,
            [str(part) for part in command],
            allowed_commands=[*self.allowed_commands, sys.executable, "python", "python3"],
            timeout_seconds=timeout_seconds,
            backend=self.execution_backend,
            docker_image=self.docker_image,
            vm_provider=self.vm_provider,
            vm_name=self.vm_name,
            vm_host=self.vm_host,
            vm_workdir=self.vm_workdir,
            vm_network_mode=self.vm_network_mode,
            vm_sync_mode=self.vm_sync_mode,
            extra_env=self.extra_env,
        )
        completed = subprocess.CompletedProcess(
            args=[str(part) for part in command],
            returncode=int(mirror_result.returncode),
            stdout=str(mirror_result.stdout or ""),
            stderr=str(mirror_result.stderr or ""),
        )
        output_text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        if function_name in {"run_lint", "run_typecheck", "run_build"} and "Can't list" in output_text:
            completed = subprocess.CompletedProcess(
                args=[str(part) for part in command],
                returncode=2,
                stdout=str(completed.stdout or ""),
                stderr=str(completed.stderr or "") + "\nvalidation target is missing or not materialized",
            )
        payload = self._write_run_output(
            function_name=function_name,
            command=command,
            completed=completed,
            timeout_seconds=timeout_seconds,
        )
        return self._raw_success(
            function_name=function_name,
            reward=0.3 if completed.returncode == 0 else 0.0,
            state="VALIDATION_RUN_COMPLETED",
            success=completed.returncode == 0,
            run_ref=payload["run_ref"],
            returncode=int(completed.returncode),
            stdout_tail=str(completed.stdout or "")[-4000:],
            stderr_tail=str(completed.stderr or "")[-4000:],
            mirror_command=mirror_result.to_dict(),
            execution_boundary=dict(mirror_result.execution_boundary),
            sandbox_label=str(mirror_result.security_boundary or "best_effort_local_mirror"),
            not_os_security_sandbox=not bool(mirror_result.real_vm_boundary),
        )

    @staticmethod
    def _validation_path_part(target: str) -> str:
        return str(target or "").split("::", 1)[0].strip()

    def _ensure_validation_target_available(self, target: str) -> str:
        path_part = self._validation_path_part(target)
        if not path_part or path_part == ".":
            return "."
        workspace_path, relative = self._resolve_workspace_path(path_part)
        if workspace_path.exists():
            return relative
        source_path = (self.source_root / relative).resolve()
        try:
            source_path.relative_to(self.source_root)
        except ValueError as exc:
            raise MirrorScopeError(f"validation target is outside source root: {path_part}") from exc
        if source_path.exists() and source_path.is_file():
            materialize_files(self.source_root, self.mirror_root, [relative])
            return relative
        if source_path.exists() and source_path.is_dir():
            paths: list[str] = []
            for child in sorted(source_path.rglob("*")):
                if len(paths) >= 250:
                    break
                if not child.is_file() or child.is_symlink():
                    continue
                child_relative = child.resolve().relative_to(self.source_root).as_posix()
                if self._path_excluded(child_relative, DEFAULT_REPO_EXCLUDES):
                    continue
                paths.append(child_relative)
            if paths:
                materialize_files(self.source_root, self.mirror_root, paths)
                return relative
        raise MirrorScopeError(f"validation target does not exist in mirror or source: {path_part}")

    def _act_run_test(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        target = str(kwargs.get("target") or ".").strip() or "."
        timeout_seconds = self._bounded_int(kwargs.get("timeout_seconds"), default=30, minimum=1, maximum=600)
        materialized_for_full_test: list[str] = []
        if target != ".":
            self._ensure_validation_target_available(target)
            target_path = self._validation_path_part(target)
            if target_path == "tests" or target_path.startswith("tests/") or Path(target_path).name.startswith("test_"):
                context = self._action_grounding_context()
                materialized_for_full_test = self._materialize_missing_pytest_context(pytest_context_paths_from_tree(context, limit=200))
        else:
            context = self._action_grounding_context()
            materialized_for_full_test = self._materialize_missing_pytest_context(pytest_context_paths_from_tree(context, limit=200))
        raw = self._run_workspace_command(
            function_name="run_test",
            command=[sys.executable, "-m", "pytest", target],
            timeout_seconds=timeout_seconds,
        )
        if materialized_for_full_test:
            raw["materialized_for_full_test"] = materialized_for_full_test
        return raw

    def _materialize_missing_pytest_context(self, paths: Sequence[str]) -> list[str]:
        missing: list[str] = []
        workspace_root = self._workspace_root()
        for path in [str(item) for item in list(paths or []) if str(item)]:
            workspace_path = (workspace_root / path).resolve()
            try:
                workspace_path.relative_to(workspace_root)
            except ValueError:
                continue
            if not workspace_path.exists():
                missing.append(path)
        if missing:
            materialize_files(self.source_root, self.mirror_root, missing)
        return missing

    def _act_run_lint(self, kwargs: Mapping[str, Any], *, function_name: str) -> Dict[str, Any]:
        target = str(kwargs.get("target") or ".").strip() or "."
        timeout_seconds = self._bounded_int(kwargs.get("timeout_seconds"), default=30, minimum=1, maximum=600)
        materialized_for_validation: list[str] = []
        if target == ".":
            context = self._action_grounding_context()
            materialized_for_validation = self._materialize_missing_pytest_context(pytest_context_paths_from_tree(context, limit=200))
            if not materialized_for_validation and int(open_mirror(self.source_root, self.mirror_root).to_manifest().get("workspace_file_count", 0) or 0) <= 0:
                raise MirrorScopeError("validation target is empty; inspect or materialize files before running validation")
            command = [sys.executable, "-m", "compileall", "-q", "."]
        else:
            relative = self._ensure_validation_target_available(target)
            workspace_path, relative = self._resolve_workspace_path(relative)
            command = (
                [sys.executable, "-m", "py_compile", relative]
                if workspace_path.is_file() and workspace_path.suffix == ".py"
                else [sys.executable, "-m", "compileall", "-q", relative]
            )
        raw = self._run_workspace_command(
            function_name=function_name,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        if materialized_for_validation:
            raw["materialized_for_validation"] = materialized_for_validation
        return raw

    def _act_read_run_output(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        run_ref = str(kwargs.get("run_ref") or "").strip()
        if not run_ref:
            run_ref = str(self._load_investigation_state().get("last_run_ref", "") or "")
        if not re.fullmatch(r"run_[A-Za-z0-9_.-]+", run_ref):
            raise MirrorScopeError("read_run_output requires a valid run_ref")
        path = self._run_output_root() / f"{run_ref}.json"
        payload = _load_json(path)
        if not payload:
            raise FileNotFoundError(str(path))
        max_chars = self._bounded_int(kwargs.get("max_chars"), default=6000, minimum=100, maximum=50000)
        stdout = str(payload.get("stdout", "") or "")
        stderr = str(payload.get("stderr", "") or "")
        return self._raw_success(
            function_name="read_run_output",
            reward=0.05,
            state="RUN_OUTPUT_READ",
            run_ref=run_ref,
            returncode=int(payload.get("returncode", 0) or 0),
            stdout=stdout[-max_chars:],
            stderr=stderr[-max_chars:],
            command=list(payload.get("command", []) or []),
        )

    def _validate_mirror_exec_request(self, command: Sequence[str], kwargs: Mapping[str, Any], *, generated_command: bool) -> None:
        if not generated_command:
            return
        command_text = " ".join(str(part) for part in command)
        if len(command_text) > 1000:
            raise MirrorScopeError("Use atomic local-machine actions instead of mirror_exec: command is too long")
        if "<<" in command_text:
            raise MirrorScopeError("Use atomic local-machine actions instead of mirror_exec: heredoc is not allowed")
        if re.search(r"\bbase64\b|zlib|gzip|bz2|lzma", command_text, re.IGNORECASE):
            raise MirrorScopeError("Use atomic local-machine actions instead of mirror_exec: encoded or compressed scripts are not allowed")
        purpose = str(kwargs.get("purpose") or "").strip().lower()
        if purpose not in {"inspect", "test", "format", "build"}:
            raise MirrorScopeError("mirror_exec fallback requires purpose: inspect, test, format, or build")
        if "timeout_seconds" not in kwargs:
            raise MirrorScopeError("mirror_exec fallback requires timeout_seconds")
        target = str(kwargs.get("target") or kwargs.get("path") or kwargs.get("root") or "").strip()
        if not target:
            raise MirrorScopeError("mirror_exec fallback requires a bounded target path")
        self._workspace_relative_path_arg(target, allow_empty=True)
        for idx, part in enumerate(command[:-1]):
            if Path(str(part)).name in {"python", "python3"} and command[idx + 1] == "-c":
                script = str(command[idx + 2] if idx + 2 < len(command) else "")
                if len(script) > 300:
                    raise MirrorScopeError("Use atomic local-machine actions instead of mirror_exec: python -c script is too long")

    def _write_raw_diff_artifact(self, diff_entries: Sequence[Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
        rows = [entry.to_dict() if hasattr(entry, "to_dict") else dict(entry) for entry in diff_entries]
        payload = {
            "schema_version": LOCAL_MACHINE_RAW_DIFF_ARTIFACT_VERSION,
            "created_at": _now(),
            "source_root": str(self.source_root),
            "mirror_root": str(self.mirror_root),
            "workspace_root": str(manifest.get("workspace_root", "") or ""),
            "entry_count": len(rows),
            "diff_entries": rows,
        }
        artifact_id = _json_hash(payload)[:24]
        root = Path(str(manifest.get("control_root", "") or self.mirror_root / "control")) / "object_store" / "raw_diff"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{artifact_id}.json"
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return {
            "schema_version": LOCAL_MACHINE_RAW_DIFF_ARTIFACT_VERSION,
            "object_type": "raw_diff",
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "entry_count": len(rows),
            "created_at": payload["created_at"],
        }

    @staticmethod
    def _diff_summary(diff_entries: Sequence[Any], *, limit: int = 20) -> Dict[str, Any]:
        rows = [entry.to_dict() if hasattr(entry, "to_dict") else dict(entry) for entry in diff_entries]
        counts: Dict[str, int] = {}
        total_size = 0
        patch_bytes = 0
        for row in rows:
            status = str(row.get("status", "") or "unknown")
            counts[status] = counts.get(status, 0) + 1
            total_size += int(row.get("size_bytes", 0) or 0)
            patch_bytes += len(str(row.get("text_patch", "") or "").encode("utf-8"))
        examples = []
        for row in rows[: max(0, int(limit))]:
            examples.append(
                {
                    "relative_path": str(row.get("relative_path", "") or ""),
                    "status": str(row.get("status", "") or ""),
                    "size_bytes": int(row.get("size_bytes", 0) or 0),
                    "source_sha256": str(row.get("source_sha256", "") or ""),
                    "mirror_sha256": str(row.get("mirror_sha256", "") or ""),
                    "patch_sha256": str(row.get("patch_sha256", "") or ""),
                }
            )
        return {
            "schema_version": "conos.local_machine.diff_summary/v1",
            "entry_count": len(rows),
            "status_counts": counts,
            "total_size_bytes": total_size,
            "total_patch_bytes": patch_bytes,
            "examples": examples,
            "examples_truncated": len(rows) > len(examples),
        }

    def _observation_payload(
        self,
        manifest: Dict[str, Any],
        sync_plan: Dict[str, Any],
        tools: Sequence[ToolSpec],
    ) -> Dict[str, Any]:
        diff_entries = compute_mirror_diff(self.source_root, self.mirror_root)
        diff_ref = self._write_raw_diff_artifact(diff_entries, manifest)
        diff_summary = self._diff_summary(diff_entries)
        diff = list(diff_summary.get("examples", []) or [])
        internet_manifest = (
            load_internet_manifest(self._internet_output_root(manifest))
            if self.internet_enabled
            else {}
        )
        investigation_state = self._load_investigation_state()
        local_mirror = {
            "adapter_version": LOCAL_MACHINE_ADAPTER_VERSION,
            "instruction": self.instruction,
            "source_root": str(self.source_root),
            "mirror_root": str(self.mirror_root),
            "workspace_root": str(manifest.get("workspace_root", "") or ""),
            "control_root": str(manifest.get("control_root", "") or ""),
            "workspace_initial_state": str(manifest.get("workspace_initial_state", "empty") or "empty"),
            "workspace_file_count": int(manifest.get("workspace_file_count", 0) or 0),
            "materialized_files": list(manifest.get("materialized_files", []) or []),
            "external_baselines": list(manifest.get("external_baselines", []) or []),
            "diff_ref": dict(diff_ref),
            "raw_diff_ref": dict(diff_ref),
            "diff_summary": dict(diff_summary),
            "diff_entries": diff,
            "sync_plan": dict(sync_plan),
            "last_action": dict(self._last_action),
            "command_executed": bool(self._command_executed),
            "latest_command_returncode": self._last_command_returncode,
            "command_failed": bool(self._command_failed),
            "default_command_present": bool(self.default_command),
            "allow_empty_exec": bool(self.allow_empty_exec),
            "default_command_timeout_seconds": int(self.default_command_timeout_seconds),
            "execution_backend": str(self.execution_backend),
            "docker_image": str(self.docker_image),
            "vm_provider": str(self.vm_provider),
            "vm_name": str(self.vm_name),
            "vm_host": str(self.vm_host),
            "vm_workdir": str(self.vm_workdir),
            "vm_network_mode": str(self.vm_network_mode),
            "vm_sync_mode": str(self.vm_sync_mode),
            "execution_boundary": self._execution_boundary_report(),
            "terminal_after_plan": bool(self.terminal_after_plan),
            "internet_enabled": bool(self.internet_enabled),
            "deterministic_fallback_enabled": bool(self.deterministic_fallback_enabled),
            "prefer_llm_kwargs": bool(self.prefer_llm_kwargs),
            "prefer_llm_patch_proposals": bool(self.prefer_llm_patch_proposals),
            "llm_thinking_mode": str(self.llm_thinking_mode),
            "require_llm_generation": bool(self.require_llm_generation),
            "require_market_evidence_reference": bool(self.require_market_evidence_reference),
            "require_non_template_product": bool(self.require_non_template_product),
            "internet_ingress": dict(internet_manifest),
            "last_internet_artifact": dict(self._last_internet_artifact),
            "investigation": dict(investigation_state),
            "investigation_phase": str(investigation_state.get("investigation_phase") or "discover"),
            "terminal_state": str(investigation_state.get("terminal_state") or ""),
            "completion_reason": str(investigation_state.get("completion_reason") or ""),
            "terminal_tick": investigation_state.get("terminal_tick"),
            "verified_completion": bool(investigation_state.get("verified_completion", False)),
            "action_schema_registry": action_schema_registry_payload(),
            "action_grounding": dict(investigation_state.get("grounding", {}) or {}),
            "action_governance": dict(investigation_state.get("action_governance", {}) or {}),
            "llm_budget": self._llm_budget_report(investigation_state),
            "formal_evidence_ledger": self._formal_evidence_summary(),
            "candidate_files": list(investigation_state.get("candidate_files", []) or []),
            "learning_hints_present": bool(self.learning_context.get("hint_text")),
            "end_to_end_learning": dict(self.learning_context),
            "acquire_attempted": bool(self._acquire_attempted),
            "last_acquire_selected_count": int(self._last_acquire_selected_count),
            "applied": bool(self._applied),
            "audit_events": list(manifest.get("audit_events", []) or []),
        }
        return {
            "success": True,
            "reward": 0.0,
            "terminal": bool(self._terminal),
            "done": bool(self._terminal),
            "state": "LOCAL_MACHINE_MIRROR",
            "local_mirror": local_mirror,
            "available_functions": [tool.name for tool in tools],
            "function_signatures": {tool.name: dict(tool.input_schema or {}) for tool in tools},
            "sandbox_label": "best_effort_local_mirror",
            "not_os_security_sandbox": True,
        }

    @staticmethod
    def _coerce_action(action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            return dict(action)
        if hasattr(action, "payload"):
            return {
                "kind": str(getattr(action, "kind", "") or ""),
                "payload": dict(getattr(action, "payload", {}) or {}),
            }
        return {"kind": str(action or "wait")}

    @staticmethod
    def _extract_tool_call(action: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        if str(action.get("kind", "") or "").strip().lower() == "wait":
            return "wait", {}
        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        function_name = str(
            action.get("function_name")
            or action.get("action")
            or tool_args.get("function_name")
            or tool_args.get("action")
            or payload.get("function_name")
            or payload.get("action")
            or ""
        ).strip()
        # Merge from the least authoritative envelope to the selected action
        # itself. LLM-filled payload/tool_args can be stale or generic; the
        # selected top-level kwargs are the action semantic authority.
        kwargs = {}
        for candidate in (
            payload.get("args"),
            payload.get("kwargs"),
            tool_args.get("args"),
            tool_args.get("kwargs"),
            action.get("args"),
            action.get("kwargs"),
        ):
            if isinstance(candidate, dict):
                kwargs.update(candidate)
        return function_name, kwargs

    def _raw_success(self, *, function_name: str, reward: float, state: str, success: bool = True, **extra: Any) -> Dict[str, Any]:
        payload = {
            "success": bool(success),
            "reward": float(reward),
            "terminal": bool(self._terminal),
            "done": bool(self._terminal),
            "state": state,
            "function_name": str(function_name),
            "local_machine_adapter": LOCAL_MACHINE_ADAPTER_VERSION,
        }
        payload.update(extra)
        return payload

    def _raw_failure(self, *, function_name: str, exc: Exception, action: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "reward": 0.0,
            "terminal": bool(self._terminal),
            "done": bool(self._terminal),
            "state": "LOCAL_MACHINE_ERROR",
            "function_name": str(function_name),
            "failure_reason": str(exc),
            "exception_type": type(exc).__name__,
            "action_input": dict(action),
            "local_machine_adapter": LOCAL_MACHINE_ADAPTER_VERSION,
        }

    @staticmethod
    def _zero_arg_schema(description: str) -> Dict[str, Any]:
        return {
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def _tool_acquire(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_acquire",
            description="Materialize source files selected from the current instruction and declared candidates.",
            input_schema=self._zero_arg_schema("Use the adapter instruction and candidate paths."),
            side_effects=["reads declared source files", "writes mirror workspace"],
            risk_notes=["does not write to source root", "selection is limited to supplied candidate paths"],
            capability_class="local_mirror_materialization",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_no_op_complete(self) -> ToolSpec:
        return ToolSpec(
            name="no_op_complete",
            description="Acknowledge that this local-machine task is already verified and terminal.",
            input_schema=self._schema(
                "No-op after verified completion.",
                {"reason": {"type": "string"}},
                [],
            ),
            side_effects=[],
            risk_notes=["does not read or write workspace files"],
            capability_class="local_machine_completion",
            side_effect_class="none",
            approval_required=False,
            risk_level="low",
        )

    def _tool_fetch(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_fetch",
            description="Materialize explicitly declared source files into the mirror workspace.",
            input_schema=self._zero_arg_schema("Use adapter fetch paths."),
            side_effects=["reads declared source files", "writes mirror workspace"],
            risk_notes=["does not write to source root"],
            capability_class="local_mirror_materialization",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="low",
        )

    @staticmethod
    def _schema(description: str, properties: Dict[str, Any], required: Sequence[str] = ()) -> Dict[str, Any]:
        return {
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": [str(item) for item in required],
            },
        }

    def _tool_repo_tree(self) -> ToolSpec:
        return ToolSpec(
            name="repo_tree",
            description="Read a bounded source-root directory inventory. Use this first when no candidate files are known.",
            input_schema=self._schema(
                "List source files and directories without exposing shell commands.",
                {
                    "path": {"type": "string", "description": "Source-relative path or source root name. Defaults to the root."},
                    "depth": {"type": "integer", "description": "Traversal depth, usually 2."},
                    "max_entries": {"type": "integer", "description": "Maximum entries to return."},
                    "exclude": {"type": "array", "items": {"type": "string"}},
                },
            ),
            side_effects=["reads source tree", "writes investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_discovery",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_repo_find(self) -> ToolSpec:
        return ToolSpec(
            name="repo_find",
            description="Find source files or directories by name pattern.",
            input_schema=self._schema(
                "Search filenames under the source root.",
                {
                    "root": {"type": "string"},
                    "name_pattern": {"type": "string"},
                    "kind": {"type": "string", "enum": ["any", "file", "dir"]},
                    "max_results": {"type": "integer"},
                },
            ),
            side_effects=["reads source tree", "writes investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_discovery",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_repo_grep(self) -> ToolSpec:
        return ToolSpec(
            name="repo_grep",
            description="Search source text with bounded globs and match limits.",
            input_schema=self._schema(
                "Search text files under the source root.",
                {
                    "root": {"type": "string"},
                    "query": {"type": "string"},
                    "globs": {"type": "array", "items": {"type": "string"}},
                    "max_matches": {"type": "integer"},
                    "case_sensitive": {"type": "boolean"},
                    "regex": {"type": "boolean"},
                },
                required=["query"],
            ),
            side_effects=["reads source files", "writes investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_discovery",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_file_read(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description="Read a bounded source or mirror file slice.",
            input_schema=self._schema(
                "Read a file by line range.",
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                required=["path"],
            ),
            side_effects=["reads one file", "writes investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_reading",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_file_outline(self) -> ToolSpec:
        return ToolSpec(
            name="file_outline",
            description="Return a compact outline of a source or mirror file.",
            input_schema=self._schema("Outline a file.", {"path": {"type": "string"}}, required=["path"]),
            side_effects=["reads one file"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_reading",
            side_effect_class="read_only",
            approval_required=False,
            risk_level="low",
        )

    def _tool_file_summary(self) -> ToolSpec:
        return ToolSpec(
            name="file_summary",
            description="Return metadata, first lines, and lightweight outline for one file.",
            input_schema=self._schema("Summarize a file.", {"path": {"type": "string"}}, required=["path"]),
            side_effects=["reads one file"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_reading",
            side_effect_class="read_only",
            approval_required=False,
            risk_level="low",
        )

    def _tool_note_write(self) -> ToolSpec:
        return ToolSpec(
            name="note_write",
            description="Persist an investigation note so later steps retain why a file or hypothesis matters.",
            input_schema=self._schema(
                "Write an investigation note.",
                {
                    "kind": {"type": "string", "enum": ["finding", "risk", "decision", "edit", "test"]},
                    "content": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                required=["content"],
            ),
            side_effects=["writes mirror investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_investigation_state",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_hypothesis_add(self) -> ToolSpec:
        return ToolSpec(
            name="hypothesis_add",
            description="Persist a lifecycle-tracked candidate explanation for the current codebase behavior.",
            input_schema=self._schema(
                "Add an investigation hypothesis.",
                {
                    "claim": {"type": "string"},
                    "family": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "competing_with": {"type": "array", "items": {"type": "string"}},
                    "predictions": {"type": "object"},
                    "falsifiers": {"type": "object"},
                },
                required=["claim"],
            ),
            side_effects=["writes mirror investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_investigation_state",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_hypothesis_update(self) -> ToolSpec:
        return ToolSpec(
            name="hypothesis_update",
            description="Update one hypothesis posterior from explicit evidence with damped belief revision.",
            input_schema=self._schema(
                "Update a hypothesis from evidence.",
                {
                    "hypothesis_id": {"type": "string"},
                    "signal": {"type": "string", "enum": ["support", "contradict", "neutral"]},
                    "evidence_ref": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "strength": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                required=["hypothesis_id", "signal"],
            ),
            side_effects=["writes mirror investigation state", "writes hypothesis lifecycle evidence"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_hypothesis_lifecycle",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_hypothesis_compete(self) -> ToolSpec:
        return ToolSpec(
            name="hypothesis_compete",
            description="Record that two hypotheses are competing explanations that need discriminating evidence.",
            input_schema=self._schema(
                "Mark two hypotheses as competing.",
                {
                    "hypothesis_a": {"type": "string"},
                    "hypothesis_b": {"type": "string"},
                    "reason": {"type": "string"},
                },
                required=["hypothesis_a", "hypothesis_b"],
            ),
            side_effects=["writes mirror investigation state", "writes hypothesis lifecycle evidence"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_hypothesis_lifecycle",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_discriminating_test_add(self) -> ToolSpec:
        return ToolSpec(
            name="discriminating_test_add",
            description="Propose a minimal action whose outcome distinguishes two competing hypotheses.",
            input_schema=self._schema(
                "Record a discriminating test proposal.",
                {
                    "hypothesis_a": {"type": "string"},
                    "hypothesis_b": {"type": "string"},
                    "action": {"type": "object"},
                    "action_name": {"type": "string"},
                    "args": {"type": "object"},
                    "expected_if_a": {"type": "string"},
                    "expected_if_b": {"type": "string"},
                    "why": {"type": "string"},
                },
                required=["hypothesis_a", "hypothesis_b", "expected_if_a", "expected_if_b", "why"],
            ),
            side_effects=["writes mirror investigation state", "writes hypothesis lifecycle evidence"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_hypothesis_lifecycle",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_candidate_files_set(self) -> ToolSpec:
        return ToolSpec(
            name="candidate_files_set",
            description="Persist the current candidate file set for focused later reads and edits.",
            input_schema=self._schema(
                "Set candidate files.",
                {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                required=["files"],
            ),
            side_effects=["writes mirror investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_investigation_state",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_investigation_status(self) -> ToolSpec:
        return ToolSpec(
            name="investigation_status",
            description="Read persisted notes, hypotheses, candidate files, and last probe summaries.",
            input_schema=self._zero_arg_schema("No arguments required."),
            side_effects=["reads mirror investigation state"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_investigation_state",
            side_effect_class="read_only",
            approval_required=False,
            risk_level="low",
        )

    def _tool_propose_patch(self) -> ToolSpec:
        return ToolSpec(
            name="propose_patch",
            description="Generate one bounded patch proposal for a localized target file and accept it only if verifier tests pass.",
            input_schema=self._schema(
                "Generate, apply, verify, and rollback a bounded candidate diff.",
                {
                    "target_file": {"type": "string"},
                    "max_changed_lines": {"type": "integer"},
                    "expected_tests": {"type": "array", "items": {"type": "string"}},
                },
                required=[],
            ),
            side_effects=["may write mirror workspace", "runs tests", "writes investigation state"],
            risk_notes=[
                "does not write source root",
                "forbids test-file edits by default",
                "rolls back rejected proposals",
                "full test verification is required before acceptance",
            ],
            capability_class="local_machine_edit",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_apply_patch_atomic(self) -> ToolSpec:
        return ToolSpec(
            name="apply_patch",
            description="Apply one bounded unified diff to the mirror workspace.",
            input_schema=self._schema(
                "Apply a small patch. Prefer one file and a few hunks.",
                {
                    "patch": {"type": "string"},
                    "max_files": {"type": "integer"},
                    "max_hunks": {"type": "integer"},
                },
                required=["patch"],
            ),
            side_effects=["writes mirror workspace", "writes investigation state"],
            risk_notes=[
                "does not write source root",
                "bounded by max_files and max_hunks",
                "requires prior evidence refs through action governance",
            ],
            capability_class="local_machine_edit",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_edit_replace_range(self) -> ToolSpec:
        return ToolSpec(
            name="edit_replace_range",
            description="Replace a small line range in one mirror workspace file, materializing it first from source if needed.",
            input_schema=self._schema(
                "Replace a line range.",
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "replacement": {"type": "string"},
                },
                required=["path", "start_line", "end_line", "replacement"],
            ),
            side_effects=["writes one mirror workspace file"],
            risk_notes=["does not write source root", "requires prior evidence refs through action governance"],
            capability_class="local_machine_edit",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_edit_insert_after(self) -> ToolSpec:
        return ToolSpec(
            name="edit_insert_after",
            description="Insert text after a specific line in one mirror workspace file.",
            input_schema=self._schema(
                "Insert text after a line.",
                {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "text": {"type": "string"},
                },
                required=["path", "line", "text"],
            ),
            side_effects=["writes one mirror workspace file"],
            risk_notes=["does not write source root", "requires prior evidence refs through action governance"],
            capability_class="local_machine_edit",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_create_file(self) -> ToolSpec:
        return ToolSpec(
            name="create_file",
            description="Create one mirror workspace file.",
            input_schema=self._schema(
                "Create a file in the mirror workspace.",
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                required=["path", "content"],
            ),
            side_effects=["writes one mirror workspace file"],
            risk_notes=["does not write source root", "requires prior evidence refs through action governance"],
            capability_class="local_machine_edit",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_delete_file(self) -> ToolSpec:
        return ToolSpec(
            name="delete_file",
            description="Delete one mirror workspace file.",
            input_schema=self._schema("Delete a mirror file.", {"path": {"type": "string"}}, required=["path"]),
            side_effects=["deletes one mirror workspace file"],
            risk_notes=["does not write source root", "requires prior evidence refs through action governance"],
            capability_class="local_machine_edit",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_run_test(self) -> ToolSpec:
        return ToolSpec(
            name="run_test",
            description="Run pytest on a bounded target in the mirror workspace.",
            input_schema=self._schema(
                "Run tests in the mirror workspace.",
                {
                    "target": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
            ),
            side_effects=["executes pytest in mirror workspace", "writes run output artifact"],
            risk_notes=["does not write source root", "not an OS security sandbox"],
            capability_class="local_machine_validation",
            side_effect_class="mirror_workspace_execution",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_run_lint(self, name: str) -> ToolSpec:
        return ToolSpec(
            name=name,
            description="Run a local Python compile check on a mirror workspace target.",
            input_schema=self._schema(
                "Run a validation check in the mirror workspace.",
                {
                    "target": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
            ),
            side_effects=["executes Python validation in mirror workspace", "writes run output artifact"],
            risk_notes=["does not write source root", "not an OS security sandbox"],
            capability_class="local_machine_validation",
            side_effect_class="mirror_workspace_execution",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_read_run_output(self) -> ToolSpec:
        return ToolSpec(
            name="read_run_output",
            description="Read stdout/stderr from a previous run_test, run_lint, run_typecheck, or run_build call.",
            input_schema=self._schema(
                "Read saved validation output.",
                {
                    "run_ref": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
            ),
            side_effects=["reads run output artifact"],
            risk_notes=["does not write source or mirror workspace"],
            capability_class="local_machine_validation",
            side_effect_class="read_only",
            approval_required=False,
            risk_level="low",
        )

    def _tool_read_test_failure(self) -> ToolSpec:
        tool = self._tool_read_run_output()
        tool.name = "read_test_failure"
        tool.description = "Read failure-focused stdout/stderr from a previous validation run."
        return tool

    def _tool_exec(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_exec",
            description="Restricted emergency fallback. Prefer atomic local-machine actions for discovery, editing, and validation.",
            input_schema={
                "description": "Run default_command, or provide a short explicit allowlisted command when atomic actions cannot express the need.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "description": "Optional short command as argv strings or a shell-like string. Long generated scripts are rejected.",
                        },
                        "purpose": {
                            "type": "string",
                            "enum": ["inspect", "test", "format", "build"],
                            "description": "Required for explicit fallback commands.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Required bounded mirror-workspace target path for explicit fallback commands.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Required for explicit fallback commands.",
                        },
                        "backend": {
                            "type": "string",
                            "enum": ["local", "docker", "vm", "managed-vm"],
                            "description": "Optional execution backend. Defaults to the adapter execution_backend.",
                        },
                        "docker_image": {
                            "type": "string",
                            "description": "Optional Docker image when backend=docker.",
                        },
                        "vm_provider": {
                            "type": "string",
                            "enum": ["auto", "managed", "managed-vm", "lima", "ssh"],
                            "description": "Optional real VM provider when backend=vm.",
                        },
                        "vm_name": {"type": "string"},
                        "vm_host": {"type": "string"},
                        "vm_workdir": {"type": "string"},
                        "vm_network_mode": {
                            "type": "string",
                            "enum": ["provider_default", "configured_isolated"],
                        },
                        "vm_sync_mode": {
                            "type": "string",
                            "enum": ["none", "push", "pull", "push-pull"],
                            "description": "Optional explicit VM workspace sync around backend=vm execution.",
                        },
                    },
                    "required": [],
                },
            },
            side_effects=["executes allowlisted command", "may write mirror workspace"],
            risk_notes=["best-effort local mirror isolation; not an OS security sandbox"],
            capability_class="local_mirror_execution",
            side_effect_class="mirror_workspace_execution",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_internet_fetch(self) -> ToolSpec:
        return ToolSpec(
            name="internet_fetch",
            description="Fetch a public HTTP/HTTPS URL into the audited internet artifact store.",
            input_schema={
                "description": "Fetch a public URL and store the response as an audited artifact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Public http or https URL. Credentials and local/private network hosts are blocked by default.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional local artifact filename hint.",
                        },
                    },
                    "required": ["url"],
                },
            },
            side_effects=["performs an HTTP GET", "writes mirror control artifact store"],
            risk_notes=[
                "best-effort URL policy gate; not a browser security sandbox",
                "blocks URL credentials and local/private IP hosts by default",
                "enforces a max byte limit before writing artifacts",
            ],
            capability_class="internet_ingress",
            side_effect_class="network_read_artifact_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_internet_fetch_project(self) -> ToolSpec:
        return ToolSpec(
            name="internet_fetch_project",
            description="Fetch an internet-hosted project into the audited artifact store using generic git or archive handling.",
            input_schema={
                "description": "Fetch a project from a public git URL or zip/tar archive URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Public http or https project URL. GitHub is treated as an ordinary host.",
                        },
                        "source_type": {
                            "type": "string",
                            "enum": ["auto", "git", "archive"],
                            "description": "Use auto detection, force git clone, or force archive download and extraction.",
                        },
                        "ref": {
                            "type": "string",
                            "description": "Optional git branch or tag for git project URLs.",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Optional git clone depth. Defaults to 1.",
                        },
                        "directory_name": {
                            "type": "string",
                            "description": "Optional project directory name hint.",
                        },
                    },
                    "required": ["url"],
                },
            },
            side_effects=["performs an HTTP/HTTPS git clone or archive download", "writes mirror control artifact store"],
            risk_notes=[
                "best-effort URL policy gate; not a VM or browser sandbox",
                "blocks URL credentials and local/private IP hosts by default",
                "stores fetched project outside the source root until a separate approved sync path is used",
            ],
            capability_class="internet_ingress_project",
            side_effect_class="network_read_artifact_write",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_plan(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_plan",
            description="Build an auditable sync plan from mirror changes.",
            input_schema=self._zero_arg_schema("No arguments required."),
            side_effects=["writes mirror control sync plan"],
            risk_notes=["does not write to source root"],
            capability_class="local_mirror_review",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_apply(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_apply",
            description="Apply an approved sync plan back to the source root.",
            input_schema=self._zero_arg_schema("Use the latest plan id with machine approval when eligible."),
            side_effects=["writes approved files to source root"],
            risk_notes=[
                "requires sync plan id",
                "machine approval is limited to text-like added or modified files",
                "code sync requires prior passing validation through action governance",
            ],
            capability_class="local_mirror_sync",
            side_effect_class="source_filesystem_write",
            approval_required=True,
            risk_level="high",
        )

    def to_dict(self) -> Dict[str, Any]:
        spec = self.get_generic_task_spec()
        obs = self.observe()
        return {
            "task_spec": asdict(spec),
            "observation": dict(obs.raw),
        }
