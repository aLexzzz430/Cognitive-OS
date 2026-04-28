from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
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
from core.runtime.state_store import RuntimeStateStore
from core.surfaces.base import ActionResult, SurfaceObservation, ToolSpec
from integrations.local_machine.execution_actions import LocalMachineExecutionActionsMixin
from integrations.local_machine.grounding_state import LocalMachineGroundingStateMixin
from integrations.local_machine.investigation_actions import LocalMachineInvestigationActionsMixin
from integrations.local_machine import tool_specs
from integrations.local_machine.action_grounding import (
    action_schema_registry_payload,
    is_completed_verified_context,
    is_local_machine_side_effect_action,
    side_effect_after_completion_event,
    validate_local_machine_action,
)
from integrations.local_machine.budget_policy import budget_policy_report
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


class LocalMachineSurfaceAdapter(
    LocalMachineGroundingStateMixin,
    LocalMachineExecutionActionsMixin,
    LocalMachineInvestigationActionsMixin,
):
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
        return tool_specs.with_internet_tools(tools, internet_enabled=self.internet_enabled)

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
        tools = tool_specs.atomic_tools()
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
                tools.append(tool_specs.tool_repo_tree())
                if self.allow_empty_exec and not self._command_executed:
                    tools.append(tool_specs.tool_exec())
                return self._with_internet_tool(tools)
            tools.extend(self._atomic_tools(sync_plan=sync_plan))
            if not research_first:
                tools = self._with_internet_tool(tools)
            if (self.default_command or self.allow_empty_exec) and not self._command_executed:
                tools.append(tool_specs.tool_exec())
            if not sync_plan:
                tools.append(tool_specs.tool_plan())
            if self.expose_apply_tool and sync_plan and not self._applied:
                tools.append(tool_specs.tool_apply())
            return tools

        workspace_count = int(manifest.get("workspace_file_count", 0) or 0)
        if workspace_count <= 0:
            tools: list[ToolSpec] = []
            can_empty_exec = bool(self.allow_empty_exec and not self._command_executed)
            if can_empty_exec:
                tools.append(tool_specs.tool_exec())
            if self.instruction or self.candidate_paths:
                skip_repeated_empty_acquire = (
                    self._acquire_attempted
                    and self._last_acquire_selected_count <= 0
                    and bool(self.default_command)
                )
                if not skip_repeated_empty_acquire:
                    tools.append(tool_specs.tool_acquire())
            if self.fetch_paths and not tools:
                tools.append(tool_specs.tool_fetch())
            if self._command_executed and not sync_plan:
                tools.append(tool_specs.tool_plan())
            return self._with_internet_tool(tools)

        if (self.default_command or self.internet_enabled or self.allow_empty_exec) and not self._command_executed:
            return self._with_internet_tool([tool_specs.tool_exec()])
        if not sync_plan:
            return self._with_internet_tool([tool_specs.tool_plan()])
        if self.expose_apply_tool and not self._applied:
            return self._with_internet_tool([tool_specs.tool_apply()])
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

    def to_dict(self) -> Dict[str, Any]:
        spec = self.get_generic_task_spec()
        obs = self.observe()
        return {
            "task_spec": asdict(spec),
            "observation": dict(obs.raw),
        }
