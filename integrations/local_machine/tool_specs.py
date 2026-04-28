"""Tool specifications for the distilled local-machine surface."""

from __future__ import annotations

from typing import Any, Sequence

from core.surfaces.base import ToolSpec


def zero_arg_schema(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def schema(description: str, properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": dict(properties),
            "required": [str(item) for item in required],
        },
    }


def tool_acquire() -> ToolSpec:
    return ToolSpec(
        name="mirror_acquire",
        description="Materialize source files selected from the current instruction and declared candidates.",
        input_schema=zero_arg_schema("Use the adapter instruction and candidate paths."),
        side_effects=["reads declared source files", "writes mirror workspace"],
        risk_notes=["does not write to source root", "selection is limited to supplied candidate paths"],
        capability_class="local_mirror_materialization",
        side_effect_class="mirror_workspace_write",
        approval_required=False,
        risk_level="low",
    )


def tool_no_op_complete() -> ToolSpec:
    return ToolSpec(
        name="no_op_complete",
        description="Acknowledge that this local-machine task is already verified and terminal.",
        input_schema=schema(
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


def tool_fetch() -> ToolSpec:
    return ToolSpec(
        name="mirror_fetch",
        description="Materialize explicitly declared source files into the mirror workspace.",
        input_schema=zero_arg_schema("Use adapter fetch paths."),
        side_effects=["reads declared source files", "writes mirror workspace"],
        risk_notes=["does not write to source root"],
        capability_class="local_mirror_materialization",
        side_effect_class="mirror_workspace_write",
        approval_required=False,
        risk_level="low",
    )


def tool_repo_tree() -> ToolSpec:
    return ToolSpec(
        name="repo_tree",
        description="Read a bounded source-root directory inventory. Use this first when no candidate files are known.",
        input_schema=schema(
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


def tool_repo_find() -> ToolSpec:
    return ToolSpec(
        name="repo_find",
        description="Find source files or directories by name pattern.",
        input_schema=schema(
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


def tool_repo_grep() -> ToolSpec:
    return ToolSpec(
        name="repo_grep",
        description="Search source text with bounded globs and match limits.",
        input_schema=schema(
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


def tool_file_read() -> ToolSpec:
    return ToolSpec(
        name="file_read",
        description="Read a bounded source or mirror file slice.",
        input_schema=schema(
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


def tool_file_outline() -> ToolSpec:
    return ToolSpec(
        name="file_outline",
        description="Return a compact outline of a source or mirror file.",
        input_schema=schema("Outline a file.", {"path": {"type": "string"}}, required=["path"]),
        side_effects=["reads one file"],
        risk_notes=["does not write source or mirror workspace"],
        capability_class="local_machine_reading",
        side_effect_class="read_only",
        approval_required=False,
        risk_level="low",
    )


def tool_file_summary() -> ToolSpec:
    return ToolSpec(
        name="file_summary",
        description="Return metadata, first lines, and lightweight outline for one file.",
        input_schema=schema("Summarize a file.", {"path": {"type": "string"}}, required=["path"]),
        side_effects=["reads one file"],
        risk_notes=["does not write source or mirror workspace"],
        capability_class="local_machine_reading",
        side_effect_class="read_only",
        approval_required=False,
        risk_level="low",
    )


def tool_note_write() -> ToolSpec:
    return ToolSpec(
        name="note_write",
        description="Persist an investigation note so later steps retain why a file or hypothesis matters.",
        input_schema=schema(
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


def tool_hypothesis_add() -> ToolSpec:
    return ToolSpec(
        name="hypothesis_add",
        description="Persist a lifecycle-tracked candidate explanation for the current codebase behavior.",
        input_schema=schema(
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


def tool_hypothesis_update() -> ToolSpec:
    return ToolSpec(
        name="hypothesis_update",
        description="Update one hypothesis posterior from explicit evidence with damped belief revision.",
        input_schema=schema(
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


def tool_hypothesis_compete() -> ToolSpec:
    return ToolSpec(
        name="hypothesis_compete",
        description="Record that two hypotheses are competing explanations that need discriminating evidence.",
        input_schema=schema(
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


def tool_discriminating_test_add() -> ToolSpec:
    return ToolSpec(
        name="discriminating_test_add",
        description="Propose a minimal action whose outcome distinguishes two competing hypotheses.",
        input_schema=schema(
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


def tool_candidate_files_set() -> ToolSpec:
    return ToolSpec(
        name="candidate_files_set",
        description="Persist the current candidate file set for focused later reads and edits.",
        input_schema=schema(
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


def tool_investigation_status() -> ToolSpec:
    return ToolSpec(
        name="investigation_status",
        description="Read persisted notes, hypotheses, candidate files, and last probe summaries.",
        input_schema=zero_arg_schema("No arguments required."),
        side_effects=["reads mirror investigation state"],
        risk_notes=["does not write source or mirror workspace"],
        capability_class="local_machine_investigation_state",
        side_effect_class="read_only",
        approval_required=False,
        risk_level="low",
    )


def tool_propose_patch() -> ToolSpec:
    return ToolSpec(
        name="propose_patch",
        description="Generate one bounded patch proposal for a localized target file and accept it only if verifier tests pass.",
        input_schema=schema(
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


def tool_apply_patch_atomic() -> ToolSpec:
    return ToolSpec(
        name="apply_patch",
        description="Apply one bounded unified diff to the mirror workspace.",
        input_schema=schema(
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


def tool_edit_replace_range() -> ToolSpec:
    return ToolSpec(
        name="edit_replace_range",
        description="Replace a small line range in one mirror workspace file, materializing it first from source if needed.",
        input_schema=schema(
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


def tool_edit_insert_after() -> ToolSpec:
    return ToolSpec(
        name="edit_insert_after",
        description="Insert text after a specific line in one mirror workspace file.",
        input_schema=schema(
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


def tool_create_file() -> ToolSpec:
    return ToolSpec(
        name="create_file",
        description="Create one mirror workspace file.",
        input_schema=schema(
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


def tool_delete_file() -> ToolSpec:
    return ToolSpec(
        name="delete_file",
        description="Delete one mirror workspace file.",
        input_schema=schema("Delete a mirror file.", {"path": {"type": "string"}}, required=["path"]),
        side_effects=["deletes one mirror workspace file"],
        risk_notes=["does not write source root", "requires prior evidence refs through action governance"],
        capability_class="local_machine_edit",
        side_effect_class="mirror_workspace_write",
        approval_required=False,
        risk_level="medium",
    )


def tool_run_test() -> ToolSpec:
    return ToolSpec(
        name="run_test",
        description="Run pytest on a bounded target in the mirror workspace.",
        input_schema=schema(
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


def tool_run_lint(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Run a local Python compile check on a mirror workspace target.",
        input_schema=schema(
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


def tool_read_run_output() -> ToolSpec:
    return ToolSpec(
        name="read_run_output",
        description="Read stdout/stderr from a previous run_test, run_lint, run_typecheck, or run_build call.",
        input_schema=schema(
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


def tool_read_test_failure() -> ToolSpec:
    tool = tool_read_run_output()
    tool.name = "read_test_failure"
    tool.description = "Read failure-focused stdout/stderr from a previous validation run."
    return tool


def tool_exec() -> ToolSpec:
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


def tool_internet_fetch() -> ToolSpec:
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


def tool_internet_fetch_project() -> ToolSpec:
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


def tool_plan() -> ToolSpec:
    return ToolSpec(
        name="mirror_plan",
        description="Build an auditable sync plan from mirror changes.",
        input_schema=zero_arg_schema("No arguments required."),
        side_effects=["writes mirror control sync plan"],
        risk_notes=["does not write to source root"],
        capability_class="local_mirror_review",
        side_effect_class="mirror_control_write",
        approval_required=False,
        risk_level="low",
    )


def tool_apply() -> ToolSpec:
    return ToolSpec(
        name="mirror_apply",
        description="Apply an approved sync plan back to the source root.",
        input_schema=zero_arg_schema("Use the latest plan id with machine approval when eligible."),
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


def atomic_tools() -> list[ToolSpec]:
    return [
        tool_repo_tree(),
        tool_repo_find(),
        tool_repo_grep(),
        tool_file_read(),
        tool_file_outline(),
        tool_file_summary(),
        tool_note_write(),
        tool_hypothesis_add(),
        tool_hypothesis_update(),
        tool_hypothesis_compete(),
        tool_discriminating_test_add(),
        tool_candidate_files_set(),
        tool_investigation_status(),
        tool_propose_patch(),
        tool_apply_patch_atomic(),
        tool_edit_replace_range(),
        tool_edit_insert_after(),
        tool_create_file(),
        tool_delete_file(),
        tool_run_test(),
        tool_run_lint("run_lint"),
        tool_run_lint("run_typecheck"),
        tool_run_lint("run_build"),
        tool_read_run_output(),
        tool_read_test_failure(),
        tool_no_op_complete(),
    ]


def with_internet_tools(tools: Sequence[ToolSpec], *, internet_enabled: bool) -> list[ToolSpec]:
    result = list(tools)
    if internet_enabled:
        existing = {tool.name for tool in result}
        if "internet_fetch" not in existing:
            result.append(tool_internet_fetch())
        if "internet_fetch_project" not in existing:
            result.append(tool_internet_fetch_project())
    return result
