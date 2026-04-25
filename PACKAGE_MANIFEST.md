## Con OS Runtime Source Snapshot

This package contains the current runnable Con OS source tree plus a minimal
public smoke test. The runtime contract is still the source surface listed
below; tests are included here only to verify the published boundary.

Included:
- `core/`
- `core/app/`
- `planner/`
- `decision/`
- `modules/`
- `modules/control_plane/`
- `evolution/`
- `self_model/`
- `integrations/arc_agi3/`
- `integrations/local_machine/`
- `integrations/webarena/`
- `scripts/conos.py`
- `scripts/build_macos_app.py`
- `scripts/local_mirror.py`
- `scripts/run_arc_agi3.py`
- `scripts/run_local_machine.py`
- `scripts/run_webarena.py`
- `scripts/eval_metrics_panel.py`
- `scripts/cognitive_loop_ablation.py`
- `scripts/check_conos_repo_layout.py`
- `scripts/check_runtime_preflight.py`
- `tests/test_public_repo_smoke.py`
- `conos_cli.py`
- `trace_runtime.py`
- `pyproject.toml`
- `README.md`

Not part of the runtime contract:
- `tests/`

Excluded on purpose:
- `conos_evals/`
- `eval/`
- `runtime/`
- `reports/`
- `audit/`
- `memory/`
- `trace/`
- `external/`
- `private_cognitive_core/`
- `private-cognitive-core/`
- `state/state.json`
- generated logs such as `*.jsonl`

Execution safety note:

The runtime marks execution isolation as best-effort. Execution tickets carry a
`sandbox_audit` envelope that records file/path, write-path, network-target, and
credential/secret-lease signals, but this package does not claim OS-level
filesystem or network sandboxing.

Action authority note:

Action governance lives in `modules/control_plane/action_governance.py`.
Tool calls are not automatically trusted: mirror writes require evidence refs,
code source sync requires passing validation, source sync requires an approved
plan, and repeated action failures can downgrade an agent's write/exec
authority.

Failure learning note:

Failure learning lives in `core/runtime/failure_learning.py` and the
`failure_learning_objects` SQLite table. Failed runs can now persist a formal
failure mode, violated assumption, evidence refs, missing tool or bad policy,
suggested regression test, suggested governance rule, and a future retrieval
object. The local-machine runner keeps these objects in the object layer and
exposes them as JSON, rather than expanding raw failure records into prompt
text.

Minimal setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
conos preflight
conos layout
```

Entry points:

```bash
conos run arc-agi3 --game vc33
conos run local-machine --instruction "inspect README" --candidate README.md --max-ticks 2
conos run webarena --task-id smoke --instruction "open the target page"
conos app
conos app --summary-json
conos mirror init --mirror-root runtime/mirrors/session-1
conos mirror fetch --mirror-root runtime/mirrors/session-1 --path README.md
conos mirror exec --mirror-root runtime/mirrors/session-1 --allow-command python3 -- python3 -c "from pathlib import Path; print(Path('README.md').exists())"
conos mirror exec --mirror-root runtime/mirrors/session-1 --backend docker --allow-command python3 -- python3 -c "print('docker backend')"
conos mirror plan --mirror-root runtime/mirrors/session-1
conos mirror apply --mirror-root runtime/mirrors/session-1 --plan-id <plan_id> --approved-by machine
conos mirror rollback --mirror-root runtime/mirrors/session-1 --plan-id <plan_id>
conos llm control-plane --route structured_answer --required-capability structured_output --permission generate_text
conos ui runtime reports audit
conos eval runtime reports audit
conos dashboard runtime reports audit
python scripts/build_macos_app.py --output-dir dist
python scripts/cognitive_loop_ablation.py --task-count 25 --seed 17
python scripts/local_mirror.py init --mirror-root runtime/mirrors/session-1
python scripts/run_arc_agi3.py
python scripts/run_local_machine.py --instruction "inspect README" --candidate README.md
python scripts/run_webarena.py
python scripts/eval_metrics_panel.py runtime reports audit
```
