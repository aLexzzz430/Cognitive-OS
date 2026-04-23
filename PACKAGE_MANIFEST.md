## Con OS Runtime Source Snapshot

This package contains the current runnable Con OS source tree plus a minimal
public smoke test. The runtime contract is still the source surface listed
below; tests are included here only to verify the published boundary.

Included:
- `core/`
- `planner/`
- `decision/`
- `modules/`
- `evolution/`
- `self_model/`
- `integrations/arc_agi3/`
- `integrations/webarena/`
- `scripts/run_arc_agi3.py`
- `scripts/run_webarena.py`
- `scripts/eval_metrics_panel.py`
- `scripts/check_conos_repo_layout.py`
- `scripts/check_runtime_preflight.py`
- `tests/test_public_repo_smoke.py`
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

Minimal setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python scripts/check_runtime_preflight.py
python scripts/check_conos_repo_layout.py
```

Entry points:

```bash
python scripts/run_arc_agi3.py
python scripts/run_webarena.py
python scripts/eval_metrics_panel.py runtime reports audit
```
