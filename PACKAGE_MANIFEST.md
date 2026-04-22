## Con OS Runtime-Only Package

This package contains the current runnable Con OS source tree only.

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
- `scripts/check_conos_repo_layout.py`
- `trace_runtime.py`
- `pyproject.toml`
- `README.md`

Excluded on purpose:
- `tests/`
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

Minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python scripts/check_conos_repo_layout.py
```

Entry points:

```bash
python scripts/run_arc_agi3.py
python scripts/run_webarena.py
```
