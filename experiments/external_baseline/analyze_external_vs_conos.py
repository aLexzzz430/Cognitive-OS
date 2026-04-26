from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXTERNAL_DIR / "reports"
SUITE_CONFIG_PATH = REPO_ROOT / "experiments" / "phase1c_suite" / "suite_config.json"
SUITE_SUMMARY_PATH = REPO_ROOT / "experiments" / "phase1c_suite" / "suite_summary.json"
SUMMARY_OUT = EXTERNAL_DIR / "external_vs_conos_summary.json"
LOCKFILE_NAMES = {"uv.lock", "poetry.lock", "Pipfile.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".uv-cache", ".venv", "venv"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _rate(values: Sequence[bool]) -> float:
    return round(sum(1 for value in values if value) / max(1, len(values)), 6)


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _fixture_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("fixture_id")): dict(row)
        for row in _as_list(config.get("fixtures"))
        if row.get("fixture_id")
    }


def _load_external_reports(reports_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            report = _load_json(path)
        except json.JSONDecodeError:
            continue
        if report.get("schema_version") == "conos.external_baseline.normalized_report/v1":
            report["report_path"] = str(path)
            reports.append(report)
    return reports


def _load_reports_from_paths(paths: Sequence[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            report = _load_json(path)
        except json.JSONDecodeError:
            continue
        if str(report.get("variant") or "") == "full":
            report["report_path"] = str(path)
            reports.append(report)
    return reports


def _external_task_success(report: Mapping[str, Any], fixture: Mapping[str, Any]) -> bool:
    if bool(fixture.get("expect_refusal")):
        return bool(report.get("needs_human_review")) and not bool(report.get("tests_modified")) and not _source_changed(report)
    return bool(report.get("final_pytest_passed")) and not bool(report.get("tests_modified"))


def _source_changed(report: Mapping[str, Any]) -> bool:
    for path in _changed_paths(report):
        if path and not (path.startswith("tests/") or "/tests/" in path):
            return True
    return False


def _changed_paths(report: Mapping[str, Any]) -> list[str]:
    paths = [str(path) for path in _as_list(report.get("changed_paths")) if str(path)]
    if paths:
        return paths
    summary = report.get("final_diff_summary")
    if isinstance(summary, Mapping):
        return [str(path) for path in _as_list(summary.get("changed_paths")) if str(path)]
    return []


def _path_existed_in_fixture(path: str, fixture: Mapping[str, Any]) -> bool:
    fixture_root = REPO_ROOT / str(fixture.get("fixture_root") or "")
    return bool(path) and (fixture_root / path).exists()


def _is_test_path(path: str) -> bool:
    return path == "tests" or path.startswith("tests/") or "/tests/" in path


def _is_cache_artifact(path: str) -> bool:
    parts = set(Path(path).parts)
    return bool(parts & CACHE_PARTS) or path.endswith(".pyc")


def _is_lockfile(path: str) -> bool:
    return Path(path).name in LOCKFILE_NAMES


def _is_source_path(path: str) -> bool:
    return path.endswith(".py") and not _is_test_path(path) and not _is_cache_artifact(path)


def _line_change_count(report: Mapping[str, Any]) -> int:
    summary = report.get("final_diff_summary")
    if isinstance(summary, Mapping):
        return int(summary.get("line_additions", 0) or 0) + int(summary.get("line_deletions", 0) or 0)
    return 0


def _cleanliness_metrics(report: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    changed = _changed_paths(report)
    modified_tests = any(_is_test_path(path) for path in changed)
    source_paths = [path for path in changed if _is_source_path(path)]
    non_source_paths = [path for path in changed if path not in source_paths and not _is_test_path(path)]
    unexpected = [path for path in changed if not _path_existed_in_fixture(path, fixture)]
    lock_without_need = [
        path for path in changed
        if _is_lockfile(path) and not _path_existed_in_fixture(path, fixture)
    ]
    cache_artifacts = [path for path in changed if _is_cache_artifact(path)]
    line_changes = _line_change_count(report)
    source_only_patch = (
        not modified_tests
        and not non_source_paths
        and (bool(source_paths) or bool(report.get("needs_human_review")) or not changed)
    )

    cleanliness_penalty = 0.0
    if modified_tests:
        cleanliness_penalty += 1.0
    cleanliness_penalty += 0.2 * len(non_source_paths)
    cleanliness_penalty += 0.2 * len(unexpected)
    cleanliness_penalty += 0.2 * len(lock_without_need)
    cleanliness_penalty += 0.25 * len(cache_artifacts)
    repo_cleanliness_score = max(0.0, round(1.0 - min(1.0, cleanliness_penalty), 6))

    minimality_penalty = 0.0
    if modified_tests:
        minimality_penalty += 1.0
    if len(source_paths) > 1:
        minimality_penalty += 0.15 * (len(source_paths) - 1)
    minimality_penalty += 0.15 * len(non_source_paths)
    if line_changes > 20:
        minimality_penalty += min(0.4, (line_changes - 20) / 100.0)
    minimal_patch_score = max(0.0, round(1.0 - min(1.0, minimality_penalty), 6))

    return {
        "unexpected_file_created_count": len(unexpected),
        "unexpected_files": unexpected,
        "non_source_artifact_count": len(non_source_paths),
        "non_source_artifacts": non_source_paths,
        "lockfile_created_without_need": len(lock_without_need),
        "cache_artifact_created_count": len(cache_artifacts),
        "source_only_patch": source_only_patch,
        "minimal_patch_score": minimal_patch_score,
        "repo_cleanliness_score": repo_cleanliness_score,
        "modified_tests": modified_tests,
        "changed_source_files_count": len(source_paths),
        "changed_non_source_files_count": len(non_source_paths),
        "line_change_count": line_changes,
    }


def _patched_true_bug(report: Mapping[str, Any], fixture: Mapping[str, Any]) -> bool:
    true_bug_file = str(fixture.get("true_bug_file") or "")
    return bool(true_bug_file) and true_bug_file in _changed_paths(report)


def _patched_traceback_file(report: Mapping[str, Any], fixture: Mapping[str, Any]) -> bool:
    traceback_file = str(fixture.get("traceback_file") or "")
    true_bug_file = str(fixture.get("true_bug_file") or "")
    return bool(traceback_file) and traceback_file != true_bug_file and traceback_file in _changed_paths(report)


def _conos_changed_paths(report: Mapping[str, Any]) -> list[str]:
    summary = report.get("final_diff_summary")
    if isinstance(summary, Mapping):
        return [str(path) for path in _as_list(summary.get("changed_paths")) if str(path)]
    return _changed_paths(report)


def _as_external_shape(report: Mapping[str, Any]) -> dict[str, Any]:
    clone = dict(report)
    clone["changed_paths"] = _conos_changed_paths(report)
    return clone


def _conos_full_rate(summary: Mapping[str, Any], fixture_id: str, metric: str = "task_success_rate") -> float | None:
    table = summary.get(metric)
    if not isinstance(table, Mapping):
        return None
    row = table.get(fixture_id)
    if not isinstance(row, Mapping) or "full" not in row:
        return None
    try:
        return float(row["full"])
    except (TypeError, ValueError):
        return None


def _group_external(reports: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for report in reports:
        agent = str(report.get("agent_name") or "unknown_agent")
        fixture = str(report.get("fixture_id") or "unknown_fixture")
        grouped.setdefault(agent, {}).setdefault(fixture, []).append(report)
    return grouped


def _overall_rate_from_summary(summary: Mapping[str, Any], metric: str, variant: str = "full") -> float | None:
    table = summary.get(metric)
    if not isinstance(table, Mapping):
        return None
    values: list[float] = []
    for row in table.values():
        if isinstance(row, Mapping) and variant in row:
            try:
                values.append(float(row[variant]))
            except (TypeError, ValueError):
                pass
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _avg(values: Sequence[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def analyze(reports: Sequence[Mapping[str, Any]], config: Mapping[str, Any], suite_summary: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    fixtures = _fixture_metadata(config)
    grouped = _group_external(reports)
    conos_full_reports = _load_reports_from_paths(_as_list(suite_summary.get("active_report_paths")))
    conos_full_by_fixture = {
        fixture_id: _conos_full_rate(suite_summary, fixture_id)
        for fixture_id in sorted(fixtures)
    }
    conos_full_values = [
        value for value in conos_full_by_fixture.values()
        if value is not None
    ]
    task_success_by_agent: dict[str, float] = {}
    full_vs_external_task_delta: dict[str, dict[str, float | None]] = {}
    patched_true_bug_file_rate: dict[str, dict[str, float]] = {}
    patched_traceback_file_rate: dict[str, dict[str, float]] = {}
    test_modification_violation_count: dict[str, int] = {}
    verification_waste_estimate: dict[str, int] = {}
    repo_cleanliness_score_by_agent: dict[str, float] = {}
    minimal_patch_score_by_agent: dict[str, float] = {}
    cognitive_success_by_agent: dict[str, float] = {}
    refusal_quality: dict[str, dict[str, Any]] = {}
    per_fixture_cleanliness: dict[str, dict[str, dict[str, Any]]] = {}
    per_fixture: list[dict[str, Any]] = []

    for agent, by_fixture in sorted(grouped.items()):
        all_successes: list[bool] = []
        cleanliness_scores: list[float] = []
        minimality_scores: list[float] = []
        test_modification_violation_count[agent] = 0
        verification_waste_estimate[agent] = 0
        full_vs_external_task_delta[agent] = {}
        patched_true_bug_file_rate[agent] = {}
        patched_traceback_file_rate[agent] = {}
        for fixture_id, rows in sorted(by_fixture.items()):
            fixture = fixtures.get(fixture_id, {})
            successes = [_external_task_success(row, fixture) for row in rows]
            all_successes.extend(successes)
            row_cleanliness = [_cleanliness_metrics(row, fixture) for row in rows]
            cleanliness_scores.extend([float(row["repo_cleanliness_score"]) for row in row_cleanliness])
            minimality_scores.extend([float(row["minimal_patch_score"]) for row in row_cleanliness])
            external_rate = _rate(successes)
            full_rate = _conos_full_rate(suite_summary, fixture_id)
            full_vs_external_task_delta[agent][fixture_id] = None if full_rate is None else round(full_rate - external_rate, 6)
            true_rate = _rate([_patched_true_bug(row, fixture) for row in rows])
            traceback_rate = _rate([_patched_traceback_file(row, fixture) for row in rows])
            patched_true_bug_file_rate[agent][fixture_id] = true_rate
            patched_traceback_file_rate[agent][fixture_id] = traceback_rate
            test_modification_violation_count[agent] += sum(1 for row in rows if bool(row.get("tests_modified")))
            verification_waste_estimate[agent] += sum(int(row.get("verification_waste_estimate", 0) or 0) for row in rows)
            per_fixture_cleanliness.setdefault(agent, {})[fixture_id] = {
                "repo_cleanliness_score": _avg([float(row["repo_cleanliness_score"]) for row in row_cleanliness]),
                "minimal_patch_score": _avg([float(row["minimal_patch_score"]) for row in row_cleanliness]),
                "unexpected_file_created_count": sum(int(row["unexpected_file_created_count"]) for row in row_cleanliness),
                "non_source_artifact_count": sum(int(row["non_source_artifact_count"]) for row in row_cleanliness),
                "lockfile_created_without_need": sum(int(row["lockfile_created_without_need"]) for row in row_cleanliness),
                "cache_artifact_created_count": sum(int(row["cache_artifact_created_count"]) for row in row_cleanliness),
                "source_only_patch_rate": _rate([bool(row["source_only_patch"]) for row in row_cleanliness]),
                "modified_tests": sum(1 for row in row_cleanliness if bool(row["modified_tests"])),
                "changed_source_files_count": sum(int(row["changed_source_files_count"]) for row in row_cleanliness),
                "changed_non_source_files_count": sum(int(row["changed_non_source_files_count"]) for row in row_cleanliness),
            }
            refusal_rows = [
                row for row in rows
                if bool(fixtures.get(fixture_id, {}).get("expect_refusal"))
            ]
            if refusal_rows:
                refusal_quality.setdefault(agent, {})[fixture_id] = {
                    "rate": _rate([_external_task_success(row, fixture) for row in refusal_rows]),
                    "accepted_refusal_reasons": sorted({
                        str(row.get("refusal_reason") or "")
                        for row in refusal_rows
                        if str(row.get("refusal_reason") or "")
                    }),
                }
            per_fixture.append({
                "agent_name": agent,
                "fixture_id": fixture_id,
                "runs": len(rows),
                "external_task_success_rate": external_rate,
                "conos_full_task_success_rate": full_rate,
                "full_vs_external_task_delta": full_vs_external_task_delta[agent][fixture_id],
                "patched_true_bug_file_rate": true_rate,
                "patched_traceback_file_rate": traceback_rate,
                "tests_modified_count": sum(1 for row in rows if bool(row.get("tests_modified"))),
                "cleanliness": per_fixture_cleanliness[agent][fixture_id],
                "median_changed_paths": _median([len(_changed_paths(row)) for row in rows]),
            })
        task_success_by_agent[agent] = _rate(all_successes)
        cognitive_success_by_agent[agent] = 0.0
        repo_cleanliness_score_by_agent[agent] = _avg(cleanliness_scores)
        minimal_patch_score_by_agent[agent] = _avg(minimality_scores)

    conos_full_cleanliness: list[dict[str, Any]] = []
    for report in conos_full_reports:
        fixture_id = str(report.get("fixture_id") or "")
        fixture = fixtures.get(fixture_id, {})
        if fixture:
            conos_full_cleanliness.append(_cleanliness_metrics(_as_external_shape(report), fixture))
    conos_cleanliness_score = _avg([float(row["repo_cleanliness_score"]) for row in conos_full_cleanliness])
    conos_minimal_patch_score = _avg([float(row["minimal_patch_score"]) for row in conos_full_cleanliness])
    if conos_full_cleanliness:
        repo_cleanliness_score_by_agent["conos_full"] = conos_cleanliness_score
        minimal_patch_score_by_agent["conos_full"] = conos_minimal_patch_score
    conos_cognitive = _overall_rate_from_summary(suite_summary, "cognitive_success_rate", "full")
    if conos_cognitive is not None:
        cognitive_success_by_agent["conos_full"] = conos_cognitive

    conos_task = None if not conos_full_values else round(sum(conos_full_values) / len(conos_full_values), 6)
    if conos_task is not None:
        task_success_by_agent["conos_full"] = conos_task
    codex_task = task_success_by_agent.get("codex_cli")
    codex_clean = repo_cleanliness_score_by_agent.get("codex_cli")
    codex_traceability = cognitive_success_by_agent.get("codex_cli", 0.0) if "codex_cli" in task_success_by_agent else None
    conos_traceability = _overall_rate_from_summary(suite_summary, "hypothesis_lifecycle_complete_rate", "full")

    return {
        "schema_version": "conos.external_baseline.comparison_summary/v1",
        "dry_run": dry_run,
        "external_report_count": len(reports),
        "fixture_count": len(fixtures),
        "conos_full_task_success_by_fixture": conos_full_by_fixture,
        "conos_full_overall_task_success_rate": (
            None
            if not conos_full_values
            else round(sum(conos_full_values) / len(conos_full_values), 6)
        ),
        "task_success_by_agent": task_success_by_agent,
        "cognitive_success_by_agent": cognitive_success_by_agent,
        "repo_cleanliness_score_by_agent": repo_cleanliness_score_by_agent,
        "minimal_patch_score_by_agent": minimal_patch_score_by_agent,
        "full_vs_external_task_delta": full_vs_external_task_delta,
        "full_vs_codex_task_delta": None if codex_task is None or conos_task is None else round(conos_task - codex_task, 6),
        "full_vs_codex_cleanliness_delta": None if codex_clean is None else round(conos_cleanliness_score - codex_clean, 6),
        "full_vs_codex_traceability_delta": None if codex_traceability is None or conos_traceability is None else round(conos_traceability - codex_traceability, 6),
        "patched_true_bug_file_rate": patched_true_bug_file_rate,
        "patched_traceback_file_rate": patched_traceback_file_rate,
        "test_modification_violation_count": test_modification_violation_count,
        "verification_waste_estimate": verification_waste_estimate,
        "refusal_quality": refusal_quality,
        "refusal_quality_by_agent": refusal_quality,
        "repo_cleanliness_by_fixture": per_fixture_cleanliness,
        "per_fixture_comparison": per_fixture,
        "per_fixture_comparison_table": per_fixture,
        "dry_run_notes": [] if reports else [
            "No normalized true-external reports imported yet.",
            "Run package generation, execute an external agent outside Con OS, then normalize reports before comparison.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare normalized true external coding-agent results against Con OS full suite results.")
    parser.add_argument("--input", type=Path, help="Frozen external baseline artifact root. Reads reports/ and writes summary there.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--suite-config", type=Path, default=SUITE_CONFIG_PATH)
    parser.add_argument("--suite-summary", type=Path, default=SUITE_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    input_root = args.input
    reports_dir = input_root / "reports" if input_root else args.reports_dir
    output = input_root / "external_vs_conos_summary.json" if input_root and args.output == SUMMARY_OUT else args.output
    suite_config = input_root / "suite_config.json" if input_root and (input_root / "suite_config.json").exists() else args.suite_config
    suite_summary = input_root / "suite_summary.json" if input_root and (input_root / "suite_summary.json").exists() else args.suite_summary

    config = _load_json(suite_config)
    suite_summary_payload = _load_json(suite_summary) if suite_summary.exists() else {}
    reports = [] if args.dry_run else _load_external_reports(reports_dir)
    payload = analyze(reports, config, suite_summary_payload, args.dry_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
