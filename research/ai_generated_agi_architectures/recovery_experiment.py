"""Isolated process-crash experiment; does not import or modify Cognitive-OS.

Run with --output PATH to preserve a JSON report. All effects use owned temporary
directories. This tests process death, not power-loss durability or hostile races.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

BEFORE = b"mode=local\n"
ADDITION = b"verified=true\n"
AFTER = BEFORE + ADDITION
CRASH_EXIT = 87
POINTS = ("before_intent", "after_intent", "after_stage", "after_replace", "after_done")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def durable_write(path, data):
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def stop(point, selected):
    if point == selected:
        os._exit(CRASH_EXIT)


def execute(root, strategy, crash):
    target = root / "config.txt"
    staged = root / "staged.txt"
    db = sqlite3.connect(root / "state.sqlite3")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("""CREATE TABLE IF NOT EXISTS actions (
        action_id TEXT PRIMARY KEY, before_hash TEXT NOT NULL,
        after_hash TEXT NOT NULL, approval_version INTEGER NOT NULL,
        status TEXT NOT NULL)""")
    db.commit()
    stop("before_intent", crash)
    db.execute("INSERT OR IGNORE INTO actions VALUES (?, ?, ?, ?, ?)",
               ("edit-1", digest(BEFORE), digest(AFTER), 1, "pending"))
    db.commit()
    stop("after_intent", crash)
    before_hash, after_hash, approval_version, status = db.execute(
        "SELECT before_hash, after_hash, approval_version, status FROM actions"
    ).fetchone()
    if status in ("done", "conflict", "denied"):
        db.close()
        return

    def finish(value):
        db.execute("UPDATE actions SET status=?", (value,))
        db.commit()

    current = target.read_bytes()
    if strategy == "reconcile":
        # Recording a completed effect requires no new file write.
        if digest(current) == after_hash:
            finish("done")
            db.close()
            return
        policy = json.loads((root / "policy.json").read_text())
        if not policy["allowed"] or policy["version"] != approval_version:
            finish("denied")
            db.close()
            return
        if digest(current) != before_hash:
            finish("conflict")
            db.close()
            return
        result = AFTER
    else:
        # Deliberately simple baseline: pending means replay the append.
        result = current + ADDITION
    durable_write(staged, result)
    stop("after_stage", crash)
    os.replace(staged, target)
    stop("after_replace", crash)
    finish("done")
    stop("after_done", crash)
    db.close()


def child(root, strategy, point="none"):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(root),
         "--strategy", strategy, "--crash", point],
        capture_output=True, text=True, timeout=30,
    )
    expected = 0 if point == "none" else CRASH_EXIT
    if result.returncode != expected:
        raise RuntimeError(f"Child exit {result.returncode}, expected {expected}: {result.stderr}")
    return result.returncode


def run_case(strategy, point, intervention="none"):
    with tempfile.TemporaryDirectory(prefix="cognitive-recovery-") as directory:
        root = Path(directory)
        durable_write(root / "config.txt", BEFORE)
        (root / "policy.json").write_text(json.dumps({"version": 1, "allowed": True}))
        exit_code = child(root, strategy, point)
        expected_content = AFTER
        expected_status = "done"
        if intervention == "external_edit":
            durable_write(root / "config.txt", b"user-changed=true\n")
            expected_content, expected_status = b"user-changed=true\n", "conflict"
        elif intervention == "revoke":
            (root / "policy.json").write_text(json.dumps({"version": 2, "allowed": False}))
            expected_content, expected_status = BEFORE, "denied"
        child(root, strategy)
        first = (root / "config.txt").read_bytes()
        child(root, strategy)
        final = (root / "config.txt").read_bytes()
        with closing(sqlite3.connect(root / "state.sqlite3")) as db:
            status = db.execute("SELECT status FROM actions").fetchone()[0]
        correct = final == expected_content and status == expected_status
        if strategy == "reconcile" and not correct:
            raise AssertionError((point, intervention, final, status))
        if strategy == "naive" and point == "after_replace" and intervention == "none":
            assert final == BEFORE + ADDITION + ADDITION, "Baseline duplicate not reproduced"
        return dict(strategy=strategy, crash_point=point, intervention=intervention,
                    injected_exit=exit_code, final_text=final.decode(), final_status=status,
                    invariant_passed=correct, second_recovery_unchanged=first == final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--strategy", choices=("naive", "reconcile"))
    parser.add_argument("--crash", default="none", choices=("none",) + POINTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        execute(args.worker, args.strategy, args.crash)
        return
    cases = [run_case(strategy, point) for strategy in ("naive", "reconcile") for point in POINTS]
    cases.extend(run_case(strategy, "after_intent", intervention)
                 for strategy in ("naive", "reconcile")
                 for intervention in ("external_edit", "revoke"))
    report = dict(scope="isolated process-crash design experiment, not Cognitive-OS runtime",
                  python=sys.version.split()[0], platform=sys.platform,
                  script_sha256=digest(Path(__file__).read_bytes()), cases=cases)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"Completed {len(cases)} cases; all 7 reconciliation cases satisfy their invariants.")


if __name__ == "__main__":
    main()
