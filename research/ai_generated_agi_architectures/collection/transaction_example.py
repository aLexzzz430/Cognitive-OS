"""Test one crash-boundary claim in a collected design, not Cognitive-OS.

Uses only temporary SQLite databases and local files. Child processes exit
abruptly after a filesystem effect but before the completion receipt commits.
This models process termination, not power loss or arbitrary external tools.
"""
from pathlib import Path
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile

def worker(root, durable_intent):
    db = sqlite3.connect(root / 'state.db')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('BEGIN IMMEDIATE')
    db.execute("INSERT INTO actions VALUES ('a1', 'PENDING')")
    if durable_intent:
        db.commit()
    with (root / 'effect.txt').open('a') as f:
        f.write('performed a1\n')
        f.flush()
        os.fsync(f.fileno())
    # No Python shutdown cleanup; the pending transaction remains uncommitted.
    os._exit(17)

def run_case(root, durable_intent):
    root.mkdir()
    with sqlite3.connect(root / 'state.db') as db:
        db.execute('CREATE TABLE actions (action_id TEXT PRIMARY KEY, status TEXT)')
    result = subprocess.run([sys.executable, __file__, '--worker', str(root), '--mode', 'durable' if durable_intent else 'single-transaction'], check=False)
    assert result.returncode == 17
    with sqlite3.connect(root / 'state.db') as db:
        actions = db.execute('SELECT action_id, status FROM actions').fetchall()
    effects = (root / 'effect.txt').read_text().splitlines()
    assert effects == ['performed a1']
    if durable_intent:
        assert actions == [('a1', 'PENDING')]
        # The action-specific read probe reconciles the effect; no second append.
        with sqlite3.connect(root / 'state.db') as db:
            db.execute("UPDATE actions SET status='VERIFIED_BY_PROBE' WHERE action_id='a1'")
        assert (root / 'effect.txt').read_text().splitlines() == effects
    else:
        assert actions == []
    return {'mode': 'durable-intent-before-effect' if durable_intent else 'intent-and-receipt-in-one-transaction', 'child_exit_code': result.returncode, 'pending_actions_found_after_exit': actions, 'filesystem_effects_after_exit': effects, 'reconciled_without_repeating_effect': durable_intent}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--mode', choices=['durable', 'single-transaction'])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.mode == 'durable')
    with tempfile.TemporaryDirectory(prefix='cognitive-design-example-') as directory:
        root = Path(directory)
        results = [run_case(root / 'one-transaction', False), run_case(root / 'durable-intent', True)]
    print(json.dumps({'sqlite_version': sqlite3.sqlite_version, 'python_version': sys.version.split()[0], 'cases': results, 'scope': 'Two isolated process-exit examples of a proposed transaction ordering. Not an upstream bug report, a power-loss test, or a general exactly-once guarantee.'}, indent=2))

if __name__ == '__main__':
    main()
