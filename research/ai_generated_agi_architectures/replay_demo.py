"""A finite synthetic counterexample. Run with Python 3; no packages or network."""
from collections import Counter
from itertools import product
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory


def setup(db, initial_room, destination, newest_last, versioned):
    other_room = "B" if initial_room == "A" else "A"
    observations = [(1, other_room), (2, initial_room)]
    if not newest_last:
        observations.reverse()
    rev, projected_room = max(observations) if versioned else observations[-1]
    db.executescript("""
        CREATE TABLE world (room TEXT, revision INTEGER);
        CREATE TABLE intent (key TEXT PRIMARY KEY, source TEXT, rev INTEGER, dest TEXT);
        CREATE TABLE receipt (key TEXT PRIMARY KEY, outcome TEXT);
        CREATE TABLE dispatch_log (outcome TEXT);
    """)
    db.execute("INSERT INTO world VALUES (?, 2)", (initial_room,))
    if projected_room != destination:
        db.execute("INSERT INTO intent VALUES ('order-1/crate-1/move', ?, ?, ?)",
                   (projected_room, rev, destination))
    db.commit()


def dispatch(db, intent, deduplicate):
    key, source, rev, dest = intent
    with db:
        if deduplicate:
            cached = db.execute("SELECT outcome FROM receipt WHERE key = ?", (key,)).fetchone()
            if cached:
                return cached[0]
        changed = db.execute("UPDATE world SET room = ?, revision = revision + 1 "
                             "WHERE room = ? AND revision = ?", (dest, source, rev)).rowcount
        outcome = "moved" if changed == 1 else "rejected"
        db.execute("INSERT INTO dispatch_log VALUES (?)", (outcome,))
        if deduplicate:
            db.execute("INSERT INTO receipt VALUES (?, ?)", (key, outcome))
        return outcome


def run(initial_room, destination, newest_last, crash, versioned, deduplicate):
    with TemporaryDirectory() as folder:
        database = Path(folder) / "fixture.sqlite"
        db = sqlite3.connect(database)
        try:
            setup(db, initial_room, destination, newest_last, versioned)
            read_intent = "SELECT key, source, rev, dest FROM intent"
            intent = db.execute(read_intent).fetchone()
            if crash == "after dispatch before ack" and intent:
                dispatch(db, intent, deduplicate)
            if crash != "uninterrupted":
                # Lose the live connection and its plan. Only committed tables
                # survive this simulated restart at the named boundary.
                db.close()
                db = sqlite3.connect(database)
                intent = db.execute(read_intent).fetchone()
            if intent:
                outcome = dispatch(db, intent, deduplicate)
                if outcome == "moved":
                    with db:
                        db.execute("DELETE FROM intent WHERE key = ?", (intent[0],))
            room, = db.execute("SELECT room FROM world").fetchone()
            calls = [row[0] for row in db.execute("SELECT outcome FROM dispatch_log")]
            return (room == destination and "rejected" not in calls
                    and len(calls) == int(initial_room != destination))
        finally:
            db.close()


def main():
    scenarios = list(product("AB", "AB", [True, False],
                             ["uninterrupted", "after intent before dispatch",
                              "after dispatch before ack"]))
    passed = Counter()
    for case in scenarios:
        for flags in product([False, True], repeat=2):
            passed[flags] += run(*case, *flags)
    print("Version order | Keyed adapter | Passed / cases")
    for flags in product([False, True], repeat=2):
        print(f"{flags[0]!s:13} | {flags[1]!s:13} | {passed[flags]} / {len(scenarios)}")
    assert passed[(True, True)] == len(scenarios)


if __name__ == "__main__":
    main()
