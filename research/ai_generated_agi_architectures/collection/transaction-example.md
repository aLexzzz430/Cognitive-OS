# One executed transaction-ordering example

**Finding:** Gemini reference candidate A §8 cannot recover its pending action at the illustrated failure boundary. Its intent record is still uncommitted when a separate filesystem effect occurs. An abrupt process exit loses the pending record while leaving that effect observable.

This is an analyst-authored functional example of generated pseudocode. It does not run Cognitive-OS and is not an upstream defect report. The script uses only a temporary directory, SQLite databases and a text file; it makes no network calls. No other experiments proposed by the model were executed.

Run from the packet root:

```sh
python collection/transaction_example.py
```

The worker uses SQLite WAL and `synchronous=FULL`. In both cases it appends one line to a file, flushes it, calls `fsync`, and exits using `os._exit(17)` before writing any completion receipt. Exit code 17 is the intended interruption, not a failed test runner. The parent reopens the database and reads the file. The recorded run used Python 3.12.14 and SQLite 3.53.1.

| Case | Intent ordering | Pending actions after exit | File effects after exit | Reconciliation |
| --- | --- | --- | --- | --- |
| A ordering | Insert intent, perform effect, exit before SQL commit | None | One `performed a1` line | A pending-action scan cannot discover this action. |
| Analyst correction | Insert and commit intent, perform effect, exit before receipt | `a1`, `PENDING` | One `performed a1` line | The action-specific read probe observes the line and marks the record verified without another append. |

Both assertions passed. The original result is preserved in [transaction-example-results.json](transaction-example-results.json). The script also verifies that the corrected case still has only one line after reconciliation.

The inference is limited: durable intent preserves the identity needed for this recovery probe; it does not atomically couple arbitrary external effects to SQLite. The probe is deliberately simple because this action has a unique, observable effect. There is no competing writer, no ambiguous third-party response, and no general exactly-once guarantee. The test is abrupt process exit, not power loss, filesystem failure or exhaustive crash-boundary coverage. The generated A trace mentions both power loss and process kill; this experiment does not validate power-loss recovery.

SQLite documents atomicity of database changes within a transaction. WAL records a commit explicitly, with durability behavior depending on synchronization. The separate file append in this example is outside those database changes. These documentation points support the interpretation; the two observed cases are evidence from this script. Sources: [Atomic Commit in SQLite](https://sqlite.org/atomiccommit.html), [Write-Ahead Logging](https://sqlite.org/wal.html), accessed 2026-09-08.

For the proposed runtime design, commit a validated action intent before dispatch. After interruption, reconcile the effect using a task-specific probe, then commit the receipt and task transition together. Preserve `UNCERTAIN` when the probe cannot decide. Those steps remain a design proposal until evaluated against the actual runtime and its full set of effect adapters.
