# What happens if the crate moved just before a crash?

A planner issues a transfer from A to B. The simulated warehouse moves the crate. Before the planner can save the acknowledgement, the process dies. On restart the last sensor packet says A. Rebuilding the planner's own log cannot tell it whether the transfer happened. A second dispatch is safe only if the warehouse can recognize the original request, or if the outcome can be reconciled before any further move.

## Smallest implementation

Use a single writer and a simulated adapter. Assign each order an immutable ID and each logical move a stable key. Store observations with an entity, source, source revision, observed time and received time. Keep the observations so that a late delivery remains available for diagnosis. For one authoritative sensor stream, project the highest revision. For independent sources with incompatible revisions, retain both hypotheses and request a fresh observation. A large sequence number from another sensor gives no ordering guarantee.

Before dispatch, persist the intent and the expected source-room revision. The adapter checks the expected state, applies the move once, and records the key and outcome in the same transaction as its simulated effect. A second request with that key returns the recorded outcome. Save the returned receipt before completing the plan. For an adapter that cannot offer that contract, mark the move pending reconciliation and ask for inspection. Bind any human approval to the exact move and revision. A later contradictory observation invalidates that approval.

A correction adds a dated evidence event and triggers a local plan revision. Learn sensor reliability on separate synthetic episodes only after a fixed-rule baseline exists. Neither a new sensor reading nor a model-generated explanation changes the safety rule.

## Reproducible comparison

Run `python replay_demo.py` from this directory. The Python standard library suffices. There are 24 fixtures: two initial rooms, two destinations, two arrival orders for a stale and a current observation, and three execution histories. Each fixture uses a fresh SQLite file. In the ordinary run a live plan issues its move. The first restart closes the connection after the intent commits, reopens the database, and reloads that intent. The second restart does that after the adapter transaction commits and before the caller acknowledges it. A pass requires the correct final room, no rejected pickup and exactly as many dispatches reaching the precondition check as a correct delivery requires. A cached receipt skips that check. A rejected dispatch counts as a call and leaves the crate in place. If the plan has no move, several interruption labels lead to equivalent behavior.

| Observation used | Adapter behavior on retry | Fixtures passed |
| --- | --- | ---: |
| Last arrival | Repeat call | 10 / 24 |
| Last arrival | Reuse receipt | 12 / 24 |
| Highest source revision | Repeat call | 20 / 24 |
| Highest source revision | Reuse receipt | 24 / 24 |

These counts measure the behavior of the four short deterministic implementations in the included finite fixture set. They say nothing about the frequency of crashes, real warehouse reliability or model performance. The baseline already solves every in-order, uninterrupted fixture. The two changes address different counterexamples. A receipt cannot repair a decision to skip a needed move, and ordered evidence cannot tell the caller that an unacknowledged move already ran.

## Fit with Cognitive-OS

At revision `e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca`, the repository already has a plan reviser with recovery and invalidated-assumption triggers, a plan-state approval mechanism and a formal evidence ledger. A first integration would exercise those paths in an offline fixture and introduce the smallest adapter contract needed for keyed receipts. The fixture here is independent of the runtime. It closes and reopens the database at committed boundaries. The next acceptance test would terminate an actual subprocess on both sides of commit and inspect the recovered plan and the adapter transaction. The fixture has one sensor and no moves away and back between observations. Conflicting independent readings and a return to the same room at a later revision each need their own test.

[Plan revision](https://github.com/aLexzzz430/Cognitive-OS/blob/e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca/planner/plan_reviser.py) · [Approval tests](https://github.com/aLexzzz430/Cognitive-OS/blob/e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca/tests/test_plan_state_approval.py) · [Evidence ledger tests](https://github.com/aLexzzz430/Cognitive-OS/blob/e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca/tests/test_evidence_ledger.py)
