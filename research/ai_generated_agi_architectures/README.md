# What should an agent remember after a warehouse move?

Eleven AI systems received one warehouse problem on 25 November 2026. The task asked for a delivery plan, recovery after a restart and a way to handle a stale crate-location observation. Each system proposed an architecture. For the synthesis we chose a specific point of failure: the simulated crate has moved, but the planner restarts before it saves the acknowledgement.

Storing a plan helps the agent resume its work. It leaves two practical questions open. Which observation describes the crate now? What can the agent establish about a move whose acknowledgement was lost? In a small SQLite example, sorting observations by their source revision and having the adapter recognize a retried move addressed different failures. The combined implementation passed 24 of 24 enumerated cases. The arrival-order implementation without keyed receipts passed 10.

| To inspect | File |
| --- | --- |
| Original final answers | [raw_outputs/](raw_outputs/) |
| Exact common prompt, limits and retries | [prompts.md](prompts.md) |
| Model attribution, interfaces and source conditions | [sources.md](sources.md) |
| Available receipt identifiers and one SHA-256 digest per answer | [provenance.csv](provenance.csv) |
| Eleven requested comparison dimensions for all eleven records | [comparison.csv](comparison.csv) |
| Agreements, disagreements and limits | [summary.md](summary.md) |
| Proposed implementation, fixture design and results | [synthesis.md](synthesis.md) · [replay_demo.py](replay_demo.py) |

The collected systems share some model ancestors and service infrastructure. Eight final texts ended normally. Three have a cut-off ending and are marked in the provenance file. Missing details remain visible in the comparison. The executable example tests one engineering decision in a finite simulated setting. It does not measure general intelligence or rank the systems.
