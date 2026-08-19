# AI-generated AGI architecture research packet

Source opportunity: https://github.com/aLexzzz430/Cognitive-OS/issues/5

Exact panel run: `b2fcca6e-fad0-483a-8f2c-c5f559900965`

Raw outputs: 8; distinct system families: 8; providers: 7.

## Collection method

Eight genuine provider API calls were executed with one comparable deterministic prompt. Raw responses are preserved separately and unedited. Provider/model identity, access timestamps, call ids, output hashes, token usage and cost are recorded in `sources.md`. No credentials or private provider material are included.

## Comparison method

`comparison.csv` has one row per system and all eleven comparison dimensions requested by the bounty. Text is extracted from the closest explicit Markdown section when available. If no matching section is detected, the cell says so rather than inventing a model position; reviewers can inspect the preserved raw output.

## Headline findings

The proposals repeatedly separate persistent memory/state, planning, tool execution, world/state representation, governance and evaluation responsibilities. The combined architecture in `synthesis.md` turns those recurring responsibilities into versioned interfaces, an append-only event kernel, typed action receipts and falsifiable promotion gates. Disagreement between proposals is preserved as an ablation/testing queue rather than averaged away.

## Files

- `prompts.md` — exact comparable prompt and prompt hash
- `raw_outputs/` — one preserved raw response per model/system family
- `comparison.csv` — consistent matrix across all requested dimensions
- `summary.md` — patterns, coverage diagnostics and interpretation limits
- `synthesis.md` — concrete implementation-oriented combined architecture
- `sources.md` — attribution, access dates, call ids, hashes, usage and edit disclosure
- `acceptance_report.json` — local deterministic verification report
