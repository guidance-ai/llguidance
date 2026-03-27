# llg_test_utils

Shared test infrastructure for the [llguidance](../parser/) workspace.
**Not** intended as an example of how to use llguidance in production — see
[sample_parser](../sample_parser/) for that.

## What it provides

* A lazily-initialised `ParserFactory` using the **Phi-3.5-mini-instruct**
  tokenizer (capped at 35 000 tokens), with `ff_tokens` and `backtrack`
  enabled.
* `get_tok_env()` / `get_parser_factory()` — accessors for the above.
* **Acceptance helpers** (`lark_str_test`, `json_schema_check`, …) — tokenize
  an input, feed tokens one-by-one, and assert accept/reject.
* **Trace-replay helpers** (`check_lark_grammar`, `check_lark_grammar_nested`,
  …) — step a parser through a recorded token trace and verify every
  intermediate mask and forced-token result.

## Consumers

| Crate | Dependency kind |
|-------|----------------|
| `llguidance` (parser) | `[dev-dependencies]` |
| `json_schema_test_suite` | `[dependencies]` |

This crate is **workspace-only** (`publish = false`) and is not included in
`default-members`.
