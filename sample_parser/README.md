# Sample Parser for llguidance

This directory demonstrates how to use the [llguidance](../parser/README.md)
crate to implement **constrained decoding** (structured output) for Large Language Models.

It contains two binaries:
- **`minimal`** — a stripped-down example that shows the core API in ~80 lines
  (start here to understand the code)
- **`sample_parser`** — a full CLI tool with real tokenizers, multiple grammar
  formats, and several operating modes

## How constrained decoding works

During LLM inference, llguidance sits between the model and the output, ensuring
every generated token is grammatically valid. The core loop is:

```
┌─────────────────────────────────────────────────────┐
│  1. compute_mask()                                  │
│     → returns a bitset of tokens allowed by the     │
│       grammar at this position (~1ms, background)   │
│                                                     │
│  2. LLM samples a token                             │
│     → logits are masked so only valid tokens remain │
│                                                     │
│  3. consume_token(token)                            │
│     → tells the parser which token was chosen       │
│       (very fast, <100μs)                           │
│                                                     │
│  4. consume_ff_tokens()                             │
│     → returns "fast-forward" tokens forced by the   │
│       grammar (e.g. `{"name":"` in JSON); these     │
│       bypass sampling entirely                      │
│                                                     │
│  5. Repeat until is_stopped() returns true          │
└─────────────────────────────────────────────────────┘
```

The `minimal` binary validates a known-good input against a grammar.
The `sample_parser` binary can also *generate* random output that satisfies
the grammar, simulating an LLM.

## Building

```bash
# Build both binaries (release mode recommended for real tokenizers)
cargo build --release -p sample_parser
```

Requires Rust 1.87+. The first run with a HuggingFace tokenizer will download
model files (~500KB–2MB).

## Usage: `minimal`

The minimal binary uses a simple single-byte tokenizer (no downloads needed):

```bash
cargo run --bin minimal -- data/blog.schema.json data/blog.sample.json
```

This validates that `blog.sample.json` conforms to the JSON Schema in
`blog.schema.json`. Read `src/minimal.rs` for line-by-line commentary on
the constrained decoding loop.

## Usage: `sample_parser`

### Validate an input against a grammar

```bash
# JSON Schema + known-good JSON input
cargo run -- data/blog.schema.json --input data/blog.sample.json

# Lark grammar + known-good XML input
cargo run -- data/rfc.lark --input data/rfc.xml

# Verbose mode shows per-token stats (lexer cost, Earley items, timing)
cargo run -- data/blog.schema.json --input data/blog.sample.json --verbose
```

### Generate random output satisfying a grammar

This simulates an LLM by sampling random tokens from the allowed set:

```bash
# Generate up to 100 random tokens conforming to the JSON Schema
cargo run -- data/blog.schema.json --rnd 100

# Use a different tokenizer
cargo run -- data/blog.schema.json --rnd 100 --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

### Just compile the grammar (no generation)

Omit both `--input` and `--rnd` to compile the grammar and compute one mask:

```bash
cargo run -- data/blog.schema.json
```

This is useful for checking that a grammar is valid and measuring compilation time.

## Supported grammar formats

The grammar format is determined by the file extension:

| Extension | Format | Example |
|-----------|--------|---------|
| `.schema.json` | [JSON Schema](../docs/json_schema.md) | `data/blog.schema.json` |
| `.ll.json` | Internal llguidance format | `data/blog.schema.ll.json` |
| `.lark` | [Lark-like CFG](../docs/syntax.md) | `data/lark.lark` |
| `.txt` | Text → substring regex | (any text file) |

JSON Schema is the most common format for structured output use cases.
Lark grammars allow arbitrary context-free grammars.

## CLI options

| Option | Description |
|--------|-------------|
| `GRAMMAR` | Grammar file (required; format determined by extension) |
| `--input FILE` | Input file to validate against the grammar |
| `--rnd N` | Generate N random tokens satisfying the grammar |
| `--tokenizer NAME` | HuggingFace tokenizer (default: `microsoft/Phi-3.5-mini-instruct`) |
| `--verbose` | Print per-token sampling details and stats |
| `--seed N` | Random seed for `--rnd` mode (default: 1) |
| `--log-level N` | stderr log level: 1=warnings, 2=verbose (default: 1) |
| `--repeat N` | Repeat generation N times for profiling |
| `--lexer-limit N` | Multiply default lexer fuel limits by N |
| `--initial-lexer-fuel N` | Set initial lexer fuel (thousands; default: 1000) |
| `--step-lexer-fuel N` | Set per-step lexer fuel (thousands; default: 200) |
| `--split-words` | For `.txt` input, split on words instead of lines |
| `--dump-tokenizer` | Print tokenizer vocabulary (hex-encoded) and exit |

## Data files

| File | Description |
|------|-------------|
| `blog.schema.json` | A JSON Schema describing a blog post object |
| `blog.sample.json` | A sample blog post conforming to the schema |
| `blog.schema.ll.json` | The same schema in llguidance's internal format |
| `rfc.lark` | A Lark grammar describing a subset of RFC/XML syntax |
| `rfc.xml` | An XML document conforming to the Lark grammar |
| `lark.lark` | A Lark grammar that parses the Lark grammar format itself |
| `from-llama.cpp/` | Grammar files ported from llama.cpp's GBNF format |
| `constnewline.schema.json` | JSON schema which exposed [Issue 326](https://github.com/guidance-ai/llguidance/issues/326) |

## Key API types

| Type | Purpose |
|------|---------|
| `TopLevelGrammar` | Grammar specification (from JSON Schema, Lark, regex, etc.) |
| `ParserFactory` | Compiles grammars; holds tokenizer state. Create once, share via `Arc`. |
| `Matcher` | Server-side API wrapping the parser. Used in this sample. |
| `Constraint` | Higher-level API (used by the [Guidance](https://github.com/guidance-ai/guidance) Python library) |
| `SimpleVob` | Token mask — bitset of allowed token IDs |

## Further reading

- [parser/README.md](../parser/README.md) — API overview and integration guide
- [docs/fast_forward.md](../docs/fast_forward.md) — How fast-forward tokens work and why canonical tokenization matters
- [docs/syntax.md](../docs/syntax.md) — Lark grammar syntax reference
- [docs/json_schema.md](../docs/json_schema.md) — Supported JSON Schema features
- [docs/optimizations.md](../docs/optimizations.md) — Performance details and the slicer optimization
