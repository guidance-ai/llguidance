# Completion-row scan experiment (#103)

This offline experiment measures the existing main completion inference loop. It
is research instrumentation, not an indexed completion implementation or a
speedup claim. It needs no tokenizer downloads or model weights, and adds no
third-party dependencies. See [issue #103](https://github.com/guidance-ai/llguidance/issues/103).

## Reproduce

From the repository root, using the installed Rust toolchain and Python 3.9+:

```sh
cargo build --locked --release -j 2 -p completion_bench --features completion-stats
python3 completion_bench/scripts/measure.py target/release/completion_bench \
  --output /tmp/completion-counted-UNIQUE.jsonl
cargo test --locked --release -j 2 -p completion_bench --features completion-stats

# Same algorithm, with instrumentation compiled out; run before rebuilding again.
cargo build --locked --release -j 2 -p completion_bench
python3 completion_bench/scripts/measure.py target/release/completion_bench \
  --output /tmp/completion-uncounted-UNIQUE.jsonl
cargo test --locked --release -j 2 -p completion_bench
python3 completion_bench/scripts/compare.py \
  /tmp/completion-counted-UNIQUE.jsonl /tmp/completion-uncounted-UNIQUE.jsonl
```

Add `--offline` to Cargo commands when dependencies are cached. Set
`CARGO_TARGET_DIR` to isolate build artifacts, and pass that binary path to the
runner. Output files must not exist; the runner refuses to overwrite evidence.
The Rust executable rejects debug builds. Each workload runs in its own child
process; the Python runner uses `wait4` to record that exact child's peak resident
set size (bytes on macOS; KiB converted to bytes on Linux). This is actual whole
process peak RSS, including grammar compilation, the tokenizer, warm-up, lexer caches,
parser clones and output. It is neither incremental parser memory nor allocator
usage. Other platforms need an equivalent external process memory measurement.

The executable accepts `--list` or `CASE ITERATIONS SAMPLES`. Defaults in the
runner are one untimed warm-up and five samples of three replays. All raw samples
are saved, with process wall time, template time, clone time, replay time and
parser statistics. No outlier removal or automatic speedup inference occurs.

## Workloads and controls

| Case | Purpose | Input |
| --- | --- | --- |
| `expression` | Small arithmetic CFG with precedence and recursion | `(12+3)*4+5*(6+7)` |
| `blog_schema` | Existing repository JSON Schema and sample | `sample_parser/data/blog.*` |
| `wide_32`, `wide_256`, `wide_1024` | Generated large CFGs with a shared atom and distinct recursive alternatives | `((atail0;))` |
| `ambiguous_16`, `ambiguous_64`, `ambiguous_128` | Ambiguous `expr: expr expr \| "a"` | N `a` bytes followed by `!` |

Large source grammar size need not imply large Earley rows; use the measured row
counts. The widest case explicitly raises its factory's row limit to 16,384 and
step item budget to 2,000,000. Other cases retain the defaults of 2,000 and
50,000. No production defaults change. Other limits stay at their defaults and
errors fail the run.

A canonical single-byte tokenizer isolates grammar work from vocabulary size.
Slicing, fast-forwarding, backtracking and cancellation are disabled. The harness
uses current `ParserFactory` / `TokenParser` APIs and checks every input token
against its computed mask before consumption, followed by EOS and full
acceptance. No synthetic logits or sampling are involved.

Template construction happens once per workload and is timed separately; it
includes the byte tokenizer, factory setup and grammar compilation. One complete
untimed replay warms the shared lexer; each measured iteration clones the
same unstarted template outside the replay timer. Replay time includes parser
initialization, mask computation, token consumption, checks and a checksum of
all masks. The checksum is intentionally extra harness work; these numbers are
not isolated completion-loop times or inference throughput. Clone time and
process wall time are also reported. Lexer/cache state can differ from a cold
request; this is a controlled warm replay experiment.

The checksum covers every mask at every input prefix (including EOS), and must
be stable within a workload. Compare it and acceptance across counted and
uncounted runs. Equality is a regression tripwire on this corpus, not a proof of
language equivalence. Unit tests also use hand-written accepted/rejected inputs
for a smaller generated grammar. They exercise counter relationships, saved
stats compatibility, deltas and maxima.

## Counters

`llguidance/completion-stats` is opt-in. Default library builds contain no counter
updates or extra stats fields. With the feature, `ParserStats.completion` has:

- `searches`: non-nullable completion searches started;
- `candidates`: origin-row items actually visited, including nonmatches;
- `matches`: matching candidates before item deduplication;
- `max_origin_row_items`: largest origin row targeted by such a search;
- `max_row_items`: largest row passed to `just_push_row`.

These include initial, definitive, speculative and repeated work. Cached rows
that bypass the loop do not contribute. Nullable completions use a different
path and are excluded. Cancellation can leave a partially visited search.
`delta()` subtracts cumulative counters with saturation; row maxima remain
lifetime high-water marks, since subtraction cannot recover an interval maximum.
`max()` takes fieldwise maxima. Deserializing older stats without `completion`
uses zero counters. Initial template work is included in each replay's saved
stats even when it preceded the replay timer. The existing `lexer_cost` field
is the shared lexer's lifetime fuel total, including warm-up and earlier clones;
it is not isolated per-replay lexer work.

Candidate/match ratios quantify how much search work fails the symbol test.
They do not establish that an index is profitable: index creation, mutation,
nullable symbols, captures, speculative rollback, cancellation and memory costs
would need a controlled implementation A/B. Preserve mask/acceptance controls
and measure representative workloads as well as stress workloads before
proposing any algorithm change.

## Recorded baseline

A sample run on macOS 26.6.2 / arm64 used Rust 1.98.1, release mode,
`CARGO_INCREMENTAL=0`, `CARGO_PROFILE_RELEASE_DEBUG=0`, and two Cargo build jobs.
The source base was `6cef6ed9c973c2a155a85f2bb1996320af28b289` plus this patch.
Each row below summarizes seven samples of 100 warm replays. Full batched raw
elapsed nanoseconds, clone/template/process times, RSS bytes, checksums and
completion counters for both instrumentation settings are in
[the CSV](results/2026-10-23-macos-arm64.csv). Repeated process-level columns
refer to one child per workload/setting, not seven separate processes.

Both settings accepted every replay and had matching full-mask checksums on
all eight workloads. Timings below are for the counted build. The counters are
from the last replay, including template work as described above.

| Case | Median replay (ms) | Peak process RSS (bytes) | Largest origin row | Searches | Candidates | Matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `expression` | 0.051471 | 4,718,592 | 7 | 153 | 957 | 225 |
| `blog_schema` | 0.391968 | 4,898,816 | 23 | 147 | 1,895 | 151 |
| `wide_32` | 0.027968 | 5,128,192 | 97 | 5 | 483 | 4 |
| `wide_256` | 0.455849 | 8,683,520 | 769 | 5 | 3,843 | 4 |
| `wide_1024` | 6.001160 | 17,252,352 | 3,073 | 5 | 15,363 | 4 |
| `ambiguous_16` | 0.049490 | 4,603,904 | 35 | 473 | 6,043 | 2,800 |
| `ambiguous_64` | 1.657429 | 4,734,976 | 131 | 6,497 | 289,891 | 141,760 |
| `ambiguous_128` | 19.459896 | 4,964,352 | 259 | 25,281 | 2,205,891 | 1,090,432 |

These are measurements from a shared host with unrandomized run order, not
a latency guarantee or an optimization A/B. No owned builds ran during the
final measurements. Host scheduling/cache effects and instrumentation overhead
can affect timings, so the counted/uncounted timing differences do not
establish an algorithmic benefit. No completion index was implemented or
measured. Large-row and ambiguous cases establish reproducible scan workloads;
a useful indexing proposal still needs its own controlled time/memory A/B.
