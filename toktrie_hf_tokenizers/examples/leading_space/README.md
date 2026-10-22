# Leading-space tokenizer and constraint experiment

This offline experiment supplies reproducible observations for
[llguidance #277](https://github.com/guidance-ai/llguidance/issues/277).
It measures tokenizer preprocessing, vocabulary bytes, and grammar validity.
It does not run a language model, estimate probability mass, or establish
whether a grammar harms generation quality. No tokenizer defaults are changed.

## Run

From the repository root, using the repository's Rust toolchain:

```sh
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=target/leading-space-research \
  cargo test --locked --release -p toktrie_hf_tokenizers --test leading_space
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=target/leading-space-research \
  cargo run --locked --release -p toktrie_hf_tokenizers --example leading_space \
  > leading-space-fixtures.json
```

The default run uses only checked-in fixtures. Once dependencies are cached,
add `--offline` to the Cargo commands. To inspect an existing local production
tokenizer, pass its `tokenizer.json` path; this code never downloads data:

```sh
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=target/leading-space-research \
  cargo run --locked --release -p toktrie_hf_tokenizers --example leading_space \
  -- --tokenizer /path/to/tokenizer.json > leading-space-local.json
```

Multiple local paths are supported. Each tokenizer must be supported by the
existing `ByteTokenizer` adapter. Use a pinned public model revision and record
the tokenizer file's SHA-256 when sharing a production-tokenizer report.
Model weights and a model runtime are unnecessary for this experiment.

## Controls and fixtures

The seven standalone Hugging Face `tokenizer.json` files share a small BPE
vocabulary: ordinary words/fragments, space-prefixed words, a space token, and
EOS. They are constructed test fixtures, not production model vocabularies.
Their 22 token IDs map to identical llguidance bytes; the spelling of the
space marker differs between ByteLevel (`Ġ`) and Metaspace (`▁`).

| Pair/group | Preprocessing setting varied |
| --- | --- |
| `bytelevel-false` / `bytelevel-true` | ByteLevel `add_prefix_space` |
| `metaspace-never` / `metaspace-first` / `metaspace-always` | Metaspace `prepend_scheme` |
| `prepend-none` / `prepend-space` | A `Prepend` step before a space-to-`▁` Replace normalizer, without a pre-tokenizer |

Within each group, only the indicated setting changes. The Metaspace and
Prepend fixtures use explicit decoder sequences with ByteFallback, Replace,
Fuse, and Strip steps: they replace `▁` with space and strip one initial space. This
deliberately exposes the difference between whole-sequence HF decoding and
the literal bytes associated with a vocabulary token.

The paired Lark grammars differ only by an optional single ASCII space at
the start. Both otherwise accept one or more ASCII word characters, optional
subsequent words separated by one ASCII space, and an optional final `!`.
The strict grammar is a subset of the word-leading language discussed in
the issue. There is no ignored whitespace rule. Samples include a word,
a phrase, a literal initial space, two initial spaces, punctuation, digits,
an underscore, and an empty string.

For each tokenizer and input the JSON records three paths separately:

1. Original HF `encode(input, add_special_tokens=false)` and HF decoding.
2. HF encoding/decoding after the existing `ByteTokenizer` adaptations.
3. Actual `TokenizerEnv::tokenize(input)` output and its concatenated trie bytes.

Source and adapted HF paths include token spellings and grammar-visible bytes.
The environment path includes a `preserves_input_bytes` flag; acceptance is
always evaluated on the emitted token bytes, so preprocessing cannot silently
change the string being tested. The report also includes original/adapted
normalizers, pre-tokenizers, decoders, and post-processors. Automatic special
token insertion is disabled for every input.

The grammar comparison records initial mask cardinalities and newly allowed
token IDs/bytes (up to 64 examples, with the uncapped count). Each of the three
encoding paths is checked against both grammars with fresh parsers, using
`validate_tokens_raw`, per-token `compute_mask`, `consume_token`, and final
`is_accepting`. A disagreement between validation and masks fails the run.
A sequence can be a valid prefix without being an accepted complete output;
the report distinguishes those states.

The tests additionally check literal input tokenizations and every non-special
fixture token alone, after a space, and after `hello`, against an independent
byte-language oracle. They check identical masks for identical vocabulary bytes
across all preprocessing settings, and explicitly check EOS rejection at the
empty prefix.

## Observed results

The release-mode example was run on 2026-10-23 with Rust 1.98.1 and the locked
`tokenizers` 0.21.2 dependency, using production source at repository base
`6cef6ed9c973c2a155a85f2bb1996320af28b289`. The seven fixture runs and two
existing cached public tokenizer files completed successfully. No model data
was downloaded.
These are initial masks for exactly the two checked-in grammars:

| Tokenizer | Vocabulary entries | Strict grammar permits | Optional-space grammar permits | Newly permitted |
| --- | ---: | ---: | ---: | ---: |
| Each of the seven controlled fixtures | 22 | 17 | 20 | 3 |
| Phi-3.5-mini-instruct snapshot below | 32,011 | 10,283 | 24,151 | 13,868 |
| Meta-Llama-3.1-8B-Instruct snapshot below | 128,256 | 34,698 | 77,751 | 43,053 |

The three newly permitted fixture tokens contain `" "`, `" hello"`, and
`" world"`. Changing preprocessing while keeping token bytes fixed left these
mask counts unchanged in all seven fixtures. This is a count of vocabulary
entries, without any model weighting.

For the literal input `"hello"`, selected raw observations were:

| Configuration | Source HF IDs | Source HF decoded text | Source IDs as llguidance bytes | Environment IDs | Environment bytes |
| --- | --- | --- | --- | --- | --- |
| ByteLevel false; Metaspace never; Prepend absent | `[14]` | `"hello"` | `"hello"` | `[14]` | `"hello"` |
| ByteLevel true | `[19]` | `" hello"` | `" hello"` | `[19]` | `" hello"` |
| Metaspace first/always; Prepend present | `[19]` | `"hello"` | `" hello"` | `[14]` | `"hello"` |
| Phi snapshot | `[22172]` | `"hello"` | `" hello"` | `[12199]` | `"hello"` |
| Llama snapshot | `[15339]` | `"hello"` | `"hello"` | `[15339]` | `"hello"` |

In these rows, a sequence whose trie bytes are `" hello"` is rejected by the
strict grammar and accepted by the optional-space grammar. Both grammars
accept the sequences whose bytes are `"hello"`. HF decoded text alone would
hide this distinction for the Metaspace/Prepend fixtures and the Phi example.
The existing adapter removes Prepend normalization and disables Metaspace
prepending. It retains ByteLevel's `add_prefix_space=true` in that fixture;
the input-byte preservation flag records the resulting initial space.

Public tokenizer provenance (only these JSON files are needed to reproduce):

- [`microsoft/Phi-3.5-mini-instruct` tokenizer.json](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/2fe192450127e6a83f7441aef6e3ca586c338b77/tokenizer.json),
  revision `2fe192450127e6a83f7441aef6e3ca586c338b77`, 1,844,408 bytes,
  SHA-256 `9e2ae3d66819f163cdcfedba5078f3a3af8118c2712491ed60912f77886bda6f`.
  Its source normalizer is a sequence of Prepend `▁` and Replace space with
  `▁`, without a pre-tokenizer.
- [`unsloth/Meta-Llama-3.1-8B-Instruct` tokenizer.json](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/a2856192dd7c25b842431f39c179a6c2c2f627d1/tokenizer.json),
  revision `a2856192dd7c25b842431f39c179a6c2c2f627d1`, 17,209,920 bytes,
  SHA-256 `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`.
  Its source ByteLevel pre-tokenizer has `add_prefix_space=false`. Its
  decoder has `add_prefix_space=true`; that is a separate component.

## Limits and follow-up

An initial mask's number of permitted tokens is not model probability mass.
Tiny fixture token IDs have no learned probabilities, and the two production
tokenizers are observational examples rather than a controlled comparison of
their models. No logits, synthetic or otherwise, are used here.

Answering the model-distribution part of #277 requires inference with fixed
model weights/revision, prompt/chat template, tokenized prompt, generation
boundary, and sampling settings. A paired experiment should vary only the
grammar's initial-space policy, measure actual pre-mask model probability
mass removed at each step, and compare generation validity and task quality
over a specified prompt set and repeated seeds. A manually tokenized completion
in isolation does not establish how a model behaves after a particular prompt.

This experiment starts parsers without a prompt. It intentionally leaves
prompt-boundary effects, model quality, latency, and distribution effects open.
