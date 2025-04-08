### Changelog

All notable changes to this project will be documented in this file. Dates are displayed in UTC.

If a release doesn't introduce any interesting changes (build fixes etc.), it's skipped.

#### [0.7.13](https://github.com/guidance-ai/llguidance/compare/v0.7.12...0.7.13) 2025-04-05

- expose LLParserLimits in Python API [`598dc8f`](https://github.com/guidance-ai/llguidance/commit/598dc8f37f69f51244e54d9885445abf02a515a7)
- pre-compute lexer states for particularly large regexes (can be disabled in ParserLimits)

#### [0.7.12](https://github.com/guidance-ai/llguidance/compare/v0.7.11...0.7.12) 2025-04-04

- performance optimizations
- use factory in C FFI (otherwise slicer was not used)
- add some null checks and safety comments in C FFI
- implement subgrammar lexeme class merging; fixes [`#113`](https://github.com/guidance-ai/llguidance/issues/113)

#### [0.7.11](https://github.com/guidance-ai/llguidance/compare/v0.7.10...0.7.11) 2025-03-27

- add StructTag python API; fixes [`#146`](https://github.com/guidance-ai/llguidance/issues/146)
- fix handling of AddedToken.special (gemma tokenizer, fixes [`#147`](https://github.com/guidance-ai/llguidance/issues/147))
- handle incomplete tokenizers (SmolLM); fixes [`#138`](https://github.com/guidance-ai/llguidance/issues/138)
- fix `validate_token_raw([EOS])` bug

#### [v0.7.10](https://github.com/guidance-ai/llguidance/compare/v0.7.9...v0.7.10) 2025-03-25

- add `llg_matcher_*()` functions to C interface [`#145`](https://github.com/guidance-ai/llguidance/pull/145)
- always pass validation of grammars with special tokens without tokenizer [`0892a2a`](https://github.com/guidance-ai/llguidance/commit/0892a2adb5c8d818c025fe554bd67f05a5770aa7)

#### [v0.7.9](https://github.com/guidance-ai/llguidance/compare/v0.7.8...v0.7.9) 2025-03-24

- improve Python `LLMatcher.validate_grammar()` [`6b5f5ed`](https://github.com/guidance-ai/llguidance/commit/6b5f5eda7ca85ae2ca9a76c3813a0162a8b99b45)

#### [v0.7.6](https://github.com/guidance-ai/llguidance/compare/v0.7.5...v0.7.6) 2025-03-21

- Stabilize JSON schema property order [`#134`](https://github.com/guidance-ai/llguidance/pull/134)

#### [v0.7.5](https://github.com/guidance-ai/llguidance/compare/v0.7.4...v0.7.5) 2025-03-21

- add toktrie_hf_downloader crate [`11bea00`](https://github.com/guidance-ai/llguidance/commit/11bea00ecd1ef3c4a8970c1748db829e0c8a14de)
- use hf tokenizers library in python ext [`61728be`](https://github.com/guidance-ai/llguidance/commit/61728be47828525e959f6db226a0f17a783442bc)
- add LLTokenizer.tokenize_partial [`893cedf`](https://github.com/guidance-ai/llguidance/commit/893cedf614e234bd86bf01a99772d846b6ea884b)

#### [v0.7.4](https://github.com/guidance-ai/llguidance/compare/v0.7.3...v0.7.4) 2025-03-20

- fix gbnf parsing [`e5828b8`](https://github.com/guidance-ai/llguidance/commit/e5828b8a7a2fffaa9cf1aa2619c603a3d4ec7e17)

#### [v0.7.3](https://github.com/guidance-ai/llguidance/compare/v0.7.2...v0.7.3) 2025-03-19

- add LLMatcher.validate_grammar(); make it never raise [`0f8ec60`](https://github.com/guidance-ai/llguidance/commit/0f8ec6088a28eda13c2dd3d537733c0648e00cb3)
- add LLMatcher.reset() [`6a70aa7`](https://github.com/guidance-ai/llguidance/commit/6a70aa7efa8121fcd1865cefa9998926852eee25)

#### [v0.7.2](https://github.com/guidance-ai/llguidance/compare/v0.7.1...v0.7.2) 2025-03-18

- don't go into error state on final EOS [`1f0f21d`](https://github.com/guidance-ai/llguidance/commit/1f0f21d41fe88427d065b09414047d76b8b32041)

#### [v0.7.1](https://github.com/guidance-ai/llguidance/compare/v0.7.0...v0.7.1) 2025-03-18

- add `LLMatcher` interface in python
- add whitespace_pattern to JsonCompileOptions [`04a5491`](https://github.com/guidance-ai/llguidance/commit/04a54912cf6d082669674340833f06385f7b66f8)
- enable mypy in CI [`#140`](https://github.com/guidance-ai/llguidance/pull/140)
- add py.typed for annotations information [`#139`](https://github.com/guidance-ai/llguidance/pull/139)
- fix clippy warnings

#### [v0.7.0](https://github.com/guidance-ai/llguidance/compare/v0.6.31...v0.7.0) 2025-03-07

- remove JSON-based grammar serialization [`#133`](https://github.com/guidance-ai/llguidance/pull/133)

#### [v0.6.31](https://github.com/guidance-ai/llguidance/compare/v0.6.29...v0.6.31) 2025-03-05

- fix https://github.com/guidance-ai/guidance/issues/1131 - backtracking+prompt healing [`#1131`](https://github.com/guidance-ai/guidance/issues/1131)
- optimize substring [`9950600`](https://github.com/guidance-ai/llguidance/commit/9950600f46e433b4c42506f8816f61cee331774f)

#### [v0.6.29](https://github.com/guidance-ai/llguidance/compare/v0.6.28...v0.6.29) 2025-02-25

- [JSON] "x-guidance" JsonCompileOptions [`#130`](https://github.com/guidance-ai/llguidance/pull/130)

#### [v0.6.28](https://github.com/guidance-ai/llguidance/compare/v0.6.27...v0.6.28) 2025-02-21

- support for rollback() [`#126`](https://github.com/guidance-ai/llguidance/pull/126)
- allow lexer to produce alternative lexemes [`#124`](https://github.com/guidance-ai/llguidance/pull/124)
- make tokenize_with_greedy_fallback() handle invalid UTF not only at the end [`4762895`](https://github.com/guidance-ai/llguidance/commit/476289558d7d1edefe42eb87a093865debae8129)
- rise default lexer state limit from 50k to 250k [`202d3d5`](https://github.com/guidance-ai/llguidance/commit/202d3d545c14c63a62017b228c424a603619eb2a)

#### [v0.6.27](https://github.com/guidance-ai/llguidance/compare/v0.6.26...v0.6.27) 2025-02-18

- fix #122: captures with nullable symbols [`#122`](https://github.com/guidance-ai/llguidance/issues/122)

#### [v0.6.26](https://github.com/guidance-ai/llguidance/compare/v0.6.25...v0.6.26) 2025-02-14

- Extend Token::Number to match floats in scientific notation [`#121`](https://github.com/guidance-ai/llguidance/pull/121)
- native [suffix=...] support [`6d648c7`](https://github.com/guidance-ai/llguidance/commit/6d648c748bed4d83db28ed96ea87ad40ea51bc7e)

#### [v0.6.25](https://github.com/guidance-ai/llguidance/compare/v0.6.16...v0.6.25) 2025-02-12

- update referencing to 0.29.0 [`#118`](https://github.com/guidance-ai/llguidance/pull/118)
- Allow passing string for `capture_name` in lark syntax [`#119`](https://github.com/guidance-ai/llguidance/pull/119)

Plus a few releases messing with, deps, unsafe code cleanup.

#### [v0.6.16](https://github.com/guidance-ai/llguidance/compare/v0.6.15...v0.6.16) 2025-02-06

- Port over guidance's 'substring' [`#116`](https://github.com/guidance-ai/llguidance/pull/116)
- add %regex {...} syntax in lark for substrings [`b5ab086`](https://github.com/guidance-ai/llguidance/commit/b5ab0861e819b6e9221ef0aed3fcc827d6bad316)

#### [v0.6.15](https://github.com/guidance-ai/llguidance/compare/v0.6.14...v0.6.15) 2025-02-04

- move gbnf_to_lark to llguidance python pkg [`53134f1`](https://github.com/guidance-ai/llguidance/commit/53134f1befc6b6019bc88406e21b51c901943b51)
- add LLExecutor and fill_next_token_bitmask_par() [`ba4b917`](https://github.com/guidance-ai/llguidance/commit/ba4b9175b8d6c5445e1c0bcc8d5ef8e62b6cf73c)

#### [v0.6.14](https://github.com/guidance-ai/llguidance/compare/v0.6.13...v0.6.14) 2025-02-03

- add llguidance.numpy and llguidance.mlx submodules [`c627a39`](https://github.com/guidance-ai/llguidance/commit/c627a39689c9147fe7b072e5075960d16d43fc73)

#### [v0.6.13](https://github.com/guidance-ai/llguidance/compare/v0.6.12...v0.6.13) 2025-02-03

- add llguidance.torch and llguidance.hf submodules [`3fcdb1d`](https://github.com/guidance-ai/llguidance/commit/3fcdb1d93af076bbc8f1b3bef6fa9ead22b3e959)

#### [v0.6.12](https://github.com/guidance-ai/llguidance/compare/v0.6.11...v0.6.12) 2025-01-31

- fixes for numeric tokens [`b7c9970`](https://github.com/guidance-ai/llguidance/commit/b7c99709a9cb7f7a8a3c4716092e4d94fae2ff2c)
- make capture explicit in lark syntax [`2a57678`](https://github.com/guidance-ai/llguidance/commit/2a57678d9397e8be54cb0c9f14c4270604f8e1a5)
