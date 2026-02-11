### Changelog

All notable changes to this project will be documented in this file. Dates are displayed in UTC.

If a release doesn't introduce any interesting changes (build fixes etc.), it's skipped.

#### [v1.5.1](https://github.com/guidance-ai/llguidance/compare/v1.5.0...v1.5.1) 2026-02-10

- Mask caching to improve performance, especially for lazy lexemes [`#279`](https://github.com/guidance-ai/llguidance/pull/279)
- Bump `bytes` from 1.10.1 to 1.11.1 [`#281`](https://github.com/guidance-ai/llguidance/pull/281)
- Add sentence to docs about negation and UTF-8 [`#280`](https://github.com/guidance-ai/llguidance/pull/280)

#### [v1.5.0](https://github.com/guidance-ai/llguidance/compare/v1.4.0...v1.5.0) 2026-02-02

- [FIX] Coerce `Metaspace` prepend_scheme to `never` [`#272`](https://github.com/guidance-ai/llguidance/pull/272)
- Speed up toktrie add_bias (~3% improvement) [`#264`](https://github.com/guidance-ai/llguidance/pull/264)
  - Thanks [@omarkilani](https://github.com/omarkilani)!
- Add discord & email to readme [`#273`](https://github.com/guidance-ai/llguidance/pull/273)
- Add parallel token consumption API (consume_token_par) [`#271`](https://github.com/guidance-ai/llguidance/pull/271)
  - Thanks [@alecsolder](https://github.com/alecsolder)!
- Add toolcalls doc [`#270`](https://github.com/guidance-ai/llguidance/pull/270)
- Fix typo in README.md [`#269`](https://github.com/guidance-ai/llguidance/pull/269)
  - Thanks [@juliendenize](https://github.com/juliendenize)!
- Mark immutable pyclasses as frozen [`#260`](https://github.com/guidance-ai/llguidance/pull/260)
  - Thanks [@ngoldbaum](https://github.com/ngoldbaum)!

#### [v1.4.0](https://github.com/guidance-ai/llguidance/compare/v1.3.0...v1.4.0) 2025-11-18

- Update pyo3 dependency to latest version to support python 3.14t [`#255`](https://github.com/guidance-ai/llguidance/pull/255)
  - Thank you [@ngoldbaum](https://github.com/ngoldbaum)!
- Fix tokenization of special tokens for ByteTokenizer [`#249`](https://github.com/guidance-ai/llguidance/pull/249)
  - Thank you [@daulet](https://github.com/daulet)!
- Bump minimum python version to 3.10 [`#252`](https://github.com/guidance-ai/llguidance/pull/252)
- Update MSRV in README [`30d1515`](https://github.com/guidance-ai/llguidance/commit/30d15150e2c6bae9e52dea58c72574c160f4697a)
  - Thank you for pointing out the error [@nteodosio](https://github.com/nteodosio)!

#### [v1.3.0](https://github.com/guidance-ai/llguidance/compare/v1.2.0...v1.3.0) 2025-10-20

- change semantics of JSON separators (now regex patterns instead of text literals) [`#246`](https://github.com/guidance-ai/llguidance/pull/246)
- fix Inconsistent handling of the EOS token in `try_consume_tokens` [`#244`](https://github.com/guidance-ai/llguidance/pull/244)
- add option for initial skip of `%ignore` lexeme [`#235`](https://github.com/guidance-ai/llguidance/pull/235)
- bump MSRV to 1.87.0 [`#243`](https://github.com/guidance-ai/llguidance/pull/243)

#### [v1.2.0](https://github.com/guidance-ai/llguidance/compare/v1.1.2...v1.2.0) 2025-08-12

- support multithreaded compute bitmask for speculative decoding [`#225`](https://github.com/guidance-ai/llguidance/pull/225)
  - thank you [@ZonePG](https://github.com/ZonePG)!
- `force_lexeme_end` -> `try_lexeme_end` in lark lexer when out of input [`#229`](https://github.com/guidance-ai/llguidance/pull/229); fixes [`#228`](https://github.com/guidance-ai/llguidance/issues/228)
- more JSON test coverage

#### [v1.1.2](https://github.com/guidance-ai/llguidance/compare/v1.1.1...v1.1.2) 2025-08-08

- add flag in ParserLimits to disable verbose errors [`#227`](https://github.com/guidance-ai/llguidance/pull/227)
- new tests and cleanups

#### [v1.1.1](https://github.com/guidance-ai/llguidance/compare/v1.1.0...v1.1.1) 2025-07-23

- prevent error state when calling `try_consume_tokens` after parser is stopped [`#213`](https://github.com/guidance-ai/llguidance/pull/213); fixes [`#211`](https://github.com/guidance-ai/llguidance/issues/211)
- set parser stop condition in `try_consume_tokens` even when some tokens are rejected [`#212`](https://github.com/guidance-ai/llguidance/pull/212)

#### [v1.1.0](https://github.com/guidance-ai/llguidance/compare/v1.0.1...v1.1.0) 2025-07-18

- disable hf tokenizer truncation and padding [`#205`](https://github.com/guidance-ai/llguidance/pull/205); fixes [`#1322`](https://github.com/guidance-ai/guidance/issues/1322)
- llama_cpp tokenizers: infer added tokens starting/ending with &lt; and &gt; to be special tokens [`#202`](https://github.com/guidance-ai/llguidance/pull/202)
- add lark syntax for "any token" and negation of token ranges [`#201`](https://github.com/guidance-ai/llguidance/pull/201)
- add de-recursion cook book to docs [`#199`](https://github.com/guidance-ai/llguidance/pull/199)

#### [v1.0.1](https://github.com/guidance-ai/llguidance/compare/v1.0.0...v1.0.1) 2025-07-03

- fix: tokenizers normalizers sequence api changed [`#195`](https://github.com/guidance-ai/llguidance/pull/195)
- Strip debug info from the wheels [`#194`](https://github.com/guidance-ai/llguidance/pull/194)

Thank you @ammar-elsabe and @Ahajha!

#### [v1.0.0](https://github.com/guidance-ai/llguidance/compare/v0.7.30...v1.0.0) 2025-06-23

This is identical to `0.7.30`, but indicates intended stability and from now on we'll try to follow semver.

#### [v0.7.30](https://github.com/guidance-ai/llguidance/compare/v0.7.29...v0.7.30) 2025-06-23

- parametric grammars [`#192`](https://github.com/guidance-ai/llguidance/pull/192)
- allow for tokens up to ~2k bytes; fixes #188 [`#188`](https://github.com/guidance-ai/llguidance/issues/188)

#### [v0.7.29](https://github.com/guidance-ai/llguidance/compare/v0.7.28...v0.7.29) 2025-06-06

- cargo fmt

#### [v0.7.28](https://github.com/guidance-ai/llguidance/compare/v0.7.27...v0.7.28) 2025-06-06

- fix lexer_stack=... panic with numeric tokens [`4e91b0f`](https://github.com/guidance-ai/llguidance/commit/4e91b0fa0c03572a5fc221ac0e0b05035af9dcfa)

#### [v0.7.27](https://github.com/guidance-ai/llguidance/compare/v0.7.26...v0.7.27) 2025-06-04

- add toktrie_tiktoken and llguidance.tiktoken.lltokenizer_from_encoding [`#154`](https://github.com/guidance-ai/llguidance/issues/154)
- implement clone on StopController [`#185`](https://github.com/guidance-ai/llguidance/issues/185)

#### [v0.7.26](https://github.com/guidance-ai/llguidance/compare/v0.7.25...v0.7.26) 2025-05-30

- add support for & and ~ in lark regexes [`96fcee3`](https://github.com/guidance-ai/llguidance/commit/96fcee373697b57bead94d1bc06c17cf1c6134e4)
- dump grammar in errors in LLInterpreter [`#183`](https://github.com/guidance-ai/llguidance/pull/183)
- don't check lexer bytes invariant when we cannot rollback [`ec22083`](https://github.com/guidance-ai/llguidance/commit/ec220837051513a70177974ca389b7bf387455f1)


#### [v0.7.25](https://github.com/guidance-ai/llguidance/compare/v0.7.24...v0.7.25) 2025-05-28

- add parse_special=False to tokenize_str/bytes() in python [`#181`](https://github.com/guidance-ai/llguidance/pull/181)

#### [v0.7.24](https://github.com/guidance-ai/llguidance/compare/v0.7.23...v0.7.24) 2025-05-23

- add the sentinel token hack, fixes #180 [`#180`](https://github.com/guidance-ai/llguidance/issues/180)

#### [v0.7.23](https://github.com/guidance-ai/llguidance/compare/v0.7.22...v0.7.23) 2025-05-22

- native llama.cpp tokenizer support [`#179`](https://github.com/guidance-ai/llguidance/pull/179)
- improve special token detection in HF tokenizers [`6cae393`](https://github.com/guidance-ai/llguidance/commit/6cae393b9c04fe67621615ff22b46beab512d069)

#### [v0.7.22](https://github.com/guidance-ai/llguidance/compare/v0.7.21...v0.7.22) 2025-05-21

- Keep EOS token bytes in `TokenizerWrapper` [`#178`](https://github.com/guidance-ai/llguidance/pull/178)
- Stop using prefix/sentinel strings for `TokenizerWrapper` [`#175`](https://github.com/guidance-ai/llguidance/pull/175)
- avoid taking poisoned locks, see [`#174`](https://github.com/guidance-ai/llguidance/issues/174) [`d41aa9a`](https://github.com/guidance-ai/llguidance/commit/d41aa9a4427967708a951506b2bc0e395871b6c8); thanks [@g-eoj](https://github.com/g-eoj)

#### [v0.7.21](https://github.com/guidance-ai/llguidance/compare/v0.7.20...v0.7.21) 2025-05-20

- include parser state in errors [`82e34da`](https://github.com/guidance-ai/llguidance/commit/82e34da704d22f04979d8cbc54a0ac00885a277d)
- tighten email format in JSON schema [`7454ea9`](https://github.com/guidance-ai/llguidance/commit/7454ea9df958f8bcc42e6bb986d6de397de65b3e)

#### [v0.7.20](https://github.com/guidance-ai/llguidance/compare/v0.7.19...v0.7.20) 2025-05-15

- use fancy-regex instead of onig as tokenizers regex library [`#172`](https://github.com/guidance-ai/llguidance/pull/172)
  - fixes compilation on GCC 15, thanks [@Slowki](https://github.com/Slowki)
- msrv 1.80 support (incl. derivre bump) [`c89e386`](https://github.com/guidance-ai/llguidance/commit/c89e386685cd911a89fd47df225de88f88c10883), thank you [@nteodosio](https://github.com/nteodosio) for initial [PR](https://github.com/guidance-ai/llguidance/pull/170)!

#### [v0.7.19](https://github.com/guidance-ai/llguidance/compare/v0.7.18...v0.7.19) 2025-04-24

- fix a numeric token bug [`1f59edf`](https://github.com/guidance-ai/llguidance/commit/1f59edfc49b44cfba74b2380f34874a0778d9441)

#### [v0.7.18](https://github.com/guidance-ai/llguidance/compare/v0.7.17...v0.7.18) 2025-04-22

- apply x-guidance also in %json{} [`2627891`](https://github.com/guidance-ai/llguidance/commit/2627891c72c7e38062cd3e052f1de146d2e21635)
- more sensible llg_validate_grammar() signature [`41928c0`](https://github.com/guidance-ai/llguidance/commit/41928c07298e69e3c8adc4a3c1f43ef9b1cc1c6b)

#### [v0.7.17](https://github.com/guidance-ai/llguidance/compare/v0.7.16...v0.7.17) 2025-04-22

- support for min/maxProperties in JSON Schema [`#168`](https://github.com/guidance-ai/llguidance/issues/168)
- give priority to &lt;[123]&gt; over "foo" in grammar [`3e9f3b5`](https://github.com/guidance-ai/llguidance/commit/3e9f3b5e8c1cac92daab6e9709f01ebccc20342b)

#### [v0.7.16](https://github.com/guidance-ai/llguidance/compare/v0.7.15...v0.7.16) 2025-04-17

- fix special token tokenization [`ae7870f`](https://github.com/guidance-ai/llguidance/commit/ae7870f05ca0de68599088607ba742b7071f92ad)

#### [v0.7.15](https://github.com/guidance-ai/llguidance/compare/v0.7.14...v0.7.15) 2025-04-16

- support for patternProperties in JSON schema [`#167`](https://github.com/guidance-ai/llguidance/pull/167)
- add lenient option to JSON schemas [`#163`](https://github.com/guidance-ai/llguidance/pull/163) [`#136`](https://github.com/guidance-ai/llguidance/issues/136)
- Add llg_validate_grammar() in C FFI [`e5c21cf`](https://github.com/guidance-ai/llguidance/commit/e5c21cf480a17e6b310e46b24b272576cfd9c4c6)

#### [v0.7.14](https://github.com/guidance-ai/llguidance/compare/v0.7.13...v0.7.14) 2025-04-11

- support %lark { ... } syntax for nested grammars [`#157`](https://github.com/guidance-ai/llguidance/pull/157)
- treat \d and \w in json schema as ASCII; fix ^$ anchors [`#158`](https://github.com/guidance-ai/llguidance/issues/158)
- make it build without "lark" feature again [`929b13f`](https://github.com/guidance-ai/llguidance/commit/929b13f1e523b4cfda6d842ef84cfabf8b99224f)
- bump derivre to 0.3.7 and use it for anchor handling in json schema (fixes a few testcases) [`bb228cb`](https://github.com/guidance-ai/llguidance/commit/bb228cbca080f1382bb992dd27bddc0223e9dd00)
- expose regex_to_lark() in Rust and Python; add \d\w\s replacement [`78fb32f`](https://github.com/guidance-ai/llguidance/commit/78fb32fe2745d30ca94a62b00e5a7299750d80b0)
- fix usage of / vs \* in python signatures [`ca73c2a`](https://github.com/guidance-ai/llguidance/commit/ca73c2abd44e75d569230b942f53c72b052ed2ab)

#### [v0.7.13](https://github.com/guidance-ai/llguidance/compare/v0.7.12...v0.7.13) 2025-04-05

- expose LLParserLimits in Python API [`598dc8f`](https://github.com/guidance-ai/llguidance/commit/598dc8f37f69f51244e54d9885445abf02a515a7)
- pre-compute lexer states for particularly large regexes (can be disabled in ParserLimits)

#### [v0.7.12](https://github.com/guidance-ai/llguidance/compare/v0.7.11...v0.7.12) 2025-04-04

- performance optimizations
- use factory in C FFI (otherwise slicer was not used)
- add some null checks and safety comments in C FFI
- implement subgrammar lexeme class merging; fixes [`#113`](https://github.com/guidance-ai/llguidance/issues/113)

#### [v0.7.11](https://github.com/guidance-ai/llguidance/compare/v0.7.10...v0.7.11) 2025-03-27

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
