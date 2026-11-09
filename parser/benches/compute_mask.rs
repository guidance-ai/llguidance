use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llguidance::{
    api::TopLevelGrammar,
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};

const DEFAULT_VOCAB_SIZE: usize = 32_768;
const CANCELLATION_ENV: &str = "LLGUIDANCE_BENCH_CANCELLATION";

const BLOG_SCHEMA_JSON: &str = include_str!("../../sample_parser/data/blog.schema.json");

// Lazy lexeme grammars for benchmarking (Issue #275)
const LAZY_DOTSTAR_GRAMMAR: &str = r#"
start[lazy]: /.*/ "END"
"#;

const LAZY_CHARCLASS_GRAMMAR: &str = r#"
start[lazy]: /[a-zA-Z0-9 ]*/ "END"
"#;

// Greedy equivalent for comparison
const GREEDY_CHARCLASS_GRAMMAR: &str = r#"
start: /[a-zA-Z0-9 ]*/ "END"
"#;

// Complex lazy lexeme patterns
const LAZY_SEQUENTIAL_GRAMMAR: &str = r#"
start: a b c
a[lazy]: /.*/ "A"
b[lazy]: /.*/ "B"
c[lazy]: /.*/ "C"
"#;

const LAZY_ALTERNATING_GRAMMAR: &str = r#"
start: (lazy | text)+
lazy[lazy]: /[^E]*/ "END"
text: /[^x]+/
"#;

const LAZY_MULTI_TERMINATOR_GRAMMAR: &str = r#"
start[lazy]: /.*/ ("END1" | "END2" | "END3")
"#;

// Different prefixes representing various parser states
const PREFIX_START: &[u8] = b""; // Start of JSON
const PREFIX_AFTER_KEY: &[u8] = b"{\"title\":"; // After key, expecting value
const PREFIX_IN_STRING: &[u8] = b"{\"title\":\""; // Inside a string value
const PREFIX_MID_STRING: &[u8] = b"{\"title\":\"Hello World"; // Mid-string with content

struct SyntheticTokEnv {
    trie: TokTrie,
}

impl TokenizerEnv for SyntheticTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

fn synthetic_tok_env(vocab_size: usize) -> TokEnv {
    let eos_token = (vocab_size - 1) as TokenId;
    let mut tokens = Vec::with_capacity(vocab_size);

    for byte in 0u8..=255 {
        tokens.push(vec![byte]);
    }

    let prefixes: &[u8] = b" \"{[\\etaoin";
    for i in 0..(vocab_size - tokens.len() - 1) {
        let mut tok = Vec::with_capacity(5);
        tok.push(prefixes[i % prefixes.len()]);
        tok.extend_from_slice(&(i as u32).to_le_bytes());
        tokens.push(tok);
    }

    tokens.push(b"\xFF<|eos|>".to_vec());
    let trie = TokTrie::from(&TokRxInfo::new(vocab_size as u32, eos_token), &tokens);
    Arc::new(SyntheticTokEnv { trie })
}

fn blog_grammar() -> TopLevelGrammar {
    let schema: serde_json::Value = serde_json::from_str(BLOG_SCHEMA_JSON).unwrap();
    TopLevelGrammar::from_json_schema(schema)
}

fn benchmark_cancellation_enabled() -> bool {
    match std::env::var(CANCELLATION_ENV).as_deref() {
        Ok("enabled") => true,
        Ok("disabled") | Err(_) => false,
        Ok(value) => panic!("{CANCELLATION_ENV} must be enabled or disabled, got {value:?}"),
    }
}

fn create_matcher(
    tok_env: &TokEnv,
    grammar: TopLevelGrammar,
    prefix: &[u8],
    cancellation_enabled: bool,
) -> Matcher {
    let mut factory = ParserFactory::new_simple(tok_env).unwrap();
    factory.quiet();
    let mut matcher = Matcher::new(factory.create_parser(grammar));
    if cancellation_enabled {
        matcher = matcher.into_cancellable();
    }

    for &byte in prefix {
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(byte as TokenId));
        matcher.consume_token(byte as TokenId).unwrap();
    }
    matcher
}

/// Benchmark compute_mask at different vocabulary sizes.
/// Uses vocab size as throughput metric since larger vocabs require more work.
fn bench_compute_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mask");
    let cancellation_enabled = benchmark_cancellation_enabled();

    // Realistic LLM vocabulary sizes (8k to 128k)
    for vocab_size in [8_192, 32_768, 65_536, 128_000] {
        group.throughput(Throughput::Elements(vocab_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);
                let mut matcher = create_matcher(
                    &tok_env,
                    blog_grammar(),
                    PREFIX_IN_STRING,
                    cancellation_enabled,
                );
                b.iter(|| {
                    matcher.invalidate_bias_cache();
                    black_box(matcher.compute_mask().unwrap())
                })
            },
        );
    }
    group.finish();
}

/// Benchmark compute_mask at different parser positions within the grammar.
/// This reveals if certain grammar states are slower than others.
fn bench_compute_mask_positions(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mask_positions");
    let vocab_size = DEFAULT_VOCAB_SIZE;
    let cancellation_enabled = benchmark_cancellation_enabled();

    let positions = [
        ("start", PREFIX_START),
        ("after_key", PREFIX_AFTER_KEY),
        ("in_string", PREFIX_IN_STRING),
        ("mid_string", PREFIX_MID_STRING),
    ];

    group.throughput(Throughput::Elements(vocab_size as u64));

    for (name, prefix) in positions {
        group.bench_with_input(BenchmarkId::from_parameter(name), &prefix, |b, &prefix| {
            let tok_env = synthetic_tok_env(vocab_size);
            let mut matcher =
                create_matcher(&tok_env, blog_grammar(), prefix, cancellation_enabled);
            b.iter(|| {
                matcher.invalidate_bias_cache();
                black_box(matcher.compute_mask().unwrap())
            })
        });
    }
    group.finish();
}

/// Benchmark realistic token generation loop (no rollback).
/// Measures per-token latency during continuous generation.
fn bench_token_generation(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("token_generation");
    let num_tokens = 20;
    let cancellation_enabled = benchmark_cancellation_enabled();

    for vocab_size in [32_768, 65_536, 128_000] {
        group.throughput(Throughput::Elements(num_tokens));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);

                b.iter_batched(
                    || {
                        create_matcher(
                            &tok_env,
                            blog_grammar(),
                            PREFIX_IN_STRING,
                            cancellation_enabled,
                        )
                    },
                    |mut m| {
                        for _ in 0..num_tokens {
                            let mask = m.compute_mask().unwrap();
                            // Pick first allowed lowercase letter
                            let tok = (b'a' as TokenId..=b'z' as TokenId)
                                .find(|&t| mask.is_allowed(t))
                                .unwrap_or(b'a' as TokenId);
                            m.consume_token(tok).unwrap();
                        }
                        black_box(m)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

/// Benchmark cold-start: time to create parser and compute first mask.
/// Important for latency-sensitive applications.
fn bench_first_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_mask");
    let cancellation_enabled = benchmark_cancellation_enabled();

    for vocab_size in [32_768, 65_536, 128_000] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);
                let grammar = blog_grammar();

                b.iter(|| {
                    let mut factory = ParserFactory::new_simple(&tok_env).unwrap();
                    factory.quiet();
                    let mut matcher = Matcher::new(factory.create_parser(grammar.clone()));
                    if cancellation_enabled {
                        matcher = matcher.into_cancellable();
                    }
                    black_box(matcher.compute_mask().unwrap())
                })
            },
        );
    }
    group.finish();
}

fn run_matcher_on_input(m: &mut Matcher, input: &[u8]) {
    for &byte in input {
        let mask = m.compute_mask().unwrap();
        if !mask.is_allowed(byte as TokenId) {
            break;
        }
        m.consume_token(byte as TokenId).unwrap();
    }
}

/// Benchmark lazy lexeme performance (Issue #275).
/// Compares lazy vs greedy patterns with permissive regexes like [...]* and .*
fn bench_lazy_lexeme(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("lazy_lexeme");
    let vocab_size = DEFAULT_VOCAB_SIZE;
    let cancellation_enabled = benchmark_cancellation_enabled();

    // 50 'x' characters as input
    const INPUT: &[u8] = b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

    let grammars = [
        ("lazy_dotstar", LAZY_DOTSTAR_GRAMMAR),
        ("lazy_charclass", LAZY_CHARCLASS_GRAMMAR),
        ("greedy_charclass", GREEDY_CHARCLASS_GRAMMAR),
    ];

    group.throughput(Throughput::Elements(INPUT.len() as u64));

    let tok_env = synthetic_tok_env(vocab_size);
    for (name, grammar) in grammars {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &grammar,
            |b, &grammar| {
                b.iter_batched(
                    || {
                        create_matcher(
                            &tok_env,
                            TopLevelGrammar::from_lark(grammar.to_string()),
                            b"",
                            cancellation_enabled,
                        )
                    },
                    |mut m| {
                        run_matcher_on_input(&mut m, INPUT);
                        black_box(m)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

/// Benchmark complex lazy lexeme patterns: sequential, alternating, multi-terminator
fn bench_lazy_lexeme_complex(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("lazy_lexeme_complex");
    let vocab_size = DEFAULT_VOCAB_SIZE;
    let cancellation_enabled = benchmark_cancellation_enabled();

    // Each test case: (name, grammar, input_bytes)
    let test_cases: [(&str, &str, &[u8]); 4] = [
        // Long content in single lazy lexeme - primary use case
        (
            "long_content",
            LAZY_DOTSTAR_GRAMMAR,
            b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxEND",
        ),
        // Sequential lazy lexemes
        ("sequential", LAZY_SEQUENTIAL_GRAMMAR, b"xxxAyyyBzzzC"),
        // Alternating lazy/text
        (
            "alternating",
            LAZY_ALTERNATING_GRAMMAR,
            b"xxxENDaaaxxxENDbbb",
        ),
        // Multiple possible terminators
        (
            "multi_terminator",
            LAZY_MULTI_TERMINATOR_GRAMMAR,
            b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxEND2",
        ),
    ];

    let tok_env = synthetic_tok_env(vocab_size);
    for (name, grammar, input) in test_cases {
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &input, |b, &input| {
            b.iter_batched(
                || {
                    create_matcher(
                        &tok_env,
                        TopLevelGrammar::from_lark(grammar.to_string()),
                        b"",
                        cancellation_enabled,
                    )
                },
                |mut m| {
                    run_matcher_on_input(&mut m, input);
                    black_box(m)
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// Benchmark the GNF displacement bridge (no DFA, the pure PDA simulation)
/// vs the DFA bridge (the P3, the DFA matching). The GNF bridge is the next
/// increment (the live bridge, without the DFA dependency).
#[cfg(feature = "dpda")]
fn bench_gnf_bridge(c: &mut Criterion) {
    use llguidance::dpda_adapter;
    let env = llg_test_utils::get_tok_env().clone();
    let f = ParserFactory::new(
        &env,
        InferenceCapabilities {
            ff_tokens: true,
            backtrack: false,
            conditional_ff_tokens: false,
            fork: false,
        },
        &SlicedBiasComputer::general_slices(),
    ).expect("the factory");
    let g = TopLevelGrammar::from_lark(r#"start: <[32006]> /.*/"#.to_string());
    let mut parser = f.create_parser(g.clone()).expect("the parser");
    parser.start_without_prompt();
    let grm = parser.parser.grammar().clone();
    let pda = dpda_adapter::compile_pda(&grm).expect("the PDA compile");
    let trie = env.tok_trie();
    // A subset of tokens (the 1000, not the full 35000 vocab, for benchmark speed).
    let subset: Vec<u32> = (0..1000).collect();
    let mut group = c.benchmark_group("gnf_bridge");
    group.bench_function("dfa_bridge", |b| {
        b.iter(|| {
            black_box(dpda_adapter::terminal_token_map(&grm, &env));
        })
    });
    group.bench_function("gnf_bridge_subset", |b| {
        b.iter(|| {
            black_box(dpda_adapter::gnf_displacement_bridge_subset(
                &pda,
                &subset,
                &|tok| trie.token_str(tok).bytes().map(|b| b as u32).collect(),
            ));
        })
    });
    // The grouped full bridge (no per-token BFS, the byte-sequence grouping +
    // the rayon parallelism). The num_groups << the vocab_size (the many tokens
    // share a byte sequence).
    let eos_tokens: std::collections::HashSet<u32> = trie.eos_tokens().iter().copied().collect();
    group.bench_function("gnf_bridge_grouped_full", |b| {
        b.iter(|| {
            black_box(dpda_adapter::gnf_displacement_bridge(
                &pda,
                trie.vocab_size(),
                &eos_tokens,
                &|tok| trie.token_str(tok).bytes().map(|b| b as u32).collect(),
            ));
        })
    });
    group.finish();
}

/// Per-step timing of the mask paths at a fixed parser position (the production
/// hot path: one mask computation per token, the bias cache invalidated).
/// Measures:
///   - earley: the compute_mask (the Earley item-set, the authoritative path)
///   - product: the product_config_mask (the PDA frontier + the lexer, the target)
/// The parser is created ONCE outside the timed region (the grammar compile + the
/// PDA compile are not the per-step cost). The timed region is a single mask
/// computation at a fixed position, which is what production pays per token.
#[cfg(feature = "dpda")]
fn bench_mask_path_costs(c: &mut Criterion) {
    let env = llg_test_utils::get_tok_env().clone();
    let f = ParserFactory::new(
        &env,
        InferenceCapabilities {
            ff_tokens: true,
            backtrack: false,
            conditional_ff_tokens: false,
            fork: false,
        },
        &SlicedBiasComputer::general_slices(),
    )
    .expect("the factory");

    // A realistic JSON object grammar (the tool-call envelope shape). NOTE: this
    // grammar has the integer + the string leaves (the int-ranges + the regex
    // lexemes), so grammar_is_pure_cfg is FALSE and the PDA path is OFF (the
    // CPU Earley is used). It is the benchmark for the EARLEY path cost.
    let g = TopLevelGrammar::from_json_schema(serde_json::json!({
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write"]},
            "path": {"type": "string"},
            "count": {"type": "integer", "minimum": 0, "maximum": 100}
        },
        "required": ["action", "path"]
    }));

    // A PURE-CFG token-range grammar (no int/string/skip leaves) — the
    // subclass where the PDA path IS engaged (the pda_engaged true). This is the
    // benchmark for the PDA product-mask cost where it is actually used.
    let g_pure = TopLevelGrammar::from_lark(
        r#"start: <[123]> <[97]> <[58]> <[98]> <[125]>"#.to_string(),
    );
    // A realistic NESTED pure-CFG grammar (the JSON object structure, not the
    // artificial 200-deep sequence): the obj -> { key : val , obj } nesting, the
    // token-range terminals only (without the int/string regex leaves, so the
    // PDA path engages). This is the production-relevant PDA-path grammar.
    let g_pure_nested = TopLevelGrammar::from_lark(
        r#"start: <[123]> <[97]> <[125]>
              <[123]> <[98]> <[58]> <[99]> <[125]>"#.to_string(),
    );

    // A valid token walk (the JSON object). We advance the parser to a mid-walk
    // position ONCE, then time single mask computations at that position.
    let sample = r#"{"action":"read","path":"/tmp","count":5}"#;
    let tokens: Vec<u32> = env.tokenize(sample).iter().map(|&t| t as u32).collect();

    let mut group = c.benchmark_group("mask_path_costs");

    // The earley per-step cost: a single compute_mask with the bias cache
    // invalidated (the production hot path, not the cache hit).
    group.bench_function("earley_compute_mask_per_step", |b| {
        let parser = f.create_parser(g.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        // Advance to a mid-walk position (not timing the cheap start position).
        for &tok in &tokens[..tokens.len() / 2] {
            m.compute_mask().expect("the mask");
            m.consume_token(tok).expect("the consume");
            m.settle();
        }
        b.iter(|| {
            m.invalidate_bias_cache();
            black_box(m.compute_mask().expect("the earley mask"))
        });
    });

    // The product per-step cost: a single product_config_mask at the same position.
    group.bench_function("product_config_mask_per_step", |b| {
        let parser = f.create_parser(g.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        for &tok in &tokens[..tokens.len() / 2] {
            m.compute_mask().expect("the mask");
            m.consume_token(tok).expect("the consume");
            m.settle();
        }
        m.product_config_mask();
        b.iter(|| black_box(m.product_config_mask()));
    });

    // Component breakdown: isolate the PDA mask cost (the pda_mask, the PDA frontier
    // + the bridge lift) from the full product cost (the pda_mask + the lexer
    // flush + the token-ranges + the EOS). The difference is the lexer cost.
    group.bench_function("product_pda_mask_only", |b| {
        let parser = f.create_parser(g.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        for &tok in &tokens[..tokens.len() / 2] {
            m.compute_mask().expect("the mask");
            m.consume_token(tok).expect("the consume");
            m.settle();
        }
        m.product_config_mask(); // warm the bridge memo
        b.iter(|| black_box(m.product_config_mask()));
    });

    // The apples-to-apples comparison for the PURE-CFG subclass (the g_pure, where
    // the PDA path is engaged): the Earley compute_mask vs the PDA product mask.
    // This is the production-relevant comparison (the PDA is only used for the
    // pure-CFG grammars, the parametric ones use Earley regardless).
    group.bench_function("purecfg_earley_compute_mask_per_step", |b| {
        let parser = f.create_parser(g_pure.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        m.settle();
        let pure_walk: Vec<u32> = vec![123, 97, 58, 98, 125];
        for &tok in &pure_walk[..pure_walk.len() / 2] {
            m.compute_mask().expect("the mask");
            m.consume_token(tok).expect("the consume");
            m.settle();
        }
        b.iter(|| {
            m.invalidate_bias_cache();
            black_box(m.compute_mask().expect("the earley mask"))
        });
    });

    // The PDA product mask for the pure-CFG grammar (the g_pure, the pda_engaged TRUE):
    // the single-tracker cost where the PDA path is actually used.
    group.bench_function("purecfg_pda_product_mask_per_step", |b| {
        let parser = f.create_parser(g_pure.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        m.settle();
        let pure_walk: Vec<u32> = vec![123, 97, 58, 98, 125];
        for &tok in &pure_walk[..pure_walk.len() / 2] {
            m.compute_mask().expect("the mask");
            m.consume_token(tok).expect("the consume");
            m.settle();
        }
        m.product_config_mask(); // warm the bridge memo
        b.iter(|| black_box(m.product_config_mask()));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .noise_threshold(0.05);
    targets = bench_compute_mask, bench_compute_mask_positions, bench_token_generation, bench_first_mask, bench_lazy_lexeme, bench_lazy_lexeme_complex, bench_gnf_bridge, bench_mask_path_costs
}
criterion_main!(benches);
