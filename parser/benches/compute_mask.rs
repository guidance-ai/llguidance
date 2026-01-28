use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};

const BLOG_SCHEMA_JSON: &str = include_str!("../../sample_parser/data/blog.schema.json");
const TITLE_STRING_PREFIX: &[u8] = b"{\"title\":\"";

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

fn matcher_at_prefix(tok_env: &TokEnv, prefix: &[u8]) -> Matcher {
    let mut factory = ParserFactory::new_simple(tok_env).unwrap();
    factory.quiet();
    let mut matcher = Matcher::new(factory.create_parser(blog_grammar()));

    for &byte in prefix {
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(byte as TokenId));
        matcher.consume_token(byte as TokenId).unwrap();
    }
    matcher
}

fn bench_compute_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mask");

    // Test different vocabulary sizes to see how optimizations scale
    for vocab_size in [1_024, 8_192, 32_768, 65_536] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);
                let mut matcher = matcher_at_prefix(&tok_env, TITLE_STRING_PREFIX);
                b.iter(|| black_box(matcher.compute_mask().unwrap()))
            },
        );
    }
    group.finish();
}

fn bench_generate_string(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("generate_string");

    for vocab_size in [1_024, 8_192, 32_768, 65_536] {
        group.throughput(Throughput::Elements(10)); // 10 tokens generated
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);
                let matcher = matcher_at_prefix(&tok_env, TITLE_STRING_PREFIX);

                b.iter_batched(
                    || matcher.deep_clone(),
                    |mut m| {
                        for _ in 0..10 {
                            let mask = m.compute_mask().unwrap();
                            let tok = (b'a' as TokenId..=b'z' as TokenId)
                                .find(|&t| mask.is_allowed(t))
                                .unwrap_or(b'a' as TokenId);
                            m.consume_token(tok).unwrap();
                        }
                        m.rollback(10).unwrap();
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(5))
        .noise_threshold(0.05);
    targets = bench_compute_mask, bench_generate_string
}
criterion_main!(benches);
