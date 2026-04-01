/// Full-featured CLI tool demonstrating llguidance constrained decoding.
///
/// This binary exercises all major features of the llguidance library:
///   - Loading grammars from JSON Schema, Lark, internal (.ll.json), and text formats
///   - Using a real HuggingFace tokenizer (downloaded on first use)
///   - Three operating modes:
///     1. **Mask-only**: Compile the grammar and compute one token mask (no `--input` or `--rnd`)
///     2. **Random generation** (`--rnd N`): Simulate an LLM by sampling random valid tokens
///     3. **Input validation** (`--input FILE`): Verify a known input conforms to the grammar
///
/// See `minimal.rs` for a stripped-down version focused on the core decoding loop.
///
/// Usage examples:
///   cargo run -- data/blog.schema.json --input data/blog.sample.json
///   cargo run -- data/rfc.lark --input data/rfc.xml
///   cargo run -- data/blog.schema.json --rnd 100 --verbose
use clap::Parser;
use std::{fs::File, io::Read, sync::Arc, vec};

use llguidance::{api::TopLevelGrammar, toktrie::TokEnv, Matcher, ParserFactory};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::json;

/// Sample a random set-bit index from a `SimpleVob`.
///
/// This is an inline copy of `llg_test_utils::sample_from_vob` to keep
/// sample_parser self-contained as a standalone demo crate.
fn sample_from_vob(rng: &mut impl Rng, vob: &llguidance::toktrie::SimpleVob) -> u32 {
    let nset = vob.num_set();
    assert!(nset > 0);
    if nset > vob.len() / 10 {
        loop {
            let idx = rng.random_range(0..vob.len());
            if vob[idx] {
                return idx as u32;
            }
        }
    } else {
        let choices = vob.to_list();
        choices[rng.random_range(0..choices.len())]
    }
}

fn dump_tokenizer(name: &str) {
    let btok = toktrie_hf_downloader::byte_tokenizer_from_name(name).unwrap();
    let vecs = btok.token_bytes();
    for v in vecs.iter() {
        let v: String = v
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("");
        println!("{v}");
    }
}

/// CLI arguments for the sample parser.
///
/// The tool operates in three modes depending on which arguments are provided:
///   - No `--input` or `--rnd`: compile the grammar and compute one mask (test grammar validity)
///   - `--rnd N`: generate N random tokens that satisfy the grammar (simulates an LLM)
///   - `--input FILE`: validate that the tokens in FILE conform to the grammar
#[derive(Parser, Debug, Default)]
#[command(version, about, long_about = None)]
pub struct CliOptions {
    /// Print out tokenizer stuff
    #[arg(long)]
    dump_tokenizer: bool,

    /// Specify HF tokenizer to use
    #[arg(long, default_value = "microsoft/Phi-3.5-mini-instruct")]
    tokenizer: String,

    /// Input file for the grammar
    #[arg(long, short = 'i')]
    input: Option<String>,

    /// Random seed
    #[arg(long, default_value = "1")]
    seed: u32,

    /// Generate N random tokens for input
    #[arg(long, short = 'r')]
    rnd: Option<usize>,

    /// Set stderr log level; 1 is warnings only, 2 is verbose (default: 1)
    #[arg(long, short = 'l', default_value = "1")]
    log_level: u32,

    /// Verbose printing
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Split .txt input on words, not lines
    #[arg(long)]
    split_words: bool,

    /// Repeat the operation N times for profiling
    #[arg(long, default_value = "1")]
    repeat: usize,

    /// Increase lexer limit N times
    #[arg(long, default_value = "1")]
    lexer_limit: usize,

    /// Allowed startup lexer cost in thousands (default 1000)
    #[arg(long)]
    initial_lexer_fuel: Option<usize>,

    /// Allowed startup lexer cost in thousands (default 200)
    #[arg(long)]
    step_lexer_fuel: Option<usize>,

    /// .ll.json/.schema.json/.lark/.txt file
    #[arg(value_name = "GRAMMAR")]
    file: String,
}

fn main() {
    let opts = CliOptions::parse();
    if opts.dump_tokenizer {
        dump_tokenizer(&opts.tokenizer);
        return;
    }

    // --- Grammar loading ---
    // llguidance supports multiple grammar formats. The file extension determines
    // how the grammar is parsed:
    //   .ll.json      — internal llguidance JSON format (most powerful, used by Guidance library)
    //   .schema.json  — JSON Schema (most common for structured output use cases)
    //   .lark         — Lark-like context-free grammar (for arbitrary grammars)
    //   .txt          — text file turned into a substring-matching regex
    let grammar_file = read_file_to_string(&opts.file);
    let grammar: TopLevelGrammar = if opts.file.ends_with(".ll.json") {
        serde_json::from_str(&grammar_file).expect("Invalid JSON in schema")
    } else if opts.file.ends_with(".schema.json") {
        let val = serde_json::from_str(&grammar_file).expect("Invalid JSON in schema");
        TopLevelGrammar::from_json_schema(val)
    } else if opts.file.ends_with(".lark") {
        TopLevelGrammar::from_lark(grammar_file)
    } else if opts.file.ends_with(".txt") {
        let regex_opts = if opts.split_words {
            json!({
                "substring_words": grammar_file
            })
        } else {
            let lines = grammar_file.split_inclusive('\n').collect::<Vec<_>>();
            json!({
                "substring_chunks": lines
            })
        };
        TopLevelGrammar::from_lark(format!(
            "start: \"foo\" sub\nsub: %regex {}",
            serde_json::to_string(&regex_opts).unwrap()
        ))
    } else {
        panic!("Unknown schema file extension")
    };

    // --- Tokenizer and factory setup ---
    // TokEnv wraps the tokenizer. In production, use the same tokenizer as your LLM.
    // toktrie_hf_downloader downloads the tokenizer from HuggingFace on first use.
    // You can also implement the TokEnv trait yourself (see ByteTokenizerEnv).
    let tok_env: TokEnv = toktrie_hf_downloader::tok_env_from_name(&opts.tokenizer).unwrap();

    // ParserFactory compiles grammars and holds shared state.
    // Create once per tokenizer; it can be shared read-only across threads (via Arc).
    let mut factory = ParserFactory::new_simple(&tok_env).unwrap();

    factory.set_stderr_log_level(opts.log_level);

    // Parser limits control resource usage for complex grammars.
    // Increase lexer_limit if you have very complex regex patterns.
    factory.limits_mut().initial_lexer_fuel *= opts.lexer_limit as u64;
    factory.limits_mut().step_lexer_fuel *= opts.lexer_limit as u64;

    if let Some(initial_lexer_fuel) = opts.initial_lexer_fuel {
        factory.limits_mut().initial_lexer_fuel = initial_lexer_fuel as u64 * 1000;
    }
    if let Some(step_lexer_fuel) = opts.step_lexer_fuel {
        factory.limits_mut().step_lexer_fuel = step_lexer_fuel as u64 * 1000;
    }

    let factory = Arc::new(factory);

    let mut t0 = std::time::Instant::now();

    // create_parser() compiles the grammar for this request.
    // Matcher wraps the parser with a simple server-side API.
    let parser = factory.create_parser(grammar.clone());
    let mut constraint = Matcher::new(parser);

    // --- Mode 1: Mask-only ---
    // When no --input or --rnd is given, just compile the grammar and compute
    // one token mask. Useful for checking that a grammar is valid and measuring
    // compilation time.
    if opts.input.is_none() && opts.rnd.is_none() {
        let _ = constraint.compute_mask().unwrap();
        return;
    }

    // --- Mode 2: Random generation (--rnd N) ---
    // This simulates an LLM by sampling random tokens from the allowed set.
    // The loop mirrors real LLM inference:
    //   1. compute_mask() → get allowed tokens (runs in background in production, ~1ms)
    //   2. sample a token from the mask (replaces LLM logit sampling)
    //   3. consume_token() → advance the parser (very fast, <100μs)
    //   4. consume_ff_tokens() → handle grammar-forced tokens
    //   5. repeat until stopped or max_tokens reached
    if let Some(max_tokens) = opts.rnd {
        let mut ttfm = vec![];
        for rep in 0..opts.repeat {
            let mut rng = SmallRng::seed_from_u64(opts.seed as u64);
            let mut tokens = vec![];
            let mut lens = vec![];
            let trie = tok_env.tok_trie();
            let mut prev_time = std::time::Instant::now();
            let mut times = vec![prev_time.duration_since(t0).as_micros() as u64];
            ttfm.push(times[0]);
            for _ in 0..max_tokens {
                // Compute the token mask — a bitset of allowed tokens.
                // In production, apply this mask to the LLM's logits before sampling.
                let mask = constraint.compute_mask().unwrap();
                // eprintln!("stats: {}", constraint.last_step_stats().unwrap());
                times.push(prev_time.elapsed().as_micros() as u64);
                prev_time = std::time::Instant::now();
                if constraint.is_stopped() {
                    break;
                }
                let mut v = mask.clone();
                // Suppress EOS 80% of the time to make generation run longer.
                // In real usage, the LLM decides when to stop via temperature/logits.
                if rng.random_range(0u32..5) != 0 {
                    v.disallow_token(trie.eos_token());
                }
                // Sample a random token from the allowed set.
                // In production: token = sample(softmax(logits * mask))
                let t = sample_from_vob(&mut rng, &v);
                // Tell the parser which token was sampled.
                constraint.consume_token(t).unwrap();
                tokens.push(t);
                // Consume fast-forward tokens — grammar-forced tokens that bypass sampling.
                // These are appended directly to the output (like speculative decoding at 100%).
                let ff = constraint.consume_ff_tokens();
                tokens.extend_from_slice(&ff);
                lens.push(ff.len());
                if constraint.is_stopped() {
                    break;
                }
            }
            if opts.repeat == 1 {
                eprintln!("Lens: {lens:?}");
                eprintln!("Tokens:\n{}\n", trie.decode_str(&tokens));
            }
            eprintln!("Mask times: {times:?}");
            if rep + 1 == opts.repeat {
                break;
            }

            t0 = std::time::Instant::now();
            constraint = Matcher::new(factory.create_parser(grammar.clone()));
        }
        ttfm.sort();
        eprintln!("Min ttfm: {:?}", ttfm[0]);
        eprintln!("Median ttfm: {:?}", ttfm[ttfm.len() / 2]);
        return;
    }

    // --- Mode 3: Input validation (--input FILE) ---
    // Validates that a pre-tokenized input file conforms to the grammar.
    // This simulates an LLM that always produces the "right" answer — useful for
    // testing grammars against known-good outputs.
    let trie = tok_env.tok_trie();

    let obj_str = read_file_to_string(opts.input.as_ref().unwrap());
    let tokens = tok_env.tokenize(&obj_str);
    eprintln!("Parsing tokens: {}", trie.tokens_dbg(&tokens));

    // constraint.parser.start_without_prompt();
    // constraint.parser.consume_token(tokens[0]).unwrap();

    let mut idx = 0;
    while idx < tokens.len() {
        // Compute the token mask for this position.
        let mask = constraint.compute_mask().unwrap();

        if constraint.is_stopped() {
            break;
        }

        // In real LLM inference, this token comes from sampling.
        // Here we use the pre-tokenized input.
        let sampled_token = tokens[idx];

        // Check that the grammar allows this token.
        let is_allowed = mask.is_allowed(sampled_token);

        let p_stats = constraint.last_step_stats().unwrap();
        if opts.verbose {
            println!(
                "SAMPLE {}: {} {}; stats: {} lex, {} items, {} us",
                idx,
                sampled_token,
                tok_env.tok_trie().token_dbg(sampled_token),
                p_stats.lexer_cost,
                p_stats.all_items,
                p_stats.compute_time_us,
            );
        }

        // Consume the token (tell the parser what was sampled), then check the mask.
        // We call consume_token() before checking is_allowed so that any diagnostics
        // from the parser include context about the failing token.
        constraint.consume_token(sampled_token).unwrap();

        if !is_allowed {
            panic!("Sampled token was not allowed by the mask");
        }

        if constraint.is_stopped() {
            break;
        }

        idx += 1;

        // Get fast-forward tokens — tokens the grammar forces deterministically.
        let splice = constraint.compute_ff_tokens();

        // Verify the fast-forward tokens match what's in our input file.
        // In production, ff_tokens are appended to the output without LLM sampling.
        if tokens[idx..idx + splice.len()] != splice {
            panic!(
                "BAD TEST: ff_tokens mismatch:\n{}\n{}",
                trie.tokens_dbg(&tokens[idx..idx + splice.len()]),
                trie.tokens_dbg(&splice)
            );
        }

        if splice.len() > 1 && opts.verbose {
            println!("FF: {}", trie.tokens_dbg(&splice));
        }

        // Advance the parser past the fast-forward tokens.
        constraint.consume_tokens(&splice).unwrap();
        idx += splice.len();
    }

    // Report why generation stopped.
    // Common reasons: NoExtension (grammar complete), EndOfSentence (EOS sampled),
    // MaxTokensTotal, or an error like LexerTooComplex.
    println!("Stop reason: {:?}", constraint.stop_reason());
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
