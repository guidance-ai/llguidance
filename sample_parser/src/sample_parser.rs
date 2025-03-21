use clap::Parser;
use std::{fs::File, hint::black_box, io::Read, vec};

use llguidance::{
    api::{ParserLimits, TopLevelGrammar},
    earley::{SlicedBiasComputer, XorShift},
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, TokenParser,
};
use serde_json::json;

fn dump_tokenizer(name: &str) {
    let btok = toktrie_hf_downloader::byte_tokenizer_from_name(name).unwrap();
    let vecs = btok.token_bytes();
    for v in vecs.iter() {
        let v: String = v
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join("");
        println!("{}", v);
    }
}

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

    // you can implement TokEnv yourself, if you have the tokenizer
    // see the ByteTokenizerEnv for an example
    let tok_env: TokEnv = toktrie_hf_downloader::tok_env_from_name(&opts.tokenizer).unwrap();

    // set to 2 for more output; 1 is warnings only
    let stderr_log_level = opts.log_level;

    // typically set to 2, to send info-level output to the user
    let buffer_log_level = 2;

    let mut t0 = std::time::Instant::now();

    let mut limits = ParserLimits::default();
    limits.initial_lexer_fuel *= opts.lexer_limit as u64;
    limits.step_lexer_fuel *= opts.lexer_limit as u64;

    let infer_caps = InferenceCapabilities {
        ff_tokens: true,  // can the engine append multiple tokens?
        backtrack: false, // can the engine remove generated tokens?

        conditional_ff_tokens: false, // not used
        fork: false,                  // not used
    };

    let parser = TokenParser::from_grammar(
        tok_env.clone(),
        grammar.clone(),
        llguidance::Logger::new(buffer_log_level, stderr_log_level),
        infer_caps.clone(),
        limits.clone(),
        SlicedBiasComputer::general_slices(),
    )
    .unwrap();
    let mut constraint = Constraint::new(parser);

    // enable sending parser results back via the logs (constraint.flush_logs())
    constraint.log_json_progress = true;

    if opts.input.is_none() && opts.rnd.is_none() {
        constraint.start_without_prompt();
        let _ = constraint.compute_mask().unwrap();
        return;
    }

    if let Some(max_tokens) = opts.rnd {
        let mut ttfm = vec![];
        for rep in 0..opts.repeat {
            constraint.start_without_prompt();
            let mut rng = XorShift::new(opts.seed);
            let mut tokens = vec![];
            let mut lens = vec![];
            let trie = tok_env.tok_trie();
            let mut prev_time = std::time::Instant::now();
            let mut times = vec![prev_time.duration_since(t0).as_micros() as u64];
            ttfm.push(times[0]);
            for _ in 0..max_tokens {
                let r = constraint.compute_mask().unwrap();
                times.push(prev_time.elapsed().as_micros() as u64);
                prev_time = std::time::Instant::now();
                if r.is_stop() {
                    break;
                }
                let mut v = r.sample_mask.clone().unwrap();
                // mostly disallow eos to make it run longer
                if !rng.one_in(5) {
                    v.disallow_token(trie.eos_token());
                }
                let t = rng.sample_from_vob(&v);
                let r = constraint.commit_token(Some(t)).unwrap();
                assert_eq!(r.backtrack, 0);
                tokens.extend_from_slice(&r.ff_tokens);
                lens.push(r.ff_tokens.len());
            }
            if opts.repeat == 1 {
                eprintln!("Lens: {:?}", lens);
                eprintln!("Tokens:\n{}\n", trie.decode_str(&tokens));
            }
            eprintln!("Mask times: {:?}", times);
            if rep + 1 == opts.repeat {
                break;
            }

            t0 = std::time::Instant::now();
            let parser = TokenParser::from_grammar(
                tok_env.clone(),
                grammar.clone(),
                llguidance::Logger::new(buffer_log_level, stderr_log_level),
                infer_caps.clone(),
                limits.clone(),
                SlicedBiasComputer::general_slices(),
            )
            .unwrap();
            constraint = Constraint::new(parser);
        }
        ttfm.sort();
        eprintln!("Min ttfm: {:?}", ttfm[0]);
        eprintln!("Median ttfm: {:?}", ttfm[ttfm.len() / 2]);
        return;
    }

    let trie = tok_env.tok_trie();

    let obj_str = read_file_to_string(opts.input.as_ref().unwrap());
    let tokens = tok_env.tokenize(&obj_str);
    eprintln!("Parsing tokens: {}", trie.tokens_dbg(&tokens));

    // constraint.parser.start_without_prompt();
    // constraint.parser.consume_token(tokens[0]).unwrap();

    let mut idx = 0;
    while idx < tokens.len() {
        let res = constraint.compute_mask().unwrap();

        if res.is_stop() {
            // stop sequence
            break;
        }

        let mut is_allowed = true;

        let sampled_token = if let Some(mask) = &res.sample_mask {
            // Simulate sampling - it should use the mask and temperature
            let sampled_token = tokens[idx];
            is_allowed = mask.is_allowed(sampled_token);
            black_box(mask);
            black_box(constraint.temperature);

            let p_stats = constraint.parser.last_step_stats();
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
            Some(sampled_token)
        } else {
            // sampling not required
            if opts.verbose {
                println!("NO SAMPLE {}", idx);
            }
            None
        };

        // run commit_token() before checking the mask - it produces more diagnostics that way
        let splice = constraint.commit_token(sampled_token).unwrap();

        if !is_allowed {
            panic!("Sampled token was not allowed by the mask");
        }

        if splice.stop {
            // stop sequence
            break;
        }

        assert!(splice.backtrack == 0); // we didn't allow backtracking in InferenceCaps

        // The splice contains the tokens (possibly more than one since we enabled ff_tokens
        // in InferenceCaps) that the parser wants to append to the output.

        // if this fails, our test data is broken
        if tokens[idx..idx + splice.ff_tokens.len()] != splice.ff_tokens {
            panic!(
                "BAD TEST: ff_tokens mismatch:\n{}\n{}",
                trie.tokens_dbg(&tokens[idx..idx + splice.ff_tokens.len()]),
                trie.tokens_dbg(&splice.ff_tokens)
            );
        }

        if splice.ff_tokens.len() > 1 && opts.verbose {
            println!("FF: {}", trie.tokens_dbg(&splice.ff_tokens));
        }

        idx += splice.ff_tokens.len();

        // send output to the user
        send_output(&constraint.flush_logs());
    }

    // flush any output
    send_output(&constraint.flush_logs());
    // the stop reason should be likely also sent to the user
    println!("Stop reason: {:?}", constraint.parser.stop_reason());

    println!("Max step stats: {:?}", constraint.parser.max_step_stats());
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}

fn send_output(user_output: &str) {
    // enable if you want to see the output
    if false {
        println!("{}", user_output);
    }
}
