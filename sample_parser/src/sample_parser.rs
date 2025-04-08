use clap::Parser;
use std::{fs::File, io::Read, sync::Arc, vec};

use llguidance::{api::TopLevelGrammar, earley::XorShift, toktrie::TokEnv, Matcher, ParserFactory};
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

    let mut factory = ParserFactory::new_simple(&tok_env).unwrap();

    factory.set_stderr_log_level(opts.log_level);

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

    let parser = factory.create_parser(grammar.clone());
    let mut constraint = Matcher::new(parser);

    if opts.input.is_none() && opts.rnd.is_none() {
        let _ = constraint.compute_mask().unwrap();
        return;
    }

    if let Some(max_tokens) = opts.rnd {
        let mut ttfm = vec![];
        for rep in 0..opts.repeat {
            let mut rng = XorShift::new(opts.seed);
            let mut tokens = vec![];
            let mut lens = vec![];
            let trie = tok_env.tok_trie();
            let mut prev_time = std::time::Instant::now();
            let mut times = vec![prev_time.duration_since(t0).as_micros() as u64];
            ttfm.push(times[0]);
            for _ in 0..max_tokens {
                let mask = constraint.compute_mask().unwrap();
                // eprintln!("stats: {}", constraint.last_step_stats().unwrap());
                times.push(prev_time.elapsed().as_micros() as u64);
                prev_time = std::time::Instant::now();
                if constraint.is_stopped() {
                    break;
                }
                let mut v = mask.clone();
                // mostly disallow eos to make it run longer
                if !rng.one_in(5) {
                    v.disallow_token(trie.eos_token());
                }
                let t = rng.sample_from_vob(&v);
                constraint.consume_token(t).unwrap();
                tokens.push(t);
                let ff = constraint.consume_ff_tokens();
                tokens.extend_from_slice(&ff);
                lens.push(ff.len());
                if constraint.is_stopped() {
                    break;
                }
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
            constraint = Matcher::new(factory.create_parser(grammar.clone()));
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
        let mask = constraint.compute_mask().unwrap();

        if constraint.is_stopped() {
            // stop sequence
            break;
        }

        // Simulate sampling - it should use the mask and temperature
        let sampled_token = tokens[idx];

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

        // run commit_token() before checking the mask - it produces more diagnostics that way
        constraint.consume_token(sampled_token).unwrap();

        if !is_allowed {
            panic!("Sampled token was not allowed by the mask");
        }

        if constraint.is_stopped() {
            // stop sequence
            break;
        }

        idx += 1;

        let splice = constraint.compute_ff_tokens();

        // The splice contains the tokens (possibly more than one since we enabled ff_tokens
        // in InferenceCaps) that the parser wants to append to the output.

        // if this fails, our test data is broken
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

        constraint.consume_tokens(&splice).unwrap();
        idx += splice.len();
    }

    // the stop reason should be likely also sent to the user
    println!("Stop reason: {:?}", constraint.stop_reason());
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
