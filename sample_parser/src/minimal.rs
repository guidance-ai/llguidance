/// Minimal example of using llguidance for constrained decoding.
///
/// This demonstrates the core loop that an LLM inference engine uses to enforce
/// grammar constraints on generated output:
///
///   1. **compute_mask()** — ask llguidance which tokens are valid next
///   2. **sample** — pick a token (in production, the LLM samples from logits masked
///      by the result of step 1)
///   3. **consume_token()** — tell llguidance which token was chosen
///   4. **compute_ff_tokens()** — get any "fast-forward" tokens that the grammar
///      forces deterministically (e.g. `{"name":"` in a JSON schema)
///   5. **consume_tokens()** — advance the parser past the fast-forward tokens
///   6. Repeat until the grammar is satisfied or an error occurs
///
/// Instead of a real LLM, this example validates a known-good input file against
/// a grammar, confirming that every token in the input would have been allowed.
///
/// Usage:
///   cargo run --bin minimal -- <grammar_file> <input_file>
///
/// where <grammar_file> is a `.schema.json` (JSON Schema) or `.ll.json` (internal format),
/// and <input_file> contains text that should conform to the grammar.
///
/// Example:
///   cargo run --bin minimal -- data/blog.schema.json data/blog.sample.json
use std::{env, fs::File, io::Read, sync::Arc};

use llguidance::{api::TopLevelGrammar, toktrie::ApproximateTokEnv, Matcher, ParserFactory};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <schema.ll.json> <sample.json>", args[0]);
        std::process::exit(1);
    }

    // --- Step 1: Load the grammar ---
    // TopLevelGrammar is llguidance's grammar representation. It can be built from:
    //   - a JSON Schema (.schema.json) — most common for structured output
    //   - the internal llguidance JSON format (.ll.json)
    //   - a Lark grammar (.lark) — for arbitrary context-free grammars
    //   - a regex string
    let schema_file = read_file_to_string(&args[1]);
    let schema: TopLevelGrammar = if args[1].ends_with(".ll.json") {
        serde_json::from_str(&schema_file).expect("Invalid JSON in schema")
    } else if args[1].ends_with(".schema.json") {
        let val = serde_json::from_str(&schema_file).expect("Invalid JSON in schema");
        TopLevelGrammar::from_json_schema(val)
    } else {
        panic!("Unknown schema file extension")
    };
    let obj_str = read_file_to_string(&args[2]);

    // --- Step 2: Set up the tokenizer environment ---
    // In production, you'd use the same tokenizer as your LLM, e.g.:
    //   let tok_env = toktrie_hf_downloader::tok_env_from_name("meta-llama/Llama-3.1-8B-Instruct")?;
    // Here we use a simple single-byte tokenizer (each byte = one token) to
    // avoid downloading model files. This works for demonstration purposes.
    let tok_env = ApproximateTokEnv::single_byte_env();

    // --- Step 3: Create the ParserFactory ---
    // The factory compiles grammars and holds shared state (tokenizer, caches).
    // Create one factory per model/tokenizer and reuse it across requests.
    let mut factory = ParserFactory::new_simple(&tok_env).unwrap();
    factory.set_stderr_log_level(1); // 1 = warnings only; 2 = verbose

    // The factory can be shared (read-only) across threads after setup.
    let factory = Arc::new(factory);

    // --- Step 4: Tokenize the input ---
    // In a real LLM, tokens come from the model's sampling loop.
    // Here, we pre-tokenize a known-good input to validate against the grammar.
    let tokens = tok_env.tokenize(&obj_str);

    // --- Step 5: Create the parser and Matcher ---
    // create_parser() compiles the grammar for this specific request.
    // Matcher wraps the parser with a simple API for the constrained decoding loop.
    // (There is also a higher-level Constraint API used by the Guidance library.)
    let parser = factory.create_parser(schema);
    let mut constraint = Matcher::new(parser);

    let trie = tok_env.tok_trie();

    eprintln!("Parsing tokens: {}", trie.tokens_dbg(&tokens));

    // --- Step 6: The constrained decoding loop ---
    // This is the core loop that an LLM inference engine would run.
    let mut idx = 0;
    while idx < tokens.len() {
        // 6a. Compute the token mask — a bitset of which tokens are grammatically
        //     valid at this position. In production, this runs in the background
        //     while the GPU computes logits (~1ms for 128k-token vocabulary).
        let mask = constraint.compute_mask().unwrap();

        // 6b. "Sample" a token — in production, the LLM generates logits, the mask
        //     zeros out invalid tokens, and a token is sampled from the distribution.
        //     Here we just take the next token from our pre-tokenized input.
        let sampled_token = tokens[idx];

        // Verify the grammar allows this token (would be guaranteed by masking in production).
        assert!(mask.is_allowed(sampled_token));

        // 6c. Tell the parser which token was sampled.
        constraint.consume_token(sampled_token).unwrap();
        idx += 1;

        // 6d. Get fast-forward tokens — tokens that the grammar forces deterministically.
        //     For example, after a JSON key the grammar may force: ":"
        //     These tokens bypass the LLM entirely (zero entropy / no choice).
        //     In production, they are appended to the output and processed like a
        //     short prefill, similar to speculative decoding with 100% acceptance.
        let splice = constraint.compute_ff_tokens();

        // Verify ff_tokens match our expected input (sanity check on test data).
        if tokens[idx..idx + splice.len()] != splice {
            panic!(
                "BAD TEST: ff_tokens mismatch:\n{}\n{}",
                trie.tokens_dbg(&tokens[idx..idx + splice.len()]),
                trie.tokens_dbg(&splice)
            );
        }

        // 6e. Advance the parser past the fast-forward tokens.
        if splice.len() > 1 {
            println!("FF: {}", trie.tokens_dbg(&splice));
            constraint.consume_tokens(&splice).unwrap();
            idx += splice.len();
        }
    }

    // --- Step 7: Check why generation stopped ---
    // In production, send the stop reason to the user (e.g., "grammar complete"
    // vs "max tokens reached" vs "error").
    println!("Stop reason: {:?}", constraint.stop_reason());
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename).expect("Unable to open file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}
