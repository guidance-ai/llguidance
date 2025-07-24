/*
Per
https://doc.rust-lang.org/book/ch11-03-test-organization.html#submodules-in-integration-tests
we do have an 'old style' mod.rs, so that the test runner doesn't look inside
 */

use anyhow::Result;
use llguidance::{
    api::{GrammarInit, TopLevelGrammar},
    toktrie::bytes::limit_str,
    TokenParser,
};
use sample_parser::*;
use serde_json::Value;

pub fn make_parser(lark: &str, quiet: bool) -> Result<TokenParser> {
    let grm = TopLevelGrammar::from_lark(lark.to_string());
    let mut parser = get_parser_factory().create_parser_from_init(
        GrammarInit::Serialized(grm),
        if quiet { 0 } else { 2 },
        if quiet { 1 } else { 2 },
    )?;
    parser.start_without_prompt();
    Ok(parser)
}

pub fn consume(parser: &mut TokenParser, tok: u32) {
    let n = parser.consume_token(tok).unwrap();
    assert!(n == 0);
}

#[allow(dead_code)]
pub fn lark_ok(lark: &str) {
    if let Err(e) = make_parser(lark, false) {
        panic!("unexpected error: {e}, grm:\n{lark}")
    }
}

pub fn lark_err_test(lark: &str, err: &str) {
    match make_parser(lark, false) {
        Err(e) => {
            let e = format!("{e}");
            if !e.contains(err) {
                panic!("unexpected error: {e}, expecting {err:?}; grm:\n{lark}");
            }
        }
        Ok(_) => panic!("expected error: {err}; grm:\n{lark}"),
    }
}

pub fn json_err_test(schema: &Value, err: &str) {
    lark_err_test(
        &format!(r#"start: %json {}"#, serde_json::to_string(schema).unwrap()),
        err,
    );
}

pub fn lark_str_test(lark: &str, should_accept: bool, input: &str, quiet: bool) {
    let trie = get_tok_env().tok_trie();
    let (final_reject, input) = if let Some(input) = input.strip_prefix("FINAL_REJECT:") {
        (true, input)
    } else {
        (false, input)
    };
    let tokens = get_tok_env().tokenize(input);
    let info = format!(
        "\ninput: {:?}, grm: {:?}",
        limit_str(input, 500),
        limit_str(lark, 100)
    );
    if !quiet {
        println!(
            "\n\ntokens: {}, accpt={}\ngrm:\n{}\n",
            trie.tokens_dbg(&tokens),
            should_accept,
            lark
        );
    }

    // let t0 = std::time::Instant::now();
    let mut p = make_parser(lark, quiet).unwrap();
    // println!("make_parser: {:?}", t0.elapsed());

    for tok in tokens.iter() {
        let m = p.compute_mask().unwrap();
        if m.is_allowed(*tok) {
            consume(&mut p, *tok);
        } else {
            if should_accept {
                panic!("unexpected token: {}{info}", trie.token_dbg(*tok));
            }
            if final_reject {
                panic!(
                    "unexpected token: {}; expecting reject only at the end{info}",
                    trie.token_dbg(*tok)
                );
            }
            return;
        }
    }

    if !final_reject && !should_accept {
        panic!(
            "expected rejection (in the middle; final accept={})",
            p.is_accepting()
        );
    }

    if p.is_accepting() == final_reject {
        if p.is_accepting() {
            panic!("unexpected accept{info}");
        } else {
            panic!("unexpected reject{info}");
        }
    }
}

#[allow(dead_code)]
pub fn lark_str_test_many_ext(quiet: bool, lark: &str, passing: &[&str], failing: &[&str]) {
    for s in passing {
        lark_str_test(lark, true, s, quiet);
    }
    for s in failing {
        lark_str_test(lark, false, s, quiet);
    }
}

pub fn json_schema_check(schema: &Value, json_obj: &Value, expect_valid: bool) {
    /*
       This is a modification of the lark_str_test function, which makes the
       assumption that the input Value completely satifies the schema.

       The subtlety is that a string of tokens might not match a grammar _yet_
       but could with the addition of more tokens. For example, if we're trying
       to construct an integer which is greater than 2, then the string "1"
       is not yet a match, but it could become one if we add more tokens.
       lark_str_test uses the magic 'FINAL_REJECT:' rule to handle this,
       but we can write something a little simpler here.
    */
    let lark_grammar = format!(r#"start: %json {}"#, serde_json::to_string(schema).unwrap());
    let json_string = serde_json::to_string(json_obj).unwrap();

    // Tokenize the string representation of the JSON object
    let tokens = get_tok_env().tokenize(&json_string);

    // Create the parser
    let mut p = make_parser(&lark_grammar, false).unwrap();

    // Work through token by token
    for (i, tok) in tokens.iter().enumerate() {
        // Compute the mask of allowed tokens
        let m = p.compute_mask().unwrap();

        if m.is_allowed(*tok) {
            // Consume the token
            consume(&mut p, *tok);
        } else {
            // Token isn't allowed, so check if we expect this
            let curr_tok_str = get_tok_env().tok_trie().token_dbg(*tok);
            assert!(
                !expect_valid,
                "Unexpected token: {curr_tok_str} at token index {i}",
            );
            // We were expecting this to fail, so return early
            return;
        }
    }

    /*
    Note that p.is_accepting() will be true if the parser has reached a valid end state.
    It does not mean that we couldn't add more tokens and remain valid.
    For example, if we have a schema for any integer, then we can always add more digits
    to a valid integer string.
     */
    assert_eq!(p.is_accepting(), expect_valid, "Final state mismatch");
}

#[allow(dead_code)]
pub fn json_test_many(schema: &Value, passing: &[Value], failing: &[Value]) {
    for s in passing {
        json_schema_check(schema, s, true);
    }
    for s in failing {
        json_schema_check(schema, s, false);
    }
}

#[allow(dead_code)]
pub fn lark_str_test_many(lark: &str, passing: &[&str], failing: &[&str]) {
    lark_str_test_many_ext(false, lark, passing, failing);
}

#[allow(dead_code)]
pub fn lark_str_test_many_quiet(lark: &str, passing: &[&str], failing: &[&str]) {
    lark_str_test_many_ext(true, lark, passing, failing);
}
