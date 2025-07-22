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

pub fn json_test_many(schema: &Value, passing: &[Value], failing: &[Value]) {
    let lark = format!(r#"start: %json {}"#, serde_json::to_string(schema).unwrap());
    for s in passing {
        let s = serde_json::to_string(s).unwrap();
        lark_str_test(&lark, true, &s, false);
    }
    for s in failing {
        let s = serde_json::to_string(s).unwrap();
        lark_str_test(&lark, false, &s, false);
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
