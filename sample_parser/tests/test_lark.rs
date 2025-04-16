use anyhow::Result;
use llguidance::{
    api::{GrammarInit, TopLevelGrammar},
    earley::XorShift,
    substring::chunk_into_words,
    toktrie::bytes::limit_str,
    TokenParser,
};
use sample_parser::*;
use serde_json::{json, Value};

fn make_parser(lark: &str, quiet: bool) -> Result<TokenParser> {
    let grm = TopLevelGrammar::from_lark(lark.to_string());
    let mut parser = get_parser_factory().create_parser_from_init(
        GrammarInit::Serialized(grm),
        if quiet { 0 } else { 2 },
        if quiet { 1 } else { 2 },
    )?;
    parser.start_without_prompt();
    Ok(parser)
}

fn consume(parser: &mut TokenParser, tok: u32) {
    let n = parser.consume_token(tok).unwrap();
    assert!(n == 0);
}

fn lark_ok(lark: &str) {
    if let Err(e) = make_parser(lark, false) {
        panic!("unexpected error: {}, grm:\n{}", e, lark)
    }
}

fn lark_err_test(lark: &str, err: &str) {
    match make_parser(lark, false) {
        Err(e) => {
            let e = format!("{}", e);
            if !e.contains(err) {
                panic!(
                    "unexpected error: {}, expecting {:?}; grm:\n{}",
                    e, err, lark
                );
            }
        }
        Ok(_) => panic!("expected error: {}; grm:\n{}", err, lark),
    }
}

fn json_err_test(schema: &Value, err: &str) {
    lark_err_test(
        &format!(r#"start: %json {}"#, serde_json::to_string(schema).unwrap()),
        err,
    );
}

fn lark_str_test(lark: &str, should_accept: bool, input: &str, quiet: bool) {
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

    if p.is_accepting() == final_reject {
        if p.is_accepting() {
            panic!("unexpected accept{info}");
        } else {
            panic!("unexpected reject{info}");
        }
    }
}

fn lark_str_test_many_ext(quiet: bool, lark: &str, passing: &[&str], failing: &[&str]) {
    for s in passing {
        lark_str_test(lark, true, s, quiet);
    }
    for s in failing {
        lark_str_test(lark, false, s, quiet);
    }
}

fn json_test_many(schema: &Value, passing: &[Value], failing: &[Value]) {
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

fn lark_str_test_many(lark: &str, passing: &[&str], failing: &[&str]) {
    lark_str_test_many_ext(false, lark, passing, failing);
}

fn lark_str_test_many_quiet(lark: &str, passing: &[&str], failing: &[&str]) {
    lark_str_test_many_ext(true, lark, passing, failing);
}

#[test]
fn test_dot_unicode() {
    lark_str_test_many(
        r#"start: /.../ "abc" /.../"#,
        &[
            "abcabcabc",
            "aaaabcccc",
            // NOTE: Also ensures that multi-byte characters still count as a single character
            "🔵🟠✅abc❌🟠🔵",
        ],
        &[
            "aaabcccc",
            "aaaaabcccc",
            "FINAL_REJECT:aaaabccc",
            "aaaabccccc",
            "🔵🟠✅❌abc❌✅🟠🔵",
            "🔵🟠abc🟠🔵",
        ],
    );
}

#[test]
fn test_lark_syntax_general() {
    lark_err_test(r#"root: "abc" "def""#, "no start");

    lark_err_test(
        r#"
            start: foo{7,6}
            foo: "a" | "b"
        "#,
        "range end must be >= start",
    );
    lark_err_test(
        r#"
            start: foo{-1,}
            foo: "a" | "b"
        "#,
        "range start must be >= 0",
    );
    lark_err_test(
        r#"
            start: foo{0,-1}
            foo: "a" | "b"
        "#,
        "range end must be >= start",
    );

    lark_err_test(
        r#"
            start: FOO
            FOO: ("a" | "b"){7,6}
        "#,
        "range end must be >= start",
    );
    lark_err_test(
        r#"
            start: FOO
            FOO: ("a" | "b"){-1,}
        "#,
        "range start must be >= 0",
    );
    lark_err_test(
        r#"
            start: FOO
            FOO: ("a" | "b"){0,-1}
        "#,
        "range end must be >= start",
    );

    lark_err_test(
        r#"
            start: FOO
            FOO: "a" | BAR
            BAR: "b" FOO
        "#,
        "circular reference in token",
    );

    lark_ok(
        r#"
            start: foo
            foo: "a" | bar
            bar: "b" foo
        "#,
    );

    lark_err_test(
        r#"
            start: FOO
            BAR: "b"
        "#,
        "unknown name",
    );

    lark_err_test(
        r#"
            start: foo
            bar: "b"
        "#,
        "unknown name",
    );

    lark_err_test(
        r#"
            start: BAR
            BAR: BAZ "a"
        "#,
        r#"unknown name: "BAZ""#,
    );

    lark_ok(
        r#"
            %import common.INT
            start: INT
        "#,
    );
    lark_err_test(
        r#"
            %import common.BLAH
            start: BLAH
        "#,
        "Unknown common",
    );

    lark_err_test(r#" start: /[abc/ "#, "invalid regex");
    lark_ok(r#" start: /[abc]/ "#);
    lark_err_test(r#" start: /[abc]/l "#, "l-flag is not supported");

    lark_err_test(
        r#"
            start: FOO
            FOO: @qux
        "#,
        "cannot be used in terminals",
    );
    lark_err_test(
        r#"
            start: FOO
            FOO: %json { }
        "#,
        "cannot be used in terminals",
    );
    lark_err_test(
        r#"
            start: FOO
            FOO: <[1234]>
        "#,
        "cannot be used in terminals",
    );
    lark_err_test(
        r#"
            start: FOO
            FOO: <|assistant|>
        "#,
        "cannot be used in terminals",
    );
    lark_err_test(
        r#"
            start: "A" | <|foobarbaz|>
        "#,
        "unknown special token",
    );

    lark_err_test(
        r#" start: "ab".."c" "#,
        "range start must be a single character",
    );
    lark_err_test(
        r#" start: "a".."cd" "#,
        "range end must be a single character",
    );
    lark_err_test(r#"  start: "d".."a" "#, "invalid range order");

    lark_err_test(r#"start: <[100-200-300]>"#, "invalid token range");
    lark_ok(r#"start: <[100-200,300-4002]>"#);
    lark_err_test(r#"start: <[100-200,100-200-300]>"#, "invalid token range");
    lark_err_test(r#"start: <[,]>"#, "empty token range");
    lark_err_test(r#"start: <[200-100]>"#, "invalid token range");
    lark_err_test(r#"start: <[200 - 100]>"#, "lexer error");

    lark_err_test(
        r#"
            start: foo
            foo: "a" | "b"
            foo: "c"
        "#,
        "duplicate rule",
    );
    lark_err_test(
        r#"
            start: FOO
            FOO: "a" | "b"
            FOO: "c"
        "#,
        "duplicate token",
    );
}

#[test]
fn test_lark_syntax_perc() {
    lark_err_test(r#"start: %json {"#, "EOF while parsing an object");
    lark_err_test(r#"start: %json { foo"#, "key must be a string");
    lark_err_test(r#"start: %json []"#, "failed to compile JSON schema");
    lark_err_test(
        r#"start: %json { "if": {} }"#,
        "failed to compile JSON schema",
    );

    lark_err_test(
        r#"
            %llguidance { "no_forcing": "yadda-dada"}
            start: "a" | "b"
        "#,
        "failed to parse %llguidance declaration",
    );

    lark_ok(r#" start: %regex { "substring_words": "foo bar" } "#);
    lark_ok(r#" start: %regex { "substring_chars": "foo bar" } "#);
    lark_ok(r#" start: %regex { "substring_chunks": ["foo", "bar"] } "#);

    lark_err_test(
        r#" start: %regex { "substring_words": true } "#,
        "failed to parse %regex",
    );

    lark_err_test(r#" start: %regex { "foobar": true } "#, "unknown field");

    lark_err_test(
        r#" start: %regex { "substring_words": "aa", "substring_chars": "bb" } "#,
        "only one field can be set on %regex",
    );

    lark_err_test(r#" start: %regex {  } "#, "no fields set on %regex");
}

#[test]
fn test_lark_syntax_attributes() {
    lark_ok(
        r#" start: foo
            foo[stop=""]: /.*/ "#,
    );

    lark_ok(
        r#" start: foo
            foo[stop="",max_tokens=12]: /.*/ "#,
    );

    lark_ok(
        r#" start: foo
            foo[capture,stop=""]: /.*/ "#,
    );

    lark_ok(
        r#" start: foo
            foo[capture="bar" , stop=""]: /.*/ "#,
    );

    lark_ok(
        r#" start: foo
            foo[stop = "foobar"]: /.*/ "#,
    );

    lark_ok(
        r#" start: foo
            foo[stop = /foobar/]: /.*/ "#,
    );

    lark_ok(
        r#" start: foo
            foo[stop = STOP]: /.*/
            STOP: "foobar"
        "#,
    );

    lark_err_test(
        r#" start: foo
            foo[foobar=12]: /.*/ "#,
        "Unknown attribute",
    );

    lark_err_test(
        r#" start: foo
            foo[stop=""="foo"]: /.*/ "#,
        "Expected token",
    );

    lark_err_test(
        r#" start: foo
            foo[max_tokens="foo"]: /.*/ "#,
        "Expected token",
    );
}

#[test]
fn test_repeat() {
    lark_str_test_many(
        r#"start:  ab{3,5}
           ab:  "a" | "b"
        "#,
        &["aba", "abaa", "aaaaa", "aabaa"],
        &["FINAL_REJECT:aa", "FINAL_REJECT:ab", "aaaaaa"],
    );

    lark_str_test_many(
        r#"start:  ab{3,}
           ab:  "a" | "b"
        "#,
        &["aba", "abaa", "aaaaa", "aabaa", "aaaaaa"],
        &["FINAL_REJECT:aa", "FINAL_REJECT:ab"],
    );

    lark_str_test_many(
        r#"start:  ab{,5}
           ab:  "a" | "b"
        "#,
        &["", "aa", "b", "aba", "abaa", "aaaaa", "aabaa"],
        &["aaaaaa"],
    );
}

#[test]
fn test_lexeme_substring_general() {
    for grm in &[
        r#" start: "A" %regex { "substring_words": "foo bar baz" } "B" "#,
        r#" start: SUB
            SUB: "A" %regex { "substring_words": "foo bar baz" } "B" "#,
    ] {
        lark_str_test_many(
            grm,
            &[
                "AfooB",
                "Abar bazB",
                "AbazB",
                "Afoo bar bazB",
                "Afoo bar B",
                "A bar bazB",
                "AB",
            ],
            &["FINAL_REJECT:Afoo bar baz", "AfoB"],
        );
    }

    lark_str_test_many(
        r#" start: "A" %regex { "substring_chunks": ["foo", " bar", " baz"] } "B" "#,
        &[
            "AfooB",
            "A bar bazB",
            "A bazB",
            "Afoo bar bazB",
            "Afoo barB",
            "AB",
            "A bar bazB",
        ],
        &["FINAL_REJECT:Afoo bar baz", "AfoB"],
    );
}

#[test]
fn test_lexeme_substring_chars_ascii() {
    lark_str_test_many(
        r#"start: %regex { "substring_chars": "The quick brown fox jumps over the lazy dog." }"#,
        &[
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox",
            "he quick brow",
            "fox jump",
            "dog.",
        ],
        &["brown fx"],
    );
}

#[test]
fn test_lexeme_substring_chars_unicode() {
    lark_str_test_many(
        r#"start: %regex { "substring_chars": "빠른 갈색 여우가 게으른 개를 뛰어넘었다." }"#,
        &[
            "빠른 갈색 여우가 게으른 개를 뛰어넘었다.",
            "빠른 갈색 여우가 게으른",
            "른 갈색 여우",
            "여우가 게으",
            "뛰어넘었다.",
        ],
        &["갈색 여가"],
    );
}

#[test]
fn test_lexeme_substring_words_ascii() {
    lark_str_test_many(
        r#"start: %regex { "substring_words": "The quick brown fox jumps over the lazy dog." }"#,
        &[
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox",
            "dog.",
        ],
        &["he quick brow", "FINAL_REJECT:fox jump", "brown fx"],
    );
}

#[test]
fn test_lexeme_substring_words_unicode() {
    lark_str_test_many(
        r#"start: %regex { "substring_words": "빠른 갈색 여우가 게으른 개를 뛰어넘었다." }"#,
        &[
            "빠른 갈색 여우가 게으른 개를 뛰어넘었다.",
            "빠른 갈색 여우가 게으른",
            "뛰어넘었다.",
        ],
        &["른 갈색 여우", "FINAL_REJECT:여우가 게으", "갈색 여가"],
    );
}

fn gen_words(seed: u32, num_words: usize) -> String {
    let letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.";
    let mut rnd = XorShift::new(seed + 1);
    let mut words = vec![];
    let num_words = rnd.from_range((num_words / 2)..num_words);
    for _ in 0..num_words {
        let mut word = String::new();
        let len = rnd.from_range(1..15);
        for _ in 0..len {
            let idx = rnd.from_range(0..letters.len());
            word.push(letters.as_bytes()[idx] as char);
        }
        words.push(word);
    }
    words.join(" ")
}

fn quote_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

#[test]
fn test_large_select() {
    // it's kind of slow in non-release mode
    let num_words = if cfg!(debug_assertions) { 100 } else { 500 };
    let num_opt = if cfg!(debug_assertions) { 100 } else { 1500 };

    let t0 = std::time::Instant::now();
    let mut grm_sz = 0;

    for start in &["start: OPTS\nOPTS: ", "start: opts\nopts: "] {
        let mut grm_head = start.to_string();
        let mut grm_tail = "".to_string();
        let options = (0..num_opt)
            .map(|i| gen_words(i, num_words))
            .collect::<Vec<_>>();
        for (i, opt) in options.iter().enumerate() {
            grm_head.push_str(&format!("OPT{} | ", i));
            grm_tail.push_str(&format!("OPT{}: {}\n", i, quote_str(opt)));
        }
        grm_head.push_str(" \"\"\n");
        let grm = format!("{}{}", grm_head, grm_tail);
        grm_sz = grm.len();

        lark_str_test_many_quiet(
            &grm,
            //&options.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &[options[2].as_str(), options[7].as_str()],
            &["something that is unlikely to be in the options"],
        );
    }

    println!("large_select: {:?}; grm={}kB", t0.elapsed(), grm_sz / 1024);
}

#[test]
fn test_large_substring_words() {
    let words_str = gen_words(1, 5000);
    let words = chunk_into_words(&words_str);
    let grm = format!(
        "start: %regex {{ \"substring_words\": {} }}",
        quote_str(&words_str)
    );

    let mtch = words[50..100].to_vec().join("");
    let no_mtch = format!("{}{}", mtch, "XXX");
    lark_str_test_many_quiet(&grm, &[&mtch], &[&no_mtch]);
}

#[test]
fn test_large_substring_chars() {
    let chars = gen_words(2, 15000)[..10000].to_string();
    let grm = format!(
        "start: %regex {{ \"substring_chars\": {} }}",
        quote_str(&chars)
    );
    let mtch = chars[50..100].to_string();
    let no_mtch = format!("{}{}", mtch, "XXX");
    lark_str_test_many_quiet(&grm, &[&mtch], &[&no_mtch]);
}

#[test]
fn test_lexer_amb() {
    lark_str_test_many(
        r#"start: "'foo'" /a+/ | STRING /b+/
           STRING: /'[^']*'/
        "#,
        &["'foo'a", "'foo'aaa", "'bar'b", "'bar'bbb", "'foo'bb"],
        &["'bar'a", "'bar'c"],
    );
}

#[test]
fn test_edits() {
    let grm = r#"
start: ( step "\n" )* step ( "\n" final_comments )?
step: plan "\n" a_file
plan[lazy]: /((.|\n)*\n)?```/
final_comments: /[^`]*/

replace: "=======\n" repl_inner " REPLACE\n```"
repl_inner[lazy]: /(.|\n)*\n>>>>>>>/
SEARCH: "\n<<<<<<< SEARCH\n"

// "generated"

file_0: "gbnf_to_lark.py" SEARCH FILE_0 replace
a_file: file_0

FILE_0: %regex {
  "substring_chunks": [
    "foo\n",
    "bar\n",
    "baz\n",
    "line\n",
    "line\n"
  ]
}
"#;

    fn repl_block(filename: &str, src: &str, dst: &str) -> String {
        format!(
            "```\n{}\n<<<<<<< SEARCH\n{}\n=======\n{}\n>>>>>>> REPLACE\n```",
            filename,
            src.trim_end_matches("\n"),
            dst.trim_end_matches("\n")
        )
    }

    let filename = "gbnf_to_lark.py";
    let repl = repl_block(filename, "foo\nbar", "qux");

    lark_str_test_many(
        grm,
        &[
            &repl,
            &format!("{}\n", repl),
            &format!("{}\n\n", repl),
            &format!("Some text\n{}", repl),
            &format!("Some text\nMore\n{}", repl),
            &format!("Some text\nMore\n{}\n", repl),
            &format!("Some text\nMore\n{}\nAnd then some", repl),
            &format!("Some text\nMore\n{}\nAnd then some\n", repl),
        ],
        &[
            "FINAL_REJECT:Some text\nSome more\n",
            "FINAL_REJECT:Some text\nSome more",
            "Foo\n```\nbar",
            "FINAL_REJECT:Foo\n```\ngbnf",
            &repl_block(filename, "fooz\nbar", "quux"),
        ],
    );
}

#[test]
fn test_json_dw_pattern() {
    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "^\\d$" }
        "#,
        &["\"1\"", "\"2\""],
        &[
            "1", "\"12\"", "\"a\"", "\"\"",  // simple stuff
            "\"১\"", // unicode should not match
        ],
    );

    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "^\\w$" }
        "#,
        &["\"1\"", "\"2\"", "\"a\"", "\"A\""],
        &[
            "1", "\"12\"", "\"\"", // simple stuff
            "\"১\"", "\"ł\"", // unicode should not match
        ],
    );

    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "^\\s$" }
        "#,
        &[
            // regular escapes:
            "\" \"",
            "\"\\t\"",
            "\"\\n\"",
            "\"\\r\"",
            // unicode escapes:
            "\"\\u0009\"",
            "\"\\u000A\"",
            "\"\\u000B\"",
            "\"\\u000C\"",
            // unicode whitespace:
            "\"\u{00A0}\"",
            "\"\u{2000}\"",
            "\"\u{2008}\"",
        ],
        &[
            // simple stuff
            "1",
            "\"12\"",
            "\"\"",
            // non-whitespace unicode
            "\"১\"",
            "\"ł\"",
            // we do not allow \uXXXX outside of \u0000-0x001F
            "\"\\u00A0\"",
        ],
    );
}

#[test]
fn test_json_anchoring() {
    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "[ab]" }
        "#,
        &["\"a\"", "\"foobar\"", "\"\\nb\\n\\n\""],
        &["1", "\"12\"", "\"\"", "\"১\"", "\"ł\""],
    );

    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "^foo" }
        "#,
        &["\"foo\"", "\"foobar\""],
        &["1", "\"afoo\"", "\"afooa\""],
    );
    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "foo$" }
        "#,
        &["\"foo\"", "\"barfoo\""],
        &["1", "\"fooa\"", "\"afooa\""],
    );
    lark_str_test_many(
        r#"
            start: %json { "type": "string", "pattern": "^foo$" }
        "#,
        &["\"foo\""],
        &["1", "\"fooa\"", "\"afoo\"", "\"afooa\""],
    );
}

#[test]
fn test_nested_lark() {
    lark_str_test_many(
        r#"
            start: /[ab]+/ foobar
            foobar: %lark {
                start: "foo" | "Bar"
            }
        "#,
        &["afoo", "abfoo", "aaaaaaBar"],
        &["FINAL_REJECT:a", "afooa"],
    );
}

#[test]
fn test_large_real_substring() {
    let data = include_str!("../data/ulysses.md");
    // 240k is the limit for 1M fuel
    let data = data[..200_000].to_string();
    let grm = format!(
        r#"
            start: %regex {{ "substring_words": {} }}
        "#,
        quote_str(&data)
    );
    let mtch = data.split_inclusive(' ').collect::<Vec<_>>()[50..250]
        .to_vec()
        .join("");
    let no_mtch = format!("{}{}", mtch, "XXX");
    lark_str_test_many_quiet(&grm, &[&mtch], &[&no_mtch]);
}

#[test]
fn test_json_pattern_properties() {
    json_err_test(
        &json!({
            "type": "object",
            "patternProperties": {
                "^fo": { "type": "integer" },
                "^foo": { "type": "number" },
            },
        }),
        "are not disjoint",
    );

    json_err_test(
        &json!({
            "type": "object",
            "patternProperties": {
                "foo": { "type": "integer" },
                "bar": { "type": "number" },
            },
        }),
        "are not disjoint",
    );

    json_err_test(
        &json!({
            "allOf": [
                {
                    "type": "object",
                    "patternProperties": {
                        "^fo": { "type": "integer" },
                    },
                },
                {
                    "type": "object",
                    "patternProperties": {
                        "^foo": { "type": "number" },
                    },
                },
            ],
        }),
        "are not disjoint",
    );

    json_err_test(
        &json!({
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "foo": { "type": "string" },
                    },
                    "required": ["foo"],
                },
                {
                    "type": "object",
                    "patternProperties": {
                        "^f": { "type": "number" },
                    },
                },
            ],
        }),
        "required property 'foo' is unsatisfiable",
    );

    json_test_many(
        &json!({
            "type": "object",
            "properties": {
                "foo": { "type": "string" },
            },
            "patternProperties": {
                "^foo": { "type": "integer" },
                "^bar": { "type": "array" },
            },
            "additionalProperties": {
                "type": "boolean",
            },
        }),
        &[
            json!({}),
            json!({
                "foo": "bar"
            }),
            json!({
                "foo": "bar",
                "foo1": 123,
                "bar": [],
                "qux": true,
                "foo2": 456,
                "bar1": [],
                "mux": false,
            }),
            json!({
                "bar": []
            }),
            json!({
                "muxxx": false
            }),
        ],
        &[
            json!({
                "foo": 123
            }),
            json!({
                "foo1": "blah"
            }),
            json!({
                "foo1": true
            }),
            json!({
                "bar11": true
            }),
        ],
    );

    json_test_many(
        &json!({
            "type": "object",
            "properties": {
                "foo": { "type": "string" },
            },
            "patternProperties": {
                "^foo": { "type": "integer" },
                "^bar": { "type": "array" },
            },
            "additionalProperties": {
                "type": "boolean",
            },
            "required": ["foo", "mux", "foo1", "bar1"],
        }),
        &[
            json!({
                "foo": "bar",
                "mux": false,
                "foo1": 123,
                "bar1": [],
            }),
            json!({
                "foo": "bar",
                "mux": false,
                "foo1": 123,
                "bar1": [],
                "blah": true
            }),
        ],
        &[
            json!({
                "foo": "bar",
                "mux": false,
                "bar1": [],
                "foo1": 123,
            }),
            json!({
                "foo": "bar",
                "mux": "blah",
                "foo1": 123,
                "bar1": [],
            }),
            json!({
                "foo": "bar",
                "mux": false,
                "foo1": "aaa",
                "bar1": [],
            }),
        ],
    );
}
