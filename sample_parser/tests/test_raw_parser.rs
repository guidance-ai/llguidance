use lazy_static::lazy_static;
use llguidance::{
    api::TopLevelGrammar,
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, TokEnv},
    ParserFactory, TokenParser,
};
use serde_json::{json, Value};

lazy_static! {
    static ref PARSER_FACTORY_PHI: ParserFactory = {
        let env = sample_parser::get_tok_env();
        let mut fact = ParserFactory::new(
            env,
            InferenceCapabilities {
                ff_tokens: false,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            &SlicedBiasComputer::general_slices(),
        )
        .unwrap();
        fact.set_stderr_log_level(2);
        fact.set_buffer_log_level(0);
        fact
    };
}

lazy_static! {
    static ref PARSER_FACTORY: ParserFactory = {
        let env =
            toktrie_hf_downloader::tok_env_from_name("unsloth/Meta-Llama-3.1-8B-Instruct").unwrap();
        let mut fact = ParserFactory::new(
            &env,
            InferenceCapabilities {
                ff_tokens: false,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            &SlicedBiasComputer::general_slices(),
        )
        .unwrap();
        fact.set_stderr_log_level(2);
        fact.set_buffer_log_level(0);
        fact
    };
}

fn make_parser(lark: &str) -> TokenParser {
    let grm = TopLevelGrammar::from_lark(lark.to_string());
    let mut parser = PARSER_FACTORY.create_parser(grm).unwrap();
    parser.start_without_prompt();
    parser
}

fn consume(parser: &mut TokenParser, tok: u32) {
    let n = parser.consume_token(tok).unwrap();
    assert!(n == 0);
}

#[test]
fn test_ff_tokens() {
    let lark = r#"
        start: <[1111]> <[311]> ( <[366]> | "s" ) <[311]> <[1111]>
    "#;
    let grm = TopLevelGrammar::from_lark(lark.to_string());
    let mut parser = PARSER_FACTORY_PHI.create_parser(grm).unwrap();
    parser.start_without_prompt();

    let t = parser.compute_ff_tokens();
    assert_eq!(t, vec![1111, 311]);
    let n = parser.validate_tokens_raw(&t).unwrap();
    assert_eq!(n, 2);
    consume(&mut parser, 1111);
    consume(&mut parser, 311);

    let n = parser.validate_tokens_raw(&[366, 311, 1111]).unwrap();
    assert_eq!(n, 3);

    let n = parser.validate_tokens_raw(&[29879, 311, 1111]).unwrap();
    assert_eq!(n, 3);

    consume(&mut parser, 29879);

    let t = parser.compute_ff_tokens();
    assert_eq!(t, vec![311, 1111]);
    let n = parser.validate_tokens_raw(&t).unwrap();
    assert_eq!(n, 2);
}

fn get_tok_env() -> &'static TokEnv {
    PARSER_FACTORY.tok_env()
}

fn json_fwd_test(schema: Value, obj: Value) {
    let mut p = make_parser(&format!(
        "start: %json {}",
        serde_json::to_string(&schema).unwrap()
    ));

    let trie = get_tok_env().tok_trie();
    let tokens = get_tok_env().tokenize(serde_json::to_string(&obj).unwrap().as_str());
    println!("\n\ntokens: {}\n", trie.tokens_dbg(&tokens));

    for tok in tokens.iter() {
        let m = p.compute_mask().unwrap();
        assert!(m.is_allowed(*tok));
        consume(&mut p, *tok);
    }
}

#[test]
fn test_ff_json1() {
    json_fwd_test(
        json!({
            "type": "object",
            "properties": {
                "someLongPropertyName": {
                    "type": "string"
                }
            },
            "additionalProperties": false
        }),
        json!({
            "someLongPropertyName": "123"
        }),
    );
}

#[test]
fn test_ff_json2() {
    json_fwd_test(
        json!({
            "additionalProperties": false,
            "properties": {
              "path": {
                "pattern": "^/contributions",
                "type": "string"
              }
            },
            "required": ["path"],
            "type": "object"
        }
        ),
        json!({"path": "/contributions/foo"}),
    );
}

#[test]
fn test_ff_json3() {
    json_fwd_test(
        json!({
            "additionalProperties": false,
            "properties": {
              "location": { "type": "string" },
              "retries": { "type": "number" },
              "retrieveDate": { "type": "string" },
              "retryInterval": { "type": "number" }
            },
            "required": [ "location", "retrieveDate" ],
            "type": "object"
        }),
        json!({
            "location": "https://example.com/firmware.bin",
            "retrieveDate": "2022-01-01T12:00:00Z",
            "retryInterval": 300
        }),
    );
}

#[test]
fn test_ff_json4() {
    let schema = json!({
        "anyOf":[{
            "type": "object",
            "properties": {
                "foo": { "type": "number" }
            },
            // "required": ["foo"], -> with required it passes
            "additionalProperties": { "type": "string" },
        }, {
            "type": "object",
            "properties": {
                "bar": { "type": "number" }
            },
            "additionalProperties": false,
        }]
    });

    json_fwd_test(schema.clone(), json!({ "foo": 123, "baz": "hello" }));
    json_fwd_test(schema.clone(), json!({ "bar": 123 }));
}
