use llguidance::{
    api::TopLevelGrammar,
    grammar_from_tools,
    toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie},
    Matcher, ParserFactory,
};
use rstest::rstest;
use serde_json::{json, Value};
use std::sync::Arc;

fn factory(markers: &[&str]) -> ParserFactory {
    let mut words: Vec<_> = (0..=255).map(|byte| vec![byte]).collect();
    words.extend(markers.iter().map(|name| {
        std::iter::once(TokTrie::SPECIAL_TOKEN_MARKER)
            .chain(name.bytes())
            .collect()
    }));
    words.push(b"\xff<eos>".to_vec());
    let info = TokRxInfo::new(words.len() as u32, words.len() as u32 - 1);
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    let mut factory = ParserFactory::new_simple(&env).unwrap();
    factory.quiet();
    factory
}

fn accepts(factory: &ParserFactory, grammar: &str, text: &str, special: bool) -> bool {
    let env = factory.tok_env();
    let tokens = if special {
        env.tok_trie()
            .tokenize_with_special(text, |text| env.tokenize(text))
    } else {
        env.tokenize(text)
    };
    let parser = factory
        .create_parser(TopLevelGrammar::from_lark(grammar.to_string()))
        .unwrap();
    let mut matcher = Matcher::new(Ok(parser));
    for token in tokens {
        if !matcher.compute_mask().unwrap().is_allowed(token) {
            return false;
        }
        matcher.consume_token(token).unwrap();
    }
    matcher.is_accepting().unwrap() && matcher.compute_mask().unwrap().is_allowed(env.eos_token())
}

fn tool(name: &str, strict: bool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "strict": strict,
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"]
            }
        }
    })
}

fn call(name: &str, arguments: &str) -> String {
    format!("<tool_call>\n<function={name}>\n{arguments}\n</function>\n</tool_call>")
}

#[test]
fn strict_tool_enforces_argument_schema() {
    let grammar = grammar_from_tools(
        &json!([tool("count", true)]),
        "qwen3_coder",
        false,
        Some(&[]),
    )
    .unwrap();
    let factory = factory(&[]);
    let valid = "<parameter=count>\n3\n</parameter>";
    assert!(accepts(&factory, &grammar, &call("count", valid), false));
    for invalid in [
        "<parameter=wrong>3</parameter>",
        "<parameter=count>three</parameter>",
        "<parameter=count>3</parameter><parameter=extra>4</parameter>",
        "",
    ] {
        assert!(
            !accepts(&factory, &grammar, &call("count", invalid), false),
            "accepted invalid arguments: {invalid:?}"
        );
    }
}

#[test]
fn mixed_tools_only_constrain_strict_arguments() {
    let tools = json!([tool("count", true), tool("loose", false)]);
    let grammar = grammar_from_tools(&tools, "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    for name in ["loose", "unknown"] {
        let text =
            call(name, "arbitrary arguments") + &call("count", "<parameter=count>3</parameter>");
        assert!(accepts(&factory, &grammar, &text, false));
    }
    assert!(!accepts(
        &factory,
        &grammar,
        &call("count", "arbitrary arguments"),
        false
    ));
}

#[test]
fn without_strict_tools_text_is_unrestricted() {
    let factory = factory(&[]);
    for tools in [json!([]), json!([tool("count", false)])] {
        let grammar = grammar_from_tools(&tools, "qwen3_coder", false, Some(&[])).unwrap();
        assert!(accepts(
            &factory,
            &grammar,
            "Ordinary text. <tool_call>not a function",
            false
        ));
    }
}

#[rstest]
fn reasoning_delays_strict_calls(#[values(false, true)] initial_reasoning: bool) {
    let grammar = grammar_from_tools(
        &json!([tool("count", true)]),
        "qwen3_coder",
        initial_reasoning,
        Some(&[]),
    )
    .unwrap();
    let factory = factory(&[]);
    let opening = if initial_reasoning { "" } else { "<think>" };
    let reasoning = format!("{opening}Bad example: {}</think>\n", call("count", "wrong"));
    for (arguments, expected) in [("<parameter=count>3</parameter>", true), ("wrong", false)] {
        let text = reasoning.clone() + &call("count", arguments);
        assert_eq!(accepts(&factory, &grammar, &text, false), expected);
    }
}

#[rstest]
fn markers_accept_text_and_special_forms(#[values(false, true)] explicit_override: bool) {
    let markers: &[&str] = if explicit_override {
        &["<tool_call>", "</think>"]
    } else {
        &["<tool_call>", "</tool_call>", "<think>", "</think>"]
    };
    let names: Vec<String> = markers.iter().map(|name| name.to_string()).collect();
    let grammar = grammar_from_tools(
        &json!([tool("count", true)]),
        "qwen3_coder",
        false,
        explicit_override.then_some(names.as_slice()),
    )
    .unwrap();
    let factory = factory(markers);
    for special in [false, true] {
        let reasoning = "<think><tool_call>not a function</think>\n";
        for (arguments, expected) in [("<parameter=count>3</parameter>", true), ("wrong", false)] {
            let text = reasoning.to_string() + &call("count", arguments);
            assert_eq!(accepts(&factory, &grammar, &text, special), expected);
        }
    }
}

#[test]
fn unknown_marker_is_rejected() {
    let error = grammar_from_tools(
        &json!([tool("count", true)]),
        "qwen3_coder",
        false,
        Some(&["<unknown>".to_string()]),
    )
    .expect_err("unknown protocol marker should be rejected");
    let message = error.to_string().to_lowercase();
    assert!(
        message.contains("marker") || message.contains("special token"),
        "unexpected error: {error}"
    );
}

#[rstest]
#[case::plain(
    r###"{"type": "string"}"###,
    r###"["", "é🧪文 <tag> & \"quoted\""]"###,
    r###"["before </parameter> after"]"###
)]
#[case::enumerated(
    r###"{"type": "string", "enum": ["blue", "red"]}"###,
    r###"["blue", "red"]"###,
    r###"["green", "\"blue\"", " blue", "blue "]"###
)]
#[case::constant(
    r###"{"const": "ready"}"###,
    r###"["ready"]"###,
    r###"["Ready", "\"ready\""]"###
)]
#[case::pattern_anchored(
    r###"{"type": "string", "pattern": "^[a-z]{2,4}$"}"###,
    r###"["ab", "abcd"]"###,
    r###"["a", "abcde", "1abc", "abc1"]"###
)]
#[case::pattern_unanchored(
    r###"{"type": "string", "pattern": "ab"}"###,
    r###"["ab", "xxabyy", "\nxabz\n"]"###,
    r###"["a", "ba"]"###
)]
#[case::pattern_start(
    r###"{"type": "string", "pattern": "^ab"}"###,
    r###"["ab", "abcd"]"###,
    r###"["zab"]"###
)]
#[case::pattern_end(
    r###"{"type": "string", "pattern": "ab$"}"###,
    r###"["ab", "zzab"]"###,
    r###"["abz"]"###
)]
#[case::pattern_special(
    r###"{"type": "string", "pattern": "^a/b\\[c\\]\\.\\+$"}"###,
    r###"["a/b[c].+"]"###,
    r###"["a/bc.+", "a/b[c]x+"]"###
)]
#[case::pattern_ascii(
    r###"{"type": "string", "pattern": "^\\w\\d$"}"###,
    r###"["a3", "_0"]"###,
    r###"["é3", "a٣"]"###
)]
#[case::pattern_alternation(
    r###"{"type": "string", "pattern": "^ab|cd$"}"###,
    r###"["abx", "xcd"]"###,
    r###"["xabx", "cdx"]"###
)]
#[case::min_length(
    r###"{"type": "string", "minLength": 2}"###,
    r###"["ab", "abc", "é🧪", "中文"]"###,
    r###"["", "a", "é", "🧪", "文"]"###
)]
#[case::max_length(
    r###"{"type": "string", "maxLength": 2}"###,
    r###"["", "a", "é🧪", "中文"]"###,
    r###"["abc", "é🧪文"]"###
)]
#[case::length_bounds(
    r###"{"type": "string", "minLength": 2, "maxLength": 3}"###,
    r###"["é🧪", "é🧪文"]"###,
    r###"["é", "é🧪文字"]"###
)]
#[case::pattern_length(
    r###"{"type": "string", "pattern": "^[a-z]+$", "minLength": 2, "maxLength": 3}"###,
    r###"["ab", "abc"]"###,
    r###"["a", "abcd", "a3"]"###
)]
#[case::enum_pattern(
    r###"{"type": "string", "enum": ["blue", "red", "black"], "pattern": "^bl"}"###,
    r###"["blue", "black"]"###,
    r###"["red", "blond"]"###
)]
#[case::enum_length(
    r###"{"type": "string", "enum": ["a", "ab", "abc"], "minLength": 2, "maxLength": 2}"###,
    r###"["ab"]"###,
    r###"["a", "abc", "cd"]"###
)]
#[case::allof(r###"{"allOf": [{"type": "string", "pattern": "^[a-z]+$"}, {"minLength": 2}, {"maxLength": 3}]}"###, r###"["ab", "abc"]"###, r###"["a", "abcd", "a3"]"###)]
#[case::anyof(
    r###"{"anyOf": [{"type": "string", "pattern": "^a+$", "maxLength": 2}, {"const": "blue"}]}"###,
    r###"["a", "aa", "blue"]"###,
    r###"["aaa", "red", "ab"]"###
)]
#[case::oneof_disjoint(r###"{"oneOf": [{"type": "string", "pattern": "^a+$"}, {"type": "string", "pattern": "^b+$"}]}"###, r###"["a", "aa", "bb"]"###, r###"["ab", "c"]"###)]
#[case::oneof_overlap(r###"{"oneOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "string", "enum": ["b", "c"]}]}"###, r###"["a", "c"]"###, r###"["b", "d"]"###)]
#[case::mixed_anyof(r###"{"anyOf": [{"type": "string", "pattern": "^x+$"}, {"type": "integer", "minimum": 2, "maximum": 4}]}"###, r###"["xx", "2", "4"]"###, r###"["a", "1", "5", "3.5", "true"]"###)]
#[case::nullable(
    r###"{"type": ["string", "null"], "pattern": "^blue$"}"###,
    r###"["blue", "null"]"###,
    r###"["red", "true", "\"blue\""]"###
)]
#[case::drop_unsatisfiable_length(
    r###"{"anyOf": [{"type": "string", "minLength": 5, "maxLength": 2}, {"const": "ok"}]}"###,
    r###"["ok"]"###,
    r###"["ab", "abcde"]"###
)]
#[case::drop_unsatisfiable_enum(
    r###"{"anyOf": [{"type": "string", "enum": ["red"], "pattern": "^blue$"}, {"const": "ok"}]}"###,
    r###"["ok"]"###,
    r###"["red", "blue"]"###
)]
#[case::drop_unsatisfiable_for_json(
    r###"{"anyOf": [{"type": "string", "minLength": 5, "maxLength": 2}, {"type": "integer"}]}"###,
    r###"["3"]"###,
    r###"["abc", "\"3\""]"###
)]
#[case::delimiter_pattern(
    r###"{"type": "string", "pattern": "^before .* after$"}"###,
    r###"["before <tag> after"]"###,
    r###"["before </parameter> after"]"###
)]
#[case::newline_pattern(
    r###"{"type": "string", "pattern": "^\nblue\n$"}"###,
    r###"["\nblue\n"]"###,
    r###"["blue", "\nblue", "blue\n"]"###
)]
#[case::format_date_time(
    r###"{"type": "string", "format": "date-time"}"###,
    r###"["2026-10-28T12:30:00Z"]"###,
    r###"["2026-13-28T25:30:00Z"]"###
)]
#[case::format_time(
    r###"{"type": "string", "format": "time"}"###,
    r###"["12:30:00+01:00"]"###,
    r###"["25:30:00Z"]"###
)]
#[case::format_date(
    r###"{"type": "string", "format": "date"}"###,
    r###"["2026-10-28"]"###,
    r###"["2026-13-28"]"###
)]
#[case::format_duration(
    r###"{"type": "string", "format": "duration"}"###,
    r###"["P1DT2H"]"###,
    r###"["1 day"]"###
)]
#[case::format_email(
    r###"{"type": "string", "format": "email"}"###,
    r###"["reader@example.com"]"###,
    r###"["reader@@example.com"]"###
)]
#[case::format_hostname(
    r###"{"type": "string", "format": "hostname"}"###,
    r###"["example.com"]"###,
    r###"["-example.com"]"###
)]
#[case::format_ipv4(
    r###"{"type": "string", "format": "ipv4"}"###,
    r###"["192.168.1.1"]"###,
    r###"["256.1.1.1"]"###
)]
#[case::format_ipv6(
    r###"{"type": "string", "format": "ipv6"}"###,
    r###"["2001:db8::1"]"###,
    r###"["2001:db8::gg"]"###
)]
#[case::format_uuid(
    r###"{"type": "string", "format": "uuid"}"###,
    r###"["123e4567-e89b-12d3-a456-426614174000"]"###,
    r###"["123e4567-e89b-12d3-a456-42661417400z"]"###
)]
#[case::format_uri(
    r###"{"type": "string", "format": "uri"}"###,
    r###"["https://example.com/a?x=1#b"]"###,
    r###"["https://example.com/a b"]"###
)]
#[case::format_pattern_length(r###"{"type": "string", "format": "email", "pattern": "@example\\.com$", "minLength": 13, "maxLength": 15}"###, r###"["a@example.com", "abc@example.com"]"###, r###"["abcd@example.com", "a@other.com", "@@example.com"]"###)]
#[case::const_pattern(
    r###"{"const": "ready", "pattern": "^re"}"###,
    r###"["ready"]"###,
    r###"["re", "Ready", "\"ready\""]"###
)]
#[case::format_enum(
    r###"{"type": "string", "format": "email", "enum": ["a@example.com", "not-an-email"]}"###,
    r###"["a@example.com"]"###,
    r###"["not-an-email", "b@example.com"]"###
)]
fn lark_string_rules(
    #[case] schema: &str,
    #[case] valid: &str,
    #[case] invalid: &str,
    #[values("direct", "ref", "allof", "anyof", "oneof", "nullable")] wrapper: &str,
) {
    let mut schema: Value = serde_json::from_str(schema).unwrap();
    let mut definitions = json!({});
    match wrapper {
        "ref" => {
            definitions = json!({"value": schema});
            schema = json!({"$ref": "#/$defs/value"});
        }
        "allof" => schema = json!({"allOf": [schema, {}]}),
        "anyof" => schema = json!({"anyOf": [schema, false]}),
        "nullable" => schema = json!({"anyOf": [schema, {"type":"null"}]}),
        "oneof" => schema = json!({"oneOf": [schema, false]}),
        _ => {}
    }
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] = json!({"type": "object", "properties": {"value": schema}, "required": ["value"], "$defs": definitions});
    let grammar =
        grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    if wrapper == "nullable" {
        assert!(accepts(&factory, &grammar, &value_call("null"), false));
    }
    for (values, expected) in [(valid, true), (invalid, false)] {
        for value in serde_json::from_str::<Vec<String>>(values).unwrap() {
            let text = call(
                "sample",
                &format!("<parameter=value>\n{value}\n</parameter>"),
            );
            assert_eq!(
                accepts(&factory, &grammar, &text, false),
                expected,
                "{value:?}"
            );
        }
    }
}

#[rstest]
#[case::minimum(
    r###"{"type": "number", "minimum": 2}"###,
    r###"["2", "2.5"]"###,
    r###"["1.9"]"###
)]
#[case::maximum(
    r###"{"type": "number", "maximum": 2}"###,
    r###"["1.5", "2"]"###,
    r###"["2.1"]"###
)]
#[case::exclusive_minimum(
    r###"{"type": "number", "exclusiveMinimum": 2}"###,
    r###"["2.1"]"###,
    r###"["2", "1"]"###
)]
#[case::exclusive_maximum(
    r###"{"type": "number", "exclusiveMaximum": 2}"###,
    r###"["1.9"]"###,
    r###"["2", "3"]"###
)]
#[case::multiple_of(
    r###"{"type": "integer", "multipleOf": 3}"###,
    r###"["-3", "0", "6"]"###,
    r###"["4", "3.5"]"###
)]
#[case::fractional_multiple(
    r###"{"type": "number", "multipleOf": 0.5}"###,
    r###"["-0.5", "1.5"]"###,
    r###"["1.25"]"###
)]
#[case::integer(
    r###"{"type": "integer"}"###,
    r###"["0", "-3"]"###,
    r###"["1.5", "\"3\""]"###
)]
#[case::number(
    r###"{"type": "number"}"###,
    r###"["-1.5", "2e3"]"###,
    r###"["true", "\"2\""]"###
)]
#[case::boolean(
    r###"{"type": "boolean"}"###,
    r###"["true", "false"]"###,
    r###"["True", "1"]"###
)]
#[case::null(
    r###"{"type": "null"}"###,
    r###"["null"]"###,
    r###"["NULL", "None"]"###
)]
fn lark_json_scalar_rules(#[case] schema: &str, #[case] valid: &str, #[case] invalid: &str) {
    let schema: Value = serde_json::from_str(schema).unwrap();
    let grammar = value_grammar(schema).unwrap();
    let factory = factory(&[]);
    for (values, expected) in [(valid, true), (invalid, false)] {
        for value in serde_json::from_str::<Vec<String>>(values).unwrap() {
            for pad in ["", "\n"] {
                assert_eq!(
                    accepts(
                        &factory,
                        &grammar,
                        &value_call(&format!("{pad}{value}{pad}")),
                        false
                    ),
                    expected,
                    "{value:?}"
                );
            }
        }
    }
}

fn value_grammar(schema: Value) -> anyhow::Result<String> {
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] =
        json!({"type": "object", "properties": {"value": schema}, "required": ["value"]});
    grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[]))
}

fn value_call(value: &str) -> String {
    call("sample", &format!("<parameter=value>{value}</parameter>"))
}

#[rstest]
#[case("boolean", "true")]
#[case("boolean", "false")]
#[case("null", "null")]
fn boolean_null_padding(
    #[case] kind: &str,
    #[case] value: &str,
    #[values("", "\n", " ", "\t", "\r", "\n\n", "\n ")] padding: &str,
) {
    let grammar = value_grammar(json!({"type":kind})).unwrap();
    let factory = factory(&[]);
    assert!(accepts(&factory, &grammar, &value_call(value), false));
    for body in [
        format!("{padding}{value}"),
        format!("{value}{padding}"),
        format!("{padding}{value}{padding}"),
    ] {
        assert_eq!(
            accepts(&factory, &grammar, &value_call(&body), false),
            padding.is_empty() || padding == "\n",
            "{body:?}"
        );
    }
}

#[rstest]
#[case::contradictory_length(
    r###"{"type": "string", "minLength": 5, "maxLength": 2}"###,
    "unsatisfiable"
)]
#[case::contradictory_enum(
    r###"{"type": "string", "enum": ["red"], "pattern": "^blue$"}"###,
    "unsatisfiable"
)]
#[case::contradictory_allof(
    r###"{"allOf": [{"type": "string", "maxLength": 2}, {"minLength": 5}]}"###,
    "unsatisfiable"
)]
#[case::empty_oneof(
    r###"{"oneOf": [{"const": "same"}, {"const": "same"}]}"###,
    "unsatisfiable"
)]
#[case::unknown_format(r###"{"type": "string", "format": "not-a-format"}"###, "format")]
#[case::remote_ref(r###"{"$ref": "https://example.invalid/value.json"}"###, "ref")]
#[case::missing_ref(r###"{"$ref": "#/$defs/missing"}"###, "ref")]
#[case::unsupported_keyword(r###"{"type": "string", "not": {"const": "x"}}"###, "not")]
#[case::unevaluated(
    r###"{"type": "object", "unevaluatedProperties": false}"###,
    "unevaluatedProperties"
)]
#[case::pattern_properties(
    r###"{"type": "object", "patternProperties": {"^a": {"type": "integer"}}}"###,
    "patternProperties"
)]
#[case::invalid_pattern(r###"{"type": "string", "pattern": "["}"###, "regex")]
#[case::lookaround_pattern(r###"{"type": "string", "pattern": "(?=a)b"}"###, "regex")]
#[case::allof_incompatible_types(
    r###"{"allOf": [{"type": "string"}, {"type": "integer"}]}"###,
    "unsatisfiable"
)]
#[case::contradictory_const_pattern(r###"{"const": "ready", "pattern": "^x"}"###, "unsatisfiable")]
fn schema_errors_identify_tool_and_parameter(
    #[case] schema: &str,
    #[case] keyword: &str,
    #[values(false, true)] referenced: bool,
) {
    let schema: Value = serde_json::from_str(schema).unwrap();
    let mut definition = tool("sample", true);
    let (value, definitions) = if referenced {
        (
            json!({"$ref":"#/$defs/constraint"}),
            json!({"constraint":schema}),
        )
    } else {
        (schema, json!({}))
    };
    definition["function"]["parameters"] = json!({"type":"object", "properties":{"value":value}, "required":["value"], "$defs":definitions});
    let error =
        grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[])).unwrap_err();
    let message = format!("{error:#}").to_lowercase();
    assert!(
        message.contains("sample") && message.contains("value"),
        "{message}"
    );
    assert!(message.contains(&keyword.to_lowercase()), "{message}");
}

#[rstest]
#[case(json!({"type":"string", "enum":["blue"]}))]
#[case(json!({"type":"string", "pattern":"^[a-z]+$", "minLength":4, "maxLength":4}))]
fn constrained_string_padding(#[case] schema: Value) {
    let grammar = value_grammar(schema).unwrap();
    let valid: Vec<_> = ["blue", "\nblue", "blue\n", "\nblue\n"]
        .map(value_call)
        .into();
    let invalid: Vec<_> = ["\n\nblue", "blue\n\n", " blue", "blue ", "\r\nblue\r\n"]
        .map(value_call)
        .into();
    llg_test_utils::lark_str_test_many_quiet(
        &grammar,
        &valid.iter().map(String::as_str).collect::<Vec<_>>(),
        &invalid.iter().map(String::as_str).collect::<Vec<_>>(),
    );
}

#[rstest]
#[case("\nblue", "\n\nblue", "\nblue")]
#[case("blue\n", "blue\n\n", "blue\n")]
#[case("\nblue\n", "\n\nblue\n\n", "\nblue\n")]
fn constant_newlines_are_value(#[case] value: &str, #[case] valid: &str, #[case] invalid: &str) {
    let grammar = value_grammar(json!({"const": value})).unwrap();
    let factory = factory(&[]);
    assert!(accepts(&factory, &grammar, &value_call(valid), false));
    assert!(!accepts(&factory, &grammar, &value_call(invalid), false));
}

#[test]
fn reference_constraints_preserve_raw_and_json_encoding() {
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] = json!({"type":"object", "$defs":{"name":{"type":"string","pattern":"^[a-z]+$","minLength":2,"maxLength":3}},"properties":{"raw":{"$ref":"#/$defs/name"},"nested":{"type":"array","items":{"$ref":"#/$defs/name"}}},"required":["raw","nested"]});
    let grammar =
        grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    for (raw, nested, expected) in [
        ("ab", r#"["abc"]"#, true),
        ("a", r#"["abc"]"#, false),
        ("ab", r#"["abcd"]"#, false),
        ("ab", r#"["a3"]"#, false),
        ("ab", "[abc]", false),
    ] {
        let arguments = format!(
            "<parameter=raw>\n{raw}\n</parameter><parameter=nested>\n{nested}\n</parameter>"
        );
        assert_eq!(
            accepts(&factory, &grammar, &call("sample", &arguments), false),
            expected
        );
    }
}

#[test]
fn large_ordered_schema_and_tool_list_fit_default_limits() {
    let properties: serde_json::Map<String, Value> = (0..50)
        .map(|i| (format!("field_{i}"), json!({"type":"integer"})))
        .collect();
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] = json!({"type":"object", "properties":properties, "required":["field_0","field_25","field_49"]});
    let tools: Vec<_> = (0..100)
        .map(|i| tool(&format!("tool_{i}"), true))
        .chain([definition])
        .collect();
    let grammar = grammar_from_tools(&json!(tools), "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    let required = [0, 25, 49]
        .map(|i| format!("<parameter=field_{i}>1</parameter>"))
        .join("");
    assert!(accepts(
        &factory,
        &grammar,
        &call("sample", &required),
        false
    ));
    let all = (0..50)
        .map(|i| format!("<parameter=field_{i}>1</parameter>"))
        .collect::<String>();
    assert!(accepts(&factory, &grammar, &call("sample", &all), false));
    assert!(!accepts(
        &factory,
        &grammar,
        &call(
            "sample",
            &all.replace("<parameter=field_49>1</parameter>", "")
        ),
        false
    ));
    let calls = (0..100)
        .map(|i| call(&format!("tool_{i}"), "<parameter=count>3</parameter>"))
        .collect::<String>();
    assert!(accepts(&factory, &grammar, &calls, false));
    assert!(!accepts(
        &factory,
        &grammar,
        &call("tool_99", "<parameter=wrong>3</parameter>"),
        false
    ));
}

#[rstest]
#[case("<parameter=count>3</parameter><parameter=mode>fast</parameter>", true)]
#[case("<parameter=count>3</parameter>", true)]
#[case(
    "<parameter=mode>fast</parameter><parameter=count>3</parameter>",
    false
)]
#[case("<parameter=count>3</parameter><parameter=count>4</parameter>", false)]
#[case("<parameter=wrong>3</parameter>", false)]
#[case("<parameter=mode>fast</parameter>", false)]
#[case("<parameter=count>3</parameter><parameter=extra>x</parameter>", true)]
#[case(
    "<parameter=count>3</parameter><parameter=extra>x</parameter><parameter=extra>y</parameter>",
    true
)]
#[case(
    "<parameter=count>3</parameter><parameter=extra>x</parameter><parameter=mode>fast</parameter>",
    false
)]
#[case("<parameter=count>{\"count\":3}</parameter>", false)]
fn ordered_parameters_and_extras(#[case] arguments: &str, #[case] expected: bool) {
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] = json!({"type":"object", "properties":{"count":{"type":"integer"}, "mode":{"enum":["fast","slow"]}}, "required":["count"], "additionalProperties":true});
    let grammar =
        grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    assert!(accepts(
        &factory,
        &grammar,
        &call("sample", "<parameter=count>3</parameter>"),
        false
    ));
    assert_eq!(
        accepts(&factory, &grammar, &call("sample", arguments), false),
        expected
    );
}

#[rstest]
fn header_whitespace_cannot_bypass_strict_tools(#[values("", "\n", " \t\r\n")] gap: &str) {
    let grammar = grammar_from_tools(
        &json!([tool("count", true)]),
        "qwen3_coder",
        false,
        Some(&[]),
    )
    .unwrap();
    let factory = factory(&[]);
    for (name, arguments, expected) in [
        ("count", "<parameter=count>3</parameter>", true),
        ("count", "<parameter=wrong>3</parameter>", false),
        (" count", "anything", false),
        ("count ", "anything", false),
        ("count_extra", "anything", true),
    ] {
        let text = format!(
            "<tool_call>{gap}<function={name}>{gap}{arguments}{gap}</function>{gap}</tool_call>"
        );
        assert_eq!(accepts(&factory, &grammar, &text, false), expected);
    }
}

#[test]
fn rollback_and_copy_preserve_value_masks_and_incomplete_call_eos() {
    let grammar = value_grammar(json!({"enum":["blue","black"]})).unwrap();
    let factory = factory(&[]);
    let env = factory.tok_env();
    let mut matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar)));
    let prefix = env.tokenize("<tool_call><function=sample><parameter=value>bl");
    matcher.consume_tokens(&prefix).unwrap();
    let before = matcher.compute_mask().unwrap();
    assert!(!before.is_allowed(env.eos_token()));
    assert!(before.is_allowed(b'u' as u32));
    assert!(before.is_allowed(b'a' as u32));
    assert!(!before.is_allowed(b'z' as u32));
    let mut copy = matcher.deep_clone();
    let draft = env.tokenize("ue</parameter></function></tool_call>");
    matcher.consume_tokens(&draft).unwrap();
    assert!(matcher.is_accepting().unwrap());
    matcher.rollback(draft.len()).unwrap();
    assert_eq!(matcher.compute_mask().unwrap(), before);
    assert_eq!(copy.compute_mask().unwrap(), before);
    copy.consume_tokens(&env.tokenize("ack</parameter></function></tool_call>"))
        .unwrap();
    assert!(copy.is_accepting().unwrap());
    assert_eq!(matcher.compute_mask().unwrap(), before);
}

const OBJECT_A: &str = r###"{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}"###;
const OBJECT_B: &str = r###"{"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"], "additionalProperties": false}"###;

fn check_value_schema(value: Value, root: Value, valid: &[&str], invalid: &[&str]) {
    let mut parameters =
        json!({"type": "object", "properties": {"value": value}, "required": ["value"]});
    parameters
        .as_object_mut()
        .unwrap()
        .extend(root.as_object().unwrap().clone());
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] = parameters;
    let grammar =
        grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    for (texts, expected) in [(valid, true), (invalid, false)] {
        for text in texts {
            assert_eq!(
                accepts(
                    &factory,
                    &grammar,
                    &value_call(&format!("\n{text}\n")),
                    false
                ),
                expected,
                "{text:?}"
            );
        }
    }
}

fn object(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[rstest]
#[case::ref_number(json!({"$ref": "#/$defs/n"}), json!({"$defs": {"n": {"type": "integer", "minimum": 1, "maximum": 5}}}), &["1", "5"], &["0", "6", "1.5", "\"3\""])]
#[case::ref_boolean(json!({"$ref": "#/$defs/b"}), json!({"$defs": {"b": {"type": "boolean"}}}), &["true", "false"], &["yes", "\"true\""])]
#[case::ref_object(json!({"$ref": "#/$defs/o"}), json!({"$defs": {"o": object(OBJECT_A)}}), &[r#"{"a":1}"#], &[r#"{"a":"x"}"#, "{}", r#"{"a":1,"b":2}"#])]
#[case::ref_array(json!({"$ref": "#/$defs/l"}), json!({"$defs": {"l": {"type": "array", "items": {"enum": ["x", "y"]}, "minItems": 1}}}), &[r#"["x"]"#, r#"["x","y"]"#], &["[]", r#"["z"]"#])]
#[case::ref_chain(json!({"$ref": "#/$defs/a"}), json!({"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"type": "string", "enum": ["ok"]}}}), &["ok"], &["no"])]
#[case::definitions_keyword(json!({"$ref": "#/definitions/s"}), json!({"definitions": {"s": {"type": "string", "pattern": "^[0-9]+$"}}}), &["123"], &["12a"])]
#[case::ref_with_sibling(json!({"$ref": "#/$defs/s", "maxLength": 3}), json!({"$defs": {"s": {"type": "string", "pattern": "^[a-z]+$"}}}), &["abc"], &["abcd", "ab1"])]
#[case::anyof_refs(json!({"anyOf": [{"$ref": "#/$defs/auto"}, {"$ref": "#/$defs/count"}]}), json!({"$defs": {"auto": {"const": "auto"}, "count": {"type": "integer", "minimum": 0}}}), &["auto", "0", "7"], &["-1", "manual"])]
#[case::anyof_object_array(json!({"anyOf": [object(OBJECT_A), {"type": "array", "items": {"type": "integer"}}]}), json!({}), &[r#"{"a":1}"#, "[1,2]"], &[r#"{"a":"x"}"#, r#"["x"]"#, "1"])]
#[case::anyof_integer_array(json!({"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]}), json!({}), &["3", "[3]"], &["x", r#"["3"]"#])]
#[case::anyof_two_objects(json!({"anyOf": [object(OBJECT_A), object(OBJECT_B)]}), json!({}), &[r#"{"a":1}"#, r#"{"b":"x"}"#], &[r#"{"c":1}"#, r#"{"a":"x"}"#])]
#[case::oneof_containers(json!({"oneOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "object"}]}), json!({}), &["[1]", "{}"], &["1"])]
#[case::nullable_object(json!({"type": ["object", "null"], "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}), json!({}), &[r#"{"a":1}"#, "null"], &["{}", "\"null\""])]
#[case::nullable_array(json!({"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]}), json!({}), &["[1]", "null"], &[r#"["a"]"#])]
#[case::nullable_ref_object(json!({"anyOf": [{"$ref": "#/$defs/o"}, {"type": "null"}]}), json!({"$defs": {"o": object(OBJECT_A)}}), &[r#"{"a":1}"#, "null"], &[r#"{"a":"x"}"#])]
#[case::allof_objects(json!({"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}, {"properties": {"a": {"maximum": 3}}}]}), json!({}), &[r#"{"a":3}"#], &[r#"{"a":4}"#, "{}"])]
#[case::const_object(json!({"const": {"a": 1}}), json!({}), &[r#"{"a":1}"#], &[r#"{"a":2}"#])]
#[case::enum_arrays(json!({"enum": [[1, 2], [3]]}), json!({}), &["[1,2]", "[3]"], &["[2,1]"])]
fn structural_value_schemas(
    #[case] value: Value,
    #[case] root: Value,
    #[case] valid: &[&str],
    #[case] invalid: &[&str],
) {
    check_value_schema(value, root, valid, invalid);
}

#[rstest]
#[case::decimal_string_lengths(json!({"type": "string", "minLength": 2.0, "maxLength": 3.0}), &["ab", "abc"], &["a", "abcd"])]
#[case::decimal_item_counts(json!({"type": "array", "items": {"type": "integer"}, "minItems": 1.0, "maxItems": 2.0}), &["[1]", "[1,2]"], &["[]", "[1,2,3]"])]
#[case::nested_pattern_properties(json!({"type": "object", "patternProperties": {"^a": {"type": "integer"}}, "additionalProperties": false}), &[r#"{"ab":1}"#], &[r#"{"ab":"x"}"#, r#"{"b":1}"#])]
#[case::nested_id_annotation(json!({"type": "object", "$id": "urn:example:object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}), &[r#"{"a":1}"#], &[r#"{"a":"x"}"#])]
#[case::string_or_object(json!({"anyOf": [{"type": "string"}, object(OBJECT_A)]}), &["hello", r#"{"a":1}"#, "{ not json"], &["a </parameter> b"])]
#[case::enum_or_object(json!({"anyOf": [{"type": "string", "enum": ["auto"]}, object(OBJECT_A)]}), &["auto", r#"{"a":1}"#], &["manual", r#"{"a":"x"}"#])]
fn matches_json_compiler_support(
    #[case] value: Value,
    #[case] valid: &[&str],
    #[case] invalid: &[&str],
) {
    check_value_schema(value, json!({}), valid, invalid);
}

#[test]
fn decimal_property_counts() {
    let mut definition = tool("sample", true);
    definition["function"]["parameters"] = json!({
        "type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"],
        "additionalProperties": true, "minProperties": 2.0, "maxProperties": 2.0
    });
    let grammar =
        grammar_from_tools(&json!([definition]), "qwen3_coder", false, Some(&[])).unwrap();
    let factory = factory(&[]);
    let count = "<parameter=count>1</parameter>";
    let extra = "<parameter=extra>x</parameter>";
    assert!(accepts(
        &factory,
        &grammar,
        &call("sample", &format!("{count}{extra}")),
        false
    ));
    assert!(!accepts(&factory, &grammar, &call("sample", count), false));
    assert!(!accepts(
        &factory,
        &grammar,
        &call(
            "sample",
            &format!("{count}{extra}<parameter=more>y</parameter>")
        ),
        false
    ));
}
