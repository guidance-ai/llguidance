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
