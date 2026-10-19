import pytest
from llguidance import LLMatcher, LLTokenizer, grammar_from


@pytest.mark.parametrize(
    "rule,accepted,rejected",
    [
        (r'"\u0061"', "a", "b"),
        (r'"\u0000"', "\x00", "a"),
        (r'"\u000a"', "\n", "a"),
        (r'"\u0022"', '"', "a"),
        (r'"\u00e9"', "é", "e"),
        (r"[\u0061-\u007a]", "m", "A"),
        (r"[\u005d]", "]", "a"),
        (r"[\u0000]", "\x00", "a"),
        (r'"\u4e2d"', "中", "a"),
        (r'"a"', "a", "b"),
        (r"[a-z]", "m", "A"),
    ],
)
def test_gbnf_unicode_escapes(rule: str, accepted: str, rejected: str) -> None:
    grammar = grammar_from("gbnf", "root ::= " + rule)
    assert not LLMatcher.validate_grammar(grammar)
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, grammar)
    rejected_tokens = tokenizer.tokenize_str(rejected)
    assert matcher.validate_tokens(rejected_tokens) < len(rejected_tokens)
    for token in tokenizer.tokenize_str(accepted):
        assert matcher.consume_token(token)
    assert matcher.is_accepting()
