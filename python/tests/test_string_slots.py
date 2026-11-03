import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from llguidance import LLGrammarTemplate, LLMatcher, LLParserLimits, LLTokenizer


def bundle():
    return {
        "grammars": [
            {
                "name": "main",
                "lark_grammar": 'start: "[" @slot "," ESCAPED_STRING "]"\n%import common.ESCAPED_STRING',
            },
            {
                "name": "slot",
                "string_slot": "targets",
                "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING",
            },
        ]
    }


def test_full_compilation_equivalence():
    tokenizer = LLTokenizer("byte")
    template = LLGrammarTemplate(tokenizer, bundle())
    for value in ["", "Kitchen", 'a"b\\c\n', "Étage 東京"]:
        literal = json.dumps(json.dumps(value, ensure_ascii=False), ensure_ascii=False)
        full = LLMatcher(
            tokenizer,
            'start: "[" VALUE "," ESCAPED_STRING "]"\nVALUE: ' + literal + "\n%import common.ESCAPED_STRING",
            log_level=0,
        )
        native = template.new_matcher({"targets": [value, value]})
        for token in json.dumps([value, "unrestricted"], ensure_ascii=False, separators=(",", ":")).encode():
            assert native.compute_bitmask() == full.compute_bitmask()
            assert native.consume_token(token)
            assert full.consume_token(token)
        assert native.is_accepting()


def test_validation():
    tokenizer = LLTokenizer("byte")
    template = LLGrammarTemplate(tokenizer, bundle())
    assert template.slot_names == ["targets"]
    for value in ({"unknown": ["x"]}, {"targets": []}, {"targets": "bad"}, {"targets": [1]}):
        with pytest.raises((ValueError, TypeError)):
            template.new_matcher(value)
    bounded = LLGrammarTemplate(tokenizer, bundle(), max_binding_bytes=2, max_binding_values=1)
    for value in (["abc"], ["a", "a"]):
        with pytest.raises(ValueError):
            bounded.new_matcher({"targets": value})
    assert not bounded.new_matcher({"targets": ["a"]}).is_error()


def test_concurrent_fallback_and_bindings():
    template = LLGrammarTemplate(LLTokenizer("byte"), bundle())

    def run(i):
        matcher = template.new_matcher({"targets": [str(i)]} if i % 2 else {})
        for token in json.dumps([str(i), "free"], separators=(",", ":")).encode():
            matcher.compute_bitmask()
            assert matcher.consume_token(token)
        assert matcher.is_accepting()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(32)))


def test_binding_replaces_fallback_and_uses_canonical_json():
    template = LLGrammarTemplate(
        LLTokenizer("byte"), {"grammars": [{"string_slot": "names", "lark_grammar": 'start: /"[A-Z]+"/'}]}
    )

    def accepts(matcher, text):
        tokens = list(text.encode("utf-8"))
        return matcher.validate_tokens(tokens) == len(tokens)

    assert not accepts(template.new_matcher(), '"Kitchen"')
    assert accepts(template.new_matcher({"names": ["Kitchen"]}), '"Kitchen"')
    assert not accepts(template.new_matcher({"names": ["Kitchen"]}), '"OTHER"')
    assert accepts(template.new_matcher({"names": ["é"]}), '"é"')
    assert not accepts(template.new_matcher({"names": ["é"]}), r'"\u00e9"')


def test_token_limit_matches_full_compilation():
    tokenizer = LLTokenizer("byte")
    limited = {**bundle(), "max_tokens": 2}
    template = LLGrammarTemplate(tokenizer, limited)
    for bindings in [None, {"targets": ["Kitchen"]}]:
        full_bundle = json.loads(json.dumps(limited))
        if bindings:
            full_bundle["grammars"][1]["lark_grammar"] = 'start: "\\"Kitchen\\""'
            del full_bundle["grammars"][1]["string_slot"]
        bound = template.new_matcher(bindings)
        full = LLMatcher(tokenizer, json.dumps(full_bundle), log_level=0)
        for token in b'["Kitchen"':
            assert bound.compute_bitmask() == full.compute_bitmask()
            assert bound.consume_token(token) == full.consume_token(token)
            assert bound.stop_reason() == full.stop_reason()
            if bound.is_stopped():
                break


def test_lexer_budget_failure_keeps_template_usable():
    template = LLGrammarTemplate(
        LLTokenizer("byte"), {"grammars": [bundle()["grammars"][1]]}, limits=LLParserLimits(initial_lexer_fuel=2000)
    )
    with pytest.raises(ValueError):
        template.new_matcher({"targets": ["x" * 50000]})
    assert not template.new_matcher({"targets": ["Kitchen"]}).is_error()
