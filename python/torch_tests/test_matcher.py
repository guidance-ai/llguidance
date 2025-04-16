from typing import Any, Dict, List, Tuple, Union
import llguidance
from llguidance.numpy import fill_next_token_bitmask_par, allocate_token_bitmask
from llguidance import LLMatcher, LLTokenizer, StructTag, LLParserLimits
import pytest
from numpy.typing import NDArray
import numpy as np

_tokenizer = None


def tokenizer() -> LLTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = LLTokenizer("byte")
    return _tokenizer


def matcher(grm: str) -> LLMatcher:
    return LLMatcher(tokenizer(), grm, log_level=1)


def check_err(matcher: LLMatcher) -> None:
    if matcher.is_error():
        raise ValueError(matcher.get_error())


def check_one_grammar(grm: str, s: str, passing: bool) -> None:
    # print("Checking", repr(s))
    interp = matcher(grm)
    check_err(interp)
    final_reject = False
    if s.startswith("FINAL_REJECT:"):
        final_reject = True
        s = s[len("FINAL_REJECT:"):]

    if "⁂" in s:
        b = (s + "⁂[13]").encode().replace("⁂".encode(), b"\xFF")
        tokens, suffix = tokenizer().tokenize_partial(b)
        assert not suffix
        assert tokens[-1] == 13
        del tokens[-1]
    else:
        tokens = tokenizer().tokenize_str(s)

    print("Check: " + tokenizer().dbg_tokens(tokens))

    for i, t in enumerate(tokens):
        next_tokens = tokens[i:]
        if passing or final_reject:
            assert interp.validate_tokens(next_tokens) == len(next_tokens)
        else:
            assert interp.validate_tokens(next_tokens) < len(next_tokens)
        mask = interp.compute_logit_bias()
        if mask[t] == 0:
            if passing or final_reject:
                print("Token not in mask",
                      tokenizer().dbg_tokens(tokens[:i + 1]), repr(s))
                assert False
            return
        else:
            assert mask[t] == 200
        interp.consume_token(t)
        check_err(interp)
    if final_reject:
        if interp.is_accepting():
            print("Expected to fail at the end", s)
            assert False
        else:
            return
    if not passing:
        print("Expected to fail", s)
        assert False
    assert interp.is_accepting()


def check_grammar(grm: str, passing: List[str], failing: List[str]) -> None:
    for s in passing:
        check_one_grammar(grm, s, True)
    for s in failing:
        check_one_grammar(grm, s, False)


def test_json() -> None:
    grm = LLMatcher.grammar_from_json_schema({"type": "object"},
                                             {"whitespace_flexible": False})
    check_grammar(grm, ["{}", '{"foo":1}'], ["FINAL_REJECT:{", " {}", "{ }"])

    grm = LLMatcher.grammar_from_json_schema({
        "type": "object",
        "properties": {
            "foo": {
                "type": "integer"
            }
        },
        "required": ["foo"]
    })
    check_grammar(grm, ['{"foo":1}', '{"foo":1,"bar":2}', '{ "foo" : 1 }'],
                  ["{}", "FINAL_REJECT:{", ' {"foo":1}', '{"bar":1}'])


def test_lark() -> None:
    check_grammar(
        'start: /.../ "abc" /.../',
        [
            "abcabcabc",
            "aaaabcccc",
            # NOTE: Also ensures that multi-byte characters still count as a single character
            "🔵🟠✅abc❌🟠🔵",
        ],
        [
            "aaabcccc",
            "aaaaabcccc",
            "FINAL_REJECT:aaaabccc",
            "aaaabccccc",
            "🔵🟠✅❌abc❌✅🟠🔵",
            "🔵🟠abc🟠🔵",
        ],
    )


def test_regex_grammar() -> None:
    grm = LLMatcher.grammar_from_regex(r"\d+")
    check_grammar(grm, ["123", "456"], ["abc", "1a2"])


def test_lark_syntax() -> None:
    m = matcher('start: /.../ no_such_rule')
    assert m.is_error()
    assert "no_such_rule" in m.get_error()

    e = LLMatcher.validate_grammar('start: /.../ no_such_rule')
    assert "no_such_rule" in e

    e = LLMatcher.validate_grammar('start: /.../ no_such_rule', tokenizer())
    assert "no_such_rule" in e


def test_regex_syntax() -> None:
    g = LLMatcher.grammar_from_regex(r"missing close paren (")
    e = LLMatcher.validate_grammar(g)
    assert "invalid regex" in e
    e = LLMatcher.validate_grammar(g, tokenizer())
    assert "invalid regex" in e


def test_slices() -> None:
    t = tokenizer()
    gen_slices = LLTokenizer.general_slices()
    assert len(gen_slices) > 0
    json_slices = LLTokenizer.json_slices()
    assert len(json_slices) > 0
    t2 = t.with_slices(json_slices)
    assert t.tokenize_str("Hello, world!") == t2.tokenize_str("Hello, world!")


def mask_has(mask: NDArray[np.int32], t: int) -> bool:
    v: int = mask[t // 32]
    return v & (1 << (t % 32)) != 0


def test_par_errors() -> None:
    t = tokenizer()
    exec = llguidance.LLExecutor()
    g0 = matcher(r"start: /[a-zA-Z ]*/")
    g1 = matcher(r"start: /[0-9 ]*/")
    mask = allocate_token_bitmask(3, t.vocab_size)

    with pytest.raises(ValueError, match="Target index out of bounds"):
        fill_next_token_bitmask_par(exec, [(g0, 0), (g1, 3)], mask)

    with pytest.raises(RuntimeError, match="Already borrowed"):
        fill_next_token_bitmask_par(exec, [(g0, 0), (g1, 1), (g1, 2)], mask)

    with pytest.raises(TypeError, match="cannot be converted"):
        l = [1, (g1, 0), (g1, 1)]
        fill_next_token_bitmask_par(exec, l, mask)  # type: ignore

    with pytest.raises(TypeError, match="cannot be converted"):
        l = [(tokenizer(), 0)]
        fill_next_token_bitmask_par(exec, l, mask)  # type: ignore

    with pytest.raises(ValueError, match=r"Expecting.*tuple"):
        l = [(tokenizer(), 0, 0)]
        fill_next_token_bitmask_par(exec, l, mask)  # type: ignore

    (three, vocab) = mask.shape
    assert three == 3
    with pytest.raises(ValueError, match="Null pointer"):
        exec.unsafe_compute_mask_ptr([(g0, 0), (g1, 1)], 0, vocab * 4, 3)
    with pytest.raises(ValueError, match="Pointer not aligned"):
        exec.unsafe_compute_mask_ptr([(g0, 0), (g1, 1)], 3, vocab * 4, 3)
    with pytest.raises(ValueError, match="Invalid buffer size"):
        exec.unsafe_compute_mask_ptr([(g0, 0), (g1, 1)], 1024, vocab * 4 + 1,
                                     3)
    with pytest.raises(ValueError, match="Invalid buffer size"):
        exec.unsafe_compute_mask_ptr([(g0, 0), (g1, 1)], 1024, vocab * 4 - 1,
                                     3)

    # should be OK
    fill_next_token_bitmask_par(exec, [(g0, 0), (g1, 2)], mask)
    t_a = t.tokenize_str("a")[0]
    t_1 = t.tokenize_str("1")[0]
    assert mask_has(mask[0, :], t_a)
    assert not mask_has(mask[0, :], t_1)
    assert not mask_has(mask[2, :], t_a)
    assert mask_has(mask[2, :], t_1)


def consume_tokens(m: LLMatcher, tokens: List[int]) -> None:
    print("Consume", tokenizer().dbg_tokens(tokens))
    assert m.stop_reason() == "NotStopped"
    assert not m.is_stopped()
    assert not m.is_accepting()
    for t in tokens:
        mask = m.compute_logit_bias()
        assert mask[t] == 200, "Token should be in mask."
        bit_mask = m.compute_bitmask()
        assert bit_mask[t // 8] & (1 <<
                                   (t % 8)) != 0, "Token should be in bitmask."
        assert m.stop_reason() == "NotStopped"
        assert not m.is_stopped()
        assert not m.is_accepting()
        m.consume_token(t)
    assert not m.is_error()


def test_stopping_conditions() -> None:
    m = LLMatcher(tokenizer(), "start: /[aA][bB][cC]/")
    consume_tokens(m, tokenizer().tokenize_str("abc"))
    assert m.is_accepting()
    assert m.is_stopped()
    assert m.stop_reason() == "NoExtension"


def test_rollback() -> None:
    m = LLMatcher(tokenizer(), "start: /[aA] [bB] [cC] [dD] [eE]/")
    m2 = m.deep_copy()
    t = tokenizer().tokenize_str("a b c d e")
    consume_tokens(m, t[0:3])
    assert not m.is_stopped() and not m.is_accepting()
    m.rollback(2)
    m3 = m.deep_copy()
    consume_tokens(m, t[1:])
    assert m.is_stopped() and m.is_accepting()
    assert not m.is_error()
    m.rollback(1)
    assert not m.is_accepting() and not m.is_stopped()
    assert m.stop_reason() == "NotStopped"
    consume_tokens(m, t[-1:])
    assert m.is_stopped() and m.is_accepting()
    assert not m.is_error()

    mask = m.compute_logit_bias()
    assert mask[tokenizer().eos_token] == 200
    assert not m.is_error()
    m.consume_token(tokenizer().eos_token)
    assert not m.is_error()

    consume_tokens(m2, t)
    assert m2.is_stopped() and m2.is_accepting() and not m2.is_error()

    m3.consume_tokens(t[1:])
    assert m2.is_stopped() and m2.is_accepting() and not m2.is_error()


def check_ff(m: LLMatcher, expected: str) -> None:
    assert m.compute_ff_bytes() == expected.encode(), "FF bytes mismatch"
    assert m.compute_ff_tokens() == tokenizer().tokenize_str(expected)


def test_fast_forward() -> None:
    m = LLMatcher(tokenizer(), "start: /(foo[12]23|bar)/")
    toks = tokenizer().tokenize_str("foo123")
    assert len(toks) == 6
    check_ff(m, "")
    consume_tokens(m, toks[0:1])
    check_ff(m, "oo")
    consume_tokens(m, toks[1:2])
    check_ff(m, "o")
    consume_tokens(m, toks[2:3])
    check_ff(m, "")
    consume_tokens(m, toks[3:4])
    check_ff(m, "23")
    consume_tokens(m, toks[4:])
    assert m.is_accepting()
    assert m.is_stopped()
    assert m.stop_reason() == "NoExtension"
    assert m.compute_ff_bytes() == b""
    assert m.compute_ff_tokens() == []
    assert not m.is_error()


def test_try_consume_tokens() -> None:
    m = LLMatcher(tokenizer(), "start: /(foo[12]23|bar)/")
    tokens = tokenizer().tokenize_str("foo723")
    assert len(tokens) == 6
    assert m.try_consume_tokens(tokens) == 3
    consume_tokens(m, tokenizer().tokenize_str("123"))
    assert m.is_stopped() and m.is_accepting() and not m.is_error()


def test_consume_token_error() -> None:
    m = LLMatcher(tokenizer(), "start: /(foo[12]23|bar)/")
    m2 = m.deep_copy()
    m3 = m.deep_copy()
    m4 = m.deep_copy()
    tokens = tokenizer().tokenize_str("foo723")

    consume_tokens(m, tokens[0:3])
    mask = m.compute_logit_bias()
    assert mask[tokens[3]] == 0
    r = m.consume_token(tokens[3])
    assert r == False
    assert m.is_error()
    assert "doesn't satisfy the grammar" in m.get_error()
    # this is internal error, since the token was not in the mask
    assert m.stop_reason() == "InternalError"

    consume_tokens(m2, tokens[0:3])
    r = m2.consume_token(tokenizer().vocab_size + 100)
    assert r == False
    assert m2.is_error()
    assert "out of range" in m2.get_error()

    r = m3.consume_tokens(tokens[0:3] + [tokenizer().vocab_size + 100])
    assert r == False
    assert m3.is_error()
    assert "out of range" in m3.get_error()

    n = m4.validate_tokens(tokens[0:3] + [tokenizer().vocab_size + 100])
    assert n == 0  # questionable
    assert m3.is_error()
    assert "out of range" in m3.get_error()


def test_struct_tag_0() -> None:
    tags = [
        StructTag(trigger="<func",
                  begin="<func=foo>",
                  grammar={"type": "object"},
                  end="</func>"),
        StructTag(trigger="<func",
                  begin="<func=bar>",
                  grammar={"type": "object"},
                  end="</func>")
    ]

    f_foo = '<func=foo>{}</func>'
    f_bar = '<func=bar>{"x":1}</func>'
    inner = f"{f_foo}1\n23{f_bar}"

    for special in [False, True]:
        if special:
            tags.append(
                StructTag(trigger="<|tool|>",
                          begin="<|tool|>blah(",
                          grammar='start: /[0-9]+/ ("," /[0-9]+/)*',
                          end=")"))
        grm = StructTag.to_grammar(tags)

        check_grammar(grm, [
            f_foo,
            f_foo + f_bar,
            inner,
            "FO" + inner,
            inner + "qux",
            "AAA" + inner + "BBB",
            "123",
            "aa<Func=123",
            "</func>",
        ], [
            "FINAL_REJECT:" + f_foo[:-1],
            '<func= foo>{}</func>',
            '<func =foo>{}</func>',
            '<func=foo>1</func>',
            '<func=foo> {}</func>',
            '<func=foo>{} </func>',
        ])

        if special:
            f_tool = "⁂<|tool|>blah(11,12)"

            check_grammar(
                grm,
                [
                    f_tool,
                    "foo" + f_tool + f_bar,
                    f_tool + "qqqq" + f_bar + "mux",
                    # this should pass since it doesn't actually use special tokens:
                    '<|tool|>qux(1)',
                ],
                ["⁂<|tool|>qux(11)"])


def test_parser_limits() -> None:
    # this one needs more than initial_lexer_fuel=20
    m = LLMatcher(tokenizer(),
                  "start: /(foo[12]23|bar)/",
                  limits=LLParserLimits(initial_lexer_fuel=5))
    assert "initial lexer configuration (grammar) too big" in m.get_error()


def check_json_schema_error(msg: str, json_schema: Dict[str, Any]) -> None:
    for lenient in [True, False]:
        grm = LLMatcher.grammar_from_json_schema(
            json_schema, overrides={"lenient": lenient})
        is_err, msgs = LLMatcher.validate_grammar_with_warnings(
            grm,
            tokenizer=tokenizer(),
        )
        assert msg in msgs[0]
        if lenient:
            assert not is_err
        else:
            assert is_err

        m = LLMatcher(tokenizer(), grm)

        if lenient:
            assert msg in m.get_grammar_warnings()[0]
        else:
            assert m.is_error()
            assert msg in m.get_error()


def test_grammar_warnings() -> None:
    is_err, msgs = LLMatcher.validate_grammar_with_warnings(
        "start: <[128]>* <[123]>",
        tokenizer=None,
    )
    assert not is_err
    assert "no tokenizer" in msgs[0]

    check_json_schema_error(
        "Unknown format", {
            "type": "object",
            "properties": {
                "foo": {
                    "type": "string",
                    "format": "my_custom_format"
                }
            },
            "required": ["foo"]
        })

    check_json_schema_error('Unimplemented keys: ["not"]',
                            {"not": {
                                "type": "object"
                            }})
