import re

import llguidance as llg
import numpy as np
import pytest
from llguidance.numpy import (
    allocate_token_bitmask,
    fill_next_token_bitmask_par_with_draft_tokens,
)


class ByteTokenizer:
    bos_token_id = None
    tokens = [bytes([value]) for value in range(255)] + [
        b"<eos>",
        b"<tool_call>",
        b"</tool_call>",
        b"<think>",
        b"</think>",
        b"<function=read_files>\n<parameter=name>",
        b"<function=read_files>\n<parameter=read_files>",
        b'<function=read_files>\n{"read_files":',
        b"<tool_call>\n<function=read_files>\n<parameter=name>",
        b"<tool_call>\n<function=read_files>\n<parameter=read_files>",
        b"\n</parameter>\n</function>\n</tool_call>\nDone.",
        b"\n</parameter>\n<parameter=unexpected>",
        b"<tool_call><function=read_files><parameter=name>",
        b"<tool_call><function=read_files><parameter=wrong>",
        b"<tool_call>\t\r\n<function=read_files>\t<parameter=name>",
        b"<tool_call>\t\r\n<function=read_files>\t<parameter=wrong>",
        b"<tool_call><function=read_files_extra><parameter=anything>",
        b"<tool_call>not a function",
        b"<tool_call><function= read_files ><parameter=name>",
    ]
    eos_token_id = tokens.index(b"<eos>")
    special_token_ids = list(map(tokens.index, [b"<eos>", b"<tool_call>", b"</tool_call>", b"<think>", b"</think>"]))

    def __call__(self, value: bytes) -> list[int]:
        return list(value)


def token_id(value: bytes) -> int:
    return ByteTokenizer.tokens.index(value)


def encode(text: str, *, special=True, markers=None) -> list[int]:
    if not special:
        return list(text.encode())
    if markers is None:
        markers = {
            marker: token_id(marker.encode()) for marker in ("<tool_call>", "</tool_call>", "<think>", "</think>")
        }
    result = []
    for part in re.split(r"(<tool_call>|</tool_call>|<think>|</think>)", text):
        if part in markers:
            result.append(markers[part])
        else:
            result.extend(part.encode())
    return result


@pytest.fixture(scope="module")
def tokenizer() -> llg.LLTokenizer:
    return llg.LLTokenizer(llg.TokenizerWrapper(ByteTokenizer()), slices=[])


def tokenizer_with_markers(names):
    source = ByteTokenizer()
    source.tokens = [bytes([value]) for value in range(255)] + [b"<eos>"] + [name.encode() for name in names]
    source.eos_token_id = source.tokens.index(b"<eos>")
    markers = {name: source.tokens.index(name.encode()) for name in names}
    source.special_token_ids = [source.eos_token_id, *markers.values()]
    return llg.LLTokenizer(llg.TokenizerWrapper(source), slices=[]), markers


@pytest.fixture(
    scope="module",
    params=[
        [],
        ["<tool_call>", "</tool_call>"],
        ["<think>", "</think>"],
        ["<tool_call>", "</think>"],
        ["<tool_call>", "</tool_call>", "<think>", "</think>"],
    ],
    ids=["byte-only", "tool-markers", "reasoning-markers", "mixed-markers", "all-markers"],
)
def marker_tokenizer(request):
    return tokenizer_with_markers(request.param)


def tool(parameters=None, *, name="read_files", strict=True):
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters, "strict": strict},
    }


def call(arguments: str, *, name="read_files") -> str:
    if arguments and not arguments.endswith("\n"):
        arguments += "\n"
    return f"<tool_call>\n<function={name}>\n{arguments}</function>\n</tool_call>"


def parameter(name: str, value: str, *, padded=True) -> str:
    if padded:
        return f"<parameter={name}>\n{value}\n</parameter>\n"
    return f"<parameter={name}>{value}</parameter>"


def matcher(tokenizer, tools, *, initial_reasoning=False, special_tokens=None):
    grammar = llg.grammar_from_tools(
        tools, format="qwen3_coder", initial_reasoning=initial_reasoning, special_tokens=special_tokens
    )
    result = llg.LLMatcher(tokenizer, grammar)
    if result.is_error():
        raise ValueError(result.get_error())
    return result


def consume(result, tokens):
    for token_id in tokens:
        mask = result.compute_logit_bias()
        assert not result.is_error(), result.get_error()
        assert mask[token_id] == 200, f"Rejected token {token_id}"
        assert result.consume_token(token_id), result.get_error()


def accepts(tokenizer, tools, text, *, special=True, initial_reasoning=False):
    result = matcher(tokenizer, tools, initial_reasoning=initial_reasoning)
    for token_id in encode(text, special=special):
        mask = result.compute_logit_bias()
        assert not result.is_error(), result.get_error()
        if mask[token_id] == 0:
            return False
        assert mask[token_id] == 200
        assert result.consume_token(token_id), result.get_error()
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    return result.is_accepting() and mask[tokenizer.eos_token] == 200


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
def test_strict_call_accepts_schema_value(tokenizer, special):
    assert accepts(tokenizer, [tool()], call(parameter("name", "report.docx")), special=special)


@pytest.mark.parametrize("options", [{}, {"special_tokens": None}], ids=["omitted", "none"])
@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
def test_default_special_tokens_accept_all_format_markers(tokenizer, options, special):
    grammar = llg.grammar_from_tools([tool()], format="qwen3_coder", **options)
    result = llg.LLMatcher(tokenizer, grammar)
    text = (
        "<think>Draft: " + call(parameter("wrong", "anything")) + "</think>\n" + call(parameter("name", "report.docx"))
    )
    consume(result, encode(text, special=special))
    assert result.is_accepting()
    assert result.compute_logit_bias()[tokenizer.eos_token] == 200


@pytest.mark.parametrize("initial_reasoning", [False, True], ids=["generated-think", "prefilled-think"])
def test_special_token_overrides_preserve_strict_calls(marker_tokenizer, initial_reasoning):
    tokenizer, markers = marker_tokenizer
    tools = [tool({"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]})]
    seed = matcher(tokenizer, tools, initial_reasoning=initial_reasoning, special_tokens=list(markers))
    reasoning = (
        ("" if initial_reasoning else "<think>") + "Draft: " + call(parameter("wrong", "anything")) + "</think>\n"
    )
    prefix = reasoning + call(parameter("count", "2")) + "\n<tool_call><function=read_files><parameter=count>\n"
    ending = "3\n</parameter>\n</function>\n</tool_call>"
    for special in [False, True]:
        result = seed.deep_copy()
        consume(result, encode(prefix, special=special, markers=markers))
        mask = result.compute_logit_bias()
        assert not result.is_error(), result.get_error()
        assert mask[ord("x")] == 0
        assert mask[tokenizer.eos_token] == 0
        consume(result, encode(ending, special=special, markers=markers))
        assert result.is_accepting()
        assert result.compute_logit_bias()[tokenizer.eos_token] == 200


@pytest.mark.parametrize("special_tokens", [None, ["<tool_call>"]], ids=["format-defaults", "explicit-marker"])
def test_requested_special_tokens_must_exist_in_the_tokenizer(special_tokens):
    tokenizer, _ = tokenizer_with_markers([])
    grammar = llg.grammar_from_tools([tool()], format="qwen3_coder", special_tokens=special_tokens)
    result = llg.LLMatcher(tokenizer, grammar)
    assert result.is_error()
    assert "special token" in result.get_error().lower()


@pytest.mark.parametrize("special_tokens", [["<tool_cal>"], ["<tool_call>", "<unknown>"], ["<eos>"]])
def test_special_token_overrides_reject_names_outside_the_format(special_tokens):
    with pytest.raises(ValueError, match="(?i)special.*token|marker"):
        llg.grammar_from_tools([tool()], format="qwen3_coder", special_tokens=special_tokens)


@pytest.mark.parametrize(
    ("prefix", "marker"),
    [
        pytest.param("", "<tool_call>", id="tool-open"),
        pytest.param("", "<think>", id="reasoning-open"),
        pytest.param("<think>Plan ", "</think>", id="reasoning-close"),
        pytest.param(
            "<tool_call>\n<function=read_files>\n" + parameter("name", "report.docx") + "</function>\n",
            "</tool_call>",
            id="tool-close",
        ),
    ],
)
def test_unlisted_special_markers_cannot_bypass_text_only_rules(tokenizer, prefix, marker):
    result = matcher(tokenizer, [tool()], special_tokens=[])
    consume(result, encode(prefix, special=False))
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    assert mask[token_id(marker.encode())] == 0
    consume(result, encode(marker, special=False))


@pytest.mark.parametrize("byte_only_first", [False, True], ids=["defaults-first", "byte-only-first"])
def test_special_token_overrides_are_isolated_between_compilations(tokenizer, byte_only_first):
    byte_tokenizer, _ = tokenizer_with_markers([])
    modes = [(tokenizer, None, True), (byte_tokenizer, [], False)]
    if byte_only_first:
        modes.reverse()
    results = [(matcher(tok, [tool()], special_tokens=names), tok, special) for tok, names, special in modes]
    text = "<think>Plan</think>\n" + call(parameter("name", "report.docx"))
    for result, tok, special in results:
        consume(result, encode(text, special=special))
        assert result.is_accepting()
        assert result.compute_logit_bias()[tok.eos_token] == 200


def test_special_token_overrides_preserve_speculative_masks_and_rollback(marker_tokenizer):
    tokenizer, markers = marker_tokenizer
    tools = [tool({"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]})]
    prefix = encode("<think>Plan ", markers=markers)
    valid_draft = "</think>\n<tool_call>\n<function=read_files>\n<parameter="
    drafted = encode(valid_draft + "wrong>", markers=markers)
    accepted_count = len(encode(valid_draft, markers=markers))
    committed = encode(valid_draft + "count>\n3\n</parameter>\n</function>\n</tool_call>", markers=markers)
    result = matcher(tokenizer, tools, special_tokens=list(markers))
    consume(result, prefix)
    before = result.compute_bitmask()
    masks = allocate_token_bitmask(len(drafted) + 1, tokenizer.vocab_size)
    fill_next_token_bitmask_par_with_draft_tokens(llg.LLExecutor(1), [(result, 0, drafted)], masks)
    assert not result.is_error(), result.get_error()
    assert result.compute_bitmask() == before

    reference = matcher(tokenizer, tools, special_tokens=list(markers))
    consume(reference, prefix)
    for index in range(accepted_count + 1):
        np.testing.assert_array_equal(masks[index], np.frombuffer(reference.compute_bitmask(), dtype=np.int32))
        if index < accepted_count:
            consume(reference, [drafted[index]])
    assert reference.compute_logit_bias()[drafted[accepted_count]] == 0

    consume(result, committed)
    assert result.is_accepting()
    assert result.rollback(len(committed))
    assert result.compute_bitmask() == before
    alternative = encode("carefully.</think>\n" + call(parameter("count", "4")), markers=markers)
    consume(result, alternative)
    fresh = matcher(tokenizer, tools, special_tokens=list(markers))
    consume(fresh, prefix + alternative)
    assert result.compute_bitmask() == fresh.compute_bitmask()
    assert result.is_accepting()


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param('{"read_files": "{\\"name\\": \\"report.docx\\"}"}', id="original-wrapper"),
        pytest.param(parameter("read_files", '{"name":"report.docx"}'), id="wrapped-parameter"),
        pytest.param(parameter("file_name", "report.docx"), id="misnamed-parameter"),
        pytest.param("", id="missing-required"),
        pytest.param(
            parameter("name", "report.docx") + parameter("name", "report.xlsx"),
            id="duplicate-parameter",
        ),
        pytest.param(
            parameter("name", "report.docx") + parameter("unexpected", "yes"),
            id="extra-parameter",
        ),
    ],
)
def test_strict_arguments_reject_malformed_calls(tokenizer, arguments):
    assert accepts(tokenizer, [tool()], call(parameter("name", "report.docx")))
    assert not accepts(tokenizer, [tool()], call(arguments))


@pytest.mark.parametrize("padded", [True, False], ids=["template", "compact"])
@pytest.mark.parametrize(
    ("schema", "value"),
    [
        pytest.param({"type": "string"}, 'C:\\work & <draft> "quoted"\nČaša 🧪', id="raw-string"),
        pytest.param({"type": "string"}, "", id="empty-string"),
        pytest.param({"type": "string"}, '<html lang="en">\n<p>A & B</p>\n</html>', id="html-string"),
        pytest.param({"type": "string"}, "null", id="literal-null-string"),
        pytest.param({"type": "string", "enum": ["red", "blue"]}, "blue", id="string-enum"),
        pytest.param({"type": "string", "pattern": "^[a-z]{3}$"}, "abc", id="string-pattern"),
        pytest.param({"type": "string", "minLength": 2, "maxLength": 2}, "é🧪", id="unicode-length"),
        pytest.param({"type": "integer", "minimum": 1, "maximum": 5}, "3", id="integer"),
        pytest.param({"type": "number"}, "-2.5", id="number"),
        pytest.param({"type": "boolean"}, "false", id="boolean"),
        pytest.param({"type": "null"}, "null", id="null"),
        pytest.param({"type": ["integer", "null"]}, "null", id="nullable-integer"),
        pytest.param({"type": "array", "items": {"type": "integer"}}, "[1,2]", id="json-array"),
        pytest.param(
            {
                "type": "object",
                "properties": {"title": {"type": "string"}, "ready": {"type": "boolean"}},
                "required": ["title", "ready"],
                "additionalProperties": False,
            },
            '{"title":"report","ready":true}',
            id="nested-json-object",
        ),
        pytest.param(
            {"type": "object", "additionalProperties": {"type": "integer"}},
            '{"apples":2,"pears":3}',
            id="explicit-nested-map",
        ),
    ],
)
def test_parameter_values_follow_their_schema(tokenizer, schema, value, padded):
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", value, padded=padded)))


@pytest.mark.parametrize(
    ("schema", "valid", "value"),
    [
        pytest.param({"type": "integer"}, "3", '"3"', id="quoted-integer"),
        pytest.param({"type": "integer"}, "3", "3.5", id="fractional-integer"),
        pytest.param({"type": "boolean"}, "true", "yes", id="wrong-boolean"),
        pytest.param({"type": "number"}, "1.25", "many", id="wrong-number"),
        pytest.param({"type": "null"}, "null", "None", id="wrong-null"),
        pytest.param({"type": "string", "enum": ["red", "blue"]}, "blue", "green", id="wrong-enum"),
        pytest.param({"type": "string", "pattern": "^[a-z]{3}$"}, "abc", "abcd", id="wrong-pattern"),
        pytest.param({"type": "string", "minLength": 2}, "ab", "a", id="short-string"),
        pytest.param({"type": "string", "maxLength": 2}, "ab", "abc", id="long-string"),
        pytest.param({"type": "integer", "minimum": 1, "maximum": 5}, "5", "6", id="out-of-range"),
        pytest.param({"type": "array", "items": {"type": "integer"}}, "[1,2]", '[1,"two"]', id="wrong-array-item"),
        pytest.param({"type": "object"}, "{}", '"{\\"name\\":1}"', id="stringified-object"),
        pytest.param(
            {"type": "object", "additionalProperties": {"type": "integer"}},
            '{"apples":2}',
            '{"apples":"two"}',
            id="wrong-map-value",
        ),
    ],
)
def test_parameter_values_reject_schema_violations(tokenizer, schema, valid, value):
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", valid)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", value)))


@pytest.mark.parametrize("include_optional", [False, True])
def test_optional_parameters_stay_optional(tokenizer, include_optional):
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "force": {"type": "boolean"}},
        "required": ["name"],
    }
    arguments = parameter("name", "report.docx")
    if include_optional:
        arguments += "\n" + parameter("force", "true")
    assert accepts(tokenizer, [tool(parameters)], call(arguments))


def test_root_ref_stays_an_xml_parameter_list(tokenizer):
    parameters = {
        "$defs": {
            "arguments": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
        "$ref": "#/$defs/arguments",
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("name", "report.docx")))
    assert not accepts(tokenizer, [tool(parameters)], call('{"name":"report.docx"}'))


def test_local_ref_preserves_raw_string_encoding(tokenizer):
    parameters = {
        "$defs": {"filename": {"type": "string", "enum": ["report.docx"]}},
        "type": "object",
        "properties": {"name": {"$ref": "#/$defs/filename"}},
        "required": ["name"],
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("name", "report.docx")))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("name", "report.xlsx")))


@pytest.mark.parametrize("nested_first", [False, True], ids=["raw-first", "json-first"])
@pytest.mark.parametrize(
    "definition",
    [{"type": "string"}, {"const": "report.docx"}, {"enum": ["report.docx", "report.xlsx"]}],
    ids=["string", "const", "enum"],
)
def test_same_ref_uses_raw_and_nested_json_encoding(tokenizer, definition, nested_first):
    properties = {
        "name": {"$ref": "#/$defs/filename"},
        "names": {"type": "array", "items": {"$ref": "#/$defs/filename"}},
    }
    if nested_first:
        properties = dict(reversed(list(properties.items())))
    parameters = {
        "$defs": {"filename": definition},
        "type": "object",
        "properties": properties,
        "required": ["name", "names"],
    }
    values = {"name": "report.docx", "names": '["report.docx"]'}
    arguments = "".join(parameter(name, values[name]) for name in properties)
    assert accepts(tokenizer, [tool(parameters)], call(arguments))
    values["names"] = "[report.docx]"
    invalid = "".join(parameter(name, values[name]) for name in properties)
    assert not accepts(tokenizer, [tool(parameters)], call(invalid))
    if "type" not in definition:
        values = {"name": '"report.docx"', "names": '["report.docx"]'}
        invalid = "".join(parameter(name, values[name]) for name in properties)
        assert not accepts(tokenizer, [tool(parameters)], call(invalid))


def test_recursive_nested_ref_stays_json(tokenizer):
    parameters = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {"next": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/node"}]}},
                "required": ["next"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"head": {"$ref": "#/$defs/node"}},
        "required": ["head"],
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("head", '{"next":{"next":null}}')))
    for invalid in ['{"next":{}}', '{"next":1}', '{"next":{"next":null,"extra":0}}']:
        assert not accepts(tokenizer, [tool(parameters)], call(parameter("head", invalid)))


@pytest.mark.parametrize("strict", [False, None, 1, "true"], ids=["false", "missing", "one", "string"])
def test_only_boolean_true_enables_constraints(tokenizer, strict):
    definition = tool(strict=strict)
    if strict is None:
        del definition["function"]["strict"]
    assert accepts(tokenizer, [definition], call('{"unexpected": "unconstrained"}'))


def test_mixed_tools_leave_nonstrict_and_unknown_calls_free(tokenizer):
    tools = [tool(), tool(name="loose", strict=False)]
    text = (
        "Preparing <parameter=anything>ordinary text</parameter>.\n"
        + call('{"unconstrained":true}', name="loose")
        + call("arbitrary arguments", name="unknown_tool")
        + call(parameter("name", "report.docx"))
        + "\nFinished."
    )
    assert accepts(tokenizer, tools, text)
    assert not accepts(tokenizer, tools, text + call(parameter("wrong_name", "report.docx")))


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize(
    "gap", ["", " ", "\t", "\n", "\r\n", "\n \t\n"], ids=["compact", "space", "tab", "lf", "crlf", "mixed"]
)
def test_header_whitespace_cannot_bypass_a_known_strict_schema(tokenizer, gap, special):
    tools = [
        tool(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["report.docx"]},
                    "count": {"type": "integer"},
                },
                "required": ["name", "count"],
            }
        )
    ]
    header = f"<tool_call>{gap}<function=read_files>{gap}"
    ending = "</function>\n</tool_call>"
    valid = parameter("name", "report.docx") + parameter("count", "3")
    assert accepts(tokenizer, tools, header + valid + ending, special=special)
    for invalid in [
        parameter("wrong", "report.docx") + parameter("count", "3"),
        parameter("name", "report.docx") + parameter("count", '"3"'),
        valid + parameter("unexpected", "yes"),
        parameter("name", "report.xlsx") + parameter("count", "3"),
        '{"read_files": "{\\"name\\": \\"report.docx\\"}"}',
    ]:
        assert not accepts(tokenizer, tools, header + invalid + ending, special=special), invalid


@pytest.mark.parametrize("name", [" read_files", "read_files ", "\tread_files\n", "\u00a0read_files\u2003"])
def test_whitespace_aliases_of_strict_names_are_rejected(tokenizer, name):
    # Whitespace aliases of strict function names must not bypass their schemas.
    parameters = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    tools = [tool(parameters)]
    arguments = parameter("count", "3")
    assert accepts(tokenizer, tools, call(arguments))
    assert not accepts(tokenizer, tools, call(arguments, name=name))


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize("suffix", ["", "garbage", "\n</tool_call>", "<function=>\n</function></tool_call>"])
def test_tool_call_marker_requires_a_function_envelope(tokenizer, special, suffix):
    text = "Planning. <tool_call>" + suffix
    assert accepts(tokenizer, [tool()], "Planning. " + call("free", name="message_added_tool"), special=special)
    assert not accepts(tokenizer, [tool()], text, special=special)
    assert accepts(tokenizer, [tool()], "<think>" + text + "</think>Done.", special=special)


@pytest.mark.parametrize("suffix", ["", "<tool_call>", "<tool_call>garbage"])
def test_no_strict_tools_allow_free_text_and_tool_shaped_text(tokenizer, suffix):
    text = "A draft: " + call("not valid arguments") + suffix
    assert accepts(tokenizer, [], text)
    assert accepts(tokenizer, [tool(strict=False)], text)


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize("initial_reasoning", [False, True], ids=["generated-think", "prefilled-think"])
def test_reasoning_is_free_but_calls_after_reasoning_are_strict(tokenizer, special, initial_reasoning):
    prefix = "" if initial_reasoning else "<think>"
    reasoning = prefix + "Bad example: " + call(parameter("wrong", "value")) + "</think>\n"
    valid = reasoning + call(parameter("name", "report.docx"))
    invalid = reasoning + call(parameter("wrong", "value"))
    assert accepts(tokenizer, [tool()], valid, special=special, initial_reasoning=initial_reasoning)
    assert not accepts(tokenizer, [tool()], invalid, special=special, initial_reasoning=initial_reasoning)


@pytest.mark.parametrize(
    ("prefix", "valid_token", "invalid_tokens"),
    [
        pytest.param(
            "<tool_call>\n",
            b"<function=read_files>\n<parameter=name>",
            [b"<function=read_files>\n<parameter=read_files>", b'<function=read_files>\n{"read_files":'],
            id="after-special-open",
        ),
        pytest.param(
            "<tool_call>",
            b"<function=read_files>\n<parameter=name>",
            [b"<function=read_files>\n<parameter=read_files>", b'<function=read_files>\n{"read_files":'],
            id="immediately-after-special-open",
        ),
        pytest.param(
            "",
            b"<tool_call>\n<function=read_files>\n<parameter=name>",
            [b"<tool_call>\n<function=read_files>\n<parameter=read_files>"],
            id="whole-byte-header",
        ),
        pytest.param(
            "",
            b"<tool_call><function=read_files><parameter=name>",
            [
                b"<tool_call><function=read_files><parameter=wrong>",
                b"<tool_call>not a function",
                b"<tool_call><function= read_files ><parameter=name>",
            ],
            id="compact-byte-header",
        ),
        pytest.param(
            "",
            b"<tool_call>\t\r\n<function=read_files>\t<parameter=name>",
            [b"<tool_call>\t\r\n<function=read_files>\t<parameter=wrong>"],
            id="whitespace-byte-header",
        ),
        pytest.param(
            "",
            b"<tool_call><function=read_files_extra><parameter=anything>",
            [b"<tool_call><function=read_files><parameter=wrong>"],
            id="unknown-prefix-overlap",
        ),
    ],
)
def test_tokens_crossing_a_strict_header_are_masked_atomically(tokenizer, prefix, valid_token, invalid_tokens):
    result = matcher(tokenizer, [tool()])
    consume(result, encode(prefix))
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    assert mask[token_id(valid_token)] == 200
    for invalid_token in invalid_tokens:
        assert mask[token_id(invalid_token)] == 0


@pytest.mark.parametrize(
    "prefix",
    [
        "<tool_call>",
        "<tool_call> \t\n",
        "<tool_call><function=",
        "<tool_call><function=read_",
        "<tool_call><function=read_files>",
        "<tool_call>\n<function=read_files>\n",
        "<tool_call>\n<function=read_files>\n<parameter=name>report",
        "<tool_call>\n<function=read_files>\n<parameter=name>report.docx</parameter>",
        "<tool_call>\n<function=read_files>\n<parameter=name>report.docx</parameter>\n</function>\n",
        "<tool_call><function=loose>free arguments",
        "<tool_call><function=unknown_tool>free arguments",
        "<tool_call><function=unknown_tool>free arguments</function>",
    ],
)
@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
def test_eos_is_blocked_inside_an_incomplete_tool_envelope(tokenizer, prefix, special):
    result = matcher(tokenizer, [tool(), tool(name="loose", strict=False)])
    consume(result, encode(prefix, special=special))
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    assert not result.is_accepting()
    assert mask[tokenizer.eos_token] == 0


def test_rollback_restores_state_across_the_strict_header(tokenizer):
    tools = [tool(), tool(name="loose", strict=False)]
    result = matcher(tokenizer, tools)
    prefix = encode("Planning. ")
    consume(result, prefix)
    before = result.compute_bitmask()
    draft = encode(call(parameter("name", "report.docx")))
    consume(result, draft)
    assert result.rollback(len(draft))
    assert result.compute_bitmask() == before
    alternative = encode(call("free arguments", name="loose"))
    consume(result, alternative)
    reference = matcher(tokenizer, tools)
    consume(reference, prefix + alternative)
    assert result.compute_bitmask() == reference.compute_bitmask()
    assert result.is_accepting()


@pytest.mark.parametrize(("draft_text", "accepted_count"), [("report.docx", 11), ("report.z", 7)])
def test_speculative_masks_restore_state_and_match_sequential_decoding(tokenizer, draft_text, accepted_count):
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["report.docx", "report.xlsx"]}},
        "required": ["name"],
    }
    tools = [tool(parameters)]
    prefix = encode("<tool_call>\n<function=read_files>\n<parameter=name>")
    result = matcher(tokenizer, tools)
    consume(result, prefix)
    before = result.compute_bitmask()
    draft = list(draft_text.encode())
    masks = allocate_token_bitmask(len(draft) + 1, tokenizer.vocab_size)
    fill_next_token_bitmask_par_with_draft_tokens(llg.LLExecutor(1), [(result, 0, draft)], masks)
    assert not result.is_error(), result.get_error()
    assert result.compute_bitmask() == before

    reference = matcher(tokenizer, tools)
    consume(reference, prefix)
    for index in range(accepted_count + 1):
        np.testing.assert_array_equal(masks[index], np.frombuffer(reference.compute_bitmask(), dtype=np.int32))
        if index < accepted_count:
            consume(reference, [draft[index]])
    if accepted_count < len(draft):
        assert reference.compute_logit_bias()[draft[accepted_count]] == 0
    else:
        assert reference.compute_logit_bias()[ord("<")] == 200

    committed = encode("report.docx</parameter>\n</function>\n</tool_call>")
    consume(result, committed)
    fresh = matcher(tokenizer, tools)
    consume(fresh, prefix + committed)
    assert result.compute_bitmask() == fresh.compute_bitmask()
    assert result.is_accepting()


def test_raw_string_cannot_contain_its_closing_delimiter(tokenizer):
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "pattern": "^before .* after$"}},
        "required": ["name"],
    }
    result = matcher(tokenizer, [tool(parameters)])
    consume(result, encode("<tool_call>\n<function=read_files>\n<parameter=name>before </parameter"))
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    assert mask[ord(">")] == 0
    consume(result, encode("x after</parameter>\n</function>\n</tool_call>"))
    assert result.is_accepting()


@pytest.mark.parametrize("format_name", ["unknown", "qwen_xml"])
def test_unknown_format_is_rejected(format_name):
    with pytest.raises(ValueError, match="(?i)format"):
        llg.grammar_from_tools([tool()], format=format_name)


def test_duplicate_function_names_are_rejected():
    with pytest.raises(ValueError, match="(?i)duplicate"):
        llg.grammar_from_tools([tool(), tool()], format="qwen3_coder")


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        pytest.param(
            {"type": "object", "unevaluatedProperties": False}, "unevaluatedProperties", id="unsupported-keyword"
        ),
        pytest.param(
            {"$ref": "https://example.invalid/tool-schema.json"}, "(?i)ref|resource|retrieve", id="remote-reference"
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": {"type": "integer"},
            },
            "additionalProperties",
            id="schema-valued-top-level-additional-properties",
        ),
    ],
)
def test_unsupported_strict_schemas_fail_explicitly(tokenizer, parameters, message):
    with pytest.raises(ValueError, match=message):
        matcher(tokenizer, [tool(parameters)])


def test_public_qwen3_coder_template_call(tokenizer):
    # Tool-call body layout from the public Qwen3-Coder template:
    # https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/chat_template.jinja
    text = """<tool_call>
<function=read_files>
<parameter=name>
report.docx
</parameter>
</function>
</tool_call>"""
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["report.docx"]}},
        "required": ["name"],
    }
    assert accepts(tokenizer, [tool(parameters)], text)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "enum": ["blue"]},
        {"type": "string", "pattern": "^[a-z]+$", "minLength": 4, "maxLength": 4},
    ],
    ids=["enum", "pattern-and-length"],
)
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("blue", True, id="compact"),
        pytest.param("\nblue\n", True, id="template"),
        pytest.param("\nblue", True, id="leading-only"),
        pytest.param("blue\n", True, id="trailing-only"),
        pytest.param("\n\nblue\n", False, id="extra-leading-newline"),
        pytest.param("\nblue\n\n", False, id="extra-trailing-newline"),
        pytest.param(" blue ", False, id="spaces-are-value"),
        pytest.param("\r\nblue\r\n", False, id="crlf-is-not-lf-padding"),
    ],
)
def test_constraints_apply_after_removing_one_edge_newline(tokenizer, schema, body, expected):
    # Qwen3-Coder's parser removes one leading and one trailing LF from parameter values:
    # https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", "blue")))
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", body, padded=False))) == expected


@pytest.mark.parametrize(
    ("value", "valid_body", "invalid_body"),
    [
        pytest.param("\nblue", "\n\nblue", "\nblue", id="leading-newline-in-value"),
        pytest.param("blue\n", "blue\n\n", "blue\n", id="trailing-newline-in-value"),
        pytest.param("\nblue\n", "\n\nblue\n\n", "\nblue\n", id="both-newlines-in-value"),
        pytest.param(" blue ", "\n blue \n", "\nblue\n", id="spaces-preserved"),
    ],
)
def test_string_constants_preserve_whitespace_in_the_parsed_value(tokenizer, value, valid_body, invalid_body):
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string", "const": value}},
        "required": ["value"],
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", valid_body, padded=False)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", invalid_body, padded=False)))


@pytest.mark.parametrize("whitespace", ["", "\n", " \t", "\r\n\t "], ids=["none", "lf", "spaces", "mixed"])
def test_whitespace_between_parameters_and_before_function_end(tokenizer, whitespace):
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}, "enabled": {"type": "boolean"}},
        "required": ["count", "enabled"],
    }
    arguments = whitespace.join(
        [
            parameter("count", "3").removesuffix("\n"),
            parameter("enabled", "true").removesuffix("\n"),
        ]
    )
    text = f"<tool_call>\n<function=read_files>{whitespace}{arguments}{whitespace}</function>\n</tool_call>"
    assert accepts(tokenizer, [tool(parameters)], text)


@pytest.mark.parametrize("explicit_required", [False, True], ids=["omitted-required", "empty-required"])
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param([], True, id="skip-all"),
        pytest.param(["name"], True, id="skip-trailing"),
        pytest.param(["force"], True, id="skip-leading"),
        pytest.param(["name", "force"], True, id="schema-order"),
        pytest.param(["force", "name"], False, id="reversed"),
    ],
)
def test_all_optional_parameters_allow_ordered_subsets(tokenizer, fields, expected, explicit_required):
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "force": {"type": "boolean"}},
        "additionalProperties": False,
    }
    if explicit_required:
        parameters["required"] = []
    values = {"name": "report.docx", "force": "false"}
    arguments = "".join(parameter(name, values[name]) for name in fields)
    assert accepts(tokenizer, [tool(parameters)], call(arguments)) == expected
    assert not accepts(tokenizer, [tool(parameters)], call(arguments + parameter("unknown", "x")))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("force", "not-a-boolean")))
    if expected and fields:
        assert not accepts(tokenizer, [tool(parameters)], call(arguments + parameter(fields[0], values[fields[0]])))


@pytest.mark.parametrize("include_parameters", [False, True], ids=["omitted-parameters", "empty-object"])
def test_zero_argument_tool_accepts_only_an_empty_parameter_list(tokenizer, include_parameters):
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    definition = tool(parameters)
    if not include_parameters:
        del definition["function"]["parameters"]
    assert accepts(tokenizer, [definition], call(""))
    assert not accepts(tokenizer, [definition], call(parameter("name", "unexpected")))


@pytest.mark.parametrize(
    ("additional", "expected"), [(None, False), (False, False), (True, True)], ids=["omitted", "false", "true"]
)
def test_top_level_additional_properties_controls_extra_parameters(tokenizer, additional, expected):
    parameters = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    if additional is not None:
        parameters["additionalProperties"] = additional
    arguments = parameter("count", "1")
    assert accepts(tokenizer, [tool(parameters)], call(arguments))
    assert accepts(tokenizer, [tool(parameters)], call(arguments + parameter("extra", "3"))) == expected


@pytest.mark.parametrize("padded", [True, False], ids=["template", "compact"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty-string"),
        pytest.param('Raw <text> & "quotes"\nČaša 🧪', id="raw-string"),
        pytest.param("7", id="integer"),
        pytest.param("-1.25", id="number"),
        pytest.param("true", id="boolean"),
        pytest.param("null", id="null"),
        pytest.param('[1,"two",false]', id="json-array"),
        pytest.param('{"count":2,"tags":["a"]}', id="json-object"),
    ],
)
def test_additional_properties_true_accepts_unconstrained_values(tokenizer, value, padded):
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": True,
    }
    arguments = parameter("count", "1", padded=padded) + parameter("extra", value, padded=padded)
    assert accepts(tokenizer, [tool(parameters)], call(arguments))


def test_additional_properties_true_accepts_duplicate_extra_names(tokenizer):
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": True,
    }
    arguments = parameter("count", "1") + parameter("extra", "first") + parameter("extra", "second")
    assert accepts(tokenizer, [tool(parameters)], call(arguments))


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param([("extra", "1")], id="missing-required"),
        pytest.param([("count", '"1"')], id="wrong-required-type"),
        pytest.param([("count", "1"), ("enabled", "yes")], id="wrong-optional-type"),
        pytest.param([("count", "1"), ("mode", "invalid")], id="wrong-optional-enum"),
        pytest.param([("count", "1"), ("count", "2")], id="duplicate-required"),
        pytest.param([("count", "1"), ("enabled", "true"), ("enabled", "false")], id="duplicate-optional"),
        pytest.param([("count", "1"), ("mode", "fast"), ("enabled", "true")], id="reordered-declared-parameters"),
        pytest.param([("extra", "x"), ("count", "1")], id="extra-before-declared"),
        pytest.param([("count", "1"), ("extra", "x"), ("enabled", "true")], id="declared-after-extra"),
        pytest.param([("count", "1"), ("extra", "x"), ("count", "2")], id="duplicate-declared-after-extra"),
    ],
)
def test_additional_properties_cannot_bypass_declared_parameter_constraints(tokenizer, fields):
    parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "enabled": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["fast", "slow"]},
        },
        "required": ["count"],
        "additionalProperties": True,
    }
    valid = parameter("count", "1") + parameter("enabled", "true") + parameter("mode", "fast") + parameter("extra", "x")
    assert accepts(tokenizer, [tool(parameters)], call(valid))
    assert accepts(tokenizer, [tool(parameters)], call(parameter("count", "1") + parameter("extra", "x")))
    assert not accepts(tokenizer, [tool(parameters)], call("".join(parameter(name, value) for name, value in fields)))


@pytest.mark.parametrize("name", ["count_extra", "coun", "countx", "file-name", "file.name", "文档"])
def test_additional_parameter_names_allow_prefix_overlaps_and_unicode(tokenizer, name):
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": True,
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("count", "1") + parameter(name, "raw text")))


@pytest.mark.parametrize("include_properties", [False, True], ids=["omitted-properties", "empty-properties"])
def test_additional_properties_true_without_declared_parameters(tokenizer, include_properties):
    parameters = {"type": "object", "additionalProperties": True}
    if include_properties:
        parameters["properties"] = {}
    assert accepts(tokenizer, [tool(parameters)], call(""))
    arguments = parameter("note", "raw text") + parameter("count", "3") + parameter("data", '{"ready":true}')
    assert accepts(tokenizer, [tool(parameters)], call(arguments))
    assert not accepts(tokenizer, [tool(parameters)], call('{"note":"raw text"}'))


@pytest.mark.parametrize("extra_names", [[], ["extra", "another"]], ids=["below-minimum", "above-maximum"])
def test_additional_parameters_count_toward_property_limits(tokenizer, extra_names):
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": True,
        "minProperties": 2,
        "maxProperties": 2,
    }
    required = parameter("count", "1")
    assert accepts(tokenizer, [tool(parameters)], call(required + parameter("extra", "2")))
    arguments = required + "".join(parameter(name, "2") for name in extra_names)
    assert not accepts(tokenizer, [tool(parameters)], call(arguments))


@pytest.mark.parametrize("padded", [True, False], ids=["template", "compact"])
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param(["count", "label"], True, id="required-schema-order"),
        pytest.param(["label", "count"], False, id="required-reversed"),
        pytest.param(["count", "force", "label"], True, id="all-schema-order"),
        pytest.param(["count", "label", "force"], False, id="optional-last"),
        pytest.param(["force", "count", "label"], False, id="optional-first"),
        pytest.param(["force", "label", "count"], False, id="optional-first-reversed"),
        pytest.param(["label", "count", "force"], False, id="optional-last-reversed"),
        pytest.param(["label", "force", "count"], False, id="all-reversed"),
        pytest.param([], False, id="missing-both-required"),
        pytest.param(["count"], False, id="missing-label"),
        pytest.param(["label"], False, id="missing-count"),
        pytest.param(["count", "force"], False, id="optional-cannot-replace-label"),
        pytest.param(["force", "label"], False, id="optional-cannot-replace-count"),
        pytest.param(["count", "count", "label"], False, id="adjacent-duplicate-required"),
        pytest.param(["count", "label", "count"], False, id="separated-duplicate-required"),
        pytest.param(["count", "force", "force", "label"], False, id="adjacent-duplicate-optional"),
    ],
)
def test_schema_order_preserves_required_fields_and_uniqueness(tokenizer, fields, expected, padded):
    parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "force": {"type": "boolean"},
            "label": {"type": "string"},
        },
        "required": ["count", "label"],
    }
    values = {"count": "2", "force": "true", "label": "report"}
    valid = parameter("count", "2") + parameter("label", "report")
    assert accepts(tokenizer, [tool(parameters)], call(valid))
    arguments = "".join(parameter(name, values[name], padded=padded) for name in fields)
    assert accepts(tokenizer, [tool(parameters)], call(arguments)) == expected


@pytest.mark.parametrize("explicit_required", [False, True], ids=["omitted-required", "empty-required"])
def test_parameter_count_bounds_with_optional_properties_are_unsupported(explicit_required):
    parameters = {
        "type": "object",
        "properties": {name: {"type": "integer"} for name in ["a", "b", "c"]},
        "minProperties": 2,
        "maxProperties": 2,
        "additionalProperties": False,
    }
    if explicit_required:
        parameters["required"] = []
    with pytest.raises(ValueError, match="min/maxProperties|minProperties|maxProperties"):
        llg.grammar_from_tools([tool(parameters)], format="qwen3_coder")


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param('{"title":"report"}', id="missing-ready"),
        pytest.param('{"ready":true}', id="missing-title"),
        pytest.param('{"title":"report","ready":"true"}', id="wrong-nested-type"),
        pytest.param('{"title":"report","ready":true,"extra":1}', id="extra-nested-key"),
        pytest.param('{"title":report,"ready":true}', id="unquoted-json-string"),
        pytest.param("<parameter=title>report</parameter><parameter=ready>true</parameter>", id="nested-xml"),
    ],
)
def test_nested_object_constraints_are_not_lost(tokenizer, invalid):
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "ready": {"type": "boolean"}},
        "required": ["title", "ready"],
        "additionalProperties": False,
    }
    parameters = {"type": "object", "properties": {"data": schema}, "required": ["data"]}
    assert accepts(tokenizer, [tool(parameters)], call(parameter("data", '{"title":"report","ready":true}')))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("data", invalid)))


@pytest.mark.parametrize(
    ("schema", "valid", "invalid"),
    [
        pytest.param({"type": "array", "items": {"type": "string"}}, "[]", "[unquoted]", id="empty-string-array"),
        pytest.param({"type": "array", "items": {"type": "string"}}, '["a","b"]', '["a",]', id="array-syntax"),
        pytest.param({"type": "array", "items": {"type": "integer"}, "minItems": 2}, "[1,2]", "[1]", id="min-items"),
        pytest.param({"type": "array", "items": {"type": "integer"}, "maxItems": 1}, "[1]", "[1,2]", id="max-items"),
        pytest.param(
            {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": False, "minItems": 2},
            '["one",2]',
            '[2,"one"]',
            id="tuple-types",
        ),
        pytest.param({"type": "string", "format": "email"}, "reader@example.com", "not-an-email", id="string-format"),
        pytest.param(
            {"type": "string", "pattern": "^[a-z]+$", "maxLength": 3}, "abc", "abcd", id="pattern-and-max-length"
        ),
        pytest.param({"type": "string", "pattern": "^[a-z]+$", "minLength": 2}, "ab", "a", id="pattern-and-min-length"),
    ],
)
def test_container_and_combined_constraints(tokenizer, schema, valid, invalid):
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", valid)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", invalid)))


@pytest.mark.parametrize(
    ("schema", "valid", "invalid"),
    [
        pytest.param({"const": "ready"}, ["ready"], '"ready"', id="raw-string-const"),
        pytest.param({"const": 7}, ["7"], "8", id="integer-const"),
        pytest.param({"const": True}, ["true"], "false", id="boolean-const"),
        pytest.param({"const": None}, ["null"], "false", id="null-const"),
        pytest.param({"const": '"\\'}, ['"\\'], '\\"\\\\', id="raw-quote-and-backslash"),
        pytest.param({"enum": ["ready", 7, False]}, ["ready", "7", "false"], "8", id="mixed-enum"),
        pytest.param(
            {"anyOf": [{"const": "ready"}, {"const": "done"}]}, ["ready", "done"], "pending", id="string-union"
        ),
        pytest.param({"anyOf": [{"const": "ready"}, {"type": "integer"}]}, ["ready", "7"], "no", id="mixed-union"),
        pytest.param({"type": ["integer", "null"]}, ["7", "null"], "false", id="nullable-integer-both-branches"),
        pytest.param(
            {"allOf": [{"type": "integer", "minimum": 2}, {"maximum": 4}]}, ["2", "4"], "5", id="intersection"
        ),
        pytest.param({"oneOf": [{"type": "integer"}, {"type": "null"}]}, ["1", "null"], "false", id="disjoint-one-of"),
    ],
)
def test_literal_and_composed_schemas_preserve_value_constraints(tokenizer, schema, valid, invalid):
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    for value in valid:
        assert accepts(tokenizer, [tool(parameters)], call(parameter("value", value)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", invalid)))


@pytest.mark.parametrize(
    "second_name", ["save_report", "read_files_extra", "read"], ids=["distinct", "longer-prefix", "shorter-prefix"]
)
def test_multiple_strict_tools_keep_their_own_schemas(tokenizer, second_name):
    second = tool(
        {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}, name=second_name
    )
    tools = [tool(), second]
    first_call = call(parameter("name", "report.docx"))
    second_call = call(parameter("count", "3"), name=second_name)
    assert accepts(tokenizer, tools, first_call + "\n" + second_call + "\n" + first_call)
    assert not accepts(tokenizer, tools, call(parameter("count", "3")))
    assert not accepts(tokenizer, tools, first_call + call(parameter("name", "report.docx"), name=second_name))


@pytest.mark.parametrize("name", ["read", "read_files_extra", "read_filesx", "read_file", "files.create_document"])
@pytest.mark.parametrize("gap", ["", "\n", " \t"], ids=["compact", "template", "whitespace"])
def test_nonstrict_or_unknown_names_stay_free(tokenizer, name, gap):
    for tools in [[tool()], [tool(), tool(name=name, strict=False)]]:
        text = f"<tool_call>{gap}<function={name}>{gap}not structured arguments</function>\n</tool_call>" + call(
            parameter("name", "report.docx")
        )
        assert accepts(tokenizer, tools, text)
        assert not accepts(tokenizer, tools, text + call(parameter("wrong", "report.docx")))


def test_schemas_are_isolated_between_tools_and_compilations(tokenizer):
    def definition(name, value):
        return tool(
            {
                "$defs": {"Value": {"const": value}},
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/Value"}},
                "required": ["name"],
            },
            name=name,
        )

    original = [definition("read_files", "report.docx"), definition("save_report", "report.xlsx")]
    changed = [definition("read_files", "report.pdf")]
    first = matcher(tokenizer, original)
    second = matcher(tokenizer, changed)
    consume(second, encode(call(parameter("name", "report.pdf"))))
    consume(
        first,
        encode(call(parameter("name", "report.docx")) + call(parameter("name", "report.xlsx"), name="save_report")),
    )
    assert first.is_accepting() and second.is_accepting()
    assert not accepts(tokenizer, changed, call(parameter("name", "report.docx")))
    assert not accepts(tokenizer, original, call(parameter("name", "report.docx"), name="save_report"))


def test_nonstrict_schema_errors_are_ignored(tokenizer):
    tools = [tool(), tool({"type": "not-a-schema-type"}, name="loose", strict=False)]
    assert accepts(tokenizer, tools, call("free", name="loose") + call(parameter("name", "report.docx")))


def test_many_optional_parameters_fit_default_limits(tokenizer):
    properties = {f"field_{index:02d}": {"type": "integer"} for index in range(50)}
    required = ["field_00", "field_25", "field_49"]
    tools = [tool({"type": "object", "properties": properties, "required": required})]
    all_arguments = "".join(parameter(name, "1") for name in properties)
    required_arguments = "".join(parameter(name, "1") for name in required)
    missing_last = "".join(parameter(name, "1") for name in required[:-1])

    assert accepts(tokenizer, tools, call(all_arguments))
    assert accepts(tokenizer, tools, call(required_arguments))
    assert not accepts(tokenizer, tools, call(missing_last))
    assert not accepts(tokenizer, tools, call(missing_last + parameter("field_49", '"1"')))


def test_many_tools_fit_default_limits(tokenizer):
    tools = [
        tool(
            {
                "type": "object",
                "properties": {"value": {"type": "integer", "const": index}},
                "required": ["value"],
            },
            name=f"tool_{index}",
        )
        for index in range(100)
    ]
    calls = "\n".join(call(parameter("value", str(index)), name=f"tool_{index}") for index in range(100))

    assert accepts(tokenizer, tools, calls)
    assert not accepts(tokenizer, tools, call(parameter("value", "0"), name="tool_99"))
    assert not accepts(tokenizer, tools, call(parameter("value", "99"), name="tool_0"))


@pytest.mark.parametrize("name", ["file-name", "文档"], ids=["hyphenated-name", "unicode-name"])
def test_declared_parameter_names_are_preserved(tokenizer, name):
    parameters = {
        "type": "object",
        "properties": {name: {"type": "string"}},
        "required": [name],
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter(name, "report")))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("wrong", "report")))


@pytest.mark.parametrize(
    "prefix",
    ["", "<think></think>", "<think>\n\n</think>\n\n"],
    ids=["no-reasoning", "empty-reasoning", "template-empty-reasoning"],
)
def test_plain_response_and_strict_call_after_optional_reasoning(tokenizer, prefix):
    assert accepts(tokenizer, [tool()], prefix + "No tool is needed.")
    assert accepts(tokenizer, [tool()], prefix + call(parameter("name", "report.docx")))
    assert not accepts(tokenizer, [tool()], prefix + call(parameter("wrong", "report.docx")))


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        pytest.param({"type": "invalid"}, "(?i)type", id="invalid-schema-type"),
        pytest.param({"type": "string"}, "(?i)object", id="non-object-string"),
        pytest.param({"type": "integer"}, "(?i)object", id="non-object-integer"),
        pytest.param({"type": "number"}, "(?i)object", id="non-object-number"),
        pytest.param({"type": "boolean"}, "(?i)object", id="non-object-boolean"),
        pytest.param({"type": "null"}, "(?i)object", id="non-object-null"),
        pytest.param({"type": "array", "items": {"type": "integer"}}, "(?i)object", id="non-object-array"),
        pytest.param({"type": "object", "required": 1}, "(?i)required", id="invalid-required"),
        pytest.param({"type": "object", "properties": []}, "(?i)properties", id="invalid-properties"),
        pytest.param({"$ref": "#/$defs/missing"}, "(?i)ref|missing|resolve", id="missing-local-reference"),
    ],
)
def test_malformed_strict_schemas_fail_explicitly(tokenizer, parameters, message):
    with pytest.raises(ValueError, match=message):
        matcher(tokenizer, [tool(parameters)])


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize(
    "scenario",
    [
        "enter-call",
        "invalid-header",
        "leave-call",
        "next-call-invalid",
        "eos-inside",
        "invalid-first-token",
        "merged-close",
        "merged-invalid-close",
        "leave-reasoning",
        "merged-open",
        "after-marker",
        "invalid-after-marker",
        "compact-invalid-header",
        "unknown-then-strict",
        "merged-compact-open",
        "merged-compact-invalid",
        "reordered-parameters",
        "skipped-optional-parameter",
        "included-optional-parameter",
        "duplicate-required-parameter",
        "duplicate-optional-parameter",
    ],
)
def test_speculative_masks_across_tool_boundaries(tokenizer, scenario, special):
    tools = [
        tool(
            {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": ["report.docx", "report.xlsx"]}},
                "required": ["name"],
            }
        )
    ]
    opening = "<tool_call>\n<function=read_files>\n<parameter=name>\n"
    closing = "\n</parameter>\n</function>\n</tool_call>"
    complete = call(parameter("name", "report.docx"))
    prefix, drafted, committed = "Planning. ", complete, complete
    valid_draft = None

    if scenario == "invalid-header":
        valid_draft = "<tool_call>\n<function=read_files>\n<parameter="
        drafted = valid_draft + "wrong>"
    elif scenario == "leave-call":
        prefix = opening + "report.docx"
        drafted = committed = closing + "\nDone."
    elif scenario == "next-call-invalid":
        prefix = complete + "\n"
        valid_draft = opening + "report."
        drafted = valid_draft + "z"
    elif scenario == "eos-inside":
        prefix = opening
        drafted = [tokenizer.eos_token]
        valid_draft = ""
        committed = "report.docx" + closing
    elif scenario == "invalid-first-token":
        prefix = "<tool_call>\n<function=read_files>\n<parameter="
        drafted, valid_draft = "wrong>", ""
        committed = "name>\nreport.docx" + closing
    elif scenario in {"merged-close", "merged-invalid-close"}:
        prefix = opening + "report.docx"
        committed = [token_id(b"\n</parameter>\n</function>\n</tool_call>\nDone.")]
        drafted = committed if scenario == "merged-close" else [token_id(b"\n</parameter>\n<parameter=unexpected>")]
        if scenario == "merged-invalid-close":
            valid_draft = ""
    elif scenario == "leave-reasoning":
        prefix = "<think>A draft plan. "
        drafted = committed = "</think>\n" + complete
    elif scenario == "merged-open":
        drafted = [token_id(b"<tool_call>\n<function=read_files>\n<parameter=name>")] + encode(
            "\nreport.docx" + closing, special=special
        )
    elif scenario in {"after-marker", "invalid-after-marker"}:
        prefix = "<tool_call>"
        drafted = committed = "<function=read_files><parameter=name>\nreport.docx" + closing
        if scenario == "invalid-after-marker":
            drafted, valid_draft = "garbage", ""
    elif scenario == "compact-invalid-header":
        valid_draft = "<tool_call><function=read_files><parameter="
        drafted = valid_draft + "wrong>"
    elif scenario == "unknown-then-strict":
        unknown = "<tool_call><function=read_files_extra>free arguments</function></tool_call>"
        valid_draft = unknown + "<tool_call><function=read_files><parameter="
        drafted = valid_draft + "wrong>"
        committed = unknown + complete
    elif scenario in {"merged-compact-open", "merged-compact-invalid"}:
        if scenario == "merged-compact-open":
            drafted = [token_id(b"<tool_call><function=read_files><parameter=name>")] + encode(
                "\nreport.docx" + closing, special=special
            )
        else:
            drafted = [token_id(b"<tool_call><function=read_files><parameter=wrong>")]
            valid_draft = ""
    elif scenario in {
        "reordered-parameters",
        "skipped-optional-parameter",
        "included-optional-parameter",
        "duplicate-required-parameter",
        "duplicate-optional-parameter",
    }:
        tools = [
            tool(
                {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "force": {"type": "boolean"},
                        "label": {"type": "string"},
                    },
                    "required": ["count", "label"],
                }
            )
        ]
        prefix = "<tool_call>\n<function=read_files>\n" + parameter("count", "2")
        ending = "</function>\n</tool_call>"
        skipped = parameter("label", "report") + ending
        included = parameter("force", "true") + skipped
        drafted, committed = skipped, included
        if scenario == "reordered-parameters":
            drafted = parameter("label", "report") + parameter("force", "true") + ending
            valid_draft = parameter("label", "report") + "<"
        elif scenario == "included-optional-parameter":
            drafted, committed = included, skipped
        elif scenario == "duplicate-required-parameter":
            drafted = parameter("count", "2") + skipped
            valid_draft = "<parameter="
        elif scenario == "duplicate-optional-parameter":
            prefix += parameter("force", "true")
            drafted, committed = included, skipped
            valid_draft = "<parameter="

    prefix_tokens = encode(prefix, special=special)
    draft_tokens = encode(drafted, special=special) if isinstance(drafted, str) else drafted
    committed_tokens = encode(committed, special=special) if isinstance(committed, str) else committed
    accepted_count = len(draft_tokens) if valid_draft is None else len(encode(valid_draft, special=special))
    result = matcher(tokenizer, tools)
    consume(result, prefix_tokens)
    before = result.compute_bitmask()
    masks = allocate_token_bitmask(len(draft_tokens) + 1, tokenizer.vocab_size)
    fill_next_token_bitmask_par_with_draft_tokens(llg.LLExecutor(1), [(result, 0, draft_tokens)], masks)
    assert not result.is_error(), result.get_error()
    assert result.compute_bitmask() == before

    reference = matcher(tokenizer, tools)
    consume(reference, prefix_tokens)
    for index in range(accepted_count + 1):
        np.testing.assert_array_equal(masks[index], np.frombuffer(reference.compute_bitmask(), dtype=np.int32))
        if index < accepted_count:
            consume(reference, [draft_tokens[index]])
    if accepted_count < len(draft_tokens):
        assert reference.compute_logit_bias()[draft_tokens[accepted_count]] == 0

    consume(result, committed_tokens)
    fresh = matcher(tokenizer, tools)
    consume(fresh, prefix_tokens + committed_tokens)
    assert result.compute_bitmask() == fresh.compute_bitmask()
    assert result.is_accepting()


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize(
    "boundary", ["tool-marker", "unknown-name", "function-name", "parameter-name", "value", "closing-tag", "reasoning"]
)
def test_partial_rollback_restores_boundary_state(tokenizer, boundary, special):
    tools = [
        tool(
            {"type": "object", "properties": {"name": {"enum": ["report.docx", "report.xlsx"]}}, "required": ["name"]}
        ),
        tool(name="read_other", strict=False),
    ]
    opening = "<tool_call>\n<function=read_files>\n<parameter=name>\n"
    closing = "\n</parameter>\n</function>\n</tool_call>"
    if boundary == "tool-marker":
        prefix = "<tool_call>"
        draft = "<function=message_added_tool>free arguments</function></tool_call>"
        alternative = "<function=read_files><parameter=name>\nreport.docx" + closing
    elif boundary == "unknown-name":
        prefix = "<tool_call><function=read_files"
        draft = "_extra>free arguments</function></tool_call>"
        alternative = "><parameter=name>\nreport.docx" + closing
    elif boundary == "function-name":
        prefix = "<tool_call>\n<function=read_"
        draft = "files>\n<parameter=name>\nreport.docx" + closing
        alternative = "other>free arguments\n</function>\n</tool_call>"
    elif boundary == "parameter-name":
        prefix = "<tool_call>\n<function=read_files>\n<parameter=na"
        draft = "me>\nreport.docx" + closing
        alternative = "me>\nreport.xlsx" + closing
    elif boundary == "value":
        prefix = opening + "report."
        draft, alternative = "docx" + closing, "xlsx" + closing
    elif boundary == "closing-tag":
        prefix = opening + "report.docx\n</para"
        draft = "meter>\n</function>\n</tool_call>"
        alternative = draft + "\n" + call(parameter("name", "report.xlsx"))
    else:
        prefix = "<think>Plan "
        draft = "</think>\n" + call(parameter("name", "report.docx"))
        alternative = "more carefully.</think>\n" + call(parameter("name", "report.xlsx"))

    result = matcher(tokenizer, tools)
    prefix_tokens = encode(prefix, special=special)
    draft_tokens = encode(draft, special=special)
    consume(result, prefix_tokens)
    before = result.compute_bitmask()
    consume(result, draft_tokens)
    assert result.rollback(len(draft_tokens))
    assert result.compute_bitmask() == before
    consume(result, encode(alternative, special=special))
    fresh = matcher(tokenizer, tools)
    consume(fresh, prefix_tokens + encode(alternative, special=special))
    assert result.compute_bitmask() == fresh.compute_bitmask()
    assert result.is_accepting()


def test_copied_matchers_do_not_share_speculative_progress(tokenizer):
    tools = [
        tool(
            {
                "type": "object",
                "properties": {"name": {"enum": ["report.docx", "report.xlsx"]}},
                "required": ["name"],
            }
        )
    ]
    result = matcher(tokenizer, tools)
    prefix = "<tool_call>\n<function=read_files>\n<parameter=name>\nreport."
    closing = "\n</parameter>\n</function>\n</tool_call>"
    consume(result, encode(prefix))
    before = result.compute_bitmask()
    fork = result.deep_copy()
    consume(fork, encode("docx" + closing))
    assert result.compute_bitmask() == before
    consume(result, encode("xlsx" + closing))
    assert result.is_accepting() and fork.is_accepting()
