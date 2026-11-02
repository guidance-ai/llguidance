import re
from typing import Any

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


def encode(text: str, *, special: bool = True, markers: dict[str, int] | None = None) -> list[int]:
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


def tokenizer_with_markers(names: list[str]) -> tuple[llg.LLTokenizer, dict[str, int]]:
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
def marker_tokenizer(request: pytest.FixtureRequest) -> tuple[llg.LLTokenizer, dict[str, int]]:
    return tokenizer_with_markers(request.param)


def tool(
    parameters: dict[str, Any] | None = None,
    *,
    name: str = "read_files",
    strict: object = True,
) -> dict[str, Any]:
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


def call(arguments: str, *, name: str = "read_files") -> str:
    if arguments and not arguments.endswith("\n"):
        arguments += "\n"
    return f"<tool_call>\n<function={name}>\n{arguments}</function>\n</tool_call>"


def parameter(name: str, value: str, *, padded: bool = True) -> str:
    if padded:
        return f"<parameter={name}>\n{value}\n</parameter>\n"
    return f"<parameter={name}>{value}</parameter>"


def matcher(
    tokenizer: llg.LLTokenizer,
    tools: list[dict[str, Any]],
    *,
    initial_reasoning: bool = False,
    special_tokens: list[str] | None = None,
) -> llg.LLMatcher:
    grammar = llg.grammar_from_tools(
        tools, format="qwen3_coder", initial_reasoning=initial_reasoning, special_tokens=special_tokens
    )
    result = llg.LLMatcher(tokenizer, grammar)
    if result.is_error():
        raise ValueError(result.get_error())
    return result


def consume(result: llg.LLMatcher, tokens: list[int]) -> None:
    for token_id in tokens:
        mask = result.compute_logit_bias()
        assert not result.is_error(), result.get_error()
        assert mask[token_id] == 200, f"Rejected token {token_id}"
        assert result.consume_token(token_id), result.get_error()


def accepts(
    tokenizer: llg.LLTokenizer,
    tools: list[dict[str, Any]],
    text: str,
    *,
    special: bool = True,
    initial_reasoning: bool = False,
) -> bool:
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
def test_strict_call_accepts_schema_value(tokenizer: llg.LLTokenizer, special: bool) -> None:
    assert accepts(tokenizer, [tool()], call(parameter("name", "report.docx")), special=special)


@pytest.mark.parametrize("options", [{}, {"special_tokens": None}], ids=["omitted", "none"])
@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
def test_default_special_tokens_accept_all_format_markers(
    tokenizer: llg.LLTokenizer,
    options: dict[str, Any],
    special: bool,
) -> None:
    grammar = llg.grammar_from_tools([tool()], format="qwen3_coder", **options)
    result = llg.LLMatcher(tokenizer, grammar)
    text = (
        "<think>Draft: " + call(parameter("wrong", "anything")) + "</think>\n" + call(parameter("name", "report.docx"))
    )
    consume(result, encode(text, special=special))
    assert result.is_accepting()
    assert result.compute_logit_bias()[tokenizer.eos_token] == 200


@pytest.mark.parametrize("initial_reasoning", [False, True], ids=["generated-think", "prefilled-think"])
def test_special_token_overrides_preserve_strict_calls(
    marker_tokenizer: tuple[llg.LLTokenizer, dict[str, int]],
    initial_reasoning: bool,
) -> None:
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
def test_requested_special_tokens_must_exist_in_the_tokenizer(special_tokens: list[str] | None) -> None:
    tokenizer, _ = tokenizer_with_markers([])
    grammar = llg.grammar_from_tools([tool()], format="qwen3_coder", special_tokens=special_tokens)
    result = llg.LLMatcher(tokenizer, grammar)
    assert result.is_error()
    assert "special token" in result.get_error().lower()


@pytest.mark.parametrize("special_tokens", [["<tool_cal>"], ["<tool_call>", "<unknown>"], ["<eos>"]])
def test_special_token_overrides_reject_names_outside_the_format(special_tokens: list[str] | None) -> None:
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
def test_unlisted_special_markers_cannot_bypass_text_only_rules(
    tokenizer: llg.LLTokenizer,
    prefix: str,
    marker: str,
) -> None:
    result = matcher(tokenizer, [tool()], special_tokens=[])
    consume(result, encode(prefix, special=False))
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    assert mask[token_id(marker.encode())] == 0
    consume(result, encode(marker, special=False))


@pytest.mark.parametrize("byte_only_first", [False, True], ids=["defaults-first", "byte-only-first"])
def test_special_token_overrides_are_isolated_between_compilations(
    tokenizer: llg.LLTokenizer,
    byte_only_first: bool,
) -> None:
    byte_tokenizer, _ = tokenizer_with_markers([])
    modes: list[tuple[llg.LLTokenizer, list[str] | None, bool]] = [
        (tokenizer, None, True),
        (byte_tokenizer, [], False),
    ]
    if byte_only_first:
        modes.reverse()
    results = [(matcher(tok, [tool()], special_tokens=names), tok, special) for tok, names, special in modes]
    text = "<think>Plan</think>\n" + call(parameter("name", "report.docx"))
    for result, tok, special in results:
        consume(result, encode(text, special=special))
        assert result.is_accepting()
        assert result.compute_logit_bias()[tok.eos_token] == 200


def test_special_token_overrides_preserve_speculative_masks_and_rollback(
    marker_tokenizer: tuple[llg.LLTokenizer, dict[str, int]],
) -> None:
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
def test_strict_arguments_reject_malformed_calls(tokenizer: llg.LLTokenizer, arguments: str) -> None:
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
def test_parameter_values_follow_their_schema(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    value: str,
    padded: bool,
) -> None:
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
def test_parameter_values_reject_schema_violations(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    valid: str,
    value: str,
) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", valid)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", value)))


@pytest.mark.parametrize("include_optional", [False, True])
def test_optional_parameters_stay_optional(tokenizer: llg.LLTokenizer, include_optional: bool) -> None:
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "force": {"type": "boolean"}},
        "required": ["name"],
    }
    arguments = parameter("name", "report.docx")
    if include_optional:
        arguments += "\n" + parameter("force", "true")
    assert accepts(tokenizer, [tool(parameters)], call(arguments))


def test_root_ref_stays_an_xml_parameter_list(tokenizer: llg.LLTokenizer) -> None:
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


def test_local_ref_preserves_raw_string_encoding(tokenizer: llg.LLTokenizer) -> None:
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
def test_same_ref_uses_raw_and_nested_json_encoding(
    tokenizer: llg.LLTokenizer,
    definition: dict[str, Any],
    nested_first: bool,
) -> None:
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


def test_recursive_nested_ref_stays_json(tokenizer: llg.LLTokenizer) -> None:
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
def test_only_boolean_true_enables_constraints(tokenizer: llg.LLTokenizer, strict: object) -> None:
    definition = tool(strict=strict)
    if strict is None:
        del definition["function"]["strict"]
    assert accepts(tokenizer, [definition], call('{"unexpected": "unconstrained"}'))


def test_mixed_tools_leave_nonstrict_and_unknown_calls_free(tokenizer: llg.LLTokenizer) -> None:
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
def test_header_whitespace_cannot_bypass_a_known_strict_schema(
    tokenizer: llg.LLTokenizer,
    gap: str,
    special: bool,
) -> None:
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
def test_whitespace_aliases_of_strict_names_are_rejected(tokenizer: llg.LLTokenizer, name: str) -> None:
    # Whitespace aliases of strict function names must not bypass their schemas.
    parameters = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    tools = [tool(parameters)]
    arguments = parameter("count", "3")
    assert accepts(tokenizer, tools, call(arguments))
    assert not accepts(tokenizer, tools, call(arguments, name=name))


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize("suffix", ["", "garbage", "\n</tool_call>", "<function=>\n</function></tool_call>"])
def test_tool_call_marker_requires_a_function_envelope(tokenizer: llg.LLTokenizer, special: bool, suffix: str) -> None:
    text = "Planning. <tool_call>" + suffix
    assert accepts(tokenizer, [tool()], "Planning. " + call("free", name="message_added_tool"), special=special)
    assert not accepts(tokenizer, [tool()], text, special=special)
    assert accepts(tokenizer, [tool()], "<think>" + text + "</think>Done.", special=special)


@pytest.mark.parametrize("suffix", ["", "<tool_call>", "<tool_call>garbage"])
def test_no_strict_tools_allow_free_text_and_tool_shaped_text(tokenizer: llg.LLTokenizer, suffix: str) -> None:
    text = "A draft: " + call("not valid arguments") + suffix
    assert accepts(tokenizer, [], text)
    assert accepts(tokenizer, [tool(strict=False)], text)


@pytest.mark.parametrize("special", [False, True], ids=["byte-markers", "special-markers"])
@pytest.mark.parametrize("initial_reasoning", [False, True], ids=["generated-think", "prefilled-think"])
def test_reasoning_is_free_but_calls_after_reasoning_are_strict(
    tokenizer: llg.LLTokenizer,
    special: bool,
    initial_reasoning: bool,
) -> None:
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
def test_tokens_crossing_a_strict_header_are_masked_atomically(
    tokenizer: llg.LLTokenizer,
    prefix: str,
    valid_token: bytes,
    invalid_tokens: list[bytes],
) -> None:
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
def test_eos_is_blocked_inside_an_incomplete_tool_envelope(
    tokenizer: llg.LLTokenizer,
    prefix: str,
    special: bool,
) -> None:
    result = matcher(tokenizer, [tool(), tool(name="loose", strict=False)])
    consume(result, encode(prefix, special=special))
    mask = result.compute_logit_bias()
    assert not result.is_error(), result.get_error()
    assert not result.is_accepting()
    assert mask[tokenizer.eos_token] == 0


def test_rollback_restores_state_across_the_strict_header(tokenizer: llg.LLTokenizer) -> None:
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
def test_speculative_masks_restore_state_and_match_sequential_decoding(
    tokenizer: llg.LLTokenizer,
    draft_text: str,
    accepted_count: int,
) -> None:
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


def test_raw_string_cannot_contain_its_closing_delimiter(tokenizer: llg.LLTokenizer) -> None:
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
def test_unknown_format_is_rejected(format_name: str) -> None:
    with pytest.raises(ValueError, match="(?i)format"):
        llg.grammar_from_tools([tool()], format=format_name)


def test_duplicate_function_names_are_rejected() -> None:
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
def test_unsupported_strict_schemas_fail_explicitly(
    tokenizer: llg.LLTokenizer,
    parameters: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        matcher(tokenizer, [tool(parameters)])


def test_public_qwen3_coder_template_call(tokenizer: llg.LLTokenizer) -> None:
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
def test_constraints_apply_after_removing_one_edge_newline(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    body: str,
    expected: bool,
) -> None:
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
def test_string_constants_preserve_whitespace_in_the_parsed_value(
    tokenizer: llg.LLTokenizer,
    value: str,
    valid_body: str,
    invalid_body: str,
) -> None:
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string", "const": value}},
        "required": ["value"],
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("value", valid_body, padded=False)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", invalid_body, padded=False)))


@pytest.mark.parametrize("whitespace", ["", "\n", " \t", "\r\n\t "], ids=["none", "lf", "spaces", "mixed"])
def test_whitespace_between_parameters_and_before_function_end(tokenizer: llg.LLTokenizer, whitespace: str) -> None:
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
def test_all_optional_parameters_allow_ordered_subsets(
    tokenizer: llg.LLTokenizer,
    fields: list[str],
    expected: bool,
    explicit_required: bool,
) -> None:
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
def test_zero_argument_tool_accepts_only_an_empty_parameter_list(
    tokenizer: llg.LLTokenizer,
    include_parameters: bool,
) -> None:
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    definition = tool(parameters)
    if not include_parameters:
        del definition["function"]["parameters"]
    assert accepts(tokenizer, [definition], call(""))
    assert not accepts(tokenizer, [definition], call(parameter("name", "unexpected")))


@pytest.mark.parametrize(
    ("additional", "expected"), [(None, False), (False, False), (True, True)], ids=["omitted", "false", "true"]
)
def test_top_level_additional_properties_controls_extra_parameters(
    tokenizer: llg.LLTokenizer,
    additional: bool | None,
    expected: bool,
) -> None:
    parameters: dict[str, Any] = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
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
def test_additional_properties_true_accepts_unconstrained_values(
    tokenizer: llg.LLTokenizer,
    value: str,
    padded: bool,
) -> None:
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": True,
    }
    arguments = parameter("count", "1", padded=padded) + parameter("extra", value, padded=padded)
    assert accepts(tokenizer, [tool(parameters)], call(arguments))


def test_additional_properties_true_accepts_duplicate_extra_names(tokenizer: llg.LLTokenizer) -> None:
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
def test_additional_properties_cannot_bypass_declared_parameter_constraints(
    tokenizer: llg.LLTokenizer,
    fields: list[tuple[str, str]],
) -> None:
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
def test_additional_parameter_names_allow_prefix_overlaps_and_unicode(tokenizer: llg.LLTokenizer, name: str) -> None:
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": True,
    }
    assert accepts(tokenizer, [tool(parameters)], call(parameter("count", "1") + parameter(name, "raw text")))


@pytest.mark.parametrize("include_properties", [False, True], ids=["omitted-properties", "empty-properties"])
def test_additional_properties_true_without_declared_parameters(
    tokenizer: llg.LLTokenizer,
    include_properties: bool,
) -> None:
    parameters = {"type": "object", "additionalProperties": True}
    if include_properties:
        parameters["properties"] = {}
    assert accepts(tokenizer, [tool(parameters)], call(""))
    arguments = parameter("note", "raw text") + parameter("count", "3") + parameter("data", '{"ready":true}')
    assert accepts(tokenizer, [tool(parameters)], call(arguments))
    assert not accepts(tokenizer, [tool(parameters)], call('{"note":"raw text"}'))


@pytest.mark.parametrize("extra_names", [[], ["extra", "another"]], ids=["below-minimum", "above-maximum"])
def test_additional_parameters_count_toward_property_limits(tokenizer: llg.LLTokenizer, extra_names: list[str]) -> None:
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
def test_schema_order_preserves_required_fields_and_uniqueness(
    tokenizer: llg.LLTokenizer,
    fields: list[str],
    expected: bool,
    padded: bool,
) -> None:
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
def test_parameter_count_bounds_with_optional_properties_are_unsupported(explicit_required: bool) -> None:
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
def test_nested_object_constraints_are_not_lost(tokenizer: llg.LLTokenizer, invalid: str) -> None:
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
def test_container_and_combined_constraints(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    valid: str,
    invalid: str,
) -> None:
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
def test_literal_and_composed_schemas_preserve_value_constraints(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    valid: list[str],
    invalid: str,
) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    for value in valid:
        assert accepts(tokenizer, [tool(parameters)], call(parameter("value", value)))
    assert not accepts(tokenizer, [tool(parameters)], call(parameter("value", invalid)))


@pytest.mark.parametrize(
    "second_name", ["save_report", "read_files_extra", "read"], ids=["distinct", "longer-prefix", "shorter-prefix"]
)
def test_multiple_strict_tools_keep_their_own_schemas(tokenizer: llg.LLTokenizer, second_name: str) -> None:
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
def test_nonstrict_or_unknown_names_stay_free(tokenizer: llg.LLTokenizer, name: str, gap: str) -> None:
    for tools in [[tool()], [tool(), tool(name=name, strict=False)]]:
        text = f"<tool_call>{gap}<function={name}>{gap}not structured arguments</function>\n</tool_call>" + call(
            parameter("name", "report.docx")
        )
        assert accepts(tokenizer, tools, text)
        assert not accepts(tokenizer, tools, text + call(parameter("wrong", "report.docx")))


def test_schemas_are_isolated_between_tools_and_compilations(tokenizer: llg.LLTokenizer) -> None:
    def definition(name: str, value: str) -> dict[str, Any]:
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


def test_nonstrict_schema_errors_are_ignored(tokenizer: llg.LLTokenizer) -> None:
    tools = [tool(), tool({"type": "not-a-schema-type"}, name="loose", strict=False)]
    assert accepts(tokenizer, tools, call("free", name="loose") + call(parameter("name", "report.docx")))


def test_many_optional_parameters_fit_default_limits(tokenizer: llg.LLTokenizer) -> None:
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


def test_many_tools_fit_default_limits(tokenizer: llg.LLTokenizer) -> None:
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
def test_declared_parameter_names_are_preserved(tokenizer: llg.LLTokenizer, name: str) -> None:
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
def test_plain_response_and_strict_call_after_optional_reasoning(tokenizer: llg.LLTokenizer, prefix: str) -> None:
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
def test_malformed_strict_schemas_fail_explicitly(
    tokenizer: llg.LLTokenizer,
    parameters: dict[str, Any],
    message: str,
) -> None:
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
def test_speculative_masks_across_tool_boundaries(tokenizer: llg.LLTokenizer, scenario: str, special: bool) -> None:
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
    prefix = "Planning. "
    drafted: str | list[int] = complete
    committed: str | list[int] = complete
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
def test_partial_rollback_restores_boundary_state(tokenizer: llg.LLTokenizer, boundary: str, special: bool) -> None:
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


def test_copied_matchers_do_not_share_speculative_progress(tokenizer: llg.LLTokenizer) -> None:
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


@pytest.mark.parametrize("wrapper", ["direct", "ref", "allof", "anyof", "oneof", "nullable"])
@pytest.mark.parametrize(
    ("schema", "valid", "invalid"),
    [
        pytest.param({'type': 'string'}, ['', 'é🧪文 <tag> & "quoted"'], ['before </parameter> after'], id='plain'),
        pytest.param({'type': 'string', 'enum': ['blue', 'red']}, ['blue', 'red'], ['green', '"blue"', ' blue', 'blue '], id='enum'),
        pytest.param({'const': 'ready'}, ['ready'], ['Ready', '"ready"'], id='const'),
        pytest.param({'type': 'string', 'pattern': '^[a-z]{2,4}$'}, ['ab', 'abcd'], ['a', 'abcde', '1abc', 'abc1'], id='pattern_anchored'),
        pytest.param({'type': 'string', 'pattern': 'ab'}, ['ab', 'xxabyy', '\nxabz\n'], ['a', 'ba'], id='pattern_unanchored'),
        pytest.param({'type': 'string', 'pattern': '^ab'}, ['ab', 'abcd'], ['zab'], id='pattern_start'),
        pytest.param({'type': 'string', 'pattern': 'ab$'}, ['ab', 'zzab'], ['abz'], id='pattern_end'),
        pytest.param({'type': 'string', 'pattern': '^a/b\\[c\\]\\.\\+$'}, ['a/b[c].+'], ['a/bc.+', 'a/b[c]x+'], id='pattern_special'),
        pytest.param({'type': 'string', 'pattern': '^\\w\\d$'}, ['a3', '_0'], ['é3', 'a٣'], id='pattern_ascii'),
        pytest.param({'type': 'string', 'pattern': '^ab|cd$'}, ['abx', 'xcd'], ['xabx', 'cdx'], id='pattern_alternation'),
        pytest.param({'type': 'string', 'minLength': 2}, ['ab', 'abc', 'é🧪', '中文'], ['', 'a', 'é', '🧪', '文'], id='min_length'),
        pytest.param({'type': 'string', 'maxLength': 2}, ['', 'a', 'é🧪', '中文'], ['abc', 'é🧪文'], id='max_length'),
        pytest.param({'type': 'string', 'minLength': 2, 'maxLength': 3}, ['é🧪', 'é🧪文'], ['é', 'é🧪文字'], id='length_bounds'),
        pytest.param({'type': 'string', 'pattern': '^[a-z]+$', 'minLength': 2, 'maxLength': 3}, ['ab', 'abc'], ['a', 'abcd', 'a3'], id='pattern_length'),
        pytest.param({'type': 'string', 'enum': ['blue', 'red', 'black'], 'pattern': '^bl'}, ['blue', 'black'], ['red', 'blond'], id='enum_pattern'),
        pytest.param({'type': 'string', 'enum': ['a', 'ab', 'abc'], 'minLength': 2, 'maxLength': 2}, ['ab'], ['a', 'abc', 'cd'], id='enum_length'),
        pytest.param({'allOf': [{'type': 'string', 'pattern': '^[a-z]+$'}, {'minLength': 2}, {'maxLength': 3}]}, ['ab', 'abc'], ['a', 'abcd', 'a3'], id='allof'),
        pytest.param({'anyOf': [{'type': 'string', 'pattern': '^a+$', 'maxLength': 2}, {'const': 'blue'}]}, ['a', 'aa', 'blue'], ['aaa', 'red', 'ab'], id='anyof'),
        pytest.param({'oneOf': [{'type': 'string', 'pattern': '^a+$'}, {'type': 'string', 'pattern': '^b+$'}]}, ['a', 'aa', 'bb'], ['ab', 'c'], id='oneof_disjoint'),
        pytest.param({'oneOf': [{'type': 'string', 'enum': ['a', 'b']}, {'type': 'string', 'enum': ['b', 'c']}]}, ['a', 'c'], ['b', 'd'], id='oneof_overlap'),
        pytest.param({'anyOf': [{'type': 'string', 'pattern': '^x+$'}, {'type': 'integer', 'minimum': 2, 'maximum': 4}]}, ['xx', '2', '4'], ['a', '1', '5', '3.5', 'true'], id='mixed_anyof'),
        pytest.param({'type': ['string', 'null'], 'pattern': '^blue$'}, ['blue', 'null'], ['red', 'true', '"blue"'], id='nullable'),
        pytest.param({'anyOf': [{'type': 'string', 'minLength': 5, 'maxLength': 2}, {'const': 'ok'}]}, ['ok'], ['ab', 'abcde'], id='drop_unsatisfiable_length'),
        pytest.param({'anyOf': [{'type': 'string', 'enum': ['red'], 'pattern': '^blue$'}, {'const': 'ok'}]}, ['ok'], ['red', 'blue'], id='drop_unsatisfiable_enum'),
        pytest.param({'anyOf': [{'type': 'string', 'minLength': 5, 'maxLength': 2}, {'type': 'integer'}]}, ['3'], ['abc', '"3"'], id='drop_unsatisfiable_for_json'),
        pytest.param({'type': 'string', 'pattern': '^before .* after$'}, ['before <tag> after'], ['before </parameter> after'], id='delimiter_pattern'),
        pytest.param({'type': 'string', 'pattern': '^\nblue\n$'}, ['\nblue\n'], ['blue', '\nblue', 'blue\n'], id='newline_pattern'),
        pytest.param({'type': 'string', 'format': 'date-time'}, ['2026-10-28T12:30:00Z'], ['2026-13-28T25:30:00Z'], id='format_date_time'),
        pytest.param({'type': 'string', 'format': 'time'}, ['12:30:00+01:00'], ['25:30:00Z'], id='format_time'),
        pytest.param({'type': 'string', 'format': 'date'}, ['2026-10-28'], ['2026-13-28'], id='format_date'),
        pytest.param({'type': 'string', 'format': 'duration'}, ['P1DT2H'], ['1 day'], id='format_duration'),
        pytest.param({'type': 'string', 'format': 'email'}, ['reader@example.com'], ['reader@@example.com'], id='format_email'),
        pytest.param({'type': 'string', 'format': 'hostname'}, ['example.com'], ['-example.com'], id='format_hostname'),
        pytest.param({'type': 'string', 'format': 'ipv4'}, ['192.168.1.1'], ['256.1.1.1'], id='format_ipv4'),
        pytest.param({'type': 'string', 'format': 'ipv6'}, ['2001:db8::1'], ['2001:db8::gg'], id='format_ipv6'),
        pytest.param({'type': 'string', 'format': 'uuid'}, ['123e4567-e89b-12d3-a456-426614174000'], ['123e4567-e89b-12d3-a456-42661417400z'], id='format_uuid'),
        pytest.param({'type': 'string', 'format': 'uri'}, ['https://example.com/a?x=1#b'], ['https://example.com/a b'], id='format_uri'),
        pytest.param({'type': 'string', 'format': 'email', 'pattern': '@example\\.com$', 'minLength': 13, 'maxLength': 15}, ['a@example.com', 'abc@example.com'], ['abcd@example.com', 'a@other.com', '@@example.com'], id='format_pattern_length'),
        pytest.param({'const': 'ready', 'pattern': '^re'}, ['ready'], ['re', 'Ready', '"ready"'], id='const_pattern'),
        pytest.param({'type': 'string', 'format': 'email', 'enum': ['a@example.com', 'not-an-email']}, ['a@example.com'], ['not-an-email', 'b@example.com'], id='format_enum'),
    ],
)
def test_lark_string_rules(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    valid: list[str],
    invalid: list[str],
    wrapper: str,
) -> None:
    definitions = {}
    if wrapper == "ref":
        definitions = {"value": schema}
        schema = {"$ref": "#/$defs/value"}
    elif wrapper == "allof":
        schema = {"allOf": [schema, {}]}
    elif wrapper == "nullable":
        schema = {"anyOf": [schema, {"type": "null"}]}
        valid = [*valid, "null"]
    elif wrapper in ("anyof", "oneof"):
        schema = {"anyOf" if wrapper == "anyof" else "oneOf": [schema, False]}
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"], "$defs": definitions}
    tools = [tool(parameters)]
    for value in valid:
        assert accepts(tokenizer, tools, call(parameter("value", value))), repr(value)
    for value in invalid:
        assert not accepts(tokenizer, tools, call(parameter("value", value))), repr(value)


@pytest.mark.parametrize(("schema", "valid", "invalid"), [
    pytest.param({'type': 'number', 'minimum': 2}, ['2', '2.5'], ['1.9'], id='minimum'),
    pytest.param({'type': 'number', 'maximum': 2}, ['1.5', '2'], ['2.1'], id='maximum'),
    pytest.param({'type': 'number', 'exclusiveMinimum': 2}, ['2.1'], ['2', '1'], id='exclusive_minimum'),
    pytest.param({'type': 'number', 'exclusiveMaximum': 2}, ['1.9'], ['2', '3'], id='exclusive_maximum'),
    pytest.param({'type': 'integer', 'multipleOf': 3}, ['-3', '0', '6'], ['4', '3.5'], id='multiple_of'),
    pytest.param({'type': 'number', 'multipleOf': 0.5}, ['-0.5', '1.5'], ['1.25'], id='fractional_multiple'),
    pytest.param({'type': 'integer'}, ['0', '-3'], ['1.5', '"3"'], id='integer'),
    pytest.param({'type': 'number'}, ['-1.5', '2e3'], ['true', '"2"'], id='number'),
    pytest.param({'type': 'boolean'}, ['true', 'false'], ['True', '1'], id='boolean'),
    pytest.param({'type': 'null'}, ['null'], ['NULL', 'None'], id='null'),
 ])
def test_lark_json_scalar_rules(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    valid: list[str],
    invalid: list[str],
) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    tools = [tool(parameters)]
    for value in valid:
        for padded in (False, True):
            assert accepts(tokenizer, tools, call(parameter("value", value, padded=padded)))
    for value in invalid:
        assert not accepts(tokenizer, tools, call(parameter("value", value)))


@pytest.mark.parametrize(("schema", "value"), [({"type": "boolean"}, "true"), ({"type": "boolean"}, "false"), ({"type": "null"}, "null")])
@pytest.mark.parametrize("padding", ["", "\n", " ", "\t", "\r", "\n\n", "\n "])
def test_lark_boolean_null_padding(
    tokenizer: llg.LLTokenizer,
    schema: dict[str, Any],
    value: str,
    padding: str,
) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    tools = [tool(parameters)]
    assert accepts(tokenizer, tools, call(parameter("value", value)))
    for body in (padding + value, value + padding, padding + value + padding):
        assert accepts(tokenizer, tools, call(parameter("value", body, padded=False))) == (padding in ("", "\n"))


@pytest.mark.parametrize(("schema", "keyword"), [
    pytest.param({'type': 'string', 'minLength': 5, 'maxLength': 2}, 'unsatisfiable', id='contradictory_length'),
    pytest.param({'type': 'string', 'enum': ['red'], 'pattern': '^blue$'}, 'unsatisfiable', id='contradictory_enum'),
    pytest.param({'allOf': [{'type': 'string', 'maxLength': 2}, {'minLength': 5}]}, 'unsatisfiable', id='contradictory_allof'),
    pytest.param({'oneOf': [{'const': 'same'}, {'const': 'same'}]}, 'unsatisfiable', id='empty_oneof'),
    pytest.param({'type': 'string', 'format': 'not-a-format'}, 'format', id='unknown_format'),
    pytest.param({'$ref': 'https://example.invalid/value.json'}, 'ref', id='remote_ref'),
    pytest.param({'$ref': '#/$defs/missing'}, 'ref', id='missing_ref'),
    pytest.param({'type': 'string', 'not': {'const': 'x'}}, 'not', id='unsupported_keyword'),
    pytest.param({'type': 'object', 'unevaluatedProperties': False}, 'unevaluatedProperties', id='unevaluated'),
    pytest.param({'oneOf': [{'type': 'string'}, {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a'], 'additionalProperties': False}]}, 'oneOf', id='string_or_object_oneof'),
    pytest.param({'type': 'string', 'pattern': '['}, 'regex', id='invalid_pattern'),
    pytest.param({'type': 'string', 'pattern': '(?=a)b'}, 'regex', id='lookaround_pattern'),
    pytest.param({'allOf': [{'type': 'string'}, {'type': 'integer'}]}, 'unsatisfiable', id='allof_incompatible_types'),
    pytest.param({'const': 'ready', 'pattern': '^x'}, 'unsatisfiable', id='contradictory_const_pattern'),
 ])
@pytest.mark.parametrize("referenced", [False, True])
def test_lark_schema_errors_identify_tool_and_parameter(schema: dict[str, Any], keyword: str, referenced: bool) -> None:
    definitions = {"constraint": schema} if referenced else {}
    value_schema = {"$ref": "#/$defs/constraint"} if referenced else schema
    parameters = {"type": "object", "properties": {"value": value_schema}, "required": ["value"], "$defs": definitions}
    with pytest.raises(ValueError) as error:
        llg.grammar_from_tools([tool(parameters, name="sample")], format="qwen3_coder")
    message = str(error.value).lower()
    assert "sample" in message and "value" in message, message
    assert keyword.lower() in message, message


def test_lark_ref_constraints_also_apply_inside_json(tokenizer: llg.LLTokenizer) -> None:
    parameters = {
        "type": "object", "$defs": {"name": {"type": "string", "pattern": "^[a-z]+$", "minLength": 2, "maxLength": 3}},
        "properties": {"raw": {"$ref": "#/$defs/name"}, "nested": {"type": "array", "items": {"$ref": "#/$defs/name"}}},
        "required": ["raw", "nested"],
    }
    for raw, nested, expected in [("ab", '["abc"]', True), ("a", '["abc"]', False), ("ab", '["abcd"]', False), ("ab", '["a3"]', False), ("ab", '[abc]', False)]:
        assert accepts(tokenizer, [tool(parameters)], call(parameter("raw", raw) + parameter("nested", nested))) == expected


OBJECT_A = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
OBJECT_B = {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"], "additionalProperties": False}


@pytest.mark.parametrize(("value", "root", "valid", "invalid"), [
    pytest.param({"$ref": "#/$defs/n"}, {"$defs": {"n": {"type": "integer", "minimum": 1, "maximum": 5}}}, ["1", "5"], ["0", "6", "1.5", '"3"'], id="ref_number"),
    pytest.param({"$ref": "#/$defs/b"}, {"$defs": {"b": {"type": "boolean"}}}, ["true", "false"], ["yes", '"true"'], id="ref_boolean"),
    pytest.param({"$ref": "#/$defs/o"}, {"$defs": {"o": OBJECT_A}}, ['{"a":1}'], ['{"a":"x"}', "{}", '{"a":1,"b":2}'], id="ref_object"),
    pytest.param({"$ref": "#/$defs/l"}, {"$defs": {"l": {"type": "array", "items": {"enum": ["x", "y"]}, "minItems": 1}}}, ['["x"]', '["x","y"]'], ["[]", '["z"]'], id="ref_array"),
    pytest.param({"$ref": "#/$defs/a"}, {"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"type": "string", "enum": ["ok"]}}}, ["ok"], ["no"], id="ref_chain"),
    pytest.param({"$ref": "#/definitions/s"}, {"definitions": {"s": {"type": "string", "pattern": "^[0-9]+$"}}}, ["123"], ["12a"], id="definitions_keyword"),
    pytest.param({"$ref": "#/$defs/s", "maxLength": 3}, {"$defs": {"s": {"type": "string", "pattern": "^[a-z]+$"}}}, ["abc"], ["abcd", "ab1"], id="ref_with_sibling"),
    pytest.param({"anyOf": [{"$ref": "#/$defs/auto"}, {"$ref": "#/$defs/count"}]}, {"$defs": {"auto": {"const": "auto"}, "count": {"type": "integer", "minimum": 0}}}, ["auto", "0", "7"], ["-1", "manual"], id="anyof_refs"),
    pytest.param({"anyOf": [OBJECT_A, {"type": "array", "items": {"type": "integer"}}]}, {}, ['{"a":1}', "[1,2]"], ['{"a":"x"}', '["x"]', "1"], id="anyof_object_array"),
    pytest.param({"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]}, {}, ["3", "[3]"], ["x", '["3"]'], id="anyof_integer_array"),
    pytest.param({"anyOf": [OBJECT_A, OBJECT_B]}, {}, ['{"a":1}', '{"b":"x"}'], ['{"c":1}', '{"a":"x"}'], id="anyof_two_objects"),
    pytest.param({"oneOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "object"}]}, {}, ["[1]", "{}"], ["1"], id="oneof_containers"),
    pytest.param({**OBJECT_A, "type": ["object", "null"]}, {}, ['{"a":1}', "null"], ["{}", '"null"'], id="nullable_object"),
    pytest.param({"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]}, {}, ["[1]", "null"], ['["a"]'], id="nullable_array"),
    pytest.param({"anyOf": [{"$ref": "#/$defs/o"}, {"type": "null"}]}, {"$defs": {"o": OBJECT_A}}, ['{"a":1}', "null"], ['{"a":"x"}'], id="nullable_ref_object"),
    pytest.param({"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}, {"properties": {"a": {"maximum": 3}}}]}, {}, ['{"a":3}'], ['{"a":4}', "{}"], id="allof_objects"),
    pytest.param({"const": {"a": 1}}, {}, ['{"a":1}'], ['{"a":2}'], id="const_object"),
    pytest.param({"enum": [[1, 2], [3]]}, {}, ["[1,2]", "[3]"], ["[2,1]"], id="enum_arrays"),
])
def test_lark_structural_value_schemas(
    tokenizer: llg.LLTokenizer,
    value: dict[str, Any],
    root: dict[str, Any],
    valid: list[str],
    invalid: list[str],
) -> None:
    parameters = {"type": "object", "properties": {"value": value}, "required": ["value"], **root}
    tools = [tool(parameters)]
    for text in valid:
        assert accepts(tokenizer, tools, call(parameter("value", text))), repr(text)
    for text in invalid:
        assert not accepts(tokenizer, tools, call(parameter("value", text))), repr(text)


@pytest.mark.parametrize(("value", "valid", "invalid"), [
    pytest.param({"type": "string", "minLength": 2.0, "maxLength": 3.0}, ["ab", "abc"], ["a", "abcd"], id="decimal_string_lengths"),
    pytest.param({"type": "array", "items": {"type": "integer"}, "minItems": 1.0, "maxItems": 2.0}, ["[1]", "[1,2]"], ["[]", "[1,2,3]"], id="decimal_item_counts"),
    pytest.param({"type": "object", "patternProperties": {"^a": {"type": "integer"}}, "additionalProperties": False}, ['{"ab":1}'], ['{"ab":"x"}', '{"b":1}'], id="nested_pattern_properties"),
    pytest.param({**OBJECT_A, "$id": "urn:example:object"}, ['{"a":1}'], ['{"a":"x"}'], id="nested_id_annotation"),
    pytest.param({"$id": "urn:example:item", "$defs": {"n": {"type": "integer"}}, "type": "object", "properties": {"x": {"$ref": "#/$defs/n"}}, "required": ["x"], "additionalProperties": False}, ['{"x":1}'], ['{"x":"a"}'], id="nested_id_ref_object"),
    pytest.param({"$id": "urn:example:item", "$defs": {"n": {"type": "integer"}}, "$ref": "#/$defs/n"}, ["1"], ["a"], id="nested_id_ref_scalar"),
    pytest.param({"anyOf": [{"type": "string"}, OBJECT_A]}, ["hello", '{"a":1}', "{ not json"], ["a </parameter> b"], id="string_or_object"),
    pytest.param({"anyOf": [{"type": "string", "enum": ["auto"]}, OBJECT_A]}, ["auto", '{"a":1}'], ["manual", '{"a":"x"}'], id="enum_or_object"),
])
def test_lark_matches_json_compiler_support(
    tokenizer: llg.LLTokenizer,
    value: dict[str, Any],
    valid: list[str],
    invalid: list[str],
) -> None:
    parameters = {"type": "object", "properties": {"value": value}, "required": ["value"]}
    tools = [tool(parameters)]
    for text in valid:
        assert accepts(tokenizer, tools, call(parameter("value", text))), repr(text)
    for text in invalid:
        assert not accepts(tokenizer, tools, call(parameter("value", text))), repr(text)


def test_lark_decimal_property_counts(tokenizer: llg.LLTokenizer) -> None:
    parameters = {
        "type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"],
        "additionalProperties": True, "minProperties": 2.0, "maxProperties": 2.0,
    }
    tools = [tool(parameters)]
    count = parameter("count", "1")
    assert accepts(tokenizer, tools, call(count + parameter("extra", "x")))
    assert not accepts(tokenizer, tools, call(count))
    assert not accepts(tokenizer, tools, call(count + parameter("extra", "x") + parameter("more", "y")))
