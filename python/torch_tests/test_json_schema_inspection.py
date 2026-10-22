"""Diagnostic invariants, without snapshots of internal rule IDs or formatting."""

import copy
import json
import re

import pytest
from llguidance import LLMatcher, LLTokenizer


def inspect(schema, defaults=None, overrides=None):
    return json.loads(LLMatcher.inspect_json_schema(schema, defaults, overrides))


def test_normalization_combines_bounds():
    result = inspect({"allOf": [{"type": "number", "minimum": 3}, {"maximum": 8}]})
    assert result["version"] == 1
    root = result["normalized"]["root"]
    assert re.search(r"minimum:\s*Some\(\s*3\.0,?\s*\)", root)
    assert re.search(r"maximum:\s*Some\(\s*8\.0,?\s*\)", root)
    assert result["stage"] == "json-lowering-before-tokenization-and-optimization"
    assert result["text_format"] == "unstable-debug"
    assert result["debug_text_may_be_truncated"] is True
    assert result["lexer_expression_character_limit"] == 512
    assert result["grammar"]
    assert result["lexer"]


def test_recursive_definitions_are_retained_without_expansion():
    schema = {
        "$defs": {
            "node": {"type": "object", "properties": {"next": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/node"}]}}}
        },
        "$ref": "#/$defs/node",
    }
    result = inspect(schema)
    definitions = result["normalized"]["definitions"]
    assert definitions
    assert list(definitions) == sorted(definitions)
    assert "Ref(" in result["normalized"]["root"]
    assert any("Ref(" in value for value in definitions.values())
    assert len(json.dumps(result)) < 100_000


def test_lexer_exposes_constraints_missing_from_rule_names():
    result = inspect({"type": "string", "pattern": "^proof[0-9]+$"})
    assert "proof" in result["lexer"]
    assert result["grammar"]


def test_order_input_and_generation_are_unchanged():
    schema = {
        "type": "object",
        "properties": {"z_last_alphabetically": {"type": "integer"}, "a_first_alphabetically": {"type": "string"}},
        "required": ["z_last_alphabetically", "a_first_alphabetically"],
        "additionalProperties": False,
    }
    before = copy.deepcopy(schema)
    tokenizer = LLTokenizer("byte")
    grammar = LLMatcher.grammar_from_json_schema(schema)
    original = LLMatcher(tokenizer, grammar)
    result = inspect(schema)
    assert schema == before
    text = result["normalized"]["root"]
    assert text.index("z_last_alphabetically") < text.index("a_first_alphabetically")
    after = LLMatcher(tokenizer, LLMatcher.grammar_from_json_schema(schema))
    assert original.compute_logit_bias() == after.compute_logit_bias()
    tokens = tokenizer.tokenize_str('{"z_last_alphabetically":1,"a_first_alphabetically":"x"}')
    assert original.validate_tokens(tokens) == len(tokens)
    assert after.validate_tokens(tokens) == len(tokens)


def test_python_option_precedence_matches_wrapper():
    schema = {"type": "string", "x-guidance": {"whitespace_flexible": True}}
    defaults = {"whitespace_flexible": False, "item_separator": ", "}
    overrides = {"whitespace_flexible": False}
    before = copy.deepcopy((schema, defaults, overrides))
    result = inspect(schema, defaults, overrides)
    assert (schema, defaults, overrides) == before
    assert result["effective_options"]["item_separator"] == ", "
    assert result["effective_options"]["whitespace_flexible"] is False
    grammar = LLMatcher.grammar_from_json_schema(schema, defaults, overrides)
    assert LLMatcher.validate_grammar(grammar) == ""


def test_warnings_are_retained():
    schema = {"type": "string", "format": "unimplemented-format", "x-guidance": {"lenient": True}}
    result = inspect(schema)
    error, ordinary_warnings = LLMatcher.validate_grammar_with_warnings(LLMatcher.grammar_from_json_schema(schema))
    assert not error
    assert ordinary_warnings
    assert result["warnings"]
    assert all(item["count"] >= 1 for item in result["warnings"])
    assert any("format" in item["message"].lower() for item in result["warnings"])


@pytest.mark.parametrize(
    "schema",
    [
        True,
        {"type": "array", "items": {"type": "boolean"}},
        {"oneOf": [{"type": "string"}, {"type": "number"}]},
        {"x-guidance": {"whitespace_pattern": None, "json_allowed_escapes": None}, "type": "string"},
    ],
)
def test_supported_schemas_and_serialized_input(schema):
    assert inspect(json.dumps(schema))["grammar"]


@pytest.mark.parametrize(
    "schema",
    [
        False,
        {"type": "number", "minimum": 8, "maximum": 3},
        {"$ref": "#/$defs/missing"},
        {"type": "string", "pattern": "(?=a)a"},
        {"type": "string", "format": "unimplemented-format"},
        {"oneOf": [{"type": "string"}, {"type": "string"}]},
        {"x-guidance": None},
        {"x-guidance": {"unknown": True}},
        {"x-guidance": {"whitespace_flexible": None}},
    ],
)
def test_errors_are_not_hidden(schema):
    error, _ = LLMatcher.validate_grammar_with_warnings(LLMatcher.grammar_from_json_schema(json.dumps(schema)))
    assert error
    with pytest.raises(ValueError):
        inspect(json.dumps(schema))


def test_invalid_json_raises_value_error():
    with pytest.raises(ValueError):
        inspect("{")


def test_boolean_schema_with_options_keeps_existing_error():
    with pytest.raises(ValueError, match="Expecting object schema"):
        inspect("true", {"lenient": True})
