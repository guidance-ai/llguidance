"""Offline preflight fixtures and explicit differences from the real compiler."""

import copy
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import NoSuchResource

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "schemas/llguidance-json-schema-2020-12-v1.json"
PROFILE = json.loads(PROFILE_PATH.read_text())


def no_retrieval(uri):
    raise NoSuchResource(ref=uri)


VALIDATOR = Draft202012Validator(PROFILE, registry=Registry(retrieve=no_retrieval))

ACCEPTED = [
    True,
    False,
    {},
    {"type": ["string", "null"]},
    {"type": "integer", "minimum": -3, "exclusiveMaximum": 4, "multipleOf": 0.5},
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False},
    {"type": "array", "prefixItems": [{"type": "string"}, True], "items": False},
    {"$defs": {"node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}}}}, "$ref": "#/$defs/node"},
    {"enum": [{"if": 12, "$ref": 34}, ["not", "a schema"]]},
    {"const": {"x-guidance": {"made_up": True}}},
    {"default": {"if": 12}, "examples": [{"$ref": False}], "x-vendor": {"items": 123}},
    {
        "x-guidance": {
            "item_separator": ",",
            "key_separator": ":",
            "whitespace_flexible": False,
            "whitespace_pattern": None,
            "coerce_one_of": True,
            "lenient": False,
            "json_allowed_escapes": None,
            "json_allow_general_unicode_escapes": True,
        }
    },
    {"format": "unknown"},
]
REJECTED = [
    None,
    [],
    "string",
    0,
    {"type": "float"},
    {"type": []},
    {"type": ["string", "string"]},
    {"minimum": "0"},
    {"exclusiveMinimum": True},
    {"multipleOf": 0},
    {"minItems": -1},
    {"minLength": 1.5},
    {"required": ["a", "a"]},
    {"properties": []},
    {"properties": {"a": 3}},
    {"items": [{"type": "string"}]},
    {"prefixItems": []},
    {"allOf": []},
    {"enum": []},
    {"enum": [1, 1]},
    {"$schema": "http://json-schema.org/draft-07/schema"},
    {"$ref": "https://example.invalid/schema"},
    {"format": "made-up"},
    {"contains": True},
    {"uniqueItems": False},
    {"if": True, "then": True},
    {"not": False},
    {"additionalItems": True},
    {"$dynamicRef": "#node"},
    {"unevaluatedProperties": False},
    {"$defs": {"bad": {"not": {}}}},
    {"allOf": [{"contains": {}}]},
    {"properties": {"child": {"x-guidance": {}}}},
    {"x-guidance": None},
    {"x-guidance": {"whitespace_flexible": None}},
    {"x-guidance": {"item_separator": 5}},
    {"x-guidance": {"retriever": None}},
    {"x-guidance": {"made_up": True}},
]


def test_profile_is_valid_2020_12_schema():
    Draft202012Validator.check_schema(PROFILE)


@pytest.mark.parametrize("schema", ACCEPTED)
def test_accepts_structural_fixtures_without_mutation(schema):
    original = copy.deepcopy(schema)
    VALIDATOR.validate(schema)
    assert schema == original


@pytest.mark.parametrize("schema", REJECTED)
def test_rejects_malformed_or_out_of_profile_fixtures(schema):
    assert not VALIDATOR.is_valid(schema)


def test_all_profile_references_resolve_locally():
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("$ref"), str):
                ref = value["$ref"]
                assert ref.startswith("#/")
                target = PROFILE
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                assert isinstance(target, (dict, bool))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(PROFILE)


def test_supported_keyword_and_option_inventories_need_review_when_changed():
    # Maintenance tripwire: an upstream addition requires reviewing this
    # profile and its limitations rather than silently drifting.
    source = (ROOT / "parser/src/json/schema.rs").read_text()
    implemented = source.split("const IMPLEMENTED:", 1)[1].split("];", 1)[0]
    names = set(re.findall(r'"([^"]+)"', implemented))
    properties = PROFILE["$defs"]["schema"]["properties"]
    assert names <= set(properties)
    assert {name for name in names if properties[name] is False} == {"additionalItems"}

    source = (ROOT / "parser/src/json/compiler.rs").read_text()
    fields = source.split("pub struct JsonCompileOptions {", 1)[1].split("\n}", 1)[0]
    names = set(re.findall(r"pub (\w+):", fields)) - {"retriever"}
    options = PROFILE["$defs"]["rootSchema"]["properties"]["x-guidance"]["properties"]
    assert names == set(options)

    source = (ROOT / "parser/src/json/formats.rs").read_text()
    formats = set(re.findall(r'(?m)^\s*"([^"]+)"\s*=>', source))
    assert formats == set(properties["format"]["enum"])


@pytest.mark.parametrize(
    ("schema", "preflight_ok", "compiles"),
    [
        ({"type": "string", "minLength": 1}, True, True),
        ({"type": "array", "items": {"type": "integer"}}, True, True),
        ({"$defs": {"n": {"type": "integer"}}, "$ref": "#/$defs/n"}, True, True),
        ({"enum": [{"if": 12, "$ref": 34}]}, True, True),
        ({"x-vendor": {"not": "a schema"}, "type": "boolean"}, True, True),
        ({"x-guidance": {"whitespace_pattern": None, "json_allowed_escapes": None}, "type": "string"}, True, True),
        (False, True, False),
        ({"type": "number", "minimum": 2, "maximum": 1}, True, False),
        ({"type": "string", "pattern": "(?=a)a"}, True, False),
        ({"$ref": "#/$defs/missing"}, True, False),
        ({"oneOf": [{"type": "string"}, {"type": "string"}]}, True, False),
        ({"type": "string", "format": "made-up", "x-guidance": {"lenient": True}}, False, True),
        ({"type": "object", "properties": {"child": {"type": "string", "x-guidance": {}}}}, False, True),
    ],
)
def test_compiler_boundary(schema, preflight_ok, compiles):
    from llguidance import LLMatcher

    assert VALIDATOR.is_valid(schema) is preflight_ok
    grammar = LLMatcher.grammar_from_json_schema(json.dumps(schema))
    is_error, _messages = LLMatcher.validate_grammar_with_warnings(grammar)
    assert (not is_error) is compiles
