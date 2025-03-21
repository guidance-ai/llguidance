from ._lib import LLMatcher
from typing import Literal, List
from .gbnf_to_lark import any_to_lark
import json

GrammarFormat = Literal[
    "lark",
    "gbnf",
    "ebnf",
    "cfg",
    "grammar",
    "json_schema",
    "json",
    "regex",
    "choice",
    "llguidance",
]


def grammar_from(format: GrammarFormat, text: str) -> str:
    """
    Create a llguidance grammar definition from a given grammar text
    of the specified type.

    Args:
        format: The format of the grammar text
            "lark": Lark grammar, see https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md
            "gbnf", "ebnf", "cfg", "grammar": Lark grammar or GBNF grammar, see https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
            "json_schema", "json": JSON schema, see https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md
            "regex": Regular expression, see https://docs.rs/regex/latest/regex/#syntax
            "choice": JSON-formatted list of strings, e.g. '["a", "b", "c"]'
            "llguidance": JSON object like: {"grammars": [{"lark_grammar": "..."},{"json_schema": {...}}]}
        text: The grammar text

    Returns:
        The llguidance grammar definition as a string.
        This can be passed to LLInterpreter or LLMatcher.

    Raises:
        ValueError: If the format is not recognized

    Note:
        To get "any JSON object" grammar use:
            grammar_from("json_schema", '{"type": "object"}')
    """

    if format == "lark":
        return LLMatcher.grammar_from_lark(text)
    if format in ("gbnf", "ebnf", "cfg", "grammar"):
        return LLMatcher.grammar_from_lark(any_to_lark(text))
    if format in ("json_schema", "json"):
        return LLMatcher.grammar_from_json_schema(text)
    if format == "regex":
        return LLMatcher.grammar_from_regex(text)
    if format == "choice":
        if isinstance(text, str):
            lst: List[str] = json.loads(text)
        else:
            lst = text
        lark = "start: " + " | ".join(json.dumps(x) for x in lst)
        return LLMatcher.grammar_from_lark(lark)
    if format == "llguidance":
        return text
    raise ValueError(f"Unknown grammar type: {format}")
