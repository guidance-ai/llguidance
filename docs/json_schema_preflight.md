# JSON Schema structural preflight

[Version 1 of the Draft 2020-12 structural profile](../schemas/llguidance-json-schema-2020-12-v1.json)
is a self-contained JSON meta-schema for checking a schema before sending it to
llguidance. Its identity is `urn:llguidance:json-schema:2020-12:structural:v1`;
the identifier is not a hosted endpoint. Obtain the JSON file from the same
repository revision as the compiler you use.

**Passing means the declared structural checks passed.** Compile the schema to
check reference resolution, satisfiability, regex support, intersections,
`oneOf` disjointness, pattern-property overlap, and compiler resource limits.
The profile does not certify another service's accepted schemas or change
llguidance's generation behavior.

## Policy and dialect

This is a strict *authoring* profile for Draft 2020-12, including schemas with
no explicit `$schema`. A legacy draft declaration is rejected. It is not a
drop-in validator for every input accepted by every compiler configuration.

| Surface | Structural check | Remaining compiler check |
| --- | --- | --- |
| Types, bounds, counts | Recognized type names; numeric bounds; nonnegative integer counts; positive `multipleOf` | Consistency, satisfiability, numeric limits |
| Objects and arrays | Recurse through schema-valued properties, definitions, applicators, and items | Intersections, property counts, pattern overlap, size limits |
| `$ref` | String containing an empty or fragment-only reference | Target existence, cycles, resolution, compilation |
| `pattern` and separators | String value | Regex syntax and supported features |
| `format` | Names present in the compiler's format lookup | Format's actual constraints and intersections |
| `x-guidance` | Top-level option names and value types; nullable optional strings | Regex validity and effective compilation behavior |
| Annotations and extensions | Standard annotation value shapes; unknown keys remain allowed | Unknown keys do not gain assertion semantics |

Known unsupported assertion keywords are rejected, including `not`,
conditionals, `contains`, `uniqueItems`, dependency assertions, property names,
unevaluated keywords, and dynamic/recursive reference keywords. The profile
also excludes legacy `additionalItems` and `contentSchema`. The exact list is
visible as `false` properties in the JSON file. `definitions` remains available
as a compatibility spelling, with the same structural recursion as `$defs`.

These checks remain strict even when `x-guidance.lenient` is true. For example,
an unknown string format may compile in lenient mode but is outside this
profile. Nested `x-guidance` is rejected to catch misplaced options; the
compiler otherwise treats it as an ignored annotation. JSON Schema authoring
rules also reject empty/duplicate enums and duplicate types or required names
even if a compiler path could tolerate them. Format `unknown` is
included because the compiler explicitly recognizes that name; it does not
impose a string-format constraint.

Unknown annotations are allowed, including their arbitrary data. Values of
`const`, `enum`, `default`, and `examples` are instance data and are **not**
recursively interpreted as schemas. This prevents an instance containing
`{"if": 12}` or `{"$ref": 34}` from being rejected as a malformed schema.
Only declared schema-valued positions recurse.

## Offline use

For example, with the development `jsonschema` dependency:

```python
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import NoSuchResource


def no_retrieval(uri):
    raise NoSuchResource(ref=uri)


profile = json.loads(
    Path("schemas/llguidance-json-schema-2020-12-v1.json").read_text()
)
Draft202012Validator.check_schema(profile)
validator = Draft202012Validator(
    profile, registry=Registry(retrieve=no_retrieval)
)
validator.validate({"type": "string", "minLength": 1})
```

All `$ref` values inside the profile resolve within that file. Validating a
schema's `$ref` string does not fetch its target. The example rejects any
attempt to retrieve an external profile resource.

After preflight, use the compiler:

```python
from llguidance import LLMatcher

schema = {"type": "string", "pattern": "(?=a)a"}
validator.validate(schema)  # The pattern has the declared string shape.
grammar = LLMatcher.grammar_from_json_schema(schema)
is_error, messages = LLMatcher.validate_grammar_with_warnings(grammar)
assert is_error  # This lookaround is not supported by the regex compiler.
```

Other deliberate false positives include unresolved references, conflicting
bounds, an unsatisfiable `false` schema, and unsupported overlapping `oneOf`
branches. Passing the profile is never a promise of a nonempty grammar.

## Maintenance and distribution

When supported keywords, formats, options, or draft behavior change, review the
profile, its fixture corpus, and this policy together. Structural contract
changes need a new versioned filename and identifier. Tests check the declared
meta-schema, offline references, schema/data recursion, known option inventory,
and explicit differences from the actual compiler.

Run `python -m pytest python/torch_tests/test_json_schema_profile.py` with the
repository's development dependencies and its installed Python extension.
No tokenizer, model weights, or network retrieval is needed by the tests.

This initial artifact is distributed as a repository file. It makes no promise
about wheel package data or a separately hosted URL.
