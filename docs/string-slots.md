# Prepared grammars with named string slots

A `GrammarWithLexer` entry can declare `"string_slot": "targets"` alongside
its name and fallback `lark_grammar`. Other entries reference it through the
existing `@name` syntax. Slot names contain 1–128 ASCII letters, digits or
underscores and must be unique within the bundle.

```python
from llguidance import LLGrammarTemplate

template = LLGrammarTemplate(tokenizer, {"grammars": [
    {"name": "main", "lark_grammar": "start: @target"},
    {"name": "target", "string_slot": "targets",
     "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING"},
]})
matcher = template.new_matcher({"targets": ["Kitchen", "Bedroom"]})
fallback = template.new_matcher()
```

Bindings contain literal string values. They are encoded as JSON strings,
sorted and deduplicated; they cannot inject grammar source. Binding replaces
the fallback language, rather than intersecting with it. Only canonical JSON
encodings of bound strings are admitted. Omission retains the fallback;
an empty list, unknown name or invalid value raises an error. An empty string
is valid. Fallbacks must resolve through unary rules to a non-nullable plain
lexeme, without captures, parameters, token limits or special lexical options.

A binding replaces the entire fallback, including any pattern or length
restriction it carries. For example, a fallback matching only uppercase strings
can be bound to `"Kitchen"`. This API is not JSON Schema intersection. Bound
values use canonical JSON encoding: `"é"` is accepted for the value `é`, while
its alternative Unicode-escaped spelling is not.

The template uses an internal prepared-grammar layer that owns the compiled
structure, tokenizer factory, parser limits and grammar token limit. Preparing
that structure is independent of slot specialization; each new matcher has its
own generation state. No public compiled variant is added to `GrammarInit`.

The template retains its tokenizer factory and parser limits. Request binding
reuses compiled structural tables, clones the grammar/regex arena, compiles
only supplied slot expressions and initializes a fresh lexer/parser. Compiled
vectors are currently copied, and DFA state is not shared between matchers.
Templates are immutable and may be shared between threads. Specialized
matchers can be cached and copied with the existing `deep_copy()` API.

`max_binding_values` (default 10,000) and `max_binding_bytes` (default 1 MiB)
bound aggregate input before specialization. Duplicate input counts toward
these budgets. Byte accounting uses UTF-8; JSON escaping can expand input by
up to six times. Existing lexer-fuel and parser limits also apply. Callers
must bound request bodies before Python/Rust deserialization as usual.

Rust callers use `GrammarTemplate::new` with an `Arc<ParserFactory>`,
`ParserLimits` and `StringSlotLimits`, then `new_parser(&bindings)`. The
template preserves the factory's inference capabilities and grammar token
limit. This API is not exposed through the C ABI yet.

An ordinary matcher still accepts such a bundle using its fallbacks. Engines
that do not implement this extension may ignore the new metadata, so serving
layers must explicitly check the template API/capability before accepting
binding requests. Do not silently downgrade a bound request to a fallback.
