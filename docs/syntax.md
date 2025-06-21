# Syntax of LLGuidance Grammars

LLGuidance supports a variant of syntax used by Python [Lark parsing toolkit](https://github.com/lark-parser/lark).
We also provide a [gbnf_to_lark.py script](../python/llguidance/gbnf_to_lark.py) to convert from
[GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) format used in
[llama.cpp](https://github.com/ggerganov/llama.cpp).
These makes it easier to get started with a new grammar,
and provide a familiar syntax, however neither is a drop-in replacement for Lark or GBNF.

For a general intro to Lark syntax, see:

- [How to write a DSL](https://blog.erezsh.com/how-to-write-a-dsl-in-python-with-lark/) blog post;
  ignore the part about syntax trees
- [Lark documentation](https://lark-parser.readthedocs.io/en/latest/);
  LLGuidance uses Earley parser, with an equivalent of `lexer='dynamic'` setting in Lark

## Extensions to Lark syntax

Following are the extensions to Lark syntax.

### Minor syntax changes

- `expr{M,N}` can be used instead of `expr~M..N`
  (with either `M` or `N` being optional; `expr{N}` is also supported)
- both `//` and `#` can be used for comments
- `-` is valid in identifiers

### Inline JSON Schemas

You can also inline JSON schema in the lark grammar, using `%json { ... }` syntax
(it behaves like a non-terminal).

For example, this defines a JSON function calling schema for Meta Llama 3.1,
according to [their website](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling):

```lark
start: TEXT | fun_call
TEXT: /[^{](.|\n)*/
fun_call: %json {
  "type": "object",
  "properties": {
    "name": { "const": "get_weather" },
    "parameters": {
      "type": "object",
      "properties": {
        "city": { "type": "string" }
      },
      "required": ["city"]
    }
  },
  "required": ["name", "parameters"]
}
```

This will accepting strings like `There is no function I can call`
and `{"name":"get_weather", "parameters": {"city":"Seattle"}}`.
If the string starts with `{`, it will be forced to follow the given JSON schema.

If you have more functions, you can use either `%json { "anyOf": [ ... ] }`,
or `fun_call1 | fun_call2` where each `fun_callX` is defined as `%json { ... }`.

### Special tokens

Special tokens can referenced via `<token_name>` syntax (i.e., any string between `<` and `>`),
for example `<|ENDOFTEXT|>`.
They cannot be used inside of terminals, but can be used in regular rules.
The exact set of available tokens depends on the tokenizer used.

You can also use numeric token ids, as in `<[128010]>` (this is `<|python_tag|>` in Meta Llama tokenizer).
Tou can also use ranges like `<[128000-128255]>` for all Llama special tokens, or
even lists of ranges like `<[128000-128100,128130-128170]>`; ranges are inclusive.

For example, this is how to constrain JSON function calling for Meta Llama 3.1,
according to their [source repo](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#model-response-format-5) (and yes, it's [different](https://github.com/meta-llama/llama-models/issues/266) than the website).

```lark
start: TEXT | fun_call
TEXT: /[^{](.|\n)*/
fun_call: <|python_tag|> json_body <|eom_id|>
json_body: %json {
  # ... same as above ...
}
```

Here, we also insist on regular text responses not starting with `{` to avoid confusing the model,
but normally the JSON response should start with the special `<|python_tag|>` token.

Another example is function calling in Llama for Brave search or Wolfram Alpha:

```lark
start: normal_text | brave | wolfram
normal_text: /(.|\n)*/
brave: <|python_tag|> "brave_search.call(query=" JSON_STRING ")" <|eom_id|>
wolfram: <|python_tag|> "wolfram_alpha.call(query=" JSON_STRING ")" <|eom_id|>
JSON_CHAR: /(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])/
JSON_STRING: "\"" JSON_CHAR* "\""
```

Note that is is [critical for performance](#terminals-vs-rules) for the `JSON_STRING` to be uppercase,
so that it is treated as a terminal (single lexeme, or regex and not a context-free grammar production).

BTW, in this case you may want to replace the JSON string definition
with a definition of a Python string, depending on how the model was trained.

### Reasoning/thinking

Yet another example is "thinking" or reasoning models distilled from DeepSeek-R1.
A grammar for forcing a JSON-formatted address may look like this:

```lark
start: <think> "\n" /(.|\n)*/ </think> address
address: %json {
    "type": "object",
    "properties": {
        "street": { "type": "string" },
        "city": { "type": "string" },
        "zip": { "type": "number" }
    },
    "required": ["street", "city", "state", "zip"]
}
```

Often, the chat format already includes initial `<think>\n` - in these cases
you can use `start: /(.|\n)*/ </think> address` as the grammar.
You can also use `/(.|\n){1000,3000}/` to place lower and upper bounds on the thinking amount.

This assumes `<think>` is a special token. If it was just a string, you would need
to use [`suffix="</think>"`](#lazy-lexemes).

### Parametric grammars (unique lists, permutations etc.)

Rules can be parameterized by a 64-bit value, which allows for expressing
combinatorial structures like unique lists and permutations.
See [Parametric grammars](./parametric.md) for details.

### Lexeme options

Some of these features (especially `stop`) are primarily for compatibility with [Guidance](https://github.com/guidance-ai/guidance).

For all rules,
`foo[capture]: ...` will generate a capture group named `foo` in the output,
while `foo[capture="bar"]: ...` will generate a capture group named `bar`.

For rules bodies of which are terminals (regexes or uppercase names), you can specify additional options:
`lazy`, `max_tokens`, `temperature`, `suffix`, and `stop`.
Example: `mygen[stop="\n", max_tokens=10, temperature=0.7]: /.*/`

The `temperature` alters temperature while sampling tokens inside of the terminal,
while `max_tokens` limits the number of tokens generated for the terminal.

### Lazy lexemes

Specifying `stop=""` will make the EOS token of the model act as the stop condition.
This is only useful if there is some other rule following this rule (otherwise the model will stop on EOS anyways),
in which case llguidance will try to "backtrack" the EOS token
(hide it from the LLM; but see notes about backtracking below).

If `stop` or `suffix` are specified and non-empty, or if `lazy` is specified, the regex
will be treated as lazy, meaning it will match as few bytes as possible.
Consider the following rules:

```lark
with_stop[capture, stop="<end>"]: /.*/        // capture: "foo"
outer_stop[capture]: with_stop "<end>"        // capture: "foo<end>"
outer_stop_problem[capture]: with_stop "b"    // capture: "foob"

with_suffix[capture, suffix="<end>"]: /.*/    // capture: "foo"
outer_suffix[capture]: with_suffix            // capture: "foo<end>"

with_lazy[capture, lazy]: /.*<end>/           // capture: "foo<end>"
outer_lazy[capture]: with_lazy                // capture: "foo<end>"
```

They are all lazy, so
given a string `"foo<end>bar<end>"`, they will all match just the prefix `"foo<end>"`
(except for `outer_stop_problem` which will match `"foo<end>b"`).
The captures are listed in the comments.

All the `with_*` rules use the same regex, but `with_stop` will attempt to "hide"
the final `"<end>"` from the model, by "backtracking" the tokens corresponding to it.
This is often not supported by server-side LLM infrastructure, and may confuse the model.
To avoid backtracking, a typical pattern (`outer_stop`) is to have another rule that appends `"<end>"`
again after `with_stop` that ate it.
The capture for `outer_stop` has `"<end>"` while the capture `with_stop` does not.
This requires certain gymnastics in llguidance implementation and may sometimes not work as expected.
Additionally, if you attempt to add something that is not `"<end>"` after `with_stop`,
as in `outer_stop_problem`, the backtracking will be necessary.

The `suffix` offers a simpler alternative: the capture for the rule with `suffix`
does not include the suffix, but the captures for upper-level rules do,
and there is never any backtracking.

For both `stop` and `suffix`, the value can be any terminal (string literal, regex, or uppercase name).
In either case you can also specify `stop_capture="my_name"` which will
cause the string matching `stop` or `suffix` to be captured as `my_name`.

If you don't care about captures, you can just put `lazy` on the rule.
It is most useful when the regular expression ends with some delimiter.
If it doesn't, the results may be surprising:
for example, `foo[lazy]: /.*/` will match only the empty string,
while `foo[lazy]: /[0-9]+/` will only match a single digit.

### Tool calling

Some models have been finetuned to use normal text, not special tokens, for tool calling.
Here's an example how `lazy` can be used to allow any text interspersed with
tool calls:

```lark
start: ( f_foo | f_bar )* f_end
f_end: TEXT
TEXT: /(.|\n)*/

f_foo_hd[lazy]: TEXT "<function"
f_foo: f_foo_hd "=foo>" %json { "type": "object" } "</function>"

f_bar_hd[lazy]: TEXT "<function"
f_bar: f_bar_hd "=bar>" /[0-9]+/ "</function>"
```

The above allows for anything, until the text `<function` is generated.
Then it will force the model to choose between `=foo>` or `=bar>`.
Then we'll either enforce valid JSON or a number, followed by string `</function>`.
Then again any text is allowed, etc.

Note that the tool call can be missing algtogether, you can change `*` to `+` if you want to enforce at least one tool call.
You can also change the definition of `TEXT` to limit the free-form text.

See [Special tokens](#special-tokens) if the "trigger" for the tool call is a special token
like `<|python_tag|>`, not a string like `<function`.

The `llguidance.StructTag` API, [inspired](https://github.com/mlc-ai/xgrammar/blob/fd9ee31/python/xgrammar/grammar.py#L211) by XGrammar, just compiles down to the above.

### And/Not operators in regexes

The regular expressions in LLGuidance can use additional operators: `&` (and) and `~` (not).
They can only be used outside of the `/.../` syntax, i.e., in the Lark terminal (token) definitions.
The `&` operator binds tighter than `|` (alternation), so `A & B | C` means `(A & B) | C`.
The `~` operator binds tighter than even `+` or `*`, so `~A+` means `(~A)+`.

The negation operator `~` is particularly tricky to use right.
For example, this is a terminal definition that matches any list of ASCII lines,
but they cannot have two newlines in a row:

```lark
ASCII_LINES: /[a-zA-Z \n]*/ & ~/(?s:.*)\n\n(?s:.*)/
```

Note that `/[a-zA-Z \n]*/ & ~/\n\n/` would mean any list of lines, also with two newlines in a row,
except for the exact string `\n\n`.
Also, `/[a-zA-Z \n]*/ & ~/(.*)\n\n(.*)/` would allow double newlines, but if there is at least two of them
(`/./` doesn't match newline).

These operators are sometimes expensive to use, so you should generally avoid them if alternatives exist.

### Structured %regex

LLGuidance supports [extended regex syntax](https://docs.rs/regex/latest/regex/#syntax) in `/.../`.
This includes character classes (`/[a-zA-Z]/`), repetition (`/a+/`, `/a*/`, `/a{10,100}/`, `/a{10,}/`, `/a?/`),
alternation (`/a|b/`), and grouping (`/(ab)+/`).

Additionally, regexes can be defined with the standard Lark syntax, using [uppercase names](#terminals-vs-rules):

```lark
// INT is equivalent to /(-)?[0-9]+/
INT: "-"? UINT
UINT: DIGIT+
DIGIT: /[0-9]/
```

Additionally, "structured" regex nodes can be defined using `%regex { ... }` syntax.

#### Substring

`%regex { "substring_chunks": lst }` will match `lst[n:m].join("")` for some `n <= m <= len(lst)`.
Additionally `substring_words` or `substring_chars` can be specified.
For example:

- `%regex { "substring_chunks": ["abc", "de", "fg"] }` matches `""`, `"abc"`, `"de"`, `"abcde"`, `"defg"`, `"abcdefg"`, etc.; it doesn't match `"ab"` not `"cde"`
- `%regex { "substring_words": "foo bar. baz" }` is equivalent to
  `%regex { "substring_chunks": ["foo", " ", "bar", ".", " ", "baz"] }`
- `%regex { "substring_chars": "ab c" }` is equivalent to
  `%regex { "substring_chunks": ["a", "b", " ", "c"] }`

We may want to switch to more JSON-schema like syntax:

```lark
ABC: %regex {
  "type": "substring",
  "chunks": ["a", "b", "c"]
}
```

#### Future extensions

Following `%regex` syntax is planned (compatible with JSON schema):

```lark
BOUNDED_NUM: %regex {
  "type": "number",
  "minimum": -17.3,
  "maximum": 33.721
}

MULT_NUM: %regex {
  "type": "integer",
  "exclusiveMinimum": 0,
  "multipleOf": 10
}
```

### Grammar options

Certain grammar options can be set by using `%llguidnace { ... }`,
by passing it a JSON object with the options;
see `LLGuidanceOptions` in [api.rs](../parser/src/api.rs#L24).
Example: `%llguidance { "no_forcing": true }`.
It can be specified multiple times, with the options being merged.

You can also start the grammar file with `%llguidance {}` to indicate
that llguidance should be used to process the grammar.

### Multiple grammars

The input to LLGuidance consists of a list of grammars. This can be accessed via
[LLGuidance API](../parser/src/api.rs). Each of these can be a Lark grammar or a JSON schema.
With the introduction of `%json` in Lark syntax
there is less need now for using multiple grammars, but it is still supported.
We may add nested lark grammars in future.

Inside of Lark grammar, you can reference other grammars using syntax like `@my_grammar`,
refering to grammar with `"name": "my_grammar"` (numeric reference like `@17` are **no longer supported**).
The top-level grammar is at index 0.

You can specify temperature for subgrammar by referencing it via
`my_temp_json[temperature=0.7]: @my_json` syntax.

Example:

```json
{
  "grammars": [
    {
      "lark_grammar": "start: /(.|\\n)*/ | fun\nfun: <|python_tag|> @fcall <|eom_id|>",
    },
    {"name": "fcall", "json_schema": { ... }}
  ]
}
```

#### Subgrammar details

Generally, subgrammars share the same context-free grammar but have a separate
set of lexemes (lexeme class).
The parser keeps track of a stack of lexeme classes, and considers the top one
to be the current lexeme class.
The `%ignore` is applied based on the top lexeme class.

Temperature and `max_tokens` can be applied to the current lexeme class as well.

There may be issues with subgrammars spanning LLM token boundaries.

If two lexeme classes share the `%ignore` regex, and `max_tokens=` and `temperature=`
are not used, the lexeme classes are merged, which generally allows for parsing
of more grammars.
For example, consider:

```lark
start: a | b
a: %json { A }
b: %json { B }
```

Normally, the parser would have to pick between lexeme class for either A or B
at the first `{` (it would always pick A since it comes first in the grammar).
However, if the classes for A and B are merged, the grammar will be equivalent to
`start: %json { "anyOf": [A, B] }` which is generally what the
[users expect](https://github.com/guidance-ai/llguidance/issues/113).


### Features to avoid

- `stop=...` - use `suffix=...` or just `lazy`
- `max_tokens=...` - any use of `max_tokens` will disable rollback, which is needed for spec-decoding; it also makes the parser slower and prevents subgrammar merging
- `temperature=...` - this is not supported in most server side integrations and prevents subgrammar merging

### Unsupported Lark features

Following features of Lark syntax are currently not supported:

- lookarounds in lexer regexes
- lazy modifier (`?`) in lexer regexes; you [can use](#lexeme-options) `[lazy]` to make the entire terminal lazy
- priorities of terminals
- templates
- imports (other than built-in `%import common`)
- regexes use Rust `regex` crate [syntax](https://docs.rs/regex/latest/regex/#syntax), not Python's `re` (though they are similar)
- certain string syntax, see [issue](https://github.com/microsoft/llguidance/issues/54)

## Performance tips

### Terminals vs rules

TL;DR: avoid regexes matching only single characters like `/[a-z]/`,
use `/[a-z]+/` or similar instead where appropriate.

A Lark file defines a context-free grammar (CFG).
Another, more well-known, type of grammar is a regular grammar, which is used by regular expressions.
Regular grammars are more limited than CFGs, but also faster to parse.
Therefore, typically programming languages are parsed in two stages:

- first, a lexer converts the input bytes into a stream of lexemes or terminals (aka tokens, but we avoid that term to avoid confusion with LLM tokens)
- the parser then uses a CFG to parse the lexemes into higher-level structures

Thus, the lexer matches strings, while the parser matches the structure.

Most often, the programming languages allow some sort of white-space between lexemes,
which is used to separate them, but otherwise ignored.
For example, the strings `foo+bar` and `foo + bar` are typically equivalent
(lexed as `foo`, `+`, `bar`), while `fo o + bar` is not.
Similarly, `x+=12` is equivalent to `x += 12`, but not to `x + = 12` nor to `x += 1 2`.
Lexemes in these cases are the fragments between which the white-space is allowed.
Note that `"xy"` and `"x y"` are not equivalent - the whole string literal is a single lexeme.

You can use `%ignore` directive to specify which kind of white-space
(or other strings, including comments) to allow between lexemes, and otherwise ignore.
For example, for JSON it would be `%ignore /[ \t\n\r]+/`.
By default nothing is ignored.

In Lark, string literals like `"while"` or `"if"` are lexemes,
and so are regular expressions enclosed in `/`...`/`, like `/[0-9]+/`.

Lark also allows using grammar operators like `|` or `*` outside of regular expressions or strings,
so you can say `/[0-9]/+` or `/[0-9]/{1,}`.
The meaning of this depends on whether you're defining a terminals (uppercase) or a rule (lowercase).
Any uppercase symbol (like `INT: /[0-9]/+` or `NUMBER: INT | FLOAT`) is treated
as a terminal (lexeme), and in particular white-space is not allowed inside of it.
Any lowercase symbol (like `digits: /[0-9]/+`) is treated as a rule,
meaning `digits` will match `123` but also `1 2 3` or `1 23` (and treat them as the same).
**Putting a repeating operator like `+` or `*` on a single-character lexeme is typically not what you want and will slow down the parser, up to the point it will refuse to run.**

All the terminals (uppersase identifiers) are compiled into regular expressions,
meaning they can't be recursive (refer to themselves, directly or indirectly).

In practice, in the grammar you can use string literals like `"while"`
to refer to keyword, but when defining a lexeme with a regular expression,
it's best to name it in uppercase, like `IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/`,
and then use `IDENTIFIER`.
Sometimes it can be useful to name the character classes (again, making sure everything is uppercase), as in:

```lark
ID_START: /[a-zA-Z_]/
ID_CHAR: /[a-zA-Z0-9_]/
IDENTIFIER: ID_START ID_CHAR*
```

### Recursive rules

TL;DR: prefer `many: one+` (or `one*`) over `many: many one | one`. Do not use `many: one many | one`.

CFGs allow for recursive rules, for example they are very convenient for defining arithmetic expressions:

```lark
expr: expr "+" term | term
term: term "*" factor | factor
factor: "(" expr ")" | NUMBER
NUMBER: /[0-9]+/
```

However, often recursion is only used to express repetition:

```lark
one: ...
many: many one | one
```

The rule `many: many one` is called left-recursive, because it refers to itself on the left side.
A rule like `many: one many` is right-recursive.
In this case, they would both express the same language.
Earley parser used by LLGuidance can handle both, but left-recursive rules are **much more efficient**
(it's often the opposite for other parsing methods).

To avoid having to think about it, you can just use `many: one+`.
This will be translated internally to the most efficient form, is easier to get right and arguably more readable.

With right-recursive rules you will hit parser item limits quite quickly (after 100-1000 repetitions) and before you do, the parser will be slower than necessary.

On related note, use `one{N}` and not `one one ... one`.
The resulting rules will be `O(log N)` in size, while the unfolded version would be `O(N)`.
Same for `one{M,N}`.
