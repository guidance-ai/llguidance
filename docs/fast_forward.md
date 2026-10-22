# Fast-forward (jump-forward, accelerated) tokens

Fast-forward tokens (also called zero-entropy tokens, forced tokens, fixed tokens, or jump tokens) are tokens that are added to the current sequence in one step due to the grammar constraint, possibly after some generation steps. These tokens can be processed in parallel, similar to a (very short) prefill run. This is similar to speculative decoding, except that the speculation is 100% correct. The initial prompt can be thought of as ff-tokens.

They were apparently independently introduced in [Guidance](https://github.com/guidance-ai/guidance/commit/5ad05eb80aca633aa66b3be0edd694559a0e85df) and [SGLang](https://arxiv.org/pdf/2312.07104).

An example, where fast forward tokens are useful is generating data adhering to a certain JSON schema. The grammar first forces `{"name":"` to be generated (we assume compact JSON representation here), then the model generates `John"`, the grammar forces `,"age":`, model generates `42`, and so on.

## Problems converting FF-string to FF-tokens

Consider JSON schema:

```json
{
    "properties": {
        "orderId": {"type": "string"},
        "orderName": {"type": "string"}
    },
    "required": [],
    "additionalProperties": false
}
```

It defines an object with two fields, `orderId` and `orderName`, both of which can be missing,
with no other fields allowed.
After sampling `{"` the grammar forces string `order`.
In Meta Llama3 tokenizer, `o`, `or`, `order`, `orderId`, `Id` and `Name` are all valid tokens,
but `orderName` is not.
If we naively force the token `order` at this point,
the set of next allowed tokens will be `I`, `N`, `Id`, and `Name`.
The problem is that model will then almost certainly sample `Name` and not `Id`,
because the training data has `orderId` as single token
and the sequence of tokens `order` `Id` is either completely absent
or at best very rare (if token regularization is used).

Therefore, by forcing token `order` we have severely impacted the distribution of model output.
We call such forcing "non-canonical".

However, consider the grammar (assume flexible whitespace):

```json
{
    "properties": {
        "name_of_the_person": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name_of_the_person", "age"],
    "additionalProperties": false
}
```

Here, after the initial `{"`, we can safely force
`name` `_of` `_the` `_person`, but not the final `"`, because the canonical
tokenization can use `":` (or `"` ` :` or `":"` or ...).
Forcing `"` [may confuse the model](https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb)
up to a point where it will just start producing whitespace forever.
If it doesn't, it may reduce accuracy of further output.

## Safely converting FF-strings to FF-tokens

The way llguidance avoids non-canonical forced tokens is by:

* tokenizing the forced bytes
* removing up to a few final tokens, if there exists a token in the tokenizer that spans the end boundary of the forced bytes and matches the following grammar

Here, we assume the tokenizer works on bytes (as it logically should).
Unfortunately, typically, tokenizers work on strings, so there is some gymnastics needed to make this work.

```python
# tokenize the bytes forced by the grammar resulting in a list of ints
tokens = tokenizer.encode(forced_bytes)

# check up to 4 tokens back
max_tokens = 4
max_length = len(tokenizer.decode(tokens[-max_tokens:]))

# find the first byte index that can start a token spanning past the end
# of the forced bytes and matching grammar
for idx in range(len(forced_bytes) - max_length, len(forced_bytes)):
    prefix = forced_bytes[:idx]
    suffix = forced_bytes[idx:]
    # of course in reality, you should have cached the list of tokens
    # "extending" a given byte string
    for tok_id in range(tokenizer.n_vocab):
        tok_bytes = tokenizer.decode([tok_id])
        if (tok_bytes.startswith(suffix) and 
            len(tok_bytes) > len(suffix) and
            grammar_allows(prefix + tok_bytes)
        ):
            break

# remove tokens that can be tokenized differently
while len(tokenizer.decode(tokens)) > idx:
    del tokens[-1]

# final forced tokens
return tokens
```

The `max_length` could be also set to a constant, instead of the length of the
last few tokens.

Also, `tokenizer.encode()` may need a few bytes preceding the forced bytes
to generate the right tokenization.

For example, consider forced string `name_of_the_person"` from the previous example.
It tokenizes to `name` `_of` `_the` `_person` `"`.
We take the last four tokens and check:
* is there a token that starts with `_of_the_person"`? No
* is there a token that starts with `of_the_person"`? No
* is there a token that starts with `f_the_person"`? No
* ...
* is there a token that starts with `n"`? No
* is there a token that starts with `"`? Yes, `":`. Is it longer than `"`? Yes. Is it allowed by the grammar? Yes.
* then we check how many tokens we have to remove to get rid of `"` (one)
* and return `name` `_of` `_the` `_person`

When running [maskbench](https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench),
with `max_tokens` above set to at least `2` and the llama3 tokenizer,
there are no non-canonical forced tokens.
If `max_tokens==1`, there is `23` cases of non-canonical tokenization (among ~10k tests),
however if `max_tokens==0` (ie., token healing is disabled),
almost all (`97%`) fast-forwarded token sequences are non-canonical.
Alternatively, when `max_length` is set to at least `7` bytes, all forced tokens are canonical.

While it [may be possible](https://arxiv.org/pdf/2309.08715) to construct
examples where `max_token==4` is not enough, we have not bee able to do so.

Note, that we can conservatively skip `grammar_allows()` check in the algorithm
above, and thus just compute once and for all the set of tokens that are not allowed
as the last token in forced bytes.
This drops the proportion of forced tokens in maskbench from `12.7%` to `12.1%`.

## Using LLTokenizer to convert bytes to tokens

You can use the `llguidance.LLTokenizer` class to convert forced bytes to tokens,
even when not using constraints:

```python
import transformers
import llguidance
import llguidance.hf

hf_tok = transformers.AutoTokenizer.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct")
ll = llguidance.hf.from_tokenizer(hf_tok)

new_tok, leftover = ll.tokenize_partial(b'order')
assert len(new_tok) == 0
assert leftover == b'order'

recent = ll.tokenize_bytes(b'{"')
new_tok, leftover = ll.tokenize_partial(b'name_of_the_person"',
                                        recent_tokens=recent)
print(ll.dbg_tokens(new_tok))
assert leftover == b'"'
assert ll.decode_str(new_tok) == "name_of_the_person"
```

You can also pass a few recent tokens to `tokenize_partial()` to help with joint tokenization.
It's only needed in some specific cases of punctuation following spaces.

## Observing available fast-forward spans

`LLMatcher.observe_ff_tokens()` returns a snapshot of the tokens currently available
to fast-forward. It does **not consume those tokens**, run a model, or count sampling
operations. The Rust `Matcher` and `TokenParser` expose the same method, returning
`Result<FastForwardObservation>`.

```python
from llguidance import LLMatcher, LLTokenizer

tokenizer = LLTokenizer("byte")
matcher = LLMatcher(tokenizer, 'start: "é🙂" /[xy]/')
span = matcher.observe_ff_tokens()
assert span["status"] == "ready"
assert span["token_bytes"] == "é🙂".encode("utf-8")
assert span["token_count"] == 6
assert span["byte_count"] == 6  # bytes, not the two Unicode characters
assert matcher.observe_ff_tokens() == span  # still the same unconsumed span
assert matcher.consume_tokens(span["tokens"])
assert matcher.observe_ff_tokens()["token_count"] == 0
```

The Python result has these fields:

| Field | Meaning |
| --- | --- |
| `status` | `ready`, `noncanonical`, `stopped`, `error`, or `cancelled` |
| `tokens` | The available token IDs, using the existing canonical fast-forward computation |
| `token_bytes` | Exactly `tokenizer.decode_bytes(tokens)`, including special-token spellings |
| `token_count` | `len(tokens)` for this observation |
| `byte_count` | `len(token_bytes)` for this observation |

A `ready` observation can have an empty span. In particular, token healing can retain
forced bytes because a safe complete token is not yet available. This does not identify
an application-level semantic choice point. `noncanonical` means the tokenizer cannot
supply canonical fast-forward tokens. `stopped` means the parser stopped normally.
The Rust status enum represents these three successful outcomes; errors and cancellation
use `Err`. Python returns `error` or `cancelled` and retains the cause in
`get_error()` / `stop_reason()`, following `LLMatcher`'s existing non-raising convention.
Every non-ready result has empty token and byte payloads and zero counts.

Byte accounting excludes both the already consumed prefix used during tokenization and
the suffix retained for token healing. `token_bytes` can contain an incomplete UTF-8
code point; observation does not decode it to a string or replace invalid bytes.
`compute_ff_bytes()` remains the separate API for raw bytes forced by the parser.
Those bytes can include a retained healing suffix and internal special-token markers,
so their length is not interchangeable with `byte_count` and should not be subtracted
from it to infer a residual count. Special tokens use the same decoded spelling as
`LLTokenizer.decode_bytes`; an engine that omits special tokens has a different output
byte accounting policy.

Observation may perform the same speculative byte forcing and cache population as
`compute_ff_tokens()`. It leaves committed token history unchanged. Clones observe their
own state, successful rollback/reset changes the next observation to match that restored
state, and validation of speculative drafts does not count as consumption. Errors and
cancellation remain terminal: observation, reset, and rollback do not recover an errored
or cancelled matcher. Rust matcher clones keep the existing independent cancellation
ownership rules.

### Counting observations, successful consumption, and sampling separately

The library cannot tell whether a token passed to `consume_token()` came from a model,
an accepted draft, replay, a prompt, or a prior fast-forward observation. Keep provenance
and request totals at the integration boundary. For example, this loop counts an actual
sampling call separately from successful matcher consumption:

```python
totals = dict(
    ff_observation_calls=0,
    ff_observed_tokens=0,
    ff_consumed_tokens=0,
    ff_consumed_bytes=0,
    sampled_tokens=0,
)

while not matcher.is_stopped():
    span = matcher.observe_ff_tokens()
    totals["ff_observation_calls"] += 1
    totals["ff_observed_tokens"] += span["token_count"]
    if span["status"] in ("error", "cancelled"):
        raise RuntimeError(matcher.get_error())
    if span["status"] == "stopped":
        break

    if span["tokens"]:
        for token in span["tokens"]:
            if not matcher.consume_token(token):
                raise RuntimeError(matcher.get_error())
            # Count only successful calls, including a successful prefix if a
            # later token fails. Append each accepted token in the engine here.
            totals["ff_consumed_tokens"] += 1
            totals["ff_consumed_bytes"] += len(tokenizer.decode_bytes([token]))
    else:
        mask = matcher.compute_bitmask()
        if matcher.is_error():
            raise RuntimeError(matcher.get_error())
        token = sample_from_model(mask)  # integration-owned
        totals["sampled_tokens"] += 1
        if not matcher.consume_token(token):
            raise RuntimeError(matcher.get_error())
        # Append the accepted sampled token in the engine here.

if matcher.is_error():
    raise RuntimeError(matcher.get_error())
```

`ff_observed_tokens` counts returned tokens across calls, so observing the same three-token
span twice adds six even if nothing has been consumed. It is not a count of unique tokens
or realized model work avoided. `ff_consumed_tokens` and `ff_consumed_bytes` above count
successful matcher consumption calls; coordinate them with the engine's actual append
boundary if reporting engine commits. `sampled_tokens` counts sampling operations even
if a sampled token is later rejected. Track draft attempts and accepted drafts separately
when using speculative decoding.

These example totals are request-level event counts, not the current sequence length.
They are not reduced by rollback. If an integration needs counts for the surviving
sequence, it must retain token provenance and undo the appropriate entries on rollback;
start new request totals on reset as appropriate. The observation API itself keeps no
cumulative ledger. An available span indicates a fast-forward opportunity, not a measured
latency improvement, accuracy improvement, or guaranteed number of model calls saved.
