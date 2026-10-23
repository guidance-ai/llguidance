from typing import List, Tuple, Dict, Any, Optional, Callable

import torch
import numpy as np
import pytest
import json
import time

from llguidance.torch import (
    apply_token_bitmask_inplace,
    get_bitmask_shape,
    fill_next_token_bitmask,
    allocate_token_bitmask,
    fill_next_token_bitmask_par,
)
from llguidance import LLMatcher, LLTokenizer, LLExecutor

import llguidance.hf

from transformers import AutoTokenizer


def _build_tokenizer() -> LLTokenizer:
    hf_tok = AutoTokenizer.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct")
    return llguidance.hf.from_tokenizer(hf_tok)


_tokenizer: Optional[LLTokenizer] = None


def tokenizer() -> LLTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _build_tokenizer()
    return _tokenizer


def lark_matcher(grm: str) -> LLMatcher:
    gstr = json.dumps({"grammars": [{"lark_grammar": grm}]})
    interp = LLMatcher(tokenizer(), gstr, log_level=1)
    return interp


def test_basic_tokenizer() -> None:
    llt = tokenizer()
    for s in [
            "Hello world!", "Hello world! こんにちは世界！", "wave 👋", "heart 👋💖",
            "1`a`b`c`d`e`f`g`h`i"
    ]:
        toks = llt.tokenize_str(s)
        print(llt.dbg_tokens(toks))
        assert llt.decode_str(toks) == s
    toks = llt.tokenize_bytes(b"\x8b")
    print(llt.dbg_tokens(toks))
    print(toks)
    assert len(toks) == 1
    assert llt.decode_bytes(toks) == b"\x8b"


def test_grammar() -> None:
    t = tokenizer()
    mask = allocate_token_bitmask(2, t.vocab_size)
    interp = lark_matcher(r"start: /[A-Z ]*/")
    fill_next_token_bitmask(interp, mask)
    allowed = []
    for idx, v in enumerate(mask[0, :].tolist()):
        for bit_idx in range(32):
            tok_idx = idx * 32 + bit_idx
            if v & (1 << bit_idx):
                if t.is_special_token(tok_idx):
                    continue
                s = t.decode_str([tok_idx])
                for c in s:
                    assert c.isupper() or c.isspace()
                allowed.append(tok_idx)
    assert len(allowed) > 100
    interp.consume_token(allowed[3])
    fill_next_token_bitmask(interp, mask, 1)
    assert torch.isclose(mask[1, :], mask[0, :]).all()


def test_par_grammar() -> None:
    n_gram = 50
    t = tokenizer()
    grammars = [(lark_matcher(r"start: /[a-zA-Z ]*/"), idx)
                for idx in range(n_gram)]
    mask = allocate_token_bitmask(n_gram, t.vocab_size)
    mask2 = allocate_token_bitmask(n_gram, t.vocab_size)
    exec = LLExecutor()
    t0 = time.monotonic()
    fill_next_token_bitmask_par(exec, grammars, mask)
    par_time = int((time.monotonic() - t0) * 1_000_000)
    for i in range(n_gram):
        assert torch.isclose(mask[i, :], mask[0, :]).all()
    t0 = time.monotonic()
    for g, idx in grammars:
        fill_next_token_bitmask(g, mask2, idx)
    seq_time = int((time.monotonic() - t0) * 1_000_000)
    assert torch.isclose(mask, mask2).all()
    print(f"Parallel: {par_time} us, Sequential: {seq_time} us")


def test_fill_bitmask_row_addressing() -> None:
    t = tokenizer()
    interp = lark_matcher(r"start: /[A-Z ]*/")
    ref = allocate_token_bitmask(1, t.vocab_size)
    fill_next_token_bitmask(interp, ref)
    assert ref.ne(0).any() and ref.ne(-1).any()

    base = torch.zeros((9, ref.shape[1]), dtype=torch.int32)
    # plain rows, negative index, a view with a storage offset, a row-strided view
    for mask, idx in [(base, 0), (base, 8), (base, -1), (base[3:], 2),
                      (base[1::2], 1)]:
        base.zero_()
        fill_next_token_bitmask(interp, mask, idx)
        assert torch.equal(mask[idx], ref[0])
        assert int(base.ne(0).any(dim=1).sum()) == 1  # no other row written

    with pytest.raises(IndexError):
        fill_next_token_bitmask(interp, base, 9)
    with pytest.raises(IndexError):
        fill_next_token_bitmask(interp, base, -10)
    wide = torch.zeros((2, ref.shape[1] * 2), dtype=torch.int32)
    with pytest.raises(AssertionError):
        fill_next_token_bitmask(interp, wide[:, ::2], 0)


@pytest.mark.parametrize("recent_tokens", [[], [1000, 3003]])
def test_tokenize_partial_basic(recent_tokens: List[int]) -> None:
    """Test tokenize_partial with a simple sentence."""
    ll_tok = tokenizer()
    assert ll_tok.is_canonical
    new_tokens, leftover = ll_tok.tokenize_partial(b" How are you",
                                                   recent_tokens=recent_tokens)
    assert isinstance(new_tokens, list)
    assert isinstance(leftover, bytes)
    assert len(new_tokens) >= 2
    assert ll_tok.decode_bytes(new_tokens) + leftover == b" How are you"
    for suff in ["", "r", "!", " "]:
        tok2 = ll_tok.tokenize_str(" How are you" + suff)
        assert tok2[0:len(new_tokens)] == new_tokens


def test_tokenize_partial_docs() -> None:
    ll = tokenizer()
    new_tok, leftover = ll.tokenize_partial(b"order")
    assert len(new_tok) == 0
    assert leftover == b"order"

    recent = ll.tokenize_bytes(b'{"')
    new_tok, leftover = ll.tokenize_partial(b'name_of_the_person"',
                                            recent_tokens=recent)
    print(ll.dbg_tokens(new_tok))
    assert leftover == b'"'
    assert ll.decode_str(new_tok) == "name_of_the_person"


def test_incomplete_tokenizer() -> None:
    hf_tok = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct")
    ll_tok = llguidance.hf.from_tokenizer(hf_tok)

    # unknown bytes are to be skipped
    # see https://github.com/guidance-ai/llguidance/issues/138
    assert len(ll_tok.tokenize_bytes(b"\xff")) == 0
    assert len(ll_tok.tokenize_bytes(b"\xff\x80")) == 1
    # make sure the special markers still work
    assert ll_tok.tokenize_partial(b"\xff[1234]") == ([1234], b"")

    tt = ll_tok.tokenize_str("\U00042000")
    tt2 = ll_tok.tokenize_bytes("\U00042000".encode()[1:])
    assert tt == tt2

    matcher = llguidance.LLMatcher(ll_tok, "start: /a.*/")
    matcher.compute_bitmask()
    assert matcher.get_error() == ""


if __name__ == "__main__":
    test_incomplete_tokenizer()
