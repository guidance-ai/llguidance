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
from llguidance import LLInterpreter, LLTokenizer, LLExecutor

import llguidance.hf
import transformers # type: ignore[import-untyped]


def _build_tokenizer() -> LLTokenizer:
    hf_tok = transformers.AutoTokenizer.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct"
    )
    return llguidance.hf.from_tokenizer(hf_tok)


_tokenizer: Optional[LLTokenizer] = None


def tokenizer() -> LLTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _build_tokenizer()
    return _tokenizer


def lark_interp(grm: str) -> LLInterpreter:
    gstr = json.dumps({"grammars": [{"lark_grammar": grm}]})
    interp = LLInterpreter(
        tokenizer(), gstr, enable_backtrack=False, enable_ff_tokens=False, log_level=1
    )
    interp.start_without_prompt()
    return interp


def test_grammar() -> None:
    t = tokenizer()
    mask = allocate_token_bitmask(2, t.vocab_size)
    interp = lark_interp(r"start: /[A-Z ]*/")
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
    bt, toks = interp.commit_token(allowed[3])
    assert bt == 0
    assert toks == [allowed[3]]
    fill_next_token_bitmask(interp, mask, 1)
    assert torch.isclose(mask[1, :], mask[0, :]).all()


def test_par_grammar() -> None:
    n_gram = 50
    t = tokenizer()
    grammars = [lark_interp(r"start: /[a-zA-Z ]*/") for _ in range(n_gram)]
    mask = allocate_token_bitmask(n_gram, t.vocab_size)
    mask2 = allocate_token_bitmask(n_gram, t.vocab_size)
    exec = LLExecutor()
    t0 = time.monotonic()
    fill_next_token_bitmask_par(exec, grammars, mask)
    par_time = int((time.monotonic() - t0) * 1_000_000)
    for i in range(n_gram):
        assert torch.isclose(mask[i, :], mask[0, :]).all()
    t0 = time.monotonic()
    for i in range(n_gram):
        fill_next_token_bitmask(grammars[i], mask2, i)
    seq_time = int((time.monotonic() - t0) * 1_000_000)
    assert torch.isclose(mask, mask2).all()
    print(f"Parallel: {par_time} us, Sequential: {seq_time} us")


def asserts(msg: str, fn: Callable[..., Any], *args: Any) -> None:
    try:
        fn(*args)
        raise AssertionError(f"Expected {msg}")
    except Exception as e:
        if msg in str(e):
            return
        raise AssertionError(f"Expected {msg}, got {e}")


def test_par_errors() -> None:
    t = tokenizer()
    exec = LLExecutor()
    g0 = lark_interp(r"start: /[a-zA-Z ]*/")
    g1 = lark_interp(r"start: /[0-9 ]*/")
    mask = allocate_token_bitmask(3, t.vocab_size)
    asserts("count mismatch", fill_next_token_bitmask_par, exec, [g0, g1], mask)
    asserts("Duplicate interpreter", fill_next_token_bitmask_par, exec, [g0, g1, g1], mask)
    # should be OK
    fill_next_token_bitmask_par(exec, [g0, g1], mask[0:2, :])


if __name__ == "__main__":
    test_par_grammar()
