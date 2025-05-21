from typing import List, Optional

from ._lib import LLTokenizer

import llama_cpp
import ctypes

def lltokenizer_from_vocab(
    vocab: llama_cpp.llama_vocab_p,
    n_vocab: Optional[int] = None,
    eos_token: Optional[int] = None,
    slices: Optional[List[str]] = None,
) -> LLTokenizer:
    """
    Create a new tokenizer from a llama.cpp vocab object.
    This is an expensive operation (~1s), so the result should be cached.

    Args:
        vocab: llama_cpp.llama_vocab_p - the vocab object to use
        n_vocab: int - override the size of the vocabulary
        eos_token: int - override the EOS token
        slices: List[str] - configuration for slicer optimization; pass [] to disable,
            or None to use the default configuration
    """

    ntok = llama_cpp.llama_vocab_n_tokens(vocab)
    if eos_token is None:
        eos_token = llama_cpp.llama_vocab_eos(vocab)
    buffer_len = 16 * 1024
    buffer = ctypes.create_string_buffer(buffer_len + 1)
    tokens: List[bytes] = []

    for token in range(ntok):
        n = llama_cpp.llama_token_to_piece(
            vocab,
            token,
            buffer,
            buffer_len,
            0,
            True
        )
        if n < 0:
            raise ValueError(f"Error writing token {token} to buffer of size {buffer_len}. Error: {n}")
        assert n <= buffer_len
        tok = bytes(buffer[:n]) # type: ignore
        attr = llama_cpp.llama_token_get_attr(vocab, token)
        if attr & llama_cpp.LLAMA_TOKEN_ATTR_CONTROL:
            tok = b"\xFF" + tok
        tokens.append(tok)

    if n_vocab is not None:
        while len(tokens) < n_vocab:
            tokens.append(b"")

    fptr = ctypes.cast(llama_cpp.llama_cpp._lib.llama_tokenize, ctypes.c_void_p).value
    return LLTokenizer.from_llamacpp( # type: ignore
        tokens=tokens,
        vocab_ptr=vocab,
        tokenize_fptr=fptr,
        eos_token=eos_token,
        slices=slices
    )
