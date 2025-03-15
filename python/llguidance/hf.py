from typing import List, Optional
from ._lib import LLTokenizer

import transformers # type: ignore[import-untyped]


def from_tokenizer(
    hf_tokenizer: transformers.PreTrainedTokenizerBase,
    n_vocab: Optional[int] = None,
    eos_token: Optional[int] = None,
    slices: Optional[List[str]] = None,
) -> LLTokenizer:
    """
    Create a new tokenizer from a Hugging Face tokenizer.
    This is an expensive operation (~1s), so the result should be cached.
    It also currently creates a non-canonical tokenizer, which means it cannot
    produce fast-forward tokens (though it can produce fast-forward bytes).

    Args:
        hf_tokenizer: transformers.PreTrainedTokenizerBase - the tokenizer to wrap
        n_vocab: int - override the size of the vocabulary
        eos_token: int - override the EOS token
        slices: List[str] - configuration for slicer optimization; pass [] to disable,
            or None to use the default configuration
    """

    if isinstance(hf_tokenizer, transformers.PreTrainedTokenizerFast):
        # this is not ideal...
        s = hf_tokenizer.backend_tokenizer.to_str()
        if n_vocab is None:
            n_vocab = hf_tokenizer.vocab_size
        if eos_token is None:
            eos_token = hf_tokenizer.eos_token_id
        return LLTokenizer(s,
                           n_vocab=n_vocab,
                           eos_token=eos_token,
                           slices=slices)
    else:
        raise ValueError("Only fast tokenizers are supported")
