from typing import List, Optional, Union, TYPE_CHECKING

from ._lib import LLTokenizer

if TYPE_CHECKING:
    import tiktoken


def lltokenizer_from_encoding(
    encoding: 'tiktoken.Encoding',
    *,
    n_vocab: Optional[int] = None,
    eos_token: Optional[Union[int, List[int]]] = None,
    slices: Optional[List[str]] = None,
) -> LLTokenizer:
    """
    Create a new tokenizer from a tiktoken Encoding object.
    This is an expensive operation (~1s), so the result should be cached.

    Args:
        encoding: tiktoken.Encoding - the encoding object to use
        n_vocab: int - override the size of the vocabulary
        eos_token: int or list of ints - override the EOS token(s)
        slices: List[str] - configuration for slicer optimization; pass [] to disable,
            or None to use the default configuration
    """

    return LLTokenizer.from_tiktoken(  # type: ignore
        encoder=encoding._mergeable_ranks,
        special_tokens=encoding._special_tokens,
        pattern=encoding._pat_str,
        eos_token=encoding.eot_token if eos_token is None else eos_token,
        n_vocab=n_vocab,
        slices=slices)
