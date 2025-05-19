from typing import List, Optional, Sequence, Any
from ._util import TokenId


class TokenizerWrapper:
    eos_token_id: TokenId
    bos_token_id: Optional[TokenId]
    tokens: Sequence[bytes]
    special_token_ids: Sequence[int]

    def __init__(self, gtokenizer: Any) -> None:
        # these are required by LLTokenizer:
        self.eos_token_id = gtokenizer.eos_token_id
        self.bos_token_id = gtokenizer.bos_token_id
        self.tokens = gtokenizer.tokens
        self.special_token_ids = getattr(gtokenizer, "special_token_ids", [])
        self.is_tokenizer_wrapper = True

        # more private stuff
        self._gtokenizer = gtokenizer
        self._accepts_bytes = True
        try:
            gtokenizer(b"test")
        except:
            self._accepts_bytes = False

    def _encode_string(self, s: str) -> List[TokenId]:
        r: List[TokenId]
        if self._accepts_bytes:
            r = self._gtokenizer(s.encode("utf-8"))
        else:
            r = self._gtokenizer(s)
        return r

    # required by LLTokenizer
    __call__ = _encode_string
