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
        # If the tokenizer used bytes, then b"\xff" would be better (since it's invalid UTF-8)
        # For now, we'll settle for "\x02" as assume it doesn't start any other token
        self._prefix_string = "\x02"
        self._prefix_tokens = self._encode_string(self._prefix_string)

    def _encode_string(self, s: str) -> List[TokenId]:
        r: List[TokenId]
        if self._accepts_bytes:
            r = self._gtokenizer(s.encode("utf-8"))
        else:
            r = self._gtokenizer(s)
        return r

    # required by LLTokenizer
    def __call__(self, s: str) -> List[TokenId]:
        tokens = self._encode_string(self._prefix_string + s)
        assert tokens[: len(self._prefix_tokens)] == self._prefix_tokens
        return tokens[len(self._prefix_tokens) :]
