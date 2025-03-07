from typing import List, Tuple, Mapping, Optional, Sequence, Union
from ._util import TokenId, StopReason
from ._tokenizer import TokenizerWrapper

class LLTokenizer:
    vocab_size: int
    eos_token: TokenId

    def __new__(
        cls,
        tokenizer: Union[str, TokenizerWrapper],
        n_vocab: Optional[int] = None,
        eos_token: Optional[TokenId] = None,
        slices: Optional[List[str]] = None,
    ) -> "LLTokenizer":
        """
        Create a new tokenizer.

        Args:
            tokenizer: str or TokenizerWrapper - if str, it is the name or path to the HF tokenizers tokenizer; otherwise it is a TokenizerWrapper
            n_vocab: int - override the size of the vocabulary
            slices: List[str] - configuration for slicer optimization; pass [] to disable,
                or None to use the default configuration
        """

    def greedy_tokenize(self, text: str) -> List[int]:
        """
        Tokenize the text using a greedy algorithm.
        This will not necessarily match BPE.
        """

    def tokenize_bytes(self, utf8bytes: bytes) -> List[int]:
        """
        Tokenize the text as bytes.
        This will use the underlying Python tokenizer to tokenize valid UTF8
        prefix of the text, and then fallback to greedy_tokenize() for the last
        few bytes.
        """

    def tokenize_str(self, text: str) -> List[int]:
        """
        Same as tokenize_bytes, but for strings.
        """

    def dbg_tokens(self, tokens: List[int]) -> str:
        """
        Return a debug string representation of the tokens.
        The result is double-quoted and tokens are separated by '‧'.
        """

    def test_trace_tokens(self, tokens: List[int]) -> str:
        """
        Return a debug string representation of the tokens
        for test traces.
        """

    def decode_str(self, tokens: List[int]) -> str:
        """
        Decode the tokens into a string.
        Invalid UTF-8 will be replaced with the Unicode replacement character.
        """

    def decode_bytes(self, tokens: List[int]) -> bytes:
        """
        Decode the tokens into a bytes object.
        """

    def is_special_token(self, token: int) -> bool:
        """
        Check if the token is a special token.
        """

class LLInterpreter:
    def __new__(
        cls,
        tokenizer: LLTokenizer,
        grammar: str,
        enable_backtrack: bool = True,
        enable_ff_tokens: bool = True,
        log_level: int = 1,
    ) -> "LLInterpreter":
        """
        Create a new interpreter.
        Args:
            tokenizer: LLTokenizer - the tokenizer to use
            grammar: str - either a Lark grammar or stringified JSON representation of LLGuidance grammar
            enable_backtrack: bool - whether to enable backtracking in the interpreter
            enable_ff_tokens: bool - whether to enable fast-forwarded tokens in the interpreter
            log_level: int - the verbosity level of the interpreter
                0 is silent, 1 is warnings, 2 is verbose
        """

    def deep_copy(self) -> "LLInterpreter":
        """
        Create a deep copy of the interpreter.
        """

    def is_accepting(self) -> bool:
        """
        Check if the last compute_mask() call resulted in overall accepting state
        of the parser.
        """

    def stop_reason(self) -> StopReason:
        """
        Get the reason why the parser stopped.
        Returns:
            "NotStopped" - Parser has not emitted stop() yet.
            "MaxTokensTotal" - max_tokens limit on the total number of tokens has been reached.
            "MaxTokensParser" - max_tokens limit on the number of tokens in the top-level parser has been reached.
            "ParserTooComplex" - Grammar is too complex (row item limit)
            "LexerTooComplex" - Lexer regex hit some limit
            "NoExtension" - Top-level parser indicates that no more bytes can be added.
            "NoExtensionBias" - Top-level parser indicates that no more bytes can be added, however it was recognized late.
            "EndOfSentence" - Top-level parser allowed EOS (as it was in an accepting state), and EOS was generated.
            "InternalError" - Something went wrong with creating a nested parser.
        """

    def process_prompt(self, prompt: List[TokenId]) -> List[TokenId]:
        """
        Perform any adjustments to the prompt before completion.
        Returns the adjusted prompt.
        """

    def start_without_prompt(self) -> None:
        """
        Start the parser without prompt processing.
        """

    def validate_tokens_raw(self, tokens: List[TokenId]) -> int:
        """
        Check if tokens are valid in the current state.
        Note that this doesn't currently check for max_tokens beyond the first token (hence 'raw').
        Return: how many of the tokens in the list can be committed
        """

    def compute_mask(self) -> Tuple[Optional[bytes], str]:
        """
        Perform next parsing step.
        Returns: optional token mask and a JSON string.
        """

    def compute_mask_into(self, trg: bytearray) -> str:
        """
        Perform next parsing step.
        Returns: a JSON string.
        """

    def unsafe_compute_mask_ptr(self, trg_pointer: int, trg_byte_size: int) -> str:
        """
        Perform next parsing step.
        Returns: a JSON string.
        """

    def commit_token(
        self, sampled_token: Optional[TokenId]
    ) -> Tuple[int, List[TokenId]]:
        """
        Perform any adjustments to the sampled token.
        Returns the number of tokens to remove from the prompt and the
        list of tokens to append.
        If compute_mask() returned None mask, this should be called immediately with None.
        If compute_mask() returned stop, you don't need to call this (but can).
        """

    def has_pending_stop(self) -> bool:
        """
        If true, next compute_mask() call will return stop
        """

class JsonCompiler:
    def __new__(
        cls,
        separators: Optional[Tuple[str, str]] = None,
        whitespace_flexible: bool = False,
        coerce_one_of: bool = False,
    ) -> "JsonCompiler":
        """
        Create a new JSON compiler.
        Args:
            compact: bool - whether to use compact JSON representation
        """

    def compile(
        self,
        schema: str,
    ) -> str:
        """
        Compile the JSON representation of the AG2 grammar/constraint.
        """

class LarkCompiler:
    def __new__(
        cls,
    ) -> "LarkCompiler":
        """
        Create a new Lark compiler.
        """

    def compile(
        self,
        lark: str,
    ) -> str:
        """
        Compile the JSON representation of the AG2 grammar/constraint.
        """

class RegexCompiler:
    def __new__(
        cls,
    ) -> "RegexCompiler":
        """
        Create a new Regex compiler.
        """

    def compile(
        self,
        regex: str,
        stop_regex: Optional[str] = None,
    ) -> str:
        """
        Compile the JSON representation of the AG2 grammar/constraint.
        """

class LLExecutor:
    def __new__(
        cls,
        num_threads: Optional[int] = None,
    ) -> "LLExecutor":
        """
        Create a new executor.
        Args:
            num_threads: int - number of threads to use for parallel execution,
                or None to use the default number of threads (80% of the available CPUs up to 32)
        """

    def unsafe_compute_mask_ptr(
        self,
        interpreters: List[LLInterpreter],
        trg_pointer: int,
        one_mask_byte_size: int,
    ) -> str:
        """
        Perform next parsing step.
        Returns: a JSON string.
        """
