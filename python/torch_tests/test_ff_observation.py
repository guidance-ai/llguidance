from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

if TYPE_CHECKING:
    from llguidance._lib import FastForwardObservation


def assert_span(
    observation: FastForwardObservation, tokenizer: LLTokenizer, expected: list[int], status: str = "ready"
) -> None:
    assert observation == {
        "status": status,
        "tokens": expected,
        "token_bytes": tokenizer.decode_bytes(expected),
        "token_count": len(expected),
        "byte_count": len(tokenizer.decode_bytes(expected)),
    }


@pytest.mark.parametrize("literal", ["abcd", "é🙂"])
def test_observation_does_not_consume_or_change_masks(literal: str) -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, f'start: "{literal}" /[xy]/')
    control = matcher.deep_copy()
    expected = tokenizer.tokenize_str(literal)

    # Establish the existing mechanism independently of the new observation API.
    assert matcher.compute_ff_tokens() == expected
    observation = matcher.observe_ff_tokens()
    assert_span(observation, tokenizer, expected)
    assert matcher.observe_ff_tokens() == observation
    assert matcher.compute_ff_tokens() == expected

    for token in tokenizer.tokenize_str(literal + "x"):
        assert matcher.compute_logit_bias() == control.compute_logit_bias()
        assert matcher.consume_token(token)
        assert control.consume_token(token)
    assert matcher.is_accepting() and control.is_accepting()
    assert matcher.stop_reason() == control.stop_reason()
    assert_span(matcher.observe_ff_tokens(), tokenizer, [], "stopped")


def test_observation_counts_bytes_without_utf8_decoding() -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, 'start: "é🙂" /[xy]/')
    assert matcher.consume_token(0xC3)
    observation = matcher.observe_ff_tokens()
    expected_bytes = "é🙂".encode()[1:]
    assert_span(observation, tokenizer, list(expected_bytes))
    assert observation["byte_count"] == 5
    with pytest.raises(UnicodeDecodeError):
        observation["token_bytes"].decode()
    assert matcher.consume_tokens(observation["tokens"])
    assert_span(matcher.observe_ff_tokens(), tokenizer, [])


class GreedyTokenizer:
    eos_token_id = 256
    bos_token_id = None
    tokens = [bytes([byte]) for byte in range(256)] + [b"", b"ab", b"cd", b"cdx", b"cdy"]

    def __call__(self, data: str | bytes) -> list[int]:
        if isinstance(data, str):
            data = data.encode()
        result = []
        while data:
            token = max(
                (index for index, value in enumerate(self.tokens) if value and data.startswith(value)),
                key=lambda index: len(self.tokens[index]),
            )
            result.append(token)
            data = data[len(self.tokens[token]) :]
        return result


def test_observation_excludes_committed_prefix_and_healing_suffix() -> None:
    tokenizer = LLTokenizer(TokenizerWrapper(GreedyTokenizer()))
    matcher = LLMatcher(tokenizer, 'start: "!abcd" /[xy]/')
    assert matcher.consume_token(ord("!"))
    control = matcher.deep_copy()
    observation = matcher.observe_ff_tokens()
    assert_span(observation, tokenizer, [257])
    assert observation["token_bytes"] == b"ab"
    assert matcher.compute_ff_bytes() == b"abcd"
    assert matcher.observe_ff_tokens() == observation
    assert matcher.compute_logit_bias() == control.compute_logit_bias()

    assert matcher.consume_tokens(observation["tokens"])
    assert_span(matcher.observe_ff_tokens(), tokenizer, [])
    assert matcher.compute_ff_bytes() == b"cd"
    # Empty token telemetry does not mean there are no forced bytes.
    assert matcher.consume_token(259)
    assert matcher.is_accepting()


def test_observation_rollback_reset_and_independent_clone() -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, 'start: "abc" /[xy]/')
    initial = matcher.observe_ff_tokens()
    assert matcher.consume_tokens([ord("a"), ord("b")])
    assert_span(matcher.observe_ff_tokens(), tokenizer, [ord("c")])
    clone = matcher.deep_copy()
    assert matcher.consume_tokens([ord("c"), ord("x")])
    assert_span(matcher.observe_ff_tokens(), tokenizer, [], "stopped")
    assert_span(clone.observe_ff_tokens(), tokenizer, [ord("c")])
    assert matcher.rollback(2)
    assert matcher.observe_ff_tokens() == clone.observe_ff_tokens()
    assert matcher.reset()
    assert matcher.observe_ff_tokens() == initial
    assert_span(clone.observe_ff_tokens(), tokenizer, [ord("c")])


def test_observation_distinguishes_empty_and_stopped() -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, "start: /[xy]/")
    assert_span(matcher.observe_ff_tokens(), tokenizer, [])
    assert not matcher.is_stopped()
    assert matcher.consume_token(ord("x"))
    assert_span(matcher.observe_ff_tokens(), tokenizer, [], "stopped")
    assert matcher.reset()
    assert_span(matcher.observe_ff_tokens(), tokenizer, [])


@pytest.mark.parametrize("invalid_grammar", [False, True])
def test_observation_reports_terminal_errors(invalid_grammar: bool) -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, "start: missing" if invalid_grammar else 'start: "abc"')
    if not invalid_grammar:
        assert not matcher.consume_token(999999)
    error = matcher.get_error()
    assert error
    assert_span(matcher.observe_ff_tokens(), tokenizer, [], "error")
    assert matcher.compute_ff_tokens() == []
    assert matcher.get_error() == error
    assert not matcher.reset()
    assert not matcher.rollback(0)
    assert_span(matcher.observe_ff_tokens(), tokenizer, [], "error")


def test_observation_after_validation_and_partial_consumption() -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, 'start: "abc" /[xy]/')
    initial = matcher.observe_ff_tokens()
    assert matcher.validate_tokens(list(b"abcy")) == 4
    assert matcher.observe_ff_tokens() == initial
    assert matcher.try_consume_tokens(list(b"ab!")) == 2
    assert_span(matcher.observe_ff_tokens(), tokenizer, [ord("c")])
    assert not matcher.is_error()


def test_observed_consumed_and_sampled_counts_have_separate_boundaries() -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, 'start: "abc" /[xy]/')
    first = matcher.observe_ff_tokens()
    second = matcher.observe_ff_tokens()
    observed_tokens = first["token_count"] + second["token_count"]
    assert observed_tokens == 6  # two observations of the same three-token span
    consumed_ff_tokens = matcher.try_consume_tokens(second["tokens"])
    assert consumed_ff_tokens == 3
    assert_span(matcher.observe_ff_tokens(), tokenizer, [])
    mask = matcher.compute_logit_bias()
    sampled_token = ord("y")
    assert mask[sampled_token]
    sampled_tokens = 1  # only the integration knows that a sampling operation occurred
    assert matcher.consume_token(sampled_token)
    assert (observed_tokens, consumed_ff_tokens, sampled_tokens) == (6, 3, 1)


def test_observation_special_tokens_use_tokenizer_decoded_bytes() -> None:
    tokenizer = LLTokenizer("byte")
    matcher = LLMatcher(tokenizer, "start: <[256]> /[xy]/")
    observation = matcher.observe_ff_tokens()
    assert_span(observation, tokenizer, [256])
    assert observation["token_bytes"] == b"<|tool|>"
    assert observation["byte_count"] == 8
