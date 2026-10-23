#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "llguidance.h"
#include "test_helpers.h"

namespace {

struct TokenizerDeleter {
  void operator()(LlgTokenizer *tok) const {
    if (tok != nullptr) {
      llg_free_tokenizer(tok);
    }
  }
};

struct MatcherDeleter {
  void operator()(LlgMatcher *matcher) const {
    if (matcher != nullptr) {
      llg_free_matcher(matcher);
    }
  }
};

struct CancellationHandleDeleter {
  void operator()(LlgCancellationHandle *handle) const {
    if (handle != nullptr) {
      llg_free_cancellation_handle(handle);
    }
  }
};

struct CancelOnTokenize {
  LlgCancellationHandle *handle = nullptr;
  std::atomic<bool> cancelled{false};
};

size_t cancel_on_tokenize(const void *user_data, const uint8_t *bytes,
                          size_t bytes_len, uint32_t *output_tokens,
                          size_t output_tokens_len) {
  auto *state = const_cast<CancelOnTokenize *>(
      static_cast<const CancelOnTokenize *>(user_data));
  if (state->handle != nullptr && !state->cancelled.exchange(true)) {
    llg_cancel(state->handle);
  }
  return byte_tokenize_callback(nullptr, bytes, bytes_len, output_tokens,
                                 output_tokens_len);
}

using TokenizerPtr = std::unique_ptr<LlgTokenizer, TokenizerDeleter>;
using MatcherPtr = std::unique_ptr<LlgMatcher, MatcherDeleter>;
using CancellationHandlePtr =
    std::unique_ptr<LlgCancellationHandle, CancellationHandleDeleter>;

struct MatcherContext {
  TokenizerPtr tok;
  LlgConstraintInit init;

  MatcherContext() : tok(create_byte_tokenizer()) {
    llg_constraint_init_set_defaults(&init, tok.get());
    init.log_stderr_level = 0;
  }

  MatcherPtr make_matcher(const char *constraint_type, const char *data) const {
    return MatcherPtr(llg_new_matcher(&init, constraint_type, data));
  }
};

bool mask_has_token(const uint32_t *mask, uint32_t token) {
  return mask != nullptr &&
         (mask[token / 32] & (uint32_t{1} << (token % 32))) != 0;
}

void check_matcher_has_no_error(LlgMatcher *matcher) {
  BOOST_REQUIRE(matcher != nullptr);
  BOOST_CHECK(!llg_matcher_is_error(matcher));
  BOOST_CHECK(llg_matcher_get_error(matcher) == nullptr);
}

} // namespace

BOOST_AUTO_TEST_SUITE(matcher)

BOOST_AUTO_TEST_CASE(new_matcher_regex) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
}

BOOST_AUTO_TEST_CASE(new_matcher_json_schema) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("json_schema", R"({"type":"number"})");

  check_matcher_has_no_error(matcher.get());
}

BOOST_AUTO_TEST_CASE(new_matcher_invalid_grammar) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("lark", "start: (");

  BOOST_REQUIRE(matcher != nullptr);
  BOOST_CHECK(llg_matcher_is_error(matcher.get()));
  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK(std::strlen(error) > 0);
}

BOOST_AUTO_TEST_CASE(compute_mask_basic) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(matcher.get()), 0);

  const uint32_t *mask = llg_matcher_get_mask(matcher.get());
  BOOST_REQUIRE(mask != nullptr);
  BOOST_CHECK(mask_has_token(mask, 97));
  BOOST_CHECK(mask_has_token(mask, 98));
  BOOST_CHECK(mask_has_token(mask, 99));
}

BOOST_AUTO_TEST_CASE(compute_mask_into) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");

  check_matcher_has_no_error(matcher.get());

  const size_t mask_byte_size = llg_matcher_get_mask_byte_size(matcher.get());
  std::vector<uint32_t> mask(mask_byte_size / sizeof(uint32_t));

  BOOST_REQUIRE_EQUAL(
      llg_matcher_compute_mask_into(matcher.get(), mask.data(), mask_byte_size),
      0);
  BOOST_CHECK(mask_has_token(mask.data(), 97));
  BOOST_CHECK(mask_has_token(mask.data(), 98));
  BOOST_CHECK(mask_has_token(mask.data(), 99));
}

BOOST_AUTO_TEST_CASE(cancellation_handle_cancels_matcher) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");
  BOOST_CHECK(!llg_matcher_is_cancelled(matcher.get()));
  CancellationHandlePtr handle(
      llg_matcher_get_cancellation_handle(matcher.get()));

  BOOST_REQUIRE(handle != nullptr);
  BOOST_CHECK(!llg_matcher_is_cancelled(matcher.get()));
  llg_cancel(handle.get());
  BOOST_CHECK(llg_matcher_is_cancelled(matcher.get()));
  BOOST_CHECK_EQUAL(llg_matcher_compute_mask(matcher.get()), -1);
  BOOST_CHECK(llg_matcher_get_mask(matcher.get()) == nullptr);
  BOOST_CHECK(llg_matcher_is_error(matcher.get()));

  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK_EQUAL(std::string(error), "operation cancelled");
  BOOST_CHECK(!llg_matcher_is_accepting(matcher.get()));
}

BOOST_AUTO_TEST_CASE(cancellation_clears_saved_mask_and_preserves_output) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");
  const size_t mask_byte_size = llg_matcher_get_mask_byte_size(matcher.get());
  std::vector<uint32_t> output(mask_byte_size / sizeof(uint32_t), 0xA5A5A5A5);

  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(matcher.get()), 0);
  BOOST_REQUIRE(llg_matcher_get_mask(matcher.get()) != nullptr);

  CancellationHandlePtr handle(
      llg_matcher_get_cancellation_handle(matcher.get()));
  BOOST_REQUIRE(handle != nullptr);
  llg_cancel(handle.get());

  BOOST_CHECK(llg_matcher_get_mask(matcher.get()) == nullptr);

  BOOST_CHECK_EQUAL(
      llg_matcher_compute_mask_into(matcher.get(), output.data(), mask_byte_size),
      -1);
  BOOST_CHECK(llg_matcher_get_mask(matcher.get()) == nullptr);
  for (uint32_t value : output) {
    BOOST_CHECK_EQUAL(value, 0xA5A5A5A5);
  }
}

BOOST_AUTO_TEST_CASE(recoverable_mask_buffer_error_does_not_mask_cancellation) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");
  const size_t mask_byte_size = llg_matcher_get_mask_byte_size(matcher.get());
  std::vector<uint32_t> output(mask_byte_size / sizeof(uint32_t));

  BOOST_CHECK_EQUAL(
      llg_matcher_compute_mask_into(matcher.get(), output.data(),
                                    mask_byte_size - sizeof(uint32_t)),
      -1);
  BOOST_CHECK(!llg_matcher_is_error(matcher.get()));

  CancellationHandlePtr handle(
      llg_matcher_get_cancellation_handle(matcher.get()));
  BOOST_REQUIRE(handle != nullptr);
  llg_cancel(handle.get());
  BOOST_CHECK_EQUAL(llg_matcher_compute_mask(matcher.get()), -1);

  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK_EQUAL(std::string(error), "operation cancelled");
}

BOOST_AUTO_TEST_CASE(cancellation_handle_clone_has_independent_lifetime) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");
  CancellationHandlePtr first(
      llg_matcher_get_cancellation_handle(matcher.get()));
  BOOST_REQUIRE(first != nullptr);
  CancellationHandlePtr second(llg_clone_cancellation_handle(first.get()));
  BOOST_REQUIRE(second != nullptr);
  BOOST_CHECK(first.get() != second.get());

  first.reset();
  llg_cancel(second.get());
  BOOST_CHECK(llg_matcher_is_cancelled(matcher.get()));
  uint32_t forced_token = 0xA5A5A5A5;
  BOOST_CHECK_EQUAL(
      llg_matcher_compute_ff_tokens(matcher.get(), &forced_token, 1), -1);
  BOOST_CHECK_EQUAL(forced_token, 0xA5A5A5A5);
  BOOST_CHECK(llg_matcher_is_error(matcher.get()));
  BOOST_CHECK(llg_matcher_is_cancelled(matcher.get()));
  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK_EQUAL(std::string(error), "operation cancelled");

  matcher.reset();
  llg_cancel(second.get());
  second.reset();
}

BOOST_AUTO_TEST_CASE(cancellation_during_ff_tokenization_does_not_write_output) {
  CancelOnTokenize state;
  TokenizerPtr tok(create_byte_tokenizer(cancel_on_tokenize, &state));
  LlgConstraintInit init;
  llg_constraint_init_set_defaults(&init, tok.get());
  init.log_stderr_level = 0;
  MatcherPtr matcher(llg_new_matcher(&init, "regex", "hello"));
  check_matcher_has_no_error(matcher.get());
  CancellationHandlePtr handle(
      llg_matcher_get_cancellation_handle(matcher.get()));
  BOOST_REQUIRE(handle != nullptr);
  state.handle = handle.get();

  uint32_t output = 0xA5A5A5A5;
  BOOST_CHECK_EQUAL(llg_matcher_compute_ff_tokens(matcher.get(), &output, 1), -1);
  BOOST_CHECK_EQUAL(output, 0xA5A5A5A5);
  BOOST_CHECK(state.cancelled.load());
  BOOST_CHECK(llg_matcher_is_error(matcher.get()));
  BOOST_CHECK(llg_matcher_is_cancelled(matcher.get()));
  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK_EQUAL(std::string(error), "operation cancelled");
}

BOOST_AUTO_TEST_CASE(cancellation_handle_is_safe_with_worker_thread) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");
  CancellationHandlePtr handle(
      llg_matcher_get_cancellation_handle(matcher.get()));
  BOOST_REQUIRE(handle != nullptr);

  std::atomic<bool> run_worker{false};
  std::atomic<int32_t> result{0};
  std::thread worker([&] {
    while (!run_worker.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    result.store(llg_matcher_compute_mask(matcher.get()),
                 std::memory_order_release);
  });

  llg_cancel(handle.get());
  run_worker.store(true, std::memory_order_release);
  worker.join();

  // This synchronization tests ABI ownership. Rust checkpoints test progress
  // inside mask computation.
  BOOST_CHECK_EQUAL(result.load(std::memory_order_acquire), -1);
  BOOST_CHECK(llg_matcher_is_cancelled(matcher.get()));
}

BOOST_AUTO_TEST_CASE(cancellation_does_not_affect_deep_matcher_clone) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[abc]+");
  MatcherPtr clone(llg_clone_matcher(matcher.get()));
  CancellationHandlePtr handle(
      llg_matcher_get_cancellation_handle(matcher.get()));

  BOOST_REQUIRE(clone != nullptr);
  BOOST_REQUIRE(handle != nullptr);
  llg_cancel(handle.get());

  BOOST_CHECK_EQUAL(llg_matcher_compute_mask(matcher.get()), -1);
  BOOST_CHECK_EQUAL(llg_matcher_compute_mask(clone.get()), 0);
}

BOOST_AUTO_TEST_CASE(consume_token_single) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0);
}

BOOST_AUTO_TEST_CASE(consume_tokens_multiple) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");
  const uint32_t tokens[] = {104, 101, 108, 108, 111};

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(
      llg_matcher_consume_tokens(matcher.get(), tokens, std::size(tokens)), 0);
}

BOOST_AUTO_TEST_CASE(is_accepting_after_valid_input) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0);
  BOOST_CHECK(llg_matcher_is_accepting(matcher.get()));
}

BOOST_AUTO_TEST_CASE(is_stopped_when_forced_eos) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "a");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0);
  BOOST_CHECK(llg_matcher_is_stopped(matcher.get()));
}

BOOST_AUTO_TEST_CASE(rollback) {
  MatcherContext ctx;
  // "ab|ac" distinguishes states: after 'a', both 'b' and 'c' are valid.
  // Consuming 'a','b' commits to the "ab" branch; rolling back 1 restores
  // the post-'a' state where 'c' is also reachable.
  auto matcher = ctx.make_matcher("regex", "ab|ac");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0); // 'a'
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 98), 0); // 'b'
  BOOST_REQUIRE_EQUAL(llg_matcher_rollback(matcher.get(), 1), 0);

  // After rollback we should be back in the post-'a' state
  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(matcher.get()), 0);
  const uint32_t *mask = llg_matcher_get_mask(matcher.get());
  BOOST_REQUIRE(mask != nullptr);
  BOOST_CHECK(mask_has_token(mask, 99));  // 'c' should be allowed
  BOOST_CHECK(mask_has_token(mask, 98));  // 'b' should be allowed
  BOOST_CHECK(!mask_has_token(mask, 48)); // '0' should be rejected
}

BOOST_AUTO_TEST_CASE(reset) {
  MatcherContext ctx;
  // "ab" requires exactly 'a' then 'b'. After consuming 'a', only 'b' is valid.
  // After reset, only 'a' (the start) should be valid, not 'b'.
  auto matcher = ctx.make_matcher("regex", "ab");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0); // 'a'
  BOOST_REQUIRE_EQUAL(llg_matcher_reset(matcher.get()), 0);
  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(matcher.get()), 0);

  const uint32_t *mask = llg_matcher_get_mask(matcher.get());
  BOOST_REQUIRE(mask != nullptr);
  BOOST_CHECK(mask_has_token(mask, 97));  // 'a' allowed (start position)
  BOOST_CHECK(!mask_has_token(mask, 98)); // 'b' not allowed (would require 'a' first)
}

BOOST_AUTO_TEST_CASE(validate_tokens) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");
  const uint32_t tokens[] = {97, 98, 99};

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(
      llg_matcher_validate_tokens(matcher.get(), tokens, std::size(tokens)), 3);
}

BOOST_AUTO_TEST_CASE(validate_tokens_partial) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");
  const uint32_t tokens[] = {97, 48};

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(
      llg_matcher_validate_tokens(matcher.get(), tokens, std::size(tokens)), 1);
}

BOOST_AUTO_TEST_CASE(compute_ff_tokens) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "hello");
  std::vector<uint32_t> output(5);
  const uint32_t expected[] = {104, 101, 108, 108, 111};

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(
      llg_matcher_compute_ff_tokens(matcher.get(), output.data(), output.size()),
      static_cast<int32_t>(output.size()));
  BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.end(), std::begin(expected),
                                std::end(expected));
}

BOOST_AUTO_TEST_CASE(compute_ff_tokens_short_buffer) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "hello");
  // Buffer of 3 + 1 canary element that must not be overwritten
  constexpr uint32_t kCanary = 0xDEADBEEF;
  std::vector<uint32_t> output(4, kCanary);
  const uint32_t expected[] = {104, 101, 108};

  check_matcher_has_no_error(matcher.get());
  // Request only 3 slots; return value should still be 3 (the clamped length)
  BOOST_REQUIRE_EQUAL(
      llg_matcher_compute_ff_tokens(matcher.get(), output.data(), 3),
      3);
  BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.begin() + 3,
                                std::begin(expected), std::end(expected));
  // Canary must be untouched
  BOOST_CHECK_EQUAL(output[3], kCanary);
}

BOOST_AUTO_TEST_CASE(consume_invalid_token) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "ab");

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(llg_matcher_consume_token(matcher.get(), 120), -1); // 'x'
  BOOST_CHECK(llg_matcher_is_error(matcher.get()));
}

BOOST_AUTO_TEST_CASE(rollback_zero) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0); // 'a'
  BOOST_REQUIRE_EQUAL(llg_matcher_rollback(matcher.get(), 0), 0);
  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(matcher.get()), 0);

  const uint32_t *mask = llg_matcher_get_mask(matcher.get());
  BOOST_REQUIRE(mask != nullptr);
  for (uint32_t token = 97; token <= 122; ++token) {
    BOOST_CHECK(mask_has_token(mask, token));
  }
}

BOOST_AUTO_TEST_CASE(rollback_too_many) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0); // 'a'
  BOOST_CHECK_EQUAL(llg_matcher_rollback(matcher.get(), 5), -1);
}

BOOST_AUTO_TEST_CASE(reset_error_state) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("lark", "start: (");

  BOOST_REQUIRE(matcher != nullptr);
  BOOST_REQUIRE(llg_matcher_is_error(matcher.get()));
  BOOST_CHECK_EQUAL(llg_matcher_reset(matcher.get()), -1);
}

BOOST_AUTO_TEST_CASE(ff_tokens_null_output_returns_error) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "hello");

  check_matcher_has_no_error(matcher.get());
  // Passing nullptr is explicitly rejected even with output_len=0
  BOOST_CHECK_EQUAL(llg_matcher_compute_ff_tokens(matcher.get(), nullptr, 0), -1);
}

BOOST_AUTO_TEST_CASE(ff_tokens_zero_length_buffer) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "hello");

  check_matcher_has_no_error(matcher.get());
  uint32_t dummy = 0xDEADBEEF;
  // Zero-length buffer: returns 0 tokens written, does not touch buffer
  int32_t n = llg_matcher_compute_ff_tokens(matcher.get(), &dummy, 0);
  BOOST_CHECK_EQUAL(n, 0);
  BOOST_CHECK_EQUAL(dummy, 0xDEADBEEF);
}

BOOST_AUTO_TEST_CASE(ff_tokens_healthy_empty_result) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");
  uint32_t output = 0xDEADBEEF;

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(llg_matcher_compute_ff_tokens(matcher.get(), &output, 1), 0);
  BOOST_CHECK_EQUAL(output, 0xDEADBEEF);
  BOOST_CHECK(!llg_matcher_is_error(matcher.get()));
  BOOST_CHECK(!llg_matcher_is_cancelled(matcher.get()));
  BOOST_CHECK(llg_matcher_get_error(matcher.get()) == nullptr);
}

BOOST_AUTO_TEST_CASE(compute_mask_into_wrong_size) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());

  const size_t mask_byte_size = llg_matcher_get_mask_byte_size(matcher.get());
  std::vector<uint32_t> mask(mask_byte_size / sizeof(uint32_t));

  BOOST_CHECK_EQUAL(
      llg_matcher_compute_mask_into(matcher.get(), mask.data(), mask_byte_size / 2),
      -1);
}

BOOST_AUTO_TEST_CASE(is_accepting_initial_plus) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK(!llg_matcher_is_accepting(matcher.get()));
}

BOOST_AUTO_TEST_CASE(is_accepting_initial_star) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]*");

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK(llg_matcher_is_accepting(matcher.get()));
}

BOOST_AUTO_TEST_CASE(validate_tokens_empty) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(llg_matcher_validate_tokens(matcher.get(), nullptr, 0), 0);
}

BOOST_AUTO_TEST_CASE(validate_tokens_all_invalid) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");
  const uint32_t tokens[] = {48, 49};

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(
      llg_matcher_validate_tokens(matcher.get(), tokens, std::size(tokens)), 0);
}

BOOST_AUTO_TEST_CASE(clone_matcher) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0);
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 98), 0);

  MatcherPtr clone(llg_clone_matcher(matcher.get()));
  BOOST_REQUIRE(clone != nullptr);
  BOOST_CHECK(llg_matcher_is_accepting(matcher.get()));
  BOOST_CHECK(llg_matcher_is_accepting(clone.get()));

  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(matcher.get()), 0);
  BOOST_REQUIRE_EQUAL(llg_matcher_compute_mask(clone.get()), 0);

  const uint32_t *matcher_mask = llg_matcher_get_mask(matcher.get());
  const uint32_t *clone_mask = llg_matcher_get_mask(clone.get());
  const size_t mask_byte_size = llg_matcher_get_mask_byte_size(matcher.get());

  BOOST_REQUIRE(matcher_mask != nullptr);
  BOOST_REQUIRE(clone_mask != nullptr);
  BOOST_CHECK_EQUAL(std::memcmp(matcher_mask, clone_mask, mask_byte_size), 0);
}

BOOST_AUTO_TEST_CASE(free_matcher_no_crash) {
  MatcherContext ctx;
  LlgMatcher *matcher = llg_new_matcher(&ctx.init, "regex", "[a-z]+");

  BOOST_REQUIRE(matcher != nullptr);
  llg_free_matcher(matcher);
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(consume_invalid_token_has_error_message) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "ab");

  check_matcher_has_no_error(matcher.get());
  BOOST_CHECK_EQUAL(llg_matcher_consume_token(matcher.get(), 120), -1); // 'x'
  BOOST_CHECK(llg_matcher_is_error(matcher.get()));

  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK(std::strlen(error) > 0);
}

BOOST_AUTO_TEST_CASE(rollback_too_many_has_error_message) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());
  BOOST_REQUIRE_EQUAL(llg_matcher_consume_token(matcher.get(), 97), 0); // 'a'
  BOOST_CHECK_EQUAL(llg_matcher_rollback(matcher.get(), 5), -1);

  const char *error = llg_matcher_get_error(matcher.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK(std::strlen(error) > 0);
}

BOOST_AUTO_TEST_CASE(matcher_get_mask_byte_size_before_compute) {
  MatcherContext ctx;
  auto matcher = ctx.make_matcher("regex", "[a-z]+");

  check_matcher_has_no_error(matcher.get());

  // llg_matcher_get_mask_byte_size is a pure query that should work
  // before any mask computation.
  const size_t byte_size = llg_matcher_get_mask_byte_size(matcher.get());
  BOOST_CHECK(byte_size > 0);
  BOOST_CHECK(byte_size % sizeof(uint32_t) == 0);
}

BOOST_AUTO_TEST_CASE(free_matcher_null) {
  llg_free_matcher(nullptr);
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(free_cancellation_handle_null) {
  llg_free_cancellation_handle(nullptr);
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END()
