#include <boost/test/unit_test.hpp>

#include <array>
#include <cstring>
#include <memory>
#include <string>

#include "test_helpers.h"
#include "llguidance.h"

namespace {

struct StopControllerDeleter {
  void operator()(LlgStopController *ptr) const {
    if (ptr != nullptr) {
      llg_free_stop_controller(ptr);
    }
  }
};

struct TokenizerDeleter {
  void operator()(LlgTokenizer *ptr) const {
    if (ptr != nullptr) {
      llg_free_tokenizer(ptr);
    }
  }
};

using StopControllerPtr =
    std::unique_ptr<LlgStopController, StopControllerDeleter>;
using TokenizerPtr = std::unique_ptr<LlgTokenizer, TokenizerDeleter>;

std::string committed_text(LlgStopController *stop_ctrl, uint32_t token,
                           size_t &output_len, bool &is_stopped) {
  const char *output =
      llg_stop_commit_token(stop_ctrl, token, &output_len, &is_stopped);
  BOOST_REQUIRE(output != nullptr);
  return std::string(output, output_len);
}

} // namespace

BOOST_AUTO_TEST_SUITE(stop_controller)

BOOST_AUTO_TEST_CASE(create_with_stop_token) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  const uint32_t stop_tokens[] = {BYTE_TOK_EOS};
  char error[256] = {};

  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), stop_tokens, 1, nullptr, error, sizeof(error)));

  BOOST_TEST(stop_ctrl.get() != nullptr);
}

BOOST_AUTO_TEST_CASE(create_with_stop_regex) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  char error[256] = {};

  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), nullptr, 0, "\\nEND", error, sizeof(error)));

  BOOST_TEST(stop_ctrl.get() != nullptr);
}

BOOST_AUTO_TEST_CASE(create_invalid_regex) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  char error[256] = {};

  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), nullptr, 0, "[z-a", error, sizeof(error)));

  BOOST_TEST(stop_ctrl.get() == nullptr);
  BOOST_TEST(error[0] != '\0');
}

BOOST_AUTO_TEST_CASE(commit_token_no_stop) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  const uint32_t stop_tokens[] = {BYTE_TOK_EOS};
  char error[256] = {};
  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), stop_tokens, 1, nullptr, error, sizeof(error)));
  BOOST_REQUIRE_MESSAGE((stop_ctrl.get() != nullptr), error);

  const std::array<uint32_t, 5> tokens = {104, 101, 108, 108, 111};
  const std::array<std::string, 5> expected = {"h", "e", "l", "l", "o"};

  for (size_t i = 0; i < tokens.size(); ++i) {
    size_t output_len = 0;
    bool is_stopped = true;
    const auto output =
        committed_text(stop_ctrl.get(), tokens[i], output_len, is_stopped);

    BOOST_TEST(is_stopped == false);
    BOOST_TEST(output_len == expected[i].size());
    BOOST_TEST(output == expected[i]);
  }
}

BOOST_AUTO_TEST_CASE(commit_token_stop_detected) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  const uint32_t stop_tokens[] = {BYTE_TOK_EOS};
  char error[256] = {};
  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), stop_tokens, 1, nullptr, error, sizeof(error)));
  BOOST_REQUIRE_MESSAGE((stop_ctrl.get() != nullptr), error);

  for (uint32_t token : {104u, 101u, 108u}) {
    size_t output_len = 0;
    bool is_stopped = true;
    const auto output =
        committed_text(stop_ctrl.get(), token, output_len, is_stopped);

    BOOST_TEST(is_stopped == false);
    BOOST_TEST(output_len == 1u);
    BOOST_TEST(output.empty() == false);
  }

  size_t output_len = 99;
  bool is_stopped = false;
  const auto output =
      committed_text(stop_ctrl.get(), BYTE_TOK_EOS, output_len, is_stopped);

  BOOST_TEST(is_stopped == true);
  BOOST_TEST(output_len == 0u);
  BOOST_TEST(output.empty());
}

BOOST_AUTO_TEST_CASE(clone_stop_controller) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  char error[256] = {};
  // Use regex stop pattern "ab" so the controller accumulates match state.
  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), nullptr, 0, "ab", error, sizeof(error)));
  BOOST_REQUIRE_MESSAGE((stop_ctrl.get() != nullptr), error);

  // Commit 'x' (120) - no match progress, text is emitted
  size_t output_len = 0;
  bool is_stopped = false;
  BOOST_TEST(committed_text(stop_ctrl.get(), 120, output_len, is_stopped) ==
             "x");
  BOOST_TEST(is_stopped == false);

  // Commit 'a' (97) - starts matching "ab"; text may be held back
  committed_text(stop_ctrl.get(), 97, output_len, is_stopped);
  BOOST_TEST(is_stopped == false);

  // Clone midway through the regex match sequence
  StopControllerPtr clone(llg_clone_stop_controller(stop_ctrl.get()));
  BOOST_REQUIRE(clone.get() != nullptr);

  // On the original: commit 'b' (98) - completes "ab", should stop
  committed_text(stop_ctrl.get(), 98, output_len, is_stopped);
  BOOST_TEST(is_stopped == true);

  // On the clone: commit 'b' (98) - should also stop (cloned state had 'a')
  committed_text(clone.get(), 98, output_len, is_stopped);
  BOOST_TEST(is_stopped == true);
}

BOOST_AUTO_TEST_CASE(create_with_multiple_stop_tokens) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  const uint32_t stop_tokens[] = {BYTE_TOK_EOS, 200};
  char error[256] = {};

  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), stop_tokens, 2, nullptr, error, sizeof(error)));
  BOOST_REQUIRE_MESSAGE((stop_ctrl.get() != nullptr), error);

  size_t output_len = 0;
  bool is_stopped = false;
  const auto output =
      committed_text(stop_ctrl.get(), 200, output_len, is_stopped);

  BOOST_TEST(is_stopped == true);
  BOOST_TEST(output_len == 0u);
  BOOST_TEST(output.empty());
}

BOOST_AUTO_TEST_CASE(commit_after_stop) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  const uint32_t stop_tokens[] = {BYTE_TOK_EOS};
  char error[256] = {};
  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), stop_tokens, 1, nullptr, error, sizeof(error)));
  BOOST_REQUIRE_MESSAGE((stop_ctrl.get() != nullptr), error);

  size_t output_len = 0;
  bool is_stopped = false;
  committed_text(stop_ctrl.get(), BYTE_TOK_EOS, output_len, is_stopped);
  BOOST_TEST(is_stopped == true);

  const char *output =
      llg_stop_commit_token(stop_ctrl.get(), static_cast<uint32_t>('a'),
                            &output_len, &is_stopped);
  BOOST_REQUIRE(output != nullptr);
}

BOOST_AUTO_TEST_CASE(create_with_both_token_and_regex) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  const uint32_t stop_tokens[] = {BYTE_TOK_EOS};
  char error[256] = {};

  StopControllerPtr stop_ctrl(llg_new_stop_controller(
      tokenizer.get(), stop_tokens, 1, "xy", error, sizeof(error)));
  BOOST_REQUIRE_MESSAGE((stop_ctrl.get() != nullptr), error);

  size_t output_len = 0;
  bool is_stopped = false;
  const auto output =
      committed_text(stop_ctrl.get(), BYTE_TOK_EOS, output_len, is_stopped);

  BOOST_TEST(is_stopped == true);
  BOOST_TEST(output_len == 0u);
  BOOST_TEST(output.empty());
}

BOOST_AUTO_TEST_SUITE_END()
