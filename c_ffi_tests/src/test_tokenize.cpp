#include <boost/test/unit_test.hpp>

#include "test_helpers.h"
#include "llguidance.h"

namespace {

struct ByteTokenizer {
  ByteTokenizer() : tok(create_byte_tokenizer()) {}
  ~ByteTokenizer() { llg_free_tokenizer(tok); }

  LlgTokenizer *tok;
};

void check_tokens(const std::vector<uint32_t> &actual,
                  const std::vector<uint32_t> &expected) {
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.end(), expected.begin(),
                                expected.end());
}

} // namespace

BOOST_AUTO_TEST_SUITE(tokenize)

BOOST_AUTO_TEST_CASE(tokenize_bytes_basic) {
  ByteTokenizer tok;
  const std::string input = "hello";
  std::vector<uint32_t> tokens(input.size());

  size_t n = llg_tokenize_bytes(
      tok.tok, reinterpret_cast<const uint8_t *>(input.data()), input.size(),
      tokens.data(), tokens.size());

  BOOST_REQUIRE_EQUAL(n, 5u);
  check_tokens(tokens, {104, 101, 108, 108, 111});
}

BOOST_AUTO_TEST_CASE(tokenize_bytes_empty) {
  ByteTokenizer tok;
  const std::string input;

  size_t n = llg_tokenize_bytes(
      tok.tok, reinterpret_cast<const uint8_t *>(input.data()), input.size(),
      nullptr, 0);

  BOOST_CHECK_EQUAL(n, 0u);
}

BOOST_AUTO_TEST_CASE(tokenize_bytes_count_only) {
  ByteTokenizer tok;
  const std::string input = "hello";

  size_t n = llg_tokenize_bytes(
      tok.tok, reinterpret_cast<const uint8_t *>(input.data()), input.size(),
      nullptr, 0);

  BOOST_CHECK_EQUAL(n, 5u);
}

BOOST_AUTO_TEST_CASE(tokenize_bytes_short_buffer) {
  ByteTokenizer tok;
  const std::string input = "hello";
  std::vector<uint32_t> tokens(3, 0);

  size_t n = llg_tokenize_bytes(
      tok.tok, reinterpret_cast<const uint8_t *>(input.data()), input.size(),
      tokens.data(), tokens.size());

  BOOST_REQUIRE_EQUAL(n, 5u);
  check_tokens(tokens, {104, 101, 108});
}

BOOST_AUTO_TEST_CASE(tokenize_bytes_marker) {
  ByteTokenizer tok;
  // Include a \xff[256] marker sequence which should produce token 256 (EOS)
  const uint8_t input[] = {'h', 'i', 0xff, '[', '2', '5', '6', ']'};
  std::vector<uint32_t> tokens(3);

  size_t n = llg_tokenize_bytes_marker(
      tok.tok, input, sizeof(input),
      tokens.data(), tokens.size());

  // Should produce 3 tokens: 'h'=104, 'i'=105, and marker token 256 (EOS)
  BOOST_REQUIRE_EQUAL(n, 3u);
  check_tokens(tokens, {104, 105, BYTE_TOK_EOS});
}

BOOST_AUTO_TEST_CASE(stringify_tokens_basic) {
  ByteTokenizer tok;
  const auto tokens = llg_tokenize(tok.tok, "hi");
  const std::string out = llg_stringify(tok.tok, tokens);

  BOOST_TEST(!out.empty());
  BOOST_TEST(out.find("h") != std::string::npos);
  BOOST_TEST(out.find("i") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(stringify_tokens_buffer_too_small) {
  ByteTokenizer tok;
  const auto tokens = llg_tokenize(tok.tok, "hi");
  char buffer[2] = {};

  size_t needed =
      llg_stringify_tokens(tok.tok, tokens.data(), tokens.size(), nullptr, 0);
  size_t n =
      llg_stringify_tokens(tok.tok, tokens.data(), tokens.size(), buffer,
                           sizeof(buffer));

  BOOST_CHECK_EQUAL(n, needed);
  BOOST_CHECK_GT(n, sizeof(buffer));
  BOOST_CHECK_EQUAL(buffer[1], '\0');
}

BOOST_AUTO_TEST_CASE(decode_tokens_none_flag) {
  ByteTokenizer tok;
  const auto tokens = llg_tokenize(tok.tok, "hello");
  char buffer[16] = {};

  size_t n = llg_decode_tokens(tok.tok, tokens.data(), tokens.size(), buffer,
                               sizeof(buffer), LLG_DECODE_NONE);

  BOOST_CHECK_EQUAL(n, 6u);
  BOOST_CHECK_EQUAL(std::string(buffer), "hello");
}

BOOST_AUTO_TEST_CASE(decode_tokens_valid_utf8_flag) {
  ByteTokenizer tok;
  const uint32_t token = 128;
  char buffer[16] = {};

  size_t n = llg_decode_tokens(tok.tok, &token, 1, buffer, sizeof(buffer),
                               LLG_DECODE_VALID_UTF8);

  BOOST_CHECK_EQUAL(n, 4u);
  BOOST_CHECK_EQUAL(std::string(buffer), "\xEF\xBF\xBD");
}

BOOST_AUTO_TEST_CASE(roundtrip_tokenize_stringify) {
  ByteTokenizer tok;
  const std::string input = "hello";
  const auto tokens = llg_tokenize(tok.tok, input);
  const std::string out = llg_stringify(tok.tok, tokens);

  BOOST_TEST(!out.empty());
  BOOST_TEST(out.find("h") != std::string::npos);
  BOOST_TEST(out.find("e") != std::string::npos);
  BOOST_TEST(out.find("l") != std::string::npos);
  BOOST_TEST(out.find("o") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(tokenize_marker_at_start) {
  ByteTokenizer tok;
  const uint8_t input[] = {0xff, '[', '2', '5', '6', ']', 'h', 'i'};
  std::vector<uint32_t> tokens(3);

  size_t n =
      llg_tokenize_bytes_marker(tok.tok, input, sizeof(input), tokens.data(),
                                tokens.size());

  BOOST_REQUIRE_EQUAL(n, 3u);
  check_tokens(tokens, {BYTE_TOK_EOS, 104, 105});
}

BOOST_AUTO_TEST_CASE(tokenize_marker_multiple) {
  ByteTokenizer tok;
  const uint8_t input[] = {'a', 0xff, '[', '2', '5', '6', ']', 0xff,
                           '[', '1', '0', '0', ']', 'b'};
  std::vector<uint32_t> tokens(4);

  size_t n =
      llg_tokenize_bytes_marker(tok.tok, input, sizeof(input), tokens.data(),
                                tokens.size());

  BOOST_REQUIRE_EQUAL(n, 4u);
  check_tokens(tokens, {97, BYTE_TOK_EOS, 100, 98});
}

BOOST_AUTO_TEST_CASE(tokenize_marker_count_only) {
  ByteTokenizer tok;
  const uint8_t input[] = {0xff, '[', '2', '5', '6', ']'};

  size_t n = llg_tokenize_bytes_marker(tok.tok, input, sizeof(input), nullptr, 0);

  BOOST_CHECK_EQUAL(n, 1u);
}

// NOTE: bare \xff handling (not followed by '[' or '<') is currently undefined
// behavior in llg_tokenize_bytes_marker — the byte may be silently dropped.
// A future fix should tokenize it literally as token 255.

BOOST_AUTO_TEST_CASE(decode_tokens_include_special) {
  ByteTokenizer tok;
  const uint32_t token = BYTE_TOK_EOS;
  char buffer[16] = {};

  size_t n = llg_decode_tokens(tok.tok, &token, 1, buffer, sizeof(buffer),
                               LLG_DECODE_INCLUDE_SPECIAL);

  BOOST_CHECK_GT(n, 1u);
  BOOST_CHECK(!std::string(buffer).empty());
}

BOOST_AUTO_TEST_CASE(decode_tokens_combined_flags) {
  ByteTokenizer tok;
  const uint32_t tokens[] = {128, BYTE_TOK_EOS};
  char buffer[64] = {};

  size_t n = llg_decode_tokens(tok.tok, tokens, 2, buffer, sizeof(buffer),
                               LLG_DECODE_INCLUDE_SPECIAL |
                                   LLG_DECODE_VALID_UTF8);

  // Token 128 is invalid UTF-8 → replacement char (3 bytes); EOS has some representation
  BOOST_CHECK_GT(n, 4u);
  std::string result(buffer);
  // Should contain the UTF-8 replacement character U+FFFD
  BOOST_CHECK(result.find("\xEF\xBF\xBD") != std::string::npos);
  // Should contain some representation of the EOS token
  BOOST_CHECK(result.size() > 3u);
}

BOOST_AUTO_TEST_CASE(decode_tokens_empty) {
  ByteTokenizer tok;
  char buffer[4] = {'x', '\0', '\0', '\0'};

  size_t n = llg_decode_tokens(tok.tok, nullptr, 0, buffer, sizeof(buffer),
                               LLG_DECODE_NONE);

  BOOST_CHECK_EQUAL(n, 1u);
  BOOST_CHECK_EQUAL(std::string(buffer), "");
}

BOOST_AUTO_TEST_CASE(decode_tokens_buffer_too_small) {
  ByteTokenizer tok;
  const uint32_t tokens[] = {104, 101, 108, 108, 111};
  char buffer[3] = {};

  size_t n = llg_decode_tokens(tok.tok, tokens, 5, buffer, sizeof(buffer),
                               LLG_DECODE_NONE);

  BOOST_CHECK_EQUAL(n, 6u);
  BOOST_CHECK_EQUAL(buffer[0], 'h');
  BOOST_CHECK_EQUAL(buffer[1], 'e');
  BOOST_CHECK_EQUAL(buffer[2], '\0');
}

BOOST_AUTO_TEST_SUITE_END()
