#include <boost/test/unit_test.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "llguidance.h"
#include "test_helpers.h"

namespace {

struct SmallVocab {
  std::array<uint32_t, 3> token_lens{1, 1, 5};
  std::array<uint8_t, 7> token_bytes{
      static_cast<uint8_t>('a'), static_cast<uint8_t>('b'),
      static_cast<uint8_t>('<'), static_cast<uint8_t>('E'),
      static_cast<uint8_t>('O'), static_cast<uint8_t>('S'),
      static_cast<uint8_t>('>')};
};

} // namespace

BOOST_AUTO_TEST_SUITE(tokenizer)

BOOST_AUTO_TEST_CASE(create_byte_tokenizer_v1) {
  LlgTokenizer *tok = create_byte_tokenizer();
  BOOST_REQUIRE_NE(tok, nullptr);
  llg_free_tokenizer(tok);
}

BOOST_AUTO_TEST_CASE(create_byte_tokenizer_v2_api) {
  LlgTokenizer *tok = create_byte_tokenizer_v2();
  BOOST_REQUIRE_NE(tok, nullptr);
  llg_free_tokenizer(tok);
}

BOOST_AUTO_TEST_CASE(clone_tokenizer) {
  LlgTokenizer *tok = create_byte_tokenizer();
  BOOST_REQUIRE_NE(tok, nullptr);

  LlgTokenizer *clone = llg_clone_tokenizer(tok);
  BOOST_REQUIRE_NE(clone, nullptr);

  llg_free_tokenizer(clone);
  llg_free_tokenizer(tok);
}

BOOST_AUTO_TEST_CASE(tokenizer_v1_missing_tokenize_fn) {
  SmallVocab vocab;
  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 2;
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = nullptr;
  tok_init.use_approximate_greedy_tokenize_fn = false;

  char error_buf[256] = {};
  LlgTokenizer *tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strlen(error_buf) > 0U);
  BOOST_TEST(std::string_view(error_buf).find(
                 "Either tokenize_fn or use_approximate_greedy_tokenize_fn must "
                 "be set") != std::string_view::npos);
}

BOOST_AUTO_TEST_CASE(tokenizer_v2_invalid_struct_size) {
  SmallVocab vocab;
  LlgTokenizerInitV2 tok_init = {};
  tok_init.struct_size = 4;
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 2;
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = byte_tokenize_callback;

  char error_buf[256] = {};
  LlgTokenizer *tok =
      llg_new_tokenizer_v2(&tok_init, error_buf, sizeof(error_buf));

  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strlen(error_buf) > 0U);
  BOOST_TEST(std::string_view(error_buf).find("struct_size") !=
             std::string_view::npos);
}

BOOST_AUTO_TEST_CASE(tokenizer_v2_null_pointer) {
  char error_buf[256] = {};
  LlgTokenizer *tok = llg_new_tokenizer_v2(nullptr, error_buf, sizeof(error_buf));

  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strcmp(error_buf, "tok_init is NULL") == 0);
}

BOOST_AUTO_TEST_CASE(tokenizer_v2_eos_out_of_range) {
  SmallVocab vocab;
  uint32_t extra_eos[] = {99};

  LlgTokenizerInitV2 tok_init = {};
  tok_init.struct_size = sizeof(tok_init);
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 2;
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = byte_tokenize_callback;
  tok_init.tok_eos_extra = extra_eos;
  tok_init.tok_eos_extra_count = 1;

  char error_buf[256] = {};
  LlgTokenizer *tok =
      llg_new_tokenizer_v2(&tok_init, error_buf, sizeof(error_buf));

  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strlen(error_buf) > 0U);
  BOOST_TEST(std::string_view(error_buf).find("out of range") !=
             std::string_view::npos);
}

BOOST_AUTO_TEST_CASE(tokenizer_v1_primary_eos_out_of_range) {
  SmallVocab vocab;
  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 99; // out of range (vocab_size=3)
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = byte_tokenize_callback;

  char error_buf[256] = {};
  LlgTokenizer *tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strlen(error_buf) > 0U);
  BOOST_TEST(std::string_view(error_buf).find("out of range") !=
             std::string_view::npos);
}

BOOST_AUTO_TEST_CASE(tokenizer_v2_primary_eos_out_of_range) {
  SmallVocab vocab;
  LlgTokenizerInitV2 tok_init = {};
  tok_init.struct_size = sizeof(tok_init);
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 99; // out of range, no extra EOS
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = byte_tokenize_callback;

  char error_buf[256] = {};
  LlgTokenizer *tok =
      llg_new_tokenizer_v2(&tok_init, error_buf, sizeof(error_buf));

  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strlen(error_buf) > 0U);
  BOOST_TEST(std::string_view(error_buf).find("out of range") !=
             std::string_view::npos);
}

BOOST_AUTO_TEST_CASE(tokenizer_with_greedy_approx) {
  SmallVocab vocab;
  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 2;
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = nullptr;
  tok_init.use_approximate_greedy_tokenize_fn = true;

  char error_buf[256] = {};
  LlgTokenizer *tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  BOOST_REQUIRE_MESSAGE((tok != nullptr),
                        "Expected tokenizer creation to succeed: " << error_buf);
  llg_free_tokenizer(tok);
}

BOOST_AUTO_TEST_CASE(tokenizer_v1_vocab_size_zero) {
  // Supply non-null dummy pointers so construction doesn't fail at the
  // "token_lens and token_bytes must be set" check — we want to reach the
  // EOS out-of-range validation.
  uint32_t dummy_lens = 0;
  uint8_t dummy_bytes = 0;
  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = 0;
  tok_init.tok_eos = 0;
  tok_init.token_lens = &dummy_lens;
  tok_init.token_bytes = &dummy_bytes;
  tok_init.tokenize_fn = byte_tokenize_callback;

  char error_buf[256] = {};
  LlgTokenizer *tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  // vocab_size=0 with tok_eos=0 is out of range (no valid token IDs)
  BOOST_TEST(tok == nullptr);
  BOOST_TEST(std::strlen(error_buf) > 0U);
  BOOST_TEST(std::string_view(error_buf).find("out of range") !=
             std::string_view::npos);
}

BOOST_AUTO_TEST_CASE(tokenizer_v1_both_fn_and_greedy) {
  SmallVocab vocab;
  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 2;
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = byte_tokenize_callback;
  tok_init.use_approximate_greedy_tokenize_fn = true;

  char error_buf[256] = {};
  LlgTokenizer *tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));

  // When both are set, tokenize_fn takes precedence — should succeed
  BOOST_REQUIRE_MESSAGE((tok != nullptr),
                        "Expected tokenizer creation to succeed: " << error_buf);
  llg_free_tokenizer(tok);
}

BOOST_AUTO_TEST_CASE(tokenizer_v2_eos_extra_null_with_count) {
  SmallVocab vocab;
  LlgTokenizerInitV2 tok_init = {};
  tok_init.struct_size = sizeof(tok_init);
  tok_init.vocab_size = 3;
  tok_init.tok_eos = 2;
  tok_init.token_lens = vocab.token_lens.data();
  tok_init.token_bytes = vocab.token_bytes.data();
  tok_init.tokenize_fn = byte_tokenize_callback;
  tok_init.tok_eos_extra = nullptr;
  tok_init.tok_eos_extra_count = 2; // count > 0 but pointer is null

  char error_buf[256] = {};
  LlgTokenizer *tok =
      llg_new_tokenizer_v2(&tok_init, error_buf, sizeof(error_buf));

  // Rust implementation gracefully ignores null tok_eos_extra regardless of count
  BOOST_REQUIRE_MESSAGE((tok != nullptr),
                        "Expected tokenizer creation to succeed: " << error_buf);
  llg_free_tokenizer(tok);
}

BOOST_AUTO_TEST_SUITE_END()
