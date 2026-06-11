#define BOOST_TEST_MODULE LlgFfiTests
#include <boost/test/unit_test.hpp>

#include "test_helpers.h"

extern "C" size_t byte_tokenize_callback(const void * /*user_data*/,
                                         const uint8_t *bytes, size_t bytes_len,
                                         uint32_t *output_tokens,
                                         size_t output_tokens_len) {
  if (output_tokens_len > 0) {
    size_t n = std::min(output_tokens_len, bytes_len);
    for (size_t i = 0; i < n; i++) {
      output_tokens[i] = bytes[i];
    }
  }
  return bytes_len;
}

LlgTokenizer *create_byte_tokenizer() {
  std::vector<std::vector<uint8_t>> tokens;
  tokens.reserve(BYTE_VOCAB_SIZE);
  for (uint32_t i = 0; i < 256; i++) {
    tokens.push_back({(uint8_t)i});
  }
  const char *eos = "<EOS>";
  tokens.push_back(std::vector<uint8_t>(eos, eos + strlen(eos)));

  std::vector<uint32_t> token_lens(tokens.size());
  size_t total_size = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    token_lens[i] = (uint32_t)tokens[i].size();
    total_size += token_lens[i];
  }
  std::vector<uint8_t> token_bytes(total_size);
  size_t offset = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    std::copy(tokens[i].begin(), tokens[i].end(), token_bytes.data() + offset);
    offset += token_lens[i];
  }

  LlgTokenizerInit tok_init = {};
  tok_init.vocab_size = (uint32_t)tokens.size();
  tok_init.tok_eos = BYTE_TOK_EOS;
  tok_init.token_lens = token_lens.data();
  tok_init.token_bytes = token_bytes.data();
  tok_init.tokenize_assumes_string = false;
  tok_init.tokenize_user_data = nullptr;
  tok_init.tokenize_fn = byte_tokenize_callback;

  char error_buf[256];
  auto tok = llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));
  BOOST_REQUIRE_MESSAGE((tok != nullptr),
                        "Failed to create byte tokenizer: " << error_buf);
  return tok;
}

LlgTokenizer *create_byte_tokenizer_v2() {
  std::vector<std::vector<uint8_t>> tokens;
  tokens.reserve(BYTE_V2_VOCAB_SIZE);
  for (uint32_t i = 0; i < 256; i++) {
    tokens.push_back({(uint8_t)i});
  }
  const char *eos = "<EOS>";
  tokens.push_back(std::vector<uint8_t>(eos, eos + strlen(eos)));
  const char *eos2 = "<EOS2>";
  tokens.push_back(std::vector<uint8_t>(eos2, eos2 + strlen(eos2)));

  std::vector<uint32_t> token_lens(tokens.size());
  size_t total_size = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    token_lens[i] = (uint32_t)tokens[i].size();
    total_size += token_lens[i];
  }
  std::vector<uint8_t> token_bytes(total_size);
  size_t offset = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    std::copy(tokens[i].begin(), tokens[i].end(), token_bytes.data() + offset);
    offset += token_lens[i];
  }

  uint32_t eos_extra[] = {BYTE_V2_TOK_EOS2};

  LlgTokenizerInitV2 tok_init = {};
  tok_init.struct_size = sizeof(tok_init);
  tok_init.vocab_size = (uint32_t)tokens.size();
  tok_init.tok_eos = BYTE_V2_TOK_EOS;
  tok_init.token_lens = token_lens.data();
  tok_init.token_bytes = token_bytes.data();
  tok_init.tokenize_assumes_string = false;
  tok_init.tokenize_user_data = nullptr;
  tok_init.tokenize_fn = byte_tokenize_callback;
  tok_init.tok_eos_extra = eos_extra;
  tok_init.tok_eos_extra_count = 1;

  char error_buf[256];
  auto tok = llg_new_tokenizer_v2(&tok_init, error_buf, sizeof(error_buf));
  BOOST_REQUIRE_MESSAGE((tok != nullptr),
                        "Failed to create byte tokenizer v2: " << error_buf);
  return tok;
}

std::string read_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return "";
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::vector<uint32_t> llg_tokenize(const LlgTokenizer *tok,
                                   const std::string &s) {
  size_t n = llg_tokenize_bytes(tok, (const uint8_t *)s.c_str(), s.size(),
                                nullptr, 0);
  std::vector<uint32_t> tokens(n);
  llg_tokenize_bytes(tok, (const uint8_t *)s.c_str(), s.size(), tokens.data(),
                     n);
  return tokens;
}

std::string llg_stringify(const LlgTokenizer *tok,
                          const std::vector<uint32_t> &tokens) {
  char buffer[2048];
  size_t n = llg_stringify_tokens(tok, tokens.data(), tokens.size(), buffer,
                                  sizeof(buffer));
  if (n >= sizeof(buffer)) {
    std::vector<char> big_buf(n + 1);
    llg_stringify_tokens(tok, tokens.data(), tokens.size(), big_buf.data(),
                         big_buf.size());
    return std::string(big_buf.data());
  }
  return std::string(buffer);
}
