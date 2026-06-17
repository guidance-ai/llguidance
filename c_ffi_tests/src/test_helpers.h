#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "llguidance.h"

// Byte-level tokenize callback: each byte is its own token.
extern "C" size_t byte_tokenize_callback(const void *user_data,
                                         const uint8_t *bytes, size_t bytes_len,
                                         uint32_t *output_tokens,
                                         size_t output_tokens_len);

// Create a byte-level tokenizer using the v1 API.
// Vocab: tokens 0..255 are single bytes, token 256 is <EOS>.
LlgTokenizer *create_byte_tokenizer();

// Create a byte-level tokenizer using the v2 API with two EOS tokens.
// Vocab: tokens 0..255 are single bytes, token 256 is <EOS>, token 257 is <EOS2>.
LlgTokenizer *create_byte_tokenizer_v2();

// Read the contents of a file into a string.
std::string read_file(const std::string &path);

// Helper: tokenize a string using llg_tokenize_bytes.
std::vector<uint32_t> llg_tokenize(const LlgTokenizer *tok, const std::string &s);

// Helper: stringify tokens using llg_stringify_tokens.
std::string llg_stringify(const LlgTokenizer *tok, const std::vector<uint32_t> &tokens);

// Vocab size for the byte tokenizer (v1).
constexpr uint32_t BYTE_VOCAB_SIZE = 257;
// EOS token ID for the byte tokenizer (v1).
constexpr uint32_t BYTE_TOK_EOS = 256;

// Vocab size for the byte tokenizer (v2).
constexpr uint32_t BYTE_V2_VOCAB_SIZE = 258;
// Primary EOS token ID for the byte tokenizer (v2).
constexpr uint32_t BYTE_V2_TOK_EOS = 256;
// Extra EOS token ID for the byte tokenizer (v2).
constexpr uint32_t BYTE_V2_TOK_EOS2 = 257;
