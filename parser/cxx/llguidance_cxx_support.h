#pragma once

#include "llguidance_cxx.h"
#include <cstdint>

namespace llguidance {

class FactoryInit {
protected:
  size_t m_vocab_size;
  uint32_t m_tok_eos;
  uint32_t m_stderr_log_level;
  bool m_allow_ff_tokens;
  bool m_allow_backtracking;
  rust::Vec<rust::String> m_slices;

public:
  FactoryInit(size_t vocab_size, uint32_t tok_eos,
              uint32_t stderr_log_level = 1, bool allow_ff_tokens = false,
              bool allow_backtracking = false);

  virtual ~FactoryInit() = default;

  // Non-virtual configuration getters
  size_t vocab_size() const { return m_vocab_size; }
  uint32_t tok_eos() const { return m_tok_eos; }
  uint32_t stderr_log_level() const { return m_stderr_log_level; }
  bool allow_ff_tokens() const { return m_allow_ff_tokens; }
  bool allow_backtracking() const { return m_allow_backtracking; }
  rust::Vec<rust::String> slices() const { return m_slices; }

  // Return bytes corresponding to a given token
  // Prepend 0xff as the first byte, if it's a special token
  virtual rust::Vec<uint8_t> token_bytes(size_t token) const = 0;

  // Tokenize given UTF-8 text into a sequence of token IDs.
  // This function *has to be thread-safe*!
  virtual rust::Vec<uint32_t> tokenize(rust::Str text) const {
    // by default return empty tokenization - this will make llguidance
    // use greedy, non-canonical tokenizer
    (void)text;
    return rust::Vec<uint32_t>();
  }

protected:
  FactoryInit(const FactoryInit &) = delete;
  FactoryInit &operator=(const FactoryInit &) = delete;
  FactoryInit(FactoryInit &&) = delete;
  FactoryInit &operator=(FactoryInit &&) = delete;
};

} // namespace llguidance