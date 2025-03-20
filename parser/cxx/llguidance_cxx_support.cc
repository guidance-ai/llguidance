#include "llguidance_cxx.h"

namespace llguidance {

FactoryInit::FactoryInit(size_t vocab_size, uint32_t tok_eos,
                         uint32_t stderr_log_level, bool allow_ff_tokens,
                         bool allow_backtracking)
    : m_vocab_size(vocab_size), m_tok_eos(tok_eos),
      m_stderr_log_level(stderr_log_level), m_allow_ff_tokens(allow_ff_tokens),
      m_allow_backtracking(allow_backtracking), m_slices(general_slices()) {}

} // namespace llguidance