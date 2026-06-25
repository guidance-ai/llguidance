#include <boost/test/unit_test.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "test_helpers.h"
#include "llguidance.h"

namespace {

constexpr size_t kMaskU32Count = (BYTE_VOCAB_SIZE + 31) / 32;
constexpr size_t kMaskByteLen = kMaskU32Count * sizeof(uint32_t);

struct ConstraintDeleter {
  void operator()(LlgConstraint *ptr) const {
    if (ptr != nullptr) {
      llg_free_constraint(ptr);
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

using ConstraintPtr = std::unique_ptr<LlgConstraint, ConstraintDeleter>;
using TokenizerPtr = std::unique_ptr<LlgTokenizer, TokenizerDeleter>;

bool mask_allows(const std::vector<uint32_t> &mask, uint32_t token) {
  return (mask[token / 32] & (uint32_t{1} << (token % 32))) != 0;
}

ConstraintPtr new_regex_constraint(const LlgTokenizer *tokenizer,
                                   const char *regex) {
  LlgConstraintInit init = {};
  llg_constraint_init_set_defaults(&init, tokenizer);
  init.log_stderr_level = 0;

  ConstraintPtr constraint(llg_new_constraint_regex(&init, regex));
  BOOST_REQUIRE(constraint.get() != nullptr);
  BOOST_REQUIRE_MESSAGE((llg_get_error(constraint.get()) == nullptr),
                        llg_get_error(constraint.get()));
  return constraint;
}

extern "C" void mark_done_callback(const void *user_data) {
  auto *done = static_cast<std::atomic<bool> *>(
      const_cast<void *>(user_data));
  done->store(true, std::memory_order_release);
}

void wait_for_done(const std::atomic<bool> &done) {
  // Use a generous deadline so the Rayon callback has time to complete.
  // We must not unwind (via BOOST_REQUIRE) while the callback may still
  // access stack-owned data through the pointer descriptors.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  BOOST_REQUIRE_MESSAGE(done.load(std::memory_order_acquire),
                        "Timed out waiting for parallel compute_mask callback");
}

std::vector<uint32_t> compute_mask_sequential(LlgConstraint *constraint) {
  LlgMaskResult result = {};
  BOOST_REQUIRE_EQUAL(llg_compute_mask(constraint, &result), 0);
  BOOST_REQUIRE(result.sample_mask != nullptr);
  return std::vector<uint32_t>(result.sample_mask,
                               result.sample_mask + kMaskU32Count);
}

std::vector<uint32_t> compute_mask_parallel(LlgConstraint *constraint) {
  std::vector<uint32_t> mask(kMaskU32Count, 0);
  LlgConstraintStep step = {constraint, mask.data(), kMaskByteLen};
  std::atomic<bool> done{false};

  llg_par_compute_mask(&step, 1, &done, mark_done_callback);
  wait_for_done(done);

  return mask;
}

} // namespace

BOOST_AUTO_TEST_SUITE(parallel)

BOOST_AUTO_TEST_CASE(par_compute_mask_basic) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  auto alpha_constraint = new_regex_constraint(tokenizer.get(), "[a-z]+");
  auto digit_constraint = new_regex_constraint(tokenizer.get(), "[0-9]+");

  std::vector<uint32_t> alpha_mask(kMaskU32Count, 0);
  std::vector<uint32_t> digit_mask(kMaskU32Count, 0);
  std::atomic<bool> done{false};
  std::array<LlgConstraintStep, 2> steps = {{
      {alpha_constraint.get(), alpha_mask.data(), kMaskByteLen},
      {digit_constraint.get(), digit_mask.data(), kMaskByteLen},
  }};

  llg_par_compute_mask(steps.data(), steps.size(), &done, mark_done_callback);
  wait_for_done(done);

  BOOST_TEST(mask_allows(alpha_mask, static_cast<uint32_t>('a')));
  BOOST_TEST(mask_allows(alpha_mask, static_cast<uint32_t>('z')));
  BOOST_TEST(!mask_allows(alpha_mask, static_cast<uint32_t>('0')));

  BOOST_TEST(mask_allows(digit_mask, static_cast<uint32_t>('0')));
  BOOST_TEST(mask_allows(digit_mask, static_cast<uint32_t>('9')));
  BOOST_TEST(!mask_allows(digit_mask, static_cast<uint32_t>('a')));
}

BOOST_AUTO_TEST_CASE(par_compute_mask_matches_sequential) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  auto constraint_seq = new_regex_constraint(tokenizer.get(), "[a-z]+");
  auto constraint_par = new_regex_constraint(tokenizer.get(), "[a-z]+");

  const auto sequential_mask = compute_mask_sequential(constraint_seq.get());
  const auto parallel_mask = compute_mask_parallel(constraint_par.get());

  BOOST_REQUIRE_EQUAL(parallel_mask.size(), sequential_mask.size());
  for (size_t i = 0; i < parallel_mask.size(); ++i) {
    BOOST_TEST(parallel_mask[i] == sequential_mask[i]);
  }
}

BOOST_AUTO_TEST_CASE(par_compute_mask_single_step) {
  TokenizerPtr tokenizer(create_byte_tokenizer());
  auto constraint = new_regex_constraint(tokenizer.get(), "[a-z]+");

  std::vector<uint32_t> mask(kMaskU32Count, 0);
  std::atomic<bool> done{false};
  LlgConstraintStep step = {constraint.get(), mask.data(), kMaskByteLen};

  llg_par_compute_mask(&step, 1, &done, mark_done_callback);
  wait_for_done(done);

  BOOST_TEST(mask_allows(mask, static_cast<uint32_t>('a')));
  BOOST_TEST(mask_allows(mask, static_cast<uint32_t>('z')));
  BOOST_TEST(!mask_allows(mask, static_cast<uint32_t>('0')));
}

BOOST_AUTO_TEST_CASE(par_compute_mask_zero_steps) {
  std::array<LlgConstraintStep, 1> steps = {{
      {nullptr, nullptr, 0},
  }};
  std::atomic<bool> done{false};

  llg_par_compute_mask(steps.data(), 0, &done, mark_done_callback);
  wait_for_done(done);
}

BOOST_AUTO_TEST_CASE(par_compute_mask_null_steps_nonzero_count) {
  std::atomic<bool> done{false};

  // Null steps with n_steps > 0 is a caller bug. The detected error is handled
  // internally (no work is performed and no masks are written), but the
  // callback is still invoked exactly once so that an async caller blocked on
  // it does not deadlock.
  llg_par_compute_mask(nullptr, 2, &done, mark_done_callback);

  wait_for_done(done);
  BOOST_CHECK(done.load(std::memory_order_acquire));
}

BOOST_AUTO_TEST_CASE(par_compute_mask_null_constraint) {
  // A step with a null constraint pointer should be silently skipped
  // (no crash, callback still invoked).
  std::vector<uint32_t> mask(kMaskU32Count, 0xDEADBEEF);
  LlgConstraintStep step = {nullptr, mask.data(), kMaskByteLen};
  std::atomic<bool> done{false};

  llg_par_compute_mask(&step, 1, &done, mark_done_callback);
  wait_for_done(done);

  // Mask buffer should be untouched since the step was skipped.
  for (size_t i = 0; i < kMaskU32Count; ++i) {
    BOOST_TEST(mask[i] == 0xDEADBEEF);
  }
}

BOOST_AUTO_TEST_CASE(par_compute_mask_mixed_null_and_valid) {
  // One null-constraint step and one valid step in the same batch.
  // The valid step should still get its mask computed correctly.
  TokenizerPtr tokenizer(create_byte_tokenizer());
  auto constraint = new_regex_constraint(tokenizer.get(), "[a-z]+");

  std::vector<uint32_t> null_mask(kMaskU32Count, 0xDEADBEEF);
  std::vector<uint32_t> valid_mask(kMaskU32Count, 0);
  std::array<LlgConstraintStep, 2> steps = {{
      {nullptr, null_mask.data(), kMaskByteLen},
      {constraint.get(), valid_mask.data(), kMaskByteLen},
  }};
  std::atomic<bool> done{false};

  llg_par_compute_mask(steps.data(), steps.size(), &done, mark_done_callback);
  wait_for_done(done);

  // Null-constraint step: mask untouched.
  for (size_t i = 0; i < kMaskU32Count; ++i) {
    BOOST_TEST(null_mask[i] == 0xDEADBEEF);
  }

  // Valid step: should allow lowercase letters.
  BOOST_TEST(mask_allows(valid_mask, static_cast<uint32_t>('a')));
  BOOST_TEST(mask_allows(valid_mask, static_cast<uint32_t>('z')));
  BOOST_TEST(!mask_allows(valid_mask, static_cast<uint32_t>('0')));
}

BOOST_AUTO_TEST_CASE(par_compute_mask_null_callback) {
  // Fire-and-forget mode: done_cb is null. The function should still compute
  // masks correctly; it just doesn't notify anyone on completion.
  TokenizerPtr tokenizer(create_byte_tokenizer());
  auto constraint = new_regex_constraint(tokenizer.get(), "[a-z]+");

  std::vector<uint32_t> mask(kMaskU32Count, 0);
  LlgConstraintStep step = {constraint.get(), mask.data(), kMaskByteLen};

  llg_par_compute_mask(&step, 1, nullptr, nullptr);

  // Without a callback we have no signal — wait a reasonable time for the
  // rayon worker to finish.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  BOOST_TEST(mask_allows(mask, static_cast<uint32_t>('a')));
  BOOST_TEST(mask_allows(mask, static_cast<uint32_t>('z')));
  BOOST_TEST(!mask_allows(mask, static_cast<uint32_t>('0')));
}

BOOST_AUTO_TEST_CASE(par_compute_mask_null_steps_zero_count) {
  // steps=nullptr with n_steps=0 is a valid no-op. The callback should still
  // be invoked (on a rayon thread).
  std::atomic<bool> done{false};

  llg_par_compute_mask(nullptr, 0, &done, mark_done_callback);
  wait_for_done(done);
}

BOOST_AUTO_TEST_CASE(par_compute_mask_error_propagation) {
  // Pass an invalid mask_byte_len (not a multiple of 4) to trigger the
  // per-constraint error path. Verify the error is recorded on the constraint.
  TokenizerPtr tokenizer(create_byte_tokenizer());
  auto constraint = new_regex_constraint(tokenizer.get(), "[a-z]+");

  std::vector<uint32_t> mask(kMaskU32Count, 0);
  // mask_byte_len = 3 is not a multiple of 4 — triggers error.
  LlgConstraintStep step = {constraint.get(), mask.data(), 3};
  std::atomic<bool> done{false};

  llg_par_compute_mask(&step, 1, &done, mark_done_callback);
  wait_for_done(done);

  // The constraint should now be in an error state.
  const char *err = llg_get_error(constraint.get());
  BOOST_REQUIRE(err != nullptr);
  BOOST_TEST(std::string(err).find("mask_byte_len") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
