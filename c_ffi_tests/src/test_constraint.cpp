#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "test_helpers.h"
#include "llguidance.h"

namespace {

using TokenizerPtr = std::unique_ptr<LlgTokenizer, decltype(&llg_free_tokenizer)>;
using ConstraintPtr =
    std::unique_ptr<LlgConstraint, decltype(&llg_free_constraint)>;

TokenizerPtr make_byte_tokenizer() {
  return TokenizerPtr(create_byte_tokenizer(), &llg_free_tokenizer);
}

LlgConstraintInit make_constraint_init(const LlgTokenizer *tokenizer,
                                       uint32_t log_buffer_level = 0,
                                       uint32_t log_stderr_level = 0) {
  LlgConstraintInit init{};
  llg_constraint_init_set_defaults(&init, tokenizer);
  init.log_buffer_level = log_buffer_level;
  init.log_stderr_level = log_stderr_level;
  return init;
}

ConstraintPtr make_constraint(LlgConstraint *constraint) {
  BOOST_REQUIRE(constraint != nullptr);
  return ConstraintPtr(constraint, &llg_free_constraint);
}

void require_no_error(const LlgConstraint *constraint) {
  const char *error = llg_get_error(constraint);
  BOOST_REQUIRE_MESSAGE((error == nullptr), (error ? error : "unexpected error"));
}

bool mask_allows(const LlgMaskResult &mask_res, uint32_t token) {
  return mask_res.sample_mask != nullptr &&
         (mask_res.sample_mask[token / 32] & (uint32_t{1} << (token % 32))) != 0;
}

LlgMaskResult compute_mask_ok(LlgConstraint *constraint) {
  LlgMaskResult mask_res{};
  const int32_t rc = llg_compute_mask(constraint, &mask_res);
  const char *error = llg_get_error(constraint);
  BOOST_REQUIRE_MESSAGE((rc == 0), (error ? error : "llg_compute_mask failed"));
  BOOST_REQUIRE_MESSAGE((error == nullptr), (error ? error : "unexpected error"));
  return mask_res;
}

LlgCommitResult commit_token_ok(LlgConstraint *constraint, uint32_t token) {
  LlgCommitResult commit_res{};
  const int32_t rc = llg_commit_token(constraint, token, &commit_res);
  const char *error = llg_get_error(constraint);
  BOOST_REQUIRE_MESSAGE((rc == 0), (error ? error : "llg_commit_token failed"));
  BOOST_REQUIRE_MESSAGE((error == nullptr), (error ? error : "unexpected error"));
  return commit_res;
}

} // namespace

BOOST_AUTO_TEST_SUITE(constraint)

BOOST_AUTO_TEST_CASE(init_set_defaults) {
  auto tokenizer = make_byte_tokenizer();

  LlgConstraintInit init{};
  llg_constraint_init_set_defaults(&init, tokenizer.get());

  BOOST_CHECK_EQUAL(init.ff_tokens_ok, false);
  BOOST_CHECK_EQUAL(init.backtrack_ok, false);
  BOOST_CHECK_EQUAL(init.log_stderr_level, 1U);
  BOOST_CHECK_EQUAL(init.log_buffer_level, 0U);
}

BOOST_AUTO_TEST_CASE(new_constraint_regex_valid) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-z]+"));

  require_no_error(constraint.get());
}

BOOST_AUTO_TEST_CASE(new_constraint_regex_invalid) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[z-a"));

  const char *error = llg_get_error(constraint.get());
  BOOST_REQUIRE(error != nullptr);
  BOOST_CHECK(std::strlen(error) > 0);
}

BOOST_AUTO_TEST_CASE(new_constraint_json_schema) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint = make_constraint(
      llg_new_constraint_json(&init, R"({"type":"string"})"));

  require_no_error(constraint.get());
}

BOOST_AUTO_TEST_CASE(new_constraint_lark) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_lark(&init, R"(start: "hello")"));

  require_no_error(constraint.get());
}

BOOST_AUTO_TEST_CASE(new_constraint_any_regex) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_any(&init, "regex", "[0-9]+"));

  require_no_error(constraint.get());
}

BOOST_AUTO_TEST_CASE(new_constraint_any_invalid_type) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_any(&init, "not-a-type", "data"));

  BOOST_REQUIRE(llg_get_error(constraint.get()) != nullptr);
}

BOOST_AUTO_TEST_CASE(compute_mask_and_commit_regex) {
  constexpr uint32_t tok_a = 'a';
  constexpr uint32_t tok_b = 'b';
  constexpr uint32_t tok_c = 'c';

  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-c]+"));

  require_no_error(constraint.get());

  auto mask_res = compute_mask_ok(constraint.get());
  BOOST_CHECK(!mask_res.is_stop);
  BOOST_REQUIRE(mask_res.sample_mask != nullptr);
  BOOST_CHECK(mask_allows(mask_res, tok_a));
  BOOST_CHECK(mask_allows(mask_res, tok_b));
  BOOST_CHECK(mask_allows(mask_res, tok_c));

  auto commit_res = commit_token_ok(constraint.get(), tok_a);
  BOOST_CHECK_EQUAL(commit_res.n_tokens, 1U);
  BOOST_REQUIRE(commit_res.tokens != nullptr);
  BOOST_CHECK_EQUAL(commit_res.tokens[0], tok_a);

  mask_res = compute_mask_ok(constraint.get());
  BOOST_CHECK(!mask_res.is_stop);
  BOOST_REQUIRE(mask_res.sample_mask != nullptr);
  BOOST_CHECK(mask_allows(mask_res, tok_a));
  BOOST_CHECK(mask_allows(mask_res, tok_b));
  BOOST_CHECK(mask_allows(mask_res, tok_c));
  BOOST_CHECK(mask_allows(mask_res, BYTE_TOK_EOS));

  commit_token_ok(constraint.get(), BYTE_TOK_EOS);

  mask_res = compute_mask_ok(constraint.get());
  BOOST_CHECK(mask_res.is_stop);
}

BOOST_AUTO_TEST_CASE(full_json_sampling_loop) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());

  const auto schema_path =
      std::string(TEST_DATA_DIR) + "/blog.schema.ll.json";
  const auto sample_path = std::string(TEST_DATA_DIR) + "/blog.sample.json";
  const auto schema = read_file(schema_path);
  const auto sample = read_file(sample_path);

  BOOST_REQUIRE(!schema.empty());
  BOOST_REQUIRE(!sample.empty());

  auto constraint =
      make_constraint(llg_new_constraint(&init, schema.c_str()));

  require_no_error(constraint.get());

  const auto tokens = llg_tokenize(tokenizer.get(), sample);
  BOOST_REQUIRE(!tokens.empty());

  for (uint32_t token : tokens) {
    auto mask_res = compute_mask_ok(constraint.get());
    BOOST_CHECK(!mask_res.is_stop);
    BOOST_REQUIRE(mask_res.sample_mask != nullptr);
    BOOST_CHECK(mask_allows(mask_res, token));

    auto commit_res = commit_token_ok(constraint.get(), token);
    BOOST_CHECK_EQUAL(commit_res.n_tokens, 1U);
    BOOST_REQUIRE(commit_res.tokens != nullptr);
    BOOST_CHECK_EQUAL(commit_res.tokens[0], token);
  }

  auto mask_res = compute_mask_ok(constraint.get());
  BOOST_CHECK(mask_res.is_stop);
  BOOST_CHECK(llg_is_stopped(constraint.get()));
}

BOOST_AUTO_TEST_CASE(get_temperature) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-z]+"));

  require_no_error(constraint.get());

  const auto mask_res = compute_mask_ok(constraint.get());
  BOOST_CHECK_EQUAL(llg_get_temperature(constraint.get()), mask_res.temperature);
}

BOOST_AUTO_TEST_CASE(is_stopped_initially_false) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-z]+"));

  require_no_error(constraint.get());
  BOOST_CHECK(!llg_is_stopped(constraint.get()));
}

BOOST_AUTO_TEST_CASE(flush_logs) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get(), 2, 0);
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-z]+"));

  require_no_error(constraint.get());

  compute_mask_ok(constraint.get());
  commit_token_ok(constraint.get(), 'a');

  const char *logs = llg_flush_logs(constraint.get());
  BOOST_REQUIRE(logs != nullptr);
  BOOST_CHECK(std::strlen(logs) > 0);
}

BOOST_AUTO_TEST_CASE(clone_constraint) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto original =
      make_constraint(llg_new_constraint_regex(&init, "ab|bc"));

  require_no_error(original.get());

  auto initial_mask = compute_mask_ok(original.get());
  BOOST_CHECK(mask_allows(initial_mask, 'a'));
  BOOST_CHECK(mask_allows(initial_mask, 'b'));

  auto clone = make_constraint(llg_clone_constraint(original.get()));
  require_no_error(clone.get());

  auto original_commit = commit_token_ok(original.get(), 'a');
  BOOST_CHECK_EQUAL(original_commit.n_tokens, 1U);
  BOOST_CHECK_EQUAL(original_commit.tokens[0], static_cast<uint32_t>('a'));

  auto clone_commit = commit_token_ok(clone.get(), 'b');
  BOOST_CHECK_EQUAL(clone_commit.n_tokens, 1U);
  BOOST_CHECK_EQUAL(clone_commit.tokens[0], static_cast<uint32_t>('b'));

  auto original_mask = compute_mask_ok(original.get());
  BOOST_CHECK(mask_allows(original_mask, 'b'));
  BOOST_CHECK(!mask_allows(original_mask, 'c'));

  auto clone_mask = compute_mask_ok(clone.get());
  BOOST_CHECK(!mask_allows(clone_mask, 'b'));
  BOOST_CHECK(mask_allows(clone_mask, 'c'));
}

BOOST_AUTO_TEST_CASE(free_constraint_with_error) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint = make_constraint(llg_new_constraint(&init, "{"));

  BOOST_REQUIRE(llg_get_error(constraint.get()) != nullptr);
}

BOOST_AUTO_TEST_CASE(compute_mask_on_error_constraint) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint = make_constraint(llg_new_constraint_json(&init, "{"));

  BOOST_REQUIRE(llg_get_error(constraint.get()) != nullptr);

  LlgMaskResult mask_res{};
  BOOST_CHECK_EQUAL(llg_compute_mask(constraint.get(), &mask_res), -1);
}

BOOST_AUTO_TEST_CASE(commit_token_on_error_constraint) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_lark(&init, R"(start: [)"));

  BOOST_REQUIRE(llg_get_error(constraint.get()) != nullptr);

  LlgCommitResult commit_res{};
  BOOST_CHECK_EQUAL(llg_commit_token(constraint.get(), 'a', &commit_res), -1);
}

BOOST_AUTO_TEST_CASE(commit_disallowed_token) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint = make_constraint(llg_new_constraint_regex(&init, "a"));

  require_no_error(constraint.get());

  const auto mask_res = compute_mask_ok(constraint.get());
  BOOST_REQUIRE(mask_res.sample_mask != nullptr);
  BOOST_CHECK(mask_allows(mask_res, 'a'));
  BOOST_CHECK(!mask_allows(mask_res, 'z'));

  LlgCommitResult commit_res{};
  BOOST_CHECK_EQUAL(llg_commit_token(constraint.get(), 'z', &commit_res), -1);
}

BOOST_AUTO_TEST_CASE(double_compute_mask) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get());
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-z]+"));

  require_no_error(constraint.get());

  auto first_mask = compute_mask_ok(constraint.get());
  BOOST_CHECK(first_mask.sample_mask != nullptr);

  LlgMaskResult second_mask{};
  BOOST_CHECK_EQUAL(llg_compute_mask(constraint.get(), &second_mask), 0);
  BOOST_CHECK(llg_get_error(constraint.get()) == nullptr);
  BOOST_CHECK(second_mask.sample_mask != nullptr);
}

BOOST_AUTO_TEST_CASE(flush_logs_returns_valid_c_string_after_no_activity) {
  auto tokenizer = make_byte_tokenizer();
  auto init = make_constraint_init(tokenizer.get(), 2, 0);
  auto constraint =
      make_constraint(llg_new_constraint_regex(&init, "[a-z]+"));

  require_no_error(constraint.get());

  // Flush logs on a fresh constraint with no compute/commit activity.
  const char *logs = llg_flush_logs(constraint.get());
  BOOST_REQUIRE(logs != nullptr);
  // The returned string must be a valid C string (NUL-terminated).
  // It may be empty if no log output was generated.
  BOOST_CHECK(std::strlen(logs) >= 0);
}

BOOST_AUTO_TEST_CASE(free_constraint_null) {
  llg_free_constraint(nullptr);
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END()