#include <boost/test/unit_test.hpp>

#include <array>
#include <memory>

#include "test_helpers.h"
#include "llguidance.h"

namespace {

struct TokenizerDeleter {
  void operator()(LlgTokenizer *tok) const {
    if (tok) llg_free_tokenizer(tok);
  }
};

using TokenizerPtr = std::unique_ptr<LlgTokenizer, TokenizerDeleter>;

struct ValidationResult {
  int32_t rc;
  std::array<char, 512> message;
};

TokenizerPtr make_byte_tokenizer() {
  return TokenizerPtr(create_byte_tokenizer());
}

LlgConstraintInit make_constraint_init(const LlgTokenizer *tokenizer) {
  LlgConstraintInit init{};
  llg_constraint_init_set_defaults(&init, tokenizer);
  init.log_stderr_level = 0;
  return init;
}

ValidationResult do_validate(const LlgConstraintInit &init,
                                  const char *constraint_type,
                                  const char *data) {
  ValidationResult result{};
  result.message.fill('\0');
  result.rc = llg_validate_grammar(&init, constraint_type, data,
                                   result.message.data(), result.message.size());
  return result;
}

} // namespace

BOOST_AUTO_TEST_SUITE(validate_grammar)

BOOST_AUTO_TEST_CASE(validate_regex_valid) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result = do_validate(init, "regex", "[a-z]+");

  BOOST_TEST(result.rc == 0);
  BOOST_TEST(result.message[0] == '\0');
}

BOOST_AUTO_TEST_CASE(validate_regex_invalid) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result = do_validate(init, "regex", "[z-a");

  BOOST_TEST(result.rc == -1);
  BOOST_TEST(result.message[0] != '\0');
}

BOOST_AUTO_TEST_CASE(validate_json_schema_valid) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result =
      do_validate(init, "json_schema", R"({"type":"string"})");

  BOOST_TEST(result.rc == 0);
  BOOST_TEST(result.message[0] == '\0');
}

BOOST_AUTO_TEST_CASE(validate_lark_valid) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result = do_validate(init, "lark", R"(start: "hello")");

  BOOST_TEST(result.rc == 0);
  BOOST_TEST(result.message[0] == '\0');
}

BOOST_AUTO_TEST_CASE(validate_lark_invalid) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result = do_validate(init, "lark", "start: (");

  BOOST_TEST(result.rc == -1);
  BOOST_TEST(result.message[0] != '\0');
}

BOOST_AUTO_TEST_CASE(validate_llguidance_type) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result = do_validate(
      init, "llguidance",
      R"({"grammars":[{"json_schema":{"type":"string"}}]})");

  BOOST_TEST(result.rc == 0);
  BOOST_TEST(result.message[0] == '\0');
}

BOOST_AUTO_TEST_CASE(validate_unknown_type) {
  auto tokenizer = make_byte_tokenizer();
  BOOST_REQUIRE(tokenizer != nullptr);

  const auto init = make_constraint_init(tokenizer.get());
  const auto result =
      do_validate(init, "not-a-type", R"({"type":"string"})");

  BOOST_TEST(result.rc == -1);
  BOOST_TEST(result.message[0] != '\0');
}

BOOST_AUTO_TEST_SUITE_END()
