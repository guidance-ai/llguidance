#include <boost/test/unit_test.hpp>

#include <cstring>

#include "test_helpers.h"
#include "llguidance.h"

BOOST_AUTO_TEST_SUITE(version)

BOOST_AUTO_TEST_CASE(get_version_non_null) {
  const char *version = llg_get_version();

  BOOST_TEST(version != nullptr);
}

BOOST_AUTO_TEST_CASE(get_version_contains_name) {
  const char *version = llg_get_version();
  BOOST_REQUIRE(version != nullptr);

  BOOST_TEST(std::strstr(version, "llguidance@") != nullptr);
}

BOOST_AUTO_TEST_CASE(get_version_stable_pointer) {
  const char *first = llg_get_version();
  const char *second = llg_get_version();

  BOOST_TEST(first == second);
}

BOOST_AUTO_TEST_SUITE_END()
