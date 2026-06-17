cmake_minimum_required(VERSION 3.16)
project(c_ffi_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost.Test
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

# Path to the llguidance static library and header
set(LLG_ROOT "${CMAKE_SOURCE_DIR}/..")
set(LLG_INCLUDE_DIR "${LLG_ROOT}/parser")
set(LLG_LIB_DIR "${LLG_ROOT}/target/release")

# Locate the static library
find_library(LLG_LIBRARY
    NAMES llguidance
    PATHS "${LLG_LIB_DIR}"
    NO_DEFAULT_PATH
)
if(NOT LLG_LIBRARY)
    message(FATAL_ERROR "libllguidance not found in ${LLG_LIB_DIR}. Run 'cargo build --release' in the parser directory first.")
endif()

# Test data directory
set(TEST_DATA_DIR "${LLG_ROOT}/sample_parser/data")
add_compile_definitions(TEST_DATA_DIR="${TEST_DATA_DIR}")

# Collect test sources
set(TEST_SOURCES
    src/test_helpers.cpp
    src/test_tokenizer.cpp
    src/test_tokenize.cpp
    src/test_constraint.cpp
    src/test_matcher.cpp
    src/test_stop_controller.cpp
    src/test_parallel.cpp
    src/test_validate_grammar.cpp
    src/test_version.cpp
)

add_executable(llg_ffi_tests ${TEST_SOURCES})

target_include_directories(llg_ffi_tests PRIVATE
    ${LLG_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    src
)

target_link_libraries(llg_ffi_tests PRIVATE
    ${LLG_LIBRARY}
    Boost::unit_test_framework
    pthread
    dl
    m
)

# Enable CTest
enable_testing()
add_test(NAME llg_ffi_tests COMMAND llg_ffi_tests)
