#pragma once
#include "cxx_gen.h"
#include "llguidance_cxx_support.h"
#include <memory>

namespace llguidance {
  using FactoryInit = ::llguidance::FactoryInit;
  struct ParserFactory;
}

namespace llguidance {
#ifndef CXXBRIDGE1_STRUCT_llguidance$ParserFactory
#define CXXBRIDGE1_STRUCT_llguidance$ParserFactory
struct ParserFactory final : public ::rust::Opaque {
  ~ParserFactory() = delete;

private:
  friend ::rust::layout;
  struct layout {
    static ::std::size_t size() noexcept;
    static ::std::size_t align() noexcept;
  };
};
#endif // CXXBRIDGE1_STRUCT_llguidance$ParserFactory

::rust::Box<::llguidance::ParserFactory> parser_factory(::std::unique_ptr<::llguidance::FactoryInit> tok_init);

// Returns slices applicable for general grammars.
// Currently the same as `json_slices`.
::rust::Vec<::rust::String> general_slices() noexcept;

// Returns slices applicable for JSON schemas.
::rust::Vec<::rust::String> json_slices() noexcept;
} // namespace llguidance
