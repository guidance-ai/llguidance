#include "cxx_gen.h"
#include "llguidance_cxx.h"
#include <memory>

namespace rust {
inline namespace cxxbridge1 {
namespace repr {
struct PtrLen final {
  void *ptr;
  ::std::size_t len;
};
} // namespace repr

namespace detail {
template <typename T, typename = void *>
struct operator_new {
  void *operator()(::std::size_t sz) { return ::operator new(sz); }
};

template <typename T>
struct operator_new<T, decltype(T::operator new(sizeof(T)))> {
  void *operator()(::std::size_t sz) { return T::operator new(sz); }
};
} // namespace detail

template <typename T>
union MaybeUninit {
  T value;
  void *operator new(::std::size_t sz) { return detail::operator_new<T>{}(sz); }
  MaybeUninit() {}
  ~MaybeUninit() {}
};

namespace {
template <>
class impl<Error> final {
public:
  static Error error(repr::PtrLen repr) noexcept {
    Error error;
    error.msg = static_cast<char const *>(repr.ptr);
    error.len = repr.len;
    return error;
  }
};

template <bool> struct deleter_if {
  template <typename T> void operator()(T *) {}
};

template <> struct deleter_if<true> {
  template <typename T> void operator()(T *ptr) { ptr->~T(); }
};
} // namespace
} // namespace cxxbridge1
} // namespace rust

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

extern "C" {
::std::size_t llguidance$cxxbridge1$FactoryInit$vocab_size(::llguidance::FactoryInit const &self) noexcept {
  ::std::size_t (::llguidance::FactoryInit::*vocab_size$)() const = &::llguidance::FactoryInit::vocab_size;
  return (self.*vocab_size$)();
}

::std::uint32_t llguidance$cxxbridge1$FactoryInit$tok_eos(::llguidance::FactoryInit const &self) noexcept {
  ::std::uint32_t (::llguidance::FactoryInit::*tok_eos$)() const = &::llguidance::FactoryInit::tok_eos;
  return (self.*tok_eos$)();
}

void llguidance$cxxbridge1$FactoryInit$token_bytes(::llguidance::FactoryInit const &self, ::std::size_t token, ::rust::Vec<::std::uint8_t> *return$) noexcept {
  ::rust::Vec<::std::uint8_t> (::llguidance::FactoryInit::*token_bytes$)(::std::size_t) const = &::llguidance::FactoryInit::token_bytes;
  new (return$) ::rust::Vec<::std::uint8_t>((self.*token_bytes$)(token));
}

void llguidance$cxxbridge1$FactoryInit$tokenize(::llguidance::FactoryInit const &self, ::rust::Str text, ::rust::Vec<::std::uint32_t> *return$) noexcept {
  ::rust::Vec<::std::uint32_t> (::llguidance::FactoryInit::*tokenize$)(::rust::Str) const = &::llguidance::FactoryInit::tokenize;
  new (return$) ::rust::Vec<::std::uint32_t>((self.*tokenize$)(text));
}

void llguidance$cxxbridge1$FactoryInit$slices(::llguidance::FactoryInit const &self, ::rust::Vec<::rust::String> *return$) noexcept {
  ::rust::Vec<::rust::String> (::llguidance::FactoryInit::*slices$)() const = &::llguidance::FactoryInit::slices;
  new (return$) ::rust::Vec<::rust::String>((self.*slices$)());
}

bool llguidance$cxxbridge1$FactoryInit$allow_ff_tokens(::llguidance::FactoryInit const &self) noexcept {
  bool (::llguidance::FactoryInit::*allow_ff_tokens$)() const = &::llguidance::FactoryInit::allow_ff_tokens;
  return (self.*allow_ff_tokens$)();
}

bool llguidance$cxxbridge1$FactoryInit$allow_backtracking(::llguidance::FactoryInit const &self) noexcept {
  bool (::llguidance::FactoryInit::*allow_backtracking$)() const = &::llguidance::FactoryInit::allow_backtracking;
  return (self.*allow_backtracking$)();
}

::std::uint32_t llguidance$cxxbridge1$FactoryInit$stderr_log_level(::llguidance::FactoryInit const &self) noexcept {
  ::std::uint32_t (::llguidance::FactoryInit::*stderr_log_level$)() const = &::llguidance::FactoryInit::stderr_log_level;
  return (self.*stderr_log_level$)();
}
::std::size_t llguidance$cxxbridge1$ParserFactory$operator$sizeof() noexcept;
::std::size_t llguidance$cxxbridge1$ParserFactory$operator$alignof() noexcept;

::rust::repr::PtrLen llguidance$cxxbridge1$parser_factory(::llguidance::FactoryInit *tok_init, ::rust::Box<::llguidance::ParserFactory> *return$) noexcept;

void llguidance$cxxbridge1$general_slices(::rust::Vec<::rust::String> *return$) noexcept;

void llguidance$cxxbridge1$json_slices(::rust::Vec<::rust::String> *return$) noexcept;
} // extern "C"

::std::size_t ParserFactory::layout::size() noexcept {
  return llguidance$cxxbridge1$ParserFactory$operator$sizeof();
}

::std::size_t ParserFactory::layout::align() noexcept {
  return llguidance$cxxbridge1$ParserFactory$operator$alignof();
}

::rust::Box<::llguidance::ParserFactory> parser_factory(::std::unique_ptr<::llguidance::FactoryInit> tok_init) {
  ::rust::MaybeUninit<::rust::Box<::llguidance::ParserFactory>> return$;
  ::rust::repr::PtrLen error$ = llguidance$cxxbridge1$parser_factory(tok_init.release(), &return$.value);
  if (error$.ptr) {
    throw ::rust::impl<::rust::Error>::error(error$);
  }
  return ::std::move(return$.value);
}

// Returns slices applicable for general grammars.
// Currently the same as `json_slices`.
::rust::Vec<::rust::String> general_slices() noexcept {
  ::rust::MaybeUninit<::rust::Vec<::rust::String>> return$;
  llguidance$cxxbridge1$general_slices(&return$.value);
  return ::std::move(return$.value);
}

// Returns slices applicable for JSON schemas.
::rust::Vec<::rust::String> json_slices() noexcept {
  ::rust::MaybeUninit<::rust::Vec<::rust::String>> return$;
  llguidance$cxxbridge1$json_slices(&return$.value);
  return ::std::move(return$.value);
}
} // namespace llguidance

extern "C" {
static_assert(::rust::detail::is_complete<::llguidance::FactoryInit>::value, "definition of FactoryInit is required");
static_assert(sizeof(::std::unique_ptr<::llguidance::FactoryInit>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::llguidance::FactoryInit>) == alignof(void *), "");
void cxxbridge1$unique_ptr$llguidance$FactoryInit$null(::std::unique_ptr<::llguidance::FactoryInit> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::llguidance::FactoryInit>();
}
void cxxbridge1$unique_ptr$llguidance$FactoryInit$raw(::std::unique_ptr<::llguidance::FactoryInit> *ptr, ::llguidance::FactoryInit *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::llguidance::FactoryInit>(raw);
}
::llguidance::FactoryInit const *cxxbridge1$unique_ptr$llguidance$FactoryInit$get(::std::unique_ptr<::llguidance::FactoryInit> const &ptr) noexcept {
  return ptr.get();
}
::llguidance::FactoryInit *cxxbridge1$unique_ptr$llguidance$FactoryInit$release(::std::unique_ptr<::llguidance::FactoryInit> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$llguidance$FactoryInit$drop(::std::unique_ptr<::llguidance::FactoryInit> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::llguidance::FactoryInit>::value>{}(ptr);
}

::llguidance::ParserFactory *cxxbridge1$box$llguidance$ParserFactory$alloc() noexcept;
void cxxbridge1$box$llguidance$ParserFactory$dealloc(::llguidance::ParserFactory *) noexcept;
void cxxbridge1$box$llguidance$ParserFactory$drop(::rust::Box<::llguidance::ParserFactory> *ptr) noexcept;
} // extern "C"

namespace rust {
inline namespace cxxbridge1 {
template <>
::llguidance::ParserFactory *Box<::llguidance::ParserFactory>::allocation::alloc() noexcept {
  return cxxbridge1$box$llguidance$ParserFactory$alloc();
}
template <>
void Box<::llguidance::ParserFactory>::allocation::dealloc(::llguidance::ParserFactory *ptr) noexcept {
  cxxbridge1$box$llguidance$ParserFactory$dealloc(ptr);
}
template <>
void Box<::llguidance::ParserFactory>::drop() noexcept {
  cxxbridge1$box$llguidance$ParserFactory$drop(this);
}
} // namespace cxxbridge1
} // namespace rust
