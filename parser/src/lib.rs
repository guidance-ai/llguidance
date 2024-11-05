/// This is the primary interface for llguidance -- the one on which the others
/// (FFI and LLInterpreter) are built.  While not cleanest of these interfaces,
/// it is the  most inclusive.
///
/// cbindgen:ignore
pub mod earley;

mod tokenparser;
pub use tokenparser::TokenParser;
pub mod api;
pub mod output;
pub use toktrie;

mod tokenizer_json;
mod constraint;
pub use constraint::{CommitResult, Constraint};

mod logging;
pub use logging::Logger;

pub use derivre;

pub mod ffi;

pub mod lark;

mod grammar_builder;
mod json;
pub use grammar_builder::{GrammarBuilder, NodeRef};
pub use json::JsonCompileOptions;
pub use tokenizer_json::token_bytes_from_tokenizer_json;

#[macro_export]
macro_rules! loginfo {
    ($s:expr, $($arg:tt)*) => {
        if $s.level_enabled(2) {
            use std::fmt::Write;
            writeln!($s.info_logger(), $($arg)*).unwrap();
        }
    };
}

#[macro_export]
macro_rules! infoln {
    ($s:expr, $($arg:tt)*) => {
        if $s.logger.level_enabled(2) {
            use std::fmt::Write;
            writeln!($s.logger.info_logger(), $($arg)*).unwrap();
        }
    };
}

#[macro_export]
macro_rules! warn {
    ($s:expr, $($arg:tt)*) => {
        if $s.logger.level_enabled(1) {
            use std::fmt::Write;
            $s.logger.write_warning("Warning: ");
            writeln!($s.logger.warning_logger(), $($arg)*).unwrap();
        }
    };
}
