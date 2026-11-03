use std::sync::Arc;

use anyhow::Result;

use crate::{
    api::{GrammarInit, ParserLimits, TopLevelGrammar},
    earley::CGrammar,
    Logger, ParserFactory, TokenParser,
};

/// Compiled structure and the configuration it was prepared against.
/// Request state is created independently; preparation does not bind slot values.
pub(crate) struct PreparedGrammar {
    factory: Arc<ParserFactory>,
    pub(crate) grammar: Arc<CGrammar>,
    pub(crate) limits: ParserLimits,
    max_tokens: Option<usize>,
}

impl PreparedGrammar {
    pub(crate) fn new(
        factory: Arc<ParserFactory>,
        input: TopLevelGrammar,
        limits: ParserLimits,
    ) -> Result<Self> {
        let max_tokens = input.max_tokens;
        let grammar = GrammarInit::Serialized(input).to_cgrammar(
            Some(factory.tok_env().clone()),
            &mut Logger::new(factory.buffer_log_level(), factory.stderr_log_level()),
            limits.clone(),
            factory.extra_lexemes(),
        )?;
        Ok(Self {
            factory,
            grammar,
            limits,
            max_tokens,
        })
    }

    pub(crate) fn new_parser(&self) -> Result<TokenParser> {
        self.new_parser_with_grammar(self.grammar.clone())
    }

    /// The replacement must retain this preparation's tokenizer and structural invariants.
    pub(crate) fn new_parser_with_grammar(&self, grammar: Arc<CGrammar>) -> Result<TokenParser> {
        self.factory
            .create_parser_from_compiled(grammar, self.max_tokens, self.limits.clone())
    }
}
