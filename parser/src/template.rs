use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::{
    api::{ParserLimits, TopLevelGrammar},
    prepared::PreparedGrammar,
    HashMap, ParserFactory, TokenParser,
};

/// Aggregate input budgets, checked before cloning or compiling a specialization.
/// Duplicate values count toward both budgets. Bytes are UTF-8 input bytes;
/// JSON escaping can expand the bounded input by up to six times.
#[derive(Clone, Debug)]
pub struct StringSlotLimits {
    pub max_values: usize,
    pub max_bytes: usize,
}

impl Default for StringSlotLimits {
    fn default() -> Self {
        Self {
            max_values: 10_000,
            max_bytes: 1_048_576,
        }
    }
}

/// An immutable grammar prepared for one tokenizer and parser configuration.
///
/// Bindings replace declared `string_slot` lexemes with finite JSON-string
/// languages. Omitted slots retain their asset-defined fallback. Empty lists
/// are rejected; they never silently select the fallback. Each parser owns
/// its specialized lexer, so a template can be used concurrently.
pub struct GrammarTemplate {
    prepared: PreparedGrammar,
    slot_limits: StringSlotLimits,
}

impl GrammarTemplate {
    pub fn new(
        factory: Arc<ParserFactory>,
        input: TopLevelGrammar,
        limits: ParserLimits,
        slot_limits: StringSlotLimits,
    ) -> Result<Self> {
        Ok(Self {
            prepared: PreparedGrammar::new(factory, input, limits)?,
            slot_limits,
        })
    }

    /// Declared slot names, sorted for stable capability inspection.
    pub fn slot_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .prepared
            .grammar
            .lexer_spec()
            .string_slots
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    pub fn new_parser(&self, bindings: &HashMap<String, Vec<String>>) -> Result<TokenParser> {
        let mut count = 0usize;
        let mut bytes = 0usize;
        for (name, values) in bindings {
            ensure!(
                self.prepared
                    .grammar
                    .lexer_spec()
                    .string_slots
                    .contains_key(name),
                "unknown string slot {name}"
            );
            ensure!(!values.is_empty(), "empty string slot binding for {name}");
            count = count.saturating_add(values.len());
            ensure!(
                count <= self.slot_limits.max_values,
                "string slot value budget exceeded"
            );
            for value in values {
                bytes = bytes.saturating_add(value.len());
                ensure!(
                    bytes <= self.slot_limits.max_bytes,
                    "string slot byte budget exceeded"
                );
            }
        }
        if bindings.is_empty() {
            return self.prepared.new_parser();
        }
        let grammar = self
            .prepared
            .grammar
            .with_string_bindings(bindings, &self.prepared.limits)?;
        self.prepared.new_parser_with_grammar(Arc::new(grammar))
    }
}
