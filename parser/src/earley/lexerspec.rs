use anyhow::{ensure, Result};
use derivre::{raw::ExprSet, ExprRef, HashMap, JsonQuoteOptions, RegexAst, RegexBuilder};
use std::{fmt::Debug, hash::Hash, ops::RangeInclusive};
use toktrie::{bytes::limit_bytes, SimpleVob, TokTrie, TokenId};

use crate::{
    api::{ParserLimits, SkipRepetition, SkipSpec},
    id32_type,
};

use super::{
    int_ranges::IntRanges,
    lexer::MatchingLexemesIdx,
    regexvec::{LexemeSet, MatchingLexemes, RegexVec, RxLexeme},
};

#[derive(Clone)]
pub struct LexerSpec {
    pub lexemes: Vec<LexemeSpec>,
    /// Interval configurations, indexed by lexeme ID.
    int_ranges: Option<Box<HashMap<LexemeIdx, IntRanges>>>,
    pub regex_builder: RegexBuilder,
    pub no_forcing: bool,
    pub allow_initial_skip: bool,
    pub num_extra_lexemes: usize,
    pub skip_by_class: Vec<LexemeIdx>,
    class_by_skip: HashMap<(ExprRef, SkipRepetition), LexemeClass>,
    pub current_class: LexemeClass,
    // regex for \xFF \[ [0-9]+ \]
    pub special_token_rx: Option<ExprRef>,
    pub has_stop: bool,
    pub has_max_tokens: bool,
    pub has_temperature: bool,
    pub grammar_warnings: Vec<(String, usize)>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct LexemeClass(u8);

impl LexemeClass {
    pub const ROOT: LexemeClass = LexemeClass(0);

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
    pub fn new(class: usize) -> Self {
        LexemeClass(class.try_into().expect("class too large"))
    }
}

#[derive(Clone)]
pub struct LexemeSpec {
    pub(crate) idx: LexemeIdx,
    pub(crate) single_set: MatchingLexemes,
    pub(crate) name: String,
    pub(crate) rx: RegexAst,
    class: LexemeClass,
    pub(crate) compiled_rx: ExprRef,
    ends_at_eos: bool,
    lazy: bool,
    contextual: bool,
    max_tokens: usize,
    pub(crate) is_extra: bool,
    pub(crate) is_suffix: bool,
    pub(crate) is_skip: bool,
    pub(crate) skip_repetition: SkipRepetition,
    json_options: Option<JsonQuoteOptions>,
    pub(crate) token_ranges: Vec<RangeInclusive<TokenId>>,
}

// LexemeIdx is an index into the lexeme table.
// It corresponds to a category like IDENTIFIER or STRING,
// or to a very specific lexeme like WHILE or MULTIPLY.
id32_type!(LexemeIdx);

pub fn token_ranges_to_string(token_ranges: &Vec<RangeInclusive<TokenId>>) -> String {
    use std::fmt::Write;
    let mut s = "<[".to_string();
    for range in token_ranges {
        if s.len() > 2 {
            s.push(',');
        }
        if range.start() == range.end() {
            write!(s, "{:?}", range.start()).unwrap();
        } else {
            write!(s, "{:?}-{:?}", range.start(), range.end()).unwrap();
        }
    }
    s.push_str("]>");
    s
}

impl LexemeSpec {
    pub fn class(&self) -> LexemeClass {
        self.class
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn to_string(&self, max_len: usize, exprset: Option<&ExprSet>) -> String {
        use std::fmt::Write;
        let mut f = String::new();
        write!(f, "[{}] {} ", self.idx.0, self.name).unwrap();
        self.rx.write_to_str(&mut f, max_len, exprset);
        if self.lazy {
            f.push_str(" lazy");
        }
        if self.is_suffix {
            f.push_str(" suffix");
        }
        if self.contextual {
            f.push_str(" contextual");
        }
        if self.is_extra {
            f.push_str(" extra");
        }
        if !self.token_ranges.is_empty() {
            write!(f, " tokens={}", token_ranges_to_string(&self.token_ranges)).unwrap();
        }
        // write!(f, " compiled={:?}", self.compiled_rx).unwrap();
        f
    }

    pub fn contains_token(&self, token: TokenId) -> bool {
        self.token_ranges.iter().any(|range| range.contains(&token))
    }

    /// Total number of token ids covered by this spec's `token_ranges`.
    /// Used to prefer the most specific (narrowest) lexeme when a committed
    /// token falls on the boundary of several active token-range specs.
    pub fn token_range_span(&self) -> u64 {
        self.token_ranges
            .iter()
            .map(|r| (*r.end()) as u64 - (*r.start()) as u64 + 1)
            .sum()
    }
}

impl Debug for LexemeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.to_string(512, None);
        f.write_str(&s)
    }
}

impl LexerSpec {
    pub fn new() -> Result<Self> {
        Ok(LexerSpec {
            lexemes: Vec::new(),
            int_ranges: None,
            special_token_rx: None,
            regex_builder: RegexBuilder::new(),
            no_forcing: false,
            allow_initial_skip: false,
            num_extra_lexemes: 0,
            skip_by_class: Vec::new(),
            current_class: LexemeClass(0),
            class_by_skip: HashMap::default(),
            has_stop: false,
            has_max_tokens: false,
            has_temperature: false,
            grammar_warnings: Vec::new(),
        })
    }

    pub fn render_warnings(&self) -> Vec<String> {
        let mut total_len = 0;
        let mut r = vec![];
        for (msg, count) in &self.grammar_warnings {
            let mut s = msg.clone();
            if count > &1 {
                s.push_str(&format!(" ({count} times)"));
            }
            total_len += s.len();
            r.push(s);
            if total_len > 16 * 1024 {
                r.push("...".to_string());
                break;
            }
        }
        r
    }

    pub fn can_rollback(&self) -> bool {
        !self.has_stop && !self.has_max_tokens
    }

    pub fn check_rollback(&self) -> Result<()> {
        ensure!(
            self.can_rollback(),
            "rollback not supported with max_tokens=... or stop=... lexemes; suffix=... is OK"
        );
        Ok(())
    }

    /// Check if the lexeme always matches bytes.
    pub fn has_forced_bytes(&self, lex_spec: &LexemeSpec, bytes: &[u8]) -> bool {
        self.regex_builder
            .exprset()
            .has_simply_forced_bytes(lex_spec.compiled_rx, bytes)
    }

    pub fn setup_lexeme_class(&mut self, skip: RegexAst) -> Result<LexemeClass> {
        self.setup_lexeme_class_with_skip(SkipSpec::unbounded(skip))
    }

    pub fn setup_lexeme_class_with_skip(&mut self, skip: SkipSpec) -> Result<LexemeClass> {
        let SkipSpec {
            regex: skip,
            repetition: skip_repetition,
        } = skip;
        let skip_node = self.regex_builder.mk(&skip)?; // validate first

        if !self.has_max_tokens && !self.has_temperature {
            if let Some(&cls) = self.class_by_skip.get(&(skip_node, skip_repetition)) {
                // re-use existing
                self.current_class = cls;
                return Ok(cls);
            }
        }

        self.current_class = LexemeClass::new(self.skip_by_class.len());
        self.class_by_skip
            .insert((skip_node, skip_repetition), self.current_class);
        self.skip_by_class.push(LexemeIdx(0)); // avoid assert in empty_spec()
        let idx = self
            .add_lexeme_spec(LexemeSpec {
                name: format!("SKIP{}", self.current_class.as_usize()),
                rx: skip,
                is_skip: true,
                skip_repetition,
                ..self.empty_spec()
            })
            .expect("already validated");
        self.skip_by_class.pop();
        self.skip_by_class.push(idx);
        Ok(self.current_class)
    }

    pub fn alloc_lexeme_set(&self) -> LexemeSet {
        LexemeSet::new(self.lexemes.len())
    }

    pub fn alloc_grammar_set(&self) -> SimpleVob {
        SimpleVob::alloc(self.skip_by_class.len())
    }

    pub fn lexeme_set(&self, cond: impl Fn(&LexemeSpec) -> bool) -> LexemeSet {
        let mut v = self.alloc_lexeme_set();
        for (idx, lex) in self.lexemes.iter().enumerate() {
            if cond(lex) {
                v.add(LexemeIdx::new(idx));
            }
        }
        v
    }

    pub fn all_lexemes(&self) -> LexemeSet {
        self.lexeme_set(|_| true)
    }

    pub fn lazy_lexemes(&self) -> LexemeSet {
        self.lexeme_set(|lex| lex.lazy)
    }

    pub fn eos_ending_lexemes(&self) -> LexemeSet {
        self.lexeme_set(|lex| lex.ends_at_eos)
    }

    pub fn token_range_lexemes(&self, possible: &LexemeSet) -> Vec<&LexemeSpec> {
        let mut res = Vec::new();
        for idx in possible.iter() {
            let spec = &self.lexemes[idx.as_usize()];
            if !spec.token_ranges.is_empty() {
                res.push(spec);
            }
        }
        res
    }

    /// The possible lexemes that are REGEX (no explicit token_ranges): their token
    /// set is determined by regex matching (the string / the parametric lexemes),
    /// which the PDA's terminal->token bridge does NOT model. A grammar with any of
    /// these (or int-ranges) is NOT pure-CFG: its mask must come from the lexer-aware
    /// Earley path, not the PDA fast-path.
    pub fn regex_lexemes(&self, possible: &LexemeSet) -> Vec<&LexemeSpec> {
        let mut res = Vec::new();
        for idx in possible.iter() {
            let spec = &self.lexemes[idx.as_usize()];
            if spec.token_ranges.is_empty() {
                res.push(spec);
            }
        }
        res
    }

    /// Whether the grammar's allowed-token set depends on RUNTIME parametric values
    /// that the PDA's terminal->token bridge does not model. Two such sources:
    /// (1) int-range lexemes (the JSON numeric bounds, where the allowed digits
    /// depend on the min/max values); (2) parametric regex lexemes (the bounded
    /// string / the length-limited parametric rules, where the allowed token set
    /// depends on the runtime length bound). A base tokenizer's regex lexemes
    /// (the single-byte matchers, the unbounded max_tokens) do NOT count: they
    /// determine token *recognition*, and the PDA bridge already maps the
    /// grammar's terminals to token IDs (the recognition is orthogonal to the
    /// The grammar's allowed-token set can include tokens the PDA's terminal-to-token
    /// bridge does not model, in which case the PDA path is unsound and the consumer
    /// must use the exact Earley mask. The sources:
    /// (1) int-range lexemes (the JSON numeric bounds, where the allowed digits depend
    ///     on the runtime min/max);
    /// (2) ANY regex lexeme with empty token_ranges (the string / the number / the
    ///     pattern, bounded or unbounded): a ComplexRegex terminal maps to an EMPTY
    ///     bridge entry (terminal_preterminals returns None for the ComplexRegex
    ///     kind, not the displacement bucket), so the PDA mask under-allows its
    ///     tokens. The bound (max_tokens) is irrelevant to the bridge gap — an
    ///     unbounded regex lexeme (the /[0-9]+/ with no length limit) is just as
    ///     unbridgeable as a bounded one. The prior `max_tokens < usize::MAX` clause
    ///     was the bug: it let unbounded regex lexemes through as "pure-CFG",
    ///     engaging the PDA path on a grammar whose bridge has empty entries (the
    ///     under-map).
    /// (3) ANY skip lexeme (the %ignore, the env-default whitespace skip): the
    ///     sampler may emit skipped tokens (the spaces) that are not PDA terminals,
    ///     so the PDA frontier desynchronizes when a skip is committed. The
    ///     env-default SKIP0 counts too — routing such a grammar to the exact CPU
    ///     mask is the safe (firewall) direction (not a PDA path that desynchronizes
    ///     on skipped tokens).
    pub fn has_lexer_augmentation(&self) -> bool {
        if self.int_ranges.is_some() {
            return true;
        }
        self.lexemes.iter().any(|spec| {
            if spec.is_skip {
                return true;
            }
            // A regex lexeme (not a token-range lexeme) with a real regex: the
            // ComplexRegex terminal, the empty bridge entry, the under-map. The bound
            // (max_tokens) is irrelevant to the bridge gap.
            spec.token_ranges.is_empty() && !matches!(spec.rx, RegexAst::NoMatch)
        })
    }

    pub fn is_nullable(&self, idx: LexemeIdx) -> bool {
        if let Some(config) = self.int_ranges.as_ref().and_then(|ranges| ranges.get(&idx)) {
            return config.min_ranges == 0;
        }
        self.regex_builder
            .is_nullable(self.lexemes[idx.as_usize()].compiled_rx)
    }

    pub fn to_regex_vec(&self, limits: &mut ParserLimits) -> Result<RegexVec> {
        // TODO
        // Find all non-contextual lexemes that are literals (we call them 'keywords')
        // This assumes that this is the only possible conflict in the lexer that we want to catch.
        // For every non literals lexeme, find all keywords that match it.
        // Replace the regex R for the lexeme with (R & ~(K1|K2|...)) where K1...
        // are the conflicting keywords.
        let rx_list: Vec<_> = self
            .lexemes
            .iter()
            .map(|lex| RxLexeme {
                rx: lex.compiled_rx,
                priority: 0,
                lazy: lex.lazy,
            })
            .collect();
        RegexVec::new_with_exprset(
            self.regex_builder.exprset().clone(),
            rx_list,
            self.special_token_rx,
            self.int_ranges.as_deref(),
            limits,
        )
    }

    /// Registers a regex lexeme.
    fn add_lexeme_spec(&mut self, spec: LexemeSpec) -> Result<LexemeIdx> {
        self.add_lexeme_spec_inner(spec, None)
    }

    /// Deduplicates lexemes by both their regex and optional interval configuration.
    fn add_lexeme_spec_inner(
        &mut self,
        mut spec: LexemeSpec,
        int_ranges: Option<IntRanges>,
    ) -> Result<LexemeIdx> {
        let compiled = if !spec.token_ranges.is_empty() {
            if let Some(rx) = self.special_token_rx {
                rx
            } else {
                let rx_ast = RegexAst::Concat(vec![
                    RegexAst::Byte(TokTrie::SPECIAL_TOKEN_MARKER),
                    RegexAst::Regex(r"\[[0-9]+\]".to_string()),
                ]);
                let compiled = self.regex_builder.mk(&rx_ast)?;
                self.special_token_rx = Some(compiled);
                compiled
            }
        } else {
            self.regex_builder.mk(&spec.rx)?
        };

        if !self.has_stop && !spec.is_suffix {
            self.has_stop = match &spec.rx {
                RegexAst::Concat(parts) => parts
                    .iter()
                    .any(|part| matches!(part, RegexAst::LookAhead(_))),
                _ => false,
            };
            if spec.ends_at_eos {
                self.has_stop = true;
            }
        }

        if spec.max_tokens < usize::MAX {
            self.has_max_tokens = true;
        }

        let compiled = if let Some(ref opts) = spec.json_options {
            self.regex_builder.json_quote(compiled, opts)?
        } else {
            compiled
        };

        if let Some(idx) = self.lexemes.iter().position(|lex| {
            lex.compiled_rx == compiled
                && lex.class == spec.class
                && lex.max_tokens == spec.max_tokens
                && lex.token_ranges == spec.token_ranges
                && lex.is_extra == spec.is_extra
                && lex.is_skip == spec.is_skip
                && lex.skip_repetition == spec.skip_repetition
                && self
                    .int_ranges
                    .as_ref()
                    .and_then(|ranges| ranges.get(&lex.idx))
                    == int_ranges.as_ref()
        }) {
            return Ok(LexemeIdx::new(idx));
        }
        let idx = LexemeIdx::new(self.lexemes.len());
        spec.idx = idx;
        spec.single_set = MatchingLexemes::One(idx);
        spec.compiled_rx = compiled;
        if spec.name.is_empty() {
            spec.name = format!("[{}]", idx.as_usize());
        }
        if let Some(config) = int_ranges {
            self.int_ranges
                .get_or_insert_with(Default::default)
                .insert(idx, config);
        }
        self.lexemes.push(spec);
        Ok(idx)
    }

    fn empty_spec(&self) -> LexemeSpec {
        assert!(
            !self.skip_by_class.is_empty(),
            "new_lexeme_class() not called"
        );
        LexemeSpec {
            idx: LexemeIdx(0),
            single_set: MatchingLexemes::One(LexemeIdx(0)),
            name: "".to_string(),
            rx: RegexAst::NoMatch,
            compiled_rx: ExprRef::INVALID,
            lazy: false,
            contextual: false,
            ends_at_eos: false,
            is_skip: false,
            skip_repetition: SkipRepetition::Unbounded,
            is_suffix: false,
            is_extra: false,
            json_options: None,
            class: self.current_class,
            max_tokens: usize::MAX,
            token_ranges: vec![],
        }
    }

    /// Registers a dynamic interval lexeme. Its placeholder regex is never
    /// scanned; the regex vector selects the specialized matcher by lexeme ID.
    #[cfg(feature = "lark")]
    pub(crate) fn add_int_ranges(&mut self, config: IntRanges) -> Result<LexemeIdx> {
        self.add_lexeme_spec_inner(
            LexemeSpec {
                name: "%int_ranges".to_string(),
                ..self.empty_spec()
            },
            Some(config),
        )
    }

    pub fn add_rx_and_stop(
        &mut self,
        name: String,
        body_rx: RegexAst,
        stop_rx: RegexAst,
        lazy: bool,
        max_tokens: usize,
        is_suffix: bool,
    ) -> Result<LexemeIdx> {
        let rx = if !matches!(stop_rx, RegexAst::EmptyString) {
            RegexAst::Concat(vec![body_rx, RegexAst::LookAhead(Box::new(stop_rx))])
        } else {
            body_rx
        };
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx,
            lazy,
            ends_at_eos: !lazy,
            max_tokens,
            is_suffix,
            ..self.empty_spec()
        })
    }

    pub fn add_simple_literal(
        &mut self,
        name: String,
        literal: &str,
        contextual: bool,
    ) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx: RegexAst::Literal(literal.to_string()),
            contextual,
            ..self.empty_spec()
        })
    }

    pub fn add_special_token(
        &mut self,
        name: String,
        token_ranges: Vec<RangeInclusive<TokenId>>,
    ) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            token_ranges,
            ..self.empty_spec()
        })
    }

    pub fn add_greedy_lexeme(
        &mut self,
        name: String,
        rx: RegexAst,
        contextual: bool,
        json_options: Option<JsonQuoteOptions>,
        max_tokens: usize,
    ) -> Result<LexemeIdx> {
        self.add_lexeme_spec(LexemeSpec {
            name,
            rx,
            contextual,
            json_options,
            max_tokens,
            ..self.empty_spec()
        })
    }

    pub fn add_extra_lexemes(&mut self, extra_lexemes: &[String]) {
        assert!(self.num_extra_lexemes == 0);
        self.num_extra_lexemes = extra_lexemes.len();
        let lex0 = self.lexemes.len();
        for (idx, added) in extra_lexemes.iter().enumerate() {
            self.add_lexeme_spec(LexemeSpec {
                name: format!("$extra_{idx}"),
                rx: RegexAst::Regex(added.clone()),
                is_extra: true,
                ..self.empty_spec()
            })
            .expect("adding lexeme");
        }
        assert!(
            self.lexemes.len() - lex0 == self.num_extra_lexemes,
            "repeating slices?"
        );
    }

    pub fn extra_lexeme(&self, idx: usize) -> LexemeIdx {
        assert!(idx < self.num_extra_lexemes);
        self.lexemes[self.lexemes.len() - self.num_extra_lexemes + idx].idx
    }

    pub fn dbg_lexeme_set(&self, vob: &LexemeSet) -> String {
        format!(
            "Lexemes( {} )",
            vob.iter()
                .map(|idx| format!("[{}]", idx.as_usize()))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    pub fn lexeme_spec(&self, idx: LexemeIdx) -> &LexemeSpec {
        &self.lexemes[idx.as_usize()]
    }

    pub fn cost(&self) -> u64 {
        self.regex_builder.exprset().cost()
    }

    pub fn skip_id(&self, class: LexemeClass) -> LexemeIdx {
        self.skip_by_class[class.as_usize()]
    }

    pub fn lexeme_def_to_string(&self, idx: LexemeIdx) -> String {
        use std::fmt::Write;
        let mut text =
            self.lexemes[idx.as_usize()].to_string(512, Some(self.regex_builder.exprset()));
        if let Some(config) = self.int_ranges.as_ref().and_then(|ranges| ranges.get(&idx)) {
            write!(text, " %int_ranges {config:?}").unwrap();
        }
        text
    }

    pub fn dbg_lexeme_set_ext(&self, vob: &SimpleVob) -> String {
        format!(
            "LexemesExt(\n    {}\n)",
            vob.iter()
                .map(|idx| self.lexeme_def_to_string(LexemeIdx::new(idx as usize)))
                .collect::<Vec<_>>()
                .join("\n    ")
        )
    }
}

impl Debug for LexerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LexerSpec {{ lexemes: [")?;
        for lex in &self.lexemes {
            let slex = self.lexeme_def_to_string(lex.idx);
            writeln!(f, "  {slex}")?;
        }
        write!(
            f,
            "]{}{} }}",
            if self.has_stop { " has_stop" } else { "" },
            if self.has_max_tokens {
                " has_max_tokens"
            } else {
                ""
            }
        )
    }
}

#[derive(Clone)]
pub struct Lexeme {
    pub idx: MatchingLexemesIdx,
    bytes: Vec<u8>,
    hidden_len: u32,
    is_suffix: bool,
}

impl Debug for Lexeme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Lexeme({:?}, {:?} + {:?}{})",
            self.idx,
            limit_bytes(self.visible_bytes(), 100),
            limit_bytes(self.hidden_bytes(), 100),
            if self.is_suffix { " suffix" } else { "" }
        )
    }
}

impl Lexeme {
    pub fn new(idx: MatchingLexemesIdx, bytes: Vec<u8>, hidden_len: u32, is_suffix: bool) -> Self {
        Lexeme {
            idx,
            bytes,
            hidden_len,
            is_suffix,
        }
    }

    pub fn just_idx(idx: MatchingLexemesIdx) -> Self {
        Lexeme {
            idx,
            hidden_len: 0,
            bytes: Vec::new(),
            is_suffix: false,
        }
    }

    pub fn single_idx(idx: LexemeIdx) -> Self {
        Self::just_idx(MatchingLexemesIdx::Single(idx))
    }

    pub fn bogus() -> Self {
        Self::single_idx(LexemeIdx(0))
    }

    pub fn is_bogus(&self) -> bool {
        self.bytes.is_empty() && matches!(self.idx, MatchingLexemesIdx::Single(LexemeIdx(0)))
    }

    pub fn is_suffix(&self) -> bool {
        self.is_suffix
    }

    #[inline(always)]
    pub fn num_hidden_bytes(&self) -> usize {
        self.hidden_len as usize
    }

    pub fn num_visible_bytes(&self) -> usize {
        self.bytes.len() - self.num_hidden_bytes()
    }

    pub fn visible_bytes(&self) -> &[u8] {
        &self.bytes[0..self.num_visible_bytes()]
    }

    pub fn upper_visible_bytes(&self, is_lexeme: bool) -> &[u8] {
        if is_lexeme || !self.is_suffix {
            self.visible_bytes()
        } else {
            self.all_bytes()
        }
    }

    pub fn hidden_bytes(&self) -> &[u8] {
        &self.bytes[self.num_visible_bytes()..]
    }

    pub fn all_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with_ranges(ranges: Vec<RangeInclusive<TokenId>>) -> LexemeSpec {
        LexemeSpec {
            idx: LexemeIdx(0),
            single_set: MatchingLexemes::None,
            name: String::new(),
            rx: RegexAst::NoMatch,
            class: LexemeClass(0),
            compiled_rx: ExprRef::INVALID,
            ends_at_eos: false,
            lazy: false,
            contextual: false,
            max_tokens: usize::MAX,
            is_extra: false,
            is_suffix: false,
            is_skip: false,
            skip_repetition: SkipRepetition::Unbounded,
            json_options: None,
            token_ranges: ranges,
        }
    }

    #[test]
    fn test_token_range_span_narrow_vs_broad() {
        // A single-token range (the EOR marker) has span 1.
        let narrow = spec_with_ranges(vec![5431..=5431]);
        assert_eq!(narrow.token_range_span(), 1);

        // A broad range (the reasoning free-text set) has a larger span.
        let broad = spec_with_ranges(vec![5426..=5432]);
        assert_eq!(broad.token_range_span(), 7);

        // The narrow spec must sort before the broad one — this is the ordering
        // the boundary fix relies on (most the most specific lexeme).
        assert!(narrow.token_range_span() < broad.token_range_span());

        // Multi-range span is the sum of the individual ranges.
        let multi = spec_with_ranges(vec![10..=12, 20..=21]);
        assert_eq!(multi.token_range_span(), 3 + 2);
    }

    /// A regex lexeme (no token_ranges, a real rx) — the ComplexRegex
    /// terminal, the empty PDA bridge entry. Bounded or unbounded, it makes the
    /// grammar lexer-augmented (the PDA path must stay off; the CPU Earley mask
    /// is authoritative).
    fn spec_with_regex(rx: RegexAst, max_tokens: usize) -> LexemeSpec {
        LexemeSpec {
            idx: LexemeIdx(0),
            single_set: MatchingLexemes::None,
            name: String::new(),
            rx,
            class: LexemeClass(0),
            compiled_rx: ExprRef::INVALID,
            ends_at_eos: false,
            lazy: false,
            contextual: false,
            max_tokens,
            is_extra: false,
            is_suffix: false,
            is_skip: false,
            skip_repetition: SkipRepetition::Unbounded,
            json_options: None,
            token_ranges: Vec::new(), // the regex lexeme (no token ranges)
        }
    }

    fn spec_with_skip() -> LexemeSpec {
        let mut s = spec_with_ranges(Vec::new());
        s.is_skip = true;
        s
    }

    /// The has_lexer_augmentation decision table (the R8 bridge-coverage gate):
    /// true iff the grammar has a token set the PDA bridge cannot model exactly.
    /// The inclusive: a regex lexeme (bounded or unbounded), a skip lexeme, or an
    /// int-range makes it augmented. The exclusive: a pure token-range grammar
    /// (no regex/skip/int-range) is NOT augmented (the PDA bridge is exact).
    #[test]
    fn has_lexer_augmentation_decision_table() {
        use crate::earley::lexerspec::LexerSpec;

        // The exclusive: a pure token-range grammar (the <[97]> form) is NOT
        // augmented — the PDA bridge (the token-range lift) is exact for it.
        let pure = LexerSpec::new().unwrap();
        let mut pure_spec = pure;
        pure_spec.lexemes.push(spec_with_ranges(vec![97..=97]));
        assert!(
            !pure_spec.has_lexer_augmentation(),
            "a pure token-range grammar is not lexer-augmented (the PDA bridge is exact)"
        );

        // The inclusive: an UNBOUNDED regex lexeme (the /[0-9]+/ with no length
        // limit, max_tokens == usize::MAX) IS augmented. This is the case the
        // prior `max_tokens < usize::MAX` clause missed (it let unbounded regexes
        // through as "pure-CFG", engaging the PDA path on an empty bridge entry).
        let unbounded_rx = LexerSpec::new().unwrap();
        let mut unbounded_spec = unbounded_rx;
        unbounded_spec.lexemes.push(spec_with_regex(
            RegexAst::Regex("[0-9]+".to_string()),
            usize::MAX, // the unbounded (no length limit)
        ));
        assert!(
            unbounded_spec.has_lexer_augmentation(),
            "an unbounded regex lexeme IS lexer-augmented (the bridge entry is empty, so the PDA path is off)"
        );

        // The inclusive: a BOUNDED regex lexeme (the length-limited string) is
        // augmented (the prior clause caught this; the regression guard).
        let bounded_rx = LexerSpec::new().unwrap();
        let mut bounded_spec = bounded_rx;
        bounded_spec.lexemes.push(spec_with_regex(
            RegexAst::Regex("a{1,3}".to_string()),
            3, // the bounded (the maxLength 3)
        ));
        assert!(
            bounded_spec.has_lexer_augmentation(),
            "a bounded regex lexeme is lexer-augmented"
        );

        // The inclusive: a skip lexeme (the %ignore) is augmented (the sampler may
        // emit the skipped tokens, which are not PDA terminals).
        let skip_spec = LexerSpec::new().unwrap();
        let mut with_skip = skip_spec;
        with_skip.lexemes.push(spec_with_skip());
        assert!(
            with_skip.has_lexer_augmentation(),
            "a skip lexeme is lexer-augmented (the skipped tokens are not PDA terminals)"
        );

        // The mixed: a pure token-range + a regex lexeme is augmented (the regex
        // dominates).
        let mixed = LexerSpec::new().unwrap();
        let mut mixed_spec = mixed;
        mixed_spec.lexemes.push(spec_with_ranges(vec![97..=97]));
        mixed_spec.lexemes.push(spec_with_regex(RegexAst::Regex("x+".to_string()), usize::MAX));
        assert!(
            mixed_spec.has_lexer_augmentation(),
            "a grammar with any regex lexeme is augmented (the PDA path is off)"
        );
    }
}
