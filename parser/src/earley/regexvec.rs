/// This file implements regex vectors.  To match tokens to lexemes, llguidance uses
/// a DFA whose nodes are regex vectors.  For more on this see
/// S. Owens, J. Reppy, and A. Turon.
/// Regular Expression Derivatives Reexamined".
/// Journal of Functional Programming 19(2):173-190, March 2009.
/// <https://www.khoury.northeastern.edu/home/turon/re-deriv.pdf> (retrieved 15 Nov 2024)
use anyhow::{bail, Result};
use derivre::raw::{DerivCache, ExprSet, NextByteCache, RelevanceCache, VecHashCons};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use toktrie::SimpleVob;

use derivre::HashMap;
pub use derivre::{AlphabetInfo, ExprRef, NextByte, StateID};

use crate::api::ParserLimits;

use super::{
    int_ranges::{IntRanges, IntRangesMatcher},
    lexerspec::LexemeIdx,
};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct LexerStats {
    pub num_regexps: usize,
    pub num_ast_nodes: usize,
    pub num_derived: usize,
    pub num_derivatives: usize,
    pub total_fuel_spent: usize,
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_bytes: usize,
    pub alphabet_size: usize,
    pub error: bool,
}

#[derive(Clone)]
pub enum MatchingLexemes {
    None,
    One(LexemeIdx),
    Two([LexemeIdx; 2]),
    Many(Vec<LexemeIdx>),
}

impl MatchingLexemes {
    pub fn is_some(&self) -> bool {
        !matches!(self, MatchingLexemes::None)
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    pub fn first(&self) -> Option<LexemeIdx> {
        match self {
            MatchingLexemes::None => None,
            MatchingLexemes::One(idx) => Some(*idx),
            MatchingLexemes::Two([idx, _]) => Some(*idx),
            MatchingLexemes::Many(v) => v.first().copied(),
        }
    }

    pub fn contains(&self, idx: LexemeIdx) -> bool {
        match self {
            MatchingLexemes::None => false,
            MatchingLexemes::One(idx2) => *idx2 == idx,
            MatchingLexemes::Two([idx1, idx2]) => *idx1 == idx || *idx2 == idx,
            MatchingLexemes::Many(v) => v.contains(&idx),
        }
    }

    pub fn add(&mut self, idx: LexemeIdx) {
        match self {
            MatchingLexemes::None => *self = MatchingLexemes::One(idx),
            MatchingLexemes::One(idx2) => {
                *self = MatchingLexemes::Two([*idx2, idx]);
            }
            MatchingLexemes::Two([idx1, idx2]) => {
                *self = MatchingLexemes::Many(vec![*idx1, *idx2, idx]);
            }
            MatchingLexemes::Many(v) => {
                v.push(idx);
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MatchingLexemes::None => 0,
            MatchingLexemes::One(_) => 1,
            MatchingLexemes::Two(_) => 2,
            MatchingLexemes::Many(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[LexemeIdx] {
        match self {
            MatchingLexemes::None => &[],
            MatchingLexemes::One(idx) => std::slice::from_ref(idx),
            MatchingLexemes::Two(v) => v,
            MatchingLexemes::Many(v) => v.as_slice(),
        }
    }
}

impl Debug for MatchingLexemes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchingLexemes::None => write!(f, "Lex:[]"),
            MatchingLexemes::One(idx) => write!(f, "Lex:[{}]", idx.as_usize()),
            MatchingLexemes::Two([idx1, idx2]) => {
                write!(f, "Lex:[{},{}]", idx1.as_usize(), idx2.as_usize())
            }
            MatchingLexemes::Many(v) => write!(
                f,
                "Lex:[{}]",
                v.iter()
                    .map(|idx| idx.as_usize().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

impl Display for LexerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "regexps: {} with {} nodes (+ {} derived via {} derivatives with total fuel {}), states: {}; transitions: {}; bytes: {}; alphabet size: {} {}",
            self.num_regexps,
            self.num_ast_nodes,
            self.num_derived,
            self.num_derivatives,
            self.total_fuel_spent,
            self.num_states,
            self.num_transitions,
            self.num_bytes,
            self.alphabet_size,
            if self.error { "ERROR" } else { "" }
        )
    }
}

impl Debug for LexerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone, Debug)]
pub struct LexemeSet {
    vob: SimpleVob,
}

impl LexemeSet {
    pub fn new(size: usize) -> Self {
        LexemeSet {
            vob: SimpleVob::alloc(size),
        }
    }

    pub fn len(&self) -> usize {
        self.vob.len()
    }

    pub fn from_vob(vob: &SimpleVob) -> Self {
        LexemeSet { vob: vob.clone() }
    }

    pub fn is_empty(&self) -> bool {
        self.vob.is_zero()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = LexemeIdx> + '_ {
        self.vob.iter().map(|e| LexemeIdx::new(e as usize))
    }

    pub fn add(&mut self, idx: LexemeIdx) {
        self.vob.set(idx.as_usize(), true);
    }

    pub fn remove(&mut self, idx: LexemeIdx) {
        self.vob.set(idx.as_usize(), false);
    }

    pub fn first(&self) -> Option<LexemeIdx> {
        self.vob.first_bit_set().map(LexemeIdx::new)
    }

    pub fn contains(&self, idx: LexemeIdx) -> bool {
        self.vob.get(idx.as_usize())
    }

    pub fn clear(&mut self) {
        self.vob.set_all(false);
    }
}

#[derive(Clone)]
pub struct RegexVec {
    exprs: ExprSet,
    deriv: DerivCache,
    next_byte: NextByteCache,
    relevance: RelevanceCache,
    alpha: AlphabetInfo,
    #[allow(dead_code)]
    rx_lexemes: Vec<RxLexeme>,
    lazy: LexemeSet,
    subsumable: LexemeSet,
    rx_list: Vec<ExprRef>,
    /// Iterative interval matchers. The state vector stores their interned position
    /// IDs in place of regex expression IDs, so rollback, cloning and mask caches
    /// include the dynamic bounds.
    int_ranges: Option<Box<IntRangesLexers>>,
    special_token_rx: Option<ExprRef>,
    rx_sets: VecHashCons,
    state_table: Vec<StateID>,
    state_descs: Vec<StateDesc>,
    num_transitions: usize,
    num_ast_nodes: usize,
    max_states: usize,
    fuel: u64,
}

#[derive(Clone, Debug)]
pub struct StateDesc {
    pub state: StateID,
    pub greedy_accepting: MatchingLexemes,
    pub possible: LexemeSet,

    /// Index of lowest matching regex if any.
    /// Lazy regexes match as soon as they accept, while greedy only
    /// if they accept and force EOI.
    pub lazy_accepting: MatchingLexemes,
    pub lazy_hidden_len: u32,

    pub has_special_token: bool,

    possible_lookahead_len: Option<usize>,
    lookahead_len: Option<Option<usize>>,
    next_byte: Option<NextByte>,
}

// public implementation
impl RegexVec {
    pub fn alpha(&self) -> &AlphabetInfo {
        &self.alpha
    }

    pub fn lazy_regexes(&self) -> &LexemeSet {
        &self.lazy
    }

    /// Create and return the initial state of a DFA for this
    /// regex vector
    pub fn initial_state(&mut self, selected: &LexemeSet) -> StateID {
        let mut vec_desc = vec![];
        for idx in selected.iter() {
            if self
                .int_ranges
                .as_ref()
                .is_some_and(|ranges| ranges.matchers.contains_key(&idx))
            {
                Self::push_state(&mut vec_desc, idx, 0);
                continue;
            }
            let rx = self.get_rx(idx);
            if rx != ExprRef::NO_MATCH {
                Self::push_state(&mut vec_desc, idx, rx.as_u32());
            }
        }
        self.insert_state(vec_desc)
    }

    #[inline(always)]
    pub fn state_desc(&self, state: StateID) -> &StateDesc {
        &self.state_descs[state.as_usize()]
    }

    pub fn possible_lookahead_len(&mut self, state: StateID) -> usize {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(len) = desc.possible_lookahead_len {
            return len;
        }
        let mut max_len = 0;
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if self
                .int_ranges
                .as_ref()
                .is_none_or(|ranges| !ranges.matchers.contains_key(&idx))
            {
                max_len = max_len.max(self.exprs.possible_lookahead_len(ExprRef::new(e)));
            }
        }
        desc.possible_lookahead_len = Some(max_len);
        max_len
    }

    pub fn lookahead_len_for_state(&mut self, state: StateID) -> Option<usize> {
        let desc = &mut self.state_descs[state.as_usize()];
        if desc.greedy_accepting.is_none() {
            return None;
        }
        if let Some(len) = desc.lookahead_len {
            return len;
        }
        let mut res = None;
        let exprs = &self.exprs;
        for (idx2, e) in iter_state(&self.rx_sets, state) {
            let is_dynamic = self
                .int_ranges
                .as_ref()
                .is_some_and(|ranges| ranges.matchers.contains_key(&idx2));
            // Preserve the constant-time nullable check for ordinary lexemes;
            // searching the accepting set here can be quadratic in its size.
            let accepting = if is_dynamic {
                desc.greedy_accepting.contains(idx2)
            } else {
                exprs.is_nullable(ExprRef::new(e))
            };
            if accepting {
                assert!(desc.greedy_accepting.contains(idx2));
                res = Some(if is_dynamic {
                    0
                } else {
                    exprs.lookahead_len(ExprRef::new(e)).unwrap_or(0)
                });
                break;
            }
        }
        desc.lookahead_len = Some(res);
        res
    }

    /// Given a transition (a from-state and a byte) of the DFA
    /// for this regex vector, return the to-state.  It is taken
    /// from the cache, if it is cached, and created otherwise.
    #[inline(always)]
    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        self.transition_with_cancellation(state, b, &None)
    }

    #[inline(always)]
    pub(crate) fn transition_with_cancellation(
        &mut self,
        state: StateID,
        b: u8,
        cancellation: &Option<crate::CancellationHandle>,
    ) -> StateID {
        let idx = self.alpha.map_state(state, b);
        let new_state = self.state_table[idx];
        if new_state != StateID::MISSING {
            new_state
        } else if self.int_ranges.is_none() {
            self.transition_inner::<false>(state, b, idx, cancellation)
        } else {
            self.transition_inner::<true>(state, b, idx, cancellation)
        }
    }

    /// "Subsumption" is a feature implementing regex containment.
    /// subsume_possible() returns true if it's possible for this
    /// state, false otherwise.
    pub fn subsume_possible(&mut self, state: StateID) -> bool {
        if state.is_dead() || self.has_error() {
            return false;
        }
        for (idx, _) in iter_state(&self.rx_sets, state) {
            if self.lazy.contains(idx)
                || self
                    .int_ranges
                    .as_ref()
                    .is_some_and(|ranges| ranges.matchers.contains_key(&idx))
            {
                return false;
            }
        }
        true
    }

    /// Part of the interface for "subsumption", a feature implementing
    /// regex containment.
    pub fn check_subsume(
        &mut self,
        state: StateID,
        lexeme_idx: LexemeIdx,
        budget: u64,
    ) -> Result<bool> {
        self.check_subsume_with_cancellation(state, lexeme_idx, budget, None)
    }

    pub(crate) fn check_subsume_with_cancellation(
        &mut self,
        state: StateID,
        lexeme_idx: LexemeIdx,
        mut budget: u64,
        cancellation: Option<&crate::CancellationHandle>,
    ) -> Result<bool> {
        let budget0 = budget;
        assert!(self.subsume_possible(state));
        let small = self.get_rx(lexeme_idx);
        let mut res = false;
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if let Some(cancellation) = cancellation {
                cancellation.check()?;
            }
            if !self.subsumable.contains(idx) {
                continue;
            }
            let c0 = self.exprs.cost();
            let cache_failures = budget > budget0 / 2;
            let is_contained = self
                .relevance
                .is_contained_in_prefixes(
                    &mut self.exprs,
                    &mut self.deriv,
                    small,
                    ExprRef::new(e),
                    budget,
                    cache_failures,
                )
                .unwrap_or(false);
            // println!("chk: {} in {} -> {}",
            //     self.exprs.expr_to_string(small),
            //     self.exprs.expr_to_string(e),
            //     is_contained
            // );
            if let Some(cancellation) = cancellation {
                cancellation.check()?;
            }
            if is_contained {
                res = true;
                break;
            }
            let cost = self.exprs.cost() - c0;
            budget = budget.saturating_sub(cost);
        }
        Ok(res)
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.exprs.num_bytes()
            + self.deriv.num_bytes()
            + self.next_byte.num_bytes()
            + self.relevance.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
            + self
                .int_ranges
                .as_ref()
                .map_or(0, |ranges| ranges.num_bytes)
    }

    /// Find the lowest, or best, match in 'state'.  It is the first lazy regex.
    /// If there is no lazy regex, and all greedy lexemes have reached the end of
    /// the lexeme, then it is the first greedy lexeme.  If neither of these
    /// criteria produce a choice for "best", 'None' is returned.
    fn lowest_match_inner<const WITH_INT_RANGES: bool>(&mut self, desc: &mut StateDesc) {
        // 'all_eoi' is true if all greedy lexemes match, that is, if we are at
        // the end of lexeme for all of them.  End of lexeme is called
        // "end of input" or EOI for consistency with the regex package.
        // Initially, 'all_eoi' is true, vacuously.
        let mut all_eoi = true;

        // 'eoi_candidate' tracks the lowest (aka first or best) greedy match.
        // Initially, there is none.
        let mut eois = MatchingLexemes::None;

        let mut lazies = MatchingLexemes::None;

        let mut hidden_len = 0;

        // For every regex in this state
        for (idx, e) in iter_state(&self.rx_sets, desc.state) {
            if WITH_INT_RANGES {
                let ranges = self.int_ranges.as_mut().unwrap();
                let at_eoi = match ranges.with_matcher(idx, &mut self.fuel, |m| {
                    // Dynamic lexemes are greedy. Only an accepting position can
                    // finish, and further probes cannot restore a false all_eoi.
                    m.is_accepting(e) && all_eoi && m.next_byte(e) == NextByte::ForcedEOI
                }) {
                    Ok(next) => next,
                    Err(()) => {
                        self.alpha.enter_error_state();
                        return;
                    }
                };
                if let Some(at_eoi) = at_eoi {
                    if at_eoi {
                        eois.add(idx);
                    } else {
                        all_eoi = false;
                    }
                    continue;
                }
            }
            let e = ExprRef::new(e);
            // If this lexeme is not a match.  (If the derivative at this point is nullable,
            // there is a match, so if it is not nullable, there is no match.)
            // println!("idx: {:?} e: {:?} {:?}", idx, e,self.special_token_rx);
            if !self.exprs.is_nullable(e) {
                // No match, so not at end of lexeme
                all_eoi = false;
                continue;
            } else if Some(self.get_rx(idx)) == self.special_token_rx {
                // the regex is /\xFF\[[0-9]+\]/ so it's guaranteed not to conflict with anything
                // else (starts with non-unicode byte); thus we ignore the rest of processing
                // when has_special_token is set, we just need to make sure lazy_accepting is non-empty,
                // the actual value is not important
                desc.lazy_accepting = MatchingLexemes::One(idx);
                desc.has_special_token = true;
                return;
            }

            // If this is the first lazy lexeme, we can cut things short.  The first
            // lazy lexeme is our lowest, or best, match.  We return it and are done.
            if self.lazy.contains(idx) {
                if lazies.is_none() {
                    all_eoi = false;
                    hidden_len = self.exprs.possible_lookahead_len(e) as u32;
                }
                lazies.add(idx);
                continue;
            }

            // If all the greedy lexemes so far are matches.
            if all_eoi {
                // If this greedy lexeme is at end of lexeme ...
                if self.next_byte.next_byte(&self.exprs, e) == NextByte::ForcedEOI {
                    // then, if we have not yet found a matching greedy lexeme, set
                    // this one to be our lowest match ...
                    eois.add(idx);
                } else {
                    // ... otherwise, if this greedy lexeme is not yet a match, then indicate
                    // that not all greedy lexemes are matches at this point.
                    all_eoi = false;
                }
            }
        }

        if lazies.is_some() {
            desc.lazy_accepting = lazies;
            desc.lazy_hidden_len = hidden_len;
        } else if all_eoi {
            desc.lazy_accepting = eois;
            // no hidden len
        }
    }

    /// Stops fast-forwarding while an integer-range matcher is possible, so the
    /// model chooses among valid tokenizations without expanding a forced sequence.
    pub(crate) fn allows_forcing(&self, state: StateID) -> bool {
        self.int_ranges.as_ref().is_none_or(|ranges| {
            !ranges
                .matchers
                .keys()
                .any(|idx| self.state_desc(state).possible.contains(*idx))
        })
    }

    /// Reports whether this lexer contains interval matchers.
    pub(crate) fn has_int_ranges(&self) -> bool {
        self.int_ranges.is_some()
    }

    /// Check if the there is only one transition out of state.
    /// This is an approximation - see docs for NextByte.
    pub fn next_byte(&mut self, state: StateID) -> NextByte {
        let desc = &self.state_descs[state.as_usize()];
        if let Some(next_byte) = desc.next_byte {
            return next_byte;
        }
        if self.int_ranges.is_none() {
            self.next_byte_inner::<false>(state)
        } else {
            self.next_byte_inner::<true>(state)
        }
    }

    /// Computes and caches the union of possible next bytes from active matchers.
    fn next_byte_inner<const WITH_INT_RANGES: bool>(&mut self, state: StateID) -> NextByte {
        let mut next_byte = NextByte::Dead;
        for (idx, e) in iter_state(&self.rx_sets, state) {
            let dynamic_next = if WITH_INT_RANGES {
                let ranges = self.int_ranges.as_mut().unwrap();
                match ranges.with_matcher(idx, &mut self.fuel, |m| m.next_byte(e)) {
                    Ok(next) => next,
                    Err(()) => {
                        self.alpha.enter_error_state();
                        return NextByte::Dead;
                    }
                }
            } else {
                None
            };
            let next = dynamic_next
                .unwrap_or_else(|| self.next_byte.next_byte(&self.exprs, ExprRef::new(e)));
            next_byte = next_byte | next;
            if next_byte.is_some_bytes() {
                break;
            }
        }

        self.state_descs[state.as_usize()].next_byte = Some(next_byte);
        next_byte
    }

    pub fn limit_state_to(&mut self, state: StateID, allowed_lexemes: &LexemeSet) -> StateID {
        let mut vec_desc = vec![];
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if allowed_lexemes.contains(idx) {
                Self::push_state(&mut vec_desc, idx, e);
            }
        }
        self.insert_state(vec_desc)
    }

    pub fn total_fuel_spent(&self) -> u64 {
        self.exprs.cost() + self.int_ranges.as_ref().map_or(0, |ranges| ranges.cost)
    }

    pub fn lexeme_weight(&mut self, lexeme_idx: LexemeIdx) -> u32 {
        let e = self.rx_list[lexeme_idx.as_usize()];
        self.exprs.get_weight(e)
    }

    pub fn set_max_states(&mut self, max_states: usize) {
        if !self.has_error() {
            self.max_states = max_states;
            if let Some(ranges) = &mut self.int_ranges {
                ranges.max_bytes = max_states
                    .saturating_sub(self.state_descs.len())
                    .saturating_mul(1024);
                if ranges.num_bytes >= ranges.max_bytes {
                    self.alpha.enter_error_state();
                }
            }
        }
    }

    // Each fuel point is on the order 100ns (though it varies).
    // So, for ~10ms limit, do a .set_fuel(100_000).
    pub fn set_fuel(&mut self, fuel: u64) {
        if !self.has_error() {
            self.fuel = fuel;
        }
    }

    pub fn get_fuel(&self) -> u64 {
        self.fuel
    }

    pub fn has_error(&self) -> bool {
        self.alpha.has_error()
    }

    pub fn get_error(&self) -> Option<String> {
        if self.has_error() {
            if self.fuel == 0 {
                Some("too many expressions constructed".to_string())
            } else if self.state_descs.len() >= self.max_states {
                Some(format!(
                    "too many states: {} >= {}",
                    self.state_descs.len(),
                    self.max_states
                ))
            } else if let Some(ranges) = &self.int_ranges {
                Some(format!(
                    "%int_ranges cache exceeds max_lexer_states: {} bytes, {} bytes available",
                    ranges.num_bytes, ranges.max_bytes
                ))
            } else {
                Some("unknown error".to_string())
            }
        } else {
            None
        }
    }

    pub fn stats(&self) -> LexerStats {
        LexerStats {
            num_regexps: self.rx_list.len(),
            num_ast_nodes: self.num_ast_nodes,
            num_derived: self.exprs.len() - self.num_ast_nodes,
            num_derivatives: self.deriv.num_deriv,
            total_fuel_spent: self.total_fuel_spent() as usize,
            num_states: self.state_descs.len(),
            num_transitions: self.num_transitions,
            num_bytes: self.num_bytes(),
            alphabet_size: self.alpha.len(),
            error: self.has_error(),
        }
    }

    pub fn print_state_table(&self) {
        for (state, row) in self.state_table.chunks(self.alpha.len()).enumerate() {
            println!("state: {state}");
            for (b, &new_state) in row.iter().enumerate() {
                println!("  s{b:?} -> {new_state:?}");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RxLexeme {
    pub rx: ExprRef,
    pub lazy: bool,
    #[allow(dead_code)]
    pub priority: i32,
}

/// Interval matchers and their aggregate resource usage.
struct IntRangesLexers {
    /// Matchers indexed by lexeme ID; their position IDs appear in the state vector.
    matchers: HashMap<LexemeIdx, IntRangesMatcher>,
    /// Running total of matcher work used for fuel accounting.
    cost: u64,
    /// Retained sequence positions and their lookup tables.
    num_bytes: usize,
    /// Each unused outer-state budget unit permits one KiB of inner cache storage.
    max_bytes: usize,
}

impl Clone for IntRangesLexers {
    /// Recomputes storage after cloning because cloned vectors can have less spare
    /// capacity. Repeated clone-and-grow cycles must not accumulate phantom usage.
    fn clone(&self) -> Self {
        let matchers = self.matchers.clone();
        let num_bytes = matchers
            .values()
            .map(IntRangesMatcher::num_bytes)
            .sum::<usize>()
            + matchers.capacity() * (std::mem::size_of::<(LexemeIdx, IntRangesMatcher)>() + 1);
        Self {
            matchers,
            cost: self.cost,
            num_bytes,
            max_bytes: self.max_bytes,
        }
    }
}

impl IntRangesLexers {
    /// Accounts for one bounded operation and stops before visiting another matcher
    /// when fuel or storage is exhausted. It leaves error publication to callers
    /// so cancellable scans can check cancellation first.
    fn with_matcher<T>(
        &mut self,
        idx: LexemeIdx,
        fuel: &mut u64,
        operation: impl FnOnce(&mut IntRangesMatcher) -> T,
    ) -> Result<Option<T>, ()> {
        let Some(matcher) = self.matchers.get_mut(&idx) else {
            return Ok(None);
        };
        if *fuel == 0 || self.num_bytes >= self.max_bytes {
            return Err(());
        }
        let cost = matcher.cost();
        let bytes = matcher.num_bytes();
        let result = operation(matcher);
        let cost = matcher.cost() - cost;
        self.cost += cost;
        *fuel = fuel.saturating_sub(cost);
        self.num_bytes = self.num_bytes - bytes + matcher.num_bytes();
        if *fuel == 0 || self.num_bytes >= self.max_bytes {
            Err(())
        } else {
            Ok(Some(result))
        }
    }
}

// private implementation
impl RegexVec {
    pub(crate) fn new_with_exprset(
        mut exprset: ExprSet,
        mut rx_lexemes: Vec<RxLexeme>,
        special_token_rx: Option<ExprRef>,
        int_ranges: Option<&HashMap<LexemeIdx, IntRanges>>,
        limits: &mut ParserLimits,
    ) -> Result<Self> {
        let spec_pos = if let Some(rx) = special_token_rx {
            rx_lexemes.iter().position(|r| r.rx == rx)
        } else {
            None
        };
        let mut roots: Vec<_> = rx_lexemes.iter().map(|r| r.rx).collect();
        // Dynamic numeric bounds must distinguish every digit. Literal separators
        // must also retain their byte identities in the shared transition cache.
        if let Some(configs) = int_ranges {
            let mut bytes = [false; 256];
            for config in configs.values() {
                for byte in b"0123456789-"
                    .iter()
                    .copied()
                    .chain(config.separator.bytes())
                {
                    bytes[byte as usize] = true;
                }
            }
            for (byte, used) in bytes.into_iter().enumerate() {
                if used {
                    roots.push(exprset.mk_byte(byte as u8));
                }
            }
        }
        let (alpha, mut exprset, mut rx_list) = AlphabetInfo::from_exprset(exprset, &roots);
        rx_list.truncate(rx_lexemes.len());
        let num_ast_nodes = exprset.len();
        let special_token_rx = spec_pos.map(|pos| rx_list[pos]);

        for idx in 0..rx_lexemes.len() {
            rx_lexemes[idx].rx = rx_list[idx];
        }

        let fuel0 = limits.initial_lexer_fuel;
        let mut relevance = RelevanceCache::new();
        for elt in rx_list.iter_mut() {
            let c0 = exprset.cost();
            match relevance.is_non_empty_limited(&mut exprset, *elt, limits.initial_lexer_fuel) {
                Ok(true) => {}
                Ok(false) => {
                    *elt = ExprRef::NO_MATCH;
                }
                Err(_) => {
                    bail!(
                        "fuel exhausted when checking relevance of lexemes ({})",
                        fuel0
                    );
                }
            }
            limits.initial_lexer_fuel = limits
                .initial_lexer_fuel
                .saturating_sub(exprset.cost() - c0);
        }

        let mut lazy = LexemeSet::new(rx_lexemes.len());
        let mut subsumable = LexemeSet::new(rx_lexemes.len());
        for (idx, r) in rx_lexemes.iter().enumerate() {
            if r.lazy {
                lazy.add(LexemeIdx::new(idx));
            } else if exprset.attr_has_repeat(r.rx) {
                subsumable.add(LexemeIdx::new(idx));
            }
        }

        let rx_sets = StateID::new_hash_cons();
        let int_ranges = if let Some(configs) = int_ranges {
            let mut ranges = IntRangesLexers {
                matchers: HashMap::default(),
                cost: 0,
                num_bytes: 0,
                max_bytes: limits.max_lexer_states.saturating_mul(1024),
            };
            for (&idx, config) in configs {
                let matcher = IntRangesMatcher::new(config.clone());
                let cost = matcher.cost();
                // Stop at the exhausted budget instead of compiling every
                // interval lexeme before checking their combined cost.
                if cost > limits.initial_lexer_fuel {
                    bail!("fuel exhausted when compiling %int_ranges");
                }
                limits.initial_lexer_fuel -= cost;
                ranges.cost += cost;
                ranges.num_bytes += matcher.num_bytes();
                ranges.matchers.insert(idx, matcher);
                // Include the matcher map's allocation in the storage budget.
                let map_bytes = ranges.matchers.capacity()
                    * (std::mem::size_of::<(LexemeIdx, IntRangesMatcher)>() + 1);
                if ranges.num_bytes + map_bytes >= ranges.max_bytes {
                    bail!("%int_ranges cache exceeds max_lexer_states during compilation");
                }
            }
            ranges.num_bytes += ranges.matchers.capacity()
                * (std::mem::size_of::<(LexemeIdx, IntRangesMatcher)>() + 1);
            Some(Box::new(ranges))
        } else {
            None
        };
        let mut r = RegexVec {
            deriv: DerivCache::new(),
            next_byte: NextByteCache::new(),
            special_token_rx,
            relevance,
            lazy,
            subsumable,
            rx_lexemes,
            exprs: exprset,
            alpha,
            rx_list,
            int_ranges,
            rx_sets,
            state_table: vec![],
            state_descs: vec![],
            num_transitions: 0,
            num_ast_nodes,
            fuel: u64::MAX,
            max_states: usize::MAX,
        };

        assert!(r.lazy.len() == r.rx_list.len());

        r.insert_state(vec![]);
        // also append state for the "MISSING"
        r.append_state(r.state_descs[0].clone());
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);
        assert!(!r.alpha.is_empty());
        if r.int_ranges.is_some() {
            r.set_max_states(limits.max_lexer_states);
            r.set_fuel(limits.initial_lexer_fuel);
            if let Some(error) = r.get_error() {
                bail!(error);
            }
        }
        Ok(r)
    }

    fn get_rx(&self, idx: LexemeIdx) -> ExprRef {
        self.rx_list[idx.as_usize()]
    }

    fn append_state(&mut self, state_desc: StateDesc) {
        let mut new_states = vec![StateID::MISSING; self.alpha.len()];
        self.state_table.append(&mut new_states);
        self.state_descs.push(state_desc);
        if self.state_descs.len() >= self.max_states {
            self.alpha.enter_error_state();
        }
        if let Some(ranges) = &mut self.int_ranges {
            ranges.max_bytes = self
                .max_states
                .saturating_sub(self.state_descs.len())
                .saturating_mul(1024);
            if ranges.num_bytes >= ranges.max_bytes {
                self.alpha.enter_error_state();
            }
        }
    }

    fn insert_state(&mut self, lst: Vec<u32>) -> StateID {
        // does this help?
        // if lst.len() == 0 {
        //     return StateID::DEAD;
        // }
        assert!(lst.len().is_multiple_of(2));
        let id = StateID::new(self.rx_sets.insert(&lst));
        if id.as_usize() >= self.state_descs.len() {
            let state_desc = if self.int_ranges.is_none() {
                self.compute_state_desc::<false>(id)
            } else {
                self.compute_state_desc::<true>(id)
            };
            self.append_state(state_desc);
        }
        if self.state_desc(id).lazy_accepting.is_some() {
            id._set_lowest_match()
        } else {
            id
        }
    }

    /// Builds acceptance metadata for the active regex and interval matchers.
    fn compute_state_desc<const WITH_INT_RANGES: bool>(&mut self, state: StateID) -> StateDesc {
        let mut res = StateDesc {
            state,
            greedy_accepting: MatchingLexemes::None,
            possible: LexemeSet::new(self.rx_list.len()),
            possible_lookahead_len: None,
            lookahead_len: None,
            next_byte: None,
            lazy_accepting: MatchingLexemes::None,
            lazy_hidden_len: 0,
            has_special_token: false,
        };
        for (idx, e) in iter_state(&self.rx_sets, state) {
            res.possible.add(idx);
            let dynamic_accepting = if WITH_INT_RANGES {
                match self
                    .int_ranges
                    .as_mut()
                    .unwrap()
                    .with_matcher(idx, &mut self.fuel, |m| m.is_accepting(e))
                {
                    Ok(accepting) => accepting,
                    Err(()) => {
                        self.alpha.enter_error_state();
                        return res;
                    }
                }
            } else {
                None
            };
            let accepting =
                dynamic_accepting.unwrap_or_else(|| self.exprs.is_nullable(ExprRef::new(e)));
            if accepting {
                res.greedy_accepting.add(idx);
            }
        }

        if res.possible.is_empty() {
            assert!(state == StateID::DEAD);
        }

        self.lowest_match_inner::<WITH_INT_RANGES>(&mut res);

        // println!("state {:?} desc: {:?}", state, res);

        res
    }

    /// Appends a regex expression ID or an iterative matcher position ID,
    /// interpreted according to the associated lexeme's matcher kind.
    fn push_state(vec_desc: &mut Vec<u32>, idx: LexemeIdx, value: u32) {
        vec_desc.push(idx.as_usize() as u32);
        vec_desc.push(value);
    }

    /// Given a transition (from-state and byte), create the to-state.
    /// It is assumed the to-state does not exist.
    /// `WITH_INT_RANGES` specializes the hot loop for the grammar's lexeme kinds.
    fn transition_inner<const WITH_INT_RANGES: bool>(
        &mut self,
        state: StateID,
        b: u8,
        idx: usize,
        cancellation: &Option<crate::CancellationHandle>,
    ) -> StateID {
        assert!(state.is_valid());

        let mut vec_desc = vec![];

        // let d0 = self.deriv.num_deriv;
        // Dynamic operations debit fuel immediately; regex work is charged below.
        let mut c0 = self.exprs.cost();
        // let t0 = crate::Instant::now();
        // let mut state_size = 0;

        for (idx, e) in iter_state(&self.rx_sets, state) {
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return StateID::DEAD;
            }
            if WITH_INT_RANGES {
                let ranges = self.int_ranges.as_mut().unwrap();
                let next = ranges.with_matcher(idx, &mut self.fuel, |m| m.transition(e, b));
                #[cfg(test)]
                if !matches!(next, Ok(None)) {
                    crate::cancellation::checkpoint("lexer");
                }
                if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                    return StateID::DEAD;
                }
                let next = match next {
                    Ok(next) => next,
                    Err(()) => {
                        self.alpha.enter_error_state();
                        return StateID::DEAD;
                    }
                };
                if let Some(next) = next {
                    if let Some(next) = next {
                        Self::push_state(&mut vec_desc, idx, next);
                    }
                    continue;
                }
            }
            let d = self.deriv.derivative(&mut self.exprs, ExprRef::new(e), b);
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return StateID::DEAD;
            }

            let fuel = self.fuel.saturating_sub(self.exprs.cost() - c0);
            let non_empty = self
                .relevance
                .is_non_empty_limited(&mut self.exprs, d, fuel);
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return StateID::DEAD;
            }
            let d = match non_empty {
                Ok(true) => d,
                Ok(false) => ExprRef::NO_MATCH,
                Err(_) => {
                    self.fuel = 0; // just in case
                    break;
                }
            };

            #[cfg(test)]
            crate::cancellation::checkpoint("lexer");
            // state_size += 1;
            if d != ExprRef::NO_MATCH {
                Self::push_state(&mut vec_desc, idx, d.as_u32());
            }
            if WITH_INT_RANGES {
                // Charge regex work before the next dynamic operation checks
                // the remaining fuel.
                self.fuel = self.fuel.saturating_sub(self.exprs.cost() - c0);
                c0 = self.exprs.cost();
                if self.fuel == 0 {
                    break;
                }
            }
        }

        // Do not publish an incomplete transition or change shared error state.
        if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
            return StateID::DEAD;
        }
        if WITH_INT_RANGES && self.fuel == 0 {
            self.alpha.enter_error_state();
            return StateID::DEAD;
        }

        // let num_deriv = self.deriv.num_deriv - d0;
        let new_state = self.insert_state(vec_desc);
        let cost = self.exprs.cost() - c0;
        self.fuel = self.fuel.saturating_sub(cost);
        if self.fuel == 0 {
            self.alpha.enter_error_state();
        }
        // if false && cost > 40 {
        //     eprintln!(
        //         "cost: {:?} {} {} size={}",
        //         t0.elapsed() / (cost as u32),
        //         num_deriv,
        //         cost,
        //         state_size
        //     );

        //     // for (idx, e) in iter_state(&self.rx_sets, state) {
        //     //     eprintln!("expr{}: {}", idx, self.exprs.expr_to_string(e));
        //     // }
        // }
        self.num_transitions += 1;
        self.state_table[idx] = new_state;
        new_state
    }
}

impl Debug for RegexVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexVec({})", self.stats())
    }
}

/// Iterates the interned vector without interpreting its matcher-specific IDs.
fn iter_state(
    rx_sets: &VecHashCons,
    state: StateID,
) -> impl Iterator<Item = (LexemeIdx, u32)> + '_ {
    let lst = rx_sets.get(state.as_u32());
    (0..lst.len())
        .step_by(2)
        .map(move |idx| (LexemeIdx::new(lst[idx] as usize), lst[idx + 1]))
}

// #[test]
// fn test_fuel() {
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(200);
//     match_(&mut rx, "abcg");
//     assert!(!rx.has_error());
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(20);
//     no_match(&mut rx, "abcg");
//     assert!(rx.has_error());
// }
