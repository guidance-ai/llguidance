#[cfg(feature = "lark")]
use anyhow::{ensure, Result};
use derivre::{NextByte, Regex, StateID};
use serde::Deserialize;

use crate::{json::numeric::rx_int_range, HashMap};

/// Lark's ordered interval sequence. Bounds are inclusive and every interval
/// consumes at least one ID, so the numeric domain also bounds the count.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct IntRanges {
    pub min: u32,
    pub max: u32,
    /// Exact decimal width; zero uses canonical, unpadded decimal notation.
    #[serde(default)]
    pub width: u32,
    #[serde(default = "IntRanges::default_separator")]
    pub separator: String,
    #[serde(default)]
    pub min_ranges: u64,
    /// An omitted limit is bounded only by the number of available IDs.
    pub max_ranges: Option<u64>,
}

impl IntRanges {
    /// Rejects configurations that cannot produce the requested sequence.
    #[cfg(feature = "lark")]
    pub fn validate(&self) -> Result<()> {
        ensure!(self.min <= self.max, "%int_ranges: min must be <= max");
        ensure!(
            self.width == 0 || self.width >= self.max.to_string().len() as u32,
            "%int_ranges: width is insufficient for max"
        );
        ensure!(
            !self.separator.is_empty()
                && !self
                    .separator
                    .bytes()
                    .any(|b| b.is_ascii_digit() || b == b'-'),
            "%int_ranges: separator must be nonempty and contain neither decimal digits nor '-'"
        );
        ensure!(
            self.min_ranges <= self.max_ranges.unwrap_or(u64::MAX),
            "%int_ranges: min_ranges must be <= max_ranges"
        );
        ensure!(
            self.min_ranges <= u64::from(self.max) - u64::from(self.min) + 1,
            "%int_ranges: impossible min_ranges for the endpoint bounds"
        );
        Ok(())
    }

    /// Supplies the literal separator when the Lark object omits it.
    fn default_separator() -> String {
        ",".to_string()
    }
}

/// One immutable position in an interval sequence. Interning these positions
/// lets the ordinary lexer state stack restore speculative and committed scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Position {
    /// Fully consumed intervals before the current endpoint or separator.
    count: u64,
    phase: Phase,
}

/// The only scanning phases; no productions or call stack accumulate per range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Phase {
    /// A bounded decimal endpoint, with its parsed value preserved for the next bound.
    Endpoint {
        is_end: bool,
        matcher: usize,
        state: StateID,
        value: u32,
        /// Width beyond ten digits is necessarily zero padding. Track it without
        /// constructing a huge regex or using the regex engine's unbounded repeat.
        padding: u32,
    },
    /// A literal separator; the previous end determines the next lower bound.
    Separator { offset: usize, end: u32 },
}

/// Cached bounded-number DFAs and interned sequence positions for one lexeme.
/// All semantic state is selected by a position ID, including during trie walks.
#[derive(Clone)]
pub(crate) struct IntRangesMatcher {
    config: IntRanges,
    positions: Vec<Position>,
    position_ids: HashMap<Position, u32>,
    matchers: Vec<Regex>,
    matcher_ids: HashMap<(u32, u32), usize>,
    /// Regex construction and derivative work, charged to the lexer fuel budget.
    cost: u64,
}

impl IntRangesMatcher {
    /// Compiles the first bounded endpoint; all subsequent bounds use the same
    /// generator and are cached on demand. Position zero is always the start.
    pub fn new(config: IntRanges) -> Self {
        let mut result = Self {
            config,
            positions: Vec::new(),
            position_ids: HashMap::default(),
            matchers: Vec::new(),
            matcher_ids: HashMap::default(),
            cost: 0,
        };
        let phase = result.endpoint(false, result.config.min, 0);
        assert_eq!(result.intern(Position { count: 0, phase }), 0);
        result
    }

    /// Advances a single byte, rejecting prefixes that cannot be completed.
    /// Bound changes occur here, so tokens may cross any number of endpoints.
    pub fn transition(&mut self, id: u32, byte: u8) -> Option<u32> {
        let mut pos = self.positions[id as usize];
        match &mut pos.phase {
            Phase::Endpoint {
                is_end,
                matcher,
                state,
                value,
                padding,
            } => {
                if *padding > 0 {
                    if byte != b'0' {
                        return None;
                    }
                    *padding -= 1;
                } else if byte.is_ascii_digit() {
                    let rx = &mut self.matchers[*matcher];
                    let cost = rx.cost();
                    let next = rx.transition(*state, byte);
                    self.cost += rx.cost() - cost;
                    if next.is_dead() {
                        return None;
                    }
                    *state = next;
                    *value = value
                        .checked_mul(10)
                        .unwrap()
                        .checked_add(u32::from(byte - b'0'))
                        .unwrap();
                } else {
                    if !self.matchers[*matcher].is_accepting(*state) {
                        return None;
                    }
                    if !*is_end && byte == b'-' {
                        pos.phase = self.endpoint(true, *value, pos.count);
                    } else if *is_end
                        && byte == self.config.separator.as_bytes()[0]
                        && pos.count + 1 < self.config.max_ranges.unwrap_or(u64::MAX)
                        && *value < self.config.max
                    {
                        pos.count += 1;
                        pos.phase = if self.config.separator.len() == 1 {
                            self.endpoint(false, *value + 1, pos.count)
                        } else {
                            Phase::Separator {
                                offset: 1,
                                end: *value,
                            }
                        };
                    } else {
                        return None;
                    }
                }
            }
            Phase::Separator { offset, end } => {
                if byte != self.config.separator.as_bytes()[*offset] {
                    return None;
                }
                *offset += 1;
                if *offset == self.config.separator.len() {
                    pos.phase = self.endpoint(false, *end + 1, pos.count);
                }
            }
        }
        Some(self.intern(pos))
    }

    /// Whether the sequence may end here, without an incomplete endpoint or separator.
    pub fn is_accepting(&mut self, id: u32) -> bool {
        if id == 0 && self.config.min_ranges == 0 {
            return true;
        }
        let pos = self.positions[id as usize];
        match pos.phase {
            Phase::Endpoint {
                is_end: true,
                matcher,
                state,
                padding: 0,
                ..
            } => {
                pos.count + 1 >= self.config.min_ranges
                    && self.matchers[matcher].is_accepting(state)
            }
            _ => false,
        }
    }

    /// Finds exact next bytes, including the delimiter and optional end of input.
    /// Query the number DFA with raw bytes, independently of its compressed alphabet.
    pub fn next_byte(&mut self, id: u32) -> NextByte {
        let mut next = if self.is_accepting(id) {
            NextByte::ForcedEOI
        } else {
            NextByte::Dead
        };
        let pos = self.positions[id as usize];
        match pos.phase {
            Phase::Endpoint {
                padding,
                is_end,
                matcher,
                state,
                value,
            } => {
                if padding > 0 {
                    return next | NextByte::ForcedByte(b'0');
                }
                let rx = &mut self.matchers[matcher];
                let cost = rx.cost();
                for byte in b'0'..=b'9' {
                    if !rx.transition(state, byte).is_dead() {
                        next = next | NextByte::ForcedByte(byte);
                    }
                }
                self.cost += rx.cost() - cost;
                if rx.is_accepting(state) {
                    if !is_end {
                        next = next | NextByte::ForcedByte(b'-');
                    } else if pos.count + 1 < self.config.max_ranges.unwrap_or(u64::MAX)
                        && value < self.config.max
                    {
                        next = next | NextByte::ForcedByte(self.config.separator.as_bytes()[0]);
                    }
                }
            }
            Phase::Separator { offset, .. } => {
                next = next | NextByte::ForcedByte(self.config.separator.as_bytes()[offset]);
            }
        }
        next
    }

    /// Returns accumulated bounded-number regex work for lexer accounting.
    pub fn cost(&self) -> u64 {
        self.cost
    }

    /// Estimates retained matcher and position storage for lexer statistics.
    pub fn num_bytes(&self) -> usize {
        self.matchers.iter().map(Regex::num_bytes).sum::<usize>()
            + self.positions.len() * (2 * std::mem::size_of::<Position>() + 16)
            + self.matcher_ids.len() * 32
    }

    /// Creates a bounded endpoint, reserving one unused ID for each still-required
    /// later interval. Split by decimal length to reuse the canonical integer regex
    /// generator while adding exactly the necessary leading zeros.
    fn endpoint(&mut self, is_end: bool, min: u32, count: u64) -> Phase {
        let reserved = self.config.min_ranges.saturating_sub(count + 1);
        let max = (u64::from(self.config.max) - reserved) as u32;
        assert!(min <= max, "endpoint must leave room for required ranges");
        let matcher = if let Some(&idx) = self.matcher_ids.get(&(min, max)) {
            idx
        } else {
            let width = self.config.width.min(10);
            let pattern = if width == 0 {
                rx_int_range(Some(i64::from(min)), Some(i64::from(max))).unwrap()
            } else {
                let mut parts = Vec::new();
                for digits in 1..=width {
                    let lo = i64::from(min).max(if digits == 1 {
                        0
                    } else {
                        10i64.pow(digits - 1)
                    });
                    let hi = i64::from(max).min(10i64.pow(digits) - 1);
                    if lo <= hi {
                        let number = rx_int_range(Some(lo), Some(hi)).unwrap();
                        parts.push(format!("0{{{}}}{number}", width - digits));
                    }
                }
                format!("({})", parts.join("|"))
            };
            let rx = Regex::new(&pattern).expect("validated u32 endpoint regex");
            self.cost += rx.cost();
            let idx = self.matchers.len();
            self.matchers.push(rx);
            self.matcher_ids.insert((min, max), idx);
            idx
        };
        Phase::Endpoint {
            is_end,
            matcher,
            state: self.matchers[matcher].initial_state(),
            value: 0,
            padding: self.config.width.saturating_sub(10),
        }
    }

    /// Assigns a stable ID to all data that can affect future acceptance.
    fn intern(&mut self, pos: Position) -> u32 {
        if let Some(&id) = self.position_ids.get(&pos) {
            id
        } else {
            let id = self
                .positions
                .len()
                .try_into()
                .expect("too many interval positions");
            self.positions.push(pos);
            self.position_ids.insert(pos, id);
            id
        }
    }
}
