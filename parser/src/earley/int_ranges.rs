#[cfg(feature = "lark")]
use anyhow::{ensure, Result};
use derivre::{NextByte, RandomState};
use indexmap::IndexSet;
use serde::Deserialize;

/// Lark's ordered interval sequence. Bounds are inclusive and every interval
/// consumes at least one ID, so the numeric domain also bounds the count.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct IntRanges {
    pub min: u32,
    pub max: u32,
    /// Exact decimal width, at most ten; zero uses canonical, unpadded notation.
    #[serde(default)]
    pub width: u32,
    /// Nonempty literal of at most sixteen UTF-8 bytes, excluding ASCII digits and '-'.
    #[serde(default = "IntRanges::default_separator")]
    pub separator: String,
    /// Zero allows an empty sequence; one requires at least one interval.
    #[serde(default)]
    pub min_ranges: u64,
    /// An omitted limit is bounded only by the number of available IDs.
    pub max_ranges: Option<u64>,
}

impl IntRanges {
    /// Rejects oversized widths or separators and impossible sequence constraints.
    #[cfg(feature = "lark")]
    pub fn validate(&self) -> Result<()> {
        ensure!(self.min <= self.max, "%int_ranges: min must be <= max");
        ensure!(self.width <= 10, "%int_ranges: width must be at most 10");
        ensure!(
            self.width == 0 || self.width >= self.max.to_string().len() as u32,
            "%int_ranges: width is insufficient for max"
        );
        ensure!(
            self.separator.len() <= 16,
            "%int_ranges: separator must be at most 16 UTF-8 bytes"
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
            self.min_ranges <= 1,
            "%int_ranges: min_ranges must be 0 or 1"
        );
        ensure!(
            self.min_ranges <= self.max_ranges.unwrap_or(u64::MAX),
            "%int_ranges: min_ranges must be <= max_ranges"
        );
        Ok(())
    }

    /// Supplies the literal separator when the Lark object omits it.
    fn default_separator() -> String {
        ",".to_string()
    }
}

/// One immutable position in an interval sequence. Interning these positions
/// lets the lexer state stack restore speculative and committed scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Position {
    /// Completed intervals before this endpoint; zero when there is no maximum.
    count: u64,
    phase: Phase,
    /// Lower bound on this endpoint, or on the next start during a separator.
    lower: u32,
    /// Value of the decimal prefix scanned so far.
    value: u32,
    digits: u8,
}

/// The only scanning phases; no productions or call stack accumulate per range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Phase {
    Start,
    End,
    /// Next byte of the literal separator, whose validated length is at most 16.
    Separator(u8),
}

/// Interned sequence positions for one lexeme. Decimal-prefix arithmetic checks
/// endpoint bounds without materializing a separate DFA for each lower bound.
/// All semantic state is selected by a position ID, including during trie walks.
#[derive(Clone)]
pub(crate) struct IntRangesMatcher {
    config: IntRanges,
    /// Insertion indices are stable IDs; positions are never removed.
    positions: IndexSet<Position, RandomState>,
    /// Matcher visits, decimal-prefix probes and position interning charged to fuel.
    cost: u64,
}

impl IntRangesMatcher {
    /// Creates position zero, the empty sequence with no endpoint scanned yet.
    pub fn new(config: IntRanges) -> Self {
        let initial = Position {
            count: 0,
            phase: Phase::Start,
            lower: config.min,
            value: 0,
            digits: 0,
        };
        Self {
            config,
            positions: IndexSet::from_iter([initial]),
            cost: 1,
        }
    }

    /// Advances a single byte, rejecting prefixes that cannot be completed.
    /// Bound changes occur here, so tokens may cross any number of endpoints.
    pub fn transition(&mut self, id: u32, byte: u8) -> Option<u32> {
        let mut pos = self.step(self.positions[id as usize], byte)?;
        self.canonicalize(&mut pos);
        self.cost += 1;
        let (id, _) = self.positions.insert_full(pos);
        Some(u32::try_from(id).expect("too many interval positions"))
    }

    /// Whether the sequence may end here, without an incomplete endpoint or separator.
    pub fn is_accepting(&mut self, id: u32) -> bool {
        self.cost += 1;
        let pos = &self.positions[id as usize];
        (id == 0 && self.config.min_ranges == 0)
            || (pos.phase == Phase::End && self.complete_number(pos))
    }

    /// Finds exact next bytes, including the delimiter and optional end of input.
    /// Probes stop once multiple continuations are known and never intern positions.
    pub fn next_byte(&mut self, id: u32) -> NextByte {
        self.cost += 1;
        let pos = self.positions[id as usize];
        if let Phase::Separator(offset) = pos.phase {
            return NextByte::ForcedByte(self.config.separator.as_bytes()[usize::from(offset)]);
        }
        let mut next = if self.is_accepting(id) {
            NextByte::ForcedEOI
        } else {
            NextByte::Dead
        };
        for byte in (b'0'..=b'9').chain([b'-', self.config.separator.as_bytes()[0]]) {
            if self.step(pos, byte).is_some() {
                next = next | NextByte::ForcedByte(byte);
                if next.is_some_bytes() {
                    break;
                }
            }
        }
        next
    }

    /// Returns accumulated work, including visits which reuse interned positions.
    pub fn cost(&self) -> u64 {
        self.cost
    }

    /// Estimates retained storage in constant time for statistics and budget checks.
    /// Includes spare capacity, cached hashes and index-table slack; IndexSet does
    /// not expose the sizes of its individual allocations.
    pub fn num_bytes(&self) -> usize {
        self.config.separator.capacity()
            + self.positions.capacity()
                * (std::mem::size_of::<Position>() + 3 * std::mem::size_of::<usize>())
    }

    /// Checks whether the current decimal prefix is a complete bounded endpoint.
    fn complete_number(&self, pos: &Position) -> bool {
        pos.digits > 0
            && (self.config.width == 0 || u32::from(pos.digits) == self.config.width)
            && pos.value >= pos.lower
            && pos.value <= self.config.max
    }

    /// Tests whether any permitted decimal extension intersects the endpoint bounds.
    /// Each candidate length costs fuel; u64 covers every ten-digit extension.
    fn viable_number(&mut self, pos: &Position) -> bool {
        let lo = u64::from(pos.lower);
        let hi = u64::from(self.config.max);
        let extra = if self.config.width == 0 {
            if pos.value == 0 {
                0
            } else {
                10 - u32::from(pos.digits)
            }
        } else {
            self.config.width - u32::from(pos.digits)
        };
        let first = if self.config.width == 0 { 0 } else { extra };
        let mut scale = 10u64.pow(first);
        for _ in first..=extra {
            self.cost += 1;
            let start = u64::from(pos.value) * scale;
            if start <= hi && start + scale > lo {
                return true;
            }
            if start > hi {
                break;
            }
            scale *= 10;
        }
        false
    }

    /// Computes a transition without interning it, for both scanning and next-byte
    /// probes. Only complete ends may start a separator; singletons require '-'.
    fn step(&mut self, mut pos: Position, byte: u8) -> Option<Position> {
        self.cost += 1;
        if let Phase::Separator(offset) = pos.phase {
            if byte != self.config.separator.as_bytes()[usize::from(offset)] {
                return None;
            }
            pos.phase = if usize::from(offset) + 1 == self.config.separator.len() {
                Phase::Start
            } else {
                Phase::Separator(offset + 1)
            };
            return Some(pos);
        }
        if byte.is_ascii_digit() {
            if pos.digits >= 10
                || (self.config.width > 0 && u32::from(pos.digits) >= self.config.width)
                || (self.config.width == 0 && pos.digits > 0 && pos.value == 0)
            {
                return None;
            }
            pos.value = pos
                .value
                .checked_mul(10)?
                .checked_add(u32::from(byte - b'0'))?;
            pos.digits += 1;
            return self.viable_number(&pos).then_some(pos);
        }
        if !self.complete_number(&pos) {
            return None;
        }
        match pos.phase {
            Phase::Start if byte == b'-' => {
                pos.phase = Phase::End;
                pos.lower = pos.value;
            }
            Phase::End
                if byte == self.config.separator.as_bytes()[0]
                    && pos.count + 1 < self.config.max_ranges.unwrap_or(u64::MAX)
                    && pos.value < self.config.max =>
            {
                if self.config.max_ranges.is_some() {
                    pos.count += 1;
                }
                pos.phase = if self.config.separator.len() == 1 {
                    Phase::Start
                } else {
                    Phase::Separator(1)
                };
                pos.lower = pos.value + 1;
            }
            _ => return None,
        }
        pos.value = 0;
        pos.digits = 0;
        Some(pos)
    }

    /// Merges positions only when they admit the same suffixes. Once a prefix
    /// clears its lower bound, retaining that bound would duplicate every later
    /// digit state for each previously selected endpoint.
    fn canonicalize(&mut self, pos: &mut Position) {
        self.cost += 1;
        if pos.digits == 0 {
            return;
        }
        if pos.value >= pos.lower {
            // Appending digits cannot decrease a nonnegative endpoint.
            pos.lower = 0;
            return;
        }
        let lower = u64::from(pos.lower);
        if self.config.width > 0 {
            let scale = 10u64.pow(self.config.width - u32::from(pos.digits));
            if u64::from(pos.value) * scale >= lower {
                pos.lower = 0;
            }
            return;
        }
        // Completions of prefix p occupy [p*10^k, (p+1)*10^k-1]. A bound
        // inside one interval must stay exact; a bound between two intervals
        // can move down to the first value above the preceding interval.
        let mut scale = 1u64;
        let mut boundary = 0;
        for _ in pos.digits..=10 {
            self.cost += 1;
            let start = u64::from(pos.value) * scale;
            let end = start + scale - 1;
            if start >= lower {
                pos.lower = boundary;
                return;
            }
            if end >= lower {
                return;
            }
            // end < lower <= u32::MAX, so this conversion cannot overflow.
            boundary = (end + 1) as u32;
            scale *= 10;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonicalization must preserve every remaining digit suffix, including
    /// thresholds inside and between decimal-prefix intervals and padded zeros.
    #[test]
    fn canonical_numeric_prefixes_preserve_suffixes() {
        for width in [0, 4] {
            let mut matcher = IntRangesMatcher::new(IntRanges {
                min: 0,
                max: 9999,
                width,
                separator: ",".into(),
                min_ranges: 1,
                max_ranges: None,
            });
            for value in [0u32, 1, 3, 4, 9, 10, 31, 32, 33, 99, 300, 350, 999] {
                for lower in [
                    0, 1, 9, 10, 19, 31, 32, 33, 40, 99, 100, 299, 300, 310, 311, 320, 999, 1000,
                    3500, 9999,
                ] {
                    for phase in [Phase::Start, Phase::End] {
                        let min_digits = value.to_string().len() as u8;
                        let max_digits = if width == 0 { min_digits } else { width as u8 };
                        for digits in min_digits..=max_digits {
                            let original = Position {
                                count: 0,
                                phase,
                                lower,
                                value,
                                digits,
                            };
                            if !matcher.viable_number(&original) {
                                continue;
                            }
                            let mut canonical = original;
                            matcher.canonicalize(&mut canonical);
                            compare_numeric_suffixes(&mut matcher, original, canonical);
                        }
                    }
                }
            }
        }
    }

    /// Recursively compares the finite decimal suffix language and both delimiters;
    /// delimiter transitions must produce identical bounds for the next endpoint.
    fn compare_numeric_suffixes(
        matcher: &mut IntRangesMatcher,
        original: Position,
        canonical: Position,
    ) {
        assert_eq!(
            matcher.complete_number(&original),
            matcher.complete_number(&canonical)
        );
        for byte in b'0'..=b'9' {
            let a = matcher.step(original, byte);
            let b = matcher.step(canonical, byte);
            assert_eq!(
                a.is_some(),
                b.is_some(),
                "{original:?} vs {canonical:?}, byte {byte}"
            );
            if let (Some(a), Some(b)) = (a, b) {
                compare_numeric_suffixes(matcher, a, b);
            }
        }
        for byte in [b'-', b','] {
            assert_eq!(matcher.step(original, byte), matcher.step(canonical, byte));
        }
    }
}
