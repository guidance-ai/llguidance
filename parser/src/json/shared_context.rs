use crate::{json::schema::OptSchemaExt, regex_to_lark, HashMap, HashSet};
use anyhow::{anyhow, bail, Result};
use derivre::{ExprRef, NextByte, Regex, RegexAst, RegexBuilder};

use super::{
    context::Context,
    schema::{ObjectSchema, Schema, IMPLEMENTED, META_AND_ANNOTATIONS},
};

pub struct SharedContext {
    defs: HashMap<String, Schema>,
    seen: HashSet<String>,
    n_compiled: usize,
    pending_warnings: Vec<String>,
    pattern_cache: PatternPropertyCache,
}

#[derive(Default)]
pub struct PatternPropertyCache {
    inner: HashMap<String, Regex>,
}

const CHECK_LIMIT: u64 = 10_000;

/// Pairs `(a, b)`, `a < b`, in order, whose regexes need a derivre check; `prefixes`
/// holds the forced prefix of each regex. Every match of a regex starts with its
/// forced prefix, so when neither prefix of a pair is a prefix of the other, no string
/// matches both regexes, and the pair needs no check.
fn pairs_to_check(prefixes: &[Vec<u8>]) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..prefixes.len())
        .flat_map(move |a| (a + 1..prefixes.len()).map(move |b| (a, b)))
        .filter(move |&(a, b)| {
            let (pa, pb) = (&prefixes[a], &prefixes[b]);
            let n = pa.len().min(pb.len());
            pa[..n] == pb[..n]
        })
}

/// Builds the intersection of `a` and `b` in a copy of `base`, and leaves `base`
/// unchanged, so each pair check copies only the parsed regexes and one `And` node.
/// Do not build the intersections of all pairs in one builder: each check copies the
/// whole `ExprSet`, so the total work would grow as N^4 in the number of regexes.
fn intersection(base: &RegexBuilder, a: ExprRef, b: ExprRef) -> Result<(RegexBuilder, ExprRef)> {
    let mut builder = base.clone();
    let both = builder.mk(&RegexAst::And(vec![
        RegexAst::ExprRef(a),
        RegexAst::ExprRef(b),
    ]))?;
    Ok((builder, both))
}

impl PatternPropertyCache {
    pub fn is_match(&mut self, regex: &str, value: &str) -> Result<bool> {
        let lark_regex = regex_to_lark(regex, "dw");
        if let Some(cached_regex) = self.inner.get_mut(lark_regex.as_str()) {
            return Ok(cached_regex.is_match(value));
        }

        let mut builder = RegexBuilder::new();
        let eref = builder.mk_regex_for_serach(lark_regex.as_str())?;
        let mut rx = builder.to_regex_limited(eref, CHECK_LIMIT)?;
        let res = rx.is_match(value);
        self.inner.insert(lark_regex, rx);
        Ok(res)
    }

    /// Literal bytes that every match of `regex` starts with; empty when there is
    /// none (e.g. the regex is unanchored) or the regex can't be built cheaply.
    fn forced_prefix(&mut self, regex: &str) -> Vec<u8> {
        const MAX_PREFIX: usize = 256;
        let lark_regex = regex_to_lark(regex, "dw");
        if !self.inner.contains_key(lark_regex.as_str()) {
            let mut builder = RegexBuilder::new();
            let rx = builder
                .mk_regex_for_serach(lark_regex.as_str())
                .and_then(|e| builder.to_regex_limited(e, CHECK_LIMIT));
            match rx {
                Ok(rx) => {
                    self.inner.insert(lark_regex.clone(), rx);
                }
                Err(_) => return vec![],
            }
        }
        let rx = self.inner.get_mut(lark_regex.as_str()).unwrap();
        let mut state = rx.initial_state();
        let mut prefix = vec![];
        while prefix.len() < MAX_PREFIX {
            match rx.next_byte(state) {
                NextByte::ForcedByte(b) => {
                    prefix.push(b);
                    state = rx.transition(state, b);
                }
                _ => break,
            }
        }
        prefix
    }

    pub fn check_disjoint(&mut self, regexes: &[&String]) -> Result<()> {
        // Parse every regex once; this also reports syntax errors up front.
        let mut base = RegexBuilder::new();
        let erefs = regexes
            .iter()
            .map(|regex| {
                let regex = regex_to_lark(regex, "dw");
                base.mk_regex_for_serach(regex.as_str())
            })
            .collect::<Result<Vec<_>>>()?;
        if regexes.len() < 2 {
            return Ok(());
        }
        let prefixes = regexes
            .iter()
            .map(|r| self.forced_prefix(r))
            .collect::<Vec<_>>();
        for (a, b) in pairs_to_check(&prefixes) {
            let (builder, both) = intersection(&base, erefs[a], erefs[b])?;
            let mut rx = builder.into_regex_limited(both, CHECK_LIMIT).map_err(|_| {
                anyhow!(
                    "can't determine if patternProperty regexes /{}/ and /{}/ are disjoint",
                    regex_to_lark(regexes[a], ""),
                    regex_to_lark(regexes[b], "")
                )
            })?;
            if !rx.always_empty() {
                return Err(anyhow!(
                    "patternProperty regexes /{}/ and /{}/ are not disjoint",
                    regex_to_lark(regexes[a], ""),
                    regex_to_lark(regexes[b], "")
                ));
            }
        }

        Ok(())
    }

    pub fn property_schema<'a>(&mut self, obj: &'a ObjectSchema, prop: &str) -> Result<&'a Schema> {
        if let Some(schema) = obj.properties.get(prop) {
            return Ok(schema);
        }

        for (key, schema) in obj.pattern_properties.iter() {
            if self.is_match(key, prop)? {
                return Ok(schema);
            }
        }

        Ok(obj.additional_properties.schema_ref())
    }
}

impl SharedContext {
    pub fn new() -> Self {
        SharedContext {
            defs: HashMap::default(),
            seen: HashSet::default(),
            n_compiled: 0,
            pending_warnings: Vec::new(),
            pattern_cache: PatternPropertyCache::default(),
        }
    }
}

impl Context<'_> {
    pub fn insert_ref(&self, uri: &str, schema: Schema) {
        self.shared
            .borrow_mut()
            .defs
            .insert(uri.to_string(), schema);
    }

    pub fn get_ref_cloned(&self, uri: &str) -> Option<Schema> {
        self.shared.borrow().defs.get(uri).cloned()
    }

    pub fn mark_seen(&self, uri: &str) {
        self.shared.borrow_mut().seen.insert(uri.to_string());
    }

    pub fn been_seen(&self, uri: &str) -> bool {
        self.shared.borrow().seen.contains(uri)
    }

    pub fn is_valid_keyword(&self, keyword: &str) -> bool {
        if !self.draft.is_known_keyword(keyword)
            || IMPLEMENTED.contains(&keyword)
            || META_AND_ANNOTATIONS.contains(&keyword)
        {
            return true;
        }
        false
    }

    pub fn increment(&self) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        shared.n_compiled += 1;
        if shared.n_compiled > self.options.max_size {
            bail!("schema too large");
        }
        Ok(())
    }

    pub fn record_warning(&self, msg: String) {
        self.shared.borrow_mut().pending_warnings.push(msg);
    }

    pub fn property_schema<'a>(&self, obj: &'a ObjectSchema, prop: &str) -> Result<&'a Schema> {
        self.shared
            .borrow_mut()
            .pattern_cache
            .property_schema(obj, prop)
    }

    pub fn property_schema_matches(&self, pattern: &str, name: &str) -> Result<bool> {
        self.shared
            .borrow_mut()
            .pattern_cache
            .is_match(pattern, name)
    }

    pub fn check_disjoint_pattern_properties(&self, regexes: &[&String]) -> Result<()> {
        self.shared
            .borrow_mut()
            .pattern_cache
            .check_disjoint(regexes)
    }

    pub fn into_result(self, schema: Schema) -> BuiltSchema {
        let mut shared = self.shared.borrow_mut();
        BuiltSchema {
            schema,
            definitions: std::mem::take(&mut shared.defs),
            warnings: std::mem::take(&mut shared.pending_warnings),
            pattern_cache: std::mem::take(&mut shared.pattern_cache),
        }
    }
}

pub struct BuiltSchema {
    pub schema: Schema,
    pub definitions: HashMap<String, Schema>,
    pub warnings: Vec<String>,
    pub pattern_cache: PatternPropertyCache,
}

impl BuiltSchema {
    pub fn simple(schema: Schema) -> Self {
        BuiltSchema {
            schema,
            definitions: HashMap::default(),
            warnings: Vec::new(),
            pattern_cache: PatternPropertyCache::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(cache: &mut PatternPropertyCache, keys: &[String]) -> Result<()> {
        let keys = keys.iter().collect::<Vec<_>>();
        cache.check_disjoint(&keys)
    }

    #[test]
    fn forced_prefix_examples() {
        let mut cache = PatternPropertyCache::default();
        for (regex, prefix) in [
            ("^p1_[a-z]+$", "p1_"),
            ("^key12$", "key12"),
            ("^x(a|b)$", "x"),
            ("^(ab|ac)d", "a"),
            ("^é", "é"),
            ("foo", ""),
            ("^a?b", ""),
            ("[a-z]+_1$", ""),
        ] {
            assert_eq!(
                cache.forced_prefix(regex),
                prefix.as_bytes(),
                "forced prefix of /{regex}/"
            );
        }
    }

    #[test]
    fn pairs_to_check_skips_pairs_with_distinct_prefixes() {
        let prefixes = [&b"p1_"[..], b"p10_", b"q", b""].map(|p| p.to_vec());
        assert_eq!(
            pairs_to_check(&prefixes).collect::<Vec<_>>(),
            vec![(0, 3), (1, 3), (2, 3)]
        );
        let prefixes = [&b"p1"[..], b"p10", b"p1"].map(|p| p.to_vec());
        assert_eq!(
            pairs_to_check(&prefixes).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (1, 2)]
        );
    }

    #[test]
    fn distinct_literal_prefixes_need_no_derivre_check() {
        let mut cache = PatternPropertyCache::default();
        let keys = (0..400)
            .map(|i| format!("^p{i}_[a-z]+$"))
            .collect::<Vec<_>>();
        let prefixes = keys
            .iter()
            .map(|k| cache.forced_prefix(k))
            .collect::<Vec<_>>();
        assert_eq!(pairs_to_check(&prefixes).count(), 0);
        check(&mut cache, &keys).unwrap();
    }

    #[test]
    fn prefix_disjoint_pair_beyond_fuel_is_accepted() {
        // Each key spells the same 250 "x" bytes in 5 different ways, so derivre
        // keeps 5 branches at every step. The derivre check of this pair needs more
        // than CHECK_LIMIT fuel and fails with "can't determine if ... are
        // disjoint". The forced prefixes ("x" * 250 + "A" and "x" * 250 + "B")
        // prove that the pair is disjoint, so the check passes.
        let key = |c: char| {
            let alts = (1..=5)
                .map(|i| format!("x{{{i}}}x{{{}}}{c}{i}", 250 - i))
                .collect::<Vec<_>>();
            format!("^({})$", alts.join("|"))
        };
        let mut cache = PatternPropertyCache::default();
        check(&mut cache, &[key('A'), key('B')]).unwrap();
    }

    #[test]
    fn keys_without_prefix_need_every_pair_checked() {
        // These keys have no forced literal prefix, so every pair needs a derivre
        // check, and each check must find the pair disjoint.
        let n = 60;
        let mut cache = PatternPropertyCache::default();
        let keys = (0..n).map(|i| format!("^[a-z]+_{i}$")).collect::<Vec<_>>();
        let prefixes = keys
            .iter()
            .map(|k| cache.forced_prefix(k))
            .collect::<Vec<_>>();
        assert_eq!(pairs_to_check(&prefixes).count(), n * (n - 1) / 2);
        check(&mut cache, &keys).unwrap();
    }

    #[test]
    fn intersection_of_parsed_regexes() {
        let mut base = RegexBuilder::new();
        let mut parse = |r: &str| {
            base.mk_regex_for_serach(regex_to_lark(r, "dw").as_str())
                .unwrap()
        };
        let one = parse("^[a-z]+_1$");
        let two = parse("^[a-z]+_2$");
        let digit = parse("^[a-z]+_[0-9]$");
        let parsed = base.exprset().len();
        for (a, b, disjoint) in [(one, two, true), (one, digit, false), (two, digit, false)] {
            let (builder, both) = intersection(&base, a, b).unwrap();
            // Each copy holds the parsed regexes and at most one new node.
            assert!(builder.exprset().len() <= parsed + 1);
            let mut rx = builder.into_regex_limited(both, CHECK_LIMIT).unwrap();
            assert_eq!(rx.always_empty(), disjoint);
        }
    }
}
