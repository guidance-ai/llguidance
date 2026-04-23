use crate::{regex_to_lark, HashMap};
use anyhow::{anyhow, Result};
use derivre::{Regex, RegexAst, RegexBuilder};

use super::schema::{ObjectSchema, OptSchemaExt, Schema};

#[derive(Default)]
pub struct PatternPropertyCache {
    inner: HashMap<String, Regex>,
}

const CHECK_LIMIT: u64 = 10_000;

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

    pub fn check_disjoint(&mut self, regexes: &[&String]) -> Result<()> {
        let mut builder = RegexBuilder::new();
        let erefs = regexes
            .iter()
            .map(|regex| {
                let regex = regex_to_lark(regex, "dw");
                builder.mk_regex_for_serach(regex.as_str())
            })
            .collect::<Result<Vec<_>>>()?;
        for (ai, a) in erefs.iter().enumerate() {
            for (bi, b) in erefs.iter().enumerate() {
                if ai >= bi {
                    continue;
                }
                let intersect = builder.mk(&RegexAst::And(vec![
                    RegexAst::ExprRef(*a),
                    RegexAst::ExprRef(*b),
                ]))?;
                let mut rx = builder
                    .to_regex_limited(intersect, CHECK_LIMIT)
                    .map_err(|_| {
                        anyhow!(
                            "can't determine if patternProperty regexes /{}/ and /{}/ are disjoint",
                            regex_to_lark(regexes[ai], ""),
                            regex_to_lark(regexes[bi], "")
                        )
                    })?;
                if !rx.always_empty() {
                    return Err(anyhow!(
                        "patternProperty regexes /{}/ and /{}/ are not disjoint",
                        regex_to_lark(regexes[ai], ""),
                        regex_to_lark(regexes[bi], "")
                    ));
                }
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
