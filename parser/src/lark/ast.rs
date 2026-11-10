use anyhow::{anyhow, bail, Result};
use serde::{de, Deserialize, Deserializer};
use serde_json::Map;

use crate::earley::{ParamCond, ParamExpr};

use super::lexer::Location;

/// Represents an item in the grammar (rule, token, or statement).
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Item {
    Rule(Rule),
    Token(TokenDef),
    Statement(Location, Statement),
}

impl Item {
    pub fn location(&self) -> &Location {
        match self {
            Item::Rule(rule) => &rule.expansions.0,
            Item::Token(token) => &token.expansions.0,
            Item::Statement(loc, _) => loc,
        }
    }
}

/// Represents a grammar rule.
#[derive(Debug)]
pub struct Rule {
    pub name: String,
    #[allow(dead_code)]
    pub cond_inline: bool,
    #[allow(dead_code)]
    pub pin_terminals: bool,
    pub params: Option<RuleParams>,
    pub priority: Option<i32>,
    pub expansions: Expansions,
    pub is_parametric: bool,

    pub stop: Option<Value>,
    pub suffix: Option<Value>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub capture_name: Option<String>,
    pub stop_capture_name: Option<String>,
}

/// Represents a token definition.
#[derive(Debug)]
pub struct TokenDef {
    pub name: String,
    pub params: Option<TokenParams>,
    pub priority: Option<i32>,
    pub expansions: Expansions,
}

/// Represents different types of statements.
#[derive(Debug)]
pub enum Statement {
    Ignore(Expansions),
    Import {
        path: String,
        alias: Option<String>,
    },
    MultiImport {
        path: String,
        names: Vec<String>,
    },
    LLGuidance(serde_json::Value),
    #[allow(dead_code)]
    OverrideRule(Box<Rule>),
    #[allow(dead_code)]
    Declare(Vec<String>),
}

/// Represents parameters for a rule.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RuleParams(pub Vec<String>);

/// Represents parameters for a token.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TokenParams(pub Vec<String>);

/// Represents an alternative (OR) of productions in a grammar.
#[derive(Debug)]
pub struct Expansions(pub Location, pub Vec<Alias>);

impl Expansions {
    pub fn single_atom(&self) -> Option<&Atom> {
        if self.1.len() == 1
            && self.1[0].param_cond.is_true()
            && self.1[0].conjuncts.len() == 1
            && self.1[0].conjuncts[0].0.len() == 1
        {
            Some(&self.1[0].conjuncts[0].0[0].atom)
        } else {
            None
        }
    }

    pub fn take_single_atom(&mut self) -> Option<Atom> {
        if self.single_atom().is_none() {
            None
        } else {
            Some(self.1[0].conjuncts.pop().unwrap().0.pop().unwrap().atom)
        }
    }
}

/// Represents an alias in the grammar.
/// Each alias consists of possibly multiple conjuncts (AND).
#[derive(Debug)]
pub struct Alias {
    pub conjuncts: Vec<Expansion>,
    pub param_cond: ParamCond,
    #[allow(dead_code)]
    pub alias: Option<String>,
}

/// Represents a concatenation of expressions in the grammar.
#[derive(Debug)]
pub struct Expansion(pub Vec<Expr>);

/// Represents an expression.
#[derive(Debug)]
pub struct Expr {
    pub atom: Atom,
    pub op: Option<Op>,
    pub range: Option<(i32, i32)>,
}

/// Represents an atom in the grammar.
#[derive(Debug)]
pub enum Atom {
    Group(Expansions),
    Maybe(Expansions),
    Value(Value),
    Not(Box<Atom>),
}

/// Represents different values in the grammar.
#[derive(Debug)]
pub enum Value {
    LiteralRange(String, String),
    Name(String),
    NameParam(String, ParamExpr),
    LiteralString(String, String),
    LiteralRegex(String, String),
    GrammarRef(String),
    SpecialToken(String),
    Json(serde_json::Value),
    /// Configuration for the specialized ordered-interval matcher.
    IntRanges(serde_json::Value),
    NestedLark(Vec<Item>),
    RegexExt(RegexExt),
    #[allow(dead_code)]
    TemplateUsage {
        name: String,
        values: Vec<Value>,
    },
}

/// Represents an operator.
#[derive(Debug, Clone)]
pub struct Op(pub String);

impl Rule {
    pub fn stop_like(&self) -> Option<&Value> {
        self.stop.as_ref().or(self.suffix.as_ref())
    }
    pub fn take_stop_like(&mut self) -> Option<Value> {
        self.stop.take().or_else(|| self.suffix.take())
    }
    // follow guidance: "lazy": node.stop_regex != "",
    pub fn is_lazy(&self) -> bool {
        match self.stop_like() {
            Some(Value::LiteralString(s, _)) => !s.is_empty(),
            Some(_) => true,
            None => false,
        }
    }
}

/// The JSON object after `%regex`.
#[derive(Debug, Clone)]
pub enum RegexExt {
    Substring(Substring),
    Number {
        integer: bool,
        keywords: Map<String, serde_json::Value>,
    },
}

/// Matches any (possibly empty) contiguous sequence of chunks.
#[derive(Debug, Clone)]
pub enum Substring {
    Chunks(Vec<String>),
    /// Chunks from `s.split(/\s+/)`
    Words(String),
    /// Chunks from `s.split('')`
    Chars(String),
}

const SUBSTRING_KEYS: [&str; 3] = ["substring_chunks", "substring_words", "substring_chars"];
const NUMBER_KEYS: [&str; 6] = [
    "type",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

#[derive(Deserialize)]
struct SubstringFields {
    substring_chunks: Option<Vec<String>>,
    substring_words: Option<String>,
    substring_chars: Option<String>,
}

impl<'de> Deserialize<'de> for RegexExt {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = Map::deserialize(d)?;
        RegexExt::from_map(map).map_err(de::Error::custom)
    }
}

impl RegexExt {
    fn from_map(map: Map<String, serde_json::Value>) -> Result<Self> {
        if let Some(key) = map.keys().find(|key| {
            !SUBSTRING_KEYS.contains(&key.as_str()) && !NUMBER_KEYS.contains(&key.as_str())
        }) {
            bail!("unknown field `{key}`");
        }
        // Null substring fields count as unset, as with a plain `Option`.
        let substrings: Vec<&str> = SUBSTRING_KEYS
            .into_iter()
            .filter(|key| map.get(*key).is_some_and(|value| !value.is_null()))
            .collect();

        if NUMBER_KEYS.iter().any(|key| map.contains_key(*key)) {
            let kind = map
                .get("type")
                .ok_or_else(|| anyhow!("numeric %regex requires \"type\""))?;
            let integer = match kind.as_str() {
                Some("integer") => true,
                Some("number") => false,
                _ => bail!("unsupported %regex type: {kind}"),
            };
            if !substrings.is_empty() {
                bail!(
                    "only one field can be set on %regex; got {:?}",
                    [substrings, vec!["type"]].concat()
                );
            }
            return Ok(RegexExt::Number {
                integer,
                keywords: map,
            });
        }

        let fields = SubstringFields::deserialize(serde_json::Value::Object(map))?;
        let substring = match (
            fields.substring_chunks,
            fields.substring_words,
            fields.substring_chars,
        ) {
            (Some(chunks), None, None) => Substring::Chunks(chunks),
            (None, Some(words), None) => Substring::Words(words),
            (None, None, Some(chars)) => Substring::Chars(chars),
            (None, None, None) => bail!("no fields set on %regex"),
            _ => bail!("only one field can be set on %regex; got {substrings:?}"),
        };
        Ok(RegexExt::Substring(substring))
    }
}
