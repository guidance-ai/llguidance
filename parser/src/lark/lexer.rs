use std::fmt::Display;

use crate::{earley::regexvec::IndentState, HashMap};
use anyhow::{bail, Result};
use derivre::RegexAst;
use serde_json::{Deserializer, Value};

use crate::{
    api::ParserLimits,
    earley::{
        lexer::{Lexer, LexerResult},
        lexerspec::LexerSpec,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    KwIgnore,
    KwImport,
    KwOverride,
    KwDeclare,
    KwJson,
    KwRegex,
    KwLLGuidance,
    Colon,
    Equals,
    Comma,
    Dot,
    DotDot,
    Arrow,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Tilde,
    // regexps
    Op, // + * ?
    String,
    Regexp,
    Rule,
    Token,
    Number,
    Newline,
    VBar,
    SpecialToken, // <something>
    GrammarRef,   // @grammar_id or @7
    // special
    SKIP,
    EOF,
}

/// Represents a lexeme with its token type, value, and position.
#[derive(Debug, Clone)]
pub struct Lexeme {
    pub token: Token,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

impl Location {
    pub fn augment(&self, err: impl Display) -> anyhow::Error {
        let err = err.to_string();
        if err.starts_with("at ") {
            // don't add more location info
            anyhow::anyhow!("{err}")
        } else {
            anyhow::anyhow!("at {}({}): {}", self.line, self.column, err)
        }
    }
}

impl Token {
    const LITERAL_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Arrow, "->"),
        (Token::Colon, ":"),
        (Token::Comma, ","),
        (Token::Dot, "."),
        (Token::DotDot, ".."),
        (Token::KwDeclare, "%declare"),
        (Token::KwLLGuidance, "%llguidance"),
        (Token::KwIgnore, "%ignore"),
        (Token::KwImport, "%import"),
        (Token::KwOverride, "%override"),
        (Token::KwJson, "%json"),
        (Token::KwRegex, "%regex"),
        (Token::LParen, "("),
        (Token::RParen, ")"),
        (Token::LBrace, "{"),
        (Token::RBrace, "}"),
        (Token::LBracket, "["),
        (Token::RBracket, "]"),
        (Token::Tilde, "~"),
        (Token::VBar, "|"),
        (Token::Equals, "="),
    ];

    const REGEX_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Op, r"[+*?]"),
        (Token::Rule, r"!?[_?]?[a-z][_a-z0-9\-]*"),
        (Token::Token, r"_?[A-Z][_A-Z0-9\-]*"),
        // use JSON string syntax
        (
            Token::String,
            r#""(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])*"(i|)"#,
        ),
        (Token::Regexp, r#"/(\\.|[^/\\])+/[imslux]*"#),
        (Token::Number, r#"[+-]?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?"#),
        (Token::Newline, r"(\r?\n)+[ \t]*"),
        (Token::SpecialToken, r"<[^<>\s]+>"),
        (Token::GrammarRef, r"@[a-zA-Z0-9_\-]+"),
    ];
}

pub fn lex_lark(input: &str) -> Result<Vec<Lexeme>> {
    let comment_or_ws = r"((#|//)[^\n]*)|[ \t]+".to_string();
    let mut spec = LexerSpec::new().unwrap();
    let cls = spec
        .new_lexeme_class(RegexAst::Regex(comment_or_ws))
        .unwrap();
    let mut lexeme_idx_to_token = HashMap::default();
    lexeme_idx_to_token.insert(spec.skip_id(cls), Token::SKIP);
    for (token, literal) in Token::LITERAL_TOKENS {
        let l = spec
            .add_simple_literal(format!("{:?}", token), *literal, false)
            .unwrap();
        lexeme_idx_to_token.insert(l, *token);
    }
    for (token, regexp) in Token::REGEX_TOKENS {
        let l = spec
            .add_simple_lexeme(format!("{:?}", token), RegexAst::Regex(regexp.to_string()))
            .unwrap();
        lexeme_idx_to_token.insert(l, *token);
    }
    let mut limits = ParserLimits::default();
    let mut lexer = Lexer::from(&spec, &mut limits, false).unwrap();
    let all_lexemes = spec.all_lexemes();
    let state0 = lexer.start_state(&all_lexemes, IndentState::default());
    let mut line_no = 1;
    let mut column_no = 1;
    let mut curr_lexeme = Lexeme {
        token: Token::EOF,
        value: String::new(),
        line: 1,
        column: 1,
    };
    let mut state = state0;
    let mut lexemes = Vec::new();
    let mut start_idx = 0;

    let input = format!("{}\n", input);
    let input_bytes = input.as_bytes();

    let mut idx = 0;

    while idx <= input_bytes.len() {
        let mut b = b'\n';
        let res = if idx == input_bytes.len() {
            lexer.force_lexeme_end(state)
        } else {
            b = input_bytes[idx];
            lexer.advance(state, b, false)
        };

        match res {
            LexerResult::Error => {
                bail!("{}({}): lexer error", line_no, column_no);
            }
            LexerResult::SpecialToken(_) => {
                bail!("{}({}): lexer special token", line_no, column_no);
            }
            LexerResult::State(s, _) => {
                state = s;
            }
            LexerResult::Lexeme(p) => {
                let transition_byte = if p.byte_next_row { p.byte } else { None };
                let lx_idx = lexer.lexemes_from_idx(p.idx).first().unwrap();

                let token = lexeme_idx_to_token[&lx_idx];
                curr_lexeme.token = token;
                let mut end_idx = if p.byte_next_row || p.byte.is_none() {
                    idx
                } else {
                    idx + 1
                };

                if token == Token::KwJson || token == Token::KwLLGuidance || token == Token::KwRegex
                {
                    let (_, n_bytes) = parse_json_prefix(&input_bytes[end_idx..])?;
                    start_idx = end_idx;
                    end_idx += n_bytes;
                    for &b in &input_bytes[start_idx..end_idx - 1] {
                        if b == b'\n' {
                            line_no += 1;
                            column_no = 1;
                        } else {
                            column_no += 1;
                        }
                    }
                    // make sure we account the line ending properly at the end of the loop
                    idx = end_idx - 1;
                    b = input_bytes[idx];
                }

                curr_lexeme.value = input[start_idx..end_idx].to_string();
                start_idx = end_idx;

                // println!("lex: {:?}", curr_lexeme);

                if curr_lexeme.token != Token::SKIP {
                    lexemes.push(curr_lexeme.clone());
                }

                state = lexer.start_state(&all_lexemes, IndentState::default());
                state = lexer.transition_start_state(state, transition_byte);

                curr_lexeme.value.clear();
                curr_lexeme.line = line_no;
                curr_lexeme.column = column_no;
            }
        }

        if b == b'\n' {
            line_no += 1;
            column_no = 1;
        } else {
            column_no += 1;
        }
        idx += 1;
    }

    Ok(lexemes)
}

fn parse_json_prefix(data: &[u8]) -> Result<(Value, usize)> {
    let cursor = std::io::Cursor::new(data);
    let mut stream = Deserializer::from_reader(cursor).into_iter::<Value>();
    if let Some(result) = stream.next() {
        match result {
            Ok(v) => {
                let bytes_read = stream.byte_offset();
                Ok((v, bytes_read))
            }
            Err(e) => Err(e.into()),
        }
    } else {
        Err(anyhow::anyhow!("empty json"))
    }
}
