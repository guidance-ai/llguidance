use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};
use tokenizers::{normalizers, pre_tokenizers, NormalizerWrapper, PreTokenizerWrapper, Tokenizer};
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

pub struct ByteTokenizer {
    pub hf_model: String,
    pub hf_tokenizer: Tokenizer,
    info: TokRxInfo,
    token_bytes: Vec<Vec<u8>>,
}

// useful when debugging this: https://www.cogsci.ed.ac.uk/~richard/utf-8.cgi

fn is_self_mapped(c: char) -> bool {
    matches!(c, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}')
}

fn build_char_map() -> HashMap<char, u8> {
    let mut res = HashMap::default();
    let mut k = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if is_self_mapped(c) {
            res.insert(c, byte);
        } else {
            res.insert(char::from_u32(k).unwrap(), byte);
            k += 1;
        }
    }
    res
}

impl ByteTokenizer {
    pub fn from_file(name: impl AsRef<Path>) -> Result<ByteTokenizer> {
        let name_str = name.as_ref().display().to_string();
        let tok = Tokenizer::from_file(name)
            .map_err(|e| anyhow!("error loading tokenizer: {}: {}", name_str, e))?;
        ByteTokenizer::from_tokenizer(tok)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<ByteTokenizer> {
        let tok =
            Tokenizer::from_bytes(bytes).map_err(|e| anyhow!("error loading tokenizer: {}", e))?;
        ByteTokenizer::from_tokenizer(tok)
    }

    pub fn from_tokenizer(mut hft: Tokenizer) -> Result<ByteTokenizer> {
        let mut is_byte_level = false;
        let mut is_byte_fallback = false;
        let mut space_ch = ' ';

        // remove the "Prepend space" normalizer if present
        fn remove_prepend_normalizer(n: NormalizerWrapper) -> Option<NormalizerWrapper> {
            match n {
                NormalizerWrapper::Prepend(_) => None,
                NormalizerWrapper::Sequence(x) => {
                    let filtered: Vec<_> = x
                        .as_ref()
                        .iter()
                        .filter_map(|n| remove_prepend_normalizer(n.clone()))
                        .collect();
                    if filtered.is_empty() {
                        None
                    } else {
                        Some(NormalizerWrapper::Sequence(normalizers::Sequence::new(
                            filtered,
                        )))
                    }
                }
                _ => Some(n),
            }
        }

        let norm = hft.get_normalizer().cloned();
        if let Some(n) = norm {
            hft.with_normalizer(remove_prepend_normalizer(n));
        }

        // fix pre-tokenizers that prepend spaces (e.g., Metaspace with prepend_scheme: First/Always)
        fn fix_metaspace(pt: PreTokenizerWrapper) -> PreTokenizerWrapper {
            match pt {
                PreTokenizerWrapper::Metaspace(ms) => {
                    let mut ms = ms.clone();
                    ms.prepend_scheme = pre_tokenizers::metaspace::PrependScheme::Never;
                    PreTokenizerWrapper::Metaspace(ms)
                }
                PreTokenizerWrapper::Sequence(x) => {
                    PreTokenizerWrapper::Sequence(pre_tokenizers::sequence::Sequence::new(
                        x.as_ref()
                            .iter()
                            .map(|pt| fix_metaspace(pt.clone()))
                            .collect(),
                    ))
                }
                _ => pt,
            }
        }

        let pretok = hft.get_pre_tokenizer().cloned();
        if let Some(pt) = pretok {
            hft.with_pre_tokenizer(Some(fix_metaspace(pt)));
        }

        if let Some(d) = hft.get_decoder() {
            // DecoderWrapper::Sequence() doesn't let one access the decoders
            // so we resort to json munching
            let v = serde_json::to_value(d).unwrap();
            if v["type"].as_str() == Some("ByteLevel") {
                is_byte_level = true;
            } else if v["type"].as_str() == Some("Sequence") {
                if let Some(decoders) = v["decoders"].as_array() {
                    for decoder in decoders {
                        if decoder["type"].as_str() == Some("ByteFallback") {
                            is_byte_fallback = true;
                        } else if decoder["type"].as_str() == Some("Replace")
                            && decoder["content"].as_str() == Some(" ")
                        {
                            if let Some(s) = decoder["pattern"]["String"].as_str() {
                                let s: Vec<char> = s.chars().collect();
                                if s.len() == 1 {
                                    space_ch = s[0];
                                }
                            }
                        }
                    }
                }
            }
        }

        if !is_byte_fallback && !is_byte_level {
            bail!("can't determine decoder type: {:?}", hft.get_decoder());
        }

        let vocab_size = hft.get_vocab_size(true) as u32;
        let mut added = hft
            .get_added_tokens_decoder()
            .into_iter()
            .collect::<Vec<_>>();
        added.sort_by_key(|(id, _)| *id);

        let mut res = ByteTokenizer {
            hf_model: "foobar".to_string(),
            info: TokRxInfo::new(vocab_size, 0),
            token_bytes: (0..vocab_size).map(|_| Vec::new()).collect(),
            hf_tokenizer: hft,
        };

        let mut specials = HashSet::new();

        for (id, info) in added.iter() {
            // we treat all added tokens of the form <...> as special tokens
            if info.special || (info.content.starts_with("<") && info.content.ends_with(">")) {
                match info.content.as_str() {
                    "</s>"
                    | "<|endoftext|>"
                    | "<|end_of_text|>"
                    | "<｜end▁of▁sentence｜>" // funky bars from DeepSeek tokenizer
                    | "<eos>" => res.info.tok_eos = *id,

                    "<|end|>" | "<|eot_id|>" | "<|im_end|>" => res.info.tok_end_of_turn = Some(*id),
                    "<unk>" | "<|unk|>" => res.info.tok_unk = Some(*id),
                    "<pad>" | "<|pad|>" => res.info.tok_pad = Some(*id),
                    _ => {}
                }
                specials.insert(*id);
            } else {
                res.token_bytes[*id as usize] = info.content.clone().into_bytes();
            }
        }

        let char_map = build_char_map();

        for tok_id in 0..vocab_size {
            if let Some(tok_name) = res.hf_tokenizer.id_to_token(tok_id) {
                let bytes = if specials.contains(&tok_id) {
                    let mut bytes = tok_name.as_bytes().to_vec();
                    bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                    bytes
                } else if is_byte_fallback {
                    if tok_name.len() == 6 && tok_name.starts_with("<0x") && tok_name.ends_with(">")
                    {
                        // parse hex number from tok_name
                        let hex_str = &tok_name[3..5];
                        let byte = u8::from_str_radix(hex_str, 16).unwrap();
                        vec![byte]
                    } else {
                        assert!(!tok_name.starts_with("<0x"));
                        let tok_name = tok_name.replace(space_ch, " ");
                        tok_name.as_bytes().to_vec()
                    }
                } else if is_byte_level {
                    let bytes: Result<Vec<u8>> = tok_name
                        .chars()
                        .map(|c| {
                            char_map
                                .get(&c)
                                .copied()
                                .ok_or_else(|| anyhow!("missing char: {}", c))
                        })
                        .collect();
                    match bytes {
                        Ok(b) => b,
                        Err(e) => {
                            log::warn!("error: {e} for {tok_name:?}");
                            continue;
                        }
                    }
                } else {
                    panic!();
                };
                res.token_bytes[tok_id as usize] = bytes;
            } else {
                log::warn!("missing token: {tok_id}");
            }
        }

        Ok(res)
    }

    pub fn tokrx_info(&self) -> TokRxInfo {
        self.info
    }
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.token_bytes.clone()
    }

    pub fn set_eos_token(&mut self, tok_id: u32) {
        self.info.tok_eos = tok_id;
    }

    pub fn into_tok_env(self, n_vocab: Option<usize>) -> Result<TokEnv> {
        let b = ByteTokenizerEnv::new(self, n_vocab)?;
        Ok(b.to_env())
    }
}

pub struct ByteTokenizerEnv {
    pub tokenizer: ByteTokenizer,
    pub tok_trie: TokTrie,
}

impl ByteTokenizerEnv {
    pub fn new(tokenizer: ByteTokenizer, n_vocab: Option<usize>) -> Result<ByteTokenizerEnv> {
        let mut info = tokenizer.tokrx_info();
        let mut token_bytes = tokenizer.token_bytes();
        if let Some(n_vocab) = n_vocab {
            if n_vocab < token_bytes.len() {
                bail!("vocab size too small; {} vs {}", n_vocab, token_bytes.len());
            }
            while n_vocab > token_bytes.len() {
                let mut name = format!(".<[{}]>", token_bytes.len()).into_bytes();
                name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                token_bytes.push(name);
            }
            info.vocab_size = n_vocab as u32;
        }
        let tok_trie = TokTrie::from(&info, &token_bytes);
        Ok(ByteTokenizerEnv {
            tokenizer,
            tok_trie,
        })
    }

    pub fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for ByteTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenizer
                .hf_tokenizer
                .encode(s, false)
                .expect("tokenizer error")
                .get_ids()
                .to_vec()
        })
    }

    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tok_trie.tokenize_with_special(s, |s| {
                self.tokenizer
                    .hf_tokenizer
                    .encode(s, false)
                    .expect("tokenizer error")
                    .get_ids()
                    .to_vec()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use tokenizers::Tokenizer;

    const MINIMAL_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "vocab": {
                "a": 0
            },
            "merges": []
        }
    }"#;

    #[test]
    fn tokenize_special_respects_toktrie_specials() {
        let hf_tokenizer = Tokenizer::from_str(MINIMAL_TOKENIZER_JSON).unwrap();
        let info = TokRxInfo::new(2, 0);
        let mut token_bytes = vec![b"a".to_vec()];
        let mut special_bytes = Vec::new();
        special_bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
        special_bytes.extend_from_slice(b"<|end|>");
        token_bytes.push(special_bytes);
        let tokenizer = ByteTokenizer {
            hf_model: "test".to_string(),
            hf_tokenizer,
            info,
            token_bytes,
        };
        let env = ByteTokenizerEnv::new(tokenizer, None).unwrap();
        let special_id = env.tok_trie().get_special_token("<|end|>").unwrap();
        assert_eq!(env.tokenize("<|end|>"), Vec::<TokenId>::new());
        let tokens = env.tokenize_special("<|end|>");
        assert_eq!(
            tokens,
            vec![special_id],
            "got: {:?}, want: {:?}",
            tokens,
            vec![special_id]
        );
    }

    use rstest::rstest;

    #[rstest]
    #[case::metaspace_always_top_level(
        r#"null"#,
        r#"{
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": false
        }"#
    )]
    #[case::metaspace_always_nested_in_sequence(
        r#"null"#,
        r#"{
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "prepend_scheme": "always",
                    "split": false
                }
            ]
        }"#
    )]
    #[case::metaspace_first_top_level(
        r#"null"#,
        r#"{
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "first",
            "split": false
        }"#
    )]
    #[case::metaspace_first_nested_in_sequence(
        r#"null"#,
        r#"{
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "prepend_scheme": "first",
                    "split": false
                }
            ]
        }"#
    )]
    #[case::prepend_normalizer_top_level(
        r#"{
            "type": "Prepend",
            "prepend": "▁"
        }"#,
        r#"null"#
    )]
    #[case::prepend_normalizer_nested_in_sequence(
        r#"{
            "type": "Sequence",
            "normalizers": [
                {
                    "type": "Prepend",
                    "prepend": "▁"
                }
            ]
        }"#,
        r#"null"#
    )]
    fn test_tokenizer_fixes(#[case] normalizer: &str, #[case] pre_tokenizer: &str) {
        let tokenizer_json = format!(
            r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {normalizer},
            "pre_tokenizer": {pre_tokenizer},
            "post_processor": null,
            "decoder": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            }},
            "model": {{
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "vocab": {{
                    "▁a": 0,
                    "▁>": 1,
                    "a": 2,
                    ">": 3,
                    "▁": 4
                }},
                "merges": [
                    "▁ a",
                    "▁ >"
                ]
            }}
        }}"#
        );

        let hf_tokenizer = Tokenizer::from_str(&tokenizer_json).unwrap();

        // Before fix: tokenizer would add unwanted ▁ prefix
        let before_encoded = hf_tokenizer.encode("a>", false).unwrap();
        let before_ids = before_encoded.get_ids();
        assert_eq!(before_ids, vec![0, 3], "Before fix: expected [▁a, >]");

        // Create ByteTokenizer which should apply the fixes
        let tokenizer = ByteTokenizer::from_tokenizer(hf_tokenizer).unwrap();

        // After fix: tokenizer should NOT add unwanted prefixes
        let after_encoded = tokenizer.hf_tokenizer.encode("a>", false).unwrap();
        let after_ids = after_encoded.get_ids();
        assert_eq!(after_ids, vec![2, 3], "After fix: expected [a, >]");
    }
}
