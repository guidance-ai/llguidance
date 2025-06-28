use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};
use tokenizers::{normalizers::Sequence, NormalizerWrapper, Tokenizer};
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

        // remove the "Prepend space"
        if let Some(n) = hft.get_normalizer() {
            let n = match n {
                NormalizerWrapper::Sequence(x) => NormalizerWrapper::Sequence(Sequence::new(
                    x.as_ref()
                        .iter()
                        .filter_map(|n| match n {
                            NormalizerWrapper::Prepend(_) => None,
                            _ => Some(n.clone()),
                        })
                        .collect(),
                )),
                _ => n.clone(),
            };
            hft.with_normalizer(Some(n));
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
                            log::warn!("error: {} for {:?}", e, tok_name);
                            continue;
                        }
                    }
                } else {
                    panic!();
                };
                res.token_bytes[tok_id as usize] = bytes;
            } else {
                log::warn!("missing token: {}", tok_id);
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
}
