use anyhow::{anyhow, bail, ensure, Result};
use clap::Parser;
use indexmap::IndexMap;
use json_stats::SchemaStats;
use jsonschema::Validator;
use llguidance::{
    api::{GrammarInit, StopReason, TopLevelGrammar},
    earley::{perf::num_with_commas, regexvec::LexerStats, XorShift},
    toktrie::{InferenceCapabilities, SimpleVob, TokEnv},
    Constraint, HashMap, JsonCompileOptions, ParserFactory, TokenParser,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::File,
    io::{Read, Write},
    sync::{atomic::AtomicUsize, Arc},
};

use rayon::prelude::*;

struct DummyResolver {}
impl jsonschema::Retrieve for DummyResolver {
    fn retrieve(
        &self,
        uri: &jsonschema::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(anyhow!("external resolver disabled (url: {})", uri).into())
    }
}

#[derive(Parser, Debug, Default)]
#[command(version, about, long_about = None)]
pub struct CliOptions {
    /// Measure grammar compilation times (only)
    #[arg(long)]
    llg_compile: bool,

    /// Validate each token (only)
    #[arg(long)]
    llg_validate_tokens: bool,

    /// Measure mask computation times (full test; default)
    #[arg(long)]
    llg_masks: bool,

    /// Don't run any llg tests, just re-write test files
    #[arg(long)]
    rewrite_files: bool,

    /// Disable the slicer optimization
    #[arg(long)]
    llg_disable_slicer: bool,

    /// Disable ff_tokens
    #[arg(long)]
    llg_no_forcing: bool,

    /// Set stderr log level; implies --num-threads 1
    #[arg(long, default_value = "0")]
    llg_log_level: u32,

    /// Test the slicer optimization against un-sliced parser
    #[arg(long)]
    llg_test_slicer: bool,

    /// Use white-space inflexible grammar (force no whitespace in JSON).
    #[arg(long)]
    compact: bool,

    /// Validate results against known good results; similar to 'diff FILE tmp/llg_sem_results.json'
    #[arg(long)]
    expected: Option<String>,

    /// Don't report missing or similar results for --expected
    #[arg(long)]
    ballpark: bool,

    /// Print out CSV mask computation histogram
    #[arg(long)]
    csv: bool,

    /// Don't print JSON output and perf counters
    #[arg(long)]
    quiet: bool,

    /// Test rollback mechanism for speculative decoding
    #[arg(long)]
    rollback: bool,

    /// Run that many threads; defaults to min(40, cpus())
    #[arg(long)]
    num_threads: Option<usize>,

    /// Delete tests that are deemed invalid by jsonschema library
    #[arg(long)]
    remove_broken_tests: bool,

    /// Skip tests with "Handwritten" or "Synthesized" in name
    #[arg(long)]
    skip_synth: bool,

    /// Generate additional JSON schema features in 'meta' fields
    #[arg(long)]
    additional_features: bool,

    /// Specify HF tokenizer to use
    #[arg(long, default_value = "unsloth/Meta-Llama-3.1-8B-Instruct")]
    tokenizer: String,

    /// Only process files with specified string in the name
    #[arg(long)]
    filter: Option<String>,

    /// Only process '"valid": true' testcases
    #[arg(long)]
    only_valid: bool,

    /// Only process '"valid": false' testcases
    #[arg(long)]
    only_invalid: bool,

    /// Treat all schemas as { "type": "object" }
    #[arg(long)]
    ignore_schema: bool,

    /// .json files or folders with .json files
    #[arg(value_name = "FILES")]
    files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
struct LlgSemanticResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    json_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parser_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_error: Option<String>,
}

impl LlgSemanticResult {
    fn from_llg_result(r: &LlgResult) -> Self {
        Self {
            json_error: r.json_error.clone(),
            parser_error: r.parser_error.clone(),
            validation_error: r.validation_error.clone(),
        }
    }

    fn error_badness(&self) -> usize {
        if self.json_error.is_some() {
            10
        } else if self.parser_error.is_some() {
            5
        } else if self.validation_error.is_some() {
            2
        } else {
            0
        }
    }

    fn info(&self) -> String {
        if let Some(e) = &self.json_error {
            format!("JSON: {}", short_limit_string(e))
        } else if let Some(e) = &self.parser_error {
            format!("PARSER: {}", short_limit_string(e))
        } else if let Some(e) = &self.validation_error {
            format!("VALIDATION: {}", short_limit_string(e))
        } else {
            "OK".to_string()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct LlgResult {
    id: String,

    ttfm_us: usize,
    json_compile_us: usize,
    parser_create_us: usize,
    first_mask_us: usize,
    max_ttfm_us: usize,
    max_mask_us: usize,
    slicer_leftover_us: usize,

    one: usize,

    num_tokens: usize,
    num_tests: usize,
    num_valid_tests: usize,
    num_invalid_tests: usize,
    num_all_tokens: usize,

    avg_parser_items: usize,
    max_avg_parser_items: usize,
    sum_parser_items: usize,
    max_sum_parser_items: usize,
    max_parser_items: usize,
    max_lexer_cost: u64,
    max_lexer_states: usize,
    lexer_cost: u64,
    trie_nodes_walked: usize,

    lexer_stats: LexerStats,

    all_mask_us: Vec<usize>,
    ff_tokens_us: Vec<usize>,

    rollback_tokens: usize,

    #[serde(skip)]
    all_hash: Vec<u64>,

    #[serde(skip)]
    all_mask_us_a: Vec<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    json_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parser_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonTest {
    #[serde(default)]
    description: String,
    meta: Option<JsonMetaInfo>,
    schema: Value,
    #[serde(default)]
    tests: Vec<JsonTestSequence>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct JsonMetaInfo {
    pub full_size: usize,
    pub stripped_size: usize,
    pub features: Vec<String>,
    #[serde(default)]
    pub raw_features: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct JsonFileInfo {
    pub id: String,
    pub meta: JsonMetaInfo,
    pub num_valid_tests: usize,
    pub num_invalid_tests: usize,
    pub size_valid_tests: usize,
    pub size_invalid_tests: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonTestSequence {
    #[serde(default)]
    description: String,
    valid: bool,
    #[serde(skip)]
    broken: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    rust_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    python_error: Option<String>,
    data: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SchemaRes {
    file_name: String,
    full_size: usize,
    file_info: JsonFileInfo,
    llg_result: Option<LlgResult>,

    test_valid_error: Option<String>,
    test_invalid_error: Option<String>,
    schema_error: Option<String>,
}

#[derive(Clone)]
struct TestEnv {
    cli: Arc<CliOptions>,
    tok_env: TokEnv,
    factory: Arc<ParserFactory>,
    ref_factory: Arc<ParserFactory>,
    file_name: String,
    hash_rnd: Arc<ahash::RandomState>,
}

fn json_sum(curr: &mut Value, v: &Value) {
    assert!(curr.is_object());
    assert!(v.is_object());
    let v = v.as_object().unwrap();
    for (k, v) in v.iter() {
        if let Some(v) = v.as_i64() {
            let c = &curr[k];
            let v2 = if c.is_null() {
                v
            } else if k.starts_with("max_") {
                std::cmp::max(c.as_i64().unwrap(), v)
            } else {
                c.as_i64().unwrap() + v
            };
            curr[k] = json!(v2);
        }
    }
}

fn log_fraction_plot(times: &mut [usize]) -> String {
    times.sort();
    let mut cutoff = 1;
    let mult = 1.3;
    let mut count = 0;
    let mut csv = String::from("cutoff time,frac left\n");
    let total = times.len() as f64;

    for &t in times.iter() {
        while t > cutoff {
            csv.push_str(&format!(
                "{:.3},{}\n",
                cutoff as f64 / 1000.0,
                (total - count as f64) / total
            ));
            cutoff = (cutoff as f64 * mult).floor() as usize + 1;
        }
        count += 1;
    }

    csv.push_str(&format!(
        "{:.3},{}\n",
        cutoff as f64 / 1000.0,
        (total - count as f64) / total
    ));

    csv
}

enum MaskResult {
    Accept { n_tokens: usize },
    Reject { reason: String },
}

enum RefResult {
    None,
    Mask(SimpleVob),
    Forced(Vec<u32>),
}

impl TestEnv {
    fn check_mask(
        &self,
        stats: &mut LlgResult,
        parser: &mut TokenParser,
        mut ref_parser: Option<&mut TokenParser>,
        roll_result: &RefResult,
        tidx: usize,
        tokens: &[u32],
    ) -> Result<MaskResult> {
        let trie = self.tok_env.tok_trie();
        let token = tokens[tidx];

        let t0 = std::time::Instant::now();

        if !self.cli.llg_no_forcing {
            let forced = parser.compute_ff_tokens();
            if !forced.is_empty() {
                let us = t0.elapsed().as_micros() as usize;
                stats.ff_tokens_us.push(us);
                let endp = std::cmp::min(tokens.len(), tidx + forced.len());

                match roll_result {
                    RefResult::None => {}
                    RefResult::Mask(_) => {
                        bail!("rollback produced mask, but main parser forced tokens")
                    }
                    RefResult::Forced(items) => {
                        ensure!(items == &forced, "forced tokens mismatch (rollback)")
                    }
                }

                if &tokens[tidx..endp] == forced.as_slice() {
                    return Ok(MaskResult::Accept {
                        n_tokens: forced.len(),
                    });
                } else {
                    return Ok(MaskResult::Reject {
                        reason: format!(
                            "forced tokens {:?} != {:?}",
                            trie.tokens_dbg(&forced),
                            trie.tokens_dbg(&tokens[tidx..endp])
                        ),
                    });
                }
            }
        }

        let m = parser.compute_mask()?; // .unwrap_or_else(|_| trie.alloc_token_set());

        // let hash = 0;

        let hash = self.hash_rnd.hash_one(m.as_slice());

        // use ring::digest::{digest, SHA256};
        // let hash = digest(&SHA256, bytemuck::cast_slice(m.as_slice()));
        // let hash = u64::from_be_bytes(hash.as_ref()[0..8].try_into().unwrap());

        // use sha2::{Sha256, Digest};
        // let mut hasher = Sha256::new();
        // hasher.update(bytemuck::cast_slice(m.as_slice()));
        // let hash = u64::from_be_bytes(hasher.finalize().as_slice()[0..8].try_into().unwrap());

        let mask_us = t0.elapsed().as_micros() as usize;
        let pstats = parser.last_step_stats();

        stats.all_hash.push(hash);

        if let Some(ref_parser) = &mut ref_parser {
            let m2 = ref_parser.compute_mask()?;
            if m != m2 {
                let mut missing_slicer = m2.clone();
                missing_slicer.sub(&m);
                let mut missing_parser = m.clone();
                missing_parser.sub(&m2);
                eprintln!(
                    "{}:\n{}\n{}",
                    tidx,
                    trie.token_set_dbg(&missing_slicer),
                    trie.token_set_dbg(&missing_parser)
                );
                panic!("mismatch");
            }
        }

        stats.all_mask_us.push(mask_us);

        // && pstats.lexer_cost < 7 * us as u64
        if self.cli.csv && mask_us > 300 {
            static CSV_LINE: AtomicUsize = AtomicUsize::new(0);
            let line_no = CSV_LINE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if line_no == 0 {
                println!("MASK,file,us,lexer_cost,slices,items,rows,cached_rows,trie_nodes,allowed_tokens,est_time");
            }
            println!(
                "{},{},{},{},{},{},{},{},{},{},{}",
                if mask_us > 1000 { "SLOW" } else { "OK" },
                stats.id,
                mask_us,
                pstats.lexer_cost,
                pstats.slices_applied,
                pstats.all_items,
                pstats.rows,
                pstats.cached_rows,
                pstats.trie_nodes_walked,
                m.num_set(),
                (pstats.trie_nodes_walked as u64 * 4 + pstats.lexer_cost * 60) / 1000
            );
            // eprintln!("{}", parser.parser.lexer_stats());

            // eprintln!("{:?}", pstats);
        }

        stats.sum_parser_items += pstats.all_items;
        stats.max_parser_items = std::cmp::max(stats.max_parser_items, pstats.all_items);
        stats.max_lexer_cost = std::cmp::max(stats.max_lexer_cost, pstats.lexer_cost);
        stats.lexer_cost += pstats.lexer_cost;
        stats.trie_nodes_walked += pstats.trie_nodes_walked;

        let is_big = m.num_set() >= 128_000 / 100 * 85;
        let sliced = pstats.slices_applied > 0;
        let cond_a = is_big && !sliced;
        if cond_a {
            stats.all_mask_us_a.push(mask_us);
        }

        stats.max_mask_us = std::cmp::max(stats.max_mask_us, mask_us);

        match roll_result {
            RefResult::None => {}
            RefResult::Mask(m2) => {
                ensure!(m == *m2, "mask mismatch (rollback)")
            }
            RefResult::Forced(_) => {
                bail!("rollback forced tokens, but main parser didn't")
            }
        }

        if m.is_allowed(token) {
            Ok(MaskResult::Accept { n_tokens: 1 })
        } else {
            Ok(MaskResult::Reject {
                reason: format!("mask: {}", trie.token_set_dbg(&m)),
            })
        }
    }

    fn try_rollback(
        &self,
        stats: &mut LlgResult,
        rnd: &mut XorShift,
        parser: &mut TokenParser,
    ) -> Result<RefResult> {
        let mut n_tokens = 0;
        loop {
            if rnd.one_in(10) {
                break;
            }
            let m = parser.compute_mask();
            if m.is_err() && parser.stop_reason() == StopReason::NoExtensionBias {
                break;
            }
            let m = m?;
            let tok = rnd.sample_from_vob(&m);
            let bt = parser.consume_token(tok)?;
            assert!(bt == 0);
            n_tokens += 1;
        }
        parser.rollback(n_tokens)?;
        stats.rollback_tokens += n_tokens;

        if !self.cli.llg_no_forcing {
            let forced = parser.compute_ff_tokens();
            if !forced.is_empty() {
                return Ok(RefResult::Forced(forced));
            }
        }

        let m = parser.compute_mask()?;
        Ok(RefResult::Mask(m))
    }

    fn run_llg_test_inner(
        &self,
        stats: &mut LlgResult,
        parser: &mut TokenParser,
        mut ref_parser: Option<TokenParser>,
        t: &JsonTestSequence,
    ) -> Result<()> {
        let dstr = serde_json::to_string(&t.data).unwrap();
        let mut rnd = XorShift::new_str(&dstr);
        let tokens = self.tok_env.tokenize(&dstr);
        let trie = self.tok_env.tok_trie();
        let masks = self.cli.llg_masks;

        let mut roll_parser = if self.cli.rollback {
            Some(parser.deep_clone())
        } else {
            None
        };

        stats.num_tests += 1;

        // println!("tokenized: {}", trie.tokens_dbg(&tokens));

        let mut tidx = 0;

        while tidx < tokens.len() {
            // eprintln!("WILL TEST {}: {}", tidx, trie.token_dbg(token));

            let roll_result = if let Some(roll_parser) = &mut roll_parser {
                self.try_rollback(stats, &mut rnd, roll_parser)?
            } else {
                RefResult::None
            };

            let mask_res = if masks {
                self.check_mask(
                    stats,
                    parser,
                    ref_parser.as_mut(),
                    &roll_result,
                    tidx,
                    &tokens,
                )?
            } else if parser.validate_token(tokens[tidx])? {
                MaskResult::Accept { n_tokens: 1 }
            } else {
                MaskResult::Reject {
                    reason: "validate_token".to_string(),
                }
            };

            let n_tokens = match mask_res {
                MaskResult::Accept { n_tokens: nt } => nt,
                MaskResult::Reject { reason } => {
                    stats.num_tokens += 1;
                    if t.valid {
                        bail!(
                            "token not accepted at {} * {} * {} {}",
                            trie.tokens_dbg(&tokens[tidx.saturating_sub(50)..tidx]),
                            trie.tokens_dbg(&tokens[tidx..tidx + 1]),
                            trie.tokens_dbg(
                                &tokens[tidx + 1..std::cmp::min(tidx + 5, tokens.len())]
                            ),
                            reason
                        )
                    } else {
                        return Ok(());
                    }
                }
            };

            stats.num_tokens += n_tokens;

            for _ in 0..n_tokens {
                let token = tokens[tidx];
                let bt = parser.consume_token(token)?;
                assert!(bt == 0);

                if let Some(ref_parser) = &mut ref_parser {
                    let bt = ref_parser.consume_token(token)?;
                    assert!(bt == 0);
                }

                if let Some(roll_parser) = &mut roll_parser {
                    let bt = roll_parser.consume_token(token)?;
                    assert!(bt == 0);
                }

                tidx += 1;
            }
        }

        if parser.is_accepting() {
            if !t.valid && !self.cli.ignore_schema {
                bail!(
                    "incorrect accept; expected {}",
                    t.rust_error
                        .clone()
                        .unwrap_or_else(|| t.python_error.clone().unwrap_or("???".to_string()))
                );
            }
        } else if t.valid {
            bail!("parser not accepting at the end");
        }

        Ok(())
    }

    fn run_llg_test(
        &self,
        stats: &mut LlgResult,
        parser: &TokenParser,
        ref_parser: Option<&TokenParser>,
        t: &JsonTestSequence,
    ) -> Result<()> {
        let mut parser = parser.deep_clone();
        parser.start_without_prompt();

        let mut ref_parser = ref_parser.map(|p| p.deep_clone());
        if let Some(p) = ref_parser.as_mut() {
            p.start_without_prompt()
        }

        let r = self.run_llg_test_inner(stats, &mut parser, ref_parser, t);

        let m = parser.parser.metrics_mut();
        stats.slicer_leftover_us += m.slicer_leftover_us;

        let lx = parser.parser.lexer_stats();
        stats.max_lexer_states = std::cmp::max(stats.max_lexer_states, lx.num_states);

        r
    }

    fn run_llg_compile(&self, id: &str, test_file: &JsonTest) -> LlgResult {
        let opts = JsonCompileOptions {
            whitespace_flexible: !self.cli.compact,
            ..Default::default()
        };
        let mut res = LlgResult {
            id: id.to_string(),
            ..Default::default()
        };

        let all_tests = test_file
            .tests
            .iter()
            .filter(|t| {
                (self.cli.only_valid && t.valid)
                    || (self.cli.only_invalid && !t.valid)
                    || (!self.cli.only_valid && !self.cli.only_invalid)
            })
            .collect::<Vec<_>>();
        for t in &all_tests {
            let dstr = serde_json::to_string(&t.data).unwrap();
            let tokens = self.tok_env.tokenize(&dstr);
            res.num_all_tokens += tokens.len();
        }

        let t0 = std::time::Instant::now();
        let mut schema = if self.cli.ignore_schema {
            json!({"type": "object"})
        } else {
            test_file.schema.clone()
        };
        opts.apply_to(&mut schema);
        let g_init = GrammarInit::Serialized(TopLevelGrammar::from_json_schema(schema));
        let g_init = g_init.to_internal(None, self.factory.limits().clone());

        res.json_compile_us = t0.elapsed().as_micros() as usize;

        let (grm, mut lex_spec) = match g_init {
            Ok(schema) => schema,
            Err(e) => {
                res.json_error = Some(format!("{e}"));
                if self.cli.llg_log_level > 0 {
                    eprintln!("{} Error JSON: {}", self.file_name, e);
                }
                limit_string(&mut res.json_error);
                return res;
            }
        };

        if self.cli.llg_no_forcing {
            lex_spec.no_forcing = true;
        }

        let g_init = GrammarInit::Internal(grm, lex_spec);

        let ref_parser = if self.cli.llg_test_slicer {
            Some(
                self.ref_factory
                    .create_parser_from_init_default(g_init.clone()),
            )
        } else {
            None
        };

        let t1 = std::time::Instant::now();
        let parser = self.factory.create_parser_from_init_default(g_init);
        res.parser_create_us = t1.elapsed().as_micros() as usize;

        let t2 = std::time::Instant::now();
        let parser = match parser {
            Ok(parser) => {
                let mut constraint = Constraint::new(parser.clone());
                constraint.compute_mask().unwrap();
                res.first_mask_us = t2.elapsed().as_micros() as usize;
                res.ttfm_us = t0.elapsed().as_micros() as usize;
                res.max_ttfm_us = res.ttfm_us;
                res.one = 1;
                parser
                // eprintln!("{} OK", file);
            }
            Err(e) => {
                // eprintln!("{} Error Parser: {}", self.file_name, e);
                res.parser_error = Some(format!("{e}"));
                if self.cli.llg_log_level > 0 {
                    eprintln!("{} Error JSON: {}", self.file_name, e);
                }
                limit_string(&mut res.parser_error);
                return res;
            }
        };

        res.lexer_stats = parser.parser.lexer_stats();

        let ref_parser = ref_parser.map(|p| p.unwrap());

        if self.cli.llg_validate_tokens {
            for (idx, t) in all_tests.iter().enumerate() {
                if let Err(e) = self.run_llg_test(&mut res, &parser, ref_parser.as_ref(), t) {
                    if res.validation_error.is_none() {
                        res.validation_error = Some(format!("test #{idx}: {e}"));
                        if self.cli.llg_log_level > 0 {
                            eprintln!("{} Error Validating: {}", self.file_name, e);
                        }
                        limit_string(&mut res.validation_error);
                    }
                } else if t.valid {
                    res.num_valid_tests += 1;
                } else {
                    res.num_invalid_tests += 1;
                }
            }

            let n_masks = res.all_mask_us.len();
            if n_masks > 0 {
                res.avg_parser_items = res.sum_parser_items / n_masks;
                res.max_avg_parser_items = res.sum_parser_items / n_masks;
            }
        }

        res
    }

    fn run_test(&self) -> SchemaRes {
        let file_name = &self.file_name;
        let schema_file = read_file_to_string(file_name);
        let mut test_file: JsonTest = serde_json::from_str(&schema_file)
            .unwrap_or_else(|_| panic!("Invalid JSON in schema file {}", file_name));

        let mut stats = SchemaRes {
            file_name: file_name.clone(),
            full_size: serde_json::to_string(&test_file.schema).unwrap().len(),
            ..Default::default()
        };

        let uuid_regex = regex::Regex::new(r"^(?P<time_low>[0-9a-fA-F]{8})-(?P<time_mid>[0-9a-fA-F]{4})-(?P<time_high_and_version>[0-9a-fA-F]{4})-(?P<clock_seq_and_reserved>[0-9a-fA-F]{2})(?P<clock_seq_low>[0-9a-fA-F]{2})-(?P<node>[0-9a-fA-F]{12})$"
    ).unwrap();
        let iri_regex = regex::Regex::new(
        r"^(?P<scheme>[A-Za-z][A-Za-z0-9+\-.]*):(?:\/\/(?P<authority>[^\s/?#]+))?(?P<path>[^\s?#]*)(?:\?(?P<query>[^\s#]*))?(?:#(?P<fragment>\S*))?$"
    ).unwrap();
        let duration_regex = regex::Regex::new(
        r"^P(?:(?P<dur_date>(?:(?P<dur_year>[0-9]+Y(?:[0-9]+M(?:[0-9]+D)?)?)|(?P<dur_month>[0-9]+M(?:[0-9]+D)?)|(?P<dur_day>[0-9]+D))(?:T(?:(?P<dur_hour>[0-9]+H(?:[0-9]+M(?:[0-9]+S)?)?)|(?P<dur_minute>[0-9]+M(?:[0-9]+S)?)|(?P<dur_second>[0-9]+S)))?)|(?P<dur_time>T(?:(?P<dur_hour2>[0-9]+H(?:[0-9]+M(?:[0-9]+S)?)?)|(?P<dur_minute2>[0-9]+M(?:[0-9]+S)?)|(?P<dur_second2>[0-9]+S)))|(?P<dur_week>[0-9]+W))$"
    ).unwrap();

        let mut schema = test_file.schema.clone();
        if !schema["$schema"].is_string() {
            schema["$schema"] = json!("http://json-schema.org/draft-07/schema#");
        }

        stats.file_info.id = file_name.split('/').next_back().unwrap().to_string();

        match Validator::options()
            .with_retriever(DummyResolver {})
            .should_validate_formats(true)
            .with_format("uuid", move |value| uuid_regex.is_match(value))
            .with_format("iri", move |value| iri_regex.is_match(value))
            .with_format("duration", move |value| duration_regex.is_match(value))
            // .with_draft(jsonschema::Draft::Draft202012)
            .build(&schema)
        {
            Ok(v) => {
                for (idx, t) in test_file.tests.iter_mut().enumerate() {
                    t.rust_error = None;
                    let res = v.validate(&t.data);
                    if t.valid {
                        stats.file_info.num_valid_tests += 1;
                        stats.file_info.size_valid_tests +=
                            serde_json::to_string(&t.data).unwrap().len();
                    } else {
                        stats.file_info.num_invalid_tests += 1;
                        stats.file_info.size_invalid_tests +=
                            serde_json::to_string(&t.data).unwrap().len();
                    }
                    match res {
                        Ok(_) if t.valid => {}
                        Err(e) if !t.valid => {
                            t.rust_error = Some(format!("{e}"));
                        }
                        Ok(_) => {
                            eprintln!("{} {idx} Error: Expected invalid, got valid", file_name);
                            t.rust_error = Some("Expected invalid, got valid".to_string());
                            stats.test_invalid_error =
                                Some("Expected invalid, got valid".to_string());
                        }
                        Err(e) => {
                            eprintln!("{} {idx} Error Validating: {}", file_name, e);
                            t.broken = true;
                            stats.test_valid_error = Some(format!("{e}"));
                        }
                    }

                    limit_string(&mut t.python_error);
                    limit_string(&mut t.rust_error);
                }
            }
            Err(e) => {
                eprintln!("{} Error Creating Validator: {}", file_name, e);
                stats.schema_error = Some(format!("{e}"));
            }
        }

        if self.cli.remove_broken_tests {
            test_file.tests.retain(|t| !t.broken);
        }

        {
            let sch_stats =
                SchemaStats::for_file(file_name, &test_file.schema, self.cli.additional_features);
            let (mut raw_features, mut features): (Vec<_>, Vec<_>) = sch_stats
                .features
                .keys()
                .cloned()
                .partition(|f| is_non_semantic_feature(f));
            features.sort();
            raw_features.sort();
            let meta = JsonMetaInfo {
                full_size: sch_stats.full_size,
                stripped_size: sch_stats.stripped_size,
                features,
                raw_features,
            };
            test_file.meta = Some(meta.clone());
            stats.file_info.meta = meta;
        }

        if self.cli.llg_compile {
            let llg = self.run_llg_compile(&stats.file_info.id, &test_file);
            stats.llg_result = Some(llg);
        } else {
            save_json_to_file(file_name, &test_file);
        }

        stats
    }
}

fn main() {
    let mut options = CliOptions::parse();

    if !options.llg_validate_tokens
        && !options.llg_masks
        && !options.llg_compile
        && !options.rewrite_files
    {
        options.llg_masks = true;
    }
    if options.llg_masks {
        options.llg_validate_tokens = true;
    }
    if options.llg_validate_tokens {
        options.llg_compile = true;
    }
    if options.llg_log_level > 0 {
        options.num_threads = Some(1);
    }

    // set max thread numbers
    let num_cores = std::thread::available_parallelism().unwrap().get();
    let num_threads = options
        .num_threads
        .unwrap_or_else(|| std::cmp::min(num_cores, 40));
    if num_threads > 1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
    }

    let options = Arc::new(options);

    let mut files = vec![];
    for arg in &options.files {
        if arg.ends_with(".json") {
            files.push(arg.to_string());
        } else {
            let dir = std::fs::read_dir(arg).expect("Unable to read directory");
            for entry in dir {
                let entry = entry.expect("Unable to read entry");
                let path = entry.path();
                if path.is_file() && path.to_str().unwrap().ends_with(".json") {
                    files.push(path.to_str().unwrap().to_string());
                }
            }
        }
    }

    files.sort();

    if options.skip_synth {
        files.retain(|f| !f.contains("Handwritten") && !f.contains("Synthesized"));
    }

    if let Some(f) = options.filter.as_ref() {
        files.retain(|f2| f2.contains(f));
    }

    let tok_env: TokEnv = toktrie_hf_downloader::tok_env_from_name(&options.tokenizer).unwrap();

    let mut slices = llguidance::earley::SlicedBiasComputer::json_slices();
    if options.llg_disable_slicer {
        slices.clear();
    }

    let caps = InferenceCapabilities {
        ff_tokens: false,
        backtrack: false,
        conditional_ff_tokens: false,
        fork: false,
    };

    let mut factory = ParserFactory::new(&tok_env, caps.clone(), &slices).unwrap();
    factory.set_buffer_log_level(0);
    factory.set_stderr_log_level(options.llg_log_level);

    // factory.limits_mut().step_lexer_fuel = 10_000_000;

    let mut ref_factory = ParserFactory::new(&tok_env, caps.clone(), &[]).unwrap();
    ref_factory.quiet();

    let factory = Arc::new(factory);
    let ref_factory = Arc::new(ref_factory);

    save_text_to_file("tmp/slices.txt", &factory.slicer().stats(false));
    save_text_to_file("tmp/slices_tokens.txt", &factory.slicer().stats(true));

    let t0 = std::time::Instant::now();
    let par = num_threads > 1;
    let hash_rnd = Arc::new(ahash::RandomState::new());
    let do_file = |file: &String| {
        let env = TestEnv {
            tok_env: tok_env.clone(),
            factory: factory.clone(),
            ref_factory: ref_factory.clone(),
            file_name: file.to_string(),
            cli: options.clone(),
            hash_rnd: hash_rnd.clone(),
        };
        env.run_test()
    };
    let results = if par {
        files.par_iter().map(do_file).collect::<Vec<_>>()
    } else {
        files.iter().map(do_file).collect::<Vec<_>>()
    };

    let mut total = TotalStats::default();
    let mut all_stats: HashMap<String, SchemaRes> = HashMap::default();
    let mut num_files_by_feature: HashMap<String, usize> = HashMap::default();
    let mut num_files_by_raw_feature: HashMap<String, usize> = HashMap::default();
    let mut all_file_info = vec![];
    let mut llg_results = vec![];
    let mut llg_sem_results: IndexMap<String, LlgSemanticResult> = IndexMap::default();
    let mut llg_totals = json!({});
    let mut all_masks_us = vec![];
    let mut all_ttfm_us = vec![];
    let mut validation_errors = vec![];

    total.mask_cache = mask_cache_stats(&results);

    for (file, s) in files.iter().zip(results.into_iter()) {
        all_stats.insert(file.clone(), s.clone());

        all_file_info.push(s.file_info.clone());

        for f in s.file_info.meta.raw_features {
            *num_files_by_raw_feature.entry(f).or_insert(0) += 1;
        }
        for f in s.file_info.meta.features {
            *num_files_by_feature.entry(f).or_insert(0) += 1;
        }

        total.num_valid_tests += s.file_info.num_valid_tests;
        total.num_invalid_tests += s.file_info.num_invalid_tests;

        total.num_files += 1;
        total.full_size += s.full_size;

        if s.file_info.num_invalid_tests + s.file_info.num_invalid_tests == 0 {
            total.num_testless_files += 1;
        }

        if let Some(llg) = s.llg_result {
            let log_err = !options.llg_masks;

            if llg.json_error.is_some() {
                total.llg.num_json_error += 1;
            } else if llg.parser_error.is_some() {
                total.llg.num_parser_error += 1;
            } else if let Some(msg) = llg.validation_error.as_ref() {
                if msg.contains("consider making your grammar left-recursive")
                    || msg.contains("try avoiding single-byte/short lexemes")
                {
                    total.llg.num_parser_limits += 1;
                } else if msg.contains("incorrect accept") {
                    total.llg.num_invalidation_error += 1;
                    validation_errors.push(format!("{}: POS {}", s.file_name, msg));
                    if log_err {
                        eprintln!("{} Error Invalidation: {}", s.file_name, msg);
                    }
                } else {
                    total.llg.num_validation_error += 1;
                    validation_errors.push(format!("{}: NEG {}", s.file_name, msg));
                    if log_err {
                        eprintln!("{} Error Validation: {}", s.file_name, msg);
                    }
                }
            } else {
                total.llg.num_correct_schemas += 1;
            }

            total.llg.num_tokens += llg.num_tokens;
            total.llg.num_masks += llg.all_mask_us.len();
            total.llg.ff_tokens_us += llg.ff_tokens_us.iter().sum::<usize>();
            total.llg.num_ff_token_seqs += llg.ff_tokens_us.len();

            json_sum(&mut llg_totals, &serde_json::to_value(&llg).unwrap());

            if llg.ttfm_us > 0 {
                total.llg.num_parsers += 1;

                all_ttfm_us.push(llg.ttfm_us);
                all_masks_us.extend_from_slice(&llg.all_mask_us);
            }

            total.llg.ttfm_us += llg.ttfm_us;
            total.llg.json_compile_us += llg.json_compile_us;
            total.llg.parser_create_us += llg.parser_create_us;
            total.llg.first_mask_us += llg.first_mask_us;
            total.llg.max_mask_us = std::cmp::max(total.llg.max_mask_us, llg.max_mask_us);

            total.llg.mask_ms_total += llg.all_mask_us.iter().sum::<usize>();
            total.llg.ff_tokens_ms_total += llg.ff_tokens_us.iter().sum::<usize>();

            total.llg.mask_ms_total_a += llg.all_mask_us_a.iter().sum::<usize>();
            total.llg.num_masks_a += llg.all_mask_us_a.len();

            llg_sem_results.insert(llg.id.clone(), LlgSemanticResult::from_llg_result(&llg));
            llg_results.push(llg);
        }

        if s.schema_error.is_some() {
            total.num_schema_error += 1;
        } else if s.test_valid_error.is_some() {
            total.num_valid_error += 1;
        } else if s.test_invalid_error.is_some() {
            total.num_invalid_error += 1;
        }
    }

    if total.llg.num_ff_token_seqs > 0 {
        total.llg.ff_tokens_us /= total.llg.num_ff_token_seqs;
    }

    total.llg.ttfm_ms_total = total.llg.ttfm_us / 1000;

    if total.llg.num_parsers > 0 {
        total.llg.ttfm_us /= total.llg.num_parsers;
        total.llg.parser_create_us /= total.llg.num_parsers;
        total.llg.first_mask_us /= total.llg.num_parsers;
        total.llg.json_compile_us /= total.llg.num_parsers;
        total.llg.num_threads = num_threads;
    }

    if total.llg.num_masks > 0 {
        total.llg.mask_us = total.llg.mask_ms_total / total.llg.num_masks;
        total.llg.num_masks_a_frac = total.llg.num_masks_a * 1000 / total.llg.num_masks;
        total.llg.mask_ms_total_a_frac = total.llg.mask_ms_total_a * 1000 / total.llg.mask_ms_total;
    }

    if total.llg.num_tokens > 0 {
        total.llg.num_ff_tokens = total.llg.num_tokens - total.llg.num_masks;
        total.llg.ff_fraction =
            (total.llg.num_ff_tokens * 10000 / total.llg.num_tokens) as f32 / 10000.0;
    }

    total.llg.mask_ms_total /= 1000;
    total.llg.ff_tokens_ms_total /= 1000;
    total.llg.mask_ms_total_a /= 1000;

    total.llg_json = llg_totals.clone();
    if !options.quiet {
        eprintln!(
            "{}\n{}",
            serde_json::to_string_pretty(&total).unwrap(),
            &factory.perf_counters(),
        );
    }
    eprintln!(
        "Total time: {}ms TTFM {}μs, mask {}μs, ff {}μs, mask+ff {}ms + compile {}ms",
        t0.elapsed().as_millis(),
        total.llg.ttfm_us,
        total.llg.mask_us,
        total.llg.ff_tokens_us,
        num_with_commas(total.llg.mask_ms_total + total.llg.ff_tokens_ms_total),
        num_with_commas(total.llg.ttfm_ms_total),
    );

    save_text_to_file("tmp/validation_errors.txt", &validation_errors.join("\n"));
    save_text_to_file(
        "tmp/llg_masks_us.csv",
        &log_fraction_plot(&mut all_masks_us),
    );
    save_text_to_file("tmp/llg_ttfm_us.csv", &log_fraction_plot(&mut all_ttfm_us));
    save_json_to_file("tmp/test_total.json", &total);
    save_json_to_file("tmp/test_all_stats.json", &all_stats);

    if let Ok(jsb_data) = std::env::var("JSB_DATA") {
        save_json_to_file(
            format!("{}/metainfo/all_stats.json", jsb_data).as_str(),
            &all_stats,
        );
    }

    if !llg_results.is_empty() {
        save_json_to_file("tmp/llg_results.json", &llg_results);
        save_json_to_file("tmp/llg_sem_results.json", &llg_sem_results);
    }

    save_sorted_json_to_file("tmp/num_files_with_feature.json", &num_files_by_feature);
    save_sorted_json_to_file(
        "tmp/num_files_with_raw_feature.json",
        &num_files_by_raw_feature,
    );

    if let Some(expected_file_name) = options.expected.as_ref() {
        let id_to_filename = |id: &str| -> String {
            all_stats
                .values()
                .find(|e| e.file_info.id == id)
                .map(|f| f.file_name.clone())
                .unwrap_or_else(|| id.to_string())
        };

        eprintln!("Expected from {}...", expected_file_name);
        let mut expected_map: HashMap<String, LlgSemanticResult> =
            serde_json::from_str(&read_file_to_string(expected_file_name)).unwrap();
        let mut num_err = 0;
        let mut num_warn = 0;
        for (id, r) in llg_sem_results.iter() {
            if let Some(exp) = expected_map.remove(id) {
                if r != &exp {
                    #[allow(clippy::comparison_chain)]
                    let status = if r.error_badness() < exp.error_badness() {
                        num_err += 1;
                        "improvement"
                    } else if r.error_badness() == exp.error_badness() {
                        if options.ballpark {
                            continue;
                        }
                        num_warn += 1;
                        "similar"
                    } else {
                        num_err += 1;
                        "regression"
                    };

                    eprintln!(
                        "{}: {}: {} -> {}",
                        id_to_filename(id),
                        status,
                        exp.info(),
                        r.info()
                    );
                }
            } else {
                num_warn += 1;
                eprintln!("{}: new ({})", id_to_filename(id), r.info());
            }
        }
        if !options.ballpark {
            for (id, exp) in expected_map {
                num_err += 1;
                eprintln!("{}: missing ({})", id_to_filename(&id), exp.info());
            }
        }

        if num_err > 0 {
            eprintln!("FAILED: {} errors, {} warnings", num_err, num_warn);
            eprintln!("MISMATCH: tmp/llg_sem_results.json {}", expected_file_name);
            std::process::exit(1);
        } else if num_warn > 0 {
            eprintln!("SOFT FAIL: {} warnings", num_warn);
            eprintln!("TRY: cp tmp/llg_sem_results.json {}", expected_file_name);
            std::process::exit(1);
        } else {
            eprintln!("PASSED");
        }
    }
}

fn mask_cache_stats(results: &[SchemaRes]) -> Value {
    let batch_size = 100;
    let hash_size = 1000;

    let mut rng = XorShift::new(1234);
    let mut results = results
        .iter()
        .filter_map(|r| r.llg_result.as_ref())
        .map(|r| {
            let mut v = r.all_hash.clone();
            v.reverse();
            v
        })
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>();
    for idx in 0..results.len() {
        let idx2 = rng.from_range(0..results.len());
        results.swap(idx, idx2);
    }

    let mut on_gpu = HashMap::default();
    let mut left = results
        .drain(0..std::cmp::min(results.len(), batch_size))
        .collect::<Vec<_>>();
    let mut num_hits: usize = 0;
    let mut num_misses: usize = 0;
    let mut round = 0;

    while !left.is_empty() {
        let curr_batch = left
            .iter_mut()
            .map(|e| e.pop().unwrap())
            .collect::<Vec<_>>();
        for h in curr_batch {
            if let std::collections::hash_map::Entry::Vacant(e) = on_gpu.entry(h) {
                num_misses += 1;
                e.insert(round);
            } else {
                num_hits += 1;
            }
        }
        if on_gpu.len() > hash_size {
            let mut to_delete = on_gpu.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
            to_delete.sort_by(|a, b| a.1.cmp(&b.1));
            for (k, _) in to_delete.drain(0..on_gpu.len() - hash_size) {
                on_gpu.remove(&k);
            }
            assert!(on_gpu.len() <= hash_size);
        }
        left.retain(|e| !e.is_empty());
        while left.len() < batch_size && !results.is_empty() {
            left.push(results.pop().unwrap());
        }
        round += 1;
    }

    json!({
        "num_hits": num_hits,
        "num_misses": num_misses,
        "hit_rate_1000": 1000 * num_hits / (num_hits + num_misses + 1),
        "num_rounds": round,
        "batch_size": batch_size,
        "hash_size": hash_size,
    })
}

fn save_json_to_file<T: Serialize>(filename: &str, data: &T) {
    let mut file =
        File::create(filename).unwrap_or_else(|_| panic!("Unable to create file {}", filename));
    file.write_all(serde_json::to_string_pretty(data).unwrap().as_bytes())
        .unwrap_or_else(|_| panic!("Unable to write file {}", filename));
    // eprintln!("Saved to {}", filename);
}

fn save_text_to_file(filename: &str, data: &str) {
    let mut file =
        File::create(filename).unwrap_or_else(|_| panic!("Unable to create file {}", filename));
    file.write_all(data.as_bytes())
        .unwrap_or_else(|_| panic!("Unable to write file {}", filename));
    // eprintln!("Saved to {}", filename);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct LlgTotalStats {
    num_json_error: usize,
    num_parser_error: usize,
    num_validation_error: usize,
    num_invalidation_error: usize,
    num_parser_limits: usize,
    num_correct_schemas: usize,
    num_tokens: usize,
    num_masks: usize,
    num_ff_tokens: usize,
    num_ff_token_seqs: usize,
    ff_fraction: f32,
    num_parsers: usize,
    num_threads: usize,
    ttfm_us: usize,
    ttfm_ms_total: usize,
    json_compile_us: usize,
    parser_create_us: usize,
    first_mask_us: usize,
    max_mask_us: usize,
    ff_tokens_us: usize,
    mask_us: usize,
    ff_tokens_ms_total: usize,
    mask_ms_total: usize,
    mask_ms_total_a: usize,
    num_masks_a: usize,
    mask_ms_total_a_frac: usize,
    num_masks_a_frac: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TotalStats {
    num_files: usize,
    num_testless_files: usize,
    num_valid_tests: usize,
    num_invalid_tests: usize,
    num_schema_error: usize,
    num_fixed_schema_error: usize,
    num_valid_error: usize,
    num_invalid_error: usize,
    full_size: usize,
    stripped_size: usize,
    llg: LlgTotalStats,
    llg_json: Value,
    mask_cache: Value,
}

fn read_file_to_string(filename: &str) -> String {
    let mut file = File::open(filename)
        .map_err(|e| format!("Unable to open file {}: {}", filename, e))
        .unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read file");
    content
}

fn save_sorted_json_to_file(filename: &str, data: &HashMap<String, usize>) {
    let mut data: Vec<_> = data.iter().collect();
    data.sort_by(|a, b| b.1.cmp(a.1));
    let data = Value::Object(
        data.iter()
            .map(|(k, v)| ((*k).clone(), Value::Number((**v as u64).into())))
            .collect::<serde_json::Map<_, _>>(),
    );
    save_json_to_file(filename, &data);
}

fn is_non_semantic_feature(feature: &str) -> bool {
    feature.starts_with("_meta:")
        || feature.starts_with("type:")
        || feature.starts_with("_nested:")
        || feature.ends_with(":trivial")
        || feature == "$id"
        || feature == "$schema"
        || feature == "definitions"
        || feature == "$defs"
        || feature == "defs"
        || feature == "id"
        // these are very widely supported and almost always used; they are not interesting
        || feature == "_boolSchema"
        || feature == "type"
        || feature == "properties"
        || feature == "required"
        || feature == "_requiredEmpty"
        // these are covered by @minmax... features
        || feature == "minimum"
        || feature == "maximum"
        || feature == "exclusiveMinimum"
        || feature == "exclusiveMaximum"
        || feature == "minLength"
        || feature == "maxLength"
        || feature == "minItems"
        || feature == "maxItems"
        || feature == "minProperties"
        || feature == "maxProperties"
}

fn short_limit_string(sp: &str) -> String {
    if sp.len() > 300 {
        format!("{}...", &String::from_utf8_lossy(&sp.as_bytes()[..300]))
    } else {
        sp.to_string()
    }
}

fn limit_string(sp: &mut Option<String>) {
    if let Some(s) = sp {
        if s.len() > 1100 {
            *sp = Some(format!(
                "{}.. ({} more)",
                &String::from_utf8_lossy(&s.as_bytes()[..1024]),
                s.len() - 1024
            ));
        }
    }
}
