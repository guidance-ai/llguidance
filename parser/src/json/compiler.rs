use crate::api::{LLGuidanceOptions, SkipSpec};
use crate::grammar_builder::GrammarResult;
use crate::json::schema::{NumberSchema, StringSchema};
use crate::{regex_to_lark, HashMap};
use anyhow::{anyhow, bail, Context, Result};
use derivre::{ExprRef, JsonQuoteOptions, RegexAst};
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::numeric::number_regex;
use super::schema::{build_schema, ArraySchema, ObjectSchema, OptSchemaExt, Schema};
use super::shared_context::PatternPropertyCache;
use super::{RetrieveWrapper, UnsatisfiableSchemaError};

use crate::{GrammarBuilder, NodeRef};

// TODO: grammar size limit
// TODO: array maxItems etc limits
// TODO: schemastore/src/schemas/json/BizTalkServerApplicationSchema.json - this breaks 1M fuel on lexer, why?!

/// A generation policy for root objects: emit only keys declared in properties
/// or required, with decoded UTF-8 byte lengths within these bounds. Nested
/// JSON objects retain their original schema language, including dynamic keys.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RootObjectKeyPolicy {
    pub min: usize,
    pub max: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct JsonCompileOptions {
    /// Restrict root objects to declared names within decoded UTF-8 byte bounds.
    /// This is an opt-in generation sublanguage, not a JSON Schema assertion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_object_key_policy: Option<RootObjectKeyPolicy>,
    pub item_separator: String,
    pub key_separator: String,
    pub whitespace_flexible: bool,
    pub whitespace_pattern: Option<String>,
    pub coerce_one_of: bool,
    pub lenient: bool,
    /// Allowed escape letters after '\' when quoting JSON strings.
    /// Defaults to full JSON set: nrbtf"u\
    /// For example, set to nrbtf"\ to disallow \uXXXX escapes.
    pub json_allowed_escapes: Option<String>,
    /// Allow printable Unicode escapes in strings and object keys without regex constraints.
    /// Paired surrogate escapes count as one character for string length limits.
    pub json_allow_general_unicode_escapes: bool,
    #[serde(skip)]
    pub retriever: Option<RetrieveWrapper>,
}

fn json_dumps(target: &serde_json::Value) -> String {
    serde_json::to_string(target).unwrap()
}

/// The original schema compiled, but its language has no instances under the
/// explicitly selected root generation policy. This is not semantic schema
/// unsatisfiability and must not be pruned by callers classifying that error.
#[derive(Debug)]
pub struct RootObjectKeyPolicyError {
    message: String,
}
impl std::fmt::Display for RootObjectKeyPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Root object key generation policy: {}", self.message)
    }
}
impl std::error::Error for RootObjectKeyPolicyError {}

// Keep both JSON string paths aligned with Derivre's regular escape policy.
const DEFAULT_JSON_ALLOWED_ESCAPES: &str = "nrbtf\\\"u";

// Match one Unicode scalar: a non-surrogate BMP escape or a high surrogate
// immediately followed by a low surrogate. Keeping each pair in one alternative
// makes string length limits count supplementary-plane characters correctly.
const UNICODE_SCALAR_ESCAPE_REGEX: &str = concat!(
    r"\\u(?:[0-9a-ce-fA-CE-F][0-9a-fA-F]{3}",
    r"|[dD][0-7][0-9a-fA-F]{2}",
    r"|[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})"
);

struct Compiler {
    builder: GrammarBuilder,
    options: JsonCompileOptions,
    definitions: HashMap<String, NodeRef>,
    pending_definitions: Vec<(String, NodeRef)>,
    pattern_cache: PatternPropertyCache,

    any_cache: Option<NodeRef>,
    string_cache: Option<NodeRef>,
    /// Reuse the decoded Unicode-scalar expression across all string length bounds.
    general_unicode_scalar_cache: Option<ExprRef>,
    /// Reuse compiled Unicode-string expressions for identical decoded length bounds.
    general_unicode_string_cache: HashMap<(usize, Option<usize>), ExprRef>,
    item_separator_cache: Option<NodeRef>,
    key_separator_cache: Option<NodeRef>,
}

impl Default for JsonCompileOptions {
    fn default() -> Self {
        Self {
            root_object_key_policy: None,
            item_separator: ",".to_string(),
            key_separator: ":".to_string(),
            whitespace_pattern: None,
            whitespace_flexible: true,
            coerce_one_of: false,
            lenient: false,
            json_allowed_escapes: None,
            json_allow_general_unicode_escapes: false,
            retriever: None,
        }
    }
}

impl JsonCompileOptions {
    pub fn json_to_llg_with_overrides(
        &self,
        builder: GrammarBuilder,
        mut schema: Value,
    ) -> Result<GrammarResult> {
        if let Some(x_guidance) = schema.get("x-guidance") {
            let opts: Self = serde_json::from_value(x_guidance.clone())?;
            // TODO: figure out why not removing this still causes problems in maskbench
            schema.as_object_mut().unwrap().remove("x-guidance");
            opts.json_to_llg(builder, schema)
        } else {
            self.json_to_llg(builder, schema)
        }
    }

    pub fn json_to_llg(&self, builder: GrammarBuilder, schema: Value) -> Result<GrammarResult> {
        let compiler = Compiler::new(self.clone(), builder);
        #[cfg(feature = "jsonschema_validation")]
        {
            use crate::json_validation::validate_schema;
            validate_schema(&schema)?;
        }

        compiler.execute(schema)
    }

    pub fn json_to_llg_no_validate(
        &self,
        builder: GrammarBuilder,
        schema: Value,
    ) -> Result<GrammarResult> {
        let compiler = Compiler::new(self.clone(), builder);
        compiler.execute(schema)
    }

    pub fn apply_to(&self, schema: &mut Value) {
        schema.as_object_mut().unwrap().insert(
            "x-guidance".to_string(),
            serde_json::to_value(self).unwrap(),
        );
    }
}

impl Compiler {
    pub fn new(options: JsonCompileOptions, builder: GrammarBuilder) -> Self {
        Self {
            builder,
            options,
            definitions: HashMap::default(),
            pending_definitions: vec![],
            any_cache: None,
            string_cache: None,
            general_unicode_scalar_cache: None,
            general_unicode_string_cache: HashMap::default(),
            item_separator_cache: None,
            key_separator_cache: None,
            pattern_cache: PatternPropertyCache::default(),
        }
    }

    pub fn execute(mut self, schema: Value) -> Result<GrammarResult> {
        let skip = if let Some(pattern) = &self.options.whitespace_pattern {
            RegexAst::Regex(pattern.clone())
        } else if self.options.whitespace_flexible {
            RegexAst::Regex(r"[\x20\x0A\x0D\x09]+".to_string())
        } else {
            RegexAst::NoMatch
        };
        let id = self
            .builder
            .add_grammar_with_skip(LLGuidanceOptions::default(), SkipSpec::once(skip))?;

        let built = build_schema(schema, &self.options)?;
        self.pattern_cache = built.pattern_cache;

        for w in built.warnings {
            self.builder.add_warning(w);
        }

        if let Some(bounds) = self.options.root_object_key_policy {
            if bounds.min > bounds.max {
                bail!("root object key byte bounds require min <= max");
            }
        }

        if self.options.root_object_key_policy.is_some() {
            // Classify the original normalized schema with the original compiler
            // before applying a spelling policy. Reuse this builder's actual
            // options, URI resolution, caches and remaining resource limits.
            // No schema is reparsed/retrieved and no byte-token parser is built.
            let _ = self.gen_json(&built.schema)?;
            self.complete_definitions(&built.definitions)?;
            self.builder.check_limits()?;
        }

        let root_result = if let Some(bounds) = self.options.root_object_key_policy {
            let root = self.gen_json_root(
                &built.schema,
                &built.definitions,
                bounds,
                &mut HashMap::default(),
                0,
            );
            // A policy error must not hide unsupported or invalid values in
            // referenced definitions that the original compiler still owns.
            match root {
                Err(error) if !error.is::<RootObjectKeyPolicyError>() => return Err(error),
                result => result,
            }
        } else {
            // Preserve the default compiler's original immediate error order.
            Ok(self.gen_json(&built.schema)?)
        };

        if let Ok(root) = &root_result {
            self.builder.set_start_node(*root);
        }
        self.complete_definitions(&built.definitions)?;

        if self.options.root_object_key_policy.is_some() {
            self.builder.check_limits()?;
        }
        let _ = root_result?;
        Ok(self.builder.finalize(id))
    }

    fn complete_definitions(&mut self, definitions: &HashMap<String, Schema>) -> Result<()> {
        while let Some((path, pl)) = self.pending_definitions.pop() {
            let schema = definitions
                .get(&path)
                .ok_or_else(|| anyhow!("Definition not found: {}", path))?;
            let compiled = self.gen_json(schema).map_err(|e| {
                let top_level = anyhow!("{e}\n  while processing {path}");
                e.context(top_level)
            })?;
            self.builder.set_placeholder(pl, compiled);
        }
        Ok(())
    }

    // Root-only compilation follows normalized schema references without moving
    // the source root or changing URI/anchor semantics. Root placeholders are
    // deliberately separate from nested definitions compiled by get_definition.
    fn gen_json_root(
        &mut self,
        schema: &Schema,
        definitions: &HashMap<String, Schema>,
        bounds: RootObjectKeyPolicy,
        roots: &mut HashMap<String, NodeRef>,
        depth: usize,
    ) -> Result<NodeRef> {
        self.builder.check_limits()?;
        if depth > super::schema::SchemaBuilderOptions::default().max_stack_level {
            bail!("root key encoding reference/alternative stack limit exceeded");
        }
        match schema {
            Schema::Ref(uri) => {
                if let Some(node) = roots.get(uri) {
                    return Ok(*node);
                }
                let referenced = definitions
                    .get(uri)
                    .ok_or_else(|| anyhow!("Definition not found: {}", uri))?;
                let placeholder = self.builder.new_node(uri);
                roots.insert(uri.clone(), placeholder);
                match self.gen_json_root(referenced, definitions, bounds, roots, depth + 1) {
                    Ok(node) => {
                        self.builder.set_placeholder(placeholder, node);
                        Ok(placeholder)
                    }
                    Err(error) => {
                        roots.remove(uri);
                        Err(error)
                    }
                }
            }
            Schema::Object(object) => self.gen_json_object(object, Some(bounds)),
            Schema::Any => {
                let num = number_regex(&NumberSchema::default())?;
                let tf = self.builder.regex.regex("true|false")?;
                let nodes = vec![
                    self.builder.string("null"),
                    self.builder.lexeme(tf),
                    self.ast_lexeme(num)?,
                    self.json_simple_string()?,
                    self.gen_json_array(&ArraySchema {
                        min_items: 0,
                        max_items: None,
                        prefix_items: vec![],
                        items: Schema::any_box(),
                    })?,
                    self.gen_json_object(
                        &ObjectSchema {
                            min_properties: 0,
                            max_properties: None,
                            properties: IndexMap::new(),
                            pattern_properties: IndexMap::new(),
                            additional_properties: Schema::any_box(),
                            required: IndexSet::new(),
                        },
                        Some(bounds),
                    )?,
                ];
                Ok(self.builder.select(&nodes))
            }
            Schema::AnyOf(options) | Schema::OneOf(options) => {
                if matches!(schema, Schema::OneOf(_)) {
                    if !(self.options.coerce_one_of || self.options.lenient) {
                        bail!("oneOf constraints are not supported. Enable 'coerce_one_of' option to approximate oneOf with anyOf");
                    }
                    self.builder.add_warning("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.".to_string());
                }
                let mut nodes = Vec::new();
                let mut errors = Vec::new();
                let mut policy_errors = Vec::new();
                for option in options {
                    match self.gen_json_root(option, definitions, bounds, roots, depth + 1) {
                        Ok(node) => nodes.push(node),
                        Err(error) if error.is::<RootObjectKeyPolicyError>() => {
                            policy_errors.push(error)
                        }
                        Err(error) if error.is::<UnsatisfiableSchemaError>() => errors.push(error),
                        Err(error) => return Err(error),
                    }
                }
                self.builder.check_limits()?;
                if !nodes.is_empty() {
                    return Ok(self.builder.select(&nodes));
                }
                if let Some(error) = policy_errors.pop() {
                    return Err(error);
                }
                let error = anyhow!(UnsatisfiableSchemaError {
                    message: "All original root schema alternatives are unsatisfiable".to_string()
                });
                Err(if let Some(source) = errors.pop() {
                    error.context(source)
                } else {
                    error
                })
            }
            _ => self.gen_json(schema),
        }
    }

    fn gen_json(&mut self, json_schema: &Schema) -> Result<NodeRef> {
        if let Some(ast) = self.regex_compile(json_schema)? {
            return self.ast_lexeme(ast);
        }
        match json_schema {
            Schema::Any => self.gen_json_any(),
            Schema::Unsatisfiable(reason) => Err(anyhow!(UnsatisfiableSchemaError {
                message: reason.to_string(),
            })),

            Schema::Array(arr) => self.gen_json_array(arr),
            Schema::Object(obj) => self.gen_json_object(obj, None),
            Schema::AnyOf(options) => self.process_any_of(options),
            Schema::OneOf(options) => self.process_one_of(options),
            Schema::Ref(uri) => self.get_definition(uri),

            Schema::Null | Schema::Boolean(_) | Schema::String(_) | Schema::Number(_) => {
                unreachable!("should be handled in regex_compile()")
            }
        }
    }

    fn process_one_of(&mut self, options: &[Schema]) -> Result<NodeRef> {
        if self.options.coerce_one_of || self.options.lenient {
            self.builder
                .add_warning("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.".to_string());
            self.process_any_of(options)
        } else {
            Err(anyhow!("oneOf constraints are not supported. Enable 'coerce_one_of' option to approximate oneOf with anyOf"))
        }
    }

    fn process_option(
        &mut self,
        option: &Schema,
        regex_nodes: &mut Vec<RegexAst>,
        cfg_nodes: &mut Vec<NodeRef>,
    ) -> Result<()> {
        match self.regex_compile(option)? {
            Some(c) => regex_nodes.push(c),
            None => cfg_nodes.push(self.gen_json(option)?),
        }
        Ok(())
    }

    fn process_any_of(&mut self, options: &[Schema]) -> Result<NodeRef> {
        let mut regex_nodes = vec![];
        let mut cfg_nodes = vec![];
        let mut errors = vec![];

        for option in options.iter() {
            if let Err(err) = self.process_option(option, &mut regex_nodes, &mut cfg_nodes) {
                match err.downcast_ref::<UnsatisfiableSchemaError>() {
                    Some(_) => errors.push(err),
                    None => return Err(err),
                }
            }
        }

        self.builder.check_limits()?;

        if !regex_nodes.is_empty() {
            let node = RegexAst::Or(regex_nodes);
            let lex = self.ast_lexeme(node)?;
            cfg_nodes.push(lex);
        }

        if !cfg_nodes.is_empty() {
            Ok(self.builder.select(&cfg_nodes))
        } else if let Some(e) = errors.pop() {
            Err(anyhow!(UnsatisfiableSchemaError {
                message: "All options in anyOf are unsatisfiable".to_string(),
            })
            .context(e))
        } else {
            Err(anyhow!(UnsatisfiableSchemaError {
                message: "No options in anyOf".to_string(),
            }))
        }
    }

    fn ast_lexeme(&mut self, ast: RegexAst) -> Result<NodeRef> {
        let id = self.builder.regex.add_ast(ast)?;
        Ok(self.builder.lexeme(id))
    }

    fn json_simple_string(&mut self) -> Result<NodeRef> {
        if let Some(node) = self.string_cache {
            return Ok(node);
        }

        let ast = if self.options.json_allow_general_unicode_escapes {
            self.json_general_unicode_string(0, None)?
        } else {
            self.json_quote(RegexAst::Regex("(?s:.*)".to_string()))
        };
        let node = self.ast_lexeme(ast)?;
        self.string_cache = Some(node);
        Ok(node)
    }

    fn item_separator(&mut self) -> Result<NodeRef> {
        if let Some(node) = self.item_separator_cache {
            return Ok(node);
        }
        let rx = self.builder.regex.regex(&self.options.item_separator)?;
        let node = self.builder.lexeme(rx);
        self.item_separator_cache = Some(node);
        Ok(node)
    }

    fn key_separator(&mut self) -> Result<NodeRef> {
        if let Some(node) = self.key_separator_cache {
            return Ok(node);
        }
        let rx = self.builder.regex.regex(&self.options.key_separator)?;
        let node = self.builder.lexeme(rx);
        self.key_separator_cache = Some(node);
        Ok(node)
    }

    fn get_definition(&mut self, reference: &str) -> Result<NodeRef> {
        if let Some(definition) = self.definitions.get(reference) {
            return Ok(*definition);
        }
        let r = self.builder.new_node(reference);
        self.definitions.insert(reference.to_string(), r);
        self.pending_definitions.push((reference.to_string(), r));
        Ok(r)
    }

    fn gen_json_any(&mut self) -> Result<NodeRef> {
        if let Some(json_any) = self.any_cache {
            return Ok(json_any);
        }

        let json_any = self.builder.new_node("json_any");
        self.any_cache = Some(json_any); // avoid infinite recursion
        let num = number_regex(&NumberSchema::default())?;
        let tf = self.builder.regex.regex("true|false")?;
        let options = vec![
            self.builder.string("null"),
            self.builder.lexeme(tf),
            self.ast_lexeme(num)?,
            self.json_simple_string()?,
            self.gen_json_array(&ArraySchema {
                min_items: 0,
                max_items: None,
                prefix_items: vec![],
                items: Schema::any_box(),
            })?,
            self.gen_json_object(
                &ObjectSchema {
                    properties: IndexMap::new(),
                    additional_properties: Schema::any_box(),
                    required: IndexSet::new(),
                    pattern_properties: IndexMap::new(),
                    min_properties: 0,
                    max_properties: None,
                },
                None,
            )?,
        ];
        let inner = self.builder.select(&options);
        self.builder.set_placeholder(json_any, inner);
        Ok(json_any)
    }

    fn gen_json_object(
        &mut self,
        obj: &ObjectSchema,
        key_bounds: Option<RootObjectKeyPolicy>,
    ) -> Result<NodeRef> {
        let mut taken_names: Vec<String> = vec![];
        let mut unquoted_taken_names: Vec<String> = vec![];
        let mut items: Vec<(NodeRef, bool)> = vec![];

        let colon = self.key_separator()?;

        let mut num_required = 0;
        let mut num_optional = 0;
        let mut policy_error = None;

        for name in obj.properties.keys().chain(
            obj.required
                .iter()
                .filter(|n| !obj.properties.contains_key(n.as_str())),
        ) {
            let property_schema = self.pattern_cache.property_schema(obj, name)?;
            let is_required = obj.required.contains(name);
            if !obj.pattern_properties.is_empty() {
                unquoted_taken_names.push(name.to_string());
            }
            // Quote (and escape) the name
            let quoted_name = json_dumps(&json!(name));
            let property = match self.gen_json(property_schema) {
                Ok(node) => node,
                Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                    // If it's not an UnsatisfiableSchemaError, just propagate it normally
                    None => return Err(e),
                    // Property is optional; don't raise UnsatisfiableSchemaError but mark name as taken
                    Some(_) if !is_required => {
                        taken_names.push(quoted_name);
                        continue;
                    }
                    // Property is required; add context and propagate UnsatisfiableSchemaError
                    Some(_) => {
                        return Err(e.context(UnsatisfiableSchemaError {
                            message: format!("required property '{name}' is unsatisfiable"),
                        }));
                    }
                },
            };
            if let Some(bounds) = key_bounds {
                if name.len() < bounds.min || name.len() > bounds.max {
                    if is_required {
                        policy_error = Some(anyhow!(RootObjectKeyPolicyError {
                            message: format!(
                                "required root key has {} UTF-8 bytes outside {}..={}",
                                name.len(),
                                bounds.min,
                                bounds.max
                            )
                        }));
                    }
                    taken_names.push(quoted_name);
                    continue;
                }
            }
            let name = self.builder.string(&quoted_name);
            taken_names.push(quoted_name);
            let item = self.builder.join(&[name, colon, property]);
            items.push((item, is_required));
            if is_required {
                num_required += 1;
            } else {
                num_optional += 1;
            }
        }

        if key_bounds.is_some() {
            // Original root/value/reference capability and emptiness were
            // established by the complete unmodified compilation preflight.
            if let Some(error) = policy_error {
                return Err(error);
            }
            return self.named_object_fields(&items, obj.min_properties, obj.max_properties);
        }

        let min_properties = obj.min_properties.saturating_sub(num_required);
        let max_properties = obj.max_properties.map(|v| v.saturating_sub(num_required));

        if num_optional > 0 && (min_properties > 0 || max_properties.is_some()) {
            // special case for min/maxProperties == 1
            // this is sometimes used to indicate that at least one property is required
            if min_properties <= 1
                && max_properties.unwrap_or(1) == 1
                && obj.pattern_properties.is_empty()
                && obj
                    .additional_properties
                    .as_ref()
                    .map(|s| s.is_unsat())
                    .unwrap_or(false)
            {
                let mut options: Vec<Vec<(NodeRef, bool)>> = vec![];
                if max_properties == Some(1) {
                    // at most one
                    for idx in 0..items.len() {
                        let (_, required) = items[idx];
                        if !required {
                            options.push(
                                items
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(i2, (n, r))| {
                                        if i2 == idx || *r {
                                            Some((*n, true))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                            );
                        }
                    }
                    if min_properties == 1 {
                        // exactly one - done
                    } else {
                        // at most one - add empty option
                        assert!(min_properties == 0);
                        options.push(items.into_iter().filter(|(_, r)| *r).collect());
                    }
                } else {
                    assert!(max_properties.is_none());
                    assert!(min_properties == 1);
                    // at least one
                    for idx in 0..items.len() {
                        let (_, required) = items[idx];
                        if !required {
                            options.push(
                                items
                                    .iter()
                                    .enumerate()
                                    .map(|(i2, (n, r))| (*n, *r || i2 == idx))
                                    .collect(),
                            );
                        }
                    }
                }
                let sel_options = options
                    .iter()
                    .map(|v| self.object_fields(v))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(self.builder.select(&sel_options));
            }

            let msg = "min/maxProperties only supported when all keys listed in \"properties\" are required";
            if self.options.lenient {
                self.builder.add_warning(msg.to_string());
            } else {
                bail!(msg);
            }
        }

        let mut taken_name_ids = taken_names
            .iter()
            .map(|n| self.builder.regex.literal(n.to_string()))
            .collect::<Vec<_>>();

        let mut pattern_options = vec![];
        for (pattern, schema) in obj.pattern_properties.iter() {
            let regex = self
                .builder
                .regex
                .add_ast(self.json_quote(RegexAst::SearchRegex(regex_to_lark(pattern, "dw"))))?;
            taken_name_ids.push(regex);

            let schema = match self.gen_json(schema) {
                Ok(r) => r,
                Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                    // If it's not an UnsatisfiableSchemaError, just propagate it normally
                    None => return Err(e),
                    // Property is optional; don't raise UnsatisfiableSchemaError but mark name as taken
                    Some(_) => continue,
                },
            };

            let exclude_names = unquoted_taken_names
                .iter()
                .enumerate()
                .filter(|(_, name)| self.pattern_cache.is_match(pattern, name).unwrap_or(true))
                .map(|(idx, _)| taken_name_ids[idx])
                .collect::<Vec<_>>();
            let regex = if exclude_names.is_empty() {
                regex
            } else {
                let options = self.builder.regex.select(exclude_names);
                let not_taken = self.builder.regex.not(options);
                self.builder.regex.and(vec![regex, not_taken])
            };

            let name = self.builder.lexeme(regex);
            pattern_options.push(self.builder.join(&[name, colon, schema]));
        }

        match self.gen_json(obj.additional_properties.schema_ref()) {
            Err(e) => {
                if e.downcast_ref::<UnsatisfiableSchemaError>().is_none() {
                    // Propagate errors that aren't UnsatisfiableSchemaError
                    return Err(e);
                }
                // Ignore UnsatisfiableSchemaError for additionalProperties
            }
            Ok(property) => {
                let name = if taken_name_ids.is_empty() {
                    self.json_simple_string()?
                } else {
                    let taken = self.builder.regex.select(taken_name_ids);
                    let not_taken = self.builder.regex.not(taken);
                    let valid_ast = self.json_general_unicode_string(0, None)?;
                    let valid = self.builder.regex.add_ast(valid_ast)?;
                    let valid_and_not_taken = self.builder.regex.and(vec![valid, not_taken]);
                    self.builder.lexeme(valid_and_not_taken)
                };
                let item = self.builder.join(&[name, colon, property]);
                pattern_options.push(item);
            }
        }

        if !pattern_options.is_empty() && max_properties != Some(0) {
            let pattern = self.builder.select(&pattern_options);
            let required = min_properties > 0;
            let seq = self.bounded_sequence(pattern, min_properties, max_properties)?;
            items.push((seq, required));
        } else if min_properties > 0 {
            return Err(anyhow!(UnsatisfiableSchemaError {
                message: format!(
                    "minProperties ({min_properties}) is greater than number of properties ({num_required})"
                ),
            }));
        }

        self.object_fields(&items)
    }

    // Compile a finite set of declared root fields in schema order. Values were
    // compiled against the original property/pattern/additional intersection.
    // This opt-in path neither generates nor tries to solve dynamic key names.
    fn named_object_fields(
        &mut self,
        items: &[(NodeRef, bool)],
        min: usize,
        max: Option<usize>,
    ) -> Result<NodeRef> {
        let maximum = max.unwrap_or(items.len()).min(items.len());
        let required = items.iter().filter(|(_, required)| *required).count();
        if min > maximum || required > maximum {
            return Err(anyhow!(RootObjectKeyPolicyError {
                message: "named root key policy cannot satisfy min/maxProperties".to_string()
            }));
        }
        let comma = self.item_separator()?;
        let empty = self.builder.string("");
        let inner = if min <= required && maximum == items.len() {
            // Same optional/required sequence language as ordered_sequence,
            // built iteratively so the new policy does not add recursive stack.
            let mut plain = empty;
            let mut prefixed = empty;
            for &(item, mandatory) in items.iter().rev() {
                self.builder.check_limits()?;
                let take_plain = self.builder.join(&[item, prefixed]);
                let take_prefixed = self.builder.join(&[comma, item, prefixed]);
                if mandatory {
                    plain = take_plain;
                    prefixed = take_prefixed;
                } else {
                    plain = self.builder.select(&[take_plain, plain]);
                    prefixed = self.builder.select(&[take_prefixed, prefixed]);
                }
            }
            plain
        } else {
            // Each row contains the grammar for each exact suffix field count.
            // Backend grammar/fuel guards bound construction; no exponential
            // subset enumeration and no recursion are introduced.
            let mut suffix = vec![Some(empty)];
            for &(item, mandatory) in items.iter().rev() {
                let limit = suffix.len().min(maximum);
                let mut next = vec![None; limit + 1];
                for (count, slot) in next.iter_mut().enumerate() {
                    self.builder.check_limits()?;
                    let mut choices = Vec::new();
                    if let Some(Some(skip)) = suffix.get(count).filter(|_| !mandatory) {
                        choices.push(*skip);
                    }
                    if let Some(Some(tail)) =
                        count.checked_sub(1).and_then(|index| suffix.get(index))
                    {
                        choices.push(if count == 1 {
                            item
                        } else {
                            self.builder.join(&[item, comma, *tail])
                        });
                    }
                    if !choices.is_empty() {
                        *slot = Some(self.builder.select(&choices));
                    }
                }
                suffix = next;
            }
            let choices = suffix.into_iter().skip(min).flatten().collect::<Vec<_>>();
            if choices.is_empty() {
                return Err(anyhow!(RootObjectKeyPolicyError {
                    message: "named root key policy has no admitted declared-field combination"
                        .to_string()
                }));
            }
            self.builder.select(&choices)
        };
        self.builder.check_limits()?;
        let open = self.builder.string("{");
        let close = self.builder.string("}");
        Ok(self.builder.join(&[open, inner, close]))
    }

    fn object_fields(&mut self, items: &[(NodeRef, bool)]) -> Result<NodeRef> {
        let opener = self.builder.string("{");
        let inner = self.ordered_sequence(items, false, &mut HashMap::default())?;
        let closer = self.builder.string("}");
        Ok(self.builder.join(&[opener, inner, closer]))
    }

    #[allow(clippy::type_complexity)]
    fn ordered_sequence<'a>(
        &mut self,
        items: &'a [(NodeRef, bool)],
        prefixed: bool,
        cache: &mut HashMap<(&'a [(NodeRef, bool)], bool), NodeRef>,
    ) -> Result<NodeRef> {
        // Cache to reduce number of nodes from O(n^2) to O(n)
        if let Some(node) = cache.get(&(items, prefixed)) {
            return Ok(*node);
        }
        if items.is_empty() {
            return Ok(self.builder.string(""));
        }
        let comma = self.item_separator()?;
        let (item, required) = items[0];
        let rest = &items[1..];

        let node = match (prefixed, required) {
            (true, true) => {
                // If we know we have preceding elements, we can safely just add a (',' + e)
                let rest_seq = self.ordered_sequence(rest, true, cache)?;
                self.builder.join(&[comma, item, rest_seq])
            }
            (true, false) => {
                // If we know we have preceding elements, we can safely just add an optional(',' + e)
                // TODO optimization: if the rest is all optional, we can nest the rest in the optional
                let comma_item = self.builder.join(&[comma, item]);
                let optional_comma_item = self.builder.optional(comma_item);
                let rest_seq = self.ordered_sequence(rest, true, cache)?;
                self.builder.join(&[optional_comma_item, rest_seq])
            }
            (false, true) => {
                // No preceding elements, so we just add the element (no comma)
                let rest_seq = self.ordered_sequence(rest, true, cache)?;
                self.builder.join(&[item, rest_seq])
            }
            (false, false) => {
                // No preceding elements, but our element is optional. If we add the element, the remaining
                // will be prefixed, else they are not.
                // TODO: same nested optimization as above
                let prefixed_rest = self.ordered_sequence(rest, true, cache)?;
                let unprefixed_rest = self.ordered_sequence(rest, false, cache)?;
                let opts = [self.builder.join(&[item, prefixed_rest]), unprefixed_rest];
                self.builder.select(&opts)
            }
        };
        cache.insert((items, prefixed), node);
        Ok(node)
    }

    fn bounded_sequence(
        &mut self,
        item: NodeRef,
        min_elts: usize,
        max_elts: Option<usize>,
    ) -> Result<NodeRef> {
        let min_elts = min_elts.saturating_sub(1);
        let max_elts = max_elts.map(|v| v.saturating_sub(1));
        let comma = self.item_separator()?;
        let item_comma = self.builder.join(&[item, comma]);
        let item_comma_rep = self.builder.repeat(item_comma, min_elts, max_elts);
        Ok(self.builder.join(&[item_comma_rep, item]))
    }

    fn sequence(&mut self, item: NodeRef) -> Result<NodeRef> {
        let comma = self.item_separator()?;
        let item_comma = self.builder.join(&[item, comma]);
        let item_comma_star = self.builder.zero_or_more(item_comma);
        Ok(self.builder.join(&[item_comma_star, item]))
    }

    fn json_quote(&self, ast: RegexAst) -> RegexAst {
        let allowed_escapes = self
            .options
            .json_allowed_escapes
            .clone()
            .unwrap_or_else(|| DEFAULT_JSON_ALLOWED_ESCAPES.to_string());
        RegexAst::JsonQuote(
            Box::new(ast),
            JsonQuoteOptions {
                allowed_escapes,
                raw_mode: false,
            },
        )
    }

    fn regex_compile(&mut self, schema: &Schema) -> Result<Option<RegexAst>> {
        fn literal_regex(rx: &str) -> Option<RegexAst> {
            Some(RegexAst::Literal(rx.to_string()))
        }

        self.builder.check_limits()?;

        let r = match schema {
            Schema::Null => literal_regex("null"),
            Schema::Boolean(None) => Some(RegexAst::Regex("true|false".to_string())),
            Schema::Boolean(Some(value)) => literal_regex(if *value { "true" } else { "false" }),

            Schema::Number(num) => Some(number_regex(num)?),

            Schema::String(opts) => return self.gen_json_string(opts.clone()).map(Some),

            Schema::Any
            | Schema::Unsatisfiable(_)
            | Schema::Array(_)
            | Schema::Object(_)
            | Schema::AnyOf(_)
            | Schema::OneOf(_)
            | Schema::Ref(_) => None,
        };
        Ok(r)
    }

    fn gen_json_string(&mut self, opts: StringSchema) -> Result<RegexAst> {
        let min_length = opts.min_length;
        let max_length = opts.max_length;
        if let Some(max_length) = max_length {
            if min_length > max_length {
                return Err(anyhow!(UnsatisfiableSchemaError {
                    message: format!(
                        "minLength ({min_length}) is greater than maxLength ({max_length})"
                    ),
                }));
            }
        }
        if opts.regex.is_none() && self.options.json_allow_general_unicode_escapes {
            return self.json_general_unicode_string(min_length, max_length);
        }
        if min_length == 0 && max_length.is_none() && opts.regex.is_none() {
            return Ok(self.json_quote(RegexAst::Regex("(?s:.*)".to_string())));
        }
        if let Some(mut ast) = opts.regex {
            let mut positive = false;

            fn mk_rx_repr(ast: &RegexAst) -> String {
                let mut rx_repr = String::new();
                ast.write_to_str(&mut rx_repr, 1_000, None);
                rx_repr
            }

            // special-case literals - the length is easy to check
            if let RegexAst::Literal(s) = &ast {
                let l = s.chars().count();

                if l < min_length || l > max_length.unwrap_or(usize::MAX) {
                    return Err(anyhow!(UnsatisfiableSchemaError {
                        message: format!("Constant {s:?} doesn't match length constraints")
                    }));
                }

                positive = true;
            } else if min_length != 0 || max_length.is_some() {
                ast = RegexAst::And(vec![
                    ast,
                    RegexAst::Regex(format!(
                        "(?s:.{{{},{}}})",
                        min_length,
                        max_length.map_or("".to_string(), |v| v.to_string())
                    )),
                ]);
            } else {
                positive = always_non_empty(&ast);
                // eprintln!("positive:{} {}", positive, mk_rx_repr(&ast));
            }

            if !positive {
                // Check if the regex is empty
                let mut builder = derivre::RegexBuilder::new();
                let expr = builder.mk(&ast)?;
                // if regex is not positive, do the more expensive non-emptiness check
                if !builder.exprset().is_positive(expr) {
                    // in JSB, 13 cases above 2000;
                    // 1 case above 5000:
                    // "format": "email",
                    // "pattern": "^\\w+([\\.-]?\\w+)*@\\w+([\\.-]?\\w+)*(\\.\\w{2,})+$",
                    // "minLength": 6,
                    //
                    // (excluding two handwritten examples with minLength:10000)
                    let mut regex = builder.to_regex_limited(expr, 10_000).map_err(|_| {
                        anyhow!(
                            "Unable to determine if regex is empty: {}",
                            mk_rx_repr(&ast)
                        )
                    })?;
                    if regex.always_empty() {
                        return Err(anyhow!(UnsatisfiableSchemaError {
                            message: format!("Regex is empty: {}", mk_rx_repr(&ast))
                        }));
                    }
                }
            }

            // The emptiness check above passed, so if `ast` intersects a literal
            // with other regexes, it matches exactly that literal. Emit the
            // literal itself so the lexer does not have to match the intersection.
            if let Some(lit) = and_literal(&ast) {
                ast = RegexAst::Literal(lit.to_string());
            }

            ast = self.json_quote(ast);

            Ok(ast)
        } else {
            Ok(self.json_quote(RegexAst::Regex(format!(
                "(?s:.{{{},{}}})",
                min_length,
                max_length.map_or("".to_string(), |v| v.to_string())
            ))))
        }
    }

    /// Builds a JSON-string regex whose repetitions count decoded Unicode scalars.
    ///
    /// This is only valid for strings without regex constraints: printable escapes
    /// are not decoded before matching a schema pattern. Complete surrogate pairs
    /// are one repetition, while unpaired surrogates are rejected. The scalar
    /// expression is compiled once and reused across all string length bounds.
    fn json_general_unicode_string(
        &mut self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> Result<RegexAst> {
        let cache_key = (min_length, max_length);
        if let Some(expr) = self.general_unicode_string_cache.get(&cache_key) {
            return Ok(RegexAst::ExprRef(*expr));
        }

        let scalar = if let Some(expr) = self.general_unicode_scalar_cache {
            expr
        } else {
            let allowed_escapes = self
                .options
                .json_allowed_escapes
                .as_deref()
                .unwrap_or(DEFAULT_JSON_ALLOWED_ESCAPES);
            let mut short_escapes = String::new();
            let mut allow_unicode_escapes = false;

            for escape in allowed_escapes.chars() {
                match escape {
                    'u' => allow_unicode_escapes = true,
                    '\\' => short_escapes.push_str(r"\\"),
                    '"' | 'b' | 'f' | 'n' | 'r' | 't' => short_escapes.push(escape),
                    _ => bail!("invalid escape character in allowed_escapes: {escape}"),
                }
            }

            let mut character_regex = r#"[^"\\\x00-\x1F\x7F]"#.to_string();
            if !short_escapes.is_empty() {
                character_regex.push_str(&format!(r"|\\[{short_escapes}]"));
            }
            if allow_unicode_escapes {
                character_regex.push('|');
                character_regex.push_str(UNICODE_SCALAR_ESCAPE_REGEX);
            }

            let expr = self.builder.regex.regex(&character_regex)?;
            self.general_unicode_scalar_cache = Some(expr);
            expr
        };

        let min = u32::try_from(min_length)
            .with_context(|| format!("minLength ({min_length}) exceeds the supported range"))?;
        let max = max_length
            .map(|max| {
                u32::try_from(max)
                    .with_context(|| format!("maxLength ({max}) exceeds the supported range"))
            })
            .transpose()?
            .unwrap_or(u32::MAX);
        let ast = RegexAst::Concat(vec![
            RegexAst::Literal("\"".to_string()),
            RegexAst::Repeat(Box::new(RegexAst::ExprRef(scalar)), min, max),
            RegexAst::Literal("\"".to_string()),
        ]);
        let expr = self.builder.regex.add_ast(ast)?;
        self.general_unicode_string_cache.insert(cache_key, expr);
        Ok(RegexAst::ExprRef(expr))
    }

    fn gen_json_array(&mut self, arr: &ArraySchema) -> Result<NodeRef> {
        let mut max_items = arr.max_items;
        let min_items = arr.min_items;

        if let Some(max_items) = max_items {
            if min_items > max_items {
                return Err(anyhow!(UnsatisfiableSchemaError {
                    message: format!(
                        "minItems ({min_items}) is greater than maxItems ({max_items})"
                    ),
                }));
            }
        }

        let additional_item_grm = match self.gen_json(arr.items.schema_ref()) {
            Ok(node) => Some(node),
            Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                // If it's not an UnsatisfiableSchemaError, just propagate it normally
                None => return Err(e),
                // Item is optional; don't raise UnsatisfiableSchemaError
                Some(_) if arr.prefix_items.len() >= min_items => None,
                // Item is required; add context and propagate UnsatisfiableSchemaError
                Some(_) => {
                    return Err(e.context(UnsatisfiableSchemaError {
                        message: "required item is unsatisfiable".to_string(),
                    }));
                }
            },
        };

        let mut required_items = vec![];
        let mut optional_items = vec![];

        // Consider at most maxItems prefix items.
        let prefix_count = max_items.map_or(arr.prefix_items.len(), |max| {
            max.min(arr.prefix_items.len())
        });

        for i in 0..prefix_count {
            let item = match self.gen_json(&arr.prefix_items[i]) {
                Ok(node) => node,
                Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                    // If it's not an UnsatisfiableSchemaError, just propagate it normally
                    None => return Err(e),
                    // Item is optional; don't raise UnsatisfiableSchemaError.
                    // Set max_items to the current index, as we can't satisfy any more items.
                    Some(_) if i >= min_items => {
                        max_items = Some(i);
                        break;
                    }
                    // Item is required; add context and propagate UnsatisfiableSchemaError
                    Some(_) => {
                        return Err(e.context(UnsatisfiableSchemaError {
                            message: format!(
                                "prefixItems[{i}] is unsatisfiable but minItems is {min_items}"
                            ),
                        }));
                    }
                },
            };

            if i < min_items {
                required_items.push(item);
            } else {
                optional_items.push(item);
            }
        }

        // Compile the homogeneous tail compactly instead of expanding every item.
        if let Some(additional_item) = additional_item_grm {
            // Array bounds include the prefix items that were already compiled.
            let explicit_items = required_items.len() + optional_items.len();
            let remaining_min = min_items.saturating_sub(explicit_items);
            let remaining_max = max_items.map(|max| max.saturating_sub(explicit_items));

            if remaining_max != Some(0) {
                let tail = if remaining_min == 0 && remaining_max.is_none() {
                    self.sequence(additional_item)?
                } else {
                    // The tail contains at least one item; its entirety may be optional below.
                    self.bounded_sequence(additional_item, remaining_min.max(1), remaining_max)?
                };

                if remaining_min == 0 {
                    optional_items.push(tail);
                } else {
                    required_items.push(tail);
                }
            }
        }

        let mut grammars: Vec<NodeRef> = vec![self.builder.string("[")];
        let comma = self.item_separator()?;

        if !required_items.is_empty() {
            grammars.push(required_items[0]);
            for item in &required_items[1..] {
                grammars.push(comma);
                grammars.push(*item);
            }
        }

        if !optional_items.is_empty() {
            let first = optional_items[0];
            let tail =
                optional_items
                    .into_iter()
                    .skip(1)
                    .rev()
                    .fold(self.builder.empty(), |acc, item| {
                        let j = self.builder.join(&[comma, item, acc]);
                        self.builder.optional(j)
                    });
            let tail = self.builder.join(&[first, tail]);

            if !required_items.is_empty() {
                let j = self.builder.join(&[comma, tail]);
                grammars.push(self.builder.optional(j));
            } else {
                grammars.push(self.builder.optional(tail));
            }
        }

        grammars.push(self.builder.string("]"));
        Ok(self.builder.join(&grammars))
    }
}

/// The literal in an intersection (possibly nested), if any.
fn and_literal(ast: &RegexAst) -> Option<&str> {
    match ast {
        RegexAst::Literal(s) => Some(s),
        RegexAst::And(args) => args.iter().find_map(and_literal),
        _ => None,
    }
}

fn always_non_empty(ast: &RegexAst) -> bool {
    match ast {
        RegexAst::Or(asts) => asts.iter().any(always_non_empty),
        RegexAst::Concat(asts) => asts.iter().all(always_non_empty),
        RegexAst::Repeat(ast, _, _) | RegexAst::JsonQuote(ast, _) | RegexAst::LookAhead(ast) => {
            always_non_empty(ast)
        }

        RegexAst::EmptyString
        | RegexAst::Literal(_)
        | RegexAst::ByteLiteral(_)
        | RegexAst::Byte(_)
        | RegexAst::ByteSet(_)
        | RegexAst::MultipleOf(_, _) => true,

        RegexAst::And(_)
        | RegexAst::Not(_)
        | RegexAst::NoMatch
        | RegexAst::Regex(_)
        | RegexAst::SearchRegex(_)
        | RegexAst::ExprRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{GrammarInit, GrammarWithLexer, ParserLimits, TopLevelGrammar};
    use serde_json::json;

    fn compile_schema(schema: Value) -> Result<GrammarResult> {
        JsonCompileOptions::default()
            .json_to_llg(GrammarBuilder::new(None, ParserLimits::default()), schema)
    }

    #[test]
    fn large_bounded_array_stays_compact() {
        let result = compile_schema(json!({
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 30_000
        }))
        .expect("large bounded array should compile");

        assert!(
            result.builder.num_nodes() < 256,
            "bounded array expanded into {} grammar symbols",
            result.builder.num_nodes()
        );
    }

    #[test]
    fn large_bounded_array_with_prefix_items_stays_compact() {
        let result = compile_schema(json!({
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}],
            "items": {"type": "boolean"},
            "minItems": 1,
            "maxItems": 30_000
        }))
        .expect("large bounded array with prefix items should compile");

        assert!(result.builder.num_nodes() < 256);
    }

    #[test]
    fn large_minimum_array_stays_compact() {
        let result = compile_schema(json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 29_999,
            "maxItems": 30_000
        }))
        .expect("large required array should compile");

        assert!(result.builder.num_nodes() < 256);
    }

    #[test]
    fn multiple_large_bounded_array_schemas_stay_compact() {
        let schema = json!({
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 30_000
                }
            },
            "required": ["values"],
            "additionalProperties": false
        });
        let variants: Vec<Value> = ["first", "second"]
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                let mut payload = schema.clone();
                payload
                    .as_object_mut()
                    .expect("payload schema should be an object")
                    .insert("$id".to_string(), json!(format!("urn:example:{index}")));
                json!({
                    "type": "object",
                    "properties": {
                        "name": {"const": name},
                        "payload": payload
                    },
                    "required": ["name", "payload"],
                    "additionalProperties": false
                })
            })
            .collect();
        let combined_schema = json!({
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"anyOf": variants}
                }
            },
            "required": ["entries"],
            "additionalProperties": false
        });

        let grammar = GrammarInit::Serialized(TopLevelGrammar {
            grammars: vec![
                GrammarWithLexer {
                    lark_grammar: Some("start: @first | @second | @combined".to_string()),
                    ..Default::default()
                },
                GrammarWithLexer {
                    name: Some("first".to_string()),
                    json_schema: Some(schema.clone()),
                    ..Default::default()
                },
                GrammarWithLexer {
                    name: Some("second".to_string()),
                    json_schema: Some(schema),
                    ..Default::default()
                },
                GrammarWithLexer {
                    name: Some("combined".to_string()),
                    json_schema: Some(combined_schema),
                    ..Default::default()
                },
            ],
            max_tokens: None,
        });

        let (compiled, _) = grammar
            .clone()
            .to_internal(None, ParserLimits::default())
            .expect("multiple bounded-array schemas should compile");
        grammar
            .to_cgrammar(
                None,
                &mut crate::Logger::new(0, 0),
                ParserLimits::default(),
                vec![],
            )
            .expect("bounded-array grammars should fit the compact symbol representation");
        assert!(
            compiled.num_symbols() < 512,
            "bounded-array grammars expanded into {} symbols",
            compiled.num_symbols()
        );
    }

    #[test]
    fn zero_maximum_array_compiles() {
        compile_schema(json!({
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 0
        }))
        .expect("empty bounded array should compile");
    }

    #[test]
    fn optional_unsatisfiable_prefix_preserves_empty_array() {
        compile_schema(json!({
            "type": "array",
            "prefixItems": [false],
            "items": {"type": "string"},
            "maxItems": 30_000
        }))
        .expect("unsatisfiable optional prefix should still allow an empty array");
    }

    #[test]
    fn unsatisfiable_array_bounds_are_rejected() {
        let error = compile_schema(json!({
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 1
        }))
        .err()
        .expect("inverted array bounds should be rejected");

        assert!(error
            .to_string()
            .contains("minItems (2) is greater than maxItems (1)"));
    }

    #[test]
    fn json_quote_default_allows_u_escape() {
        let compiler = Compiler::new(
            JsonCompileOptions::default(),
            GrammarBuilder::new(None, ParserLimits::default()),
        );
        let quoted = compiler.json_quote(RegexAst::EmptyString);
        match quoted {
            RegexAst::JsonQuote(_, opts) => {
                assert_eq!(opts.allowed_escapes, DEFAULT_JSON_ALLOWED_ESCAPES)
            }
            _ => panic!("expected JsonQuote AST"),
        }
        assert_eq!(
            DEFAULT_JSON_ALLOWED_ESCAPES,
            JsonQuoteOptions::regular().allowed_escapes
        );
    }

    #[test]
    fn json_quote_uses_override_from_options() {
        let compiler = Compiler::new(
            JsonCompileOptions {
                json_allowed_escapes: Some("nrbtf\\\"".to_string()),
                ..Default::default()
            },
            GrammarBuilder::new(None, ParserLimits::default()),
        );
        let quoted = compiler.json_quote(RegexAst::EmptyString);
        match quoted {
            RegexAst::JsonQuote(_, opts) => assert_eq!(opts.allowed_escapes, "nrbtf\\\""),
            _ => panic!("expected JsonQuote AST"),
        }
    }

    /// Reuses one Unicode-scalar expression while keeping different string bounds distinct.
    #[test]
    fn general_unicode_strings_reuse_compiled_regex() {
        let mut compiler = Compiler::new(
            JsonCompileOptions::default(),
            GrammarBuilder::new(None, ParserLimits::default()),
        );

        let unbounded = compiler.json_general_unicode_string(0, None).unwrap();
        let scalar = compiler.general_unicode_scalar_cache.unwrap();
        let repeated = compiler.json_general_unicode_string(0, None).unwrap();
        let bounded = compiler.json_general_unicode_string(1, Some(1)).unwrap();

        match (unbounded, repeated, bounded) {
            (RegexAst::ExprRef(first), RegexAst::ExprRef(second), RegexAst::ExprRef(third)) => {
                assert_eq!(first, second);
                assert_ne!(first, third);
            }
            _ => panic!("expected cached regex expression references"),
        }
        assert_eq!(compiler.general_unicode_scalar_cache, Some(scalar));
        assert_eq!(compiler.general_unicode_string_cache.len(), 2);
    }

    /// Returns invalid escape-policy errors for every schema shape instead of panicking.
    #[test]
    fn invalid_json_allowed_escapes_propagate_for_all_schema_shapes() {
        for allow_general_unicode_escapes in [false, true] {
            let options = JsonCompileOptions {
                json_allowed_escapes: Some("x".to_string()),
                json_allow_general_unicode_escapes: allow_general_unicode_escapes,
                ..JsonCompileOptions::default()
            };

            for schema in [
                json!({ "type": "string" }),
                json!({ "type": "object" }),
                json!({}),
            ] {
                let error = options
                    .json_to_llg(GrammarBuilder::new(None, ParserLimits::default()), schema)
                    .err()
                    .expect("invalid JSON escape settings should return an error");
                assert!(
                    error
                        .to_string()
                        .contains("invalid escape character in allowed_escapes: x"),
                    "unexpected error: {error}"
                );
            }
        }
    }

    #[test]
    fn x_guidance_accepts_json_allowed_escapes() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "x": { "type": "string" }
            },
            "required": ["x"],
            "additionalProperties": false,
            "x-guidance": {
                "json_allowed_escapes": "nrbtf\\\""
            }
        });

        let builder = GrammarBuilder::new(None, ParserLimits::default());
        JsonCompileOptions::default()
            .json_to_llg_with_overrides(builder, schema.clone())
            .expect("schema with x-guidance override should compile");

        // Ensure callers can still stamp options back into schema.
        JsonCompileOptions {
            json_allowed_escapes: Some("nrbtf\\\"".to_string()),
            ..Default::default()
        }
        .apply_to(&mut schema);
        assert_eq!(
            schema["x-guidance"]["json_allowed_escapes"],
            Value::String("nrbtf\\\"".to_string())
        );
    }
}

#[cfg(test)]
mod root_key_encoding_tests {
    use super::*;
    use crate::api::ParserLimits;

    fn bounded() -> JsonCompileOptions {
        JsonCompileOptions {
            root_object_key_policy: Some(RootObjectKeyPolicy { min: 1, max: 256 }),
            ..Default::default()
        }
    }
    fn compile(options: &JsonCompileOptions, schema: Value) -> Result<GrammarResult> {
        options.json_to_llg(GrammarBuilder::new(None, ParserLimits::default()), schema)
    }

    fn failure<T>(result: Result<T>) -> anyhow::Error {
        match result {
            Err(error) => error,
            Ok(_) => panic!("expected compiler error"),
        }
    }

    #[test]
    fn root_key_bounds_follow_references_without_restricting_nested_values_or_defaults() {
        let invalid = json!({"$id":"https://example.invalid/root","$defs":{"Args":{"type":"object","properties":{"":{"type":"integer"}},"required":[""],"additionalProperties":false}},"$ref":"#/$defs/Args"});
        assert!(compile(&JsonCompileOptions::default(), invalid.clone()).is_ok());
        assert!(
            compile(&bounded(), invalid).is_err_and(|error| error.is::<RootObjectKeyPolicyError>())
        );
        for key in ["x".repeat(257), "é".repeat(129)] {
            let invalid = json!({"const":{key:1}});
            assert!(compile(&bounded(), invalid)
                .is_err_and(|error| error.is::<RootObjectKeyPolicyError>()));
        }
        let nested = json!({"$id":"https://example.invalid/root","$defs":{"Args":{"type":"object","properties":{"outer":{"const":{"":1}}},"required":["outer"],"additionalProperties":false}},"$ref":"#/$defs/Args"});
        assert!(compile(&bounded(), nested).is_ok());
        assert!(compile(&bounded(), json!({"type":"object","patternProperties":{"^$":{"type":"integer"}},"additionalProperties":false,"minProperties":1})).is_err_and(|error| error.is::<RootObjectKeyPolicyError>()));
        assert!(compile(
            &bounded(),
            json!({"anyOf":[{"const":{"":1}},{"const":{"valid":1}}]})
        )
        .is_ok());
    }

    #[test]
    fn root_key_policy_counts_decoded_utf8_bytes_and_preserves_option_serialization() {
        for (key, allowed) in [
            ("".to_owned(), false),
            ("x".repeat(256), true),
            ("x".repeat(257), false),
            ("é".repeat(128), true),
            ("é".repeat(129), false),
            ("\"&<>\0é".to_owned(), true),
        ] {
            let schema = json!({"type":"object","properties":{(key.clone()):{"type":"integer"}},"required":[key],"additionalProperties":false});
            let result = compile(&bounded(), schema);
            assert_eq!(result.is_ok(), allowed, "{key:?}");
            if !allowed {
                assert!(result.is_err_and(|error| error.is::<RootObjectKeyPolicyError>()));
            }
        }
        assert!(compile(&bounded(), json!({"type":"object","patternProperties":{"^a$":{"type":"integer"}},"additionalProperties":false,"minProperties":2})).is_err_and(|error| error.is::<RootObjectKeyPolicyError>()));
        assert!(compile(&bounded(), json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"minProperties":2,"maxProperties":2})).is_ok());
        assert!(compile(&bounded(), json!({"type":"object","properties":{"a":{"type":"integer"}},"patternProperties":{"^a$":false},"required":["a"]})).is_err_and(|error| error.is::<UnsatisfiableSchemaError>()));
        assert!(serde_json::to_value(JsonCompileOptions::default())
            .unwrap()
            .get("root_object_key_policy")
            .is_none());
        let original = bounded();
        let cloned = original.clone();
        let encoded = serde_json::to_value(cloned).unwrap();
        assert_eq!(original.root_object_key_policy.unwrap().min, 1);
        let decoded: JsonCompileOptions = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.root_object_key_policy.unwrap().max, 256);
        assert!(compile(&decoded, json!({"const":{"":1}}))
            .is_err_and(|error| error.is::<RootObjectKeyPolicyError>()));
    }
    #[test]
    fn root_key_policy_preserves_unsupported_values_before_name_capability_errors() {
        let unsupported = json!({"oneOf":[{"type":"number"},{"type":"integer"}]});
        for required in [false, true] {
            for referenced in [false, true] {
                let mut schema = json!({"$id":"https://example.invalid/policy-errors","$defs":{"Value":unsupported},"type":"object","properties":{"":if referenced {json!({"$ref":"#/$defs/Value"})} else {unsupported.clone()}},"additionalProperties":false});
                if required {
                    schema["required"] = json!([""]);
                }
                let original = failure(compile(&JsonCompileOptions::default(), schema.clone()));
                let constrained = failure(compile(&bounded(), schema));
                assert!(!constrained.is::<RootObjectKeyPolicyError>());
                assert!(!constrained.is::<UnsatisfiableSchemaError>());
                assert_eq!(
                    original.root_cause().to_string(),
                    constrained.root_cause().to_string()
                );
            }
        }
        // A deferred required-name capability error cannot conceal a later
        // declared field's unsupported schema either.
        let schema = json!({"type":"object","properties":{"":{"type":"integer"},"later":unsupported},"required":[""]});
        let error = failure(compile(&bounded(), schema));
        assert!(!error.is::<RootObjectKeyPolicyError>());
        assert!(!error.is::<UnsatisfiableSchemaError>());
    }

    #[test]
    fn root_key_policy_preserves_dynamic_and_alternative_value_capability_errors() {
        let unsupported = json!({"oneOf":[{"type":"number"},{"type":"integer"}]});
        for schema in [
            json!({"type":"object","additionalProperties":unsupported}),
            json!({"type":"object","patternProperties":{"^a$":unsupported},"additionalProperties":false,"minProperties":1}),
            json!({"anyOf":[{"type":"object","properties":{"":unsupported},"required":[""]},{"const":{"valid":1}}]}),
        ] {
            let original = failure(compile(&JsonCompileOptions::default(), schema.clone()));
            let constrained = failure(compile(&bounded(), schema));
            assert!(!constrained.is::<RootObjectKeyPolicyError>());
            assert!(!constrained.is::<UnsatisfiableSchemaError>());
            assert_eq!(
                original.root_cause().to_string(),
                constrained.root_cause().to_string()
            );
        }
    }
    #[test]
    fn root_key_policy_preserves_original_empty_cardinality_and_alternatives() {
        let cases = [
            json!({"type":"object","additionalProperties":false,"minProperties":1}),
            json!({"type":"object","properties":{"ok":false},"additionalProperties":false,"minProperties":1}),
            json!({"type":"object","minProperties":2,"maxProperties":1}),
            json!({"anyOf":[false,{"type":"object","additionalProperties":false,"minProperties":1}]}),
        ];
        for schema in cases {
            let original = failure(compile(&JsonCompileOptions::default(), schema.clone()));
            let policy = failure(compile(&bounded(), schema));
            assert!(original.is::<UnsatisfiableSchemaError>());
            assert!(policy.is::<UnsatisfiableSchemaError>());
            assert!(!policy.is::<RootObjectKeyPolicyError>());
            assert_eq!(
                original.root_cause().to_string(),
                policy.root_cause().to_string()
            );
        }
        let unsupported = json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"minProperties":2,"maxProperties":2});
        let original = failure(compile(&JsonCompileOptions::default(), unsupported.clone()));
        let policy = failure(compile(&bounded(), unsupported));
        assert!(!policy.is::<RootObjectKeyPolicyError>());
        assert!(!policy.is::<UnsatisfiableSchemaError>());
        assert_eq!(
            original.root_cause().to_string(),
            policy.root_cause().to_string()
        );
        let empty_only_under_policy = json!({"type":"object","properties":{"":{"type":"integer"}},"required":[""],"additionalProperties":false});
        assert!(compile(
            &JsonCompileOptions::default(),
            empty_only_under_policy.clone()
        )
        .is_ok());
        assert!(compile(&bounded(), empty_only_under_policy)
            .is_err_and(|error| error.is::<RootObjectKeyPolicyError>()));
    }
    #[test]
    fn invalid_root_key_policy_precedes_an_originally_false_schema() {
        let options = JsonCompileOptions {
            root_object_key_policy: Some(RootObjectKeyPolicy { min: 2, max: 1 }),
            ..Default::default()
        };
        let error = failure(compile(&options, json!(false)));
        assert!(!error.is::<UnsatisfiableSchemaError>());
        assert!(!error.is::<RootObjectKeyPolicyError>());
        assert_eq!(
            error.to_string(),
            "root object key byte bounds require min <= max"
        );
    }
    #[test]
    fn root_policy_preserves_builder_resource_errors_before_capability_errors() {
        let limits = ParserLimits {
            max_grammar_size: 0,
            ..Default::default()
        };
        let result =
            bounded().json_to_llg(GrammarBuilder::new(None, limits), json!({"const":{"":1}}));
        let error = failure(result);
        assert!(!error.is::<RootObjectKeyPolicyError>());
        assert!(!error.is::<UnsatisfiableSchemaError>());
        assert!(error.to_string().contains("grammar"), "{error:#}");
    }

    #[cfg(feature = "referencing")]
    #[test]
    fn root_policy_reuses_resolved_external_references_without_retrieval() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Retriever(Arc<AtomicUsize>);
        impl crate::json::Retrieve for Retriever {
            fn retrieve(&self, _: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(
                    json!({"type":"object","properties":{"valid":{"type":"integer"}},"required":["valid"],"additionalProperties":false}),
                )
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let options = JsonCompileOptions {
            retriever: Some(crate::json::RetrieveWrapper::new(Arc::new(Retriever(
                calls.clone(),
            )))),
            ..bounded()
        };
        compile(
            &options,
            json!({"$ref":"https://example.com/root-arguments"}),
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
