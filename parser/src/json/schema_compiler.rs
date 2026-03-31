//! Iterative schema compiler: converts RawSchema → Schema IR.
//!
//! Replaces the recursive compilation logic from schema.rs with a compiler struct
//! that owns all state directly (no `Rc<RefCell>`).

use crate::{regex_to_lark, HashMap, HashSet, JsonCompileOptions};
use anyhow::{anyhow, bail, Result};
use derivre::RegexAst;
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

use super::formats::lookup_format;
use super::numeric::Decimal;
use super::raw_schema::{
    Applicator, BoolOrNumber, Constraint, ItemsValue, KeywordEntry, RawSchema, SchemaObject,
    TypeValue,
};
use super::schema::{
    ArraySchema, NumberSchema, ObjectSchema, OptSchemaExt, Schema, SchemaBuilderOptions,
    StringSchema, IMPLEMENTED, META_AND_ANNOTATIONS,
};
use super::shared_context::{BuiltSchema, PatternPropertyCache};
use super::{Draft, RefResolver};

const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];

/// Deferred intersection for circular $ref with sibling keywords.
/// Created when `intersect_ref` encounters a ref that is currently being compiled.
struct PendingIntersection {
    /// Synthetic definition name (e.g. `#/circular_0`) to store the result in.
    synthetic_uri: String,
    /// The original ref URI whose definition will be available after compilation.
    ref_uri: String,
    /// The sibling-derived schema to intersect with the resolved ref.
    sibling_schema: Schema,
    /// Whether the ref should be on the left side of the intersection.
    ref_first: bool,
}

/// Schema compiler that owns all state directly.
pub(crate) struct SchemaCompiler {
    definitions: HashMap<String, Schema>,
    seen_refs: HashSet<String>,
    pattern_cache: PatternPropertyCache,
    warnings: Vec<String>,
    n_compiled: usize,
    options: SchemaBuilderOptions,
    draft: Draft,
    resolver: RefResolver,
    /// Stack of base URIs for nested `$id` scoping.
    /// Empty string means use the root base URI.
    base_uri_stack: Vec<String>,
    /// Deferred intersections for circular refs with sibling keywords.
    pending_intersections: Vec<PendingIntersection>,
    pending_counter: usize,
    /// Refs currently being intersected — used to detect intersection cycles.
    /// When a ref is encountered while already in this set, the cycle is broken
    /// by returning Ref(ref_uri) directly (an over-approximation).
    intersecting_refs: HashSet<String>,
}

impl SchemaCompiler {
    /// Entry point: compile a JSON value into a BuiltSchema.
    pub fn compile(contents: Value, opts: &JsonCompileOptions) -> Result<BuiltSchema> {
        if let Some(b) = contents.as_bool() {
            let s = if b {
                Schema::Any
            } else {
                Schema::false_schema()
            };
            return Ok(BuiltSchema::simple(s));
        }

        let resolver = RefResolver::new(&contents, opts.retriever.as_ref())?;
        let draft = resolver.draft();

        let mut compiler = SchemaCompiler {
            definitions: HashMap::default(),
            seen_refs: HashSet::default(),
            pattern_cache: PatternPropertyCache::default(),
            warnings: Vec::new(),
            n_compiled: 0,
            options: SchemaBuilderOptions {
                lenient: opts.lenient,
                ..SchemaBuilderOptions::default()
            },
            draft,
            resolver,
            base_uri_stack: Vec::new(),
            pending_intersections: Vec::new(),
            pending_counter: 0,
            intersecting_refs: HashSet::default(),
        };

        let raw: RawSchema = serde_json::from_value(contents)
            .map_err(|e| anyhow!("Failed to parse JSON schema: {}", e))?;
        let schema = compiler.compile_raw(&raw)?;

        // Process deferred circular-ref intersections now that all definitions are resolved.
        while let Some(pending) = compiler.pending_intersections.pop() {
            let resolved = compiler
                .definitions
                .get(&pending.ref_uri)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "circular ref definition still missing after compilation: {}",
                        pending.ref_uri
                    )
                })?;
            // Register both the original ref and synthetic URI so that:
            // - define_ref won't try to JSON-pointer-resolve the synthetic URI
            // - cycle detection catches self-references during intersection
            compiler.seen_refs.insert(pending.synthetic_uri.clone());
            compiler
                .intersecting_refs
                .insert(pending.ref_uri.clone());
            compiler
                .intersecting_refs
                .insert(pending.synthetic_uri.clone());
            let result = if pending.ref_first {
                compiler.intersect(resolved, pending.sibling_schema, 0)?
            } else {
                compiler.intersect(pending.sibling_schema, resolved, 0)?
            };
            compiler.intersecting_refs.remove(&pending.ref_uri);
            compiler.intersecting_refs.remove(&pending.synthetic_uri);
            compiler.definitions.insert(pending.synthetic_uri, result);
        }

        Ok(BuiltSchema {
            schema,
            definitions: compiler.definitions,
            warnings: compiler.warnings,
            pattern_cache: compiler.pattern_cache,
        })
    }

    fn increment(&mut self) -> Result<()> {
        self.n_compiled += 1;
        if self.n_compiled > self.options.max_size {
            bail!("schema too large");
        }
        Ok(())
    }

    fn record_warning(&mut self, msg: String) {
        self.warnings.push(msg);
    }

    // ---- Compilation ----

    fn compile_raw(&mut self, raw: &RawSchema) -> Result<Schema> {
        match raw {
            RawSchema::Bool(true) => Ok(Schema::Any),
            RawSchema::Bool(false) => Ok(Schema::false_schema()),
            RawSchema::Object(obj) => {
                let schema = self.compile_schema_object(obj)?;
                Ok(self.normalize(schema))
            }
        }
    }

    fn compile_schema_object(&mut self, obj: &SchemaObject) -> Result<Schema> {
        self.increment()?;

        // If the schema has no constraints or applicators, it's unconstrained
        if self.is_only_meta(obj) {
            return Ok(Schema::Any);
        }

        // If this schema has $id (or `id` for Draft 4), push a new base URI scope
        // so that $ref resolution within this schema uses the correct base.
        // Draft 4 only recognizes bare `id`; Draft 6+ recognizes `$id`.
        // In legacy drafts (Draft 4-7), `$ref` replaces the entire schema, so `$id`
        // is ignored when `$ref` is present (matching the referencing crate's behavior).
        let has_ref = obj
            .keyword_order
            .iter()
            .any(|e| matches!(e, KeywordEntry::Applicator(Applicator::Ref(_))));
        let schema_id = match self.draft {
            Draft::Draft4 => obj.legacy_id.as_deref(),
            Draft::Draft6 | Draft::Draft7 => obj.dollar_id.as_deref(),
            _ => obj.dollar_id.as_deref(),
        };
        // In legacy drafts, $id/$id is ignored when $ref is present or when it starts with #
        let schema_id = schema_id.filter(|id| {
            if self.draft <= Draft::Draft7 && (has_ref || id.starts_with('#')) {
                return false;
            }
            true
        });
        if let Some(id) = schema_id {
            let new_base = self.normalize_ref(id).unwrap_or_default();
            // Strip fragment — only the URI path is used as the base for $ref resolution.
            let new_base = match new_base.split_once('#') {
                Some((base, _)) => base.to_string(),
                None => new_base,
            };
            self.base_uri_stack.push(new_base);
        }

        let result = self.compile_schema_object_inner(obj);

        if schema_id.is_some() {
            self.base_uri_stack.pop();
        }

        result
    }

    fn compile_schema_object_inner(&mut self, obj: &SchemaObject) -> Result<Schema> {
        // Check for unimplemented keys
        self.check_unimplemented_keys(obj)?;

        // Pre-scan the full keyword_order to find the type constraint, all property keys
        // and all prefixItems. This enables us to inject context into earlier batches when
        // additionalProperties or items appears before properties/prefixItems in JSON order.
        let type_constraint = obj.keyword_order.iter().find_map(|e| match e {
            KeywordEntry::Constraint(c @ Constraint::Type(_)) => Some(c),
            _ => None,
        });
        let all_property_keys: Vec<String> = obj
            .keyword_order
            .iter()
            .filter_map(|e| match e {
                KeywordEntry::Constraint(Constraint::Properties(p)) => Some(p),
                _ => None,
            })
            .flat_map(|p| p.keys().cloned())
            .collect();
        let all_prefix_count: usize = obj
            .keyword_order
            .iter()
            .filter_map(|e| match e {
                KeywordEntry::Constraint(Constraint::PrefixItems(p)) => Some(p.len()),
                KeywordEntry::Constraint(Constraint::Items(
                    crate::json::raw_schema::ItemsValue::Array(a),
                )) => Some(a.len()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        // Build dummy constraints to inject when needed.
        // Dummy properties: a Properties map with just the keys (values are Schema::Any/true)
        // so additionalProperties knows which keys to exempt.
        let dummy_properties = if !all_property_keys.is_empty() {
            let map: IndexMap<String, RawSchema> = all_property_keys
                .iter()
                .map(|k| (k.clone(), RawSchema::Bool(true)))
                .collect();
            Some(Constraint::Properties(Box::new(map)))
        } else {
            None
        };
        let dummy_prefix_items = if all_prefix_count > 0 {
            Some(Constraint::PrefixItems(vec![
                RawSchema::Bool(true);
                all_prefix_count
            ]))
        } else {
            None
        };

        // Process keywords in keyword_order, batching constraints and flushing on applicators.
        let mut result = Schema::Any;
        let mut batch: Vec<&Constraint> = Vec::new();

        for entry in &obj.keyword_order {
            match entry {
                KeywordEntry::Constraint(c) => {
                    batch.push(c);
                }
                KeywordEntry::Unknown(_) => continue,
                KeywordEntry::Applicator(applicator) => {
                    // Flush pending constraints
                    if !batch.is_empty() {
                        self.flush_batch(
                            &mut result,
                            &mut batch,
                            type_constraint,
                            &dummy_properties,
                            &dummy_prefix_items,
                        )?;
                    }
                    // Apply the applicator
                    result = self.apply_applicator(result, applicator)?;
                }
            }
        }

        // Flush any remaining constraints
        if !batch.is_empty() {
            self.flush_batch(
                &mut result,
                &mut batch,
                type_constraint,
                &dummy_properties,
                &dummy_prefix_items,
            )?;
        }

        Ok(result)
    }

    /// Flush a constraint batch, injecting type/properties/prefixItems context when needed.
    fn flush_batch<'a>(
        &mut self,
        result: &mut Schema,
        batch: &mut Vec<&'a Constraint>,
        type_constraint: Option<&'a Constraint>,
        dummy_properties: &'a Option<Constraint>,
        dummy_prefix_items: &'a Option<Constraint>,
    ) -> Result<()> {
        // Always include type in every batch (as the old code did)
        let has_type = batch.iter().any(|c| matches!(c, Constraint::Type(_)));
        if !has_type {
            if let Some(tc) = type_constraint {
                batch.push(tc);
            }
        }
        // Inject dummy properties if AP is present but Properties is not
        let has_props = batch.iter().any(|c| matches!(c, Constraint::Properties(_)));
        let has_ap = batch
            .iter()
            .any(|c| matches!(c, Constraint::AdditionalProperties(_)));
        if has_ap && !has_props {
            if let Some(dp) = dummy_properties {
                batch.push(dp);
            }
        }
        // Inject dummy prefixItems if Items is present but PrefixItems is not
        let has_prefix = batch.iter().any(|c| {
            matches!(c, Constraint::PrefixItems(_))
                || matches!(
                    c,
                    Constraint::Items(crate::json::raw_schema::ItemsValue::Array(_))
                )
        });
        let has_items = batch
            .iter()
            .any(|c| matches!(c, Constraint::Items(_) | Constraint::AdditionalItems(_)));
        if has_items && !has_prefix {
            if let Some(dp) = dummy_prefix_items {
                batch.push(dp);
            }
        }

        let base = self.compile_constraints(batch)?;
        *result = self.intersect(result.clone(), base, 0)?;
        batch.clear();
        Ok(())
    }

    fn is_only_meta(&self, obj: &SchemaObject) -> bool {
        obj.keyword_order.iter().all(|entry| match entry {
            KeywordEntry::Constraint(_) | KeywordEntry::Applicator(_) => false,
            KeywordEntry::Unknown(key) => !self.draft.is_known_keyword(key),
        })
    }

    fn check_unimplemented_keys(&mut self, obj: &SchemaObject) -> Result<()> {
        let mut unimplemented: Vec<&str> = obj
            .keyword_order
            .iter()
            .filter_map(|entry| match entry {
                KeywordEntry::Unknown(key)
                    if self.draft.is_known_keyword(key)
                        && !IMPLEMENTED.contains(&key.as_str())
                        && !META_AND_ANNOTATIONS.contains(&key.as_str()) =>
                {
                    Some(key.as_str())
                }
                _ => None,
            })
            .collect();
        if !unimplemented.is_empty() {
            unimplemented.sort();
            let msg = format!("Unimplemented keys: {unimplemented:?}");
            if self.options.lenient {
                self.record_warning(msg);
            } else {
                bail!(msg);
            }
        }
        Ok(())
    }

    /// Compile a batch of constraints into a Schema.
    /// Extracts `type` from the batch to determine dispatch.
    fn compile_constraints(&mut self, batch: &[&Constraint]) -> Result<Schema> {
        let type_value = batch.iter().find_map(|c| match c {
            Constraint::Type(tv) => Some(tv),
            _ => None,
        });

        let types: &[&str] = match type_value {
            Some(TypeValue::Single(tp)) => &[tp],
            Some(TypeValue::Array(types)) => {
                return self
                    .compile_typed(&types.iter().map(|s| s.as_str()).collect::<Vec<_>>(), batch);
            }
            None => &TYPES,
        };
        self.compile_typed(types, batch)
    }

    fn compile_typed(&mut self, types: &[&str], batch: &[&Constraint]) -> Result<Schema> {
        let mut options = Vec::new();
        for tp in types {
            let schema = self.compile_type(tp, batch)?;
            options.push(schema);
        }
        if options.len() == 1 {
            Ok(options.swap_remove(0))
        } else {
            Ok(Schema::AnyOf(options))
        }
    }

    fn compile_type(&mut self, tp: &str, batch: &[&Constraint]) -> Result<Schema> {
        self.increment()?;
        match tp {
            "null" => Ok(Schema::Null),
            "boolean" => Ok(Schema::Boolean(None)),
            "number" | "integer" => self.compile_numeric(batch, tp == "integer"),
            "string" => self.compile_string(batch),
            "array" => self.compile_array(batch),
            "object" => self.compile_object(batch),
            _ => bail!("Invalid type: {}", tp),
        }
    }

    fn compile_numeric(&mut self, batch: &[&Constraint], integer: bool) -> Result<Schema> {
        let mut minimum = None;
        let mut maximum = None;
        let mut excl_min = None;
        let mut excl_max = None;
        let mut mult_of = None;

        for c in batch {
            match c {
                Constraint::Minimum(n) => minimum = Some(*n),
                Constraint::Maximum(n) => maximum = Some(*n),
                Constraint::ExclusiveMinimum(v) => excl_min = Some(v),
                Constraint::ExclusiveMaximum(v) => excl_max = Some(v),
                Constraint::MultipleOf(n) => mult_of = Some(*n),
                _ => {}
            }
        }

        let exclusive_minimum = match excl_min {
            None | Some(BoolOrNumber::Bool(false)) => None,
            Some(BoolOrNumber::Bool(true)) => minimum,
            Some(BoolOrNumber::Number(n)) => Some(*n),
        };
        let exclusive_maximum = match excl_max {
            None | Some(BoolOrNumber::Bool(false)) => None,
            Some(BoolOrNumber::Bool(true)) => maximum,
            Some(BoolOrNumber::Number(n)) => Some(*n),
        };
        let multiple_of = match mult_of {
            None => None,
            Some(val) => Some(Decimal::try_from(val.abs())?),
        };
        Ok(Schema::Number(NumberSchema {
            minimum,
            maximum,
            exclusive_minimum,
            exclusive_maximum,
            integer,
            multiple_of,
        }))
    }

    fn compile_string(&mut self, batch: &[&Constraint]) -> Result<Schema> {
        let mut min_length = 0;
        let mut max_length = None;
        let mut pattern = None;
        let mut format = None;

        for c in batch {
            match c {
                Constraint::MinLength(n) => min_length = *n,
                Constraint::MaxLength(n) => max_length = Some(*n),
                Constraint::Pattern(s) => pattern = Some(s.as_str()),
                Constraint::Format(s) => format = Some(s.as_str()),
                _ => {}
            }
        }

        let pattern_rx = pattern.map(|s| RegexAst::SearchRegex(regex_to_lark(s, "dw")));
        let format_rx = match format {
            None => None,
            Some(key) => {
                if let Some(fmt) = lookup_format(key) {
                    Some(RegexAst::Regex(fmt.to_string()))
                } else {
                    let msg = format!("Unknown format: {key}");
                    if self.options.lenient {
                        self.record_warning(msg);
                        None
                    } else {
                        bail!(msg);
                    }
                }
            }
        };
        let regex = match (pattern_rx, format_rx) {
            (None, None) => None,
            (None, Some(fmt)) => Some(fmt),
            (Some(pat), None) => Some(pat),
            (Some(pat), Some(fmt)) => Some(RegexAst::And(vec![pat, fmt])),
        };
        Ok(Schema::String(StringSchema {
            min_length,
            max_length,
            regex,
        }))
    }

    fn compile_array(&mut self, batch: &[&Constraint]) -> Result<Schema> {
        let mut min_items = 0;
        let mut max_items = None;
        let mut items = None;
        let mut additional_items = None;
        let mut prefix_items = None;

        for c in batch {
            match c {
                Constraint::MinItems(n) => min_items = *n,
                Constraint::MaxItems(n) => max_items = Some(*n),
                Constraint::Items(v) => items = Some(v),
                Constraint::AdditionalItems(v) => additional_items = Some(v.as_ref()),
                Constraint::PrefixItems(v) => prefix_items = Some(v.as_slice()),
                _ => {}
            }
        }

        // Resolve draft differences: in Draft 4–2019-09, `items` as an array means
        // prefixItems, and `additionalItems` means items. In Draft 2020-12, `items`
        // is always a single schema and `prefixItems` is the array form.
        let use_legacy = self.draft <= Draft::Draft201909
            || additional_items.is_some()
            || matches!(items, Some(ItemsValue::Array(_)));

        let (prefix_raw, items_raw) = if use_legacy {
            match items {
                // items is array → treat as prefixItems; additionalItems becomes items
                Some(ItemsValue::Array(arr)) => (Some(arr.as_slice()), additional_items),
                // items is a schema → use as items; additionalItems is ignored
                Some(ItemsValue::Schema(s)) => (None, Some(s.as_ref())),
                // No items keyword → additionalItems has nothing to be "additional" to,
                // so it is ignored (matches the old code behavior).
                None => (None, None),
            }
        } else {
            (
                prefix_items,
                items.map(|iv| match iv {
                    ItemsValue::Schema(s) => s.as_ref(),
                    ItemsValue::Array(_) => unreachable!("items array in draft 2020-12"),
                }),
            )
        };

        let compiled_prefix = match prefix_raw {
            None => vec![],
            Some(arr) => arr
                .iter()
                .map(|item| self.compile_raw(item))
                .collect::<Result<Vec<_>>>()?,
        };
        let compiled_items = match items_raw {
            None => None,
            Some(val) => Some(Box::new(self.compile_raw(val)?)),
        };

        Ok(Schema::Array(ArraySchema {
            min_items,
            max_items,
            prefix_items: compiled_prefix,
            items: compiled_items,
        }))
    }

    fn compile_object(&mut self, batch: &[&Constraint]) -> Result<Schema> {
        let mut raw_properties = None;
        let mut raw_pattern_properties = None;
        let mut raw_additional_properties = None;
        let mut raw_required = None;
        let mut min_properties = 0;
        let mut max_properties = None;

        for c in batch {
            match c {
                Constraint::Properties(p) => raw_properties = Some(p.as_ref()),
                Constraint::PatternProperties(p) => raw_pattern_properties = Some(p.as_ref()),
                Constraint::AdditionalProperties(p) => raw_additional_properties = Some(p.as_ref()),
                Constraint::Required(r) => raw_required = Some(r),
                Constraint::MinProperties(n) => min_properties = *n,
                Constraint::MaxProperties(n) => max_properties = Some(*n),
                _ => {}
            }
        }

        let mut properties = match raw_properties {
            None => IndexMap::new(),
            Some(props) => props
                .iter()
                .map(|(k, v)| self.compile_raw(v).map(|s| (k.clone(), s)))
                .collect::<Result<IndexMap<String, Schema>>>()?,
        };
        let pattern_properties = match raw_pattern_properties {
            None => IndexMap::new(),
            Some(pp) => {
                let result: IndexMap<String, Schema> = pp
                    .iter()
                    .map(|(k, v)| self.compile_raw(v).map(|s| (k.clone(), s)))
                    .collect::<Result<_>>()?;
                self.pattern_cache
                    .check_disjoint(&result.keys().collect::<Vec<_>>())?;
                result
            }
        };

        // Per JSON Schema spec, a named property must validate against BOTH its
        // properties schema AND any matching patternProperties schema. Pre-intersect
        // here so that stage 2 (which excludes named properties from pattern regexes)
        // produces correct constraints.
        if !pattern_properties.is_empty() {
            for (name, prop_schema) in properties.iter_mut() {
                for (pattern, pat_schema) in pattern_properties.iter() {
                    if self.pattern_cache.is_match(pattern, name)? {
                        let owned = std::mem::replace(prop_schema, Schema::Null);
                        *prop_schema = self.intersect(owned, pat_schema.clone(), 0)?;
                        break; // patterns are disjoint, at most one match
                    }
                }
            }
        }

        let additional_properties = match raw_additional_properties {
            None => None,
            Some(val) => Some(Box::new(self.compile_raw(val)?)),
        };
        let required = match raw_required {
            None => IndexSet::new(),
            Some(arr) => arr.iter().cloned().collect(),
        };

        Ok(mk_object_schema(ObjectSchema {
            properties,
            pattern_properties,
            additional_properties,
            required,
            min_properties,
            max_properties,
        }))
    }

    // ---- Applicators ----

    fn apply_applicator(&mut self, result: Schema, applicator: &Applicator) -> Result<Schema> {
        match applicator {
            Applicator::Const(v) => {
                let schema = compile_const(v)?;
                self.intersect(result, schema, 0)
            }
            Applicator::Enum(instances) => {
                let options = instances
                    .iter()
                    .map(compile_const)
                    .collect::<Result<Vec<_>>>()?;
                self.intersect(result, Schema::AnyOf(options), 0)
            }
            Applicator::AllOf(all_of) => {
                let mut r = result;
                for value in all_of {
                    let schema = self.compile_raw(value)?;
                    r = self.intersect(r, schema, 0)?;
                }
                Ok(r)
            }
            Applicator::AnyOf(any_of) => {
                let options = any_of
                    .iter()
                    .map(|value| self.compile_raw(value))
                    .collect::<Result<Vec<_>>>()?;
                self.intersect(result, Schema::AnyOf(options), 0)
            }
            Applicator::OneOf(one_of) => {
                let options = one_of
                    .iter()
                    .map(|value| self.compile_raw(value))
                    .collect::<Result<Vec<_>>>()?;
                self.intersect(result, Schema::OneOf(options), 0)
            }
            Applicator::Ref(reference) => {
                let uri = self.normalize_ref(reference)?;
                if matches!(result, Schema::Any) {
                    self.define_ref(&uri)?;
                    Ok(Schema::Ref(uri))
                } else {
                    self.intersect_ref(&uri, result, false, 0)
                }
            }
        }
    }

    // ---- Ref handling ----

    /// Normalize a `$ref` using the current base URI scope.
    fn normalize_ref(&self, reference: &str) -> Result<String> {
        if let Some(base) = self.base_uri_stack.last().filter(|b| !b.is_empty()) {
            self.resolver.normalize_ref_relative(reference, base)
        } else {
            self.resolver.normalize_ref(reference)
        }
    }

    fn define_ref(&mut self, ref_uri: &str) -> Result<()> {
        if !self.seen_refs.contains(ref_uri) && !self.definitions.contains_key(ref_uri) {
            self.seen_refs.insert(ref_uri.to_string());
            let resolved = self.lookup_and_compile_ref(ref_uri)?;
            self.definitions.insert(ref_uri.to_string(), resolved);
        }
        Ok(())
    }

    fn intersect_ref(
        &mut self,
        ref_uri: &str,
        schema: Schema,
        ref_first: bool,
        stack_level: usize,
    ) -> Result<Schema> {
        self.define_ref(ref_uri)?;

        // Cycle detection: if this ref is already being intersected up the
        // call stack, return it directly to break the cycle. This is exact
        // for A ∩ A and a safe over-approximation for A ∩ B.
        if self.intersecting_refs.contains(ref_uri) {
            return Ok(Schema::Ref(ref_uri.to_string()));
        }

        match self.definitions.get(ref_uri).cloned() {
            Some(resolved_schema) => {
                self.intersecting_refs.insert(ref_uri.to_string());
                let result = if ref_first {
                    self.intersect(resolved_schema, schema, stack_level + 1)
                } else {
                    self.intersect(schema, resolved_schema, stack_level + 1)
                };
                self.intersecting_refs.remove(ref_uri);
                result
            }
            None => {
                // Ref is currently being compiled (not yet in definitions).
                let synthetic_uri = format!("#/circular_{}", self.pending_counter);
                self.pending_counter += 1;
                self.pending_intersections.push(PendingIntersection {
                    synthetic_uri: synthetic_uri.clone(),
                    ref_uri: ref_uri.to_string(),
                    sibling_schema: schema,
                    ref_first,
                });
                Ok(Schema::Ref(synthetic_uri))
            }
        }
    }

    fn lookup_and_compile_ref(&mut self, reference: &str) -> Result<Schema> {
        let (content, _new_base) =
            if let Some(base) = self.base_uri_stack.last().filter(|b| !b.is_empty()) {
                self.resolver.lookup_relative(reference, base)?
            } else {
                self.resolver.lookup(reference)?
            };
        let raw: RawSchema = serde_json::from_value(content)
            .map_err(|e| anyhow!("Failed to parse $ref target '{}': {}", reference, e))?;
        self.compile_raw(&raw)
    }

    // ---- Intersection ----

    fn intersect(&mut self, a: Schema, b: Schema, stack_level: usize) -> Result<Schema> {
        self.increment()?;
        if stack_level > self.options.max_stack_level {
            bail!("Schema intersection stack level exceeded");
        }
        let next = stack_level + 1;

        // Unwrap single-element unions early to enable fast paths (e.g., same-ref detection)
        let a = match a {
            Schema::AnyOf(mut opts) | Schema::OneOf(mut opts) if opts.len() == 1 => {
                opts.swap_remove(0)
            }
            other => other,
        };
        let b = match b {
            Schema::AnyOf(mut opts) | Schema::OneOf(mut opts) if opts.len() == 1 => {
                opts.swap_remove(0)
            }
            other => other,
        };

        let merged = match (a, b) {
            // Identity and annihilator
            (Schema::Any, s) | (s, Schema::Any) => s,
            (Schema::Unsatisfiable(r), _) | (_, Schema::Unsatisfiable(r)) => {
                Schema::Unsatisfiable(r)
            }
            // Same ref: intersection with itself is identity
            (Schema::Ref(ref u1), Schema::Ref(ref u2)) if u1 == u2 => Schema::Ref(u1.clone()),
            // Refs: resolve and intersect
            (Schema::Ref(uri), s) => self.intersect_ref(&uri, s, true, next)?,
            (s, Schema::Ref(uri)) => self.intersect_ref(&uri, s, false, next)?,
            // Distribute intersection over unions
            (Schema::OneOf(opts), s) => {
                self.intersect_distribute(opts, s, true, Schema::OneOf, next)?
            }
            (s, Schema::OneOf(opts)) => {
                self.intersect_distribute(opts, s, false, Schema::OneOf, next)?
            }
            (Schema::AnyOf(opts), s) => {
                self.intersect_distribute(opts, s, true, Schema::AnyOf, next)?
            }
            (s, Schema::AnyOf(opts)) => {
                self.intersect_distribute(opts, s, false, Schema::AnyOf, next)?
            }
            // Per-type intersection
            (Schema::Null, Schema::Null) => Schema::Null,
            (Schema::Boolean(v1), Schema::Boolean(v2)) => {
                if v1 == v2 || v2.is_none() {
                    Schema::Boolean(v1)
                } else if v1.is_none() {
                    Schema::Boolean(v2)
                } else {
                    Schema::unsat("incompatible boolean values")
                }
            }
            (Schema::Number(n1), Schema::Number(n2)) => Schema::Number(intersect_numbers(n1, n2)),
            (Schema::String(s1), Schema::String(s2)) => Schema::String(intersect_strings(s1, s2)),
            (Schema::Array(mut a1), Schema::Array(mut a2)) => {
                let len = a1.prefix_items.len().max(a2.prefix_items.len());
                a1.prefix_items.resize_with(len, || a1.items.schema());
                a2.prefix_items.resize_with(len, || a2.items.schema());
                Schema::Array(ArraySchema {
                    min_items: a1.min_items.max(a2.min_items),
                    max_items: opt_min(a1.max_items, a2.max_items),
                    prefix_items: a1
                        .prefix_items
                        .into_iter()
                        .zip(a2.prefix_items)
                        .map(|(i1, i2)| self.intersect(i1, i2, next))
                        .collect::<Result<_>>()?,
                    items: self.intersect_boxed_opts(a1.items, a2.items, next)?,
                })
            }
            (Schema::Object(mut o1), Schema::Object(o2)) => {
                self.intersect_objects(&mut o1, o2, next)?
            }
            // Incompatible types
            _ => Schema::unsat("incompatible types"),
        };
        Ok(self.normalize(merged))
    }

    fn intersect_objects(
        &mut self,
        o1: &mut ObjectSchema,
        o2: ObjectSchema,
        stack_level: usize,
    ) -> Result<Schema> {
        // Merge named properties, intersecting with pattern matches from the other side
        let mut properties = IndexMap::new();
        for (key, prop1) in std::mem::take(&mut o1.properties) {
            let prop2 = self.pattern_cache.property_schema(&o2, &key)?;
            properties.insert(key, self.intersect(prop1, prop2.clone(), stack_level)?);
        }
        for (key, prop2) in o2.properties {
            if properties.contains_key(&key) {
                continue;
            }
            let prop1 = self.pattern_cache.property_schema(o1, &key)?;
            properties.insert(key, self.intersect(prop1.clone(), prop2, stack_level)?);
        }

        // Merge pattern properties
        let pattern_properties = self.intersect_pattern_properties(
            std::mem::take(&mut o1.pattern_properties),
            o2.pattern_properties,
            &o1.additional_properties,
            &o2.additional_properties,
            stack_level,
        )?;

        let additional_properties = self.intersect_boxed_opts(
            o1.additional_properties.take(),
            o2.additional_properties,
            stack_level,
        )?;

        let mut required = std::mem::take(&mut o1.required);
        required.extend(o2.required);

        Ok(mk_object_schema(ObjectSchema {
            properties,
            pattern_properties,
            additional_properties,
            required,
            min_properties: o1.min_properties.max(o2.min_properties),
            max_properties: opt_min(o1.max_properties, o2.max_properties),
        }))
    }

    /// Merge pattern properties from two object schemas during intersection.
    fn intersect_pattern_properties(
        &mut self,
        o1_pp: IndexMap<String, Schema>,
        o2_pp: IndexMap<String, Schema>,
        o1_ap: &Option<Box<Schema>>,
        o2_ap: &Option<Box<Schema>>,
        stack_level: usize,
    ) -> Result<IndexMap<String, Schema>> {
        let mut result = IndexMap::new();
        let mut o2_remaining = o2_pp;
        for (key, prop1) in o1_pp.into_iter() {
            if let Some(prop2) = o2_remaining.shift_remove(&key) {
                result.insert(key, self.intersect(prop1, prop2, stack_level + 1)?);
            } else if let Some(ap) = o2_ap {
                result.insert(
                    key,
                    self.intersect(prop1, ap.as_ref().clone(), stack_level + 1)?,
                );
            } else {
                result.insert(key, prop1);
            }
        }
        for (key, prop2) in o2_remaining.into_iter() {
            if let Some(ap) = o1_ap {
                result.insert(
                    key,
                    self.intersect(prop2, ap.as_ref().clone(), stack_level + 1)?,
                );
            } else {
                result.insert(key, prop2);
            }
        }
        let keys = result.keys().collect::<Vec<_>>();
        if !keys.is_empty() {
            self.pattern_cache.check_disjoint(&keys)?;
        }
        Ok(result)
    }

    /// Distribute intersection of `schema` across each option in `opts`.
    fn intersect_distribute(
        &mut self,
        opts: Vec<Schema>,
        schema: Schema,
        opts_first: bool,
        wrap: fn(Vec<Schema>) -> Schema,
        stack_level: usize,
    ) -> Result<Schema> {
        let results = opts
            .into_iter()
            .map(|opt| {
                if opts_first {
                    self.intersect(opt, schema.clone(), stack_level)
                } else {
                    self.intersect(schema.clone(), opt, stack_level)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(wrap(results))
    }

    /// Intersect two `Option<Box<Schema>>` fields (items, additionalProperties).
    fn intersect_boxed_opts(
        &mut self,
        a: Option<Box<Schema>>,
        b: Option<Box<Schema>>,
        stack_level: usize,
    ) -> Result<Option<Box<Schema>>> {
        match (a, b) {
            (None, None) => Ok(None),
            (None, some) | (some, None) => Ok(some),
            (Some(a), Some(b)) => Ok(Some(Box::new(self.intersect(*a, *b, stack_level)?))),
        }
    }

    // ---- Normalization ----

    /// Flatten nested unions, remove unsatisfiable branches, collapse singletons.
    fn normalize(&mut self, schema: Schema) -> Schema {
        match schema {
            Schema::AnyOf(options) => self.normalize_union(options, false),
            Schema::OneOf(options) => self.normalize_union(options, true),
            other => other,
        }
    }

    /// Shared normalization for AnyOf/OneOf.
    fn normalize_union(&mut self, options: Vec<Schema>, is_one_of: bool) -> Schema {
        let mut valid = Vec::new();
        let mut last_unsat = None;

        for option in options {
            if matches!(option, Schema::Any) && !is_one_of {
                return Schema::Any;
            }
            match option {
                Schema::Unsatisfiable(_) => last_unsat = Some(option),
                // Flatten nested same-kind unions
                Schema::AnyOf(nested) if !is_one_of => valid.extend(nested),
                Schema::OneOf(nested) if is_one_of => valid.extend(nested),
                other => valid.push(other),
            }
        }

        if valid.is_empty() {
            return last_unsat.unwrap_or_else(|| {
                Schema::unsat(if is_one_of {
                    "oneOf is empty"
                } else {
                    "anyOf is empty"
                })
            });
        }
        if valid.len() == 1 {
            return valid.swap_remove(0);
        }

        // OneOf with all-disjoint options can be treated as AnyOf
        if is_one_of && self.all_pairwise_disjoint(&valid) {
            return Schema::AnyOf(valid);
        }

        if is_one_of {
            Schema::OneOf(valid)
        } else {
            Schema::AnyOf(valid)
        }
    }

    fn all_pairwise_disjoint(&mut self, items: &[Schema]) -> bool {
        items.iter().enumerate().all(|(i, x)| {
            items
                .iter()
                .skip(i + 1)
                .all(|y| self.is_verifiably_disjoint(x, y))
        })
    }

    fn is_verifiably_disjoint(&mut self, a: &Schema, b: &Schema) -> bool {
        self.is_disjoint_inner(a, b, 0)
    }

    fn is_disjoint_inner(&mut self, a: &Schema, b: &Schema, depth: usize) -> bool {
        if depth > 16 {
            return false;
        }
        match (a, b) {
            (Schema::Unsatisfiable(_), _) | (_, Schema::Unsatisfiable(_)) => true,
            (Schema::Any, _) | (_, Schema::Any) => false,
            (Schema::Ref(name), _) => match self.definitions.get(name).cloned() {
                Some(resolved) => self.is_disjoint_inner(&resolved, b, depth + 1),
                None => false,
            },
            (_, Schema::Ref(name)) => match self.definitions.get(name).cloned() {
                Some(resolved) => self.is_disjoint_inner(a, &resolved, depth + 1),
                None => false,
            },
            (Schema::AnyOf(opts), _) | (Schema::OneOf(opts), _) => {
                opts.iter()
                    .all(|opt| self.is_disjoint_inner(opt, b, depth + 1))
            }
            (_, Schema::AnyOf(opts)) | (_, Schema::OneOf(opts)) => {
                opts.iter()
                    .all(|opt| self.is_disjoint_inner(a, opt, depth + 1))
            }
            (Schema::Boolean(v1), Schema::Boolean(v2)) => v1.is_some() && v2.is_some() && v1 != v2,
            (
                Schema::String(StringSchema {
                    regex: Some(RegexAst::Literal(l1)),
                    ..
                }),
                Schema::String(StringSchema {
                    regex: Some(RegexAst::Literal(l2)),
                    ..
                }),
            ) => l1 != l2,
            (Schema::Object(o1), Schema::Object(o2)) => {
                o1.required.union(&o2.required).any(|key| {
                    let p1 = self
                        .pattern_cache
                        .property_schema(o1, key)
                        .unwrap_or(&Schema::Any);
                    let p2 = self
                        .pattern_cache
                        .property_schema(o2, key)
                        .unwrap_or(&Schema::Any);
                    self.is_disjoint_inner(p1, p2, depth + 1)
                })
            }
            _ => std::mem::discriminant(a) != std::mem::discriminant(b),
        }
    }
}

// ---- Free functions ----

fn intersect_numbers(n1: NumberSchema, n2: NumberSchema) -> NumberSchema {
    NumberSchema {
        minimum: opt_max(n1.minimum, n2.minimum),
        maximum: opt_min(n1.maximum, n2.maximum),
        exclusive_minimum: opt_max(n1.exclusive_minimum, n2.exclusive_minimum),
        exclusive_maximum: opt_min(n1.exclusive_maximum, n2.exclusive_maximum),
        integer: n1.integer || n2.integer,
        multiple_of: merge_opts(n1.multiple_of, n2.multiple_of, |a, b| a.lcm(&b)),
    }
}

fn intersect_strings(s1: StringSchema, s2: StringSchema) -> StringSchema {
    StringSchema {
        min_length: s1.min_length.max(s2.min_length),
        max_length: opt_min(s1.max_length, s2.max_length),
        regex: merge_opts(s1.regex, s2.regex, |a, b| RegexAst::And(vec![a, b])),
    }
}

fn compile_const(instance: &Value) -> Result<Schema> {
    match instance {
        Value::Null => Ok(Schema::Null),
        Value::Bool(b) => Ok(Schema::Boolean(Some(*b))),
        Value::Number(n) => {
            let value = n
                .as_f64()
                .ok_or_else(|| anyhow!("Expected f64 for numeric const, got {}", instance))?;
            Ok(Schema::Number(NumberSchema {
                minimum: Some(value),
                maximum: Some(value),
                exclusive_minimum: None,
                exclusive_maximum: None,
                integer: n.is_i64(),
                multiple_of: None,
            }))
        }
        Value::String(s) => Ok(Schema::String(StringSchema {
            min_length: 0,
            max_length: None,
            regex: Some(RegexAst::Literal(s.to_string())),
        })),
        Value::Array(items) => {
            let prefix_items = items
                .iter()
                .map(compile_const)
                .collect::<Result<Vec<Schema>>>()?;
            Ok(Schema::Array(ArraySchema {
                min_items: prefix_items.len(),
                max_items: Some(prefix_items.len()),
                prefix_items,
                items: Some(Box::new(Schema::false_schema())),
            }))
        }
        Value::Object(mapping) => {
            let properties = mapping
                .iter()
                .map(|(k, v)| Ok((k.clone(), compile_const(v)?)))
                .collect::<Result<IndexMap<String, Schema>>>()?;
            let required = properties.keys().cloned().collect();
            Ok(Schema::Object(ObjectSchema {
                properties,
                pattern_properties: IndexMap::default(),
                additional_properties: Some(Box::new(Schema::false_schema())),
                required,
                min_properties: 0,
                max_properties: None,
            }))
        }
    }
}

fn mk_object_schema(obj: ObjectSchema) -> Schema {
    if let Some(max) = obj.max_properties {
        if obj.min_properties > max {
            return Schema::unsat("minProperties > maxProperties");
        }
    }
    if obj.required.len() > obj.max_properties.unwrap_or(usize::MAX) {
        return Schema::unsat("required > maxProperties");
    }
    Schema::Object(obj)
}

fn merge_opts<T>(a: Option<T>, b: Option<T>, f: impl FnOnce(T, T) -> T) -> Option<T> {
    match (a, b) {
        (None, None) => None,
        (None, Some(v)) | (Some(v), None) => Some(v),
        (Some(a), Some(b)) => Some(f(a, b)),
    }
}

fn opt_max<T: PartialOrd>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a >= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn opt_min<T: PartialOrd>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a <= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn compile(schema: Value) -> Result<BuiltSchema> {
        let opts = JsonCompileOptions::default();
        SchemaCompiler::compile(schema, &opts)
    }

    fn compile_lenient(schema: Value) -> Result<BuiltSchema> {
        let opts = JsonCompileOptions {
            lenient: true,
            ..Default::default()
        };
        SchemaCompiler::compile(schema, &opts)
    }

    #[test]
    fn test_bool_schema_true() {
        let r = compile(json!(true)).unwrap();
        assert!(matches!(r.schema, Schema::Any));
    }

    #[test]
    fn test_bool_schema_false() {
        let r = compile(json!(false)).unwrap();
        assert!(matches!(r.schema, Schema::Unsatisfiable(_)));
    }

    #[test]
    fn test_null_type() {
        let r = compile(json!({"type": "null"})).unwrap();
        assert!(matches!(r.schema, Schema::Null));
    }

    #[test]
    fn test_boolean_type() {
        let r = compile(json!({"type": "boolean"})).unwrap();
        assert!(matches!(r.schema, Schema::Boolean(None)));
    }

    #[test]
    fn test_number_type() {
        let r = compile(json!({"type": "number"})).unwrap();
        match &r.schema {
            Schema::Number(n) => {
                assert!(!n.integer);
                assert!(n.minimum.is_none());
            }
            other => panic!("Expected Number, got {:?}", other),
        }
    }

    #[test]
    fn test_integer_type() {
        let r = compile(json!({"type": "integer", "minimum": 0, "maximum": 100})).unwrap();
        match &r.schema {
            Schema::Number(n) => {
                assert!(n.integer);
                assert_eq!(n.minimum, Some(0.0));
                assert_eq!(n.maximum, Some(100.0));
            }
            other => panic!("Expected Number, got {:?}", other),
        }
    }

    #[test]
    fn test_string_type_with_constraints() {
        let r = compile(json!({"type": "string", "minLength": 1, "maxLength": 10})).unwrap();
        match &r.schema {
            Schema::String(s) => {
                assert_eq!(s.min_length, 1);
                assert_eq!(s.max_length, Some(10));
            }
            other => panic!("Expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_string_with_pattern() {
        let r = compile(json!({"type": "string", "pattern": "^[a-z]+$"})).unwrap();
        match &r.schema {
            Schema::String(s) => assert!(s.regex.is_some()),
            other => panic!("Expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_array_type() {
        let r = compile(json!({"type": "array", "minItems": 1, "maxItems": 5})).unwrap();
        match &r.schema {
            Schema::Array(a) => {
                assert_eq!(a.min_items, 1);
                assert_eq!(a.max_items, Some(5));
                assert!(a.prefix_items.is_empty());
            }
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_array_with_items() {
        let r = compile(json!({
            "type": "array",
            "items": {"type": "string"}
        }))
        .unwrap();
        match &r.schema {
            Schema::Array(a) => {
                assert!(a.items.is_some());
                assert!(matches!(a.items.as_deref(), Some(Schema::String(_))));
            }
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_array_with_prefix_items() {
        let r = compile(json!({
            "type": "array",
            "prefixItems": [
                {"type": "string"},
                {"type": "number"}
            ]
        }))
        .unwrap();
        match &r.schema {
            Schema::Array(a) => {
                assert_eq!(a.prefix_items.len(), 2);
                assert!(matches!(&a.prefix_items[0], Schema::String(_)));
                assert!(matches!(&a.prefix_items[1], Schema::Number(_)));
            }
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_object_type() {
        let r = compile(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }))
        .unwrap();
        match &r.schema {
            Schema::Object(o) => {
                assert_eq!(o.properties.len(), 2);
                assert!(o.required.contains("name"));
                assert!(!o.required.contains("age"));
            }
            other => panic!("Expected Object, got {:?}", other),
        }
    }

    #[test]
    fn test_object_with_additional_properties() {
        let r = compile(json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": false
        }))
        .unwrap();
        match &r.schema {
            Schema::Object(o) => {
                assert!(o.additional_properties.is_some());
                assert!(matches!(
                    o.additional_properties.as_deref(),
                    Some(Schema::Unsatisfiable(_))
                ));
            }
            other => panic!("Expected Object, got {:?}", other),
        }
    }

    #[test]
    fn test_const_string() {
        let r = compile(json!({"const": "hello"})).unwrap();
        match &r.schema {
            Schema::String(s) => {
                assert!(matches!(&s.regex, Some(RegexAst::Literal(l)) if l == "hello"));
            }
            other => panic!("Expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_enum_values() {
        let r = compile(json!({"enum": ["a", "b", "c"]})).unwrap();
        match &r.schema {
            Schema::AnyOf(options) => {
                assert_eq!(options.len(), 3);
            }
            _ => panic!("Expected AnyOf schema"),
        }
    }

    #[test]
    fn test_all_of_basic() {
        let r = compile(json!({
            "allOf": [
                {"type": "number", "minimum": 0},
                {"type": "number", "maximum": 100}
            ]
        }))
        .unwrap();
        match &r.schema {
            Schema::Number(n) => {
                assert_eq!(n.minimum, Some(0.0));
                assert_eq!(n.maximum, Some(100.0));
            }
            other => panic!("Expected Number, got {:?}", other),
        }
    }

    #[test]
    fn test_any_of() {
        let r = compile(json!({
            "anyOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        }))
        .unwrap();
        match &r.schema {
            Schema::AnyOf(options) => {
                assert_eq!(options.len(), 2);
            }
            _ => panic!("Expected AnyOf schema, got {:?}", r.schema),
        }
    }

    #[test]
    fn test_ref_basic() {
        let r = compile(json!({
            "$ref": "#/$defs/Name",
            "$defs": {
                "Name": {"type": "string", "minLength": 1}
            }
        }))
        .unwrap();
        assert!(matches!(r.schema, Schema::Ref(_)));
        let def = r
            .definitions
            .values()
            .next()
            .expect("should have one definition");
        assert!(matches!(def, Schema::String(_)));
    }

    #[test]
    fn test_incompatible_types_unsatisfiable() {
        let r = compile(json!({
            "allOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        }))
        .unwrap();
        assert!(matches!(r.schema, Schema::Unsatisfiable(_)));
    }

    #[test]
    fn test_all_of_objects_property_ordering() {
        let r = compile(json!({
            "type": "object",
            "properties": {"a": true, "b": true},
            "allOf": [
                {"properties": {"c": true, "d": true}}
            ]
        }))
        .unwrap();
        match &r.schema {
            Schema::Object(o) => {
                let keys: Vec<&str> = o.properties.keys().map(|k| k.as_str()).collect();
                assert_eq!(keys, vec!["a", "b", "c", "d"]);
            }
            other => panic!("Expected Object, got {:?}", other),
        }
    }

    #[test]
    fn test_pattern_properties_intersects_named_properties() {
        let r = compile(json!({
            "type": "object",
            "properties": {"foo": {"type": "string"}},
            "patternProperties": {"^f": {"minLength": 5}}
        }))
        .unwrap();
        match &r.schema {
            Schema::Object(o) => {
                let foo = o.properties.get("foo").unwrap();
                match foo {
                    Schema::String(s) => assert_eq!(s.min_length, 5),
                    other => panic!("Expected String for 'foo', got {:?}", other),
                }
                assert!(o.pattern_properties.contains_key("^f"));
            }
            other => panic!("Expected Object, got {:?}", other),
        }
    }

    #[test]
    fn test_unimplemented_key_strict() {
        let r = compile(json!({
            "type": "object",
            "if": {"properties": {"x": true}},
            "then": {"required": ["x"]}
        }));
        assert!(r.is_err());
    }

    #[test]
    fn test_unimplemented_key_lenient() {
        let r = compile_lenient(json!({
            "type": "object",
            "if": {"properties": {"x": true}},
            "then": {"required": ["x"]}
        }))
        .unwrap();
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn test_one_of_disjoint_becomes_any_of() {
        let r = compile(json!({
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"}
            ]
        }))
        .unwrap();
        assert!(
            matches!(&r.schema, Schema::AnyOf(_)),
            "Expected AnyOf for disjoint oneOf, got {:?}",
            r.schema
        );
    }

    #[test]
    fn test_empty_schema_is_any() {
        let r = compile(json!({})).unwrap();
        assert!(
            matches!(r.schema, Schema::Any),
            "Expected Any for empty schema, got {:?}",
            r.schema
        );
    }

    #[test]
    fn test_recursive_ref_without_siblings() {
        let r = compile(json!({
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "child": {"$ref": "#"}
            }
        }))
        .unwrap();
        assert!(matches!(r.schema, Schema::Object(_)));
    }

    #[test]
    fn test_recursive_ref_with_siblings() {
        let r = compile(json!({
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "child": {
                    "$ref": "#",
                    "required": ["value"]
                }
            }
        }))
        .unwrap();
        match &r.schema {
            Schema::Object(obj) => {
                let child = obj.properties.get("child").unwrap();
                match child {
                    Schema::Object(child_obj) => {
                        assert!(
                            child_obj.required.contains("value"),
                            "Expected 'value' in required, got {:?}",
                            child_obj.required
                        );
                    }
                    _ => panic!("Expected Object for child, got {:?}", child),
                }
            }
            _ => panic!("Expected Object, got {:?}", r.schema),
        }
    }

    #[test]
    fn test_problem_child() {
        let schema = json!({
            "allOf" : [
                {"$ref": "#/$defs/tree1"},
                {"$ref": "#/$defs/tree2"}
            ],
            "$defs" : {
                "tree1": {
                    "type": "object",
                    "properties": {
                        "child": {
                            "$ref": "#/$defs/tree1"
                        }
                    }
                },
                "tree2": {
                    "type": "object",
                    "properties": {
                        "child": {
                            "$ref": "#/$defs/tree2"
                        }
                    }
                }
            }
        });
        // Test failure amounts to this resulting in a stack overflow
        let opts = JsonCompileOptions::default();
        let _ = SchemaCompiler::compile(schema, &opts);
    }
}
