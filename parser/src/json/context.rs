//! JSON Schema `$ref` resolution and draft detection.
//!
//! Provides `Draft` (schema version detection + keyword recognition) and `RefResolver`
//! (URI resolution for `$ref`). The `RefResolver` implementation is feature-gated:
//! with `referencing`, it uses a full URI registry; without, it only handles local
//! `#/...` JSON Pointer refs.

#[cfg(not(feature = "referencing"))]
use anyhow::anyhow;
use anyhow::Result;
use serde_json::Value;

use super::RetrieveWrapper;

const DEFAULT_DRAFT: Draft = Draft::Draft202012;

// ---- Draft ----

// When the `referencing` feature is enabled, use the Draft type from the
// referencing crate (which includes detect(), is_known_keyword(), etc.).
// Otherwise, use our local copy (originally derived from the same crate).

#[cfg(feature = "referencing")]
pub use referencing::Draft;

// Based on https://github.com/Stranger6667/jsonschema/blob/b025051b3c4dd4df4d30b83d23e6c8a13717b62b/crates/jsonschema-referencing/src/specification/mod.rs
// MIT License — Copyright (c) 2020-2024 Dmitry Dygalo
#[cfg(not(feature = "referencing"))]
mod draft_impl {
    use anyhow::Error;
    use serde_json::Value;

    /// JSON Schema specification versions.
    #[non_exhaustive]
    #[derive(Debug, Default, PartialEq, Copy, Clone, Hash, Eq, PartialOrd, Ord)]
    pub enum Draft {
        /// JSON Schema Draft 4
        Draft4,
        /// JSON Schema Draft 6
        Draft6,
        /// JSON Schema Draft 7
        Draft7,
        /// JSON Schema Draft 2019-09
        Draft201909,
        /// JSON Schema Draft 2020-12
        #[default]
        Draft202012,
    }

    impl Draft {
        pub fn detect(self, contents: &Value) -> Result<Draft, Error> {
            if let Some(schema) = contents
                .as_object()
                .and_then(|contents| contents.get("$schema"))
                .and_then(|schema| schema.as_str())
            {
                Ok(match schema.trim_end_matches('#') {
                    "https://json-schema.org/draft/2020-12/schema" => Draft::Draft202012,
                    "https://json-schema.org/draft/2019-09/schema" => Draft::Draft201909,
                    "http://json-schema.org/draft-07/schema" => Draft::Draft7,
                    "http://json-schema.org/draft-06/schema" => Draft::Draft6,
                    "http://json-schema.org/draft-04/schema" => Draft::Draft4,
                    value => return Err(anyhow::anyhow!("Unknown specification: {}", value)),
                })
            } else {
                Ok(self)
            }
        }

        /// Identifies known JSON schema keywords per draft.
        #[must_use]
        pub fn is_known_keyword(&self, keyword: &str) -> bool {
            match keyword {
                "$ref"
                | "$schema"
                | "additionalItems"
                | "additionalProperties"
                | "allOf"
                | "anyOf"
                | "dependencies"
                | "enum"
                | "exclusiveMaximum"
                | "exclusiveMinimum"
                | "format"
                | "items"
                | "maxItems"
                | "maxLength"
                | "maxProperties"
                | "maximum"
                | "minItems"
                | "minLength"
                | "minProperties"
                | "minimum"
                | "multipleOf"
                | "not"
                | "oneOf"
                | "pattern"
                | "patternProperties"
                | "properties"
                | "required"
                | "type"
                | "uniqueItems" => true,

                "id" if *self == Draft::Draft4 => true,

                "$id" | "const" | "contains" | "propertyNames" if *self >= Draft::Draft6 => true,

                "contentEncoding" | "contentMediaType"
                    if matches!(self, Draft::Draft6 | Draft::Draft7) =>
                {
                    true
                }

                "else" | "if" | "then" if *self >= Draft::Draft7 => true,

                "$anchor"
                | "$defs"
                | "$recursiveAnchor"
                | "$recursiveRef"
                | "dependentRequired"
                | "dependentSchemas"
                | "maxContains"
                | "minContains"
                | "prefixItems"
                | "unevaluatedItems"
                | "unevaluatedProperties"
                    if *self >= Draft::Draft201909 =>
                {
                    true
                }

                "$dynamicAnchor" | "$dynamicRef" if *self == Draft::Draft202012 => true,

                _ => false,
            }
        }
    }
}

#[cfg(not(feature = "referencing"))]
pub use draft_impl::Draft;

// ---- RefResolver (with `referencing` feature) ----

#[cfg(feature = "referencing")]
const DEFAULT_ROOT_URI: &str = "json-schema:///";

#[cfg(feature = "referencing")]
/// Resolves `$ref` URIs using the `referencing` crate's `Registry` and `Resolver`.
///
/// Handles `$id` scoping, `$anchor` resolution, JSON Pointer escaping (RFC 6901),
/// percent-encoding (RFC 3986), and relative URI resolution.
pub struct RefResolver {
    registry: referencing::Registry,
    draft: Draft,
    base_uri: String,
}

/// A retriever that always fails — ensures no external resource fetching occurs
/// unless an explicit retriever is provided. This is important for security.
#[cfg(feature = "referencing")]
struct NoOpRetriever;

#[cfg(feature = "referencing")]
impl referencing::Retrieve for NoOpRetriever {
    fn retrieve(
        &self,
        uri: &referencing::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(format!(
            "No retriever configured; cannot fetch external resource '{}'",
            uri
        )
        .into())
    }
}

#[cfg(feature = "referencing")]
impl RefResolver {
    pub fn new(contents: &Value, retriever: Option<&RetrieveWrapper>) -> Result<Self> {
        let draft = DEFAULT_DRAFT.detect(contents).unwrap_or(DEFAULT_DRAFT);
        let resource = draft.create_resource(contents.clone());
        let base_uri = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();

        let no_op = NoOpRetriever;
        let retriever: &dyn referencing::Retrieve =
            if let Some(r) = retriever { r } else { &no_op };

        let registry = {
            let empty = referencing::Registry::try_from_resources(std::iter::empty::<(
                String,
                referencing::Resource,
            )>())?;
            empty.try_with_resources_and_retriever(vec![(&base_uri, resource)], retriever, draft)?
        };

        Ok(RefResolver {
            registry,
            draft,
            base_uri,
        })
    }

    pub fn draft(&self) -> Draft {
        self.draft
    }

    /// Normalize a `$ref` string to an absolute URI against the current base.
    pub fn normalize_ref(&self, reference: &str) -> Result<String> {
        let resolver = self.registry.try_resolver(&self.base_uri)?;
        Ok(resolver
            .resolve_against(&resolver.base_uri().borrow(), reference)?
            .normalize()
            .into_string())
    }

    /// Look up a `$ref` and return the referenced schema content as a `Value`.
    pub fn lookup(&self, reference: &str) -> Result<(Value, String)> {
        let resolver = self.registry.try_resolver(&self.base_uri)?;
        let resolved = resolver.lookup(reference)?;
        let new_base = resolved.resolver().base_uri().as_str().to_string();
        Ok((resolved.contents().clone(), new_base))
    }

    /// Look up a `$ref` relative to a specific base URI (for nested `$id` scoping).
    pub fn lookup_relative(&self, reference: &str, base_uri: &str) -> Result<(Value, String)> {
        let resolver = self.registry.try_resolver(base_uri)?;
        let resolved = resolver.lookup(reference)?;
        let new_base = resolved.resolver().base_uri().as_str().to_string();
        Ok((resolved.contents().clone(), new_base))
    }

    /// Normalize a `$ref` string relative to a specific base URI.
    pub fn normalize_ref_relative(&self, reference: &str, base_uri: &str) -> Result<String> {
        let resolver = self.registry.try_resolver(base_uri)?;
        Ok(resolver
            .resolve_against(&resolver.base_uri().borrow(), reference)?
            .normalize()
            .into_string())
    }
}

#[cfg(feature = "referencing")]
impl referencing::Retrieve for RetrieveWrapper {
    fn retrieve(
        &self,
        uri: &referencing::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let value = self.0.retrieve(uri.as_str())?;
        Ok(value)
    }
}

// ---- RefResolver (without `referencing` feature) ----

#[cfg(not(feature = "referencing"))]
use std::sync::Arc;

#[cfg(not(feature = "referencing"))]
/// Fallback ref resolver for when the `referencing` feature is not enabled.
/// Only supports local `#/...` JSON Pointer refs against the root document.
pub struct RefResolver {
    draft: Draft,
    root_doc: Arc<Value>,
    retriever: Option<RetrieveWrapper>,
}

#[cfg(not(feature = "referencing"))]
impl RefResolver {
    pub fn new(contents: &Value, retriever: Option<&RetrieveWrapper>) -> Result<Self> {
        let draft = DEFAULT_DRAFT.detect(contents).unwrap_or(DEFAULT_DRAFT);
        Ok(RefResolver {
            draft,
            root_doc: Arc::new(contents.clone()),
            retriever: retriever.cloned(),
        })
    }

    pub fn draft(&self) -> Draft {
        self.draft
    }

    /// Without `referencing`, normalization is a no-op.
    pub fn normalize_ref(&self, reference: &str) -> Result<String> {
        Ok(reference.to_string())
    }

    /// Normalize a `$ref` string relative to a specific base URI.
    /// Without `referencing`, this is a no-op.
    pub fn normalize_ref_relative(&self, reference: &str, _base_uri: &str) -> Result<String> {
        Ok(reference.to_string())
    }

    /// Look up a `$ref` and return the referenced schema content as a `Value`,
    /// along with the resolved base URI for nested `$ref` resolution.
    /// Only supports local `#/...` JSON Pointer refs.
    pub fn lookup(&self, reference: &str) -> Result<(Value, String)> {
        let value = self.lookup_value(reference)?;
        Ok((value, String::new()))
    }

    /// Look up a `$ref` relative to a specific base URI.
    /// Without `referencing`, the base_uri is ignored.
    pub fn lookup_relative(&self, reference: &str, _base_uri: &str) -> Result<(Value, String)> {
        self.lookup(reference)
    }

    fn lookup_value(&self, reference: &str) -> Result<Value> {
        if reference == "#" || reference == "#/" {
            return Ok(self.root_doc.as_ref().clone());
        }

        if reference.starts_with("#/") {
            let mut content = self.root_doc.as_ref();
            for segment in reference[2..].split('/') {
                // Decode JSON Pointer escapes (RFC 6901)
                let decoded = segment.replace("~1", "/").replace("~0", "~");
                // Decode percent-encoding
                let decoded = percent_decode(&decoded);

                if content.is_array() {
                    let index = decoded.parse::<usize>()?;
                    content = content
                        .get(index)
                        .ok_or_else(|| anyhow!("Reference segment '{}' not found.", segment))?;
                } else if let Some(next) = content.get(decoded.as_str()) {
                    content = next;
                } else {
                    return Err(anyhow!(
                        "Reference segment '{}' not found in '{}'.",
                        segment,
                        reference
                    ));
                }
            }
            return Ok(content.clone());
        }

        if let Some(retriever) = &self.retriever {
            let value = retriever
                .0
                .retrieve(reference)
                .map_err(|e| anyhow!("Failed to retrieve '{}': {}", reference, e))?;
            return Ok(value);
        }

        Err(anyhow!(
            "Only local $ref's (#/...) are supported without 'referencing' feature; ref '{}'",
            reference
        ))
    }
}

/// Simple percent-decoding for JSON Pointer segments (RFC 3986).
/// Collects decoded bytes before converting to UTF-8 to handle multi-byte sequences.
#[cfg(not(feature = "referencing"))]
fn percent_decode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut iter = s.bytes();
    while let Some(b) = iter.next() {
        if b == b'%' {
            let hi = iter.next();
            let lo = iter.next();
            if let (Some(hi), Some(lo)) = (hi, lo) {
                if let Ok(byte) = u8::from_str_radix(&format!("{}{}", hi as char, lo as char), 16) {
                    bytes.push(byte);
                    continue;
                }
                // Invalid hex digits — preserve all three bytes as-is
                bytes.push(b'%');
                bytes.push(hi);
                bytes.push(lo);
            } else {
                // Incomplete sequence — preserve what we consumed
                bytes.push(b'%');
                if let Some(hi) = hi {
                    bytes.push(hi);
                }
            }
        } else {
            bytes.push(b);
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}
