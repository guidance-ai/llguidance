use core::fmt;
use std::str::FromStr;

use super::{uri, Error};
use ahash::AHashSet;
use fluent_uri::Uri;
use serde_json::Value;

/// A JSON Schema vocabulary identifier, representing standard vocabularies (Core, Applicator, etc.)
/// or custom ones via URI.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Vocabulary {
    Core,
    Applicator,
    Unevaluated,
    Validation,
    Metadata,
    Format,
    FormatAnnotation,
    Content,
    Custom(Uri<String>),
}

impl FromStr for Vocabulary {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "https://json-schema.org/draft/2020-12/vocab/core"
            | "https://json-schema.org/draft/2019-09/vocab/core" => Ok(Vocabulary::Core),
            "https://json-schema.org/draft/2020-12/vocab/applicator"
            | "https://json-schema.org/draft/2019-09/vocab/applicator" => {
                Ok(Vocabulary::Applicator)
            }
            "https://json-schema.org/draft/2020-12/vocab/unevaluated" => {
                Ok(Vocabulary::Unevaluated)
            }
            "https://json-schema.org/draft/2020-12/vocab/validation"
            | "https://json-schema.org/draft/2019-09/vocab/validation" => {
                Ok(Vocabulary::Validation)
            }
            "https://json-schema.org/draft/2020-12/vocab/meta-data"
            | "https://json-schema.org/draft/2019-09/vocab/meta-data" => Ok(Vocabulary::Metadata),
            "https://json-schema.org/draft/2020-12/vocab/format"
            | "https://json-schema.org/draft/2019-09/vocab/format" => Ok(Vocabulary::Format),
            "https://json-schema.org/draft/2020-12/vocab/format-annotation" => {
                Ok(Vocabulary::FormatAnnotation)
            }
            "https://json-schema.org/draft/2020-12/vocab/content"
            | "https://json-schema.org/draft/2019-09/vocab/content" => Ok(Vocabulary::Content),
            _ => Ok(Vocabulary::Custom(uri::from_str(s)?)),
        }
    }
}

/// A set of enabled JSON Schema vocabularies.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct VocabularySet {
    known: u8,
    custom: AHashSet<Uri<String>>,
}

impl fmt::Debug for VocabularySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_list = f.debug_list();

        // Add known vocabularies
        if self.known & (1 << 0) != 0 {
            debug_list.entry(&"core");
        }
        if self.known & (1 << 1) != 0 {
            debug_list.entry(&"applicator");
        }
        if self.known & (1 << 2) != 0 {
            debug_list.entry(&"unevaluated");
        }
        if self.known & (1 << 3) != 0 {
            debug_list.entry(&"validation");
        }
        if self.known & (1 << 4) != 0 {
            debug_list.entry(&"meta-data");
        }
        if self.known & (1 << 5) != 0 {
            debug_list.entry(&"format");
        }
        if self.known & (1 << 6) != 0 {
            debug_list.entry(&"format-annotation");
        }
        if self.known & (1 << 7) != 0 {
            debug_list.entry(&"content");
        }

        // Add custom vocabularies
        if !self.custom.is_empty() {
            let mut custom: Vec<_> = self.custom.iter().map(Uri::as_str).collect();
            custom.sort_unstable();
            for uri in custom {
                debug_list.entry(&uri);
            }
        }
        debug_list.finish()
    }
}

impl VocabularySet {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn from_known(known: u8) -> Self {
        Self {
            known,
            custom: AHashSet::new(),
        }
    }

    pub(crate) fn add(&mut self, vocabulary: Vocabulary) {
        match vocabulary {
            Vocabulary::Core => self.known |= 1 << 0,
            Vocabulary::Applicator => self.known |= 1 << 1,
            Vocabulary::Unevaluated => self.known |= 1 << 2,
            Vocabulary::Validation => self.known |= 1 << 3,
            Vocabulary::Metadata => self.known |= 1 << 4,
            Vocabulary::Format => self.known |= 1 << 5,
            Vocabulary::FormatAnnotation => self.known |= 1 << 6,
            Vocabulary::Content => self.known |= 1 << 7,
            Vocabulary::Custom(uri) => {
                self.custom.insert(uri);
            }
        }
    }
    #[must_use]
    pub fn contains(&self, vocabulary: &Vocabulary) -> bool {
        match vocabulary {
            Vocabulary::Core => self.known & (1 << 0) != 0,
            Vocabulary::Applicator => self.known & (1 << 1) != 0,
            Vocabulary::Unevaluated => self.known & (1 << 2) != 0,
            Vocabulary::Validation => self.known & (1 << 3) != 0,
            Vocabulary::Metadata => self.known & (1 << 4) != 0,
            Vocabulary::Format => self.known & (1 << 5) != 0,
            Vocabulary::FormatAnnotation => self.known & (1 << 6) != 0,
            Vocabulary::Content => self.known & (1 << 7) != 0,
            Vocabulary::Custom(uri) => self.custom.contains(uri),
        }
    }
}

pub(crate) const DRAFT_2020_12_VOCABULARIES: u8 = 0b1111_1111;
pub(crate) const DRAFT_2019_09_VOCABULARIES: u8 = 0b1001_1011;

pub(crate) fn find(document: &Value) -> Result<Option<VocabularySet>, Error> {
    if let Some(schema) = document.get("$id").and_then(|s| s.as_str()) {
        match schema {
            "https://json-schema.org/schema" | "https://json-schema.org/draft/2020-12/schema" => {
                // All known vocabularies
                Ok(Some(VocabularySet::from_known(DRAFT_2020_12_VOCABULARIES)))
            }
            "https://json-schema.org/draft/2019-09/schema" => {
                // Core, Applicator, Validation, Metadata, Content
                Ok(Some(VocabularySet::from_known(DRAFT_2019_09_VOCABULARIES)))
            }
            "https://json-schema.org/draft-07/schema"
            | "https://json-schema.org/draft-06/schema"
            | "https://json-schema.org/draft-04/schema" => Ok(None),
            _ => {
                // For unknown schemas, parse the $vocabulary object
                if let Some(vocab_obj) = document.get("$vocabulary").and_then(|v| v.as_object()) {
                    let mut set = VocabularySet::new();
                    for (uri, enabled) in vocab_obj {
                        if enabled.as_bool().unwrap_or(false) {
                            set.add(Vocabulary::from_str(uri)?);
                        }
                    }
                    Ok(Some(set))
                } else {
                    Ok(None)
                }
            }
        }
    } else {
        Ok(None)
    }
}
