//! Built-in JSON Schema meta-schemas.
//!
//! This module provides access to the official JSON Schema meta-schemas for different draft versions.
use std::sync::Arc;

use once_cell::sync::Lazy;
use serde_json::Value;

macro_rules! schema {
    ($vis:vis $name:ident, $path:expr) => {
        $vis static $name: once_cell::sync::Lazy<Arc<serde_json::Value>> =
            once_cell::sync::Lazy::new(|| {
                Arc::new(serde_json::from_slice(include_bytes!($path)).expect("Invalid schema"))
            });
    };
    ($name:ident, $path:expr) => {
        schema!(pub(crate) $name, $path);
    };
}

schema!(pub DRAFT4, "../metaschemas/draft4.json");
schema!(pub DRAFT6, "../metaschemas/draft6.json");
schema!(pub DRAFT7, "../metaschemas/draft7.json");
schema!(pub DRAFT201909, "../metaschemas/draft2019-09/schema.json");
schema!(
    pub DRAFT201909_APPLICATOR,
    "../metaschemas/draft2019-09/meta/applicator.json"
);
schema!(
    pub DRAFT201909_CONTENT,
    "../metaschemas/draft2019-09/meta/content.json"
);
schema!(
    pub DRAFT201909_CORE,
    "../metaschemas/draft2019-09/meta/core.json"
);
schema!(
    pub DRAFT201909_FORMAT,
    "../metaschemas/draft2019-09/meta/format.json"
);
schema!(
    pub DRAFT201909_META_DATA,
    "../metaschemas/draft2019-09/meta/meta-data.json"
);
schema!(
    pub DRAFT201909_VALIDATION,
    "../metaschemas/draft2019-09/meta/validation.json"
);
schema!(pub DRAFT202012, "../metaschemas/draft2020-12/schema.json");
schema!(
    pub DRAFT202012_CORE,
    "../metaschemas/draft2020-12/meta/core.json"
);
schema!(
    pub DRAFT202012_APPLICATOR,
    "../metaschemas/draft2020-12/meta/applicator.json"
);
schema!(
    pub DRAFT202012_UNEVALUATED,
    "../metaschemas/draft2020-12/meta/unevaluated.json"
);
schema!(
    pub DRAFT202012_VALIDATION,
    "../metaschemas/draft2020-12/meta/validation.json"
);
schema!(
    pub DRAFT202012_META_DATA,
    "../metaschemas/draft2020-12/meta/meta-data.json"
);
schema!(
    pub DRAFT202012_FORMAT_ANNOTATION,
    "../metaschemas/draft2020-12/meta/format-annotation.json"
);
schema!(
    pub DRAFT202012_FORMAT_ASSERTION,
    "../metaschemas/draft2020-12/meta/format-assertion.json"
);
schema!(
    pub DRAFT202012_CONTENT,
    "../metaschemas/draft2020-12/meta/content.json"
);
pub(crate) static META_SCHEMAS: Lazy<[(&'static str, &Value); 18]> = Lazy::new(|| {
    [
        ("http://json-schema.org/draft-04/schema#", &*DRAFT4),
        ("http://json-schema.org/draft-06/schema#", &*DRAFT6),
        ("http://json-schema.org/draft-07/schema#", &*DRAFT7),
        (
            "https://json-schema.org/draft/2019-09/schema",
            &*DRAFT201909,
        ),
        (
            "https://json-schema.org/draft/2019-09/meta/applicator",
            &*DRAFT201909_APPLICATOR,
        ),
        (
            "https://json-schema.org/draft/2019-09/meta/content",
            &*DRAFT201909_CONTENT,
        ),
        (
            "https://json-schema.org/draft/2019-09/meta/core",
            &*DRAFT201909_CORE,
        ),
        (
            "https://json-schema.org/draft/2019-09/meta/format",
            &*DRAFT201909_FORMAT,
        ),
        (
            "https://json-schema.org/draft/2019-09/meta/meta-data",
            &*DRAFT201909_META_DATA,
        ),
        (
            "https://json-schema.org/draft/2019-09/meta/validation",
            &*DRAFT201909_VALIDATION,
        ),
        (
            "https://json-schema.org/draft/2020-12/schema",
            &*DRAFT202012,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/core",
            &*DRAFT202012_CORE,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/applicator",
            &*DRAFT202012_APPLICATOR,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/unevaluated",
            &*DRAFT202012_UNEVALUATED,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/validation",
            &*DRAFT202012_VALIDATION,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/meta-data",
            &*DRAFT202012_META_DATA,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/format-annotation",
            &*DRAFT202012_FORMAT_ANNOTATION,
        ),
        (
            "https://json-schema.org/draft/2020-12/meta/content",
            &*DRAFT202012_CONTENT,
        ),
    ]
});
