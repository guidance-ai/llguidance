// Regression tests for schemas whose compile cost doubled (or tripled) with each nesting
// level, because a sub-schema was deep-copied once per use: into each anyOf/oneOf
// alternative, for each $ref beside sibling keywords, and for the additionalProperties,
// patternProperties and items fallbacks. The `*_grows_linearly` and `*_stays_small`
// tests check the grammar size; the `*_language` tests check that the language is kept.

use llg_test_utils::{json_err_test, json_test_many};
use llguidance::api::{GrammarInit, ParserLimits, TopLevelGrammar};
use serde_json::{json, Value};

/// Applies `level` `n` times, starting from `{"type": "string"}`.
fn nest(n: usize, level: impl Fn(Value) -> Value) -> Value {
    let mut s = json!({"type": "string"});
    for _ in 0..n {
        s = level(s);
    }
    s
}

fn num_symbols(schema: Value) -> usize {
    let (grammar, _) = GrammarInit::Serialized(TopLevelGrammar::from_json_schema(schema))
        .to_internal(None, ParserLimits::default())
        .unwrap();
    grammar.num_symbols()
}

/// Fails when the grammar of `shape(large)` has `max_ratio` times the symbols of
/// `shape(small)`, or more. Linear growth gives a ratio near 1.
fn assert_grows_linearly(shape: fn(usize) -> Value, small: usize, large: usize, max_ratio: f64) {
    let s_small = num_symbols(shape(small));
    let s_large = num_symbols(shape(large));
    eprintln!("symbols: {small} levels = {s_small}, {large} levels = {s_large}");
    assert!(
        (s_large as f64) < (s_small as f64) * max_ratio,
        "symbols: {small} levels = {s_small}, {large} levels = {s_large}"
    );
}

/// {type: object, properties: {p: inner}, required: [p], anyOf: [{required: [p]}, {required: [p]}]}
fn main_shape(depth: usize) -> Value {
    nest(depth, |s| {
        json!({"type": "object", "properties": {"p": s}, "required": ["p"],
               "anyOf": [{"required": ["p"]}, {"required": ["p"]}]})
    })
}

/// The main shape with `anyOf: [{}, {}]`, which allows everything.
fn empty_alternatives_shape(depth: usize) -> Value {
    nest(depth, |s| {
        json!({"type": "object", "properties": {"p": s}, "required": ["p"],
               "anyOf": [{}, {}]})
    })
}

/// The main shape without the `anyOf`.
fn properties_only_shape(depth: usize) -> Value {
    nest(
        depth,
        |s| json!({"type": "object", "properties": {"p": s}, "required": ["p"]}),
    )
}

/// D_i = {type: object, properties: {p0: {$ref: D_{i+1}, anyOf: [{type: object}, {type: object}]}}}
fn ref_shape(levels: usize) -> Value {
    let mut defs = serde_json::Map::new();
    for i in 0..levels {
        defs.insert(
            format!("D{i}"),
            json!({"type": "object", "properties": {"p0": {"$ref": format!("#/$defs/D{}", i + 1),
                   "anyOf": [{"type": "object"}, {"type": "object"}]}}}),
        );
    }
    defs.insert(format!("D{levels}"), json!({"type": "string"}));
    json!({"$ref": "#/$defs/D0", "$defs": defs})
}

/// No anyOf: D_i = {type: object, properties: {a: {$ref: D_{i+1}, type: object}, b: <same>}}.
/// Each $ref beside a sibling keyword copies the definition.
fn ref_pair_shape(levels: usize) -> Value {
    let mut defs = serde_json::Map::new();
    for i in 0..levels {
        let r = format!("#/$defs/D{}", i + 1);
        defs.insert(
            format!("D{i}"),
            json!({"type": "object", "properties": {
                "a": {"$ref": r.clone(), "type": "object"},
                "b": {"$ref": r, "type": "object"}}}),
        );
    }
    defs.insert(format!("D{levels}"), json!({"type": "object"}));
    json!({"$ref": "#/$defs/D0", "$defs": defs})
}

/// The nested level is the `additionalProperties` of an anyOf alternative.
fn anyof_additional_properties_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "object", "properties": {"a": {}, "b": {}},
               "anyOf": [{"type": "object", "additionalProperties": s}, {"type": "null"}]})
    })
}

/// The nested level is the `additionalProperties` of an allOf subschema.
fn allof_additional_properties_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "object", "properties": {"a": {}, "b": {}},
               "allOf": [{"type": "object", "additionalProperties": s}]})
    })
}

/// The nested level is the `items` of an allOf subschema, beside `prefixItems`.
fn prefix_items_beside_items_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "array", "prefixItems": [{}, {}],
               "allOf": [{"type": "array", "items": s}]})
    })
}

/// No applicator: the nested level is a pattern schema that matches two named properties.
fn pattern_properties_beside_properties_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "object", "properties": {"pa": {}, "pb": {}},
               "patternProperties": {"^p": s}})
    })
}

// The four shapes above with a typed sibling instead of `{}`. The shared nested level then
// meets a schema other than `Any`, so it is unwrapped; that copy must be one level deep.

fn anyof_additional_properties_typed_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "object", "properties": {"a": {"type": "object"}, "b": {"type": "object"}},
               "anyOf": [{"type": "object", "additionalProperties": s}, {"type": "null"}]})
    })
}

fn allof_additional_properties_typed_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "object", "properties": {"a": {"type": "object"}, "b": {"type": "object"}},
               "allOf": [{"type": "object", "additionalProperties": s}]})
    })
}

fn prefix_items_beside_items_typed_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "array", "prefixItems": [{"type": "array"}, {"type": "array"}],
               "allOf": [{"type": "array", "items": s}]})
    })
}

fn pattern_properties_beside_properties_typed_shape(n: usize) -> Value {
    nest(n, |s| {
        json!({"type": "object",
               "properties": {"pa": {"type": "object"}, "pb": {"type": "object"}},
               "patternProperties": {"^p": s}})
    })
}

// ---- the grammar must grow linearly with the nesting depth ----

#[test]
fn anyof_beside_properties_grammar_grows_linearly() {
    // without sharing: 22533 -> 90117 symbols
    assert_grows_linearly(main_shape, 10, 12, 1.5);
}

#[test]
fn anyof_beside_properties_depth_16_stays_small() {
    // without sharing, this hits the 500000-symbol grammar limit
    let n = num_symbols(main_shape(16));
    assert!(n < 1_000, "symbols: {n}");
}

#[test]
fn empty_alternatives_stays_small() {
    // `anyOf: [{}, {}]` allows everything, so it must not add to the grammar
    let with_anyof = num_symbols(empty_alternatives_shape(12));
    let without_anyof = num_symbols(properties_only_shape(12));
    assert_eq!(with_anyof, without_anyof);
}

#[test]
fn anyof_beside_ref_grammar_grows_linearly() {
    // without sharing: 10254 -> 40974 symbols
    assert_grows_linearly(ref_shape, 10, 12, 1.5);
}

#[test]
fn ref_beside_type_grammar_grows_linearly() {
    // without sharing: 24586 -> 98314 symbols
    assert_grows_linearly(ref_pair_shape, 10, 12, 1.5);
}

#[test]
fn anyof_additional_properties_grows_linearly() {
    assert_grows_linearly(anyof_additional_properties_shape, 4, 6, 2.0);
}

#[test]
fn allof_additional_properties_grows_linearly() {
    assert_grows_linearly(allof_additional_properties_shape, 4, 6, 2.0);
}

#[test]
fn prefix_items_beside_items_grows_linearly() {
    assert_grows_linearly(prefix_items_beside_items_shape, 4, 6, 2.0);
}

#[test]
fn pattern_properties_beside_properties_grows_linearly() {
    assert_grows_linearly(pattern_properties_beside_properties_shape, 4, 6, 2.0);
}

#[test]
fn anyof_additional_properties_typed_grows_linearly() {
    assert_grows_linearly(anyof_additional_properties_typed_shape, 4, 6, 2.0);
}

#[test]
fn allof_additional_properties_typed_grows_linearly() {
    assert_grows_linearly(allof_additional_properties_typed_shape, 4, 6, 2.0);
}

#[test]
fn prefix_items_beside_items_typed_grows_linearly() {
    assert_grows_linearly(prefix_items_beside_items_typed_shape, 4, 6, 2.0);
}

#[test]
fn pattern_properties_beside_properties_typed_grows_linearly() {
    assert_grows_linearly(pattern_properties_beside_properties_typed_shape, 4, 6, 2.0);
}

// ---- the language must not change ----

#[test]
fn anyof_beside_properties_language() {
    json_test_many(
        &main_shape(3),
        &[
            json!({"p": {"p": {"p": "x"}}}),
            json!({"p": {"p": {"p": "x"}, "z": 1}}),
        ],
        &[
            json!({}),
            json!({"p": "x"}),
            json!({"p": {"p": {}}}),
            json!({"p": {"p": {"p": 1}}}),
        ],
    );
}

#[test]
fn anyof_beside_ref_language() {
    // D3 is a string, so D2.p0 = D3 ∩ object is unsatisfiable and p0 is not allowed in D2
    json_test_many(
        &ref_shape(3),
        &[json!({}), json!({"p0": {}}), json!({"p0": {"p0": {}}})],
        &[json!({"p0": 1}), json!({"p0": {"p0": {"p0": "s"}}})],
    );
}

#[test]
fn ref_beside_type_language() {
    json_test_many(
        &ref_pair_shape(2),
        &[json!({}), json!({"a": {"a": {}, "b": {"z": 1}}, "b": {}})],
        &[json!({"a": 1}), json!({"a": {"b": []}})],
    );
}

#[test]
fn anyof_additional_properties_language() {
    json_test_many(
        &anyof_additional_properties_shape(2),
        &[
            json!({}),
            json!({"a": {}}),
            json!({"a": {"a": "x"}}),
            json!({"z": {"b": "y", "c": "w"}}),
        ],
        &[
            json!(null),
            json!({"a": "x"}),
            json!({"a": {"a": 1}}),
            json!({"a": {"a": "x"}, "b": 3}),
        ],
    );
}

#[test]
fn allof_additional_properties_language() {
    json_test_many(
        &allof_additional_properties_shape(2),
        &[json!({}), json!({"b": {"a": "x", "q": "y"}})],
        &[json!({"b": "x"}), json!({"a": {"a": null}})],
    );
}

#[test]
fn prefix_items_beside_items_language() {
    json_test_many(
        &prefix_items_beside_items_shape(2),
        &[json!([]), json!([[]]), json!([["a"], ["b", "c"], ["d"]])],
        &[json!(["a"]), json!([["a", 1]]), json!([[1]])],
    );
}

#[test]
fn pattern_properties_beside_properties_language() {
    json_test_many(
        &pattern_properties_beside_properties_shape(2),
        &[
            json!({"q": 5}),
            json!({"pa": {"pa": "x", "q": 1}}),
            json!({"pz": {"pb": "s"}}),
        ],
        &[
            json!({"pa": "x"}),
            json!({"pa": {"pa": 1}}),
            json!({"pq": {"px": true}}),
        ],
    );
}

#[test]
fn pattern_properties_beside_properties_typed_language() {
    // inner level: pa and pb are object ∩ string, so they are unsatisfiable and not allowed
    json_test_many(
        &pattern_properties_beside_properties_typed_shape(2),
        &[
            json!({"q": 5}),
            json!({"pa": {"pz": "x", "q": 1}}),
            json!({"pb": {}, "pz": {"pq": "s"}}),
        ],
        &[
            json!({"pa": "x"}),
            json!({"pa": {"pb": {}}}),
            json!({"pz": {"pa": "s"}}),
            json!({"pb": {"pz": {}}}),
        ],
    );
}

#[test]
fn anyof_additional_properties_typed_language() {
    // inner level: a and b are object ∩ string, so they are unsatisfiable and not allowed
    json_test_many(
        &anyof_additional_properties_typed_shape(2),
        &[
            json!({}),
            json!({"a": {}}),
            json!({"a": {"z": "x"}}),
            json!({"b": {}, "z": {"q": "y"}}),
        ],
        &[
            json!(null),
            json!({"a": "x"}),
            json!({"a": {"a": {}}}),
            json!({"a": {"z": 1}}),
            json!({"z": {"b": {}}}),
        ],
    );
}

#[test]
fn allof_additional_properties_typed_language() {
    // inner level: a and b are object ∩ string, so they are unsatisfiable and not allowed
    json_test_many(
        &allof_additional_properties_typed_shape(2),
        &[
            json!({}),
            json!({"b": {}}),
            json!({"b": {"q": "x"}}),
            json!({"a": {}, "z": {"q": "y"}}),
        ],
        &[
            json!([]),
            json!({"b": "x"}),
            json!({"b": {"b": {}}}),
            json!({"a": {"q": null}}),
            json!({"z": {"a": {}}}),
        ],
    );
}

#[test]
fn prefix_items_beside_items_typed_language() {
    // inner level: the prefix items are array ∩ string, so the inner array must be empty
    json_test_many(
        &prefix_items_beside_items_typed_shape(2),
        &[json!([]), json!([[]]), json!([[], [], []])],
        &[json!({}), json!(["a"]), json!([["a"]]), json!([[], [[]]])],
    );
}

/// D has a nested object property and an `anyOf`. `allOf` of the same `$ref` twice
/// intersects D with itself, so the shared property meets the same `Rc` on both sides.
#[test]
fn allof_same_ref_with_anyof_language() {
    let schema = json!({
        "$defs": {"D": {
            "type": "object",
            "properties": {"p": {"type": "object",
                                 "properties": {"q": {"type": "string"}},
                                 "required": ["q"]}},
            "required": ["p"],
            "anyOf": [{"required": ["a"]}, {"required": ["b"]}]
        }},
        "allOf": [{"$ref": "#/$defs/D"}, {"$ref": "#/$defs/D"}]
    });
    json_test_many(
        &schema,
        &[
            json!({"p": {"q": "x"}, "a": 1}),
            json!({"p": {"q": "x", "r": 2}, "b": null}),
        ],
        &[
            json!({"p": {"q": "x"}}),
            json!({"a": 1}),
            json!({"p": {}, "a": 1}),
            json!({"p": {"q": 1}, "b": 1}),
        ],
    );
}

/// The same as above, with a `oneOf` inside the shared property.
#[test]
fn allof_same_ref_with_oneof_language() {
    let schema = json!({
        "$defs": {"D": {
            "type": "object",
            "properties": {"p": {"type": "object",
                                 "properties": {"v": {"oneOf": [{"type": "string"},
                                                                {"type": "integer"}]}},
                                 "required": ["v"]}},
            "required": ["p"]
        }},
        "allOf": [{"$ref": "#/$defs/D"}, {"$ref": "#/$defs/D"}]
    });
    json_test_many(
        &schema,
        &[json!({"p": {"v": "x"}}), json!({"p": {"v": 3}, "z": 1})],
        &[
            json!({}),
            json!({"p": {}}),
            json!({"p": {"v": 1.5}}),
            json!({"p": {"v": null}}),
        ],
    );
}

#[test]
fn oneof_discriminator_beside_shared_properties() {
    // `data` is shared between the oneOf alternatives; `kind` decides disjointness
    let schema = json!({
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "data": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        },
        "required": ["kind", "data"],
        "oneOf": [{"properties": {"kind": {"const": "a"}}}, {"properties": {"kind": {"const": "b"}}}]
    });
    json_test_many(
        &schema,
        &[
            json!({"kind": "a", "data": {"x": 1}}),
            json!({"kind": "b", "data": {"x": 2}}),
        ],
        &[
            json!({"kind": "c", "data": {"x": 1}}),
            json!({"kind": "a", "data": {}}),
        ],
    );
}

fn unsat_child(required: bool) -> Value {
    // p's schema is unsatisfiable, but only the grammar builder finds out (minLength > maxLength)
    json!({
        "type": "object",
        "properties": {"p": {"type": "object",
                             "properties": {"q": {"type": "string", "minLength": 3, "maxLength": 1}},
                             "required": ["q"]}},
        "required": if required { json!(["p"]) } else { json!([]) },
        "anyOf": [{"required": ["a"]}, {"required": ["b"]}]
    })
}

#[test]
fn shared_unsatisfiable_optional_property_is_dropped() {
    json_test_many(
        &unsat_child(false),
        &[json!({"a": 1}), json!({"b": 1})],
        &[json!({}), json!({"a": 1, "p": {"q": "x"}})],
    );
}

#[test]
fn shared_unsatisfiable_required_property_is_an_error() {
    json_err_test(
        &unsat_child(true),
        "Unsatisfiable schema: required property 'p' is unsatisfiable",
    );
}
