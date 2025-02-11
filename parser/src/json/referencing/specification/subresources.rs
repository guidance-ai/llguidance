use core::slice;
use std::iter::FlatMap;

use serde_json::Value;

use super::super::{resource::InnerResourcePtr, segments::Segment, Error, Resolver, Segments};

type ObjectIter<'a> = FlatMap<
    serde_json::map::Iter<'a>,
    SubresourceIteratorInner<'a>,
    fn((&'a std::string::String, &'a Value)) -> SubresourceIteratorInner<'a>,
>;

/// A simple iterator that either wraps an iterator producing &Value or is empty.
pub(crate) enum SubresourceIterator<'a> {
    Object(ObjectIter<'a>),
    Empty,
}

impl<'a> Iterator for SubresourceIterator<'a> {
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SubresourceIterator::Object(iter) => iter.next(),
            SubresourceIterator::Empty => None,
        }
    }
}

pub(crate) enum SubresourceIteratorInner<'a> {
    Once(&'a Value),
    Array(slice::Iter<'a, Value>),
    Object(serde_json::map::Values<'a>),
    FilteredObject(serde_json::map::Values<'a>),
    Empty,
}

impl<'a> Iterator for SubresourceIteratorInner<'a> {
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SubresourceIteratorInner::Once(_) => {
                let SubresourceIteratorInner::Once(value) =
                    std::mem::replace(self, SubresourceIteratorInner::Empty)
                else {
                    unreachable!()
                };
                Some(value)
            }
            SubresourceIteratorInner::Array(iter) => iter.next(),
            SubresourceIteratorInner::Object(iter) => iter.next(),
            SubresourceIteratorInner::FilteredObject(iter) => {
                for next in iter.by_ref() {
                    if !next.is_object() {
                        continue;
                    }
                    return Some(next);
                }
                None
            }
            SubresourceIteratorInner::Empty => None,
        }
    }
}

pub(crate) fn object_iter<'a>(
    (key, value): (&'a String, &'a Value),
) -> SubresourceIteratorInner<'a> {
    match key.as_str() {
        "additionalProperties"
        | "contains"
        | "contentSchema"
        | "else"
        | "if"
        | "items"
        | "not"
        | "propertyNames"
        | "then"
        | "unevaluatedItems"
        | "unevaluatedProperties" => SubresourceIteratorInner::Once(value),
        "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
            if let Some(arr) = value.as_array() {
                SubresourceIteratorInner::Array(arr.iter())
            } else {
                SubresourceIteratorInner::Empty
            }
        }
        "$defs" | "definitions" | "dependentSchemas" | "patternProperties" | "properties" => {
            if let Some(obj) = value.as_object() {
                SubresourceIteratorInner::Object(obj.values())
            } else {
                SubresourceIteratorInner::Empty
            }
        }
        _ => SubresourceIteratorInner::Empty,
    }
}

pub(crate) fn maybe_in_subresource<'r>(
    segments: &Segments,
    resolver: &Resolver<'r>,
    subresource: &InnerResourcePtr,
) -> Result<Resolver<'r>, Error> {
    const IN_VALUE: &[&str] = &[
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    ];
    const IN_CHILD: &[&str] = &[
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    ];

    let mut iter = segments.iter();
    while let Some(segment) = iter.next() {
        if let Segment::Key(key) = segment {
            if !IN_VALUE.contains(&key.as_ref())
                && (!IN_CHILD.contains(&key.as_ref()) || iter.next().is_none())
            {
                return Ok(resolver.clone());
            }
        }
    }
    resolver.in_subresource_inner(subresource)
}

#[inline]
pub(crate) fn maybe_in_subresource_with_items_and_dependencies<'r>(
    segments: &Segments,
    resolver: &Resolver<'r>,
    subresource: &InnerResourcePtr,
    in_value: &[&str],
    in_child: &[&str],
) -> Result<Resolver<'r>, Error> {
    let mut iter = segments.iter();
    while let Some(segment) = iter.next() {
        if let Segment::Key(key) = segment {
            if (*key == "items" || *key == "dependencies") && subresource.contents().is_object() {
                return resolver.in_subresource_inner(subresource);
            }
            if !in_value.contains(&key.as_ref())
                && (!in_child.contains(&key.as_ref()) || iter.next().is_none())
            {
                return Ok(resolver.clone());
            }
        }
    }
    resolver.in_subresource_inner(subresource)
}
