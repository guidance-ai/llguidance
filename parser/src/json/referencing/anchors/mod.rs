use std::{
    hash::Hash,
    sync::atomic::{AtomicPtr, Ordering},
};

use serde_json::Value;

mod keys;

use super::{resource::InnerResourcePtr, Draft, Error, Resolved, Resolver};
pub(crate) use keys::{AnchorKey, AnchorKeyRef};

#[derive(Debug)]
pub(crate) struct AnchorName {
    ptr: AtomicPtr<u8>,
    len: usize,
}

impl AnchorName {
    fn new(s: &str) -> Self {
        Self {
            ptr: AtomicPtr::new(s.as_ptr().cast_mut()),
            len: s.len(),
        }
    }

    #[allow(unsafe_code)]
    fn as_str(&self) -> &str {
        // SAFETY: The pointer is valid as long as the registry exists
        unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                self.ptr.load(Ordering::Relaxed),
                self.len,
            ))
        }
    }
}

impl Clone for AnchorName {
    fn clone(&self) -> Self {
        Self {
            ptr: AtomicPtr::new(self.ptr.load(Ordering::Relaxed)),
            len: self.len,
        }
    }
}

impl Hash for AnchorName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl PartialEq for AnchorName {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for AnchorName {}

/// An anchor within a resource.
#[derive(Debug, Clone)]
pub(crate) enum Anchor {
    Default {
        name: AnchorName,
        resource: InnerResourcePtr,
    },
    Dynamic {
        name: AnchorName,
        resource: InnerResourcePtr,
    },
}

impl Anchor {
    /// Anchor's name.
    pub(crate) fn name(&self) -> AnchorName {
        match self {
            Anchor::Default { name, .. } | Anchor::Dynamic { name, .. } => name.clone(),
        }
    }
    /// Get the resource for this anchor.
    pub(crate) fn resolve<'r>(&'r self, resolver: Resolver<'r>) -> Result<Resolved<'r>, Error> {
        match self {
            Anchor::Default { resource, .. } => Ok(Resolved::new(
                resource.contents(),
                resolver,
                resource.draft(),
            )),
            Anchor::Dynamic { name, resource } => {
                let mut last = resource;
                for uri in &resolver.dynamic_scope() {
                    match resolver.registry.anchor(uri, name.as_str()) {
                        Ok(anchor) => {
                            if let Anchor::Dynamic { resource, .. } = anchor {
                                last = resource;
                            }
                        }
                        Err(Error::NoSuchAnchor { .. }) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Ok(Resolved::new(
                    last.contents(),
                    resolver.in_subresource_inner(last)?,
                    last.draft(),
                ))
            }
        }
    }
}

pub(crate) enum AnchorIter {
    Empty,
    One(Anchor),
    Two(Anchor, Anchor),
}

impl Iterator for AnchorIter {
    type Item = Anchor;

    fn next(&mut self) -> Option<Self::Item> {
        match std::mem::replace(self, AnchorIter::Empty) {
            AnchorIter::Empty => None,
            AnchorIter::One(anchor) => Some(anchor),
            AnchorIter::Two(first, second) => {
                *self = AnchorIter::One(second);
                Some(first)
            }
        }
    }
}

pub(crate) fn anchor(draft: Draft, contents: &Value) -> AnchorIter {
    let Some(schema) = contents.as_object() else {
        return AnchorIter::Empty;
    };

    // First check for top-level anchors
    let default_anchor =
        schema
            .get("$anchor")
            .and_then(Value::as_str)
            .map(|name| Anchor::Default {
                name: AnchorName::new(name),
                resource: InnerResourcePtr::new(contents, draft),
            });

    let dynamic_anchor = schema
        .get("$dynamicAnchor")
        .and_then(Value::as_str)
        .map(|name| Anchor::Dynamic {
            name: AnchorName::new(name),
            resource: InnerResourcePtr::new(contents, draft),
        });

    match (default_anchor, dynamic_anchor) {
        (Some(default), Some(dynamic)) => AnchorIter::Two(default, dynamic),
        (Some(default), None) => AnchorIter::One(default),
        (None, Some(dynamic)) => AnchorIter::One(dynamic),
        (None, None) => AnchorIter::Empty,
    }
}

pub(crate) fn anchor_2019(draft: Draft, contents: &Value) -> AnchorIter {
    match contents
        .as_object()
        .and_then(|schema| schema.get("$anchor"))
        .and_then(Value::as_str)
    {
        Some(name) => AnchorIter::One(Anchor::Default {
            name: AnchorName::new(name),
            resource: InnerResourcePtr::new(contents, draft),
        }),
        None => AnchorIter::Empty,
    }
}

pub(crate) fn legacy_anchor_in_dollar_id(draft: Draft, contents: &Value) -> AnchorIter {
    match contents
        .as_object()
        .and_then(|schema| schema.get("$id"))
        .and_then(Value::as_str)
        .and_then(|id| id.strip_prefix('#'))
    {
        Some(id) => AnchorIter::One(Anchor::Default {
            name: AnchorName::new(id),
            resource: InnerResourcePtr::new(contents, draft),
        }),
        None => AnchorIter::Empty,
    }
}

pub(crate) fn legacy_anchor_in_id(draft: Draft, contents: &Value) -> AnchorIter {
    match contents
        .as_object()
        .and_then(|schema| schema.get("id"))
        .and_then(Value::as_str)
        .and_then(|id| id.strip_prefix('#'))
    {
        Some(id) => AnchorIter::One(Anchor::Default {
            name: AnchorName::new(id),
            resource: InnerResourcePtr::new(contents, draft),
        }),
        None => AnchorIter::Empty,
    }
}
