use std::{
    borrow::Cow,
    sync::atomic::{AtomicPtr, Ordering},
};

use serde_json::Value;

use super::{Anchor, Draft, Error, Resolved, Resolver, Segments};

pub(crate) trait JsonSchemaResource {
    fn contents(&self) -> &Value;
    fn draft(&self) -> Draft;
    fn id(&self) -> Option<&str> {
        self.draft()
            .id_of(self.contents())
            .map(|id| id.trim_end_matches('#'))
    }
}

/// An owned document with a concrete interpretation under a JSON Schema specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resource {
    contents: Value,
    draft: Draft,
}

impl Resource {
    pub(crate) fn new(contents: Value, draft: Draft) -> Self {
        Self { contents, draft }
    }
    pub(crate) fn into_inner(self) -> (Draft, Value) {
        (self.draft, self.contents)
    }
    /// Resource contents.
    #[must_use]
    pub fn contents(&self) -> &Value {
        &self.contents
    }
    /// JSON Schema draft under which this contents is interpreted.
    #[must_use]
    pub fn draft(&self) -> Draft {
        self.draft
    }
    /// Create a resource with automatically detecting specification which applies to the contents.
    ///
    /// # Errors
    ///
    /// On unknown `$schema` value it returns [`Error::UnknownSpecification`]
    pub fn from_contents(contents: Value) -> Result<Resource, Error> {
        Self::from_contents_and_specification(contents, Draft::default())
    }

    pub(crate) fn from_contents_and_specification(
        contents: Value,
        draft: Draft,
    ) -> Result<Resource, Error> {
        Ok(draft.detect(&contents)?.create_resource(contents))
    }
    /// Resource identifier.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.draft
            .id_of(&self.contents)
            .map(|id| id.trim_end_matches('#'))
    }
    #[must_use]
    pub fn as_ref(&self) -> ResourceRef<'_> {
        ResourceRef {
            contents: &self.contents,
            draft: self.draft,
        }
    }
}

/// A borrowed document with a concrete interpretation under a JSON Schema specification.
#[derive(Debug, Clone, Copy)]
pub struct ResourceRef<'a> {
    contents: &'a Value,
    draft: Draft,
}

impl<'a> ResourceRef<'a> {
    #[must_use]
    pub fn new(contents: &'a Value, draft: Draft) -> Self {
        Self { contents, draft }
    }
    #[must_use]
    pub fn contents(&self) -> &'a Value {
        self.contents
    }
    #[must_use]
    pub fn draft(&self) -> Draft {
        self.draft
    }
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        JsonSchemaResource::id(self)
    }
}

impl JsonSchemaResource for ResourceRef<'_> {
    fn contents(&self) -> &Value {
        self.contents
    }

    fn draft(&self) -> Draft {
        self.draft
    }
}

/// A pointer to a pinned resource.
pub(crate) struct InnerResourcePtr {
    contents: AtomicPtr<Value>,
    draft: Draft,
}

impl Clone for InnerResourcePtr {
    fn clone(&self) -> Self {
        Self {
            contents: AtomicPtr::new(self.contents.load(Ordering::Relaxed)),
            draft: self.draft,
        }
    }
}

impl std::fmt::Debug for InnerResourcePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerResourcePtr")
            .field("contents", self.contents())
            .field("draft", &self.draft)
            .finish()
    }
}

impl InnerResourcePtr {
    pub(crate) fn new(contents: *const Value, draft: Draft) -> Self {
        Self {
            contents: AtomicPtr::new(contents.cast_mut()),
            draft,
        }
    }

    #[allow(unsafe_code)]
    pub(crate) fn contents(&self) -> &Value {
        // SAFETY: The pointer is valid as long as the registry exists
        unsafe { &*self.contents.load(Ordering::Relaxed) }
    }

    #[inline]
    pub(crate) fn draft(&self) -> Draft {
        self.draft
    }

    pub(crate) fn anchors(&self) -> impl Iterator<Item = Anchor> + '_ {
        self.draft().anchors(self.contents())
    }

    pub(crate) fn pointer<'r>(
        &'r self,
        pointer: &str,
        mut resolver: Resolver<'r>,
    ) -> Result<Resolved<'r>, Error> {
        // INVARIANT: Pointer always starts with `/`
        let mut contents = self.contents();
        let mut segments = Segments::new();
        let original_pointer = pointer;
        let pointer = percent_encoding::percent_decode_str(&pointer[1..])
            .decode_utf8()
            .map_err(|err| Error::invalid_percent_encoding(original_pointer, err))?;
        for segment in pointer.split('/') {
            if let Some(array) = contents.as_array() {
                let idx = segment
                    .parse::<usize>()
                    .map_err(|err| Error::invalid_array_index(original_pointer, segment, err))?;
                if let Some(next) = array.get(idx) {
                    contents = next;
                } else {
                    return Err(Error::pointer_to_nowhere(original_pointer));
                }
                segments.push(idx);
            } else {
                let segment = unescape_segment(segment);
                if let Some(next) = contents.get(segment.as_ref()) {
                    contents = next;
                } else {
                    return Err(Error::pointer_to_nowhere(original_pointer));
                }
                segments.push(segment);
            }
            let last = &resolver;
            let new_resolver = self.draft().maybe_in_subresource(
                &segments,
                &resolver,
                &InnerResourcePtr::new(contents, self.draft()),
            )?;
            if new_resolver != *last {
                segments = Segments::new();
            }
            resolver = new_resolver;
        }
        Ok(Resolved::new(contents, resolver, self.draft()))
    }
}

impl JsonSchemaResource for InnerResourcePtr {
    fn contents(&self) -> &Value {
        self.contents()
    }

    fn draft(&self) -> Draft {
        self.draft
    }
}

pub(crate) fn unescape_segment(mut segment: &str) -> Cow<str> {
    // Naively, checking for `~` and then replacing implies two passes
    // over the input buffer. First, search in the first `contains('~')` call
    // and then replacing `~1` & `~0` at once in a single pass.
    //
    // This implementation is ~3x faster than the naive one.
    //
    // **NOTE**: Heavily inspired by the implementation in `boon`:
    // `https://github.com/santhosh-tekuri/boon/blob/fb09df2db19be75c32c0970b4bdedf1655f5f943/src/util.rs#L31`
    let Some(mut tilde_idx) = segment.find('~') else {
        return Cow::Borrowed(segment);
    };

    let mut buffer = String::with_capacity(segment.len());
    loop {
        let (before, after) = segment.split_at(tilde_idx);
        buffer.push_str(before);
        segment = &after[1..];
        let next_char_size = match segment.chars().next() {
            Some('1') => {
                buffer.push('/');
                1
            }
            Some('0') => {
                buffer.push('~');
                1
            }
            Some(next) => {
                buffer.push('~');
                buffer.push(next);
                next.len_utf8()
            }
            None => {
                buffer.push('~');
                break;
            }
        };
        segment = &segment[next_char_size..];
        let Some(next_tilde_idx) = segment.find('~') else {
            buffer.push_str(segment);
            break;
        };
        tilde_idx = next_tilde_idx;
    }
    Cow::Owned(buffer)
}
