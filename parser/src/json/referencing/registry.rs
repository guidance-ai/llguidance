use std::{
    collections::{hash_map::Entry, HashSet, VecDeque},
    hash::{Hash, Hasher},
    pin::Pin,
    sync::Arc,
};

use ahash::{AHashMap, AHashSet, AHasher};
use fluent_uri::Uri;
use once_cell::sync::Lazy;
use serde_json::Value;

use super::{
    anchors::{AnchorKey, AnchorKeyRef},
    cache::{SharedUriCache, UriCache},
    hasher::BuildNoHashHasher,
    list::List,
    meta,
    resource::{unescape_segment, InnerResourcePtr, JsonSchemaResource},
    uri,
    vocabularies::{self, VocabularySet},
    Anchor, DefaultRetriever, Draft, Error, Resolver, Resource, Retrieve,
};

// SAFETY: `Pin` guarantees stable memory locations for resource pointers,
// while `Arc` enables cheap sharing between multiple registries
type DocumentStore = AHashMap<Arc<Uri<String>>, Pin<Arc<Value>>>;
type ResourceMap = AHashMap<Arc<Uri<String>>, InnerResourcePtr>;


/// A registry of JSON Schema resources, each identified by their canonical URIs.
///
/// Registries store a collection of in-memory resources and their anchors.
/// They eagerly process all added resources, including their subresources and anchors.
/// This means that subresources contained within any added resources are immediately
/// discoverable and retrievable via their own IDs.
///
/// # Resource Retrieval
///
/// Registry supports both blocking and non-blocking retrieval of external resources.
///
/// ## Blocking Retrieval
///
/// ```rust
/// use referencing::{Registry, Resource, Retrieve, Uri};
/// use serde_json::{json, Value};
///
/// struct ExampleRetriever;
///
/// impl Retrieve for ExampleRetriever {
///     fn retrieve(
///         &self,
///         uri: &Uri<String>
///     ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
///         // Always return the same value for brevity
///         Ok(json!({"type": "string"}))
///     }
/// }
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let registry = Registry::options()
///     .retriever(ExampleRetriever)
///     .build([
///         // Initial schema that might reference external schemas
///         (
///             "https://example.com/user.json",
///             Resource::from_contents(json!({
///                 "type": "object",
///                 "properties": {
///                     // Should be retrieved by `ExampleRetriever`
///                     "role": {"$ref": "https://example.com/role.json"}
///                 }
///             }))?
///         )
///     ])?;
/// # Ok(())
/// # }
/// ```
///
/// ## Non-blocking Retrieval
///
/// ```rust
/// # #[cfg(feature = "retrieve-async")]
/// # mod example {
/// use referencing::{Registry, Resource, AsyncRetrieve, Uri};
/// use serde_json::{json, Value};
///
/// struct ExampleRetriever;
///
/// #[async_trait::async_trait]
/// impl AsyncRetrieve for ExampleRetriever {
///     async fn retrieve(
///         &self,
///         uri: &Uri<String>
///     ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
///         // Always return the same value for brevity
///         Ok(json!({"type": "string"}))
///     }
/// }
///
///  # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let registry = Registry::options()
///     .async_retriever(ExampleRetriever)
///     .build([
///         (
///             "https://example.com/user.json",
///             Resource::from_contents(json!({
///                 // Should be retrieved by `ExampleRetriever`
///                 "$ref": "https://example.com/common/user.json"
///             }))?
///         )
///     ])
///     .await?;
/// # Ok(())
/// # }
/// # }
/// ```
///
/// The registry will automatically:
///
/// - Resolve external references
/// - Cache retrieved schemas
/// - Handle nested references
/// - Process JSON Schema anchors
///
#[derive(Debug)]
pub struct Registry {
    documents: DocumentStore,
    pub(crate) resources: ResourceMap,
    anchors: AHashMap<AnchorKey, Anchor>,
    resolution_cache: SharedUriCache,
}


/// Configuration options for creating a [`Registry`].
pub struct RegistryOptions<R> {
    retriever: R,
    draft: Draft,
}

impl<R> RegistryOptions<R> {
    /// Set specification version under which the resources should be interpreted under.
    #[must_use]
    pub fn draft(mut self, draft: Draft) -> Self {
        self.draft = draft;
        self
    }
}

impl RegistryOptions<Arc<dyn Retrieve>> {
    /// Create a new [`RegistryOptions`] with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            retriever: Arc::new(DefaultRetriever),
            draft: Draft::default(),
        }
    }
    /// Set a custom retriever for the [`Registry`].
    #[must_use]
    pub fn retriever(mut self, retriever: impl IntoRetriever) -> Self {
        self.retriever = retriever.into_retriever();
        self
    }
    /// Set a custom async retriever for the [`Registry`].
    #[cfg(feature = "retrieve-async")]
    #[must_use]
    pub fn async_retriever(
        self,
        retriever: impl IntoAsyncRetriever,
    ) -> RegistryOptions<Arc<dyn crate::AsyncRetrieve>> {
        RegistryOptions {
            retriever: retriever.into_retriever(),
            draft: self.draft,
        }
    }
   
}


pub trait IntoRetriever {
    fn into_retriever(self) -> Arc<dyn Retrieve>;
}

impl<T: Retrieve + 'static> IntoRetriever for T {
    fn into_retriever(self) -> Arc<dyn Retrieve> {
        Arc::new(self)
    }
}

impl IntoRetriever for Arc<dyn Retrieve> {
    fn into_retriever(self) -> Arc<dyn Retrieve> {
        self
    }
}

impl Default for RegistryOptions<Arc<dyn Retrieve>> {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// Get [`RegistryOptions`] for configuring a new [`Registry`].
    #[must_use]
    pub fn options() -> RegistryOptions<Arc<dyn Retrieve>> {
        RegistryOptions::new()
    }
  
    /// Create a new [`Registry`] from an iterator of (URI, Resource) pairs.
    ///
    /// # Arguments
    ///
    /// * `pairs` - An iterator of (URI, Resource) pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid or if there's an issue processing the resources.
    pub fn try_from_resources(
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    ) -> Result<Self, Error> {
        Self::try_from_resources_impl(pairs, &DefaultRetriever, Draft::default())
    }
  
    fn try_from_resources_impl(
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
        retriever: &dyn Retrieve,
        draft: Draft,
    ) -> Result<Self, Error> {
        let mut documents = AHashMap::new();
        let mut resources = ResourceMap::new();
        let mut anchors = AHashMap::new();
        let mut resolution_cache = UriCache::new();
        process_resources(
            pairs,
            retriever,
            &mut documents,
            &mut resources,
            &mut anchors,
            &mut resolution_cache,
            draft,
        )?;
        Ok(Registry {
            documents,
            resources,
            anchors,
            resolution_cache: resolution_cache.into_shared(),
        })
    }
    /// Create a new registry with new resources and using the given retriever.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid or if there's an issue processing the resources.
    pub fn try_with_resources_and_retriever(
        self,
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
        retriever: &dyn Retrieve,
        draft: Draft,
    ) -> Result<Registry, Error> {
        let mut documents = self.documents;
        let mut resources = self.resources;
        let mut anchors = self.anchors;
        let mut resolution_cache = self.resolution_cache.into_local();
        process_resources(
            pairs,
            retriever,
            &mut documents,
            &mut resources,
            &mut anchors,
            &mut resolution_cache,
            draft,
        )?;
        Ok(Registry {
            documents,
            resources,
            anchors,
            resolution_cache: resolution_cache.into_shared(),
        })
    }
    /// Create a new [`Resolver`] for this registry with the given base URI.
    ///
    /// # Errors
    ///
    /// Returns an error if the base URI is invalid.
    pub fn try_resolver(&self, base_uri: &str) -> Result<Resolver, Error> {
        let base = uri::from_str(base_uri)?;
        Ok(self.resolver(base))
    }
    /// Create a new [`Resolver`] for this registry with a known valid base URI.
    #[must_use]
    pub fn resolver(&self, base_uri: Uri<String>) -> Resolver {
        Resolver::new(self, Arc::new(base_uri))
    }
   
    pub(crate) fn anchor<'a>(&self, uri: &'a Uri<String>, name: &'a str) -> Result<&Anchor, Error> {
        let key = AnchorKeyRef::new(uri, name);
        if let Some(value) = self.anchors.get(key.borrow_dyn()) {
            return Ok(value);
        }
        let resource = &self.resources[uri];
        if let Some(id) = resource.id() {
            let uri = uri::from_str(id)?;
            let key = AnchorKeyRef::new(&uri, name);
            if let Some(value) = self.anchors.get(key.borrow_dyn()) {
                return Ok(value);
            }
        }
        if name.contains('/') {
            Err(Error::invalid_anchor(name.to_string()))
        } else {
            Err(Error::no_such_anchor(name.to_string()))
        }
    }
    /// Resolves a reference URI against a base URI using registry's cache.
    ///
    /// # Errors
    ///
    /// Returns an error if base has not schema or there is a fragment.
    pub fn resolve_against(&self, base: &Uri<&str>, uri: &str) -> Result<Arc<Uri<String>>, Error> {
        self.resolution_cache.resolve_against(base, uri)
    }
   
}
struct ProcessingState {
    queue: VecDeque<(Arc<Uri<String>>, InnerResourcePtr)>,
    seen: HashSet<u64, BuildNoHashHasher>,
    external: AHashSet<Uri<String>>,
    scratch: String,
    refers_metaschemas: bool,
}

impl ProcessingState {
    fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(32),
            seen: HashSet::with_hasher(BuildNoHashHasher::default()),
            external: AHashSet::new(),
            scratch: String::new(),
            refers_metaschemas: false,
        }
    }
}

fn process_input_resources(
    pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    documents: &mut DocumentStore,
    resources: &mut ResourceMap,
    state: &mut ProcessingState,
) -> Result<(), Error> {
    for (uri, resource) in pairs {
        let uri = uri::from_str(uri.as_ref().trim_end_matches('#'))?;
        let key = Arc::new(uri);
        match documents.entry(Arc::clone(&key)) {
            Entry::Occupied(_) => {}
            Entry::Vacant(entry) => {
                let (draft, contents) = resource.into_inner();
                let boxed = Arc::pin(contents);
                let contents = std::ptr::addr_of!(*boxed);
                let resource = InnerResourcePtr::new(contents, draft);
                resources.insert(Arc::clone(&key), resource.clone());
                state.queue.push_back((key, resource));
                entry.insert(boxed);
            }
        }
    }
    Ok(())
}

fn process_queue(
    state: &mut ProcessingState,
    resources: &mut ResourceMap,
    anchors: &mut AHashMap<AnchorKey, Anchor>,
    resolution_cache: &mut UriCache,
) -> Result<(), Error> {
    while let Some((mut base, resource)) = state.queue.pop_front() {
        if let Some(id) = resource.id() {
            base = resolution_cache.resolve_against(&base.borrow(), id)?;
            resources.insert(base.clone(), resource.clone());
        }

        for anchor in resource.anchors() {
            anchors.insert(AnchorKey::new(base.clone(), anchor.name()), anchor);
        }

        for contents in resource.draft().subresources_of(resource.contents()) {
            let subresource = InnerResourcePtr::new(contents, resource.draft());
            state.queue.push_back((base.clone(), subresource));
        }
    }
    Ok(())
}

fn handle_fragment(
    uri: &Uri<String>,
    resource: &InnerResourcePtr,
    key: &Arc<Uri<String>>,
    default_draft: Draft,
    queue: &mut VecDeque<(Arc<Uri<String>>, InnerResourcePtr)>,
) -> Result<(), Error> {
    if let Some(fragment) = uri.fragment() {
        if let Some(resolved) = pointer(resource.contents(), fragment.as_str()) {
            let draft = default_draft.detect(resolved)?;
            let contents = std::ptr::addr_of!(*resolved);
            let resource = InnerResourcePtr::new(contents, draft);
            queue.push_back((Arc::clone(key), resource));
        }
    }
    Ok(())
}

fn handle_metaschemas(
    refers_metaschemas: bool,
    resources: &mut ResourceMap,
    anchors: &mut AHashMap<AnchorKey, Anchor>,
) {
   
}

fn create_resource(
    retrieved: Value,
    fragmentless: Uri<String>,
    default_draft: Draft,
    documents: &mut DocumentStore,
    resources: &mut ResourceMap,
) -> Result<(Arc<Uri<String>>, InnerResourcePtr), Error> {
    let draft = default_draft.detect(&retrieved)?;
    let boxed = Arc::pin(retrieved);
    let contents = std::ptr::addr_of!(*boxed);
    let resource = InnerResourcePtr::new(contents, draft);
    let key = Arc::new(fragmentless);
    documents.insert(Arc::clone(&key), boxed);
    resources.insert(Arc::clone(&key), resource.clone());
    Ok((key, resource))
}

fn process_resources(
    pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    retriever: &dyn Retrieve,
    documents: &mut DocumentStore,
    resources: &mut ResourceMap,
    anchors: &mut AHashMap<AnchorKey, Anchor>,
    resolution_cache: &mut UriCache,
    default_draft: Draft,
) -> Result<(), Error> {
    let mut state = ProcessingState::new();
    process_input_resources(pairs, documents, resources, &mut state)?;

    loop {
        if state.queue.is_empty() && state.external.is_empty() {
            break;
        }

        process_queue(&mut state, resources, anchors, resolution_cache)?;

        // Retrieve external resources
        for uri in state.external.drain() {
            let mut fragmentless = uri.clone();
            fragmentless.set_fragment(None);
            if !resources.contains_key(&fragmentless) {
                let retrieved = retriever
                    .retrieve(&fragmentless)
                    .map_err(|err| Error::unretrievable(fragmentless.as_str(), err))?;

                let (key, resource) =
                    create_resource(retrieved, fragmentless, default_draft, documents, resources)?;

                handle_fragment(&uri, &resource, &key, default_draft, &mut state.queue)?;

                state.queue.push_back((key, resource));
            }
        }
    }

    handle_metaschemas(state.refers_metaschemas, resources, anchors);

    Ok(())
}

// A slightly faster version of pointer resolution based on `Value::pointer` from `serde_json`.
fn pointer<'a>(document: &'a Value, pointer: &str) -> Option<&'a Value> {
    if pointer.is_empty() {
        return Some(document);
    }
    if !pointer.starts_with('/') {
        return None;
    }
    pointer.split('/').skip(1).map(unescape_segment).try_fold(
        document,
        |target, token| match target {
            Value::Object(map) => map.get(&*token),
            Value::Array(list) => parse_index(&token).and_then(|x| list.get(x)),
            _ => None,
        },
    )
}

// Taken from `serde_json`.
fn parse_index(s: &str) -> Option<usize> {
    if s.starts_with('+') || (s.starts_with('0') && s.len() != 1) {
        return None;
    }
    s.parse().ok()
}

