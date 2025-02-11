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

impl Clone for Registry {
    fn clone(&self) -> Self {
        Self {
            documents: self.documents.clone(),
            resources: self.resources.clone(),
            anchors: self.anchors.clone(),
            resolution_cache: self.resolution_cache.clone(),
        }
    }
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
    /// Create a [`Registry`] from multiple resources using these options.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any URI is invalid
    /// - Any referenced resources cannot be retrieved
    pub fn build(
        self,
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    ) -> Result<Registry, Error> {
        Registry::try_from_resources_impl(pairs, &*self.retriever, self.draft)
    }
}

#[cfg(feature = "retrieve-async")]
impl RegistryOptions<Arc<dyn crate::AsyncRetrieve>> {
    /// Create a [`Registry`] from multiple resources using these options with async retrieval.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any URI is invalid
    /// - Any referenced resources cannot be retrieved
    pub async fn build(
        self,
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    ) -> Result<Registry, Error> {
        Registry::try_from_resources_async_impl(pairs, &*self.retriever, self.draft).await
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

#[cfg(feature = "retrieve-async")]
pub trait IntoAsyncRetriever {
    fn into_retriever(self) -> Arc<dyn crate::AsyncRetrieve>;
}

#[cfg(feature = "retrieve-async")]
impl<T: crate::AsyncRetrieve + 'static> IntoAsyncRetriever for T {
    fn into_retriever(self) -> Arc<dyn crate::AsyncRetrieve> {
        Arc::new(self)
    }
}

#[cfg(feature = "retrieve-async")]
impl IntoAsyncRetriever for Arc<dyn crate::AsyncRetrieve> {
    fn into_retriever(self) -> Arc<dyn crate::AsyncRetrieve> {
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
    /// Create a new [`Registry`] with a single resource.
    ///
    /// # Arguments
    ///
    /// * `uri` - The URI of the resource.
    /// * `resource` - The resource to add.
    ///
    /// # Errors
    ///
    /// Returns an error if the URI is invalid or if there's an issue processing the resource.
    pub fn try_new(uri: impl AsRef<str>, resource: Resource) -> Result<Self, Error> {
        Self::try_new_impl(uri, resource, &DefaultRetriever, Draft::default())
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
    fn try_new_impl(
        uri: impl AsRef<str>,
        resource: Resource,
        retriever: &dyn Retrieve,
        draft: Draft,
    ) -> Result<Self, Error> {
        Self::try_from_resources_impl([(uri, resource)], retriever, draft)
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
    /// Create a new [`Registry`] from an iterator of (URI, Resource) pairs using an async retriever.
    ///
    /// # Arguments
    ///
    /// * `pairs` - An iterator of (URI, Resource) pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid or if there's an issue processing the resources.
    #[cfg(feature = "retrieve-async")]
    async fn try_from_resources_async_impl(
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
        retriever: &dyn crate::AsyncRetrieve,
        draft: Draft,
    ) -> Result<Self, Error> {
        let mut documents = AHashMap::new();
        let mut resources = ResourceMap::new();
        let mut anchors = AHashMap::new();
        let mut resolution_cache = UriCache::new();

        process_resources_async(
            pairs,
            retriever,
            &mut documents,
            &mut resources,
            &mut anchors,
            &mut resolution_cache,
            draft,
        )
        .await?;

        Ok(Registry {
            documents,
            resources,
            anchors,
            resolution_cache: resolution_cache.into_shared(),
        })
    }
    /// Create a new registry with a new resource.
    ///
    /// # Errors
    ///
    /// Returns an error if the URI is invalid or if there's an issue processing the resource.
    pub fn try_with_resource(
        self,
        uri: impl AsRef<str>,
        resource: Resource,
    ) -> Result<Registry, Error> {
        let draft = resource.draft();
        self.try_with_resources([(uri, resource)], draft)
    }
    /// Create a new registry with new resources.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid or if there's an issue processing the resources.
    pub fn try_with_resources(
        self,
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
        draft: Draft,
    ) -> Result<Registry, Error> {
        self.try_with_resources_and_retriever(pairs, &DefaultRetriever, draft)
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
    /// Create a new registry with new resources and using the given non-blocking retriever.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid or if there's an issue processing the resources.
    #[cfg(feature = "retrieve-async")]
    pub async fn try_with_resources_and_retriever_async(
        self,
        pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
        retriever: &dyn crate::AsyncRetrieve,
        draft: Draft,
    ) -> Result<Registry, Error> {
        let mut documents = self.documents;
        let mut resources = self.resources;
        let mut anchors = self.anchors;
        let mut resolution_cache = self.resolution_cache.into_local();
        process_resources_async(
            pairs,
            retriever,
            &mut documents,
            &mut resources,
            &mut anchors,
            &mut resolution_cache,
            draft,
        )
        .await?;
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
    #[must_use]
    pub fn resolver_from_raw_parts(
        &self,
        base_uri: Arc<Uri<String>>,
        scopes: List<Uri<String>>,
    ) -> Resolver {
        Resolver::from_parts(self, base_uri, scopes)
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
    /// Returns vocabulary set configured for given draft and contents.
    #[must_use]
    pub fn find_vocabularies(&self, draft: Draft, contents: &Value) -> VocabularySet {
        match draft.detect(contents) {
            Ok(draft) => draft.default_vocabularies(),
            Err(Error::UnknownSpecification { specification }) => {
                // Try to lookup the specification and find enabled vocabularies
                if let Ok(Some(resource)) =
                    uri::from_str(&specification).map(|uri| self.resources.get(&uri))
                {
                    if let Ok(Some(vocabularies)) = vocabularies::find(resource.contents()) {
                        return vocabularies;
                    }
                }
                draft.default_vocabularies()
            }
            _ => unreachable!(),
        }
    }
}

fn process_meta_schemas(
    pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    documents: &mut DocumentStore,
    resources: &mut ResourceMap,
    anchors: &mut AHashMap<AnchorKey, Anchor>,
    resolution_cache: &mut UriCache,
) -> Result<(), Error> {
    let mut queue = VecDeque::with_capacity(32);

    for (uri, resource) in pairs {
        let uri = uri::from_str(uri.as_ref().trim_end_matches('#'))?;
        let key = Arc::new(uri);
        let (draft, contents) = resource.into_inner();
        let boxed = Arc::pin(contents);
        let contents = std::ptr::addr_of!(*boxed);
        let resource = InnerResourcePtr::new(contents, draft);
        documents.insert(Arc::clone(&key), boxed);
        resources.insert(Arc::clone(&key), resource.clone());
        queue.push_back((key, resource));
    }

    // Process current queue and collect references to external resources
    while let Some((mut base, resource)) = queue.pop_front() {
        if let Some(id) = resource.id() {
            base = resolution_cache.resolve_against(&base.borrow(), id)?;
            resources.insert(base.clone(), resource.clone());
        }

        // Look for anchors
        for anchor in resource.anchors() {
            anchors.insert(AnchorKey::new(base.clone(), anchor.name()), anchor);
        }

        // Process subresources
        for contents in resource.draft().subresources_of(resource.contents()) {
            let subresource = InnerResourcePtr::new(contents, resource.draft());
            queue.push_back((base.clone(), subresource));
        }
    }
    Ok(())
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

        collect_external_resources(
            &base,
            resource.contents(),
            &mut state.external,
            &mut state.seen,
            resolution_cache,
            &mut state.scratch,
            &mut state.refers_metaschemas,
        )?;

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

#[cfg(feature = "retrieve-async")]
async fn process_resources_async(
    pairs: impl IntoIterator<Item = (impl AsRef<str>, Resource)>,
    retriever: &dyn crate::AsyncRetrieve,
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

        if !state.external.is_empty() {
            let data = state
                .external
                .drain()
                .filter_map(|uri| {
                    let mut fragmentless = uri.clone();
                    fragmentless.set_fragment(None);
                    if resources.contains_key(&fragmentless) {
                        None
                    } else {
                        Some((uri, fragmentless))
                    }
                })
                .collect::<Vec<_>>();

            let results = {
                let futures = data
                    .iter()
                    .map(|(_, fragmentless)| retriever.retrieve(fragmentless));
                futures::future::join_all(futures).await
            };

            for ((uri, fragmentless), result) in data.iter().zip(results) {
                let retrieved =
                    result.map_err(|err| Error::unretrievable(fragmentless.as_str(), err))?;

                let (key, resource) = create_resource(
                    retrieved,
                    fragmentless.clone(),
                    default_draft,
                    documents,
                    resources,
                )?;

                handle_fragment(uri, &resource, &key, default_draft, &mut state.queue)?;

                state.queue.push_back((key, resource));
            }
        }
    }

    handle_metaschemas(state.refers_metaschemas, resources, anchors);

    Ok(())
}

fn collect_external_resources(
    base: &Uri<String>,
    contents: &Value,
    collected: &mut AHashSet<Uri<String>>,
    seen: &mut HashSet<u64, BuildNoHashHasher>,
    resolution_cache: &mut UriCache,
    scratch: &mut String,
    refers_metaschemas: &mut bool,
) -> Result<(), Error> {
    // URN schemes are not supported for external resolution
    if base.scheme().as_str() == "urn" {
        return Ok(());
    }

    macro_rules! on_reference {
        ($reference:expr, $key:literal) => {
            // Skip well-known schema references
            if $reference.starts_with("https://json-schema.org/draft/")
                || $reference.starts_with("http://json-schema.org/draft-")
                || base.as_str().starts_with("https://json-schema.org/draft/")
            {
                if $key == "$ref" {
                    *refers_metaschemas = true;
                }
            } else if $reference != "#" {
                let mut hasher = AHasher::default();
                (base.as_str(), $reference).hash(&mut hasher);
                let hash = hasher.finish();
                if seen.insert(hash) {
                    // Handle local references separately as they may have nested references to external resources
                    if $reference.starts_with('#') {
                        if let Some(referenced) =
                            pointer(contents, $reference.trim_start_matches('#'))
                        {
                            collect_external_resources(
                                base,
                                referenced,
                                collected,
                                seen,
                                resolution_cache,
                                scratch,
                                refers_metaschemas,
                            )?;
                        }
                    } else {
                        let resolved = if base.has_fragment() {
                            let mut base_without_fragment = base.clone();
                            base_without_fragment.set_fragment(None);

                            let (path, fragment) = match $reference.split_once('#') {
                                Some((path, fragment)) => (path, Some(fragment)),
                                None => ($reference, None),
                            };

                            let mut resolved = (*resolution_cache
                                .resolve_against(&base_without_fragment.borrow(), path)?)
                            .clone();
                            // Add the fragment back if present
                            if let Some(fragment) = fragment {
                                // It is cheaper to check if it is properly encoded than allocate given that
                                // the majority of inputs do not need to be additionally encoded
                                if let Some(encoded) = uri::EncodedString::new(fragment) {
                                    resolved = resolved.with_fragment(Some(encoded));
                                } else {
                                    uri::encode_to(fragment, scratch);
                                    resolved = resolved.with_fragment(Some(
                                        uri::EncodedString::new_or_panic(scratch),
                                    ));
                                    scratch.clear();
                                }
                            }
                            resolved
                        } else {
                            (*resolution_cache.resolve_against(&base.borrow(), $reference)?).clone()
                        };

                        collected.insert(resolved);
                    }
                }
            }
        };
    }

    if let Some(object) = contents.as_object() {
        if object.len() < 3 {
            for (key, value) in object {
                if key == "$ref" {
                    if let Some(reference) = value.as_str() {
                        on_reference!(reference, "$ref");
                    }
                } else if key == "$schema" {
                    if let Some(reference) = value.as_str() {
                        on_reference!(reference, "$schema");
                    }
                }
            }
        } else {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                on_reference!(reference, "$ref");
            }
            if let Some(reference) = object.get("$schema").and_then(Value::as_str) {
                on_reference!(reference, "$schema");
            }
        }
    }
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

