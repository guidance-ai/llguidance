use anyhow::{bail, Result};
use referencing::{uri, Registry, Resolver, Resource};
use serde_json::Value;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
};

use super::{schema::SchemaBuilderOptions, shared_context::SharedContext, RetrieveWrapper};

const DEFAULT_DRAFT: Draft = Draft::Draft202012;
const DEFAULT_ROOT_URI: &str = "json-schema:///";

pub use referencing::{Draft, ResourceRef};

fn draft_for(value: &Value) -> Draft {
    DEFAULT_DRAFT.detect(value).unwrap_or(DEFAULT_DRAFT)
}

pub struct PreContext {
    registry: Registry,
    draft: Draft,
    pub base_uri: String,
    duplicate_ids: HashSet<String>,
}

pub struct Context<'a> {
    resolver: Resolver<'a>,
    duplicate_ids: &'a HashSet<String>,
    pub draft: Draft,
    pub shared: Rc<RefCell<SharedContext>>,
    pub options: SchemaBuilderOptions,
}

impl PreContext {
    pub fn new(contents: Value, retriever: Option<RetrieveWrapper>) -> Result<Self> {
        let draft = draft_for(&contents);
        let resource = draft.create_resource(contents);
        let base_uri = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();
        let duplicate_ids = find_duplicate_ids(resource.contents(), draft, &base_uri)?;

        let retriever: &dyn referencing::Retrieve = if let Some(retriever) = retriever.as_ref() {
            retriever
        } else {
            &referencing::DefaultRetriever
        };

        let registry = {
            // Weirdly no apparent way to instantiate a new registry with a retriever, so we need to
            // make an empty one and then add the retriever + resource that may depend on said retriever
            let empty_registry =
                Registry::try_from_resources(std::iter::empty::<(String, Resource)>())?;
            empty_registry.try_with_resources_and_retriever(
                vec![(&base_uri, resource)],
                retriever,
                draft,
            )?
        };

        Ok(PreContext {
            registry,
            draft,
            base_uri,
            duplicate_ids,
        })
    }
}

/// Find each URI that more than one schema in the document claims with `$id` (`id` in draft 4).
/// The registry keeps only the last of these schemas, so a lookup of such a URI can return
/// the wrong schema. This walk visits the schemas in the same order as the registry does.
/// It must stay in sync with `process_queue` in `referencing::registry`.
/// Copies with equal JSON values (key order ignored) are not duplicates. A lookup of their URI
/// returns the last copy, as before this check.
fn find_duplicate_ids(root: &Value, draft: Draft, root_uri: &str) -> Result<HashSet<String>> {
    let root_uri = Rc::new(uri::from_str(root_uri.trim_end_matches('#'))?);
    let mut owners: HashMap<String, &Value> =
        HashMap::from([(root_uri.as_str().to_string(), root)]);
    let mut duplicates = HashSet::new();
    let mut queue = VecDeque::from([(root_uri, root)]);
    while let Some((mut base, contents)) = queue.pop_front() {
        if let Some(id) = draft.create_resource_ref(contents).id() {
            base = Rc::new(uri::resolve_against(&base.borrow(), id)?);
            let owner = owners.entry(base.as_str().to_string()).or_insert(contents);
            if !std::ptr::eq(*owner, contents) && *owner != contents {
                duplicates.insert(base.as_str().to_string());
            }
        }
        for child in draft.subresources_of(contents) {
            queue.push_back((Rc::clone(&base), child));
        }
    }
    Ok(duplicates)
}

impl<'a> Context<'a> {
    pub fn new(pre_context: &'a PreContext) -> Result<Self> {
        let resolver = pre_context.registry.try_resolver(&pre_context.base_uri)?;
        let ctx = Context {
            resolver,
            duplicate_ids: &pre_context.duplicate_ids,
            draft: pre_context.draft,
            shared: Rc::new(RefCell::new(SharedContext::new())),
            options: SchemaBuilderOptions::default(),
        };

        Ok(ctx)
    }

    pub fn in_subresource(&'a self, resource: ResourceRef) -> Result<Context<'a>> {
        let resolver = self.resolver.in_subresource(resource)?;
        Ok(Context {
            resolver,
            duplicate_ids: self.duplicate_ids,
            draft: resource.draft(),
            shared: Rc::clone(&self.shared),
            options: self.options.clone(),
        })
    }

    pub fn as_resource_ref<'r>(&'a self, contents: &'r Value) -> ResourceRef<'r> {
        self.draft
            .detect(contents)
            .unwrap_or(DEFAULT_DRAFT)
            .create_resource_ref(contents)
    }

    pub fn normalize_ref(&self, reference: &str) -> Result<String> {
        Ok(self
            .resolver
            .resolve_against(&self.resolver.base_uri().borrow(), reference)?
            .normalize()
            .into_string())
    }

    pub fn lookup_resource(&'a self, reference: &str) -> Result<ResourceRef<'a>> {
        // A URI must not identify more than one schema (JSON Schema 2020-12 core, section 9.1.2).
        if !self.duplicate_ids.is_empty() {
            let uri = reference.rsplit_once('#').map_or(reference, |(uri, _)| uri);
            let uri = self
                .resolver
                .resolve_against(&self.resolver.base_uri().borrow(), uri)?;
            if self.duplicate_ids.contains(uri.as_str()) {
                bail!("duplicate schema URI: more than one schema has the URI {uri}");
            }
        }
        let resolved = self.resolver.lookup(reference)?;
        Ok(self.as_resource_ref(resolved.contents()))
    }
}

impl referencing::Retrieve for RetrieveWrapper {
    fn retrieve(
        &self,
        uri: &referencing::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let value = self.0.retrieve(uri.as_str())?;
        Ok(value)
    }
}
