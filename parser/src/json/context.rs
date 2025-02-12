use crate::{HashMap, HashSet};
use anyhow::{bail, Result};
use referencing::{Registry, Resolver, Resource};
use serde_json::Value;
use std::{any::type_name_of_val, cell::RefCell, rc::Rc};

use super::schema::{Schema, SchemaBuilderOptions, IMPLEMENTED, META_AND_ANNOTATIONS};

const DEFAULT_DRAFT: Draft = Draft::Draft202012;
const DEFAULT_ROOT_URI: &str = "json-schema:///";

pub use referencing::{Draft, ResourceRef, Retrieve};

struct SharedContext {
    defs: HashMap<String, Schema>,
    seen: HashSet<String>,
    n_compiled: usize,
}

impl SharedContext {
    fn new() -> Self {
        SharedContext {
            defs: HashMap::default(),
            seen: HashSet::default(),
            n_compiled: 0,
        }
    }
}

fn draft_for(value: &Value) -> Draft {
    DEFAULT_DRAFT.detect(value).unwrap_or(DEFAULT_DRAFT)
}

pub struct PreContext {
    pub registry: Registry,
    pub draft: Draft,
    pub base_uri: String,
}

pub struct Context<'a> {
    resolver: Resolver<'a>,
    pub draft: Draft,
    shared: Rc<RefCell<SharedContext>>,
    options: SchemaBuilderOptions,
}

impl PreContext {
    pub fn new(contents: Value, retriever: Option<&dyn Retrieve>) -> Result<Self> {
        let draft = draft_for(&contents);
        let resource = draft.create_resource(contents);
        let base_uri = resource.id().unwrap_or(DEFAULT_ROOT_URI).to_string();

        let registry = {
            // Weirdly no apparent way to instantiate a new registry with a retriever, so we need to
            // make an empty one and then add the retriever + resource that may depend on said retriever
            let empty_registry =
                Registry::try_from_resources(std::iter::empty::<(String, Resource)>())?;
            empty_registry.try_with_resources_and_retriever(
                vec![(&base_uri, resource)],
                retriever.unwrap_or(&referencing::DefaultRetriever),
                draft,
            )?
        };

        Ok(PreContext {
            registry,
            draft,
            base_uri,
        })
    }
}

impl<'a> Context<'a> {
    pub fn new(pre_context: &'a PreContext) -> Result<Self> {
        let resolver = pre_context.registry.try_resolver(&pre_context.base_uri)?;
        let ctx = Context {
            resolver,
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
        let resolved = self.resolver.lookup(reference)?;
        Ok(self.as_resource_ref(&resolved.contents()))
    }

    pub fn insert_ref(&self, uri: &str, schema: Schema) {
        self.shared
            .borrow_mut()
            .defs
            .insert(uri.to_string(), schema);
    }

    pub fn get_ref_cloned(&self, uri: &str) -> Option<Schema> {
        self.shared.borrow().defs.get(uri).cloned()
    }

    pub fn mark_seen(&self, uri: &str) {
        self.shared.borrow_mut().seen.insert(uri.to_string());
    }

    pub fn been_seen(&self, uri: &str) -> bool {
        self.shared.borrow().seen.contains(uri)
    }

    pub fn is_valid_keyword(&self, keyword: &str) -> bool {
        if !self.draft.is_known_keyword(keyword)
            || IMPLEMENTED.contains(&keyword)
            || META_AND_ANNOTATIONS.contains(&keyword)
        {
            return true;
        }
        return false;
    }

    pub fn increment(&self) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        shared.n_compiled += 1;
        if shared.n_compiled > self.options.max_size {
            bail!("schema too large");
        }
        Ok(())
    }

    pub fn take_defs(&self) -> HashMap<String, Schema> {
        std::mem::take(&mut self.shared.borrow_mut().defs)
    }
}

#[derive(Clone)]
pub struct RetrieveWrapper(pub Rc<dyn Retrieve>);
impl RetrieveWrapper {
    pub fn new(retrieve: Rc<dyn Retrieve>) -> Self {
        RetrieveWrapper(retrieve)
    }
}
impl std::ops::Deref for RetrieveWrapper {
    type Target = dyn Retrieve;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::fmt::Debug for RetrieveWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", type_name_of_val(&self.0))
    }
}
