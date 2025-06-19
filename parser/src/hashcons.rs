use derivre::HashMap;
use std::{hash::Hash, num::NonZeroU32};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HashId<T> {
    id: NonZeroU32,
    _marker: std::marker::PhantomData<T>,
}

impl<T: std::clone::Clone> Copy for HashId<T> {}

#[derive(Debug, Clone)]
pub struct HashCons<T> {
    by_t: HashMap<T, NonZeroU32>,
    all_t: Vec<T>,
}

impl<T: Eq + Hash + Clone> Default for HashCons<T> {
    fn default() -> Self {
        HashCons {
            by_t: HashMap::default(),
            all_t: Vec::new(),
        }
    }
}

impl<T: Eq + Hash + Clone> HashCons<T> {
    pub fn insert(&mut self, t: T) -> HashId<T> {
        if let Some(&id) = self.by_t.get(&t) {
            return HashId {
                id,
                _marker: std::marker::PhantomData,
            };
        }
        let idx = self.all_t.len();
        let id = NonZeroU32::new((idx + 1) as u32).unwrap();
        self.by_t.insert(t.clone(), id);
        self.all_t.push(t);
        HashId {
            id,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn get(&self, id: HashId<T>) -> &T {
        &self.all_t[id.id.get() as usize - 1]
    }

    // pub fn len(&self) -> usize {
    //     self.all_t.len()
    // }
}
