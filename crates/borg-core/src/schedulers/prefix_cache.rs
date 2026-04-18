use radix_trie::{Trie, TrieCommon};
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct CacheEntry {
    lock_count: u32,
    last_touch: u64,
    child_count: u32,
    parent: Option<Vec<u32>>,
}

#[derive(Debug)]
pub struct PrefixCache {
    trie: Trie<Vec<u32>, CacheEntry>,
    evictable_leaves: BTreeSet<(u64, Vec<u32>)>,
    next_touch: u64,
}

impl PrefixCache {
    pub fn bytes_used(&self, block_size: u32, kv_bytes_per_token_per_device: u64) -> u64 {
        self.trie.len() as u64 * u64::from(block_size) * kv_bytes_per_token_per_device
    }

    pub fn longest_hit(&mut self, tokens: &[u32]) -> u32 {
        let tokens = tokens.to_vec();
        let Some(key) = self
            .trie
            .get_ancestor(&tokens)
            .and_then(|subtrie| subtrie.key())
            .cloned()
        else {
            return 0;
        };

        self.touch_entry(&key);
        key.len() as u32
    }

    pub fn insert_prefix(&mut self, prefix: &[u32]) {
        let prefix = prefix.to_vec();
        let touch = self.next_touch();
        let mut remove_leaf = None;
        let mut insert_leaf = None;
        if let Some(entry) = self.trie.get_mut(&prefix) {
            remove_leaf = Self::evictable_leaf_key(&prefix, entry);
            entry.last_touch = touch;
            insert_leaf = Self::evictable_leaf_key(&prefix, entry);
        }
        if let Some(leaf) = remove_leaf {
            self.evictable_leaves.remove(&leaf);
        }
        if let Some(leaf) = insert_leaf {
            self.evictable_leaves.insert(leaf);
            return;
        }

        let parent = self
            .trie
            .get_ancestor(&prefix)
            .and_then(|subtrie| subtrie.key())
            .cloned();
        let mut parent_leaf = None;
        if let Some(parent_key) = parent.as_ref() {
            if let Some(parent_entry) = self.trie.get_mut(parent_key) {
                parent_leaf = Self::evictable_leaf_key(parent_key, parent_entry);
                parent_entry.child_count += 1;
            }
        }
        if let Some(leaf) = parent_leaf {
            self.evictable_leaves.remove(&leaf);
        }

        self.trie.insert(
            prefix.clone(),
            CacheEntry {
                lock_count: 0,
                last_touch: touch,
                child_count: 0,
                parent,
            },
        );
        self.evictable_leaves.insert((touch, prefix));
    }

    pub fn lock_prefix(&mut self, prefix: &[u32]) {
        let prefix = prefix.to_vec();
        let touch = self.next_touch();
        let mut remove_leaf = None;
        let mut insert_leaf = None;
        if let Some(entry) = self.trie.get_mut(&prefix) {
            remove_leaf = Self::evictable_leaf_key(&prefix, entry);
            entry.lock_count += 1;
            entry.last_touch = touch;
            insert_leaf = Self::evictable_leaf_key(&prefix, entry);
        }
        if let Some(leaf) = remove_leaf {
            self.evictable_leaves.remove(&leaf);
        }
        if let Some(leaf) = insert_leaf {
            self.evictable_leaves.insert(leaf);
        }
    }

    pub fn unlock_prefix(&mut self, prefix: &[u32]) {
        let prefix = prefix.to_vec();
        let touch = self.next_touch();
        let mut remove_leaf = None;
        let mut insert_leaf = None;
        if let Some(entry) = self.trie.get_mut(&prefix) {
            remove_leaf = Self::evictable_leaf_key(&prefix, entry);
            entry.lock_count = entry.lock_count.saturating_sub(1);
            entry.last_touch = touch;
            insert_leaf = Self::evictable_leaf_key(&prefix, entry);
        }
        if let Some(leaf) = remove_leaf {
            self.evictable_leaves.remove(&leaf);
        }
        if let Some(leaf) = insert_leaf {
            self.evictable_leaves.insert(leaf);
        }
    }

    pub fn evict_one_leaf(&mut self) -> bool {
        let Some((_, key)) = self.evictable_leaves.pop_first() else {
            return false;
        };
        let Some(removed_entry) = self.trie.remove(&key) else {
            return false;
        };

        if let Some(parent_key) = removed_entry.parent {
            let mut parent_leaf = None;
            if let Some(parent_entry) = self.trie.get_mut(&parent_key) {
                parent_entry.child_count = parent_entry.child_count.saturating_sub(1);
                parent_leaf = Self::evictable_leaf_key(&parent_key, parent_entry);
            }
            if let Some(leaf) = parent_leaf {
                self.evictable_leaves.insert(leaf);
            }
        }
        true
    }

    fn next_touch(&mut self) -> u64 {
        let touch = self.next_touch;
        self.next_touch += 1;
        touch
    }

    fn touch_entry(&mut self, prefix: &[u32]) {
        let prefix = prefix.to_vec();
        let touch = self.next_touch();
        let mut remove_leaf = None;
        let mut insert_leaf = None;
        if let Some(entry) = self.trie.get_mut(&prefix) {
            remove_leaf = Self::evictable_leaf_key(&prefix, entry);
            entry.last_touch = touch;
            insert_leaf = Self::evictable_leaf_key(&prefix, entry);
        }
        if let Some(leaf) = remove_leaf {
            self.evictable_leaves.remove(&leaf);
        }
        if let Some(leaf) = insert_leaf {
            self.evictable_leaves.insert(leaf);
        }
    }

    fn evictable_leaf_key(key: &[u32], entry: &CacheEntry) -> Option<(u64, Vec<u32>)> {
        (entry.lock_count == 0 && entry.child_count == 0).then(|| (entry.last_touch, key.to_vec()))
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self {
            trie: Trie::new(),
            evictable_leaves: BTreeSet::new(),
            next_touch: 0,
        }
    }
}
