use std::collections::HashMap;

use crate::Partitioner;

pub struct PartitionerCache {
    cache: HashMap<usize, Partitioner>
}

impl PartitionerCache {
    pub fn new() -> Self {
        PartitionerCache {
            cache: HashMap::new()
        }
    }

    /// Allows getting a partition of given size if it already exists, otherwise it will create it and then store it.
    pub fn get_or_add(&mut self, count: usize, partition_count: usize) -> &Partitioner {
        self.cache.entry(count).or_insert_with_key(|key| { 
            Partitioner::with_partitions(*key, partition_count)
        })
    }
}