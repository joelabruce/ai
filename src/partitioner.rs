use std::{ops::Index, thread};

use crate::partition::Partition;

/// Partions data to be operated on, and provides for multi-threading.
#[derive(Hash, Debug, Clone, PartialEq, Default)]
pub struct Partitioner {
    partitions: Vec<Partition>
}

impl Index<usize> for Partitioner {
    type Output = Partition;

    fn index(&self, index: usize) -> &Self::Output {
        &self.partitions[index]
    }
}

impl Partitioner {
    pub fn new(partitions: Vec<Partition>) -> Self { Partitioner { partitions } }

    /// Creates a partitioner with partitions that are mostly equal in size, with no more than a difference of 1.    
    pub fn with_partitions(count: usize, partition_count: usize) -> Self {
        let partition_size = count / partition_count;

        let mut partitions: Vec<Partition>;
        if partition_size < 1 {
        // Count is not large enough to split into partitions
            partitions = vec![Partition::new(0, count - 1)];

            return Partitioner { partitions };
        } else {
            partitions = Vec::with_capacity(partition_count);
        }

        // Calculates left over items and distributes remainder
        let spread = count % partition_count;
        let mut cursor = 0;
        let mut end: usize;
        for partition_index in 0..partition_count {
            let adjusted_partition_size = partition_size + if partition_index < spread { 1 } else { 0 };
            let start = cursor;
            cursor = start + adjusted_partition_size;
            end = cursor - 1;

            partitions.push(Partition::new(start, end));
        } 

        Partitioner { partitions }
    }

    /// Parallelizes work among partitions as evenly as possible.
    /// Ensures result is aggregated in correct order. 
    pub fn parallelized<T, F>(&self, function: F) -> Vec<T> 
    where
        F: FnOnce(&Partition) -> Vec<T> + Send + Copy,
        T : Send
    {
        if self.partitions.len() == 1 {
        // Since only 1 partition, do not use threading.
            return function(&self.partitions[0]);
        }

        let mut values: Vec<T> = Vec::new();
        thread::scope(|s| {
            let mut scope_join_handles = Vec::with_capacity(self.partitions.len());

            for partition in &self.partitions[..] {
                scope_join_handles.push(s.spawn(move || {
                    function(&partition)
                }));
            }

            for scope_join_handle in scope_join_handles {
                match scope_join_handle.join() {
                    Ok(result) => { 
                        values.extend(result); 
                    },
                    Err(_err) => panic!("{:?}", _err)
                }
            }
        });

        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelizable_simple() {
        let tc1 = 1000;
        let tc = Partitioner::with_partitions(tc1, 8);

        let actual = tc.parallelized(|partition| {
            let partition_values = (partition.range()).collect();
            partition_values
        });

        let expected:Vec<_> = (0..tc1).collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_parallelizable_use_case() {
        let tc1 = 1000;
        let tc = Partitioner::with_partitions(tc1, 8);

        let actual = tc.parallelized(|partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for index in partition.range() {
                partition_values.push(2 * index);
            }
            partition_values
        });

        let expected:Vec<_> = (0..tc1).map(|x| x * 2).collect();

        assert_eq!(actual, expected);        
    }

    #[test]
    fn test_single_threaded() {
        let tc1 = 15;
        let tc = Partitioner::with_partitions(tc1, 16);

        let actual = tc.parallelized(|partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for index in partition.range() {
                partition_values.push(2 * index);
            }
            partition_values
        });

        let expected:Vec<_> = (0..tc1).map(|x| x * 2).collect();

        assert_eq!(actual, expected);        
    }

    #[test]
    #[should_panic]
    fn test_thread_errored() {
        let partitioner = Partitioner::with_partitions(10, 2);

        let _p: Vec<f32> = partitioner.parallelized(|_| {
            let mut _partition_values = vec![0.];

            panic!("Thread panicked! (TEST)");
        });
    }
}