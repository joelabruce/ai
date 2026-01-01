use std::{fmt::Debug, ops::Index, result, thread};

use rand_distr::num_traits::ToPrimitive;

use crate::partition::Partition;

pub const CORE16: usize = 16;
pub const CORE8: usize = 8;
pub const CORE4: usize = 4;
pub const CORE2: usize = 2;

pub const PARALLELISM: usize = CORE16;

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
    #[deprecated]   
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
    #[deprecated]
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

    pub fn chunked_parallelized<T, F>(chunk_size: usize, buffer: &mut [T], function: F) 
    where
        F: FnOnce(usize) -> T + Send + Copy,
        T: Send
    {
        //let mut values = &Vec::new();
        //let chunk_size = self.partitions.len();

        thread::scope(|s| {
            for (chunk_index, chunk) in buffer.chunks_mut(chunk_size).enumerate() {
                s.spawn(move || {
                    for (slice_index, elem) in chunk.iter_mut().enumerate() {
                        let buffer_index = chunk_index * chunk_size + slice_index;
                        let x = function(buffer_index);
                        *elem = x;
                        //println!("slice_index: {slice_index}, chunk_index: {chunk_index}, buffer_index: {buffer_index}");
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_parallelizable() {
        let tc_value_count = 10000;
        let mut buffer = vec![0.0f32; tc_value_count];
        let chunk_size = thread::available_parallelism().unwrap().get();
        println!("chunk_size: {chunk_size}");
        //let tc = Partitioner::with_partitions(tc_value_count, chunk_size);

        Partitioner::chunked_parallelized(chunk_size, &mut buffer, |index| {
            index.to_f32().unwrap() * 2.0f32
        });

        let expected: Vec<_> = (0..tc_value_count).map(|x| x.to_f32().unwrap() * 2.0f32).collect();
        assert_eq!(buffer, expected);
    }

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