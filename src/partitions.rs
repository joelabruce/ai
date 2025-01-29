use std::{ops::RangeInclusive, thread};

pub struct Partitioned<T> {
    pub partitioned: T,
    pub partitioner: Partitioner
}

/// Partions data to be operated on, and provides for multi-threading.
#[derive(Hash, Debug, Clone, PartialEq)]
pub struct Partitioner {
    partitions: Vec<Partition>
}

impl Partitioner {
    /// Creates a partitioner with partitions that are mostly equal in size, with no more than a difference of 1.    
    pub fn with_partitions(count: usize, partition_count: usize) -> Self {
        let partition_size = count / partition_count;

        let mut partitions: Vec<Partition>;
        if partition_size < 1 {
        // Count is not large enough to split into partitions
            partitions = vec![
                Partition {
                    start: 0,
                    end: count - 1
                }];

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

            partitions.push(
                Partition {
                    start,
                    end
                }
            );
        } 

        Partitioner { partitions }
    }

    /// Returns partition if it exists.
    pub fn get_partition(&self, partition_index: usize) -> &Partition {
        assert!(partition_index < self.partitions.len(), "Index for partition out of bounds.");
        &self.partitions[partition_index]
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
                    Err(_err) => { println!("{:?}", _err); }
                }
            }
        });

        values
    }
}

/// A partition to be used when processing a subset of data
#[derive(Hash, Debug, Clone, PartialEq)]
pub struct Partition {
    start: usize,
    end: usize
}

impl Partition {
    /// Returns size of the partition.
    pub fn get_size(&self) -> usize { self.end - self.start + 1 }

    /// Creates a range to work with when processing data for the partition.
    pub fn get_range(&self) -> RangeInclusive<usize> { self.start..=self.end }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_partition_sizes() {
        let tc1_count = 98;
        let partition_count = 4;
        let actual = Partitioner::with_partitions(tc1_count, partition_count);

        assert_eq!(actual.get_partition(0).start, 0);
        assert_eq!(actual.get_partition(0).end, 24);
        assert_eq!(actual.get_partition(0).get_range(), (0..=24));
        assert_eq!(actual.get_partition(0).get_size(), 25);

        assert_eq!(actual.get_partition(1).start, 25);
        assert_eq!(actual.get_partition(1).end, 49);
        assert_eq!(actual.get_partition(1).get_range(), (25..=49));
        assert_eq!(actual.get_partition(1).get_size(), 25);

        assert_eq!(actual.get_partition(2).start, 50);
        assert_eq!(actual.get_partition(2).end, 73);
        assert_eq!(actual.get_partition(2).get_range(), (50..=73));
        assert_eq!(actual.get_partition(2).get_size(), 24);

        assert_eq!(actual.get_partition(3).start, 74);
        assert_eq!(actual.get_partition(3).end, 97);
        assert_eq!(actual.get_partition(0).get_range(), (74..=97));
        assert_eq!(actual.get_partition(0).get_size(), 24);

        actual.get_partition(4);
    }

    #[test]
    fn test_parallelizable_simple() {
        let tc1 = 1000;
        let tc = Partitioner::with_partitions(tc1, 8);

        let actual = tc.parallelized(|partition| {
            let partition_values = (partition.get_range()).collect();
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
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for index in partition.get_range() {
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
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for index in partition.get_range() {
                partition_values.push(2 * index);
            }
            partition_values
        });

        let expected:Vec<_> = (0..tc1).map(|x| x * 2).collect();

        assert_eq!(actual, expected);        
    }
}