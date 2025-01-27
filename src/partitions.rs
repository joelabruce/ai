use std::{ops::RangeInclusive, thread};

/// Partions data to be operated on, and provides for multi-threading.
pub struct Partitioner {
    partitions: Vec<Partition>
}

impl Partitioner {
    /// Creates a partitioner with partitions that are mostly equal in size, with no more than a difference of 1.    
    pub fn with_partitions(count: usize, partition_count: usize) -> Self {
        let partition_size = count / partition_count;

        let mut partitions = Vec::with_capacity(partition_count);
        if partition_size < 1 {
        // Count is not large enough to split into partitions
            partitions.push(Partition {
                start: 0,
                end: count
            });

            return Partitioner {
                partitions
            };
        }

        // Calculates left over items and distributes remainder
        let spread = count % partition_count;
        let mut cursor = 0;
        for partition_index in 0..partition_count {
            let adjusted_partition_size = partition_size + if partition_index < spread { 1 } else { 0 };
            let start = cursor;
            let end = start + adjusted_partition_size - 1;
            cursor = end + 1;

            partitions.push(
                Partition {
                    start,
                    end
                }
            );
        } 

        Partitioner {
            partitions
        }
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
                    Err(_err) => { }
                }
            }
        });

        values
    }
}

/// A partition to be used when processing a subset of data
pub struct Partition {
    start: usize,
    end: usize
}

impl Partition {
    /// Returns size of the partition.
    pub fn get_size(&self) -> usize {
        self.end - self.start
    }

    /// Returns start of partition.
    pub fn get_start(&self) -> usize {
        self.start
    }

    /// Returns end of partition.
    pub fn get_end(&self) -> usize {
        self.end
    }

    /// Creates a range to work with when processing data for the partition.
    pub fn get_range(&self) -> RangeInclusive<usize>{
        self.start..=self.end
    }
}

/// Caclulates strict partitions of n into 3 distinct parts
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_3_fastest(n: i128) -> i128 {
    (n*n + 3)/12
}

/// Calculates strict partitions of n into 3 distinct parts using n choose
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_3_fast(n: i128) -> i128 {
    let n_choose = ((n - 1) * (n - 2)) / 2;
    let floor_term = 3 * ((n - 1) / 2);
    let adjustment = if n % 3 == 0 { 2 } else { 0 };

    (n_choose - floor_term + adjustment) / 6 
}

/// Attempting to find a simple closed form version of strict partitions of n into 4 distinct parts
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_4_experimental(n: i128) -> i128 {
    // really close, but gets 10th term incorrect
//     let n_choose = ((n - 4) * (n - 5) * (n - 6)) / 6;
//     let floor_term = 0;//4 * ((n - 2) / 6);
//     let adjustment = 0;//if n % 4 == 0 { 6 } else { 0 }; 
//     (n_choose - floor_term + adjustment) / 24
    let o = n - 4;
    (o*o*o - strict_partitions_n_into_3_fastest(n))/144 //- strict_partitions_n_into_3_fastest(n)
}

/// Gives exact result of strict partitions of n into 4 parts using a loop
/// # Arguments
/// # Returns
pub fn strict_partitions_n_into_4_recursive(n: i128) -> i128 {
    let terms = (n - 4) / 4;
    let offset = n % 4;

    let mut accumulator = 0;
    for i in 1..=terms {
        accumulator += strict_partitions_n_into_3_fastest(4*i + offset);
    }

    accumulator
}

/// Gives exact result of strict partitions of n into 5 parts using a loop
/// # Arguments
/// # Returns
fn _strict_partitions_n_into_5_recursive(n: i128) -> i128 {
    let terms = (n - 5) / 5;
    let offset = n % 5;

    let mut accumulator = 0;
    for i in 1..=terms {
        accumulator += strict_partitions_n_into_4_recursive(5*i + offset);
    }

    accumulator
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

        assert_eq!(actual.get_partition(1).start, 25);
        assert_eq!(actual.get_partition(1).end, 49);

        assert_eq!(actual.get_partition(2).start, 50);
        assert_eq!(actual.get_partition(2).end, 73);

        assert_eq!(actual.get_partition(3).start, 74);
        assert_eq!(actual.get_partition(3).end, 97);

        actual.get_partition(4);
    }

    #[test]
    fn test_parallelizable() {
        let tc1 = 1000;
        let tc = Partitioner::with_partitions(tc1, 8);

        let actual = tc.parallelized(|partition| {
            let partition_values = (partition.get_start()..=partition.get_end()).collect();
            partition_values
        });

        let expected:Vec<_> = (0..tc1).collect();

        assert_eq!(actual, expected);
    }
}