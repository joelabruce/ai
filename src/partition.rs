use std::ops::RangeInclusive;

/// A partition to be used when processing a subset of data
#[derive(Hash, Debug, Clone, PartialEq, Copy)]
pub struct Partition {
    start: usize,
    end: usize
}

impl Partition {
    pub fn new(start: usize, end: usize) -> Self {
        assert!(start <= end);
        Self { start, end }
    }

    pub fn get_start(&self) -> usize { self.start }
    pub fn get_end(&self) -> usize { self.end }

    /// Returns size of the partition.
    pub fn size(&self) -> usize { self.end - self.start + 1 }

    /// Creates a range to work with when processing data for the partition.
    pub fn range(&self) -> RangeInclusive<usize> { self.start..=self.end }
}


#[cfg(test)]
mod tests {
    use crate::partitioner::Partitioner;

    //use super::*;

    #[test]
    fn test_partition_sizes() {
        let tc1_count = 98;
        let partition_count = 4;
        let actual = Partitioner::with_partitions(tc1_count, partition_count);

        assert_eq!(actual[0].start, 0);
        assert_eq!(actual[0].range(), (0..=24));
        assert_eq!(actual[0].end, 24);
        assert_eq!(actual[0].size(), 25);

        assert_eq!(actual[1].start, 25);
        assert_eq!(actual[1].end, 49);
        assert_eq!(actual[1].range(), (25..=49));
        assert_eq!(actual[1].size(), 25);

        assert_eq!(actual[2].start, 50);
        assert_eq!(actual[2].end, 73);
        assert_eq!(actual[2].range(), (50..=73));
        assert_eq!(actual[2].size(), 24);

        assert_eq!(actual[3].start, 74);
        assert_eq!(actual[3].end, 97);
        assert_eq!(actual[3].range(), (74..=97));
        assert_eq!(actual[3].size(), 24);
    }

    #[test]
    #[should_panic]
    fn test_panic_on_invalid_partition() {
        let tc1_count = 98;
        let partition_count = 4;
        let actual = Partitioner::with_partitions(tc1_count, partition_count);

        actual[4];        
    }
}