use rand::Rng;

pub struct Sample<T> {
    all_samples: Vec<T>,
    unused: Vec<usize>,
    used: Vec<usize>,
}

impl<T> Sample<T> {
    /// Creates a sample of data from specified vector.
    /// Make sure to move the data into sample for it to be consumed.
    /// Requires no expensive shuffling since the data load is likely already expensive.
    pub fn create_sample(data: Vec<T>) -> Sample<T> {
        let unused = (0..data.len()).collect();
        let used = Vec::with_capacity(data.len());
        let all_samples = data;

        Sample {
            all_samples,
            unused,
            used
        }
    }

    /// Creates a random batch from sample data.
    /// Repeated calls to random_batch are guaranteed to never uses the same element more than once before call to reset.
    /// Must call reset once the sample is exhausted to use in a new series of batches.
    /// Data is owned by sample, and should only ever be immutably borrowed.
    pub fn random_batch(&mut self, requested_batch_size: usize) -> Vec<&T> {
        // Ensure batch size picks up stragglers
        let batch_size = std::cmp::min(requested_batch_size, self.unused.len());

        let mut rng = rand::thread_rng();
        let mut newbatch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let unused_index = rng.gen_range(0..self.unused.len());
            let index_to_use = self.unused.remove(unused_index);
            self.used.push(index_to_use);

            let data_reference = &self.all_samples[index_to_use];
            newbatch.push(data_reference);
        }

        newbatch
    }

    /// Resets batch sample to allow a fresh series of random batches.
    pub fn reset(&mut self) {
        self.unused.extend(self.used.iter());
        self.used.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch() {
        let mut sample = Sample::create_sample( 
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        );

        for reset in 0..5 {
            println!("TC1 Reset: {reset}");
            for _ in 0..6 {
                let _new_batch = sample.random_batch(2);
                
                println!("sample: {:?}", _new_batch);
            }

            sample.reset();
        }

        for reset in 0..4 {
            println!("TC2 Reset: {reset}");
            for _ in 0..4 {
                let _new_batch = sample.random_batch(3);
                
                println!("sample: {:?}", _new_batch);
            }

            sample.reset();
        }
    }
}