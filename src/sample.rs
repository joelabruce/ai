use rand::Rng;

pub struct Sample<T> {
    all_samples: Vec<T>,
    unused: Vec<usize>,
    consumed: Vec<usize>,
}

impl<T> Sample<T> {
    /// Creates a sample of data from specified vector.
    pub fn create_sample(data: Vec<T>) -> Sample<T> {
        let unused = data.iter().enumerate().map(|(i, _)| i).collect();
        let all_samples = data;
        let consumed = Vec::with_capacity(all_samples.len());

        Sample {
            all_samples,
            unused,
            consumed
        }
    }

    /// Creates a random batch from sample data.
    pub fn random_batch(&mut self, requested_batch_size: usize) -> Vec<&T> {
        // Ensure batch size picks up stragglers
        let batch_size = std::cmp::min(requested_batch_size, self.unused.len());

        let mut rng = rand::thread_rng();

        let mut newbatch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let index = rng.gen_range(0..self.unused.len());
            let img_index = self.unused.remove(index);
            self.consumed.push(img_index);

            let img = &self.all_samples[img_index];
            newbatch.push(img);
        }

        newbatch
    }

    /// Resets batch.
    pub fn reset(&mut self) {
        self.unused.extend(self.consumed.iter());
        self.consumed = Vec::with_capacity(self.unused.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch() {
        let mut sample = Sample::create_sample( 
            vec!["a", "b", "c", "d", "e", "f", "g"]
        );

        for _ in 0..4 {
            let _new_batch = sample.random_batch(2);
            
            //println!("batch: {:?}", _new_batch);
        }

        sample.reset();
        for _ in 0..3 {
            let _new_batch = sample.random_batch(3);
            
            //println!("batch: {:?}", _new_batch);
        }
    }
}