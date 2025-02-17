pub struct LearningRate {
    rate: f32
}

impl LearningRate {
    /// Starting point for the learning rate.
    pub fn new(rate: f32) -> Self { Self { rate } }

    /// Return the current learning rate.
    pub fn rate(&self) -> f32 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_rate() {
        // For now learning rate is constant, but looking into ability to adapt it later.
        let learning_rate = LearningRate::new(0.1);
        assert_eq!(learning_rate.rate(), 0.1);
    }
}