use crate::geoalg::f64_math::optimized_functions::*;

/// Converts label into a vector.
/// If label is greater than bounds, than vector will be zeroes.
pub fn one_hot_encode(label: f64, bounds: usize) -> Vec<f64> {
    (0..bounds)
        .map(|x| kronecker_delta_f64(label, x as f64))
        .collect()
}

/// Greyscale 28x28 image that can be 0 through 9
pub struct DigitImage {
    pub pixels: Vec<f64>,
    pub label: f64
}

impl DigitImage {
    pub fn one_hot_encoded_label(&self) -> Vec<f64> {
        one_hot_encode(self.label, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::DigitImage;

    #[test]
    fn test_one_hot_encode() {
        let tc = DigitImage {
            pixels: vec![],
            label: 3.0
        };

        let actual = tc.one_hot_encoded_label();
        let expected = vec![0.,0.,0.,1.,0.,0.,0.,0.,0.,0.];

        assert_eq!(actual, expected);
    }
}