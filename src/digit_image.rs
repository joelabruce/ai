use crate::geoalg::f32_math::optimized_functions::*;

/// Converts label into a vector.
/// If label is greater than bounds, than vector will be zeroes.
pub fn one_hot_encode(label: f32, bounds: usize) -> Vec<f32> {
    (0..bounds)
        .map(|x| kronecker_delta_f32(label, x as f32))
        .collect()
}

/// Greyscale 28x28 image that can be 0 through 9
pub struct GreyscaleDigitImage {
    pub pixels: Vec<f32>,
    pub label: f32
}

impl GreyscaleDigitImage {
    pub fn one_hot_encoded_label(&self) -> Vec<f32> {
        one_hot_encode(self.label, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::GreyscaleDigitImage;

    #[test]
    fn test_one_hot_encode() {
        let tc = GreyscaleDigitImage {
            pixels: vec![],
            label: 3.0
        };

        let actual = tc.one_hot_encoded_label();
        let expected = vec![0.,0.,0.,1.,0.,0.,0.,0.,0.,0.];

        assert_eq!(actual, expected);
    }
}