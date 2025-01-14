use crate::geoalg::f64_math::matrix::one_hot_encode;

pub struct DigitImage {
    pub pixels: Vec<f64>,
    pub label: f64
}

impl DigitImage {
    pub fn one_hot_encoded_label(&self) -> Vec<f64> {
        one_hot_encode(self.label, 10)
    }
}