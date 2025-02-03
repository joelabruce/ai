pub mod dense_layer;
pub mod input_layer;
pub mod convolution_layer;

extern crate rand;
use crate::geoalg::f32_math::matrix::*;

/// Learning rate to affect how layers apply back-propagation.
pub fn learning_rate() -> f32 {
    0.01
}

pub trait Propagates {
    fn forward(&mut self, inputs: &Matrix) -> Matrix;
    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix;
}

#[cfg(test)]
mod tests {
    #[test]
    fn testflow() {

    }
}