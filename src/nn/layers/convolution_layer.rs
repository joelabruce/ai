use crate::geoalg::f32_math::tensor::Tensor;

use super::{Matrix, Propagates};

///
pub struct ConvolutionalLayer {
    pub kernel: Tensor,
    pub biases: Matrix,
    pub stride: usize
}

impl ConvolutionalLayer {
    
}

impl Propagates for ConvolutionalLayer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        inputs.len();
        todo!()
    }

    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        dvalues.len();
        inputs.len();
        todo!()
    }
}