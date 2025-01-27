use super::{Matrix, Propagates};

///
pub struct ConvolutionalLayer {
    pub kernel: Matrix,
    pub biases: Matrix,
    pub stride: usize
}

impl ConvolutionalLayer {
    //fn 
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