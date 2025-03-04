
use crate::geoalg::f32_math::{matrix::Matrix, simd_extensions::{d_relu_simd, relu_simd, SliceExt}};

pub struct Activation {
    f: fn(&Matrix) -> Matrix,
    b: fn(&Matrix, &Matrix) -> Matrix
}

impl Activation {
    pub fn forward(&self, inputs: &Matrix) -> Matrix {
        (self.f)(inputs)
    }

    pub fn backward(&self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        (self.b)(dvalues, inputs)
    }
}

//#[allow(unused)]
pub const RELU: Activation = Activation {
    f: |inputs| -> Matrix {
        let values = relu_simd(inputs.read_values());
        Matrix::new(inputs.row_count(), inputs.column_count(), values)
    },
    b: |dvalues, inputs| -> Matrix {
        let values = (*d_relu_simd(inputs.read_values())).mul_simd(dvalues.read_values());
        Matrix::new(dvalues.row_count(), dvalues.column_count(), values)
    }
};