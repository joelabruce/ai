use crate::geoalg::f32_math::tensor::Tensor;

pub struct Activation {
    f: fn(&Tensor) -> Tensor,
    b: fn(&Tensor, &Tensor) -> Tensor
}

pub trait ActivationPropagates {
    fn forward(&self, inputs: &Tensor) -> Tensor;
    fn backward(&self, dvalues: &Tensor, inputs: &Tensor) -> Tensor;
}

impl ActivationPropagates for Activation {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        (self.f)(inputs)
    }

    #[allow(unused_variables)]
    fn backward(&self, dvalues: &Tensor, inputs: &Tensor) -> Tensor {
        (self.b)(dvalues, inputs)
    }
}

pub const RELU: Activation = Activation {
    f: |inputs| -> Tensor {
        inputs.relu_simd()
    },
    b: |dvalues, inputs| -> Tensor {
        inputs.d_relu_simd().mul_element_wise_simd(dvalues)
    }
};

#[cfg(test)]
mod tests {
    use crate::geoalg::f32_math::tensor::Tensor;

    use super::*;

    #[test]
    fn test_relu_forward_and_back() {
        let inputs = &Tensor::matrix(3, 3, vec![
            -1., 2., 3.,
            -4., -5., 6.,
            -7., 8., -9.
        ]);

        let _forward = RELU.forward(inputs);
    }
}