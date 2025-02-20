// use std::f32::consts::E;
// use crate::geoalg::f32_math::matrix::Matrix;
// use crate::geoalg::f32_math::optimized_functions::vector_row_max;
// use crate::geoalg::f32_math::tensor::Tensor;
// use crate::nn::layers::*;

// use super::learning_rate::LearningRate;

// pub struct Activation {
//     pub f: fn(&f32) -> f32,
//     pub d: fn(&f32) -> f32
// }

// impl Propagates for Activation {
//     fn forward(&mut self, inputs: &Tensor) -> Tensor {
//         inputs.map(self.f)
//     }

//     #[allow(unused_variables)]
//     fn backward<'a>(&mut self, learning_rate: &mut LearningRate, dvalues: &'a Matrix, inputs: &'a Matrix) -> Matrix {
//         assert_eq!(dvalues.row_count(), inputs.row_count(), "Backpropagation for Activation needs inputs and dvalues to have same rows.");
//         assert_eq!(dvalues.column_count(), inputs.column_count(), "Backpropagation for Activation needs inputs and dvalues to have same columns.");

//         inputs.map(self.d).mul_element_wise_simd(&dvalues)
//     }
// }

// /// Sigmoid Activation.
// pub const SIGMOID: Activation = Activation {
//     f: |x| 1.0 / (1.0 - E.powf(-(x))),
//     d: |x| (x) * (1.0 - (x))
// };

// /// ReLU (Rectified Linear Unit) Activation.
// pub const RELU: Activation = Activation {
//     f: |&x| if x <= 0.0 { 0.0 } else { x },
//     d: |&x| if x <= 0.0 { 0.0 } else { 1.0 }
// };

// /// H-Switsh Activation.
// pub const H_SWISH: Activation = Activation {
//     f: |&x| {
//         if x <= -3.0                   { 0.0 }
//         else if -3.0 < x && x < 3.0    { x * (x + 3.0) / 6.0 }
//         else                           { x }
//     },
//     d: |&x| {
//         if x <= -3.0                   { 0.0 }
//         else if -3.0 < x && x < 3.0    { (2.0 * x + 3.0) / 6.0 }
//         else                           { 1.0 }  
//     }
// };


use std::f32::consts::E;

use crate::geoalg::f32_math::{optimized_functions::vector_row_max, tensor::Tensor};


// /// Calculates the cross-entropy (used with softmax) for each input sample.
pub fn forward_categorical_cross_entropy_loss(predictions: &Tensor, expected: &Tensor) -> Tensor {
    let t: Tensor = predictions.mul_element_wise_simd(expected);
    let mut r = Vec::with_capacity(t.shape[0]);
    for row in 0..t.shape[0] {
        let loss = -t.dim_slice(0,row).iter().sum::<f32>().log10();
        r.push(loss);
    }

    Tensor::matrix(1, t.shape[0], r)
}

// /// Softmax with categorical cross entropy loss derivative 
pub fn backward_categorical_cross_entropy_loss_wrt_softmax(predictions: &Tensor, expected: &Tensor) -> Tensor {
    predictions.sub_element_wise_simd(&expected)
}

pub struct FoldActivation {
    pub f: fn(&Tensor) -> Tensor,
    pub d: fn(&Tensor) -> Tensor
}

pub const SOFTMAX: FoldActivation = FoldActivation {
    f: |m| {
        let mut values = Vec::with_capacity(m.shape.size());

        for row in 0..m.shape[0] {
            let v = m.dim_slice(0, row);
            let max = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();

            let exp_numerators: Vec<f32> = v.iter().map(|&x| E.powf(x - max)).collect();
            let denominator: f32 = exp_numerators.iter().sum();
            
            values.extend(exp_numerators.iter().map(|x| x / denominator));
        }

        Tensor::matrix(m.shape[0], m.shape[1], values)
    },
    d: |_v| {
        todo!();
    }
};

/// Calculats accurace given predicted and expected values
pub fn accuracy(predicted: &Tensor, expected: &Tensor) -> f32 {
    assert_eq!(predicted.shape[0], expected.shape[0]);
    assert_eq!(predicted.shape[1], expected.shape[1]);

    let mut matches = 0.;
    for row in 0..predicted.shape[0] {
        let (actual_i, _) = vector_row_max(predicted.dim_slice(0,row));
        let (expected_i, _) = vector_row_max(expected.dim_slice(0, row));
        if actual_i == expected_i { matches += 1. }
    }

    matches / predicted.shape[0] as f32
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn softmax_tes() {
//         let v = vec![1.0, 2.0, 3.0];
//         let mat = Matrix::from(1, 3, v);
//         let actual = (SOFTMAX.f)(&mat);

//         let expected = Matrix::from(1, 3, vec![
//             0.09003058,
//             0.24472848,
//             0.66524094            
//         ]);

//         assert_eq!(actual, expected);
//     }

//     #[test]
//     fn relu() {
//         let tc1 = -3.0;
//         let actual = (RELU.f)(&tc1);
//         let expected = 0.0;
//         assert_eq!(actual, expected);

//         let actual = (RELU.d)(&tc1);
//         let expected = 0.;
//         assert_eq!(actual, expected);

//         let tc2 = 7.2;
//         let actual = (RELU.f)(&tc2);
//         let expected = 7.2;
//         assert_eq!(actual, expected);

//         let actual = (RELU.d)(&tc2);
//         let expected = 1.;
//         assert_eq!(actual, expected);
//     }
// }