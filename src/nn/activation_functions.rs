use std::f32::consts::E;
use crate::geoalg::f32_math::matrix::Matrix;


/// Argmax. Returns index and value at the index.
pub fn vector_row_max(values: &[f32]) -> (usize, f32) {
    let mut max = values[0];
    let mut index = 0;
    for i in 1..values.len() {
        if values[i] > max { 
            max = values[i];
            index = i;
        }
    }

    (index, max)
}

/// Calculates the cross-entropy (used with softmax) for each input sample.
pub fn forward_categorical_cross_entropy_loss(predictions: &Matrix, expected: &Matrix) -> Matrix {
    let t: Matrix = predictions.mul_element_wise_simd(expected);
    let mut r = Vec::with_capacity(t.row_count());
    for row in 0..t.row_count() {
        let loss = -t.row(row).iter().sum::<f32>().log10();
        r.push(loss);
    }

    Matrix::new(1, t.row_count(), r)
}

/// Softmax with categorical cross entropy loss derivative 
pub fn backward_categorical_cross_entropy_loss_wrt_softmax(predictions: &Matrix, expected: &Matrix) -> Matrix {
    predictions.sub(&expected)
}

pub struct FoldActivation {
    pub f: fn(&Matrix) -> Matrix,
    pub d: fn(&Matrix) -> Matrix
}

pub const SOFTMAX: FoldActivation = FoldActivation {
    f: |m| {
        let mut values = Vec::with_capacity(m.len());

        for row in 0..m.row_count() {
            let v = m.row(row);
            let max = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();

            let exp_numerators: Vec<f32> = v.iter().map(|&x| E.powf(x - max)).collect();
            let denominator: f32 = exp_numerators.iter().sum();
            
            values.extend(exp_numerators.iter().map(|x| x / denominator));
        }

        Matrix::new(m.row_count(), m.column_count(), values)
    },
    d: |v| {
        v.clone()
    }
};

/// Calculats accurace given predicted and expected values
pub fn accuracy(predicted: &Matrix, expected: &Matrix) -> f32 {
    assert_eq!(predicted.row_count(), expected.row_count());
    assert_eq!(predicted.column_count(), expected.column_count());

    let mut matches = 0.;
    for row in 0..predicted.row_count() {
        let (actual_i, _) = vector_row_max(predicted.row(row));
        let (expected_i, _) = vector_row_max(expected.row(row));
        if actual_i == expected_i { matches += 1. }
    }

    matches / predicted.row_count() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_row_max() {
        let tc = vec![2., 5., 7., 3.2, 1.7];
        let (i, m) = vector_row_max(&tc);

        assert_eq!(i, 2);
        assert_eq!(m, 7.);
    }

    #[test]
    fn softmax_tes() {
        let v = vec![1.0, 2.0, 3.0];
        let mat = Matrix::new(1, 3, v);
        let actual = (SOFTMAX.f)(&mat);

        let expected = Matrix::new(1, 3, vec![
            0.09003058,
            0.24472848,
            0.66524094            
        ]);

        assert_eq!(actual, expected);
    }
}