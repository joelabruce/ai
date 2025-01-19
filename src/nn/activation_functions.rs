use std::f64::consts::E;
use crate::geoalg::f64_math::matrix::Matrix;
use crate::nn::layers::*;

pub struct Activation {
    pub f: fn(&f64) -> f64,
    pub d: fn(&f64) -> f64
}

impl Activation {

}

impl Propagates for Activation {
    fn forward(&self, inputs: &Matrix) -> Matrix {
        inputs.map(self.f)
    }
    fn backward<'a>(&mut self, dvalues: &'a Matrix, inputs: &'a Matrix) -> Matrix {
        assert_eq!(dvalues.rows, inputs.rows, "Backpropagation for Activation needs inputs and dvalues to have same rows.");
        assert_eq!(dvalues.columns, inputs.columns, "Backpropagation for Activation needs inputs and dvalues to have same columns.");

        inputs.map(self.d).elementwise_multiply_threaded(&dvalues)
    }
}

/// Sigmoid Activation.
pub const SIGMOID: Activation = Activation {
    f: |x| 1.0 / (1.0 - E.powf(-(x))),
    d: |x| (x) * (1.0 - (x))
};

/// ReLU (Rectified Linear Unit) Activation.
pub const RELU: Activation = Activation {
    f: |&x| if x <= 0.0 { 0.0 } else { x },
    d: |&x| if x <= 0.0 { 0.0 } else { 1.0 }
};

/// H-Switsh Activation.
pub const H_SWISH: Activation = Activation {
    f: |&x| {
        if x <= -3.0                   { 0.0 }
        else if -3.0 < x && x < 3.0    { x * (x + 3.0) / 6.0 }
        else                           { x }
    },
    d: |&x| {
        if x <= -3.0                   { 0.0 }
        else if -3.0 < x && x < 3.0    { (2.0 * x + 3.0) / 6.0 }
        else                           { 1.0 }  
    }
};

/// Calculates the cross-entropy (used with softmax) for each input sample.
pub fn forward_categorical_cross_entropy_loss(predictions: &Matrix, expected: &Matrix) -> Matrix {
    let t: Matrix = predictions.elementwise_multiply_threaded(expected);
    let mut r = Vec::with_capacity(t.rows);
    for row in 0..t.rows {
        let loss = -t.get_row_vector_slice(row).iter().sum::<f64>().log10();
        r.push(loss);
    }

    Matrix::from_vec(r, 1, t.rows)
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
        let mut r = Vec::with_capacity(m.len());

        for row in 0..m.rows {
            let v = m.get_row_vector_slice(row);
            let max = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();

            let exp_numerators: Vec<f64> = v.iter().map(|x| E.powf(*x - max)).collect();
            let denominator: f64 = exp_numerators.iter().sum();
            
            r.extend(exp_numerators.iter().map(|x| x / denominator));
        }

        Matrix::from_vec(r, m.rows, m.columns)
    },
    d: |v| {
        v.clone()
    }
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_tes() {
        let v = vec![1.0, 2.0, 3.0];
        let mat = Matrix::from_vec(v, 1, 3);
        let actual = (SOFTMAX.f)(&mat);

        let expected = Matrix::from_vec(vec![
            0.09003057317038046,
            0.24472847105479764,
            0.6652409557748218            
        ], 1, 3);

        assert_eq!(actual, expected);
    }

    #[test]
    fn relu() {
        let tc1 = -3.0;
        let actual = (RELU.f)(&tc1);
        let expected = 0.0;
        assert_eq!(actual, expected);

        let actual = (RELU.d)(&tc1);
        let expected = 0.;
        assert_eq!(actual, expected);

        let tc2 = 7.2;
        let actual = (RELU.f)(&tc2);
        let expected = 7.2;
        assert_eq!(actual, expected);

        let actual = (RELU.d)(&tc2);
        let expected = 1.;
        assert_eq!(actual, expected);
    }
}