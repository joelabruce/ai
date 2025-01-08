use std::f64::consts::E;
use crate::geoalg::f64_math::matrix::Matrix;
use crate::nn::layers::*;

pub struct Activation {
    pub f: fn(&f64) -> f64,
    pub d: fn(&f64) -> f64
}

impl Activation {
    pub fn forward(&self, layer: &HiddenLayer, inputs: &Matrix) -> Matrix {
        layer.weights
            .mul(&inputs)
            .add_column_vector(&layer.biases)
    }

    pub fn backward<'a>(&self, mut layer: &'a HiddenLayer, inputs: &'a Matrix, dvalues: &'a Matrix) -> Matrix {
        let t = inputs.get_transpose();
        
        inputs.clone()
    }
}

pub const SIGMOID: Activation = Activation {
    f: |x| 1.0 / (1.0 - E.powf(-(x))),
    d: |x| (x) * (1.0 - (x))
};

pub const RELU: Activation = Activation {
    f: |x| if *x <= 0.0 { 0.0 } else { *x },
    d: |x| if *x <= 0.0 { 0.0 } else { 1.0 }
};

pub const H_SWISH: Activation = Activation {
    f: |x| {
        if *x <= -3.0                   { 0.0 }
        else if -3.0 < *x && *x < 3.0   { x * (x + 3.0) / 6.0 }
        else                            { *x }
    },
    d: |x| {
        if *x <= -3.0                   { 0.0 }
        else if -3.0 < *x && *x < 3.0   { (2.0 * x + 3.0) / 6.0 }
        else                            { 1.0 }  
    }
};

pub struct FoldActivation {
    pub f: fn(&Matrix) -> Matrix,
    pub d: fn(&Matrix) -> Matrix
}

pub const SOFTMAX: FoldActivation = FoldActivation {
    f: |m| {
        let t_mat = m.get_transpose();

        let mut r = Vec::with_capacity(m.get_element_count());

        for row in 0..t_mat.rows {
            let v = t_mat.get_row_vector_slice(row).to_vec();
            let max = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();

            let exp: Vec<f64> = v.iter().map(|x| E.powf(*x - max)).collect();
            let den: f64 = exp.iter().sum();
            
            r.extend(exp.iter().map(|x| x / den));
        }

        let r = Matrix {
            rows: m.columns,
            columns: m.rows,
            values: r
        };

        r.get_transpose()
    },
    d: |v| {
        //v.to_vec()
        Matrix::new_zeroed(v.columns, v.rows)
    }
};

/// Calculates the cross-entropy (used with softmax) for each input sample.
pub fn forward_categorical_cross_entropy_loss(predictions: &Matrix, expected: &Matrix) -> Vec<f64> {
    let t = expected.get_transpose();

    let mut r = Vec::with_capacity(t.rows);
    for row in 0..t.rows {
        let (index, _) = t.get_row_vector_slice(row)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();

        let loss = -predictions.get(index, row).unwrap().log10();

        r.push(loss);
    }

    r
}

/// Error, same as softmax derivative??
pub fn backward_categorical_cross_entropy_loss(predictions: &Matrix, expected: &Matrix, samples: usize) -> Matrix {
    predictions.sub(&expected).div_by_scalar(samples as f64)
}

#[cfg(test)]
mod tests {
    //use super::*;

    #[test]
    fn softmax_tes() {
        // let v = vec![1.0, 2.0, 3.0];
        // let actual = (SOFTMAX.function)(&v);

        // // [2.7182818284590452353602874713527, 7.389056098930650227230427460575, 20.085536923187667740928529654582]
        // // 30.192874850577363203519244586509
        // let expected = vec![
        //     0.09003057317038045799802210148449,
        //     0.24472847105479765247295961834077,
        //     0.66524095577482188952901828017475
        // ];

        // println!("{:?} v {:?}", actual, expected);
    }
}