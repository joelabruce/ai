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
        assert_eq!(dvalues.rows, inputs.rows, "Backpropagation RELU needs inputs and dvalues to have same rows.");
        assert_eq!(dvalues.columns, inputs.columns, "Backpropagation RELU needs inputs and dvalues to have same columns.");
        
        let values = dvalues.values
            .iter()
            .zip(inputs.values.clone())
            .map(|(x, y)| {
                if y <= 0.0 { 0.0 } else { *x }
            }).collect();

        Matrix {
            rows: dvalues.rows,
            columns: dvalues.columns,
            values
        }
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

/// Calculates the cross-entropy (used with softmax) for each input sample.
pub fn forward_categorical_cross_entropy_loss(predictions: &Matrix, expected: &Matrix) -> Matrix {
    let t: Matrix = predictions.elementwise_multiply(expected);
    let mut r = Vec::with_capacity(t.rows);
    for row in 0..t.rows {
        let loss = -t.get_row_vector_slice(row).iter().sum::<f64>().log10();
        r.push(loss);
    }

    Matrix {
        rows: 1,
        columns: t.rows,
        values: r
    }
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
        let mut r = Vec::with_capacity(m.get_element_count());

        for row in 0..m.rows {
            let v = m.get_row_vector_slice(row);
            let max = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();

            let exp_numerators: Vec<f64> = v.iter().map(|x| E.powf(*x - max)).collect();
            let denominator: f64 = exp_numerators.iter().sum();
            
            r.extend(exp_numerators.iter().map(|x| x / denominator));
        }

        let r = Matrix {
            rows: m.rows,
            columns: m.columns,
            values: r
        };

        r
    },
    d: |v| {
        //v.to_vec()
        Matrix::new_zeroed(v.rows, v.columns)
    }
};

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