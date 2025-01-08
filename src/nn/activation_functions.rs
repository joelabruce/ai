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

    pub fn backward<'a>(&self, inputs: &'a Matrix) -> Matrix {
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

            let nums: Vec<f64> = v.iter().map(|x| E.powf(*x - max)).collect();
            let den: f64 = nums.iter().sum();
            
            r.extend(nums.iter().map(|x| x / den));
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