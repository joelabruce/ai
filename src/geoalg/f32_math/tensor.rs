pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>
}

impl Tensor {
    pub fn new(shape: Vec<usize>, values: Vec<f32>) -> Self {
        Tensor { shape, values }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn test_shape_contiguousness() {
        let _x = Tensor::new(vec![4, 3, 2], vec![
            1., 2.,
            3., 4.,
            5., 6.,

            10., 11.,
            12., 13.,
            14., 15.,

            100., 200.,
            300., 400.,
            500., 600.,

            -1., 0.4,
            32., 6.,
            23., 34.
        ]);
    }
}