pub struct Tensor {
    pub values: Vec<f64>,
    pub dimensions: Vec<usize>
}

impl Tensor {
    pub fn new(values: Vec<f64>, shape: Vec<usize>) {
        let zero = 0;
        let no_zeroes = !shape.contains(&zero);
        assert!(no_zeroes);

        let len = shape.iter().product();
        assert_eq!(values.len(), len);
    }
}