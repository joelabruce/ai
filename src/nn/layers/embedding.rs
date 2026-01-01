pub struct Embedding {

}

impl Propagates for Embedding {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {

    }

    fn backward<'a>(&'a mut self, learning_rate: &mut LearningRate, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
    }
}