use crate::geoalg::f32_math::matrix::*;

fn relu(x: f32) -> f32 {
    if x > 0f32 {x} else {0f32}
}

fn sigmoid(x: f32) -> f32 {
    0f32
}
    
pub struct Activation {
    x: f32
}

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f32
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f32) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::new_randomized(layers[i], layers[i+1]));
            biases.push(Matrix::new_randomized(1,layers[i+1]));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate
        }
    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert_eq!(self.layers[0], inputs.values.len());

        let mut current = inputs;
        self.data = vec![current.clone()];

        Matrix::new_zeroed(1, 2)
    }
}