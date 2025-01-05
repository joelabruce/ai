use crate::geoalg::f64_math::matrix::*;
use crate::nn::activation_functions::*;

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Network {
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

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .mul_using_transpose(&current)
                .add(&self.biases[i])
                .map(self.activation.function);

            self.data.push(current.clone());
        }

        current
    }

    pub fn back_propogate(&mut self, inputs: Matrix, targets: Matrix) {
        let mut errors = targets.sub(&inputs);

        let mut gradients = inputs.clone().map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.elementwise_multiply(&errors)
                .map(|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.mul_using_transpose(&self.data[i].get_transpose()));

            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].get_transpose().mul_using_transpose(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for i in 1..=epochs {
            println!{"Processing epoch {i}"};

            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propogate(outputs, Matrix::from(targets[j].clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "reason"]
    #[test]
    fn test() {
        let inputs = vec![
            vec![0f64, 0f64],
            vec![0f64, 1f64],
            vec![1f64, 0f64],
            vec![1f64, 1f64]
        ];

        let targets = vec![
            vec![0f64], vec![1f64], vec![1f64], vec![0f64]
        ];

        let mut network = Network::new(vec![2,3,1], RELU, 0.01f64);

        network.train(inputs, targets, 100000);

        println!("{:?}", network.feed_forward(Matrix::from(vec![0.0f64, 0.0f64])));
        println!("{:?}", network.feed_forward(Matrix::from(vec![0.0f64, 1.0f64])));
        println!("{:?}", network.feed_forward(Matrix::from(vec![1.0f64, 0.0f64])));
        println!("{:?}", network.feed_forward(Matrix::from(vec![1.0f64, 1.0f64])));
    }
}