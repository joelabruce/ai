use rand_distr::Uniform;

use crate::geoalg::f32_math::shape::Shape;
use super::*;

/// A fully connected layer.
/// Assumes tensor shape to be flattened to (batch size, data)
/// Should a reshape be put in here to guarantee this?
#[derive(Debug)]
pub struct Dense {
    pub weights: Tensor,
    pub biases: Tensor,
}

impl Dense {
    /// Initializes using He initialization.
    /// Might consider allowing for more flexibility here, but have to think carefuly about the cleanest way to do this.
    fn random_weight_biases(features: usize, neuron_count: usize) -> (Tensor, Tensor) {
        //let std_dev = 2. / (prev_layer_input_count as f32);
        //let normal = Normal::new(0., std_dev).unwrap();
        //let weights = Matrix::new_randomized_normal(prev_layer_input_count, neuron_count, normal);
        
        let term = (6. / (features as f32)).sqrt();
        let uniform = Uniform::new_inclusive(-term, term);
        let weights = Tensor::new_randomized_uniform(Shape::d2(features, neuron_count), uniform);
        let biases = Tensor::matrix(1, neuron_count, vec![0.0; neuron_count]);

        (weights, biases)
    }

    pub fn new(features: usize, neuron_count: usize) -> Dense {
        let (weights, biases) = Dense::random_weight_biases(features, neuron_count);

        Dense { weights, biases }
    }

    /// Instantiates and returns a new Hidden Layer based off self's shape.
    pub fn feed_into_dense(&self, neuron_count: usize) -> Dense {
        //assert!(self.weights.shape.len() > 0);
        //assert!(self.biases.len() > 0);

        Dense::new(self.biases.shape.size(), neuron_count)
    }
}

impl LayerPropagates for Dense {
    /// Forward propagates by performing weights dot inputs + biases.
    fn forward<'a>(&mut self, inputs: &'a Tensor) -> Tensor {
        let r = inputs
            .mul_transpose_simd(&self.weights.transpose())
            .broadcast_vector_add(&self.biases);
            //.mul_with_transpose(&self.weights.transpose())
            //.add_row_partitioned(&self.biases);
        r
    }

    /// Back propagates.
    /// Will return dvalues to be used in next back propagation layer.
    fn backward<'a>(& mut self, learning_rate: &mut LearningRate, dvalues: &Tensor, inputs: &Tensor) -> Tensor {
        // Mutate the weights based on derivative weights
        let inputs_t = inputs.transpose();
        let dweights = inputs_t.mul_transpose_simd(&dvalues.transpose());

        self.weights = self.weights.sub_element_wise_simd(&dweights.scale_simd(learning_rate.rate()));

        // Mutate the biases based on derivative biases
        let dbiases = dvalues.reduce_vector_add();
        self.biases = self.biases.sub_element_wise_simd(&dbiases.scale_simd(learning_rate.rate()));

        let result = dvalues.mul_transpose_simd(&self.weights);
        result
    }
}

#[cfg(test)]
mod test {
    //use crate::{geoalg::f32_math::matrix::Matrix, nn::{layers::Propagates, learning_rate::LearningRate}};

    use rand_distr::Uniform;

    use super::*;
    use crate::{geoalg::f32_math::{shape::Shape, tensor::Tensor}, nn::learning_rate::LearningRate};

    #[test]
    fn test_propagations() {
        let features = 784;
        let neuron_count = 128;

        let mut dense = Dense::new(features, neuron_count);

        let batches = 500;
        let uniform = Uniform::new_inclusive(-1., 1.);
        let inputs = &Tensor::new_randomized_uniform(Shape::d2(batches, features), uniform);

        let _forward = &dense.forward(inputs);
        //println!("{BRIGHT_YELLOW}{:?}{RESET}", forward.shape());

        let learning_rate = &mut LearningRate::new(0.01);
        let uniform = Uniform::new_inclusive(-1., 1.);
        let dvalues = &&Tensor::new_randomized_uniform(Shape::d2(batches, neuron_count), uniform);
        let _dz = dense.backward(learning_rate, dvalues, inputs);
    }

    #[test]
    fn test_feed_into_dense() {
        let features = 784;
        let neuron_count_1 = 128;
        let mut dense1 = Dense::new(features, neuron_count_1);

        let neuron_count_2 = 64;
        let mut dense2 = dense1.feed_into_dense(neuron_count_2);

        let batches = 500;
        let uniform = Uniform::new_inclusive(-1., 1.);
        let inputs = &Tensor::new_randomized_uniform(Shape::d2(batches, features), uniform);

        let forward1 = &dense1.forward(inputs);

        let _forward2 = &dense2.forward(forward1);

        let learning_rate = &mut LearningRate::new(0.01);
        let uniform = Uniform::new_inclusive(-1., 1.);
        let dvalues = &Tensor::new_randomized_uniform(Shape::d2(batches, neuron_count_2), uniform);

        let dz2 = &dense2.backward(learning_rate, dvalues, forward1);

        let _dz1 = dense1.backward(learning_rate, dz2, inputs);
    }
}