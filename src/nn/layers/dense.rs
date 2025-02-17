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
    pub fn influences_dense(&self, neuron_count: usize) -> Dense {
        //assert!(self.weights.shape.len() > 0);
        //assert!(self.biases.len() > 0);

        Dense::new(self.biases.shape.size(), neuron_count)
    }
}

impl Propagates for Dense {
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
mod tests {
    use rand_distr::Normal;

    use crate::{geoalg::f32_math::{shape::Shape, tensor::Tensor}, nn::{layers::Propagates, learning_rate::{self, LearningRate}}};

    use super::Dense;

    use crate::prettify::*;

    #[test]
    fn test_propagations() {
        let features = 10;
        let neuron_count = 6;
        let mut dense = Dense::new(features, neuron_count);
        
        let batch_size = 200;
        let inputs= &Tensor::matrix(batch_size, features, vec![1. ; batch_size * features]);
        
        let forward = dense.forward(inputs);
        //assert_eq!(forward.shape[0], batch_size);
        //assert_eq!(forward.shape[1], neuron_count);

        let mut learning_rate = LearningRate::new(0.1);
        let normal = Normal::new(0., 1.).unwrap();
        let dz = Tensor::new_randomized_normal(
            Shape::d2(batch_size, neuron_count * features),
            normal);
        let backward = dense.backward(&mut learning_rate, &forward, &forward);
        //assert_eq!(backward.shape[0], batch_size);
        //assert_eq!(backward.shape[1], features);
    }

    #[test]
    fn test_influences_dense() {
        let features = 78;
        let neuron_count_1 = 10;
        let mut dense = Dense::new(features, neuron_count_1);

        let neuron_count_2 = 64;
        let mut dense_next = dense.influences_dense(neuron_count_2);

        assert_eq!(dense_next.weights.shape[0], neuron_count_1);
        assert_eq!(dense_next.weights.shape[1], neuron_count_2);

        let batch_size = 1000;
        let inputs= &Tensor::matrix(batch_size, features, vec![1. ; batch_size * features]);

        let forward1 = dense.forward(&inputs);
        let forward2 = dense_next.forward(&forward1);

        let learning_rate = &mut LearningRate::new(0.01);
        //let back2 = dense_next.backward(learning_rate, &forward2);
    }
}