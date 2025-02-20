use crate::{geoalg::f32_math::{shape::Shape, tensor::Tensor}, nn::learning_rate::LearningRate};

use super::{dense::Dense, LayerPropagates};

pub struct Reshaper {
    source_shape: Shape,
    destination_shape: Shape
}

impl Reshaper {
    /// Assume first dimension is a place-holder for batch_size for source and destination shapes.
    /// This assumption means instead of comparing size of shapes, we only compare the stride up to first element.
    pub fn new(source_shape: &Shape, destination_shape: &Shape) -> Self {
        assert_eq!(source_shape.stride_for(0), destination_shape.stride_for(0));

        Reshaper { source_shape: source_shape.clone(), destination_shape: destination_shape.clone() }
    }

    /// Assume since this is feeding into dense, that there are only 2 dimensions.
    /// Use this info to know how to construct the dense layer.
    pub fn feed_into_dense(&self, neuron_count: usize) -> Dense {
        let features = self.destination_shape[1];
        Dense::new(features, neuron_count)
    }
}

impl LayerPropagates for Reshaper {
    fn forward(&mut self, inputs: &Tensor) -> Tensor {
        let input_shape_dims = &mut self.destination_shape.dims().to_vec();
        input_shape_dims[0] = inputs.shape[0];

        Tensor::new(Shape::new(input_shape_dims.clone()), inputs.stream().to_vec())
    }

    #[allow(unused_variables)]
    fn backward<'a>(&'a mut self, learning_rate: &mut LearningRate, dvalues: &Tensor, inputs: &Tensor) -> Tensor {
        let input_shape_dims = &mut self.source_shape.dims().to_vec();
        input_shape_dims[0] = inputs.shape[0];

        Tensor::new(Shape::new(input_shape_dims.clone()), inputs.stream().to_vec())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_reshape() {

    }
}