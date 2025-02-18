use crate::{geoalg::f32_math::{shape::Shape, tensor::Tensor}, nn::learning_rate::LearningRate};
use super::{convolution2d::{Convolution2dDeprecated, Dimensions}, dense::Dense, LayerPropagates};

/// Currently only supports valid pooling layers, with no padding.
pub struct MaxPooling {
    filters: usize,
    /// Input dimensions
    i_d: Dimensions,
    /// Pooling dimensions
    p_d: Dimensions,    
    stride: usize,
    max_indices: Vec<usize>
}

impl MaxPooling {
    pub fn new(filters: usize, p_d: Dimensions, i_d: Dimensions, stride: usize) -> Self {
        MaxPooling {
            filters, p_d, i_d, stride, max_indices: vec![0]
        }
    }

    pub fn influences_dense(&self, neuron_count: usize) -> Dense {
        let (rows, columns) = self.output_dimensions().shape();

        let features = self.filters * rows * columns;
        Dense::new(features, neuron_count)
    }

    pub fn influences_convolution2d(&self, filters: usize, channels: usize, k_d: Dimensions) -> Convolution2dDeprecated {
        Convolution2dDeprecated::new(filters, channels, k_d, self.output_dimensions())
    }

    fn output_dimensions(&self) -> Dimensions {
        Dimensions {
            height: (self.i_d.height - self.p_d.height) / self.stride + 1,
            width: (self.i_d.width - self.p_d.width) / self.stride + 1
        }
    }
}

impl LayerPropagates for MaxPooling {
    fn forward(&mut self, inputs: &Tensor) -> Tensor {
        let (r, max_indices) = inputs
            .maxpool2d(
                self.filters,
                self.stride,
                &self.i_d,
                &self.p_d,
                &self.output_dimensions());
        self.max_indices = max_indices;
        r
    }

    #[allow(unused_variables)]
    fn backward<'a>(&'a mut self, learning_rate: &mut LearningRate, dvalues: &Tensor, inputs: &Tensor) -> Tensor {
        let batches = inputs.shape[0];
        let columns = self.filters * self.i_d.height * self.i_d.width;

        let mut values = vec![0.; batches * columns];
        for i in 0..self.max_indices.len() {
            values[self.max_indices[i]] += dvalues[i];
        }

        Tensor::new(Shape::d4(batches, self.filters, self.i_d.height, self.i_d.width), values)
    }
}

#[cfg(test)]
mod tests {
    use crate::geoalg::f32_math::shape::Shape;
    use crate::geoalg::f32_math::tensor::Tensor;
    use crate::nn::layers::convolution2d::Dimensions;
    use crate::nn::layers::LayerPropagates;
    use crate::nn::learning_rate::LearningRate;
    use crate::prettify::*;
    use super::MaxPooling;

    #[test]
    fn test_forward2x2() {
        // Will need to update code if we ever change to use tensors instead of matrices everywhere.
        // Create a matrix that represents 2 filters that are 4 x 4 for a single image.
        let tc = Tensor::new(Shape::d4(1, 2, 6, 6), vec![
            1., 3., 2., 1., 3., 7., 
            4., 2., 1., 5., 9., 4.,
            3., 1., 4., 2., 9., 6.,
            8., 6., 7., 9., 6., 3.,
            10., 5., 4., 3., 6., 8.,
            9., 6., 8., 10., 12., 6.,

            6., 4., 3., 8., 6., 9.,
            2., 4., 3., 7., 7., 3.,
            1., 5., 4., 3., 1., 10.,
            4., 7., 6., 4., 4., 6.,
            10., 5., 4., 3., 6., 8.,
            9., 6., 8., 10., 12., 6.
        ]);

        let mut pooling = MaxPooling::new(
            2, 
            Dimensions { width: 2, height: 2 }, 
            Dimensions { width: 6, height: 6 },
            2);

        // Assume 2 filters
        let forward_output = pooling.forward(&tc);

        let expected = Tensor::new(Shape::d4(1, 2, 3, 3), vec![
            4.0, 5.0, 9.,
            8.0, 9.0, 9.,
            10., 10., 12.,
            
            6.0, 8.0, 9.,
            7.0, 6.0, 10.,
            10., 10., 12.]);

        // After the forward pass, the output should be a matrix that represents 2 filters of 2x2.
        println!("{BRIGHT_MAGENTA}Forward pooling: {:?} <{:?}>{RESET}", forward_output, pooling.max_indices);
        assert_eq!(forward_output, expected);

        // Assume these values came from the previous layers back-propagation.
        let dvalues = Tensor::new(Shape::d3(1, 3, 4), vec![
            0.2, -0.5, 3.,
            0.3, 0.1, -4.,
            0., 1., 0.,

            -1., 4., 2.,
            3., 8., 3.,
            1., 1., 4.
        ]);

        let learning_rate = &mut LearningRate::new(0.01);
        let backward_output = pooling.backward(learning_rate, &dvalues, &forward_output);
        let expected = Tensor::new(Shape::d4(1, 2, 6, 6), vec![
            0.0, 0.0, 0.0, 0.0, 0., 0.,
            0.2, 0.0, 0.0, -0.5, 3., 0.,
            0.0, 0.0, 0.0, 0.0, -4., 0.,
            0.3, 0.0, 0.0, 0.1, 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0.,

            -1.0, 0.0, 0.0, 4.0, 0., 2.,
            0.0, 0.0, 0.0, 0.0, 0., 0.,
            0.0, 0.0, 0.0, 0.0, 0., 3.,
            0.0, 3.0, 8.0, 0.0, 0., 0.,
            1., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 4.0, 0.
        ]);
        assert_eq!(backward_output, expected);

        // After backprop, we should get back to 2 filters of 4x4
        println!("{BRIGHT_MAGENTA}Backprop pooling: {:?}{RESET}", backward_output);
    }
}